"""Stream the Lichess database and sample human positions to imitate.

AUTHORED: the Elo band, the time-control floor, the sampling rule. Nothing is
measured and nothing is trained here.

The dump is tens of gigabytes, so it is streamed and decompressed on the fly and
never written to disk. Games are filtered with a visitor that returns SKIP from
end_headers(), which makes python-chess drop the movetext of a rejected game
without parsing it - that is the difference between minutes and hours.

Run: docker compose run --rm sim python -u dataset.py --n 200000
"""
import argparse
import io
import json
import os

import chess
import chess.pgn
import pyarrow as pa
import pyarrow.parquet as pq
import requests
import zstandard

URL = "https://database.lichess.org/standard/lichess_db_standard_rated_%s.pgn.zst"

ELO_LO, ELO_HI = 1500, 2000
MIN_BASE_SECONDS = 300
MIN_PLY = 8               # skip book moves; they teach the head nothing
SAMPLE_PROB = 0.5
MAX_PER_GAME = 20
PART_ROWS = 50_000

OUT_DIR = "/app/data"
STATE = os.path.join(OUT_DIR, "dataset_state.json")

SCHEMA = pa.schema([("fen", pa.string()), ("move_uci", pa.string()),
                    ("white_elo", pa.int16()), ("black_elo", pa.int16()),
                    ("game_id", pa.string())])


class Filtered(chess.pgn.GameBuilder):
    """Parse the movetext only for games inside the Elo / time-control band."""

    def end_headers(self):
        h = self.game.headers
        try:
            welo, belo = int(h.get("WhiteElo", 0)), int(h.get("BlackElo", 0))
        except ValueError:
            return chess.pgn.SKIP
        if not (ELO_LO <= welo <= ELO_HI and ELO_LO <= belo <= ELO_HI):
            return chess.pgn.SKIP
        if h.get("Termination") != "Normal":
            return chess.pgn.SKIP
        tc = h.get("TimeControl", "")
        base = tc.split("+")[0]
        if not base.isdigit() or int(base) < MIN_BASE_SECONDS:
            return chess.pgn.SKIP
        return None


def stream(month):
    """Text handle over the decompressed dump. Nothing touches disk."""
    resp = requests.get(URL % month, stream=True, timeout=60)
    resp.raise_for_status()
    reader = zstandard.ZstdDecompressor().stream_reader(resp.raw)
    return io.TextIOWrapper(reader, encoding="utf-8", errors="replace")


def sample_game(game, rng, seen, rows):
    """Append up to MAX_PER_GAME (fen, move) pairs from one accepted game."""
    h = game.headers
    gid = h.get("Site", "").rsplit("/", 1)[-1]
    welo, belo = int(h.get("WhiteElo", 0)), int(h.get("BlackElo", 0))
    board = game.board()
    taken = 0
    for ply, move in enumerate(game.mainline_moves()):
        if ply >= MIN_PLY and taken < MAX_PER_GAME and rng.random() < SAMPLE_PROB:
            fen = board.fen()
            if fen not in seen:
                seen.add(fen)
                rows.append((fen, move.uci(), welo, belo, gid))
                taken += 1
        board.push(move)
    return taken


def write_part(rows, part):
    path = os.path.join(OUT_DIR, "positions_part_%03d.parquet" % part)
    table = pa.Table.from_arrays(
        [pa.array([r[0] for r in rows]), pa.array([r[1] for r in rows]),
         pa.array([r[2] for r in rows], type=pa.int16()),
         pa.array([r[3] for r in rows], type=pa.int16()),
         pa.array([r[4] for r in rows])], schema=SCHEMA)
    pq.write_table(table, path)
    return path


def combine():
    """Merge the parts into data/positions.parquet and drop duplicate FENs."""
    parts = sorted(f for f in os.listdir(OUT_DIR)
                   if f.startswith("positions_part_") and f.endswith(".parquet"))
    if not parts:
        raise SystemExit("no parts to combine; run the collector first")
    table = pa.concat_tables([pq.read_table(os.path.join(OUT_DIR, p)) for p in parts])
    fens, keep = set(), []
    col = table.column("fen").to_pylist()
    for i, fen in enumerate(col):
        if fen not in fens:
            fens.add(fen)
            keep.append(i)
    table = table.take(keep)
    out = os.path.join(OUT_DIR, "positions.parquet")
    pq.write_table(table, out)
    print("wrote %s: %d unique positions from %d parts" % (out, table.num_rows, len(parts)))
    return out


def collect(month, n_target, resume):
    import random
    import time
    os.makedirs(OUT_DIR, exist_ok=True)
    state = {"games_read": 0, "part": 0, "n_pos": 0, "month": month}
    if resume and os.path.exists(STATE):
        state = json.load(open(STATE))
        print("resuming after %d games, %d positions" % (state["games_read"], state["n_pos"]))

    rng = random.Random(1234)
    seen, rows = set(), []
    skip_games, games, t0 = state["games_read"], 0, time.time()
    fh = stream(month)
    try:
        while state["n_pos"] < n_target:
            game = chess.pgn.read_game(fh, Visitor=Filtered)
            if game is None:
                print("end of dump after %d games" % games)
                break
            games += 1
            if games <= skip_games:
                continue
            state["games_read"] = games
            if not game.variations:            # rejected by the header filter
                continue
            state["n_pos"] += sample_game(game, rng, seen, rows)
            if len(rows) >= PART_ROWS:
                path = write_part(rows[:PART_ROWS], state["part"])
                rows = rows[PART_ROWS:]
                state["part"] += 1
                json.dump(state, open(STATE, "w"))
                rate = state["n_pos"] / max(time.time() - t0, 1e-6)
                print("%s  %d games, %d positions, %.0f pos/s, eta %.0f min"
                      % (path, games, state["n_pos"], rate,
                         (n_target - state["n_pos"]) / max(rate, 1e-6) / 60))
    finally:
        fh.close()
    if rows:
        write_part(rows, state["part"])
        state["part"] += 1
    json.dump(state, open(STATE, "w"))
    print("collected %d positions from %d games in %.0f s"
          % (state["n_pos"], games, time.time() - t0))


def verify(path=None, sample=2000):
    """Every sampled move must be legal in its FEN and inside the Elo band."""
    import random
    path = path or os.path.join(OUT_DIR, "positions.parquet")
    table = pq.read_table(path)
    n = table.num_rows
    rng = random.Random(0)
    picks = rng.sample(range(n), min(sample, n))
    fens = table.column("fen").to_pylist()
    moves = table.column("move_uci").to_pylist()
    welo = table.column("white_elo").to_pylist()
    belo = table.column("black_elo").to_pylist()
    bad_move = bad_elo = 0
    for i in picks:
        board = chess.Board(fens[i])
        if chess.Move.from_uci(moves[i]) not in board.legal_moves:
            bad_move += 1
        if not (ELO_LO <= welo[i] <= ELO_HI and ELO_LO <= belo[i] <= ELO_HI):
            bad_elo += 1
    ok = bad_move == 0 and bad_elo == 0
    print("%s %d rows; checked %d: %d illegal moves, %d out-of-band Elo"
          % ("OK " if ok else "BAD", n, len(picks), bad_move, bad_elo))
    return ok, n


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--month", default="2017-01",
                    help="lichess dump, YYYY-MM. Older months are far smaller.")
    ap.add_argument("--n", type=int, default=200_000)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--combine-only", action="store_true")
    ap.add_argument("--verify-only", action="store_true")
    args = ap.parse_args()

    if args.verify_only:
        good, _ = verify()
        raise SystemExit(0 if good else 1)
    if not args.combine_only:
        collect(args.month, args.n, args.resume)
    combine()
    good, _ = verify()
    print("SELFCHECK", "PASS" if good else "FAIL")
    raise SystemExit(0 if good else 1)
