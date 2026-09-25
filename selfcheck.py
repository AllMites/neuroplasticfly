"""Every assert-based check in the pipeline, in one run.

Run: docker compose run --rm sim python -u selfcheck.py

Checks that need artefacts (a dataset, reservoir shards, a checkpoint) are
skipped with a SKIP line rather than failing, so this stays runnable from a cold
clone. Skips never turn the run green on their own - the summary says how many.
"""
import os
import sys

import chess
import numpy as np

import encode
import reservoir as R
import vocab

DATA = "/app/data"


def section(title):
    print("\n== %s ==" % title)


def run_vocab():
    section("vocab: 1858-move index")
    ok, checked = vocab.selfcheck(n_positions=300)
    assert len(vocab.POLICY_INDEX) == 1858
    print("    %d legal moves round-tripped" % checked)
    return ok


def run_encode():
    section("encode: board -> Poisson drive")
    return encode.selfcheck()


def run_reservoir():
    section("reservoir: 300 ms whole-brain sim")
    return R.selfcheck()


def run_masking():
    section("head masking: illegal moves get zero probability")
    ok = True
    for fen in [chess.STARTING_FEN,
                "k7/7R/8/8/8/8/8/K7 b - - 0 1",               # exactly one legal move
                "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 4 4"]:
        board = chess.Board(fen)
        mask, pairs = vocab.legal_mask(board)
        logits = np.random.default_rng(0).normal(size=vocab.N_MOVES)
        logits[~mask] = -np.inf
        z = logits - logits.max()
        p = np.exp(z)
        p /= p.sum()
        illegal = float(p[~mask].sum())
        good = illegal == 0.0 and abs(p.sum() - 1.0) < 1e-9 and len(pairs) == \
            board.legal_moves.count()
        ok &= good
        print("%s %-2d legal moves, P(illegal)=%.1e, sum(P)=%.6f"
              % ("OK " if good else "BAD", len(pairs), illegal, p.sum()))
    return ok


def run_search():
    section("search: depth-2 alpha-beta finds mate and takes free material")
    import search
    return search.selfcheck()


def run_dataset():
    section("dataset: sampled human positions")
    path = os.path.join(DATA, "positions.parquet")
    if not os.path.exists(path):
        print("SKIP no positions.parquet yet (run dataset.py)")
        return None
    import dataset
    ok, n = dataset.verify(path, sample=1000)
    return ok


def run_shards():
    section("precompute: reservoir shards")
    shard_dir = os.path.join(DATA, "reservoir")
    if not os.path.isdir(shard_dir) or not os.listdir(shard_dir):
        print("SKIP no reservoir shards yet (run precompute.py)")
        return None
    import precompute
    return precompute.verify()


def run_player():
    section("untrained head plays legal, near-uniform moves")
    ckpt = "/app/checkpoints/full/epoch_00.pt"
    if not os.path.exists(ckpt):
        print("SKIP no epoch_00.pt yet (run train.py)")
        return None
    import play
    player = play.FlyPlayer(ckpt, temperature=1.0)
    board = chess.Board()
    ok = True
    for ply in range(2):
        move, top5, _res = player.choose(board)
        legal = move in board.legal_moves
        ok &= legal
        spread = max(t["p"] for t in top5) if top5 else 1.0
        near_uniform = spread < 0.25
        ok &= near_uniform
        print("%s ply %d: %s legal=%s  top-1 probability %.3f (untrained, expect < 0.25)"
              % ("OK " if legal and near_uniform else "BAD", ply, move.uci(),
                 legal, spread))
        board.push(move)
    return ok


def run_npz_schema():
    section("recorded-move npz schema")
    games = "/app/games"
    found = None
    for root, _dirs, files in os.walk(games):
        for f in files:
            if f.endswith(".npz") and not f.endswith("reward.npz"):
                found = os.path.join(root, f)
                break
        if found:
            break
    if not found:
        print("SKIP no recorded game yet (run play.py --record)")
        return None
    blob = np.load(found, allow_pickle=False)
    need = {"rates", "spike_t", "spike_i", "fen", "move", "top5",
            "dn_table", "sensory_summary"}
    missing = need - set(blob.files)
    size_mb = os.path.getsize(found) / 1e6
    ok = not missing
    print("%s %s: %.2f MB, missing keys: %s"
          % ("OK " if ok else "BAD", found, size_mb, sorted(missing) or "none"))

    reward = os.path.join(os.path.dirname(found),
                          os.path.basename(found).replace(".npz", "_reward.npz"))
    if os.path.exists(reward):
        rb = np.load(reward, allow_pickle=False)
        label = str(rb["label"])
        good = label in ("fed", "bitter")
        ok &= good
        print("%s reward label %r, ingestion %.1f Hz"
              % ("OK " if good else "BAD", label, float(rb["ingestion_motor_hz"])))
    return ok


def run_disclosure():
    section("disclosure: every authored constant appears in WHAT_IS_REAL.md")
    import re
    here = os.path.dirname(os.path.abspath(__file__))
    names = []
    for mod in ("encode.py", "search.py"):
        src = open(os.path.join(here, mod)).read()
        names += re.findall(r"^([A-Z][A-Z0-9_]+)\s*=", src, flags=re.M)
    doc = open(os.path.join(here, "WHAT_IS_REAL.md")).read()
    # word boundaries: plain `in` lets OUT pass on $FLYCHESS_OUT and any short
    # constant pass on an unrelated word that contains it
    missing = [n for n in names if not re.search(r"\b%s\b" % n, doc)]
    ok = not missing
    print("%s %d module-level constants in encode.py + search.py, %d undocumented%s"
          % ("OK " if ok else "BAD", len(names), len(missing),
             (": " + ", ".join(missing)) if missing else ""))
    return ok


CHECKS = [run_vocab, run_encode, run_reservoir, run_masking, run_search,
          run_disclosure, run_dataset, run_shards, run_player, run_npz_schema]

if __name__ == "__main__":
    results = {}
    for check in CHECKS:
        results[check.__name__] = check()
    failed = [k for k, v in results.items() if v is False]
    skipped = [k for k, v in results.items() if v is None]
    print("\n%d passed, %d failed, %d skipped"
          % (sum(1 for v in results.values() if v is True), len(failed), len(skipped)))
    if skipped:
        print("skipped (artefacts not built yet): %s" % ", ".join(skipped))
    if failed:
        print("failed: %s" % ", ".join(failed))
    print("SELFCHECK", "PASS" if not failed else "FAIL")
    sys.exit(0 if not failed else 1)
