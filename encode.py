"""Board -> Poisson drive on four real sensory systems of the FlyWire v783 brain.

WHAT IS REAL HERE: nothing. Every number in this file is AUTHORED. The wiring
the drive lands on is real and frozen; the choice of which neurons stand for
which chess fact is a tuning decision, disclosed here and in WHAT_IS_REAL.md so
no one mistakes it for something the fly does.

Four channels, chosen because flypet measured that each one actually propagates
into the central brain within 300 ms (see results/timing.json):

  vision  T4/T5 motion columns, 8x8 quantile patches per eye. Own pieces drive
          the left eye, the opponent's the right. NOT photoreceptors: R1-6 drive
          never leaves the lamina in this model, so it would be a dead channel.
  smell   ~50 ORN classes as a high-gain state bus (one ORN class alone reaches
          27% of the central brain). Piece counts, castling, check, material.
  taste   sugar/water when ahead on material, bitter when behind.
  touch   Johnston's organ on check, grooming mechanosensors per hanging piece.

Mirroring: if black is to move the board is mirrored first, so "own" is always
white and the head only ever sees one perspective (matches vocab.py).
"""
import json
import os

import chess
import numpy as np

# flypoke and flypet live in the sim container only. board_planes() is pure
# numpy, so train.py can import this module on the host; the two heavy imports
# are deferred into the functions that actually touch the connectome.

# ---------------------------------------------------------------- authored constants
T45 = "cell_type=T4a|T4b|T4c|T4d|T5a|T5b|T5c|T5d"
SUGAR = "cell_sub_class=sugar/water"
BITTER = "cell_sub_class=bitter"
GROOMING = "cell_sub_class=grooming"
JOHNSTON = "cell_type=JO-B1_a|JO-FV"

MAX_HZ = 150.0                 # flypet's drive level; above this the sim saturates

# Vision: Hz per piece type landing on that square's retinal patch.
PIECE_HZ = {chess.PAWN: 60.0, chess.KNIGHT: 80.0, chess.BISHOP: 80.0,
            chess.ROOK: 100.0, chess.QUEEN: 130.0, chess.KING: 150.0}
PIECE_VALUE = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
               chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}

COUNT_HZ = 40.0                # smell: Hz per piece of that type, capped at MAX_HZ
MATERIAL_HZ = 30.0             # smell + taste: Hz per pawn of material difference
HANGING_HZ = 50.0              # touch: Hz per hanging own piece
BIT_HZ = 120.0                 # smell: a set boolean bit
CHECK_HZ = MAX_HZ

# Global multiplier on every drive rate above. 1.0 is the level the 200k
# precompute ran at. Lower values move the network away from the chaotic regime
# in which one-ply-apart boards map to unrelated pooled vectors; see
# WHAT_IS_REAL.md "the reservoir destroys board-space locality".
DRIVE_GAIN = float(os.environ.get("FLYCHESS_DRIVE_GAIN", "1.0"))

GRID = 8                       # 8x8 retinal patches per eye
CACHE = os.environ.get("FLYCHESS_VISUAL_MAP", "/app/data/visual_map.npz")

# Smell bus: 40 named channels, assigned to the first 40 ORN classes in sorted
# order. The order is frozen here so a checkpoint stays meaningful across runs.
ORN_CHANNELS = (
    ["own_%s" % p for p in "PNBRQK"] +                  # 0-5   piece counts
    ["opp_%s" % p for p in "PNBRQK"] +                  # 6-11  piece counts
    ["own_castle_K", "own_castle_Q", "opp_castle_K", "opp_castle_Q"] +   # 12-15
    ["in_check"] +                                      # 16
    ["material_ahead", "material_behind"] +             # 17-18
    ["last_move_file_%d" % i for i in range(8)] +       # 19-26
    ["last_move_rank_%d" % i for i in range(8)] +       # 27-34
    ["halfmove_clock", "fullmove_phase"] +              # 35-36
    ["reserved_0", "reserved_1", "reserved_2"]          # 37-39 (always silent)
)
assert len(ORN_CHANNELS) == 40

_maps = None


# ---------------------------------------------------------------- geometry
def _soma_xyz(n):
    """pos_x/y/z per neuron index.

    flypoke drops the position columns when it loads the annotation TSV, so read
    the same file again and rely on its root_id sort to line the rows up.
    """
    import pandas as pd
    path = os.path.join(os.environ.get("FLYPOKE_DATA", "/data"),
                        "Supplemental_file1_neuron_annotations.tsv")
    df = pd.read_csv(path, sep="\t", usecols=["root_id", "pos_x", "pos_y", "pos_z"],
                     dtype={"root_id": np.int64}, low_memory=False)
    df = df.sort_values("root_id").reset_index(drop=True)
    if len(df) != n.n or not np.array_equal(df.root_id.to_numpy(),
                                            n.neurons.root_id.to_numpy()):
        raise RuntimeError("annotation TSV does not line up with the network index")
    return df[["pos_x", "pos_y", "pos_z"]].to_numpy(dtype=np.float64)


def _pca2(xyz):
    """The medulla/lobula T4/T5 sheet is ~2-D; take its two principal axes."""
    centered = xyz - xyz.mean(axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return centered @ vt[:2].T


def _grid_split(idx, uv):
    """Nested quantile split into GRID x GRID cells, none of them empty.

    Independent quantiles on u and v leave corner cells empty whenever the two
    axes correlate, and the encoder needs all 64 squares to exist.
    """
    cells = {}
    order_u = np.argsort(uv[:, 0], kind="stable")
    for f, u_chunk in enumerate(np.array_split(order_u, GRID)):
        order_v = u_chunk[np.argsort(uv[u_chunk, 1], kind="stable")]
        for r, cell in enumerate(np.array_split(order_v, GRID)):
            cells[(f, r)] = idx[cell]
    return cells


def visual_map(n=None):
    """{side: {(file, rank): indices}} plus the ORN class -> indices table."""
    global _maps
    if _maps is not None:
        return _maps
    if os.path.exists(CACHE):
        blob = np.load(CACHE, allow_pickle=False)
        _maps = {"left": {}, "right": {}}
        for side in ("left", "right"):
            for f in range(GRID):
                for r in range(GRID):
                    _maps[side][(f, r)] = blob["%s_%d_%d" % (side, f, r)]
        _maps["orn"] = [blob["orn_%02d" % k] for k in range(len(ORN_CHANNELS))]
        _maps["orn_names"] = [str(x) for x in blob["orn_names"]]
        return _maps

    import behaviors as B
    n = n or B.net()
    xyz = _soma_xyz(n)
    maps = {"left": {}, "right": {}}
    for side in ("left", "right"):
        idx = n.select("%s,side=%s" % (T45, side))
        if idx.size == 0:
            raise RuntimeError("no T4/T5 neurons on side=%s" % side)
        maps[side] = _grid_split(idx, _pca2(xyz[idx]))

    types = n.neurons.cell_type.astype("string")
    is_orn = types.str.startswith("ORN_").fillna(False).to_numpy()
    orn_names = sorted(set(types[is_orn].tolist()))
    if len(orn_names) < len(ORN_CHANNELS):
        raise RuntimeError("only %d ORN classes; need %d"
                           % (len(orn_names), len(ORN_CHANNELS)))
    orn_names = orn_names[:len(ORN_CHANNELS)]
    orn = [n.select("cell_type=%s" % name) for name in orn_names]

    maps["orn"], maps["orn_names"] = orn, orn_names
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    blob = {"%s_%d_%d" % (side, f, r): maps[side][(f, r)]
            for side in ("left", "right") for f in range(GRID) for r in range(GRID)}
    blob.update({"orn_%02d" % k: v for k, v in enumerate(orn)})
    blob["orn_names"] = np.array(orn_names)
    np.savez_compressed(CACHE, **blob)
    _maps = maps
    return _maps


# ---------------------------------------------------------------- board features
def own_view(board):
    """A board on which the side to move is always white."""
    return board if board.turn == chess.WHITE else board.mirror()


def board_planes(board):
    """float32[18, 8, 8]: 12 piece planes, 4 castling, en passant, halfmove clock."""
    b = own_view(board)
    planes = np.zeros((18, 8, 8), dtype=np.float32)
    for square, piece in b.piece_map().items():
        p = (piece.piece_type - 1) + (0 if piece.color == chess.WHITE else 6)
        planes[p, chess.square_rank(square), chess.square_file(square)] = 1.0
    for k, bb in enumerate([chess.BB_H1, chess.BB_A1, chess.BB_H8, chess.BB_A8]):
        planes[12 + k, :, :] = float(bool(b.castling_rights & bb))
    if b.ep_square is not None:
        planes[16, chess.square_rank(b.ep_square), chess.square_file(b.ep_square)] = 1.0
    planes[17, :, :] = min(b.halfmove_clock, 100) / 100.0
    return planes


def material_diff(b):
    """Own minus opponent in pawn units, on an already-mirrored board."""
    total = 0
    for _square, piece in b.piece_map().items():
        v = PIECE_VALUE[piece.piece_type]
        total += v if piece.color == chess.WHITE else -v
    return total


def hanging(b):
    """Own pieces the opponent attacks more times than we defend them."""
    k = 0
    for square, piece in b.piece_map().items():
        if piece.color != chess.WHITE or piece.piece_type == chess.KING:
            continue
        if len(b.attackers(chess.BLACK, square)) > len(b.attackers(chess.WHITE, square)):
            k += 1
    return k


def orn_levels(board):
    """Hz for each of the 40 smell channels, in ORN_CHANNELS order."""
    b = own_view(board)
    hz = np.zeros(len(ORN_CHANNELS), dtype=np.float64)
    for i, pt in enumerate([chess.PAWN, chess.KNIGHT, chess.BISHOP,
                            chess.ROOK, chess.QUEEN, chess.KING]):
        hz[i] = min(MAX_HZ, COUNT_HZ * len(b.pieces(pt, chess.WHITE)))
        hz[6 + i] = min(MAX_HZ, COUNT_HZ * len(b.pieces(pt, chess.BLACK)))
    for k, bb in enumerate([chess.BB_H1, chess.BB_A1, chess.BB_H8, chess.BB_A8]):
        hz[12 + k] = BIT_HZ if b.castling_rights & bb else 0.0
    hz[16] = CHECK_HZ if b.is_check() else 0.0
    diff = material_diff(b)
    hz[17] = min(MAX_HZ, MATERIAL_HZ * diff) if diff > 0 else 0.0
    hz[18] = min(MAX_HZ, MATERIAL_HZ * -diff) if diff < 0 else 0.0
    if b.move_stack:
        last = b.move_stack[-1].to_square
        hz[19 + chess.square_file(last)] = BIT_HZ
        hz[27 + chess.square_rank(last)] = BIT_HZ
    hz[35] = MAX_HZ * min(b.halfmove_clock, 100) / 100.0
    hz[36] = MAX_HZ * min(b.fullmove_number, 60) / 60.0
    return hz


# ---------------------------------------------------------------- drive assembly
def rate_by_neuron(board, n=None):
    """{neuron index: Hz}. Collisions take the max, so no neuron is driven twice."""
    import behaviors as B
    n = n or B.net()
    maps = visual_map(n)
    b = own_view(board)
    rates = {}

    def add(indices, hz):
        if hz <= 0.0 or len(indices) == 0:
            return
        for i in indices:
            i = int(i)
            if hz > rates.get(i, 0.0):
                rates[i] = hz

    # vision: own pieces on the left eye, the opponent's on the right
    for square, piece in b.piece_map().items():
        side = "left" if piece.color == chess.WHITE else "right"
        cell = maps[side][(chess.square_file(square), chess.square_rank(square))]
        add(cell, PIECE_HZ[piece.piece_type])

    # smell
    for hz, indices in zip(orn_levels(board), maps["orn"]):
        add(indices, float(hz))

    # taste
    diff = material_diff(b)
    if diff > 0:
        add(n.select(SUGAR), min(MAX_HZ, MATERIAL_HZ * diff))
    elif diff < 0:
        add(n.select(BITTER), min(MAX_HZ, MATERIAL_HZ * -diff))

    # touch
    if b.is_check():
        add(n.select(JOHNSTON), CHECK_HZ)
    k = hanging(b)
    if k:
        add(n.select(GROOMING), min(MAX_HZ, HANGING_HZ * k))
    if DRIVE_GAIN != 1.0:
        rates = {i: hz * DRIVE_GAIN for i, hz in rates.items()}
    return rates


def stimuli(board, n=None):
    """list[S.Stimulus], one per distinct rate, with disjoint index sets.

    flypoke turns any neuron inside a Stimulus into a pure Poisson source and
    drops its network input, so a neuron appearing in two Stimulus objects would
    silently take whichever the loop reached last. Merging first is not an
    optimisation, it is the only correct way to build this list.
    """
    from flypoke import sim as S
    rates = rate_by_neuron(board, n)
    groups = {}
    for i, hz in rates.items():
        groups.setdefault(round(hz, 1), []).append(i)
    return [S.Stimulus(np.array(sorted(v), dtype=np.int64), float(hz))
            for hz, v in sorted(groups.items())]


def summary(board, n=None):
    """Human- and body-readable record of what the fly was actually shown."""
    import behaviors as B
    n = n or B.net()
    b = own_view(board)
    eyes = {"left": np.zeros((GRID, GRID)), "right": np.zeros((GRID, GRID))}
    for square, piece in b.piece_map().items():
        side = "left" if piece.color == chess.WHITE else "right"
        f, r = chess.square_file(square), chess.square_rank(square)
        eyes[side][f, r] = max(eyes[side][f, r], PIECE_HZ[piece.piece_type])
    diff = material_diff(b)
    levels = orn_levels(board)
    return {
        "fen": board.fen(),
        "mirrored": board.turn == chess.BLACK,
        "retina_hz": {side: eyes[side].round(1).tolist() for side in eyes},
        "taste_hz": {
            "sugar": round(min(MAX_HZ, MATERIAL_HZ * diff), 1) if diff > 0 else 0.0,
            "bitter": round(min(MAX_HZ, MATERIAL_HZ * -diff), 1) if diff < 0 else 0.0,
        },
        "touch_hz": {"johnston": CHECK_HZ if b.is_check() else 0.0,
                     "grooming": round(min(MAX_HZ, HANGING_HZ * hanging(b)), 1)},
        "smell_hz": {name: round(float(hz), 1)
                     for name, hz in zip(ORN_CHANNELS, levels) if hz > 0},
        "material_diff": diff,
    }


def selfcheck(verbose=True):
    import behaviors as B
    n = B.net()
    maps = visual_map(n)
    ok = True

    for side in ("left", "right"):
        sizes = [len(v) for v in maps[side].values()]
        good = min(sizes) > 0 and len(sizes) == GRID * GRID
        ok &= good
        if verbose:
            print("%s retina %-5s %d/%d non-empty  min=%d median=%d max=%d"
                  % ("OK " if good else "BAD", side,
                     sum(1 for s in sizes if s), GRID * GRID,
                     min(sizes), int(np.median(sizes)), max(sizes)))

    start = chess.Board()
    stims = stimuli(start, n)
    all_idx = np.concatenate([s.indices for s in stims])
    disjoint = len(all_idx) == len(set(all_idx.tolist()))
    ok &= disjoint
    if verbose:
        print("%s start position: %d Stimulus objects, %d neurons, disjoint=%s"
              % ("OK " if disjoint else "BAD", len(stims), len(all_idx), disjoint))

    smell = summary(start, n)["smell_hz"]
    enough = len(smell) >= 10
    ok &= enough
    if verbose:
        print("%s start position drives %d smell channels (need >= 10)"
              % ("OK " if enough else "BAD", len(smell)))

    a = rate_by_neuron(start, n)
    again = rate_by_neuron(chess.Board(), n)
    other = rate_by_neuron(chess.Board(
        "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 4 4"), n)
    same, differs = a == again, other != a
    ok &= same and differs
    if verbose:
        print("%s deterministic=%s   a different FEN gives a different drive=%s"
              % ("OK " if same and differs else "BAD", same, differs))

    bare = chess.Board(None)
    bare.set_piece_at(chess.E1, chess.Piece(chess.KING, chess.WHITE))
    bare.set_piece_at(chess.E8, chess.Piece(chess.KING, chess.BLACK))
    try:
        stimuli(bare, n)
        if verbose:
            print("OK  bare-kings board still encodes")
    except Exception as exc:                   # reported, never swallowed
        ok = False
        if verbose:
            print("BAD bare-kings board raised", exc)
    return ok


if __name__ == "__main__":
    import sys
    good = selfcheck()
    print(json.dumps(summary(chess.Board())["smell_hz"], indent=1))
    print("SELFCHECK", "PASS" if good else "FAIL")
    sys.exit(0 if good else 1)
