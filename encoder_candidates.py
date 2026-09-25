"""Can an encoder preserve board-space locality? No simulation required.

Measured 2026-09-16 (`where_locality_dies.py`): the authored board -> drive map
correlates with board distance at r=0.019 and puts near-identical boards at 0.76x
the distance of median pairs. The LIF sim raises that to 0.227 and the pooling to
0.274, so the fly brain is the part that helps. The encoder is the bottleneck.

Stage 1 is pure numpy over ~4k neurons, so candidates cost milliseconds instead
of the 7.8 h a re-precompute would. This scores alternatives against the shipped
encoding before anyone pays for new shards.

The constraint a candidate must respect: it is still a set of Poisson rates on
real sensory neurons, so it may only change WHICH neurons carry WHAT, never add
channels the fly does not have.

Run: docker compose run --rm sim python -u encoder_candidates.py
"""
import argparse
import os

import chess
import numpy as np
import pyarrow.parquet as pq

import behaviors as B
import encode

DATA = "/app/data"
PIECES = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]


def shipped(board, maps, n):
    """What encode.rate_by_neuron does today: one patch per square, type = rate."""
    v = np.zeros(n, dtype=np.float32)
    b = encode.own_view(board)
    for sq, pc in b.piece_map().items():
        side = "left" if pc.color == chess.WHITE else "right"
        cell = maps[side][(chess.square_file(sq), chess.square_rank(sq))]
        v[cell] = np.maximum(v[cell], encode.PIECE_HZ[pc.piece_type])
    return v


def by_type(board, maps, n):
    """Piece type gets its own neurons, mirroring how board planes work.

    Each square's patch is split into 6 disjoint slices, one per piece type, so a
    pawn and a queen on the same square no longer differ only in amplitude.
    """
    v = np.zeros(n, dtype=np.float32)
    b = encode.own_view(board)
    for sq, pc in b.piece_map().items():
        side = "left" if pc.color == chess.WHITE else "right"
        cell = maps[side][(chess.square_file(sq), chess.square_rank(sq))]
        parts = np.array_split(cell, len(PIECES))
        v[parts[PIECES.index(pc.piece_type)]] = encode.MAX_HZ
    return v


def blurred(board, maps, n, spread=0.45):
    """Adjacent squares share drive, so a one-square move is a small change.

    The shipped patches are disjoint, so moving a piece one square is as large a
    change as moving it across the board. This leaks a fraction of each piece's
    rate into the 4-neighbour squares.
    """
    v = np.zeros(n, dtype=np.float32)
    b = encode.own_view(board)
    for sq, pc in b.piece_map().items():
        side = "left" if pc.color == chess.WHITE else "right"
        f, r = chess.square_file(sq), chess.square_rank(sq)
        hz = encode.PIECE_HZ[pc.piece_type]
        for df, dr, w in ((0, 0, 1.0), (1, 0, spread), (-1, 0, spread),
                          (0, 1, spread), (0, -1, spread)):
            if 0 <= f + df < 8 and 0 <= r + dr < 8:
                cell = maps[side][(f + df, r + dr)]
                v[cell] = np.maximum(v[cell], hz * w)
    return v


def by_type_blurred(board, maps, n):
    """Both fixes at once."""
    v = np.zeros(n, dtype=np.float32)
    b = encode.own_view(board)
    for sq, pc in b.piece_map().items():
        side = "left" if pc.color == chess.WHITE else "right"
        f, r = chess.square_file(sq), chess.square_rank(sq)
        k = PIECES.index(pc.piece_type)
        for df, dr, w in ((0, 0, 1.0), (1, 0, 0.45), (-1, 0, 0.45),
                          (0, 1, 0.45), (0, -1, 0.45)):
            if 0 <= f + df < 8 and 0 <= r + dr < 8:
                cell = maps[side][(f + df, r + dr)]
                parts = np.array_split(cell, len(PIECES))
                v[parts[k]] = np.maximum(v[parts[k]], encode.MAX_HZ * w)
    return v


def score(vecs, dp, i, j):
    x = np.log1p(np.clip(vecs, 0, None))
    keep = x.std(axis=0) > 1e-4
    x = x[:, keep]
    x = (x - x.mean(0)) / (x.std(0) + 1e-6)
    d = np.linalg.norm(x[i] - x[j], axis=1)
    r = float(np.corrcoef(dp, d)[0, 1])
    near = dp <= np.quantile(dp, 0.05)
    med = dp >= np.quantile(dp, 0.5)
    return r, float(d[near].mean() / d[med].mean()), int(keep.sum())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--positions", type=int, default=400)
    args = ap.parse_args()

    net = B.net()
    maps = encode.visual_map(net)
    fens = pq.read_table(os.path.join(DATA, "positions.parquet"),
                         columns=["fen"]).column("fen").to_pylist()
    rng = np.random.default_rng(0)
    picks = [fens[k] for k in rng.choice(len(fens), args.positions, replace=False)]
    boards = [chess.Board(f) for f in picks]

    planes = np.stack([encode.board_planes(b).reshape(-1) for b in boards])
    i, j = np.triu_indices(len(boards), k=1)
    sel = rng.choice(len(i), min(30000, len(i)), replace=False)
    i, j = i[sel], j[sel]
    dp = np.linalg.norm(planes[i] - planes[j], axis=1)

    print("\n%d positions, %d pairs. Higher r and lower near/med is better." % (len(boards), len(i)))
    print("%-34s %8s %10s %9s" % ("candidate", "r", "near/med", "live dims"))
    for name, fn in (("shipped (one patch, type=rate)", shipped),
                     ("type gets its own neurons", by_type),
                     ("blurred into neighbours", blurred),
                     ("both", by_type_blurred)):
        v = np.stack([fn(b, maps, net.n) for b in boards])
        r, ratio, live = score(v, dp, i, j)
        print("%-34s %8.4f %10.3f %9d" % (name, r, ratio, live))
