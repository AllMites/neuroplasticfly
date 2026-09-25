"""Which stage destroys board-space locality?

Measured 2026-09-16: the pooled vector correlates with board-space distance at
r=0.08, and a head reading it can only memorise -- val top-1 gains 0.010 per
doubling of data against the board planes' 0.045, and no amount of capacity or
regularisation moves it. That is the ceiling on the whole fly channel.

Locality has to die somewhere in board -> drive -> 139k rates -> pooled. Each
stage has a very different fix, so measure the same statistic at all three
instead of guessing:

  drive   encode.rate_by_neuron, pure authored code, cheap to change
  rates   the LIF sim over the real connectome, expensive/impossible to change
  pooled  reservoir.pool, a readout choice, cheap to change

Whichever stage the correlation collapses at is the one to work on. If it
collapses at `rates`, that is the fly itself and the honest answer is that the
project's premise does not hold.

Run: docker compose run --rm sim python -u where_locality_dies.py
"""
import argparse
import multiprocessing as mp
import os

import chess
import numpy as np
import pyarrow.parquet as pq

import behaviors as B
import encode
import reservoir as R

DATA = "/app/data"
_net = None


def _init():
    global _net
    _net = B.net()


def _one(fen):
    board = chess.Board(fen)
    drive = np.zeros(_net.n, dtype=np.float32)
    for i, hz in encode.rate_by_neuron(board, _net).items():
        drive[i] = hz
    out = R.simulate(board, full=True, n=_net)
    return (drive,
            np.asarray(out["rates"], dtype=np.float32),
            np.asarray(out["pooled"], dtype=np.float32))


def locality(x, dp, i, j, standardise=True):
    """(pearson r, near/median distance ratio) against board distance."""
    x = np.log1p(np.clip(x, 0, None))
    keep = x.std(axis=0) > 1e-4
    x = x[:, keep]
    if standardise:
        x = (x - x.mean(0)) / (x.std(0) + 1e-6)
    d = np.linalg.norm(x[i] - x[j], axis=1)
    r = float(np.corrcoef(dp, d)[0, 1])
    near = dp <= np.quantile(dp, 0.05)
    med = dp >= np.quantile(dp, 0.5)
    return r, float(d[near].mean() / d[med].mean()), int(keep.sum())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--positions", type=int, default=90)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    fens = pq.read_table(os.path.join(DATA, "positions.parquet"),
                         columns=["fen"]).column("fen").to_pylist()
    rng = np.random.default_rng(0)
    picks = [fens[i] for i in rng.choice(len(fens), args.positions, replace=False)]

    B.net()
    R.pooling()
    ctx = mp.get_context("fork")
    with ctx.Pool(args.workers, initializer=_init) as pool:
        got = pool.map(_one, picks, chunksize=1)

    drive = np.stack([g[0] for g in got])
    rates = np.stack([g[1] for g in got])
    pooled = np.stack([g[2] for g in got])

    planes = np.stack([encode.board_planes(chess.Board(f)).reshape(-1) for f in picks])
    i, j = np.triu_indices(len(picks), k=1)
    dp = np.linalg.norm(planes[i] - planes[j], axis=1)

    print("\n%d positions, %d pairs" % (len(picks), len(i)))
    print("%-34s %8s %10s %9s" % ("stage", "r", "near/med", "live dims"))
    print("%-34s %8.4f %10.3f %9d"
          % ("board planes (reference)", 1.0, 0.0, planes.shape[1]))
    for name, x in (("1. drive (encode.py, authored)", drive),
                    ("2. rates 139k (the LIF sim)", rates),
                    ("3. pooled 8865 (the readout)", pooled)):
        r, ratio, live = locality(x, dp, i, j)
        print("%-34s %8.4f %10.3f %9d" % (name, r, ratio, live))
    print("\nlower near/med and higher r = more locality preserved.")
    print("the stage where it collapses is the one to fix.")
