"""Does lowering the drive restore board-space locality?

Measured 2026-09-16 at the shipped drive (DRIVE_GAIN=1.0): the pooled vector is
reproducible (SNR 4.31) and separates one-ply-apart boards as strongly as random
ones (ratio 1.1x), but correlates with board-space distance at only r=0.08. That
is a chaotic hash, not a feature map, and an MLP cannot generalise across one.

Reservoir computing works at the edge of chaos, not past it. This sweeps the
drive down and measures, at each level, whether nearby boards start landing on
nearby pooled vectors.

Run: docker compose run --rm sim python -u drive_sweep.py
"""
import argparse
import multiprocessing as mp
import os
import subprocess
import sys

import chess
import numpy as np
import pyarrow.parquet as pq

DATA = "/app/data"
_net = None


def _init():
    global _net
    import behaviors as B
    _net = B.net()


def _one(fen):
    import reservoir as R
    return np.asarray(R.simulate(chess.Board(fen), n=_net)["pooled"], dtype=np.float32)


def measure(n_pos, workers):
    """One drive level: (locality r, near/median ratio, seed-relative spread)."""
    import behaviors as B
    import encode
    import reservoir as R

    fens = pq.read_table(os.path.join(DATA, "positions.parquet"),
                         columns=["fen"]).column("fen").to_pylist()
    rng = np.random.default_rng(0)
    picks = [fens[i] for i in rng.choice(len(fens), n_pos, replace=False)]

    B.net()
    R.pooling()
    ctx = mp.get_context("fork")
    with ctx.Pool(workers, initializer=_init) as pool:
        vecs = pool.map(_one, picks, chunksize=1)

    x = np.log1p(np.clip(np.stack(vecs), 0, None))
    live = x.std(axis=0) > 1e-4
    x = x[:, live]
    x = (x - x.mean(0)) / (x.std(0) + 1e-6)

    planes = np.stack([encode.board_planes(chess.Board(f)).reshape(-1) for f in picks])
    i, j = np.triu_indices(len(picks), k=1)
    sel = rng.choice(len(i), min(20000, len(i)), replace=False)
    i, j = i[sel], j[sel]
    dp = np.linalg.norm(planes[i] - planes[j], axis=1)
    dr = np.linalg.norm(x[i] - x[j], axis=1)

    r = float(np.corrcoef(dp, dr)[0, 1])
    near = dp <= np.quantile(dp, 0.02)
    med = dp >= np.quantile(dp, 0.5)
    ratio = float(dr[near].mean() / dr[med].mean())
    return r, ratio, int(live.sum())


CHILD = "FLYCHESS_SWEEP_CHILD"

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--positions", type=int, default=150)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--gains", default="1.0,0.5,0.25,0.1")
    args = ap.parse_args()

    if os.environ.get(CHILD):
        r, ratio, live = measure(args.positions, args.workers)
        print("RESULT %.4f %.4f %d" % (r, ratio, live))
        sys.exit(0)

    # DRIVE_GAIN is read at import time, so each level needs a fresh interpreter
    print("gain   locality_r   near/median   live dims   verdict")
    for g in [float(v) for v in args.gains.split(",")]:
        env = dict(os.environ, FLYCHESS_DRIVE_GAIN=str(g), **{CHILD: "1"})
        out = subprocess.run([sys.executable, "-u", __file__,
                              "--positions", str(args.positions),
                              "--workers", str(args.workers)],
                             capture_output=True, text=True, env=env)
        line = [l for l in out.stdout.splitlines() if l.startswith("RESULT")]
        if not line:
            print("%-6.2f FAILED: %s" % (g, out.stderr.strip().splitlines()[-1:]))
            continue
        _, r, ratio, live = line[0].split()
        r, ratio = float(r), float(ratio)
        verdict = "locality" if r > 0.3 and ratio < 0.7 else "chaotic"
        print("%-6.2f %11.4f %13.3f %11s   %s" % (g, r, ratio, live, verdict))
