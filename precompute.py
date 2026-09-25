"""Run every sampled position through the brain once and store the pooled vector.

This is the expensive step: one 300 ms whole-brain sim per position, ~0.9 s each,
so 200k positions is ~5 h on 16 workers. The pool is forked AFTER the network is
built so all workers share the ~100 MB connectivity matrix copy-on-write instead
of each loading its own.

Only the pooled vector is stored. Full 139k rates exist only for recorded games.

Run: docker compose run --rm sim python -u precompute.py --workers 16
"""
import argparse
import multiprocessing as mp
import os
import time

import chess
import numpy as np
import pyarrow.parquet as pq

import behaviors as B
import reservoir as R

DATA = "/app/data"
SHARD_DIR = os.environ.get("FLYCHESS_SHARDS", os.path.join(DATA, "reservoir"))
SHARD_ROWS = 1000

_net = None


def _init():
    """Each forked worker keeps using the already-built network."""
    global _net
    _net = B.net()


def _one(args):
    row_id, fen = args
    out = R.simulate(chess.Board(fen), n=_net, all_views=True)
    return (row_id,
            np.asarray(out["pooled"], dtype=np.float16),
            np.asarray(out["live"], dtype=np.float16))


def shard_path(k):
    return os.path.join(SHARD_DIR, "shard_%04d.npz" % k)


def run(workers, limit=None):
    os.makedirs(SHARD_DIR, exist_ok=True)
    table = pq.read_table(os.path.join(DATA, "positions.parquet"), columns=["fen"])
    fens = table.column("fen").to_pylist()
    if limit:
        fens = fens[:limit]
    n_shards = (len(fens) + SHARD_ROWS - 1) // SHARD_ROWS
    print("%d positions -> %d shards of %d" % (len(fens), n_shards, SHARD_ROWS))

    B.net()                                   # build once, before the fork
    R.pooling()
    ctx = mp.get_context("fork")
    t0, done_positions = time.time(), 0
    with ctx.Pool(workers, initializer=_init) as pool:
        for k in range(n_shards):
            path = shard_path(k)
            if os.path.exists(path):
                continue
            lo, hi = k * SHARD_ROWS, min((k + 1) * SHARD_ROWS, len(fens))
            batch = [(i, fens[i]) for i in range(lo, hi)]
            rows = pool.map(_one, batch, chunksize=8)
            row_ids = np.array([r[0] for r in rows], dtype=np.int32)
            pooled = np.stack([r[1] for r in rows]).astype(np.float16)
            live = np.stack([r[2] for r in rows]).astype(np.float16)
            np.savez_compressed(path, pooled=pooled, live=live, row_ids=row_ids)
            done_positions += len(rows)
            rate = done_positions / max(time.time() - t0, 1e-6)
            left = (n_shards - k - 1) * SHARD_ROWS
            nz = float((pooled.astype(np.float32) > 0).mean())
            print("%s  %d/%d shards  %.1f pos/s  nonzero=%.3f  eta %.1f h"
                  % (path, k + 1, n_shards, rate, nz, left / max(rate, 1e-6) / 3600))
    print("done in %.1f h" % ((time.time() - t0) / 3600))
    return n_shards


def verify(expect_shards=None):
    """Shard count matches, no NaN, and the vectors are not all zero."""
    shards = sorted(f for f in os.listdir(SHARD_DIR) if f.endswith(".npz"))
    ok = bool(shards)
    total, nonzero = 0, 0.0
    for f in shards:
        blob = np.load(os.path.join(SHARD_DIR, f))
        pooled = blob["pooled"].astype(np.float32)
        if not np.isfinite(pooled).all():
            print("BAD non-finite values in", f)
            ok = False
        total += pooled.shape[0]
        nonzero += float((pooled > 0).sum())
    frac = nonzero / max(total * (blob["pooled"].shape[1] if shards else 1), 1)
    if expect_shards is not None and len(shards) != expect_shards:
        print("BAD %d shards, expected %d" % (len(shards), expect_shards))
        ok = False
    print("%s %d shards, %d rows, mean nonzero fraction %.3f"
          % ("OK " if ok else "BAD", len(shards), total, frac))
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None,
                    help="only the first N positions (smoke test)")
    ap.add_argument("--verify-only", action="store_true")
    args = ap.parse_args()
    if args.verify_only:
        raise SystemExit(0 if verify() else 1)
    n_shards = run(args.workers, args.limit)
    good = verify(n_shards)
    print("SELFCHECK", "PASS" if good else "FAIL")
    raise SystemExit(0 if good else 1)
