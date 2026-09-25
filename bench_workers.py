"""Throwaway: how many precompute workers is this box actually worth?

Measures sims/second at several pool sizes on real positions. Builds the network
once and warms each pool before timing, so what is measured is steady-state
throughput, not import and page-in cost.

Run: docker compose run --rm sim python -u bench_workers.py
"""
import multiprocessing as mp
import os
import time

import chess
import numpy as np
import pyarrow.parquet as pq

import behaviors as B
import reservoir as R

N_SIMS = 96
COUNTS = [1, 8, 16, 24, 32]

_net = None


def _init():
    global _net
    _net = B.net()
    R.pooling(_net)


def _one(fen):
    return float(np.asarray(R.simulate(chess.Board(fen), n=_net)["pooled"]).sum())


if __name__ == "__main__":
    fens = pq.read_table("/app/data/positions.parquet",
                         columns=["fen"]).column("fen").to_pylist()[:N_SIMS + 64]
    print("cpu_count=%d  loadavg=%.1f" % (os.cpu_count(), os.getloadavg()[0]))
    B.net()
    R.pooling()

    ctx = mp.get_context("fork")
    rows = []
    for k in COUNTS:
        with ctx.Pool(k, initializer=_init) as pool:
            pool.map(_one, fens[N_SIMS:N_SIMS + k])        # warm every worker
            t0 = time.time()
            pool.map(_one, fens[:N_SIMS], chunksize=1)
            secs = time.time() - t0
        rate = N_SIMS / secs
        rows.append((k, rate, secs))
        base = rows[0][1]
        print("%2d workers  %6.2f sims/s  %5.1f s for %d sims  %4.1fx speedup  "
              "%3.0f%% efficiency  -> %.1f h for 200k"
              % (k, rate, secs, N_SIMS, rate / base, 100 * rate / base / k,
                 200_000 / rate / 3600))
    best = max(rows, key=lambda r: r[1])
    print("\nbest: %d workers at %.2f sims/s (%.1f h for 200k positions)"
          % (best[0], best[1], 200_000 / best[1] / 3600))
