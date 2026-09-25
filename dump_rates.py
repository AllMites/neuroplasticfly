"""Save full 139k rate vectors for N positions, once.

Every readout question so far has cost a fresh batch of simulations, because
precompute only ever stored the 8,865-dim `pooled` view. With the raw rates on
disk, any candidate readout -- any grouping, projection, subset or region mask --
can be scored in seconds on the host without touching the sim.

float16 at 139,248 neurons is 272 KB per position, so 5,000 positions is ~1.4 GB.

Run: docker compose run --rm sim python -u dump_rates.py --positions 5000
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
OUT = os.path.join(DATA, "rates_dump")
CHUNK = 500
_net = None


def _init():
    global _net
    _net = B.net()
    R.pooling(_net)


def _one(args):
    row_id, fen = args
    out = R.simulate(chess.Board(fen), full=True, n=_net)
    return row_id, np.asarray(out["rates"], dtype=np.float16)


def run(n_pos, workers):
    os.makedirs(OUT, exist_ok=True)
    fens = pq.read_table(os.path.join(DATA, "positions.parquet"),
                         columns=["fen"]).column("fen").to_pylist()
    rng = np.random.default_rng(0)
    pick = np.sort(rng.choice(len(fens), n_pos, replace=False))
    jobs = [(int(i), fens[int(i)]) for i in pick]

    B.net()
    R.pooling()
    ctx = mp.get_context("fork")
    with ctx.Pool(workers, initializer=_init) as pool:
        buf, ids, part = [], [], 0
        for k, (row_id, rates) in enumerate(pool.imap(_one, jobs, chunksize=1), 1):
            buf.append(rates)
            ids.append(row_id)
            if len(buf) == CHUNK or k == len(jobs):
                np.savez(os.path.join(OUT, "rates_%03d.npz" % part),
                         rates=np.stack(buf), row_ids=np.array(ids, dtype=np.int32))
                print("  wrote part %d (%d/%d)" % (part, k, len(jobs)), flush=True)
                buf, ids, part = [], [], part + 1

    # the board itself, so the host never needs python-chess over the FEN list
    occ = np.stack([encode.board_planes(chess.Board(fens[int(i)])).reshape(-1)
                    for i in pick]).astype(np.float32)
    np.savez_compressed(os.path.join(OUT, "targets.npz"),
                        planes=occ, row_ids=pick.astype(np.int32),
                        fens=np.array([fens[int(i)] for i in pick]))
    print("wrote %s: %d positions, %d parts" % (OUT, len(jobs), part))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--positions", type=int, default=5000)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    run(args.positions, args.workers)
