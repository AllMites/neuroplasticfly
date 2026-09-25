"""Run the reference engine (flypoke, in the container) and freeze its answers.

`verify_gpu.py` on the host cannot import flypoke, so the oracle has to be
recorded first. This dumps raw spike COUNTS, not rates or pooled vectors, so the
host can divide them exactly the way `SpikeRecord.rates` does and then run the
real `reservoir.views` on both sides.

Also dumps the exact `(stim_idx, stim_prob)` arrays `run_trial` built, so the
host's drive builder can be checked against the container's byte for byte before
any dynamics are compared. If the drives disagree, nothing downstream means
anything.

Run: docker compose run --rm sim python -u ref_dump.py --fens 64 --workers 4
"""
import argparse
import multiprocessing as mp
import os

import chess
import numpy as np
import pyarrow.parquet as pq

from flypoke import sim as S

import behaviors as B
import encode
import reservoir as R

OUT = os.environ.get("FLYCHESS_REF", "/app/data/ref_dump.npz")
_net = None


def _init():
    global _net
    _net = B.net()


def _counts(stims, seed, t_run, n):
    rec = S.run(n, stims, S.Params(t_run=t_run), n_trials=1, seed=seed)
    return np.bincount(rec.ids, minlength=n.n).astype(np.int32)


def _one(args):
    fen, seed, t_run = args
    n = _net
    stims = encode.stimuli(chess.Board(fen), n)
    idx = (np.concatenate([s.indices for s in stims]) if stims
           else np.empty(0, dtype=np.int64))
    prob = (np.concatenate([np.full(len(s.indices), s.rate * 0.1 / 1000.0)
                            for s in stims]) if stims else np.empty(0))
    return idx.astype(np.int64), prob.astype(np.float64), \
        _counts(stims, seed, t_run, n)


def _behavior(args):
    sel, rate, seed, t_run = args
    n = _net
    stims = ([] if sel is None
             else [S.Stimulus(n.select(sel).astype(np.int64), rate)])
    return _counts(stims, seed, t_run, n)


def main(n_fens, workers, t_run, extra_seeds):
    n = B.net()
    R.pooling(n)
    fens = pq.read_table("/app/data/positions.parquet",
                         columns=["fen"]).column("fen").to_pylist()[:n_fens]
    jobs = [(f, R.seed_for(chess.Board(f)), t_run) for f in fens]
    # Repeat runs of the first few positions with different seeds: this is the
    # CPU engine's own seed-noise floor, the bar the GPU has to beat.
    reruns = [(fens[i], s, t_run) for i in range(extra_seeds) for s in (1, 2)]

    ctx = mp.get_context("fork")
    with ctx.Pool(workers, initializer=_init) as pool:
        rows = pool.map(_one, jobs, chunksize=1)
        rr = pool.map(_one, reruns, chunksize=1) if reruns else []
        bh = pool.map(_behavior,
                      [(sel, rate, 0, 1000.0)
                       for sel, rate in B.STIMULI.values()], chunksize=1)

    idx = np.concatenate([r[0] for r in rows])
    prob = np.concatenate([r[1] for r in rows])
    offs = np.cumsum([0] + [len(r[0]) for r in rows]).astype(np.int64)
    blob = {"fens": np.array(fens), "t_run": np.float64(t_run),
            "seeds": np.array([j[1] for j in jobs], dtype=np.int64),
            "stim_idx": idx, "stim_prob": prob, "stim_offsets": offs,
            "counts": np.stack([r[2] for r in rows]),
            "rerun_counts": (np.stack([r[2] for r in rr]) if rr
                             else np.zeros((0, n.n), dtype=np.int32)),
            "rerun_seeds": np.array([j[1] for j in reruns], dtype=np.int64),
            "rerun_fen_index": np.array([i for i in range(extra_seeds)
                                         for _ in (1, 2)], dtype=np.int64),
            "behavior_names": np.array(list(B.STIMULI)),
            "behavior_counts": np.stack(bh),
            "behavior_t_run": np.float64(1000.0)}
    np.savez(OUT, **blob)
    print("wrote %s: %d fens, %d reruns, %d behaviours"
          % (OUT, len(fens), len(rr), len(bh)))
    print("  total spikes per sim: mean %.0f"
          % blob["counts"].sum(axis=1).mean())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fens", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--t-run", type=float, default=R.T_RUN)
    ap.add_argument("--extra-seeds", type=int, default=4,
                    help="positions re-run under seeds 1 and 2 for the noise floor")
    a = ap.parse_args()
    main(a.fens, a.workers, a.t_run, a.extra_seeds)
