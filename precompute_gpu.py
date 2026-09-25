"""precompute.py's shard writer, on the GPU engine. Same schema, different dir.

Kept as its own file rather than a `--device gpu` flag inside `precompute.py`
because the 200k CPU run is live and every process in it imports that module.
The one-line integration is described in `.claude/PRPs/reports/gpu-sim-port.md`.

Shards are byte-compatible with the CPU ones: 1,000 rows each, float16,
`row_ids` the identity slice of `data/positions.parquet`, and one key per readout
`reservoir.views` currently keeps (`pooled` + `live` as of 2026-09-16). The keys
are never hardcoded here - the shard writer calls `reservoir.views` itself, so a
readout redesign lands on both engines at once. Provenance goes in a sidecar
`ENGINE.json` rather than an extra npz key, so `train.py`'s key sniffing sees
exactly what it saw before.

NEVER point this at the CPU shard directory. Two engines in one directory makes
the provenance unrecoverable; set FLYCHESS_SHARDS first.

Drive vectors are built on CPU (pure python over ~3.8k neurons, measured ~8 ms a
position) in a small fork/spawn pool while the GPU integrates the previous
batch; the default of 4 workers is deliberate, the box is shared.

Run: FLYCHESS_SHARDS=data/reservoir_gpu uv run precompute_gpu.py --batch 1024
"""
import argparse
import json
import os
import time

import numpy as np

DATA = os.environ.get("FLYCHESS_DATA", "data")
SHARD_DIR = os.environ.get("FLYCHESS_SHARDS", os.path.join(DATA, "reservoir_gpu"))
SHARD_ROWS = 1000

_net = None


def _drive(fen):
    """Worker side: (stim_idx, stim_prob, seed). No torch, no CUDA context."""
    global _net
    import chess
    import gpu_sim as G
    import reservoir as R
    if _net is None:
        _net = G.brain()
    board = chess.Board(fen)
    idx, prob = G.drive_of(board, _net)
    return idx, prob, R.seed_for(board)


def shard_path(k):
    return os.path.join(SHARD_DIR, "shard_%04d.npz" % k)


def run(batch, limit=None, workers=4, t_run=None, compress=True):
    import multiprocessing as mp

    import pyarrow.parquet as pq
    import torch

    import gpu_sim as G
    import reservoir as R

    t_run = t_run or R.T_RUN
    if os.path.abspath(SHARD_DIR) == os.path.abspath(
            os.path.join(DATA, "reservoir")):
        raise SystemExit("refusing to write GPU shards into the CPU shard "
                         "directory; set FLYCHESS_SHARDS")
    os.makedirs(SHARD_DIR, exist_ok=True)

    fens = pq.read_table(os.path.join(DATA, "positions.parquet"),
                         columns=["fen"]).column("fen").to_pylist()
    if limit:
        fens = fens[:limit]
    n_shards = (len(fens) + SHARD_ROWS - 1) // SHARD_ROWS
    print("%d positions -> %d shards of %d, batch %d, engine gpu"
          % (len(fens), n_shards, SHARD_ROWS, batch))

    sim = G.GpuSim()
    json.dump({"engine": "gpu", "module": "gpu_sim.py", "rng": "counter",
               "t_run_ms": t_run, "device": torch.cuda.get_device_name(0),
               "written": time.strftime("%Y-%m-%d %H:%M:%S")},
              open(os.path.join(SHARD_DIR, "ENGINE.json"), "w"), indent=1)

    buf = {"row_ids": []}                    # view keys appear on the first batch
    next_shard, t0, done = 0, time.time(), 0
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers) as pool:
        for lo in range(0, len(fens), batch):
            chunk = fens[lo:lo + batch]
            built = pool.map(_drive, chunk, chunksize=16)
            drives = [(d[0], d[1]) for d in built]
            seeds = [d[2] for d in built]
            counts = sim.run_batch(drives, seeds, t_run=t_run, rng="counter")
            views = sim.views_of(counts, t_run)
            for k, v in views.items():       # whatever reservoir.views keeps
                buf.setdefault(k, []).append(v.astype(np.float16))
            buf["row_ids"].append(np.arange(lo, lo + len(chunk), dtype=np.int32))
            done += len(chunk)

            rows = sum(len(x) for x in buf["row_ids"])
            while rows >= SHARD_ROWS or (lo + batch >= len(fens) and rows):
                merged = {k: np.concatenate(v) for k, v in buf.items()}
                take = min(SHARD_ROWS, rows)
                blob = {k: merged[k][:take] for k in merged}
                path = shard_path(next_shard)
                (np.savez_compressed if compress else np.savez)(path, **blob)
                buf = {k: [merged[k][take:]] for k in merged}
                rows -= take
                next_shard += 1
                rate = done / max(time.time() - t0, 1e-6)
                print("%s  %d/%d shards  %.1f pos/s  eta %.2f h"
                      % (path, next_shard, n_shards, rate,
                         (len(fens) - done) / max(rate, 1e-6) / 3600))
    print("done in %.2f h" % ((time.time() - t0) / 3600))
    return next_shard


def compare_to_cpu(cpu_dir, gpu_dir=None, rows=1000):
    """r between GPU and CPU shards over the rows both directories contain."""
    gpu_dir = gpu_dir or SHARD_DIR
    out = {}
    ca = np.load(os.path.join(cpu_dir, "shard_0000.npz"))
    cb = np.load(os.path.join(gpu_dir, "shard_0000.npz"))
    for view in [k for k in ca.files if k != "row_ids" and k in cb.files]:
        a = ca[view][:rows]
        b = cb[view][:rows]
        a, b = a.astype(np.float64), b.astype(np.float64)
        m = min(len(a), len(b))
        a, b = a[:m].ravel(), b[:m].ravel()
        a1, b1 = a - a.mean(), b - b.mean()
        d = np.linalg.norm(a1) * np.linalg.norm(b1)
        out[view] = {"rows": int(m), "pearson": round(float(a1 @ b1 / d), 6),
                     "mean_abs_diff": round(float(np.abs(a - b).mean()), 5)}
        print("  %-7s r=%.5f over %d rows, mean|d|=%.4f"
              % (view, out[view]["pearson"], m, out[view]["mean_abs_diff"]))
    return out


def compare_to_reference(gpu_dir=None, ref="data/ref_dump.npz"):
    """Compare GPU shard rows against flypoke counts run through reservoir.views.

    Preferred over compare_to_cpu because it does not depend on a CPU shard
    directory that a live precompute may be rewriting: `ref_dump.npz` holds the
    reference engine's own spike counts for rows 0..63 of positions.parquet at
    the canonical seeds, and `reservoir.views` is the same code precompute.py
    calls, so this is precisely "what would the CPU shard have contained".
    """
    import gpu_sim as G
    G.brain()
    import reservoir as R
    gpu_dir = gpu_dir or SHARD_DIR
    blob = np.load(os.path.join(gpu_dir, "shard_0000.npz"))
    rd = np.load(ref, allow_pickle=False)
    counts, t = rd["counts"], float(rd["t_run"])
    m = min(len(counts), blob["row_ids"].shape[0])
    cpu = [R.views(counts[i] / (t / 1000.0), G.brain()) for i in range(m)]
    out = {}
    for view in [k for k in blob.files if k != "row_ids" and k in cpu[0]]:
        a = np.stack([np.asarray(c[view]) for c in cpu]).astype(np.float64).ravel()
        b = blob[view][:m].astype(np.float64).ravel()
        a1, b1 = a - a.mean(), b - b.mean()
        d = np.linalg.norm(a1) * np.linalg.norm(b1)
        out[view] = {"rows": int(m), "pearson": round(float(a1 @ b1 / d), 6),
                     "mean_abs_diff": round(float(np.abs(a - b).mean()), 5)}
        print("  %-7s r=%.5f over %d rows vs flypoke, mean|d|=%.4f"
              % (view, out[view]["pearson"], m, out[view]["mean_abs_diff"]))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--compare-cpu", type=str, default=None,
                    help="CPU shard dir to correlate the first shard against")
    ap.add_argument("--compare-ref", action="store_true",
                    help="correlate shard 0 against flypoke's own counts")
    a = ap.parse_args()
    if a.compare_ref:
        compare_to_reference()
    elif a.compare_cpu:
        compare_to_cpu(a.compare_cpu)
    else:
        n = run(a.batch, a.limit, a.workers)
        # The existing checker, unmodified: it reads FLYCHESS_SHARDS too, and on
        # the host `gpu_sim.brain()` has already stubbed flypoke so the import
        # works without the container.
        import gpu_sim  # noqa: F401  (installs the stubs precompute needs)
        gpu_sim.brain()
        import precompute
        good = precompute.verify(n)
        print("SELFCHECK", "PASS" if good else "FAIL")
        raise SystemExit(0 if good else 1)
