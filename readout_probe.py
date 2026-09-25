"""Is `cell_type` pooling what loses the board, or was it already gone?

Measured 2026-09-16: the board is ~74% recoverable (occupied-F1) from the shipped
8,865-dim pooled vector, and a move head on it tops out at 0.092 against the
board planes' 0.269. A decoded board with a quarter of its piece-squares wrong
cannot play chess, which is exactly what the Elo run found.

The 139,248 -> 8,865 pooling averages every neuron of a cell type together, and
that is a readout choice, not the fly. This compares three readouts of the SAME
simulations at the same decoder budget:

  pooled    what ships: mean rate per cell_type group
  proj      a fixed sparse random projection to the same 8,865 dims
  raw       a random 8,865-neuron subset, no pooling at all

If `proj` or `raw` reconstructs the board much better than `pooled` at equal
width, the pooling is the lossy step and a different readout is worth new shards.
If all three are about equal, the loss is upstream in the encoding or the sim and
a new readout buys nothing.

Rates are reduced to 8,865 dims inside the worker and thrown away, so this never
holds the 139k x N array.

Run: docker compose run --rm sim python -u readout_probe.py --positions 5000
"""
import argparse
import multiprocessing as mp
import os

import chess
import numpy as np
import pyarrow.parquet as pq

import behaviors as B
import reservoir as R

DATA = "/app/data"
WIDTH = 8865
FAN = 64
_net = None
_proj = None
_subset = None


def _tables(n_neurons):
    """(projection index table, raw subset) -- fixed, so workers agree."""
    rng = np.random.default_rng(12345)
    proj = rng.integers(0, n_neurons, size=(WIDTH, FAN), dtype=np.int32)
    subset = rng.choice(n_neurons, WIDTH, replace=False).astype(np.int32)
    return proj, subset


def _init():
    global _net, _proj, _subset
    _net = B.net()
    R.pooling(_net)
    _proj, _subset = _tables(_net.n)


def _one(fen):
    out = R.simulate(chess.Board(fen), full=True, n=_net)
    rates = np.asarray(out["rates"], dtype=np.float32)
    return (np.asarray(out["pooled"], dtype=np.float32),
            rates[_proj].sum(axis=1) / FAN,
            rates[_subset])


def decode(X, occ, tr, val, tag, epochs=16):
    import torch
    import torch.nn as nn
    dev = "cuda" if torch.cuda.is_available() else "cpu"   # container torch is CPU-only
    X = np.log1p(np.clip(X, 0, None))
    keep = X.std(axis=0) > 1e-4
    X = X[:, keep]
    X = ((X - X.mean(0)) / (X.std(0) + 1e-6)).astype(np.float32)
    Xt, T = torch.from_numpy(X), torch.from_numpy(occ)
    m = nn.Sequential(nn.Linear(X.shape[1], 1024), nn.GELU(),
                      nn.Linear(1024, 768)).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.01)
    lf = nn.BCEWithLogitsLoss()
    best = 0.0
    for _ in range(epochs):
        m.train()
        o = tr[torch.randperm(len(tr)).numpy()]
        for i in range(0, len(o), 256):
            b = o[i:i + 256]
            opt.zero_grad(set_to_none=True)
            lf(m(Xt[b].to(dev)), T[b].to(dev)).backward()
            opt.step()
        m.eval()
        tp = fp = fn = 0.0
        with torch.no_grad():
            for i in range(0, len(val), 256):
                b = val[i:i + 256]
                p = (torch.sigmoid(m(Xt[b].to(dev))) > 0.5).float()
                t = T[b].to(dev)
                tp += float((p * t).sum())
                fp += float((p * (1 - t)).sum())
                fn += float(((1 - p) * t).sum())
        best = max(best, 2 * tp / max(2 * tp + fp + fn, 1.0))
    print("  %-46s occupied-F1 %.4f  (%d live dims)" % (tag, best, int(keep.sum())))
    return best


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--positions", type=int, default=5000)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    import encode

    fens = pq.read_table(os.path.join(DATA, "positions.parquet"),
                         columns=["fen"]).column("fen").to_pylist()
    rng = np.random.default_rng(0)
    picks = [fens[i] for i in rng.choice(len(fens), args.positions, replace=False)]

    B.net()
    R.pooling()
    ctx = mp.get_context("fork")
    got = []
    with ctx.Pool(args.workers, initializer=_init) as pool:
        for k, row in enumerate(pool.imap(_one, picks, chunksize=1), 1):
            got.append(row)
            if k % 500 == 0:
                print("  %d/%d" % (k, len(picks)), flush=True)

    occ = np.stack([encode.board_planes(chess.Board(f)).reshape(-1)[:768]
                    for f in picks]).astype(np.float32)
    n = len(picks)
    perm = np.random.default_rng(1).permutation(n)
    nv = max(200, int(n * 0.15))
    val, tr = perm[:nv], perm[nv:]

    print("\nboard reconstruction from %d sims, %d train / %d val, equal width:"
          % (n, len(tr), len(val)))
    for idx, tag in ((0, "pooled by cell_type (SHIPPED)"),
                     (1, "sparse random projection, same 8865 dims"),
                     (2, "raw 8865-neuron subset, no pooling")):
        decode(np.stack([g[idx] for g in got]), occ, tr, val, tag)
