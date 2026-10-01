"""Data-derived plausibility bound for fitted gains (path 2, chunk-1 fit prereg input).

Question: in a PUBLISHED fitted connectome model, how far does training push per-type synaptic efficacy
away from "synapse count alone sets the weight"? Source: Lappalainen et al. 2024 flyvis ensemble
flow/0000 (50 trained models, 604 edge types, 65 cell types), weights = sign x count x alpha, alpha
trained per (source type, target type) pair.

If counts were sufficient, one global alpha would fit every edge type; our engine's default gain = 1
assumes exactly that, with the global scale fixed in w_scale. So the reference distribution is
alpha / median(alpha) within each model:
  - edge level: one value per (source, target) type pair  -> bound for a per-edge-type gain;
  - target level: synapse-count-weighted mean over each target type's inputs -> bound for our
    per-pool (per-target) gain.
trained/init is printed too but is NOT the reference: init = 0.01/<N> is an arbitrary normalisation,
so the ratio mixes in the init's own 1/<N> spread and a global rescale.

Run (flyvis venv, which has flyvis data + h5py):
  ../flyvis/.venv/Scripts/python.exe rate/gain_bound.py
"""
import glob
import json
import os

import h5py
import numpy as np
import pandas as pd
import torch

FLYVIS = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                      "flyvis", "data")
EDGES = os.path.join(FLYVIS, "connectome", "ConnectomeFromAvgFilters_0000", "edges")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gain_bound.json")
Q = [50, 90, 95, 99, 99.9]


def rd(k):
    with h5py.File(os.path.join(EDGES, k + ".h5"), "r") as f:
        node = f[list(f.keys())[0]]
        v = node[:] if isinstance(node, h5py.Dataset) else node[list(node.keys())[0]][:]
    return np.array([x.decode() for x in v]) if v.dtype.kind in "SO" else v


def main():
    e = pd.DataFrame({k: rd(k) for k in ["source_type", "target_type", "n_syn"]})
    # same grouping and order as flyvis SynapseCountScaling (initialization.py: groupby sort=False)
    g = e.groupby(["source_type", "target_type"], as_index=False, sort=False).mean()
    init = 0.01 / g.n_syn.values
    paths = sorted(glob.glob(os.path.join(FLYVIS, "results", "flow", "0000", "[0-9]" * 3, "best_chkpt")))
    edge, tgt, ratio = [], [], []
    for p in paths:
        a = torch.load(p, map_location="cpu", weights_only=False)["network"]["edges_syn_strength"].numpy()
        assert len(a) == len(g), (len(a), len(g))
        rel = a / np.median(a)
        edge.append(rel)
        ratio.append(a / init)
        df = pd.DataFrame({"t": g.target_type.values, "a": rel, "w": g.n_syn.values})
        t = df.groupby("t").apply(lambda x: np.average(x.a, weights=x.w), include_groups=False).values
        tgt.append(t / np.median(t))
    edge, tgt, ratio = map(np.array, (edge, tgt, ratio))
    pct = lambda x: dict(zip([str(q) for q in Q], np.percentile(x.ravel(), Q).round(3).tolist()))  # noqa: E731
    out = {"source": "flyvis flow/0000, %d models, %d edge types, %d target types"
                     % (len(paths), len(g), tgt.shape[1]),
           "edge_level": dict(pct(edge), max=float(edge.max()), frac_zero=float((edge == 0).mean())),
           "target_level": dict(pct(tgt), max=float(tgt.max())),
           "trained_over_init_NOT_reference": dict(pct(ratio), max=float(ratio.max()))}
    json.dump(out, open(OUT, "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
