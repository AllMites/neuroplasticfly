"""Drive ORN channels and report the regime. Used by P0.3 and by every P1 run.

One call = one 300 ms batch. `channels` are selector strings already exported
in brain_gpu.npz (e.g. "cell_type=ORN_DA1,side=left"); `measure` drives each
channel separately in one batch and reports sparsity, overlap and ignition.
"""
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _HERE)
import gpu_sim as G

META = os.path.join(_HERE, "data", "neuron_meta.npz")
_meta = np.load(META, allow_pickle=False)
CC = _meta["cell_class"].astype(str)
CT = _meta["cell_type"].astype(str)
CEN = _meta["super_class"].astype(str) == "central"
KC = CC == "Kenyon_Cell"
ACTIVE_HZ = 1.0


def drive(sim, selector, hz):
    idx = sim.net.select(selector).astype(np.int64)
    return idx, np.full(len(idx), hz * G.DT / 1000.0, np.float64)


def measure(sim, selectors, hz, t_run=300.0, seed0=0):
    """Returns (rates [C, N] float32, report dict). One drive per selector."""
    drives = [drive(sim, s, hz) for s in selectors]
    # Same seed for every channel. The counter RNG hashes (seed, step, neuron),
    # so one seed gives both channels the same background noise stream and
    # the only difference between them is the odour. Distinct seeds
    # (range(seed0, seed0+C), the bug in every number before 2026-09-19 pm)
    # confounded odour with noise.
    counts = sim.run_batch(drives, [seed0] * len(drives), t_run=t_run)
    rates = counts.cpu().numpy().astype(np.float32) / (t_run / 1000.0)
    act = rates > ACTIVE_HZ
    rep = {"hz": hz, "selectors": list(selectors),
           "kc_active": [float(a[KC].mean()) for a in act],
           "kc_hz": [float(r[KC].mean()) for r in rates],
           "central_active": [float(a[CEN].mean()) for a in act],
           "apl_hz": [float(r[CT == "APL"].mean()) for r in rates]}
    rep["jaccard"] = jaccard_matrix(act[:, KC])
    return rates, rep


def jaccard_matrix(act_kc):
    """act_kc: bool [C, n_kc] -> C x C list of lists."""
    C = act_kc.shape[0]
    out = [[0.0] * C for _ in range(C)]
    for i in range(C):
        for j in range(C):
            inter = int((act_kc[i] & act_kc[j]).sum())
            union = int((act_kc[i] | act_kc[j]).sum())
            out[i][j] = float(inter / union) if union else 0.0
    return out


def pair_jaccard(rep):
    """The off-diagonal Jaccard for a two-selector measure()."""
    return rep["jaccard"][0][1]
