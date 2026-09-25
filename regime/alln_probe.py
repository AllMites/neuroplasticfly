"""Is the excitatory ALLN recurrent loop what merges the two odours at the AL?

In-memory weight variants on one GpuSim (nothing on disk changes). 134 of 429
ALLNs carry positive out-edges (FlyWire top_nt says cholinergic); their
ALLN->ALLN and ALLN->ALPN excitation is the largest input onto the 345 ALPNs
that fire for DA1 without receiving a DA1 ORN synapse.

Run: .venv/Scripts/python.exe regime/alln_probe.py [hz] [seed]
"""
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _HERE)
import gpu_sim as G
from regime import probe as PR
from regime.layers import LAYERS, SELECTORS, jac, CC


def rows(sim, neurons):
    ip = sim.net.W_indptr
    return np.concatenate([np.arange(ip[i], ip[i + 1]) for i in neurons])


def main(hz=60.0, seed=0):
    sim = G.GpuSim()
    w0 = sim.data.clone()
    w_np = sim.net.W_data
    ip = sim.net.W_indptr
    alln = np.where(CC == "ALLN")[0]
    fp = np.array([(w_np[ip[i]:ip[i + 1]] > 0).mean() if ip[i + 1] > ip[i] else 0.0
                   for i in alln])
    exc = alln[fp > 0.5]
    print("ALLN %d, excitatory %d" % (len(alln), len(exc)))
    r_exc = torch.from_numpy(rows(sim, exc)).to(sim.device)
    r_all = torch.from_numpy(rows(sim, alln)).to(sim.device)
    variants = {
        "base": lambda: None,
        "exc_alln_zero": lambda: sim.data.index_fill_(0, r_exc, 0.0),
        "exc_alln_flip": lambda: sim.data.index_copy_(0, r_exc, -w0[r_exc]),
        "all_alln_zero": lambda: sim.data.index_fill_(0, r_all, 0.0),
    }
    print("%-15s %s" % ("variant", " ".join("%12s" % n for n, _ in LAYERS)))
    for name, fn in variants.items():
        sim.data.copy_(w0)
        fn()
        rates, _ = PR.measure(sim, SELECTORS, hz, seed0=seed)
        act = rates > PR.ACTIVE_HZ
        cells = []
        for _, m in LAYERS:
            cells.append("%3d/%3d %.2f" % (act[0, m].sum(), act[1, m].sum(),
                                           jac(act[0, m], act[1, m])))
        print("%-15s %s" % (name, " ".join("%12s" % c for c in cells)))
    sim.data.copy_(w0)


if __name__ == "__main__":
    a = sys.argv[1:]
    main(float(a[0]) if a else 60.0, int(a[1]) if len(a) > 1 else 0)
