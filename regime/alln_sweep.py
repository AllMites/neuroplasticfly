"""Sweep exc-ALLN output scale x ORN drive. Companion to alln_probe.py.
Run: .venv/Scripts/python.exe regime/alln_sweep.py
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
from regime.alln_probe import rows

SHOW = ["ALLN", "ALPN", "KC", "central"]


def main(scales=(0.0, 0.25, 0.5, 0.75), rates=(60, 120, 200, 300), seed=0):
    sim = G.GpuSim()
    w0 = sim.data.clone()
    ip = sim.net.W_indptr
    alln = np.where(CC == "ALLN")[0]
    fp = np.array([(sim.net.W_data[ip[i]:ip[i + 1]] > 0).mean() if ip[i + 1] > ip[i]
                   else 0.0 for i in alln])
    r_exc = torch.from_numpy(rows(sim, alln[fp > 0.5])).to(sim.device)
    print("%-6s %5s %s" % ("scale", "hz", " ".join("%16s" % n for n in SHOW)))
    for sc in scales:
        sim.data.copy_(w0)
        sim.data.index_copy_(0, r_exc, w0[r_exc] * sc)
        for hz in rates:
            rates_, _ = PR.measure(sim, SELECTORS, float(hz), seed0=seed)
            act = rates_ > PR.ACTIVE_HZ
            cells = []
            for name, m in LAYERS:
                if name in SHOW:
                    cells.append("%4d/%4d %.2f" % (act[0, m].sum(), act[1, m].sum(),
                                                   jac(act[0, m], act[1, m])))
            print("%-6.2f %5d %s" % (sc, hz, " ".join("%16s" % c for c in cells)), flush=True)
    sim.data.copy_(w0)


if __name__ == "__main__":
    main()
