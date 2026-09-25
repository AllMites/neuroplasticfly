"""Flip only the excitatory ALLNs whose sign has no literature support.

data/nt_conf.npz (FlyWire v783 annotation TSV, sorted by root_id == neuron order,
verified on cell_class/cell_type/side/super_class) gives top_nt, top_nt_conf and
known_nt. Variants: flip all 134 / flip the 90 without known_nt=acetylcholine /
flip the 54 predicted serotonin, dopamine or octopamine.
The 44-neuron case is now gpu_sim.ELN_NEGATE; this file keeps the in-memory flip
because it sweeps sets the switch cannot express (134 / 90 / 54 / 21).
Run: .venv/Scripts/python.exe regime/alln_known.py
"""
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _HERE)
import gpu_sim as G
from regime import probe as PR
from regime.layers import LAYERS, jac, CC
from regime.alln_probe import rows
from regime.alln_multiglom import run, SHOW


def main(hz=60.0, seeds=(0, 2)):
    nt = np.load(os.path.join(_HERE, "data", "nt_conf.npz"))
    top, known = nt["top_nt"].astype(str), nt["known_nt"].astype(str)
    sim = G.GpuSim()
    w0 = sim.data.clone()
    ip = sim.net.W_indptr
    alln = np.where(CC == "ALLN")[0]
    fp = np.array([(sim.net.W_data[ip[i]:ip[i + 1]] > 0).mean() if ip[i + 1] > ip[i]
                   else 0.0 for i in alln])
    exc = alln[fp > 0.5]
    sets = {"base": exc[:0],
            "flip_all134": exc,
            "flip_unsupported": exc[known[exc] != "acetylcholine"],
            "flip_nonACh_pred": exc[~np.isin(top[exc], ["acetylcholine"])]}
    print("%-18s %4s %3s %s" % ("variant", "n", "hz", " ".join("%16s" % n for n in SHOW)))
    for name, ns in sets.items():
        sim.data.copy_(w0)
        if len(ns):
            r = torch.from_numpy(rows(sim, ns)).to(sim.device)
            sim.data.index_copy_(0, r, -w0[r])
        for s in seeds:
            act = run(sim, hz, s)
            cells = ["%4d/%4d %.2f" % (act[0, m].sum(), act[1, m].sum(), jac(act[0, m], act[1, m]))
                     for n, m in LAYERS if n in SHOW]
            print("%-18s %4d %3d %s  seed%d" % (name, len(ns), hz, " ".join("%16s" % c for c in cells), s), flush=True)
    sim.data.copy_(w0)


if __name__ == "__main__":
    main()
