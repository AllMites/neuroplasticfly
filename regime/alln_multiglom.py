"""Standard two-odour multi-glomerular probe: two 8-glomerulus odours (2 shared)
driven into the left ORNs, per-layer active counts and Jaccard at ORN, ALLN,
ALPN, KC and central. With --negate the same probe is re-run with
gpu_sim.ELN_NEGATE on (the 44 immuno-confirmed cholinergic antennal-lobe eLNs
negated on the device weight copy at GpuSim.__init__). A single glomerulus
cannot drive a KC (5 KCs get >=3 DA1 PN inputs), so single-glomerulus Jaccard at
KC is not a test of anything downstream of the AL.
Run: .venv/Scripts/python.exe regime/alln_multiglom.py [--negate] [--hz 20 60 120] [--seeds 0 2 4]
"""
import argparse
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _HERE)
import gpu_sim as G
from regime import probe as PR
from regime.layers import LAYERS, jac

A = ["DA1", "VA1d", "VL1", "DL3", "DM1", "VA2", "DM3", "DM2"]
B = ["DM4", "VM5d", "VM4", "DL1", "V", "VA6", "DM3", "DM2"]   # 2 shared
SHOW = ["ORN", "ALLN", "ALPN", "KC", "central"]


def multi_drive(sim, gloms, hz):
    idx = np.concatenate([sim.net.select("cell_type=ORN_%s,side=left" % g) for g in gloms])
    return idx.astype(np.int64), np.full(len(idx), hz * G.DT / 1000.0, np.float64)


def run(sim, hz, seed):
    drives = [multi_drive(sim, A, hz), multi_drive(sim, B, hz)]
    counts = sim.run_batch(drives, [seed, seed], t_run=300.0)
    return counts.cpu().numpy().astype(np.float32) / 0.3 > PR.ACTIVE_HZ


def row_cells(act):
    """The per-layer `actA/actB J` cells of one probe row, in SHOW order.

    Hoisted so `regime/gap_sweep.py` prints byte-identical rows: a sweep whose
    rows do not line up with the probe's cannot be compared against them.
    """
    return ["%4d/%4d %.2f" % (act[0, m].sum(), act[1, m].sum(),
                              jac(act[0, m], act[1, m]))
            for n, m in LAYERS if n in SHOW]


def main(rates=(20, 60, 120), seeds=(0,), negate=False):
    prev = G.ELN_NEGATE
    variants = [("base", False)] + ([("ELN_NEGATE", True)] if negate else [])
    print("%-10s %5s %s  %s" % ("variant", "hz", " ".join("%16s" % n for n in SHOW), "seed"))
    try:
        for name, flag in variants:
            G.ELN_NEGATE = flag
            sim = G.GpuSim()
            for hz in rates:
                for seed in seeds:
                    act = run(sim, float(hz), seed)
                    cells = row_cells(act)
                    print("%-10s %5d %s  seed%d" % (
                        name, hz, " ".join("%16s" % c for c in cells), seed), flush=True)
            del sim
            torch.cuda.empty_cache()
    finally:
        G.ELN_NEGATE = prev


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--negate", action="store_true")
    p.add_argument("--hz", type=int, nargs="+", default=[20, 60, 120])
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    a = p.parse_args()
    main(a.hz, a.seeds, a.negate)
