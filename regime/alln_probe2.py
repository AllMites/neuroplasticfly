"""Which exc-ALLN edges carry the runaway, and does global gain work once the
loop is gone? Companion to alln_probe.py / alln_sweep.py.
Run: .venv/Scripts/python.exe regime/alln_probe2.py
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


def main(rates=(60, 120), seed=0):
    sim = G.GpuSim()
    w0 = sim.data.clone()
    ip, ci = sim.net.W_indptr, sim.net.W_indices
    alln = np.where(CC == "ALLN")[0]
    fp = np.array([(sim.net.W_data[ip[i]:ip[i + 1]] > 0).mean() if ip[i + 1] > ip[i]
                   else 0.0 for i in alln])
    exc = alln[fp > 0.5]
    r_exc = rows(sim, exc)
    post = CC[ci[r_exc]]
    dev = sim.device
    t = lambda a: torch.from_numpy(a).to(dev)
    r_all, r_ln, r_pn = t(r_exc), t(r_exc[post == "ALLN"]), t(r_exc[post == "ALPN"])
    r_other = t(r_exc[~np.isin(post, ["ALLN", "ALPN"])])
    print("exc ALLN out-edges %d: ->ALLN %d ->ALPN %d ->other %d" % (
        len(r_exc), r_ln.numel(), r_pn.numel(), r_other.numel()))

    def var(zero=None, gain=1.0):
        sim.data.copy_(w0 * gain)
        if zero is not None:
            sim.data.index_fill_(0, zero, 0.0)

    variants = [("base", dict()),
                ("zero LN->LN", dict(zero=r_ln)),
                ("zero LN->PN", dict(zero=r_pn)),
                ("zero LN->other", dict(zero=r_other)),
                ("zero all, x1.5", dict(zero=r_all, gain=1.5)),
                ("zero all, x2", dict(zero=r_all, gain=2.0)),
                ("zero all, x3", dict(zero=r_all, gain=3.0))]
    print("%-16s %5s %s" % ("variant", "hz", " ".join("%16s" % n for n in SHOW)))
    for name, kw in variants:
        var(**kw)
        for hz in rates:
            rr, _ = PR.measure(sim, SELECTORS, float(hz), seed0=seed)
            act = rr > PR.ACTIVE_HZ
            cells = ["%4d/%4d %.2f" % (act[0, m].sum(), act[1, m].sum(), jac(act[0, m], act[1, m]))
                     for n, m in LAYERS if n in SHOW]
            print("%-16s %5d %s" % (name, hz, " ".join("%16s" % c for c in cells)), flush=True)
    sim.data.copy_(w0)


if __name__ == "__main__":
    main()
