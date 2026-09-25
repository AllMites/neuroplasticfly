"""Per-layer Jaccard between two ORN channels. Next-step 4 of the handoff:
measure at the antennal lobe, not three synapses past it.

Run: .venv/Scripts/python.exe regime/layers.py [hz] [seeds...]
"""
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _HERE)
import gpu_sim as G
from regime import probe as PR

SELECTORS = ["cell_type=ORN_DA1,side=left", "cell_type=ORN_DM4,side=left"]
_m = np.load(os.path.join(_HERE, "data", "neuron_meta.npz"), allow_pickle=False)
CC = _m["cell_class"].astype(str)
SUB = _m["cell_sub_class"].astype(str)
LAYERS = [("ORN", CC == "olfactory"), ("ALLN", CC == "ALLN"),
          ("ALPN", CC == "ALPN"), ("uPN", (CC == "ALPN") & (SUB == "uniglomerular")),
          ("LHLN", CC == "LHLN"), ("KC", CC == "Kenyon_Cell"),
          ("MBON", CC == "MBON"), ("central", PR.CEN)]


# Criterion-1 thresholds, verbatim from the Success Metrics table of
# `.claude/PRPs/prds/eln-electrical-coupling.prd.md`. One copy, so a probe and a
# sweep cannot disagree about what passing means. Pairs are inclusive bands.
CRIT1 = dict(alpn_j=0.35,            # ALPN Jaccard <=, at SCORE_HZ
             kc_j=0.20,              # KC Jaccard <=
             kc_act=(0.02, 0.10),    # KC active fraction, BOTH odours, EVERY seed
             central=(0.01, 0.05),   # central active fraction
             cen120=0.08)            # central active fraction <, at IGNITION_HZ


def jac(a, b):
    u = int((a | b).sum())
    return float((a & b).sum() / u) if u else 0.0


def main(hz=60.0, seeds=(0, 2, 4)):
    sim = G.GpuSim()
    rows = {name: [] for name, _ in LAYERS}
    for s in seeds:
        rates, _ = PR.measure(sim, SELECTORS, hz, seed0=s)
        act = rates > PR.ACTIVE_HZ
        for name, m in LAYERS:
            rows[name].append((int(act[0, m].sum()), int(act[1, m].sum()),
                               jac(act[0, m], act[1, m])))
    print("hz %g seeds %s (same seed both channels)" % (hz, list(seeds)))
    print("%-8s %6s %6s %6s  %s" % ("layer", "n", "DA1", "DM4", "jaccard mean (per seed)"))
    for name, m in LAYERS:
        r = np.array(rows[name])
        print("%-8s %6d %6.0f %6.0f  %.3f  (%s)" % (
            name, int(m.sum()), r[:, 0].mean(), r[:, 1].mean(), r[:, 2].mean(),
            " ".join("%.3f" % j for j in r[:, 2])))


if __name__ == "__main__":
    a = sys.argv[1:]
    main(float(a[0]) if a else 60.0, tuple(int(x) for x in a[1:]) or (0, 2, 4))
