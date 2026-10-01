"""Recompute the per-odour held-out correlation of the chosen chunk-0 rate model (descriptive, for Fig 2a).

Uses rate/chunk0.py's own functions (apply, respond, pearson_nz, drives_of) on the chosen lin solution;
asserts the median reproduces results/rate_chunk0/result.json L2_median_r before writing fig2_r.json.
Deterministic engine, no training; needs a CUDA GPU and the built data/ (scripts/build_data.py).
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("NEUROPLASTICFLY_ROOT") or os.path.normpath(os.path.join(HERE, "..", ".."))
FC = ROOT
sys.path.insert(0, FC)
sys.path.insert(0, os.path.join(FC, "rate"))
import chunk0 as C0  # noqa: E402
from rate import suite as S  # noqa: E402
from rate.engine import RateSim  # noqa: E402


def main():
    res = json.load(open(os.path.join(C0.OUT, "result.json")))
    fam = res["families"]["lin"]
    cell = json.load(open(os.path.join(C0.CELLS, "lin_S%d.json" % fam["chosen"])))
    d, _ = C0.load_ref()
    names, split = d["names"].astype(str), d["split"].astype(str)
    held = np.flatnonzero(split != "cal")
    lif_on = d["rate_on"][held].mean(1).astype(np.float64)
    sim = RateSim(1.0, regime="eln8")
    C0.apply(sim, "lin", np.array(cell["x"]))
    on, _ = C0.respond(sim, C0.drives_of(names[held], S.C1.sets()["ct"]))
    r = [C0.pearson_nz(on[h], lif_on[h]) for h in range(len(held))]
    med, want = float(np.nanmedian(r)), fam["metrics"]["L2_median_r"]
    print("median r %.6f vs recorded %.6f" % (med, want))
    assert abs(med - want) < 1e-4, "per-odour recompute does not reproduce the recorded median"
    json.dump({"names": names[held].tolist(), "r": r, "median": med, "C": res["C"], "bar": fam["metrics"]["L2_bar"],
               "source": "chunk0 lin_S%d recomputed" % fam["chosen"]}, open(os.path.join(HERE, "fig2_r.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
