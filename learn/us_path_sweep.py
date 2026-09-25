"""check_us_path across seeds, with KC/central active fractions. Matrix from FLYCHESS_BRAIN.
Run: FLYCHESS_BRAIN=data/brain_gpu_danfloor1.npz .venv/Scripts/python.exe learn/us_path_sweep.py --seeds 10 --tag danfloor1
"""
import argparse
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _HERE); sys.path.insert(0, os.path.join(_HERE, "flywatch"))
import gpu_sim as G
from learn import condition as C
from learn import plastic as PL

# Same threshold and DAN mapping as condition.check_us_path (condition.py:99-107).
ARM_DAN = (("punish", "ppl1_hz"), ("reward", "pam_hz"))
THRESH = 1.0


def mn_index(spec):
    """spec: 'class:<cell_class>' | 'subclass:<cell_sub_class>' | '<cell_type>'."""
    meta = np.load(os.path.join(_HERE, "data", "neuron_meta.npz"), allow_pickle=False)
    if spec.startswith("class:"):
        col, val = meta["cell_class"].astype(str), spec.split(":", 1)[1]
    elif spec.startswith("subclass:"):
        col, val = meta["cell_sub_class"].astype(str), spec.split(":", 1)[1]
    else:
        col, val = meta["cell_type"].astype(str), spec
    idx = np.flatnonzero(col == val).astype(np.int64)
    assert len(idx) > 0, "no neuron matches --mn %s" % spec
    return idx


def stats(vals):
    a = np.asarray(vals, np.float64)
    return float(a.mean()), float(a.std(ddof=0)), float(a.min()), float(a.max())


def row(label, vals, n_over=None, n=None):
    m, sd, lo, hi = stats(vals)
    over = "-" if n_over is None else "%d/%d" % (n_over, n)
    return "| %-38s | %9.3f | %6.3f | %7.3f - %7.3f | %s |" % (label, m, sd, lo, hi, over)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--mn", default=None,
                    help="cell_type, or class:<cell_class>, or subclass:<cell_sub_class>")
    a = ap.parse_args()

    t0 = time.time()
    # Resolved before the brain load so a bad --mn fails in a second, not a minute.
    mn = mn_index(a.mn) if a.mn else None
    sim = G.GpuSim(); P = PL.Plastic.real()

    rows = []
    for seed in range(a.seeds):
        r_base = C.trial(sim, "misery", seed=seed)
        base = C.readout(r_base, P)
        rec = {"seed": seed,
               "base": {k: base[k] for k in ("pam_hz", "ppl1_hz", "kc_active", "central_active")}}
        if mn is not None:
            rec["base"]["mn_hz"] = float(r_base[mn].mean())
        for us, dan in ARM_DAN:
            rates = C.trial(sim, "misery", us=us, us_path="grn", seed=seed)
            r = C.readout(rates, P)
            arm = {k: r[k] for k in ("pam_hz", "ppl1_hz", "kc_active", "central_active")}
            if mn is not None:
                arm["mn_hz"] = float(rates[mn].mean())
            arm["ok"] = bool(r[dan] > base[dan] + THRESH)
            rec[us] = arm
        rec["decision"] = "grn" if all(rec[us]["ok"] for us, _ in ARM_DAN) else "dan"
        rows.append(rec)
        print("seed %d: punish ok=%s reward ok=%s -> %s"
              % (seed, rec["punish"]["ok"], rec["reward"]["ok"], rec["decision"]))

    b = np.load(G.BRAIN, allow_pickle=False)
    dan_floor = int(b["dan_floor"]) if "dan_floor" in b.files else None
    out = {"brain": os.path.abspath(G.BRAIN), "n_edges": int(sim.net.n_edges), "dan_floor": dan_floor,
           "seeds": a.seeds, "mn": a.mn, "mn_n": None if mn is None else int(len(mn)),
           "rows": rows}
    os.makedirs(C.RESULTS, exist_ok=True)
    path = os.path.join(C.RESULTS, "us_path_sweep_%s.json" % a.tag)
    json.dump(out, open(path, "w"), indent=1)

    N = a.seeds
    print("")
    print("brain=%s  n_edges=%d  dan_floor=%s  seeds=%d" % (G.BRAIN, out["n_edges"], dan_floor, N))
    print("| arm    | population                             | mean (Hz) |     SD |           range | over thr |")
    print("|--------|----------------------------------------|-----------|--------|-----------------|----------|")
    print("| punish " + row("PPL1 under 150 Hz bitter",
                            [r["punish"]["ppl1_hz"] for r in rows],
                            sum(r["punish"]["ok"] for r in rows), N))
    print("| reward " + row("PAM under 150 Hz sugar",
                            [r["reward"]["pam_hz"] for r in rows],
                            sum(r["reward"]["ok"] for r in rows), N))
    print("| reward " + row("PAM at baseline", [r["base"]["pam_hz"] for r in rows]))
    print("| punish " + row("PPL1 at baseline", [r["base"]["ppl1_hz"] for r in rows]))
    if mn is not None:
        print("| reward " + row("MN (%s, n=%d) under sugar" % (a.mn, len(mn)),
                                [r["reward"]["mn_hz"] for r in rows]))
        print("| base   " + row("MN (%s, n=%d) at baseline" % (a.mn, len(mn)),
                                [r["base"]["mn_hz"] for r in rows]))
    print("")
    print("decision: grn %d/%d, dan %d/%d"
          % (sum(r["decision"] == "grn" for r in rows), N,
             sum(r["decision"] == "dan" for r in rows), N))
    print("")
    # kc_active cannot move: fingerprint.kc_drive (learn/fingerprint.py:57-65) always drives
    # exactly round(SPARSITY * n_KC) KCs at KC_HZ=50, so (rates[KC] > 1).mean() is pinned at
    # SPARSITY=0.0500 in every arm. Reported anyway as a check that the authored drive landed.
    print("| regime   | kc_active (authored, invariant) | central_active |")
    print("|----------|---------------------------------|----------------|")
    for key, lab in (("base", "baseline"), ("punish", "bitter"), ("reward", "sugar")):
        print("| %-8s | %31.4f | %14.4f |"
              % (lab, np.mean([r[key]["kc_active"] for r in rows]),
                 np.mean([r[key]["central_active"] for r in rows])))
    print("")
    print("wrote %s  (%.1f s)" % (path, time.time() - t0))


if __name__ == "__main__":
    main()
