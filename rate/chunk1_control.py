"""Chunk-1 unfitted control, exactly as PREREGISTER_rate_chunk1_control.md (194be67).

Two arms (A = LIF-matched, primary; B = Lappalainen default), five conditions, one deterministic run
each, the 11-row battery from rate/chunk1_targets.md. No parameter is fitted or tuned here.

Run: .venv/Scripts/python.exe tools/gpu_slot.py -- .venv/Scripts/python.exe rate/chunk1_control.py
--base c0: one arm "C0" on the chunk-0 base (PREREGISTER_rate_chunk1_control_c0.md), output
results/rate_chunk1_control_c0/; arms A/B untouched.
"""
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _HERE)
from rate.engine import RateSim  # noqa: E402
from rate.chunk1 import (HZ, T_RUN, battery, lif_constants, measure, selftest,  # noqa: E402
                         sets)

OUT = os.path.join(_HERE, "results", "rate_chunk1_control")


def main():
    selftest()
    S = sets()
    ct, sugar, bitter, fox = S["ct"], S["sugar"], S["bitter"], S["fox"]
    comp_idx, type_idx, fda = S["comp"], S["type"], S["fda"]

    c0 = sys.argv[1:] == ["--base", "c0"]
    if c0:
        from rate import suite
        sim = suite.build(chunks=[])
    else:
        sim = RateSim(1.0)
    cb0233 = np.flatnonzero(ct == "CB0233")
    sub = sim.W[fox][:, cb0233]
    g0 = {"fox_n": 2, "fox_to_cb0233_syn": float(abs(sub).sum()), "edges": int(sub.nnz)}
    print("G0: Fox -> CB0233 %.0f synapses over %d edges (Christie Data S2A: 217, floor 0)"
          % (g0["fox_to_cb0233_syn"], g0["edges"]), flush=True)
    assert g0["edges"] > 0, "G0: Fox -> CB0233 absent"
    print("sets: sugar %d bitter %d fox %d; R %d neurons"
          % (len(sugar), len(bitter), len(fox), len(comp_idx["R"])), flush=True)

    rho = sim.spectral_bound()
    wA, bA, rmaxA = lif_constants()
    assert abs(wA - 6.880548e-3) < 1e-9 and abs(bA + 35.0282) < 1e-4, (wA, bA)
    arms = {"A": (wA, bA, rmaxA), "B": (0.5 / rho, 0.0, float("inf"))}
    prereg = "PREREGISTER_rate_chunk1_control.md @ 194be67"
    if c0:
        arms = {"C0": (sim.w_scale, float(sim.bias[0]), sim.r_max)}
        prereg += " + PREREGISTER_rate_chunk1_control_c0.md @ 2c0b86d"

    out = {"prereg": prereg, "G0": g0, "rho": rho, "hz": HZ, "t_run": T_RUN, "arms": {}}
    if c0:
        Wabs = abs(sim.W.tocsc()[:, fox])
        into_fox = float(Wabs.sum())
        out["anatomy"] = {"sugar_to_fox_syn": float(Wabs[sugar].sum()), "fox_input_syn": into_fox,
                          "sugar_share_of_fox_input": float(Wabs[sugar].sum()) / into_fox,
                          "ref_fdaI_share_of_pam_input": 0.0081}
        print("anatomy: sugar GRN -> Fox %.0f syn = %.2f%% of Fox input (FDA-I -> PAM: 0.81%%)"
              % (out["anatomy"]["sugar_to_fox_syn"], 100 * out["anatomy"]["sugar_share_of_fox_input"]),
              flush=True)
    for arm, (w, b, rmax) in arms.items():
        sim.w_scale, sim.r_max = w, rmax
        sim.bias.fill_(b)
        rates, D = measure(sim, S)
        finite = all(np.isfinite(v).all() for v in rates.values())
        mx = max(float(v.max()) for v in rates.values())
        at_max = {k: float((v >= 0.99 * rmax).mean()) for k, v in rates.items()}
        stable = finite and (max(at_max.values()) < 0.01 if arm != "B" else mx < 1e4)
        rows = battery(D)
        label = ("UNSTABLE" if not stable else
                 "PASSES-UNFITTED" if all(p for p, _ in rows.values()) else "FAILS-UNFITTED")
        out["arms"][arm] = {
            "w_scale": w, "bias": b, "r_max": rmax, "stable": stable, "finite": finite,
            "max_rate": mx, "frac_at_rmax": at_max, "label": label,
            "rows": {k: {"pass": p, "detail": d} for k, (p, d) in rows.items()},
            "D_comp": D,
            "D_type": {k: {t: float((v[i] - rates["BASE"][i]).mean()) for t, i in type_idx.items()}
                       for k, v in rates.items() if k != "BASE"},
            "fox_fda_rate": {k: {s: float(v[i].mean()) for s, i in fda.items()}
                             for k, v in rates.items()},
            "active_frac": {k: float((v > 1.0).mean()) for k, v in rates.items()},
        }
        print("\narm %s  w_scale %.4e bias %.4f r_max %s  -> %s  (max %.1f Hz, at r_max %s)"
              % (arm, w, b, rmax, label, mx, {k: round(x, 4) for k, x in at_max.items()}),
              flush=True)
        for k, (p, d) in rows.items():
            print("  %-3s %s  %s" % (k, "PASS" if p else "fail", d), flush=True)

    out_dir = OUT + "_c0" if c0 else OUT
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "result.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("\nwrote %s" % os.path.join(out_dir, "result.json"))
    if c0:
        rec = {"source": "results/rate_chunk1_control_c0/result.json (unfitted, chunk-0 base)",
               "rows": {k: v["pass"] for k, v in out["arms"]["C0"]["rows"].items()}}
        with open(os.path.join(_HERE, "rate", "regress", "c1_record.json"), "w") as f:
            json.dump(rec, f, indent=1)
        print("re-recorded rate/regress/c1_record.json on the chunk-0 base")


if __name__ == "__main__":
    main()
