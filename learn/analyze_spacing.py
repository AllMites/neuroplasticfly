"""Does the rung-3 interference ratio hold at other spacings?

Item 2 of "Do next" in the 2026-09-22 project notes: interference
was measured at exactly ONE interval (30 relax trials) and one training length,
and "whether the 5.7x ratio holds elsewhere is unknown".

No new protocol is needed. learn/persist.py already takes --relax, and it already
records it in the output json, so this is a sweep and an analysis, not a feature.

THE TRAP, and the reason this script computes a residual rather than a retention.
The decay component of any spacing result is PL.LAM read back out: recovery is
`lam * (w0 - w)` applied once per relax trial, so predicted weight retention is
exactly `(1 - lam) ** relax`. Rung 3 measured 0.740 against a predicted 0.7397
with ZERO variance at both gains. Quoting a retention per spacing would be
quoting `0.99 ** relax` in four different fonts.

What is NOT predictable from lam is the INTERFERENCE: how much more DC2 loses
while a second odour is being trained than it loses over an equal stretch of idle
time. Rung 3 measured 5.7x at relax=30. That ratio is what this script reports,
and the falsifier fixed in advance is:

  the interference:idle ratio stays within 2x of 5.7 across spacings
  -> spacing does not matter, the rung-3 number stands as written.
  outside that -> spacing matters, and the rung-3 write-up needs a spacing
  caveat added IN PLACE (stale-doc protocol), not appended somewhere else.

Reads results/persist_<tag>.json + .jsonl for every tag given.

Run: .venv/Scripts/python.exe learn/analyze_spacing.py sp00s0 sp10s0 sp30s0 sp90s0
"""
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(_HERE, "results")


def load(tag):
    meta = json.load(open(os.path.join(RESULTS, "persist_%s.json" % tag)))
    rows = [json.loads(l) for l in open(os.path.join(RESULTS, "persist_%s.jsonl" % tag))]
    return meta, rows


def avoid(rows, arm, stim, phase):
    """Mean avoid_index over the LAST probe block of `phase` for one arm/stimulus."""
    sel = [r for r in rows if r["arm"] == arm and r["song"] == stim
           and r["phase"] == phase]
    if not sel:
        return None
    last = max(r["trial"] for r in sel)
    return float(np.mean([r["avoid_index"] for r in sel if r["trial"] == last]))


def weight_frac(rows, arm, phase):
    sel = [r for r in rows if r["arm"] == arm and r["phase"] == phase and "w_mean_frac" in r]
    if not sel:
        return None
    return float(sel[-1]["w_mean_frac"])


def main(tags):
    print("| relax | seeds | predicted lam retention | measured w retention | DC2 idle loss | "
          "DC2 interference loss | ratio |")
    out = []
    for tag in tags:
        meta, rows = load(tag)
        a = meta["args"]
        relax, lam = a["relax"], a["lam"]
        base = avoid(rows, "retain", meta["cs_a"], "baseline")
        end_a = avoid(rows, "retain", meta["cs_a"], "trainA_test")
        end_b = avoid(rows, "retain", meta["cs_a"], "relax_test")
        end_c = avoid(rows, "interfere", meta["cs_a"], "trainC_test")
        # Interference is measured on the interfere arm, whose phase B is the same
        # length; its own end-of-B value is the reference, not the retain arm's.
        ib = avoid(rows, "interfere", meta["cs_a"], "relax_test")

        predicted = (1.0 - lam) ** relax
        wa = weight_frac(rows, "retain", "trainA")
        wb = weight_frac(rows, "retain", "relax")
        measured = (wb / wa) if (wa and wb) else float("nan")

        idle = (end_b - end_a) if (end_a is not None and end_b is not None) else float("nan")
        interf = (end_c - ib) if (end_c is not None and ib is not None) else float("nan")
        # Per-trial, so spacings of different length are comparable at all.
        n_c = a["train"]
        ratio = ((interf / n_c) / (idle / relax)) if relax and idle else float("nan")
        print("| %5d | %5s | %23.4f | %20.4f | %13.3f | %21.3f | %5.2f |"
              % (relax, tag, predicted, measured, idle, interf, ratio))
        out.append({"tag": tag, "relax": relax, "lam": lam, "predicted_retention": predicted,
                    "measured_weight_retention": measured, "baseline": base,
                    "end_trainA": end_a, "end_relax": end_b, "end_trainC": end_c,
                    "idle_loss": idle, "interference_loss": interf, "ratio": ratio})

    ratios = [o["ratio"] for o in out if np.isfinite(o["ratio"])]
    if ratios:
        lo, hi = min(ratios), max(ratios)
        print("\nratio range %.2f .. %.2f; rung 3 measured 5.7 at relax=30" % (lo, hi))
        ok = all(5.7 / 2 <= r <= 5.7 * 2 for r in ratios)
        print("FALSIFIER: %s" % ("PASS - every spacing within 2x of 5.7, the rung-3 number "
                                 "stands as written" if ok else
                                 "FAIL - spacing matters. Add a spacing caveat IN PLACE to "
                                 "docs/superpowers/option-a/persistence_interference_2026-09-21.md"))
    dst = os.path.join(RESULTS, "spacing_sweep.json")
    json.dump(out, open(dst, "w"), indent=1)
    print("wrote", dst)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    main(sys.argv[1:])
