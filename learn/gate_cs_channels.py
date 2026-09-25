"""Track 2 gate: are the chosen CS channels usable, measured?

Runs BEFORE any conditioning. kc_excitability_2026-09-20.md measured the regime
over eight channels in aggregate and explicitly refused to declare the gate
passed with a reinterpreted criterion. The criterion was narrowed in advance to
"the chosen CS channels are live and in band" - this script is that criterion,
measured on those channels alone.

The chosen channels are **DC2 and D**, with DA1 measured alongside but NOT used
as a CS. project-plan decision 2026-09-20: DA1 is the cVA pheromone glomerulus
(Or67d; Kurtovic, Widmer & Dickson 2007), cVA carries innate valence, so a shift
on DA1 could be innate rather than learned and the shuffle arm would not
separate the two. DA1 is reported here so the never-paired probe channel is
known to be live, and its band/Jaccard results are informational only.

Regime: ELN_NEGATE + PN_KC_GAIN = 8.0, stock V_TH (KC_V_TH_DELTA = 0.0), 80 Hz,
300 ms. Same summarise/jaccard machinery as kc_excitability_sweep.py so the
numbers are directly comparable.

PASS requires all four, evaluated on the CS channels ONLY:
  1. kc_active in [0.05, 0.10] for BOTH CS channels
  2. Jaccard(CS[0], CS[1]) < 0.3 (over ignited pairs; a silent channel scores 0
     against everything and would read as a false pass)
  3. 0 silent channels among the CS pair
  4. kc_frac at zero drive == 0 (no spontaneous firing)

FAIL -> stop. Do not run the conditioning arms, do not tune.

Run: .venv/Scripts/python.exe learn/gate_cs_channels.py
"""
import argparse
import itertools
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "flywatch"))
import gpu_sim as G

OUT_FMT = os.path.join(_HERE, "results", "gate_cs_channels_g%s.json")
T_RUN = 300.0
ORN_HZ = 80
CS = ["ORN_DC2", "ORN_D"]        # the pair that gets paired with the US
INFO = ["ORN_DA1"]                # measured, reported, never used as a CS
BAND = (0.05, 0.10)
JACCARD_MAX = 0.3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gain", type=float, default=8.0, help="PN_KC_GAIN to gate at")
    ap.add_argument("--probe", default="", help="extra comma-separated channels to MEASURE "
                    "only (e.g. ORN_DA3,ORN_DA2). They join INFO: reported, never scored, "
                    "and they cannot move the PASS/FAIL, which is computed on CS alone.")
    a = ap.parse_args()
    info = INFO + [c for c in a.probe.split(",") if c.strip()]
    G.ELN_NEGATE = True
    G.KC_V_TH_DELTA = 0.0
    G.PN_KC_GAIN = a.gain
    print("gating at PN_KC_GAIN = %.1f" % a.gain)

    meta = np.load(os.path.join(_HERE, "data", "neuron_meta.npz"), allow_pickle=False)
    ct = meta["cell_type"].astype(str)
    cc = meta["cell_class"].astype(str)
    kc = np.flatnonzero(cc == "Kenyon_Cell")
    pn = np.flatnonzero(np.char.find(ct, "PN") >= 0)
    mbon = cc == "MBON"
    apl = np.char.startswith(ct, "APL")
    central = meta["super_class"].astype(str) == "central"

    # compile_=False: the regime constants are read inside the step loop and a
    # compiled graph bakes the previous value in.
    sim = G.GpuSim(compile_=False)
    drives = []
    for c in CS + info:
        idx = np.flatnonzero(ct == c).astype(np.int64)
        assert len(idx) > 0, "channel %s not in this matrix" % c
        print("%s: %d ORNs" % (c, len(idx)))
        drives.append((idx, np.full(len(idx), ORN_HZ * G.DT / 1000.0)))
    drives.append((np.zeros(0, np.int64), np.zeros(0, np.float64)))  # zero-drive control

    out = sim.run_batch(drives, seeds=list(range(len(drives))), t_run=T_RUN)
    out = np.asarray(out.cpu() if hasattr(out, "cpu") else out)
    hz = out / (T_RUN / 1000.0)

    kc_sets = [set(np.flatnonzero(h[kc] > 1).tolist()) for h in hz[:-1]]
    pn_sets = [set(np.flatnonzero(h[pn] > 1).tolist()) for h in hz[:-1]]
    # Gate arithmetic is over the CS pair only. DA1 is measured but must not be
    # able to pass or fail the gate it is excluded from.
    cs_i = list(range(len(CS)))
    live = [i for i in cs_i if kc_sets[i]]
    n_silent = len(cs_i) - len(live)
    js = [len(kc_sets[a] & kc_sets[b]) / len(kc_sets[a] | kc_sets[b])
          for a, b in itertools.combinations(live, 2)]
    pn_live = [pn_sets[i] for i in cs_i if pn_sets[i]]
    pn_js = [len(a & b) / len(a | b) for a, b in itertools.combinations(pn_live, 2)]
    quiet = hz[-1]

    per = []
    for i, c in enumerate(CS + info):
        per.append({"channel": c, "role": "cs" if i < len(CS) else "probe-only",
                    "n_orn": int((ct == c).sum()),
                    "kc_frac": len(kc_sets[i]) / len(kc),
                    "kc_n": len(kc_sets[i]),
                    "pn_frac": len(pn_sets[i]) / len(pn),
                    "mbon_active": int((hz[i][mbon] > 1).sum()),
                    "apl_hz": float(hz[i][apl].mean()),
                    "central_active_frac": float((hz[i][central] > 1).mean())})

    res = {"regime": {"ELN_NEGATE": True, "PN_KC_GAIN": a.gain, "KC_V_TH_DELTA": 0.0,
                      "orn_hz": ORN_HZ, "t_run_ms": T_RUN},
           "cs": CS, "probe_only": info,
           "channels": per, "n_silent": n_silent,
           "kc_jaccard": float(js[0]) if js else None,
           "pn_jaccard": float(pn_js[0]) if pn_js else None,
           "kc_frac_no_drive": float((quiet[kc] > 1).mean()),
           "band": list(BAND), "jaccard_max": JACCARD_MAX}

    print("")
    print("| channel | role | ORNs | KC frac | KC n | PN frac | MBON | APL Hz | central |")
    for r in per:
        print("| %-8s | %-10s | %4d | %.4f | %5d | %.4f | %4d | %6.1f | %.4f |"
              % (r["channel"], r["role"], r["n_orn"], r["kc_frac"], r["kc_n"],
                 r["pn_frac"], r["mbon_active"], r["apl_hz"], r["central_active_frac"]))
    print("")
    print("Jaccard(%s, %s): KC %s  PN %s"
          % (CS[0], CS[1],
             "%.4f" % js[0] if js else "n/a (a channel is silent)",
             "%.4f" % pn_js[0] if pn_js else "n/a"))
    # The never-paired probe channel has to be separable from the paired one too,
    # or "DA1 did not move" would be a tautology rather than a specificity check.
    for i, c in enumerate(info, start=len(CS)):
        if kc_sets[i] and kc_sets[0]:
            ov = len(kc_sets[0] & kc_sets[i]) / len(kc_sets[0] | kc_sets[i])
            print("Jaccard(%s, %s): KC %.4f  [informational]" % (CS[0], c, ov))
    print("silent CS channels: %d/%d   KC frac at zero drive: %.4f"
          % (n_silent, len(CS), res["kc_frac_no_drive"]))

    checks = {
        "band": all(BAND[0] <= r["kc_frac"] <= BAND[1]
                    for r in per if r["role"] == "cs"),
        "jaccard": js and js[0] < JACCARD_MAX,
        "no_silent": n_silent == 0,
        "no_spontaneous": res["kc_frac_no_drive"] == 0.0,
    }
    res["checks"] = {k: bool(v) for k, v in checks.items()}
    res["pass"] = all(checks.values())
    print("")
    for k, v in res["checks"].items():
        print("  %-16s %s" % (k, "PASS" if v else "FAIL"))
    print("GATE: %s" % ("PASS - proceed to conditioning"
                        if res["pass"] else "FAIL - stop, do not tune"))

    out = OUT_FMT % ("%g" % a.gain)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(res, open(out, "w", encoding="utf-8"), indent=1)
    print("wrote %s" % out)
    return 0 if res["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
