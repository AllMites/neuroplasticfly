"""Paper-0 reference table for paper-1 chunk 2 (Task 1).

Recomputes every paper-0 number the chunk-2 guard thresholds are relative to, from the committed
jsonl rows under results/, hashes the sources, writes results/rate_chunk2/reference.json.
Shift metric = learn/analyze_seeds.py:40-41 (last probe trial - trial 0, mean over noise replicates
at each trial, which is what condition.curves() stores). SD = ddof 1 across seeds.
Run: .venv/Scripts/python.exe rate/chunk2_reference.py
"""
import hashlib
import json
import os

import numpy as np

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(_HERE, "results")
OUT = os.path.join(R, "rate_chunk2", "reference.json")

RUNG_A = ["condition_o1s%d" % s for s in range(5)]
RUNG3 = ["persist_p1s%d" % s for s in range(5)]
CYCLES = "cycles_2026-09-23_0814.json"
# paper-0 regime per learn/condition.py REGIMES["eln8"] = (ELN_NEGATE True, PN_KC_GAIN 8.0)
REGIME_DETAIL = {"ELN_NEGATE": True, "PN_KC_GAIN": 8.0}


def _sha256(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def rows_of(stem):
    return [json.loads(l) for l in open(os.path.join(R, stem + ".jsonl"))]


def mean_at(rows, arm, song, trial, phases, key="avoid_index"):
    v = [r[key] for r in rows if r["arm"] == arm and r["song"] == song and r["trial"] == trial
         and r["phase"] in phases]
    assert v, (arm, song, trial, phases)
    return float(np.mean(v))


def shift(rows, arm, song, key="avoid_index"):
    t = max(r["trial"] for r in rows if r["arm"] == arm and r["song"] == song and r["phase"] == "test")
    return mean_at(rows, arm, song, t, ("test",), key) - mean_at(rows, arm, song, 0, ("baseline",), key)


def stats(v):
    v = np.array(v, float)
    return {"per_seed": [float(x) for x in v], "mean": float(v.mean()),
            "sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0, "n": int(len(v)),
            "n_same_sign": int(max((v > 0).sum(), (v < 0).sum()))}


def wfrac(rows, arm, phase, trial):
    w = [r for r in rows if r["arm"] == arm and r["phase"] == phase and r["trial"] == trial
         and "w_mean_frac" in r]
    assert w, (arm, phase, trial)
    return float(w[-1]["w_mean_frac"])   # last row of that trial (after all songs)


def main():
    sources = {}
    for s in RUNG_A + RUNG3:
        for ext in (".jsonl", ".json"):
            sources["results/" + s + ext] = _sha256(os.path.join(R, s + ext))
    sources["results/" + CYCLES] = _sha256(os.path.join(R, CYCLES))

    meta = {}
    for s in RUNG_A + RUNG3:
        a = json.load(open(os.path.join(R, s + ".json")))["args"]
        meta[s] = {"regime": a["regime"], "eta": a["eta"], "lam": a["lam"], "train": a["train"],
                   "seed0": a["seed0"]}
        assert a["regime"] == "eln8" and a["eta"] == 5e-6 and a["lam"] == 0.01, (s, a)

    A = [rows_of(s) for s in RUNG_A]
    ra = {}
    for od in ("dc2", "d", "da1"):
        ra["learn_" + od + "_shift"] = stats([shift(r, "learn", od) for r in A])
    ra["baseline"] = {}
    for od in ("dc2", "d"):
        ra["baseline"][od] = {k: stats([mean_at(r, "learn", od, 0, ("baseline",), k) for r in A])
                              for k in ("mbon_approach_hz", "mbon_avoid_hz", "kc_active", "avoid_index")}
    ra["endpoint_learn"] = {od: {k: stats([mean_at(r, "learn", od, 30, ("test",), k) for r in A])
                                 for k in ("mbon_approach_hz", "mbon_avoid_hz")}
                            for od in ("dc2", "d")}
    ra["learn_w_mean_frac_trial30"] = stats([wfrac(r, "learn", "train", 30) for r in A])
    ra["test_trial"] = 30
    # approach-component references (fix round 1): approach drop = approach(trial 0) - approach(trial 30), per seed;
    # normalised = mean drop / mean baseline approach (ratio of means, as ratio_of_means). Avoid component reported only.
    app = lambda r, arm, od, t, ph, k="mbon_approach_hz": mean_at(r, arm, od, t, ph, k)  # noqa: E731
    dc2_drop = [app(r, "learn", "dc2", 0, ("baseline",)) - app(r, "learn", "dc2", 30, ("test",)) for r in A]
    nv_drop = [app(r, "reversed", "d", 0, ("baseline",)) - app(r, "reversed", "d", 30, ("test",)) for r in A]
    nv_base = [app(r, "reversed", "d", 0, ("baseline",)) for r in A]
    ra["learn_dc2_approach_drop"] = stats(dc2_drop)
    ra["learn_dc2_approach_drop_norm"] = float(np.mean(dc2_drop) / ra["baseline"]["dc2"]["mbon_approach_hz"]["mean"])
    ra["naive_d_approach_drop"] = stats(nv_drop)
    ra["naive_d_approach_drop_norm"] = float(np.mean(nv_drop) / np.mean(nv_base))
    ra["learn_dc2_avoid_component_shift"] = stats([app(r, "learn", "dc2", 30, ("test",), "mbon_avoid_hz")
                                                    - app(r, "learn", "dc2", 0, ("baseline",), "mbon_avoid_hz") for r in A])

    P = [rows_of(s) for s in RUNG3]
    r3 = {}
    phaseA = [mean_at(r, "interfere", "dc2", 30, ("trainA_test",)) - mean_at(r, "interfere", "dc2", 0, ("baseline",)) for r in P]
    dC = [mean_at(r, "interfere", "d", 90, ("trainC_test",)) - mean_at(r, "interfere", "d", 60, ("relax_test",)) for r in P]
    naive = [shift(r, "reversed", "d") for r in A]
    r3["phaseA_dc2_shift"] = stats(phaseA)
    r3["trained_brain_d_shift"] = stats(dC)
    r3["naive_d_shift"] = stats(naive)
    r3["ratio_of_means"] = float(np.mean(dC) / np.mean(naive))
    dC_app = [mean_at(r, "interfere", "d", 60, ("relax_test",), "mbon_approach_hz")
              - mean_at(r, "interfere", "d", 90, ("trainC_test",), "mbon_approach_hz") for r in P]
    r3["trained_brain_d_approach_drop"] = stats(dC_app)
    r3["approach_ratio_of_means"] = float(np.mean(dC_app) / np.mean(nv_drop))
    r3["end_training_approach_hz"] = {
        od: stats([mean_at(r, "interfere", od, 90, ("trainC_test",), "mbon_approach_hz") for r in P])
        for od in ("dc2", "d")}
    r3["baseline_approach_hz"] = {
        od: stats([mean_at(r, "interfere", od, 0, ("baseline",), "mbon_approach_hz") for r in P])
        for od in ("dc2", "d")}
    r3["w_mean_frac"] = {name: stats([wfrac(r, "interfere", ph, trial) for r in P])
                         for ph, name, trial in (("trainA", "end_trainA", 30), ("relax", "end_relax", 60),
                                                 ("trainC", "end_trainC", 90))}

    cy = json.load(open(os.path.join(R, CYCLES)))
    bidir = {"label": cy["outcome"], "file": "results/" + CYCLES, "prereg": cy["prereg"],
             "readout": "walking speed, closed loop (DN_HZ_REF readout)",
             "note": "paper 0 has no MB-level reference for bidir; the MB-level criterion is new in paper 1"}

    ref = {"rung_a": ra, "rung3": r3, "bidir": bidir, "sources": sources,
           "source_meta": meta, "regime": "eln8", "regime_detail": REGIME_DETAIL,
           "pn_kc_gain": 8.0, "floor": 5,
           "floor_note": "min_syn=5 is not recorded in these result files (they pre-date the w_syn/min_syn fields); "
                         "taken from paper0-build PROVENANCE (floor 5 canonical)",
           "naive_d_definition": "rung-A reversed arm (condition_o1s0..4.jsonl): D paired with punish on a naive brain; "
                                 "shift = avoid_index(test t30) - avoid_index(baseline t0) on song d",
           "shift_definition": "mean avoid_index over noise replicates at last test trial minus trial 0; SD ddof 1 over seeds"}

    chk = [("rungA DC2 shift", ra["learn_dc2_shift"]["mean"], 11.05, 0.01),
           ("rungA DC2 SD", ra["learn_dc2_shift"]["sd"], 0.195, 0.01),
           ("rungA D shift", ra["learn_d_shift"]["mean"], 1.426, 0.01),
           ("rungA DA1 shift", ra["learn_da1_shift"]["mean"], 0.556, 0.01),
           ("rung3 phaseA DC2", r3["phaseA_dc2_shift"]["mean"], 10.96, 0.01),
           ("rung3 D trained", r3["trained_brain_d_shift"]["mean"], 9.66, 0.01),
           ("rung3 D SD", r3["trained_brain_d_shift"]["sd"], 0.451, 0.01),
           ("rung3 naive D", r3["naive_d_shift"]["mean"], 11.29, 0.01),
           ("rung3 ratio", r3["ratio_of_means"], 0.855, 0.001),
           ("rung3 appr end DC2", r3["end_training_approach_hz"]["dc2"]["mean"], 1.556, 0.01),
           ("rung3 appr end D", r3["end_training_approach_hz"]["d"]["mean"], 1.103, 0.01),
           ("rung3 appr base DC2", r3["baseline_approach_hz"]["dc2"]["mean"], 11.195, 0.01),
           ("rung3 appr base D", r3["baseline_approach_hz"]["d"]["mean"], 12.414, 0.01)]
    for n, v in (("rungA DC2 approach drop", ra["learn_dc2_approach_drop"]["mean"]), ("rungA DC2 approach drop norm", ra["learn_dc2_approach_drop_norm"]),
                 ("rungA DC2 avoid component shift", ra["learn_dc2_avoid_component_shift"]["mean"]),
                 ("naive D approach drop", ra["naive_d_approach_drop"]["mean"]), ("naive D approach drop norm", ra["naive_d_approach_drop_norm"]),
                 ("rung3 trained D approach drop", r3["trained_brain_d_approach_drop"]["mean"]), ("rung3 approach-drop ratio", r3["approach_ratio_of_means"])):
        print("NEW %-32s %.5f" % (n, v))
    bad = 0
    print("%-22s %10s %10s" % ("quantity", "recomputed", "preprint"))
    for n, got, exp, tol in chk:
        ok = abs(got - exp) <= tol
        bad += not ok
        print("%-22s %10.4f %10.4f %s" % (n, got, exp, "OK" if ok else "MISMATCH"))
    print("rungA w_mean_frac t30 learn: mean %.5f per-seed %s" % (
        ra["learn_w_mean_frac_trial30"]["mean"], ra["learn_w_mean_frac_trial30"]["per_seed"]))
    print("rung3 w_mean_frac:", {k: round(v["mean"], 5) for k, v in r3["w_mean_frac"].items()})
    print("DC2 same sign: %d of %d" % (ra["learn_dc2_shift"]["n_same_sign"], ra["learn_dc2_shift"]["n"]))
    assert bad == 0, "STOP: %d numbers do not reproduce" % bad
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(ref, open(OUT, "w"), indent=1)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
