"""The only labeller for PREREGISTER_christie_match.md. Reads results/christie_match/parts/*.jsonl, writes
results/christie_match/analysis.json + analysis.txt. No hand edits.

  python learn/analyze_christie_match.py [--selftest]
"""
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR = os.path.join(HERE, "results", "christie_match")
CW = (0.37, 0.38, 0.382, 0.386, 0.39)
RATES = [float(r) for r in range(10, 201, 10)]


def load(d=DIR):
    rows = {}
    for p in glob.glob(os.path.join(d, "parts", "*.jsonl")):
        for l in open(p):
            if l.strip():
                r = json.loads(l)
                rows[(r["min_syn"], r["w_syn"], r["cond"])] = r
    return rows


def global_frac(row, ref):
    """Fraction of PAM-contributing trials that are GLOBAL (central > 2x the W 0.275 condition's mean)."""
    if ref is None:
        return None
    thr = 2.0 * sum(ref["trial_central_hz"]) / len(ref["trial_central_hz"])
    contrib = [i for i, n in enumerate(row["trial_pam_n"]) if n > 0]
    if not contrib:
        return 0.0
    return sum(row["trial_central_hz"][i] > thr for i in contrib) / len(contrib)


def label_floor(rows, k):
    cells, repro = [], []
    for w in CW:
        for r in RATES:
            it = rows.get((k, w, "gg%d" % r))
            if it is None:
                continue
            sil = rows.get((k, w, "gg%d-fox" % r))
            gf = global_frac(it, rows.get((k, 0.275, "gg%d" % r)))
            c = {"w": w, "r": r, "PAM_resp": it["PAM_resp"], "PAM_resp_5": it["PAM_resp_5"],
                 "fox_sil": None if sil is None else sil["PAM_resp"], "global_frac": gf,
                 "n_trials_with_pam": sum(n > 0 for n in it["trial_pam_n"])}
            cells.append(c)
            if c["PAM_resp"] >= 150 and c["fox_sil"] is not None and c["fox_sil"] <= 15:
                c["kind"] = "selective" if (gf is not None and gf < 0.5) else "runaway"
                repro.append(c)
    if not cells:
        return {"label": "NOT RUN", "cells_run": 0}
    best = max(cells, key=lambda c: c["PAM_resp"])
    if repro:
        top = max(repro, key=lambda c: c["PAM_resp"])
        lab = "REPRODUCED (%s)" % top["kind"]
    elif best["PAM_resp"] <= 15:
        lab = "NOT REPRODUCED"
    elif best["fox_sil"] is None:
        lab = "UNDETERMINED (Fox-silenced NOT RUN at max cell)"
    elif best["fox_sil"] > 15:
        lab = "NOT FOX-DEPENDENT"
    else:
        lab = "PARTIAL" if best["PAM_resp"] <= 149 else "UNDETERMINED"
    # Q-flicker: sign changes of (PAM_resp >= 150) across r at the best W
    seq = [c["PAM_resp"] >= 150 for c in cells if c["w"] == best["w"]]
    return {"label": lab, "best": best, "reproduced_cells": repro, "cells_run": len(cells),
            "q_flicker_flips_at_best_w": sum(a != b for a, b in zip(seq, seq[1:]))}


def analyse(rows):
    out = {"floors": {}}
    for k in (1, 5):
        f = label_floor(rows, k)
        rep = f.get("reproduced_cells", [])
        f["Q_trials"] = {"n_repro_cells_resp5_lt_resp30": sum(c["PAM_resp_5"] < c["PAM_resp"] for c in rep),
                         "n_repro_cells": len(rep)}
        gg = [rows[(k, 0.39, "gg%d" % r)]["PAM_resp"] for r in RATES if (k, 0.39, "gg%d" % r) in rows]
        grn = [rows[(k, 0.39, "grn%d" % r)]["PAM_resp"] for r in RATES if (k, 0.39, "grn%d" % r) in rows]
        f["Q_GINs"] = {"max_grn_gin_w039": max(gg) if gg else None, "max_grn_only_w039": max(grn) if grn else None}
        ctl = [rows[(k, 0.275, "gg%d" % r)]["PAM_resp"] for r in RATES if (k, 0.275, "gg%d" % r) in rows]
        f["Q_control"] = {"max_PAM_resp_w0275": max(ctl) if ctl else None,
                          "fail": bool(ctl) and max(ctl) > 15}
        out["floors"][k] = f
    out["Q_floor"] = {k: out["floors"][k]["label"] for k in (1, 5)}
    g = rows.get((1, 0.39, "grn200"))                    # second golden: main-run ablation seeds 0-4
    out["golden_in_run"] = None if g is None else {"got": g["trial_pam_n"][:5], "pass": g["trial_pam_n"][:5] == [7, 11, 7, 14, 0]}
    return out


def selftest():
    def row(k, w, cond, resp, central=1.0, pam_trials=30):
        return {"min_syn": k, "w_syn": w, "cond": cond, "PAM_resp": resp, "PAM_resp_5": resp // 2,
                "trial_pam_n": [1] * pam_trials + [0] * (30 - pam_trials), "trial_central_hz": [central] * 30}
    rows = {}
    for k in (1, 5):
        for w in (0.275,) + CW:
            for r in RATES:
                rows[(k, w, "gg%d" % r)] = row(k, w, "gg%d" % r, 0)
                rows[(k, w, "gg%d-fox" % r)] = row(k, w, "gg%d-fox" % r, 0)
    rows[(1, 0.37, "gg160")] = row(1, 0.37, "gg160", 191, central=1.5)           # selective
    rows[(5, 0.39, "gg200")] = row(5, 0.39, "gg200", 190, central=5.0)           # runaway
    a = analyse(rows)
    assert a["floors"][1]["label"] == "REPRODUCED (selective)", a["floors"][1]["label"]
    assert a["floors"][5]["label"] == "REPRODUCED (runaway)", a["floors"][5]["label"]
    rows[(5, 0.39, "gg200-fox")] = row(5, 0.39, "gg200-fox", 100)
    assert analyse(rows)["floors"][5]["label"] == "NOT FOX-DEPENDENT"
    rows[(5, 0.39, "gg200")] = row(5, 0.39, "gg200", 40); rows[(5, 0.39, "gg200-fox")] = row(5, 0.39, "gg200-fox", 3)
    assert analyse(rows)["floors"][5]["label"] == "PARTIAL"
    rows[(5, 0.39, "gg200")] = row(5, 0.39, "gg200", 10)
    assert analyse(rows)["floors"][5]["label"] == "NOT REPRODUCED"
    print("selftest PASS")


def main():
    if "--selftest" in sys.argv:
        return selftest()
    a = analyse(load())
    json.dump(a, open(os.path.join(DIR, "analysis.json"), "w"), indent=1)
    lines = ["floor %d: %s | best %s | Q-GINs %s | Q-control %s | Q-trials %s" % (
        k, f["label"], {x: f.get("best", {}).get(x) for x in ("w", "r", "PAM_resp", "fox_sil", "global_frac")},
        f["Q_GINs"], f["Q_control"], f["Q_trials"]) for k, f in a["floors"].items()]
    lines.append("golden in run: %s" % a["golden_in_run"])
    open(os.path.join(DIR, "analysis.txt"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
