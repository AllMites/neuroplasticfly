"""ELN_NEGATE separates the PN code but starves the mushroom body. Raise drive.

apl_jaccard.py: under ELN_NEGATE the PN layer is clean (Jaccard 0.057 vs 0.961)
but only 0.7% of KCs fire and half the channels reach none. APL gain cannot fix
the merge because the merge is upstream of the MB; the open question is whether
the separated code simply needs more drive to cross the KC threshold.

Sweeps ORN rate under ELN_NEGATE and asks for the gate in one place: KC active
fraction 0.05-0.10, all channels live, KC Jaccard < 0.3.

Run: .venv/Scripts/python.exe eln_drive_sweep.py
"""
import itertools
import json
import os

import numpy as np

import gpu_sim as G

OUT = os.path.join(G._HERE, "results", "eln_drive_sweep.json")
T_RUN = 300.0
RATES = [80, 120, 200, 300, 500, 800]


def main():
    G.ELN_NEGATE = True
    meta = np.load(os.path.join(G._HERE, "data", "neuron_meta.npz"), allow_pickle=False)
    ct = meta["cell_type"].astype(str)
    kc = np.flatnonzero(meta["cell_class"].astype(str) == "Kenyon_Cell")
    pn = np.flatnonzero(np.char.find(ct, "PN") >= 0)
    mbon = meta["cell_class"].astype(str) == "MBON"
    central = meta["super_class"].astype(str) == "central"
    chans = sorted({t for t in ct if t.startswith("ORN_")})[:8]

    sim = G.GpuSim(compile_=False)
    drives, labels = [], []
    for r in RATES:
        for c in chans:
            idx = np.flatnonzero(ct == c).astype(np.int64)
            drives.append((idx, np.full(len(idx), r * G.DT / 1000.0))); labels.append(r)
    counts = sim.run_batch(drives, seeds=list(range(len(drives))), t_run=T_RUN)
    counts = np.asarray(counts.cpu() if hasattr(counts, "cpu") else counts)
    hz = counts / (T_RUN / 1000.0)

    rows = []
    for r in RATES:
        h = hz[[i for i, x in enumerate(labels) if x == r]]
        act = [set(np.flatnonzero(x[kc] > 1).tolist()) for x in h]
        live = [a for a in act if a]
        js = [len(a & b) / len(a | b) for a, b in itertools.combinations(live, 2)]
        pj = [set(np.flatnonzero(x[pn] > 1).tolist()) for x in h]
        pjl = [a for a in pj if a]
        pjs = [len(a & b) / len(a | b) for a, b in itertools.combinations(pjl, 2)]
        row = {"orn_hz": r, "n_silent": len(act) - len(live),
               "kc_frac_mean": float(np.mean([len(a) / len(kc) for a in live])) if live else 0.0,
               "kc_jaccard_mean": float(np.mean(js)) if js else None,
               "kc_jaccard_max": float(np.max(js)) if js else None,
               "pn_frac_mean": float(np.mean([len(a) / len(pn) for a in pjl])) if pjl else 0.0,
               "pn_jaccard_mean": float(np.mean(pjs)) if pjs else None,
               "mbon_active_mean": float(np.mean([(x[mbon] > 1).sum() for x in h])),
               "central_active_frac": float(np.mean([(x[central] > 1).mean() for x in h]))}
        rows.append(row)
        print("%4d Hz  %d silent  KC frac %.4f  KC Jacc %s (max %s)  PN frac %.4f  PN Jacc %s  MBON %4.1f  central %.4f"
              % (r, row["n_silent"], row["kc_frac_mean"],
                 "%.3f" % row["kc_jaccard_mean"] if js else " n/a",
                 "%.3f" % row["kc_jaccard_max"] if js else "n/a",
                 row["pn_frac_mean"],
                 "%.3f" % row["pn_jaccard_mean"] if pjs else "n/a",
                 row["mbon_active_mean"], row["central_active_frac"]), flush=True)
    json.dump({"t_run": T_RUN, "eln_negate": True, "channels": chans, "rows": rows},
              open(OUT, "w"), indent=1)
    print("wrote", OUT)
    ok = [r for r in rows if r["n_silent"] == 0 and 0.05 <= r["kc_frac_mean"] <= 0.10
          and r["kc_jaccard_mean"] is not None and r["kc_jaccard_mean"] < 0.3]
    print("GATE PASSED:", [r["orn_hz"] for r in ok] or "none")


if __name__ == "__main__":
    main()
