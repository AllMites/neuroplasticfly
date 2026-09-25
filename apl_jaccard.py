"""Second half of the option-A gate: does a sparse MB separate odours?

apl_sparsity_sweep.py found that divisive APL gain moves KC active fraction
smoothly to 5-10% where drive strength could not. The gate's other half is
odour identity: on the corrected matrix the KC code for different ORN channels
overlapped at Jaccard 0.94, i.e. every odour lit the same cells. Target < 0.3.

Drives eight single ORN channels at 40 Hz under a sweep of APL_DIV_SCALE and
reports the mean pairwise Jaccard of the active-KC sets, alongside the active
fraction so a "sparse because nothing fires" result is visible.

Run: .venv/Scripts/python.exe apl_jaccard.py
"""
import itertools
import json
import os

import numpy as np

import gpu_sim as G

OUT = os.path.join(G._HERE, "results", "apl_jaccard.json")
T_RUN = 300.0
ORN_HZ = 80  # 40 Hz leaves half the channels below their own ignition threshold
# ELN_NEGATE is the one regime known to un-merge the antennal lobe
# (docs/superpowers, regime plan, closed 2026-09-20). If the KC codes stay
# merged under it too, the overlap is not the AL broadcast.
CONFIGS = [("baseline spiking", False, False, None),
           ("ELN_NEGATE", False, False, None, True),
           ("ELN_NEGATE + div 6.0", True, True, 6.0, True),
           ("ELN_NEGATE + div 5.0", True, True, 5.0, True),
           ("divisive (8.0)", True, True, 8.0),
           ("divisive (6.0)", True, True, 6.0),
           ("divisive (5.0)", True, True, 5.0),
           ("divisive (4.5)", True, True, 4.5)]


def main():
    meta = np.load(os.path.join(G._HERE, "data", "neuron_meta.npz"), allow_pickle=False)
    ct = meta["cell_type"].astype(str)
    kc = np.flatnonzero(meta["cell_class"].astype(str) == "Kenyon_Cell")
    # The PN layer feeds the KCs. If PN codes already overlap, the merge is
    # upstream of the mushroom body and no APL setting can undo it.
    pn = np.flatnonzero(np.char.find(ct, "PN") >= 0)
    mbon = meta["cell_class"].astype(str) == "MBON"
    chans = sorted({t for t in ct if t.startswith("ORN_")})[:8]
    print("channels:", ", ".join(c.replace("ORN_", "") for c in chans))

    rows = []
    for cfg in CONFIGS:
        label, graded, divisive, scale = cfg[:4]
        G.APL_GRADED, G.APL_DIVISIVE = graded, divisive
        G.ELN_NEGATE = len(cfg) > 4 and cfg[4]
        if scale is not None:
            G.APL_DIV_SCALE = scale
        sim = G.GpuSim(compile_=False)
        drives = [(np.flatnonzero(ct == c).astype(np.int64), None) for c in chans]
        drives = [(i, np.full(len(i), ORN_HZ * G.DT / 1000.0)) for i, _ in drives]
        c = sim.run_batch(drives, seeds=list(range(len(drives))), t_run=T_RUN)
        c = np.asarray(c.cpu() if hasattr(c, "cpu") else c)
        hz = c / (T_RUN / 1000.0)
        active = [set(np.flatnonzero(h[kc] > 1).tolist()) for h in hz]
        pn_act = [set(np.flatnonzero(h[pn] > 1).tolist()) for h in hz]
        pn_live = [a for a in pn_act if a]
        pn_js = [len(a & b) / len(a | b) for a, b in itertools.combinations(pn_live, 2)]
        # Silent channels score Jaccard 0 against everything and would drag the
        # mean below the gate for the wrong reason. Only ignited pairs count.
        live = [a for a in active if a]
        js = [len(a & b) / len(a | b) for a, b in itertools.combinations(live, 2)]
        fr = [len(a) / len(kc) for a in live] or [0.0]
        r = {"config": label, "div_scale": scale,
             "jaccard_mean": float(np.mean(js)) if js else None,
             "jaccard_min": float(np.min(js)) if js else None,
             "jaccard_max": float(np.max(js)) if js else None,
             "kc_frac_mean": float(np.mean(fr)), "kc_frac_min": float(np.min(fr)),
             "n_empty": int(sum(1 for a in active if not a)), "n_live": len(live),
             "kc_frac_per_channel": [round(len(a) / len(kc), 4) for a in active],
             "mbon_active_mean": float(np.mean([(h[mbon] > 1).sum() for h in hz])),
             "pn_jaccard_mean": float(np.mean(pn_js)) if pn_js else None,
             "pn_frac_mean": float(np.mean([len(a) / len(pn) for a in pn_live])) if pn_live else 0.0}
        rows.append(r)
        print("%-18s KC frac %.4f (min %.4f, %d silent)  Jaccard mean %s  min %s  max %s  MBON %.1f"
              % (label, r["kc_frac_mean"], r["kc_frac_min"], r["n_empty"],
                 "%.3f" % r["jaccard_mean"] if js else "n/a",
                 "%.3f" % r["jaccard_min"] if js else "n/a",
                 "%.3f" % r["jaccard_max"] if js else "n/a",
                 r["mbon_active_mean"]), flush=True)
        print("%-18s   PN frac %.4f  PN Jaccard %s" % (
            "", r["pn_frac_mean"],
            "%.3f" % r["pn_jaccard_mean"] if pn_js else "n/a"), flush=True)
        del sim
    json.dump({"t_run": T_RUN, "orn_hz": ORN_HZ, "channels": chans, "rows": rows},
              open(OUT, "w"), indent=1)
    print("wrote", OUT)
    ok = [r for r in rows if r["jaccard_mean"] is not None and r["jaccard_mean"] < 0.3
          and 0.05 <= r["kc_frac_mean"] <= 0.10 and r["n_empty"] == 0]
    print("GATE PASSED:", [r["config"] for r in ok] or "none")


if __name__ == "__main__":
    main()
