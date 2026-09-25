"""The option-A gate: can a KC threshold offset buy recruitment without losing the odour code?

ELN_NEGATE separates the odour code (PN Jaccard 0.961 -> 0.057) but starves the
mushroom body: 2.3% of KCs active at 800 Hz ORN, 2 of 8 channels silent. ORN drive
and APL gain are both measured out
(docs/superpowers/option-a/kc_sparsity_gate_2026-09-20.md), which leaves the Kenyon
cell's own threshold. There is room to pay: ELN_NEGATE's Jaccard is 0.016 against a
0.3 target.

Sweeps KC_V_TH_DELTA (negative = more excitable) with ELN_NEGATE on, driving eight
single ORN channels at 80 Hz, and reports KC recruitment against odour separation.

GATE: KC active fraction 0.05-0.10, zero silent channels, KC Jaccard < 0.3.

Two traps this script exists to avoid, both paid for once already:
  - A Jaccard mean over channels where some are SILENT reads far too good, because a
    silent channel scores 0 against everything. Only ignited pairs are scored, and the
    silent count and the max are reported beside the mean.
  - 40 Hz leaves half the channels below their own ignition threshold. 80 Hz does not.

Run: .venv/Scripts/python.exe kc_excitability_sweep.py
"""
import itertools
import json
import os

import numpy as np

import gpu_sim as G

OUT = os.path.join(G._HERE, "results", "kc_excitability_sweep.json")
T_RUN = 300.0
ORN_HZ = 80
# Negative is more excitable. 0.0 is the current ELN_NEGATE baseline and must
# reproduce the 2.3%-at-800-Hz picture at this lower rate.
# Hard bound: V_TH + delta must stay above V_REST (-52). V_TH is -45, so -7.0 puts a
# Kenyon cell's threshold AT rest and it free-runs. -6.5 is the last usable point and
# -7.0/-7.5 are included to document the ceiling, caught by the no-drive control.
DELTAS = [0.0, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -6.5, -7.0, -7.5]
# (KC_V_TH_DELTA, PN_KC_GAIN). The gain arm keeps the threshold at a mild, defensible
# offset and buys recruitment by raising ALPN->KC drive instead.
ARMS = [(d, 1.0) for d in DELTAS] +        [(0.0, g) for g in (1.5, 2.0, 3.0, 4.0, 6.0, 8.0)] +        [(-3.0, g) for g in (1.5, 2.0, 3.0, 4.0)]


def summarise(h, kc, pn, mbon, central):
    """Active-set stats for one drive's [n_neurons] rate vector."""
    return (set(np.flatnonzero(h[kc] > 1).tolist()),
            set(np.flatnonzero(h[pn] > 1).tolist()),
            int((h[mbon] > 1).sum()), float((h[central] > 1).mean()))


def jaccard(sets):
    """Mean/min/max over IGNITED pairs only. Silent sets score 0 against everything."""
    live = [a for a in sets if a]
    js = [len(a & b) / len(a | b) for a, b in itertools.combinations(live, 2)]
    return live, js


def main():
    G.ELN_NEGATE = True
    meta = np.load(os.path.join(G._HERE, "data", "neuron_meta.npz"), allow_pickle=False)
    ct = meta["cell_type"].astype(str)
    cc = meta["cell_class"].astype(str)
    kc = np.flatnonzero(cc == "Kenyon_Cell")
    pn = np.flatnonzero(np.char.find(ct, "PN") >= 0)
    mbon = cc == "MBON"
    apl = np.char.startswith(ct, "APL")
    central = meta["super_class"].astype(str) == "central"
    chans = sorted({t for t in ct if t.startswith("ORN_")})[:8]
    print("ELN_NEGATE on, %d channels at %d Hz:" % (len(chans), ORN_HZ),
          ", ".join(c.replace("ORN_", "") for c in chans), flush=True)

    rows = []
    for delta, gain in ARMS:
        G.KC_V_TH_DELTA = delta
        G.PN_KC_GAIN = gain
        # compile_=False: the constants above are read inside the step loop, and a
        # compiled graph bakes the previous value in.
        sim = G.GpuSim(compile_=False)
        drives = []
        for c in chans:
            idx = np.flatnonzero(ct == c).astype(np.int64)
            drives.append((idx, np.full(len(idx), ORN_HZ * G.DT / 1000.0)))
        # Zero-drive control: a delta large enough to make Kenyon cells fire without
        # any input would otherwise read as a recruitment win.
        drives.append((np.zeros(0, np.int64), np.zeros(0, np.float64)))
        c_out = sim.run_batch(drives, seeds=list(range(len(drives))), t_run=T_RUN)
        c_out = np.asarray(c_out.cpu() if hasattr(c_out, "cpu") else c_out)
        hz = c_out / (T_RUN / 1000.0)

        stats = [summarise(h, kc, pn, mbon, central) for h in hz[:-1]]
        kc_sets = [s[0] for s in stats]
        pn_sets = [s[1] for s in stats]
        live, js = jaccard(kc_sets)
        pn_live, pn_js = jaccard(pn_sets)
        quiet = hz[-1]
        row = {
            "delta": delta,
            "pn_kc_gain": gain,
            "orn_hz": ORN_HZ,
            "n_silent": len(kc_sets) - len(live),
            "kc_frac_mean": float(np.mean([len(a) / len(kc) for a in live])) if live else 0.0,
            "kc_frac_min": float(np.min([len(a) / len(kc) for a in live])) if live else 0.0,
            "kc_jaccard_mean": float(np.mean(js)) if js else None,
            "kc_jaccard_max": float(np.max(js)) if js else None,
            "pn_frac_mean": float(np.mean([len(a) / len(pn) for a in pn_live])) if pn_live else 0.0,
            "pn_jaccard_mean": float(np.mean(pn_js)) if pn_js else None,
            "apl_hz": float(np.mean([h[apl].mean() for h in hz[:-1]])),
            "mbon_active_mean": float(np.mean([s[2] for s in stats])),
            "central_active_frac": float(np.mean([s[3] for s in stats])),
            "kc_frac_no_drive": float((quiet[kc] > 1).mean()),
        }
        rows.append(row)
        print("delta %+.1f gain %.1f  %d silent  KC frac %.4f (min %.4f)  KC Jacc %s (max %s)  "
              "APL %6.1f  MBON %4.1f  central %.4f  KC@nodrive %.4f"
              % (delta, gain, row["n_silent"], row["kc_frac_mean"], row["kc_frac_min"],
                 "%.3f" % row["kc_jaccard_mean"] if js else " n/a",
                 "%.3f" % row["kc_jaccard_max"] if js else "n/a",
                 row["apl_hz"], row["mbon_active_mean"], row["central_active_frac"],
                 row["kc_frac_no_drive"]), flush=True)
        del sim

    json.dump({"t_run": T_RUN, "orn_hz": ORN_HZ, "eln_negate": True,
               "channels": chans, "rows": rows}, open(OUT, "w"), indent=1)
    print("wrote", OUT)
    ok = [r for r in rows
          if r["n_silent"] == 0
          and 0.05 <= r["kc_frac_mean"] <= 0.10
          and r["kc_jaccard_mean"] is not None and r["kc_jaccard_mean"] < 0.3
          and r["kc_frac_no_drive"] < 0.01]
    print("GATE PASSED:", [(r["delta"], r["pn_kc_gain"]) for r in ok] or "none")


if __name__ == "__main__":
    main()
