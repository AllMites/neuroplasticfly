"""Does APL gain control buy the mushroom body a sparse regime?

The gate (docs/superpowers/option-a/kc_sparsity_gate_2026-09-20.md) fails on
drive strength alone: KC active fraction is bistable, 0% or ~33%. gpu_sim
already carries the lever - a graded APL (APL_GRADED) in a subtractive or a
divisive form (APL_DIVISIVE), both off by default. This runs the same probe
under each, and sweeps APL_DIV_SCALE, where SMALLER is STRONGER.

Target: KC active fraction 0.05-0.10 with the brain still responsive
(MBON/DAN nonzero, central active fraction not collapsed to 0).

Run: .venv/Scripts/python.exe apl_sparsity_sweep.py
"""
import json
import os

import numpy as np

import gpu_sim as G

OUT = os.path.join(G._HERE, "results", "apl_sparsity_sweep.json")
T_RUN = 300.0
# One weak (single channel) and one strong (the chess set) drive. Both are
# above the ignition cliff on the default config, so any sparsening is APL's.
DRIVES = [("1ch ORN_DA1", ["ORN_DA1"], 40), ("40ch chess", None, 40)]
# (label, APL_GRADED, APL_DIVISIVE, APL_DIV_SCALE). 11.75 is the authored
# "one unit of divisive gain"; smaller multiplies the gain.
CONFIGS = [
    ("baseline spiking", False, False, None),
    ("graded subtractive", True, False, None),
    ("divisive 1x  (11.75)", True, True, 11.75),
    ("divisive (8.0)", True, True, 8.0),
    ("divisive (6.0)", True, True, 6.0),
    ("divisive (5.0)", True, True, 5.0),
    ("divisive (4.5)", True, True, 4.5),
    ("divisive (4.0)", True, True, 4.0),
    ("divisive (3.5)", True, True, 3.5),
    ("divisive (3.0)", True, True, 3.0),
    ("divisive 5x  (2.35)", True, True, 2.35),
    ("divisive 10x (1.175)", True, True, 1.175),
    ("divisive 50x (0.235)", True, True, 0.235),
]


def main():
    meta = np.load(os.path.join(G._HERE, "data", "neuron_meta.npz"), allow_pickle=False)
    ct = meta["cell_type"].astype(str)
    cc = meta["cell_class"].astype(str)
    kc, mbon, dan = cc == "Kenyon_Cell", cc == "MBON", cc == "DAN"
    apl = np.char.startswith(ct, "APL")
    central = meta["super_class"].astype(str) == "central"
    orn = sorted({t for t in ct if t.startswith("ORN_")})

    rows = []
    for label, graded, divisive, scale in CONFIGS:
        G.APL_GRADED, G.APL_DIVISIVE = graded, divisive
        if scale is not None:
            G.APL_DIV_SCALE = scale
        # compile_=False: the flags above are module constants read inside the
        # step loop, and a compiled graph would bake the previous config in.
        sim = G.GpuSim(compile_=False)
        drives = []
        for _, types, hz in DRIVES:
            sel = orn[:40] if types is None else types
            idx = np.flatnonzero(np.isin(ct, sel)).astype(np.int64)
            drives.append((idx, np.full(len(idx), hz * G.DT / 1000.0)))
        c = sim.run_batch(drives, seeds=list(range(len(drives))), t_run=T_RUN)
        c = np.asarray(c.cpu() if hasattr(c, "cpu") else c)
        hz_all = c / (T_RUN / 1000.0)
        for (dname, _, dhz), h in zip(DRIVES, hz_all):
            r = {"config": label, "graded": graded, "divisive": divisive,
                 "div_scale": scale, "drive": dname, "orn_hz": dhz,
                 "kc_active_frac": float((h[kc] > 1).mean()),
                 "kc_mean_hz": float(h[kc].mean()),
                 "mbon_active": int((h[mbon] > 1).sum()),
                 "dan_active": int((h[dan] > 1).sum()),
                 "apl_hz": float(h[apl].mean()),
                 "central_active_frac": float((h[central] > 1).mean())}
            rows.append(r)
            print("%-21s %-12s KC act %.4f  KC %5.1f Hz  MBON %2d  DAN %2d  APL %6.1f  central %.4f"
                  % (label, dname, r["kc_active_frac"], r["kc_mean_hz"], r["mbon_active"],
                     r["dan_active"], r["apl_hz"], r["central_active_frac"]), flush=True)
        del sim
    json.dump({"t_run": T_RUN, "rows": rows}, open(OUT, "w"), indent=1)
    print("wrote", OUT)
    hits = [r for r in rows if 0.05 <= r["kc_active_frac"] <= 0.10 and r["mbon_active"] > 0]
    print("SPARSE REGIME:", hits if hits else "none found")


if __name__ == "__main__":
    main()
