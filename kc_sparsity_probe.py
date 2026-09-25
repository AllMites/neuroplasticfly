"""KC sparsity probe: what olfactory drive makes Kenyon cells code sparsely?

Evidence before any plasticity work. Measured 2026-09-19: under audio-only
song drive (gain 3) zero of 5,177 KCs spike; under the chess drive
(40 ORN channels at up to 120 Hz) 86% of KCs are active per position. Biology
says 5-10% of KCs per odor. The dopamine-gated KC->MBON synapse is the only
well-characterised plastic site in this brain, and it can learn nothing from
a silent or a saturated KC layer.

This drives ONE or a FEW ORN channels at a sweep of rates for 300 ms and
reports KC active fraction, KC mean Hz, MBON/DAN activity. Reads brain_gpu.npz
only. ~1 s per sim on the GPU.

Run: .venv/Scripts/python.exe kc_sparsity_probe.py
"""
import json
import os

import numpy as np

import gpu_sim as G

OUT = os.path.join(G._HERE, "results", "kc_sparsity_probe.json")
T_RUN = 300.0
RATES = [10, 20, 40, 60, 80, 120, 150]


def main():
    net = G.brain()
    meta = np.load(os.path.join(G._HERE, "data", "neuron_meta.npz"), allow_pickle=False)
    ct = meta["cell_type"].astype(str)
    cc = meta["cell_class"].astype(str)
    kc, mbon, dan = cc == "Kenyon_Cell", cc == "MBON", cc == "DAN"
    apl = np.char.startswith(ct, "APL")
    orn_types = sorted({t for t in ct if t.startswith("ORN_")})
    sets = {
        "1ch ORN_DA1": ["ORN_DA1"],
        "1ch ORN_DM4": ["ORN_DM4"],
        "3ch DA1+DM4+VA1v": ["ORN_DA1", "ORN_DM4", "ORN_VA1v"],
        "8ch first8": orn_types[:8],
        "40ch all(chess set)": orn_types[:40],
    }
    sim = G.GpuSim(net)
    drives, labels = [], []
    for name, types in sets.items():
        idx = np.flatnonzero(np.isin(ct, types)).astype(np.int64)
        for r in RATES:
            drives.append((idx, np.full(len(idx), r * G.DT / 1000.0)))
            labels.append((name, r, int(len(idx))))
    counts = sim.run_batch(drives, seeds=list(range(len(drives))), t_run=T_RUN)
    counts = np.asarray(counts.cpu() if hasattr(counts, "cpu") else counts)
    hz = counts / (T_RUN / 1000.0)
    rows = []
    for (name, r, n_orn), h in zip(labels, hz):
        rows.append({
            "set": name, "orn_hz": r, "n_orn": n_orn,
            "kc_active_frac": float((h[kc] > 1).mean()),
            "kc_mean_hz": float(h[kc].mean()),
            "mbon_active": int((h[mbon] > 1).sum()),
            "dan_active": int((h[dan] > 1).sum()),
            "apl_hz": float(h[apl].mean()) if apl.any() else None,
            "central_active_frac": float((h[meta["super_class"].astype(str) == "central"] > 1).mean()),
        })
        print("%-22s %4d Hz  KC act %.3f  KC %.1f Hz  MBON %2d  DAN %2d  APL %s  central %.3f"
              % (name, r, rows[-1]["kc_active_frac"], rows[-1]["kc_mean_hz"],
                 rows[-1]["mbon_active"], rows[-1]["dan_active"],
                 "%.1f" % rows[-1]["apl_hz"] if rows[-1]["apl_hz"] is not None else "-",
                 rows[-1]["central_active_frac"]), flush=True)
    json.dump({"t_run": T_RUN, "rows": rows}, open(OUT, "w"), indent=1)
    print("wrote", OUT)
    # ponytail: one check that fails if KC readout is broken
    assert any(0.02 < x["kc_active_frac"] < 0.2 for x in rows) or print("NO SPARSE REGIME FOUND")


if __name__ == "__main__":
    main()
