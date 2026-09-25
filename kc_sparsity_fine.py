"""Fine sweep of the 20-40 Hz cliff on one ORN channel, both directions."""
import json, os
import numpy as np
import gpu_sim as G

T_RUN = 300.0
RATES = [20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40]
net = G.brain()
meta = np.load(os.path.join(G._HERE, "data", "neuron_meta.npz"), allow_pickle=False)
ct = meta["cell_type"].astype(str); cc = meta["cell_class"].astype(str)
kc, mbon, dan = cc == "Kenyon_Cell", cc == "MBON", cc == "DAN"
apl = np.char.startswith(ct, "APL")
central = meta["super_class"].astype(str) == "central"
sim = G.GpuSim(net)
drives, labels = [], []
for name in ("ORN_DA1", "ORN_DM4"):
    idx = np.flatnonzero(ct == name).astype(np.int64)
    for r in RATES:
        drives.append((idx, np.full(len(idx), r * G.DT / 1000.0))); labels.append((name, r))
c = sim.run_batch(drives, seeds=list(range(len(drives))), t_run=T_RUN)
c = np.asarray(c.cpu() if hasattr(c, "cpu") else c); hz = c / (T_RUN / 1000.0)
rows = []
for (name, r), h in zip(labels, hz):
    row = {"set": name, "orn_hz": r,
           "kc_active_frac": float((h[kc] > 1).mean()), "kc_mean_hz": float(h[kc].mean()),
           "mbon_active": int((h[mbon] > 1).sum()), "dan_active": int((h[dan] > 1).sum()),
           "apl_hz": float(h[apl].mean()), "central_active_frac": float((h[central] > 1).mean())}
    rows.append(row)
    print("%-9s %3d Hz  KC act %.4f  KC %5.1f Hz  MBON %2d  DAN %2d  APL %6.1f  central %.4f"
          % (name, r, row["kc_active_frac"], row["kc_mean_hz"], row["mbon_active"],
             row["dan_active"], row["apl_hz"], row["central_active_frac"]), flush=True)
json.dump({"t_run": T_RUN, "rows": rows}, open(os.path.join(G._HERE, "results", "kc_sparsity_fine.json"), "w"), indent=1)
