"""KC probe: does the board reach the mushroom body?

Question before any plasticity work: with only the fly's own (non-driven)
mushroom-body neurons as the readout, how much of the board is decodable? If the
Kenyon cells carry less board than the same number of random live neurons, a
plastic KC->MBON layer has nothing to learn from and option C is dead on
arrival. Same decoder, same 5,000 sims and same split as readout_analysis.py so
the numbers join that table.

Also reports KC sparsity in the sim, which biology says should be ~5-10% of KCs
active per stimulus. If the sim's KCs are dense or silent, the MB is not doing
what the MB does.

Host-only, no sim: reads data/rates_dump. ~5 min on the GPU decoder.
Run: uv run kc_probe.py
"""
import json
import os
import time

import numpy as np

import readout_analysis as RA

OUT = os.path.join(RA.REPO, "results", "kc_probe.json")


def main():
    t0 = time.time()
    rates, occ = RA.load_rates()
    meta = np.load(RA.META, allow_pickle=False)
    n = rates.shape[1]
    cc = meta["cell_class"].astype(str)
    sc = meta["super_class"].astype(str)
    driven = RA.driven_mask(meta, n)
    live = np.load(os.path.join(RA.REPO, "data", "live_mask.npz"))["sd"] > 0
    print("rates %s, driven %d, live %d, live&~driven %d"
          % (rates.shape, driven.sum(), live.sum(), (live & ~driven).sum()))

    pops = {"KC": cc == "Kenyon_Cell", "MBON": cc == "MBON", "DAN": cc == "DAN"}
    pops["MB_all"] = pops["KC"] | pops["MBON"] | pops["DAN"]
    pops["DN"] = sc == "descending"
    stats = {}
    for name, m in pops.items():
        m = m & ~driven
        r = rates[:, m].astype(np.float32)
        active = (r > 1.0).mean(axis=1)            # fraction of pop above 1 Hz, per position
        stats[name] = {"n": int(m.sum()), "n_live": int((m & live).sum()),
                       "mean_hz": round(float(r.mean()), 3),
                       "frac_active_per_pos": round(float(active.mean()), 4),
                       "frac_ever_active": round(float((r > 1.0).any(axis=0).mean()), 4)}
        print("  %-7s n=%-5d live=%-5d mean %.2f Hz  active/pos %.3f  ever-active %.3f"
              % (name, stats[name]["n"], stats[name]["n_live"], stats[name]["mean_hz"],
                 stats[name]["frac_active_per_pos"], stats[name]["frac_ever_active"]))

    bench = RA.Bench(occ, len(rates))
    rng = np.random.default_rng(0)
    pool = np.flatnonzero(live & ~driven)
    arms = [
        ("KC (non-driven)", pops["KC"] & ~driven),
        ("MBON", pops["MBON"] & ~driven),
        ("DAN", pops["DAN"] & ~driven),
        ("KC+MBON+DAN", pops["MB_all"] & ~driven),
        ("DN (descending)", pops["DN"] & ~driven),
    ]
    for tag, m in arms:
        bench.run(tag, rates[:, m])
    # Controls: same width as the KC readout, random live non-driven neurons.
    k_live = int((pops["KC"] & live & ~driven).sum())
    for seed in (0, 1):
        idx = np.sort(np.random.default_rng(seed).choice(pool, size=k_live, replace=False))
        bench.run("random live non-driven, width=KC-live (seed %d)" % seed, rates[:, idx])
    # Reference: everything the fly computes, no driven neurons.
    bench.run("all live non-driven (reference)", rates[:, pool])

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump({"pop_stats": stats, "rows": bench.rows,
               "minutes": round((time.time() - t0) / 60, 1)}, open(OUT, "w"), indent=1)
    print("wrote", OUT, "(%.1f min)" % ((time.time() - t0) / 60))


if __name__ == "__main__":
    main()
