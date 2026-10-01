"""Chunk-0 LIF reference, exactly as PREREGISTER_rate_chunk0_base.md (1a29779) + amendment 1 (b554245).

--regime eln8 (amendment 1; default) or stock (the original run, kept for the record). LIF, 53 single-glomerulus ORN drives at 40 Hz, 300 ms on from rest then 200 ms off (10 x 20 ms
carried bins), seeds 0-4. Writes results/rate_chunk0/lif_ref.npz + noise ceiling C (held-out only)
and, for eln8, the LIF latch reference (suite latch-check protocol) -> rate/regress/latch_ref.json.

Run: .venv/Scripts/python.exe tools/gpu_slot.py -- .venv/Scripts/python.exe rate/lif_ref.py
"""
import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _HERE)
import gpu_sim as G  # noqa: E402
from learn import condition as C  # noqa: E402

OUT = os.path.join(_HERE, "results", "rate_chunk0")
HZ, T_ON, T_OFF, N_BINS = 40.0, 300.0, 200.0, 10
SEEDS = [0, 1, 2, 3, 4]


def conditions(ct):
    """Prereg split: all ORN_* types sorted; even positions calibration, odd held-out; DM4 forced to cal."""
    names = sorted({t for t in ct if t.startswith("ORN_")})
    assert all((ct == t).sum() >= 10 for t in names)
    split = np.array(["cal" if i % 2 == 0 or t == "ORN_DM4" else "held" for i, t in enumerate(names)])
    return names, split


def ceiling(on, nz_any=None):
    """on [5, N] per-seed rates -> Pearson r(log1p mean(s0-1), log1p mean(s2-4)) over neurons nonzero in either."""
    a, b = on[:2].mean(0), on[2:].mean(0)
    m = (a > 0) | (b > 0)
    return float(np.corrcoef(np.log1p(a[m]), np.log1p(b[m]))[0, 1])


def latch_reference(sim):
    """Amendment 1 change 3: the eln8 LIF under the suite latch-check protocol, per seed."""
    from rate import suite as S
    off = (np.zeros(0, np.int64), np.zeros(0, np.float64))
    ref = {}
    for k, (idx, hz) in S.latch_conditions().items():
        drv = C.to_prob(np.asarray(idx, np.int64), np.full(len(idx), hz))
        _, st = sim.run_batch([drv] * len(SEEDS), SEEDS, t_run=S.LATCH_ON, return_state=True)
        _, st = sim.run_batch([off] * len(SEEDS), SEEDS, t_run=S.LATCH_OFF - S.LATCH_WIN, state=st,
                              return_state=True)
        c = sim.run_batch([off] * len(SEEDS), SEEDS, t_run=S.LATCH_WIN, state=st).cpu().numpy()
        n = [int(x) for x in ((c / (S.LATCH_WIN / 1000.0)) > S.LATCH_HZ).sum(1)]
        ref[k] = {"n_per_seed": n, "bound": max(n) + 139}
        print("latch ref %-8s n_LIF per seed %s -> bound %d" % (k, n, ref[k]["bound"]), flush=True)
    return ref


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", default="eln8", choices=["eln8", "stock"])
    a = ap.parse_args()
    G.ELN_NEGATE, G.PN_KC_GAIN = C.REGIMES[a.regime]
    assert G.W_SYN == 0.275 and (G.ELN_NEGATE, G.PN_KC_GAIN) == C.REGIMES[a.regime], "wrong LIF regime"
    out = OUT if a.regime == "eln8" else os.path.join(OUT, "stock")
    ct = np.load(os.path.join(_HERE, "data", "neuron_meta.npz"), allow_pickle=False)["cell_type"].astype(str)
    names, split = conditions(ct)
    assert (split == "cal").sum() == 28 and (split == "held").sum() == 25, split
    sim = G.GpuSim(compile_=False)
    N = sim.net.n
    assert N == len(ct), (N, len(ct))
    off = (np.zeros(0, np.int64), np.zeros(0, np.float64))

    # selftest: rest is silent; counter RNG reruns are identical
    c0 = sim.run_batch([off], [0], t_run=50.0).cpu().numpy()
    assert c0.sum() == 0, "LIF not silent at rest"
    dm4 = C.to_prob(np.flatnonzero(ct == "ORN_DM4"), np.full((ct == "ORN_DM4").sum(), HZ))
    x1 = sim.run_batch([dm4], [0], t_run=50.0).cpu().numpy()
    x2 = sim.run_batch([dm4], [0], t_run=50.0).cpu().numpy()
    assert (x1 == x2).all() and x1.sum() > 0, "LIF rerun not deterministic / DM4 silent"

    on_all = np.zeros((len(names), len(SEEDS), N), np.float32)
    off_all = np.zeros((len(names), N_BINS, N), np.float32)
    for k, t in enumerate(names):
        idx = np.flatnonzero(ct == t)
        drv = C.to_prob(idx, np.full(len(idx), HZ))
        cnt, st = sim.run_batch([drv] * len(SEEDS), SEEDS, t_run=T_ON, return_state=True)
        on_all[k] = cnt.cpu().numpy() / (T_ON / 1000.0)
        bw = T_OFF / N_BINS
        for j in range(N_BINS):
            c, st = sim.run_batch([off] * len(SEEDS), SEEDS, t_run=bw, state=st, return_state=True)
            off_all[k, j] = c.cpu().numpy().mean(0) / (bw / 1000.0)
        print("%-10s %-4s n=%3d  active %5d  last-100ms active %d"
              % (t, split[k], len(idx), int((on_all[k].mean(0) > 1).sum()),
                 int((off_all[k, 5:].mean(0) > 1).sum())), flush=True)

    r_held = [ceiling(on_all[k]) for k in np.flatnonzero(split == "held")]
    Cc = float(np.median(r_held))
    kc = np.load(os.path.join(_HERE, "data", "neuron_meta.npz"), allow_pickle=False)["cell_class"].astype(str) == "Kenyon_Cell"
    k_dm4 = names.index("ORN_DM4")
    print("noise ceiling C = %.4f (held-out split-half r: min %.4f max %.4f)" % (Cc, min(r_held), max(r_held)))
    print("DM4 KC active frac %.4f (stock 09-20 probe 0.328; eln8 probe 0.100)" % float((on_all[k_dm4].mean(0)[kc] > 1).mean()))
    latch = latch_reference(sim) if a.regime == "eln8" else None
    os.makedirs(out, exist_ok=True)
    np.savez_compressed(os.path.join(out, "lif_ref.npz"), rate_on=on_all, rate_off=off_all,
                        names=np.array(names), split=split, seeds=np.array(SEEDS))
    if latch is not None:
        with open(os.path.join(_HERE, "rate", "regress", "latch_ref.json"), "w") as f:
            json.dump({"prereg": "PREREGISTER_rate_chunk0_base.md amendment 1 @ b554245", "regime": a.regime,
                       "slack": 139, "conditions": latch}, f, indent=1)
    with open(os.path.join(out, "lif_ref.json"), "w") as f:
        json.dump({"prereg": "PREREGISTER_rate_chunk0_base.md @ 1a29779 + am.1 b554245", "regime": a.regime, "C": Cc, "latch": latch,
                   "r_held": dict(zip([names[k] for k in np.flatnonzero(split == "held")], r_held)),
                   "hz": HZ, "t_on": T_ON, "t_off": T_OFF, "n_bins": N_BINS}, f, indent=1)
    print("wrote %s" % out)


if __name__ == "__main__":
    main()
