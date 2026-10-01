"""Chunk 0: calibrate the rate base to the eln8 LIF, exactly as PREREGISTER_rate_chunk0_base.md (1a29779)
+ amendment 1 (b554245). Global params only (ladder step 1); Nelder-Mead from 5 fixed starts per family.

One (family, start) cell per file in results/rate_chunk0/cells/, so a killed run resumes.
Held-out conditions are loaded ONLY by --analyze.

Run:  .venv/Scripts/python.exe tools/gpu_slot.py -- .venv/Scripts/python.exe rate/chunk0.py --family lin
      .venv/Scripts/python.exe tools/gpu_slot.py -- .venv/Scripts/python.exe rate/chunk0.py --family lif
      .venv/Scripts/python.exe tools/gpu_slot.py -- .venv/Scripts/python.exe rate/chunk0.py --analyze
"""
import argparse
import json
import os
import sys

import numpy as np
import scipy.sparse as sp
import torch
from scipy.optimize import minimize

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _HERE)
import gpu_sim as G  # noqa: E402
from rate import suite as S  # noqa: E402
from rate.engine import RateSim  # noqa: E402

OUT = os.path.join(_HERE, "results", "rate_chunk0")
CELLS = os.path.join(OUT, "cells")
HZ, T_ON, N_BINS, BW = 40.0, 300.0, 10, 20.0
W0 = {"lin": 6.880548e-3, "lif": G.W_SYN * G.TAU_SYN / 1000.0}   # prereg: arm A / derived LIF mean-field
STARTS = {  # prereg table: (w factor, bias, tau[, r_max])
    "lin": [(1, -35.0282, 20, 454.5), (0.5, -35.0282, 20, 454.5), (0.25, -35.0282, 20, 454.5),
            (1, -70.0564, 20, 454.5), (0.5, -35.0282, 5, 454.5)],
    "lif": [(1, 0, 20), (0.5, 0, 20), (0.25, 0, 20), (1, -7, 20), (0.5, 0, 5)],
}
NM = dict(maxfev=300, xatol=1e-3, fatol=1e-4)
PENALTY = 10.0


def x0(family, s):
    x = [np.log(W0[family] * s[0]), s[1], np.log(s[2])]
    return np.array(x + ([np.log(s[3])] if family == "lin" else []), dtype=np.float64)


def apply(sim, family, x):
    """Unconstrained x -> sim. tau clamped to [2, 200] ms, r_max to [10, 1000] Hz (prereg bounds)."""
    sim.transfer = family
    sim.w_scale = float(np.exp(x[0]))
    with torch.no_grad():
        sim.bias.fill_(float(x[1]))
        sim.log_tau.fill_(float(np.log(np.clip(np.exp(x[2]), 2.0, 200.0))))
    sim.r_max = float(np.clip(np.exp(x[3]), 10.0, 1000.0)) if family == "lin" else float("inf")


def respond(sim, drives):
    """-> on-window rates [B, N] and off-bin rates [B, N_BINS, N] (state carried), no grad."""
    empty = [(np.zeros(0, np.int64), 0.0)] * len(drives)
    with torch.no_grad():
        on, st = sim.run(drives, T_ON, return_state=True)
        off = []
        for _ in range(N_BINS):
            r, st = sim.run(empty, BW, state=st, return_state=True)
            off.append(r.cpu().double().numpy())
    return on.cpu().double().numpy(), np.stack(off, 1)


def loss(sim, drives, on_t, off_t, latch):
    """Prereg loss on CALIBRATION targets only (log1p MSE on + off) + PENALTY if the latch check fails."""
    on, off = respond(sim, drives)
    val = float(((np.log1p(on) - on_t) ** 2).mean() + ((np.log1p(off) - off_t) ** 2).mean())
    ok, per = latch(sim)
    return val + (0.0 if ok else PENALTY), val, ok, {k: v["n_above"] for k, v in per.items()}


def pearson_nz(a, b):
    m = (a > 0) | (b > 0)
    if m.sum() < 3:
        return float("nan")
    return float(np.corrcoef(np.log1p(a[m]), np.log1p(b[m]))[0, 1])


def metrics(on, off, lif_on, lif_off, kc, C, latch_ok):
    """Held-out L1'-L4 (prereg + am.1). on [H, N], off [H, bins, N]; lif_* are LIF 5-seed means."""
    r = [pearson_nz(on[h], lif_on[h]) for h in range(len(on))]
    kc_m, kc_l = (on[:, kc] > 1).mean(1), (lif_on[:, kc] > 1).mean(1)
    tail_m, tail_l = (off[:, 5:].mean(1) > 1).mean(1), (lif_off[:, 5:].mean(1) > 1).mean(1)
    return {"L1": bool(latch_ok),
            "L2": bool(np.nanmedian(r) >= 0.8 * C), "L2_median_r": float(np.nanmedian(r)), "L2_bar": 0.8 * C,
            "L3": bool((np.abs(kc_m - kc_l) <= 0.5 * kc_l + 0.01).all()),
            "L3_kc_model": kc_m.tolist(), "L3_kc_lif": kc_l.tolist(),
            "L4": bool((tail_m <= tail_l + 0.001).all()),
            "L4_tail_model": tail_m.tolist(), "L4_tail_lif": tail_l.tolist()}


def label(fam):
    """fam: {family: {"L1".."L4": bool, "cal_loss": float} or None (no latch-passing start)}."""
    ok = {f: m for f, m in fam.items() if m is not None}
    passing = {f: m for f, m in ok.items() if all(m[k] for k in ("L1", "L2", "L3", "L4"))}
    if passing:
        return "BASE-PASS", min(passing, key=lambda f: passing[f]["cal_loss"])
    if any(m["L1"] for m in ok.values()):
        return "LATCH-ONLY", None
    return "NO-BASE", None


def selftest():
    T = dict(L1=True, L2=True, L3=True, L4=True)
    assert label({"lin": dict(T, cal_loss=2.0), "lif": dict(T, cal_loss=1.0)}) == ("BASE-PASS", "lif")
    assert label({"lin": dict(T, L3=False, cal_loss=1.0), "lif": None}) == ("LATCH-ONLY", None)
    assert label({"lin": None, "lif": None}) == ("NO-BASE", None)
    assert label({"lin": dict(T, L1=False, cal_loss=0.1), "lif": None}) == ("NO-BASE", None)
    # apply() round trip and clamps, on a 3-neuron toy chain
    toy = RateSim(1.0, device="cpu", csr=sp.csr_matrix(([1.0, 1.0], ([0, 1], [1, 2])), shape=(3, 3)),
                  pool=np.array([0, 1, 2]))
    apply(toy, "lin", x0("lin", (1, -35.0282, 1000.0, 5000.0)))
    assert abs(toy.w_scale / W0["lin"] - 1) < 1e-12 and abs(float(toy.log_tau[0].exp()) - 200.0) < 1e-3 \
        and abs(toy.r_max - 1000.0) < 1e-9
    apply(toy, "lif", x0("lif", (1, 0, 0.5)))
    assert abs(float(toy.log_tau[0].exp()) - 2.0) < 1e-5 and toy.r_max == float("inf")
    # the loss never reads held-out targets: NaN held rows are never touched
    on_t, off_t = np.zeros((1, 3)), np.zeros((1, N_BINS, 3))
    v = loss(toy, [(np.array([0]), 10.0)], on_t, off_t, lambda s: (True, {}))
    assert np.isfinite(v[0]), v
    print("selftest ok", flush=True)


def load_ref():
    d = np.load(os.path.join(OUT, "lif_ref.npz"), allow_pickle=False)
    j = json.load(open(os.path.join(OUT, "lif_ref.json")))
    assert j["regime"] == "eln8", "chunk 0 calibrates to the eln8 LIF (amendment 1)"
    return d, j


def drives_of(names, ct):
    return [(np.flatnonzero(ct == t).astype(np.int64), HZ) for t in names]


def run_family(family, smoke=False):
    d, _ = load_ref()
    names, split = d["names"].astype(str), d["split"].astype(str)
    cal = np.flatnonzero(split == "cal")
    on_t = np.log1p(d["rate_on"][cal].mean(1).astype(np.float64))   # held-out rows never loaded here
    off_t = np.log1p(d["rate_off"][cal].astype(np.float64))
    ct = S.C1.sets()["ct"]
    drives = drives_of(names[cal], ct)
    sim = RateSim(1.0, regime="eln8")
    os.makedirs(CELLS, exist_ok=True)
    for i, s in enumerate(STARTS[family][:1] if smoke else STARTS[family]):
        path = os.path.join(CELLS, "%s_S%d%s.json" % (family, i, "_smoke" if smoke else ""))
        if os.path.exists(path):
            print("skip %s (done)" % path, flush=True)
            continue
        hist = []

        def f(x):
            apply(sim, family, x)
            tot, val, ok, n = loss(sim, drives, on_t, off_t, S.latch_check)
            hist.append({"x": x.tolist(), "total": tot, "fit": val, "latch_ok": ok, "n_latched": n})
            print("%s S%d eval %3d  fit %.5f  latch %s %s" % (family, i, len(hist), val, "ok" if ok else "FAIL", n),
                  flush=True)
            return tot

        res = minimize(f, x0(family, s), method="Nelder-Mead",
                       options=dict(NM, maxfev=5) if smoke else NM)
        apply(sim, family, res.x)
        end_ok, end_per = S.latch_check(sim)
        with open(path, "w") as fh:
            json.dump({"family": family, "start": i, "x0": x0(family, s).tolist(), "x": res.x.tolist(),
                       "fun": float(res.fun), "nfev": int(res.nfev), "end_latch_ok": bool(end_ok),
                       "end_latch": end_per, "history": hist}, fh, indent=1)
        print("%s S%d done: fun %.5f nfev %d latch %s" % (family, i, res.fun, res.nfev, end_ok), flush=True)


def analyze():
    d, j = load_ref()
    names, split = d["names"].astype(str), d["split"].astype(str)
    held = np.flatnonzero(split == "held")
    meta = np.load(os.path.join(_HERE, "data", "neuron_meta.npz"), allow_pickle=False)
    kc = meta["cell_class"].astype(str) == "Kenyon_Cell"
    ct = S.C1.sets()["ct"]
    sim = RateSim(1.0, regime="eln8")
    out = {"prereg": "PREREGISTER_rate_chunk0_base.md @ 1a29779 + am.1 b554245", "C": j["C"], "families": {}}
    fam = {}
    for family in ("lin", "lif"):
        cells = [json.load(open(os.path.join(CELLS, "%s_S%d.json" % (family, i)))) for i in range(5)]
        spread = {"x": [c["x"] for c in cells], "fun": [c["fun"] for c in cells],
                  "latch_ok": [c["end_latch_ok"] for c in cells]}
        good = [c for c in cells if c["end_latch_ok"]]
        if not good:
            fam[family] = None
            out["families"][family] = {"chosen": None, "starts": spread}
            print("%s: no start ends latch-OK" % family, flush=True)
            continue
        best = min(good, key=lambda c: c["fun"])
        apply(sim, family, np.array(best["x"]))
        latch_ok, latch_per = S.latch_check(sim)
        on, off = respond(sim, drives_of(names[held], ct))
        m = metrics(on, off, d["rate_on"][held].mean(1), d["rate_off"][held], kc, j["C"], latch_ok)
        m["cal_loss"] = best["fun"]
        fam[family] = m
        out["families"][family] = {"chosen": best["start"], "x": best["x"], "latch": latch_per,
                                   "metrics": m, "starts": spread}
        print("%s chosen S%d fun %.5f: L1 %s L2 %s (median r %.4f vs %.4f) L3 %s L4 %s"
              % (family, best["start"], best["fun"], m["L1"], m["L2"], m["L2_median_r"], m["L2_bar"],
                 m["L3"], m["L4"]), flush=True)
    lab, chosen = label(fam)
    out["label"], out["base_family"] = lab, chosen
    print("LABEL: %s%s" % (lab, " (%s)" % chosen if chosen else ""), flush=True)
    with open(os.path.join(OUT, "result.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    if lab == "BASE-PASS":
        x = out["families"][chosen]["x"]
        base = {"family": chosen, "regime": "eln8", "w_scale": float(np.exp(x[0])), "bias": float(x[1]),
                "tau": float(np.clip(np.exp(x[2]), 2.0, 200.0)),
                "r_max": float(np.clip(np.exp(x[3]), 10.0, 1000.0)) if chosen == "lin" else None}
        rdir = os.path.join(_HERE, "rate", "regress")
        json.dump(base, open(os.path.join(rdir, "c0_base.json"), "w"), indent=1)
        m = out["families"][chosen]["metrics"]
        json.dump({"rows": {k: m[k] for k in ("L1", "L2", "L3", "L4")}}, open(os.path.join(rdir, "c0_record.json"), "w"),
                  indent=1)
        print("wrote rate/regress/c0_base.json + c0_record.json", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", choices=["lin", "lif"])
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    selftest()
    if a.analyze:
        analyze()
    elif a.family:
        run_family(a.family, a.smoke)


if __name__ == "__main__":
    main()
