"""Ignition loop test, exactly as PREREGISTER_rate_chunk1_pass2_loops.md Part 2 (0f0b2f7). No fitting.

For each seed: the pass-2 fit at the lowest cap where >= 1 reward compartment is up under FOX.
(a) Fox drive swept 0 -> 200 -> 0 Hz with state carry: BISTABLE / STEP / GRADED / FLAT.
(b) One cut at a time (C1 PAM->PAM, C2 PAM->FDA, C3 FDA->FDA, C4 all three, C5 silence active
    non-owned neurons): which cut stops ignition -> OWNED-LOOP / NEEDS-REST-OF-BRAIN / FEEDFORWARD.

Run: .venv/Scripts/python.exe tools/gpu_slot.py -- .venv/Scripts/python.exe rate/chunk1_loops.py
"""
import collections
import json
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _HERE)
from rate import chunk1 as C1  # noqa: E402
from rate import chunk1_fit as F  # noqa: E402
from rate import suite as S  # noqa: E402

CELLS = os.path.join(_HERE, "results", "rate_chunk1_fit_pass2", "cells")
OUT = os.path.join(_HERE, "results", "rate_chunk1_loops")
SWEEP = list(range(0, 201, 10))
STEP_MS = 300.0


def sweep_label(up, down):
    """up/down: D_R per drive level (down listed in the same drive order as up)."""
    up, down = np.asarray(up, float), np.asarray(down, float)
    if np.max(np.abs(up - down)) >= 1.0:
        return "BISTABLE"
    rng = up.max() - up.min()
    if rng < 1.0:
        return "FLAT"
    return "STEP" if np.max(np.diff(up)) >= 0.5 * rng else "GRADED"


def mechanism_label(kills):
    """kills: {cut: bool}."""
    if kills["C4"]:
        alone = [c for c in ("C1", "C2", "C3") if kills[c]]
        return "OWNED-LOOP(" + ("+".join(alone) if alone else "distributed") + ")"
    return "NEEDS-REST-OF-BRAIN" if kills["C5"] else "FEEDFORWARD"


def majority(labels):
    c = collections.Counter(labels).most_common()
    return c[0][0] if len(c) == 1 or c[0][1] > c[1][1] else "MIXED(%s)" % dict(c)


def _selftest():
    lv = np.arange(21)
    assert sweep_label(np.zeros(21), np.zeros(21)) == "FLAT"
    assert sweep_label(lv * 0.5, lv * 0.5) == "GRADED"
    step = np.where(lv >= 10, 20.0, 0.0)
    assert sweep_label(step, step) == "STEP"
    assert sweep_label(step, np.where(lv >= 5, 20.0, 0.0)) == "BISTABLE"
    k = dict(C1=False, C2=False, C3=False, C4=True, C5=False)
    assert mechanism_label(k) == "OWNED-LOOP(distributed)"
    assert mechanism_label(dict(k, C1=True)) == "OWNED-LOOP(C1)"
    assert mechanism_label(dict(k, C4=False, C5=True)) == "NEEDS-REST-OF-BRAIN"
    assert mechanism_label(dict(k, C4=False)) == "FEEDFORWARD"
    assert majority(["A", "A", "B"]) == "A" and majority(["A", "B"]).startswith("MIXED")


def main():
    _selftest()
    cells = [json.load(open(os.path.join(CELLS, f))) for f in sorted(os.listdir(CELLS))]
    St = C1.sets()
    sim = S.build(chunks=S.CHUNKS[1:], base="A")  # history: ran on the arm-A base
    pool = sim.pool.cpu().numpy()
    tpools = S.owned_pools(pool, St["ct"], F.OWN)
    owned = np.isin(pool, tpools)
    pam_p = S.owned_pools(pool, St["ct"], ["PAM%02d" % i for i in range(1, 16)])
    fda_p = S.owned_pools(pool, St["ct"], C1.FDA_I + C1.FDA_II)
    Ridx = torch.as_tensor(St["comp"]["R"], device=sim.device)
    empty = np.zeros(0, np.int64)

    def fox_D(silence=None):
        with torch.no_grad():
            base = sim.run([(empty, 0.0)], C1.T_RUN, silence=silence)[0]
            r = sim.run([(St["fox"], C1.HZ)], C1.T_RUN, silence=silence)[0]
        D = {c: float((r[i] - base[i]).mean()) for c, i in St["comp"].items()}
        return r, D, sum(D[c] >= C1.UP for c in C1.R_COMP)

    chosen = []
    for seed in F.SEEDS:
        ok = [c for c in cells if c["seed"] == seed
              and any(c["D_comp"]["FOX"][k] >= C1.UP for k in C1.R_COMP)]
        if ok:
            chosen.append(min(ok, key=lambda c: c["cap"]))
    out = {"prereg": "PREREGISTER_rate_chunk1_pass2_loops.md @ 0f0b2f7, Part 2", "fits": []}
    if not chosen:
        out["label"] = "NO-IGNITION"
        print("NO-IGNITION: no pass-2 cell has a reward compartment up under FOX")
    for c in chosen:
        tab = sim.set_edge_gains(tpools, c["cap"])
        x0 = torch.as_tensor(c["edge_x"], device=sim.device, dtype=sim.dtype)
        src, tgt = np.asarray(tab["src_pool"]), np.asarray(tab["tgt_pool"])
        cls = {"C1": np.isin(src, pam_p) & np.isin(tgt, pam_p),
               "C2": np.isin(src, pam_p) & np.isin(tgt, fda_p),
               "C3": np.isin(src, fda_p) & np.isin(tgt, fda_p)}
        cls["C4"] = cls["C1"] | cls["C2"] | cls["C3"]
        sim.edge_x = x0.clone()
        r0, D0, n0 = fox_D()
        # (a) sweep with carry
        up, down, st = [], [], None
        for hz in SWEEP + SWEEP[::-1]:
            with torch.no_grad():
                r, st = sim.run([(St["fox"], float(hz))], STEP_MS, state=st, return_state=True)
            (up if len(up) < len(SWEEP) else down).append(float(r[0][Ridx].mean()))
        down = down[::-1]
        # (b) cuts
        res = {"none": {"n_up": n0, "D_R": D0["R"]}}
        kills = {}
        for cut, m in cls.items():
            x = x0.clone()
            x[torch.as_tensor(np.flatnonzero(m), device=sim.device)] = float("-inf")
            sim.edge_x = x
            _, D, n = fox_D()
            res[cut] = {"n_up": n, "D_R": D["R"], "n_groups": int(m.sum())}
            kills[cut] = n == 0
        sim.edge_x = x0.clone()
        act = np.flatnonzero((r0.cpu().numpy() > 0) & ~owned)
        act = np.setdiff1d(act, St["fox"])
        _, D, n = fox_D(silence=act)
        res["C5"] = {"n_up": n, "D_R": D["R"], "n_silenced": int(len(act))}
        kills["C5"] = n == 0
        f = {"cap": c["cap"], "seed": c["seed"], "sweep_up": up, "sweep_down": down,
             "sweep_label": sweep_label(up, down), "cuts": res, "kills": kills,
             "mechanism": mechanism_label(kills) if n0 > 0 else "NOT-IGNITED-ON-RERUN"}
        out["fits"].append(f)
        print("seed %d cap %g: n_up %d  sweep %s  kills %s  -> %s"
              % (c["seed"], c["cap"], n0, f["sweep_label"], kills, f["mechanism"]), flush=True)
    if chosen:
        out["sweep_summary"] = majority([f["sweep_label"] for f in out["fits"]])
        out["mechanism_summary"] = majority([f["mechanism"] for f in out["fits"]])
        print("SUMMARY: sweep %s, mechanism %s" % (out["sweep_summary"], out["mechanism_summary"]))
    os.makedirs(OUT, exist_ok=True)
    json.dump(out, open(os.path.join(OUT, "result.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
