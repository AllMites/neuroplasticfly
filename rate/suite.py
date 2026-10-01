"""Regression suite for the staged rate-model fit (path 2, PRD phase 4, harness v1).

Every chunk registers: the cell types it OWNS (its parameters; each pool owned by exactly one chunk),
the file holding its fitted values, its battery, and a RECORD of which battery rows passed when the
chunk was closed. After any change, every chunk's battery is re-run on the full model; a row that the
record says passed and that now fails is a REGRESSION. "Nothing learned earlier is silently lost" (D44).
Plus one GLOBAL invariant, the latch check (PREREGISTER_rate_latch_check.md, amended by
PREREGISTER_rate_chunk0_base.md amendment 1): after a stimulus ends, no more neurons stay above 1 Hz than
in the eln8 LIF under the same protocol (+139, rate/regress/latch_ref.json); a condition with no LIF
reference must return to rest. The suite is RED if either fails.

Base model = chunk 0 (rate/regress/c0_base.json: eln8 W, 4 global params calibrated to the eln8 LIF,
PREREGISTER_rate_chunk0_base.md, BASE-PASS 7839719), with its held-out battery L1-L4 as the first record.
base="A" rebuilds the pre-chunk-0 arm-A base (history: chunk-1 fit/loops ran on it). Chunks load on top.

Run: .venv/Scripts/python.exe tools/gpu_slot.py -- .venv/Scripts/python.exe rate/suite.py
Exit 1 on any regression or ownership error.
"""
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _HERE)
from rate import chunk1 as C1  # noqa: E402
from rate.engine import RateSim  # noqa: E402


def c1_battery(sim, _S={}):
    if not _S:
        _S.update(C1.sets())
    return {k: p for k, (p, _) in C1.battery(C1.measure(sim, _S)[1]).items()}


def c0_battery(sim):
    from rate import chunk0 as C0  # lazy: chunk0 imports this module
    d, j = C0.load_ref()
    names, held = d["names"].astype(str), np.flatnonzero(d["split"].astype(str) == "held")
    kc = np.load(os.path.join(_HERE, "data", "neuron_meta.npz"))["cell_class"].astype(str) == "Kenyon_Cell"
    on, off = C0.respond(sim, C0.drives_of(names[held], C1.sets()["ct"]))
    m = C0.metrics(on, off, d["rate_on"][held].mean(1), d["rate_off"][held], kc, j["C"], latch_check(sim)[0])
    return {k: bool(m[k]) for k in ("L1", "L2", "L3", "L4")}


C0_BASE = os.path.join(_HERE, "rate", "regress", "c0_base.json")

CHUNKS = [
    {"name": "c0_base", "owns": None, "params": None,
     "record": "rate/regress/c0_record.json", "battery": c0_battery},
    # ponytail: owns/params stay None until the chunk-1 fit prereg names the owned types
    {"name": "c1_sugar_pam", "owns": None, "params": None,
     "record": "rate/regress/c1_record.json", "battery": c1_battery},
]


def owned_pools(pool, ct, types):
    """Pool ids of `types`; refuses a pool that also holds a neuron of an unowned type."""
    idx = np.flatnonzero(np.isin(ct, types))
    assert len(idx), "owned types %s match no neuron" % (types,)
    pools = np.unique(pool[idx])
    leak = np.isin(pool, pools) & ~np.isin(ct, types)
    assert not leak.any(), "pools of %s also hold %s" % (types, sorted(set(ct[leak]))[:5])
    return pools


def check_ownership(chunks, pool, ct):
    seen = {}
    for ch in chunks:
        if ch["owns"] is None:
            continue
        for p in owned_pools(pool, ct, ch["owns"]):
            assert p not in seen, "pool %d owned by %s and %s" % (p, seen[p], ch["name"])
            seen[p] = ch["name"]
    return seen


def build(chunks=CHUNKS, base="c0", **kw):
    if base == "c0":
        b = json.load(open(C0_BASE))
        sim = RateSim(1.0, regime=b["regime"], **kw)
        sim.transfer, sim.w_scale, sim.r_max = b["family"], b["w_scale"], b["r_max"] or float("inf")
        import torch
        with torch.no_grad():
            sim.bias.fill_(b["bias"])
            sim.log_tau.fill_(float(np.log(b["tau"])))
    else:
        sim = RateSim(1.0, **kw)
        sim.w_scale, bias, sim.r_max = C1.lif_constants()
        sim.bias.fill_(bias)
    ct = C1.sets()["ct"]
    pool = sim.pool.cpu().numpy()
    check_ownership(chunks, pool, ct)
    for ch in chunks:
        if ch["params"] is not None:
            vals = np.load(os.path.join(_HERE, ch["params"]))
            sim.load_params(owned_pools(pool, ct, ch["owns"]), vals)
    return sim


LATCH_ON, LATCH_OFF, LATCH_WIN, LATCH_HZ = 1000.0, 1000.0, 500.0, 1.0  # PREREGISTER_rate_latch_check.md
LATCH_REF = os.path.join(_HERE, "rate", "regress", "latch_ref.json")  # chunk-0 amendment 1


def latch_bounds():
    if not os.path.exists(LATCH_REF):
        return {}
    return {k: v["bound"] for k, v in json.load(open(LATCH_REF))["conditions"].items()}


def latch_conditions():
    St = C1.sets()
    orn = np.flatnonzero(St["ct"] == "ORN_DM4").astype(np.int64)
    return {"FOX": (St["fox"], C1.HZ), "SUGAR": (St["sugar"], C1.HZ),
            "BITTER": (St["bitter"], C1.HZ), "ORN_DM4": (orn, 40.0)}


def latch_check(sim, conds=None, bounds=None):
    """Global invariant: after each drive (LATCH_ON ms, from rest) is removed, the number of neurons
    above LATCH_HZ over the last LATCH_WIN ms of LATCH_OFF (state carried) is <= that condition's LIF
    bound (latch_bounds(); 0 = must return to rest, the default for unreferenced conditions)."""
    bounds = latch_bounds() if bounds is None else bounds
    import torch
    conds = latch_conditions() if conds is None else conds
    names = list(conds)
    empty = [(np.zeros(0, np.int64), 0.0)] * len(names)
    with torch.no_grad():
        _, st = sim.run([conds[k] for k in names], LATCH_ON, return_state=True)
        _, st = sim.run(empty, LATCH_OFF - LATCH_WIN, state=st, return_state=True)
        r = sim.run(empty, LATCH_WIN, state=st).cpu().double().numpy()
    Rset = set(C1.sets()["comp"]["R"].tolist()) if sim.n > 1000 else set()
    per = {}
    for b, k in enumerate(names):
        hot = np.flatnonzero(r[b] > LATCH_HZ)
        per[k] = {"n_above": int(len(hot)), "bound": int(bounds.get(k, 0)), "max_rate": float(r[b].max()),
                  "n_above_in_R": int(sum(i in Rset for i in hot.tolist()))}
    return all(v["n_above"] <= v["bound"] for v in per.values()), per


def run_suite(sim, chunks=CHUNKS):
    report, ok = {}, True
    lok, lper = latch_check(sim)
    report["_latch"] = {"ok": lok, "conditions": lper}
    ok = ok and lok
    for ch in chunks:
        rec = json.load(open(os.path.join(_HERE, ch["record"])))["rows"]
        rows = ch["battery"](sim)
        assert set(rows) == set(rec), "battery rows %s != record rows %s" % (sorted(rows), sorted(rec))
        reg = sorted(r for r in rec if rec[r] and not rows[r])
        gained = sorted(r for r in rec if not rec[r] and rows[r])
        report[ch["name"]] = {"regressions": reg, "gained": gained, "rows": rows}
        ok = ok and not reg
    return ok, report


def main():
    ok, report = run_suite(build())
    for name, r in report.items():
        if name == "_latch":
            print("%-14s %s  %s" % ("latch", "PASS" if r["ok"] else "LATCH",
                                    {k: v["n_above"] for k, v in r["conditions"].items()}))
            continue
        print("%-14s regressions %s  gained %s" % (name, r["regressions"] or "none",
                                                     r["gained"] or "none"))
    print("SUITE GREEN" if ok else "SUITE RED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
