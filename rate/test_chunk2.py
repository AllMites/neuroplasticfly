"""Checks for the chunk-2 plumbing (paper 1, phase 1): RateSim.set_plastic, RateAsGpuSim, chunk2 schedules.
Toy circuit on CPU float64 first, real C0 base on the GPU second. No learning number is read.

Run: .venv/Scripts/python.exe tools/gpu_slot.py -- .venv/Scripts/python.exe rate/test_chunk2.py
"""
import io
import json
import os
import sys

import numpy as np
import scipy.sparse as sp
import torch

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _HERE)
from learn import condition as C  # noqa: E402
from learn import plastic as PL  # noqa: E402
from rate import chunk2 as K  # noqa: E402
from rate.engine import RateSim  # noqa: E402

# 1. Toy 4-neuron, 5-edge pre-major CSR: set_plastic lands on the right edge in BOTH copies.
rows_, cols_ = np.array([0, 0, 1, 2, 3]), np.array([1, 2, 3, 3, 0])
toy = lambda d: sp.csr_matrix((d, (rows_, cols_)), shape=(4, 4))  # noqa: E731
d0 = np.array([2.0, -1.0, 3.0, 4.0, 0.5])
t = RateSim(0.1, device="cpu", dtype=torch.float64, csr=toy(d0.copy()), pool=np.zeros(4, np.int64))
t.bias.fill_(1.0)
off, val = np.array([1, 3]), np.array([-7.0, 9.0])
t.set_plastic(off, val)
d1 = d0.copy(); d1[off] = val
want = toy(d1).toarray()
post = sp.csr_matrix((t.data[:, 0].numpy(), t.indices.numpy(), t.indptr.numpy()), shape=(4, 4)).toarray()
pre = sp.csr_matrix((t.data_pre[:, 0].numpy(), t.indices_pre.numpy(), t.indptr_pre.numpy()), shape=(4, 4)).toarray()
assert np.array_equal(post.T, want) and np.array_equal(pre, want) and np.array_equal(t.W.toarray(), want)
print("1 toy: set_plastic edits both CSR copies and W", flush=True)

# 2. Toy: run after set_plastic == a fresh RateSim built on the edited CSR.
f = RateSim(0.1, device="cpu", dtype=torch.float64, csr=toy(d1), pool=np.zeros(4, np.int64))
f.bias.fill_(1.0)
drive = [(np.array([0]), 10.0)]
assert torch.equal(t.run(drive, 200.0), f.run(drive, 200.0))
print("2 toy: edited run == fresh run on edited CSR", flush=True)

# 9. (was: set_plastic refuses with edge gains.) Now set_plastic works with gains active; only an offset on
#    a GAINED edge refuses. Pool split [0,0,1,1], gains into pool 1: edges 0->2, 1->3, 2->3 (flat 1,2,3)
#    are gained; flat 0 (0->1) and 4 (3->0) stay frozen. Goldens 9 (a: gains == 1.0 is the no-gain run),
#    9b (gained offset refuses), 9c (full2frozen maps every frozen edge to the same weight). Engine-level
#    but kept here with the set_plastic goldens.
pool4 = np.array([0, 0, 1, 1])
mk4 = lambda: RateSim(0.1, device="cpu", dtype=torch.float64, csr=toy(d0.copy()), pool=pool4)  # noqa: E731
off_ok, val_ok = np.array([0, 4]), np.array([-7.0, 9.0])
ng, gd = mk4(), mk4()
for q in (ng, gd):
    q.bias.fill_(1.0)
gd.set_edge_gains([1], cap=4.0)
assert np.allclose(gd.edge_gain().numpy(), 1.0, rtol=1e-12), gd.edge_gain()
ng.set_plastic(off_ok, val_ok)
gd.set_plastic(off_ok, val_ok)
d2 = d0.copy(); d2[off_ok] = val_ok
assert np.array_equal(gd.W.toarray(), toy(d2).toarray()) and gd.W.nnz == 5, "W must stay the full matrix"
base_run = mk4(); base_run.bias.fill_(1.0)
r_ng, r_gd = ng.run(drive, 200.0), gd.run(drive, 200.0)
assert torch.allclose(r_ng, r_gd, rtol=1e-6, atol=0), (r_ng - r_gd).abs().max()
assert not torch.allclose(r_gd, base_run.run(drive, 200.0)), "plastic edit had no effect (vacuous golden)"
print("9 toy: gains 1.0 + set_plastic == no-gain set_plastic run (max |diff| %.1e)" % float((r_ng - r_gd).abs().max()), flush=True)
try:
    gd.set_plastic(np.array([1]), np.array([5.0]))
    raise SystemExit("9b FAIL: set_plastic accepted a gained edge")
except AssertionError:
    print("9b toy: a plastic offset on a gained edge refuses", flush=True)
assert gd.W.data[1] == d2[1], "refused call must not touch W"
rng = np.random.default_rng(3)
n9 = 40
A = sp.random(n9, n9, density=0.2, random_state=3, format="csr", data_rvs=lambda k: rng.integers(1, 9, k) * rng.choice([-1, 1], k)).astype(np.float64)
big = RateSim(0.1, device="cpu", dtype=torch.float64, csr=A.copy(), pool=np.arange(n9) % 5)
big.set_edge_gains([1, 3], cap=4.0)
pre = sp.csr_matrix((big.data_pre[:, 0].numpy(), big.indices_pre.numpy(), big.indptr_pre.numpy()), shape=(n9, n9))
f2f = big._full2frozen
gained = np.isin(big.pool.numpy()[A.indices], [1, 3])
assert ((f2f < 0) == gained).all() and big.data_pre.shape[0] == int((~gained).sum())
assert np.array_equal(pre.data[f2f[~gained]], A.data[~gained]) and np.array_equal(pre.indices[f2f[~gained]], A.indices[~gained])
ok_off = np.flatnonzero(~gained)[::3]
big.set_plastic(ok_off, np.full(len(ok_off), 11.0))
assert (big.data_pre[big._full2frozen[ok_off], 0] == 11.0).all() and (big.W.data[ok_off] == 11.0).all()
post = sp.csr_matrix((big.data[:, 0].numpy(), big.indices.numpy(), big.indptr.numpy()), shape=(n9, n9))
assert np.array_equal(post.T.toarray(), sp.csr_matrix((big.data_pre[:, 0].numpy(), big.indices_pre.numpy(), big.indptr_pre.numpy()), shape=(n9, n9)).toarray())
print("9c toy: full2frozen maps 40-neuron random W (weights + columns), both CSR copies agree", flush=True)

# 10. Toy: the stability guard. Reference on first sighting (ratios exactly 1.0), a forced breach raises
#     GuardAbort and leaves a guard row + a FAIL-UNSTABLE abort row; a reference MBON06 < 0.5 Hz switches to
#     the absolute band [0, 1] Hz.
def toy_guard(bias_ref, bias_now, log):
    g = K.GuardedSim(RateSim(0.1, device="cpu", dtype=torch.float64, csr=toy(d0.copy()), pool=np.zeros(4, np.int64)),
                     mbon06=[3])
    drv = C.to_prob(np.array([0]), np.array([10.0]))
    g.arm_guard("toy", (), log)
    g.songs[K._key(*drv)] = "toy_probe"
    g.r.bias.fill_(bias_ref)
    g.run_batch([drv], [0], t_run=C.T_RUN)
    g.r.bias.fill_(bias_now)
    return lambda: g.run_batch([drv], [0], t_run=C.T_RUN)


lg = io.StringIO()
toy_guard(1.0, 1.0, lg)()                                  # reference + identical rerun: no breach
r = [json.loads(x) for x in lg.getvalue().splitlines()]
assert len(r) == 2 and all(x["phase"] == "guard" and not x["breach"] and x["active_frac_ratio"] == 1.0
                           and x["mbon06_ratio"] == 1.0 and x["mbon06_mode"] == "ratio" for x in r), r
lg = io.StringIO()
try:
    toy_guard(1.0, 0.0, lg)()                              # MBON06 1.62 -> 0.48 Hz = 0.30x: below 0.5x
    raise SystemExit("10 FAIL: MBON06 collapse did not abort")
except K.GuardAbort as e:
    assert e.info["breached"] == ["mbon06"] and e.info["breach"] and e.info["mbon06_mode"] == "ratio", e.info
lg = io.StringIO()
try:
    toy_guard(-5.0, 1.0, lg)()                             # ref MBON06 0 Hz -> absolute band; active 1/4 -> 3/4
    raise SystemExit("10 FAIL: active-fraction blow-up did not abort")
except K.GuardAbort as e:
    assert e.info["mbon06_mode"] == "absolute" and e.info["breached"] == ["active_frac", "mbon06"], e.info
lg = io.StringIO()
assert K.guarded(K.GuardedSim.__new__(K.GuardedSim), "toy", (), lg, lambda: 7) == 7     # non-arming passthrough
lg = io.StringIO()
def boom():
    raise K.GuardAbort({"arm": "toy", "phase": "guard", "breach": True})
assert K.guarded(None, "toy", (), lg, boom) is None
ab = json.loads(lg.getvalue())
assert ab["label_hint"] == "FAIL-UNSTABLE" and ab["arm"] == "toy" and ab["breach"] is True and ab["phase"] == "abort", ab
print("10 toy: guard raises on forced breach, absolute band for MBON06 < 0.5 Hz, abort row written", flush=True)


# 13. Headroom gate decision on a fake readout dict (pure function; real reference.json numbers). Normalised effects
#     need only a nonzero, unsaturated baseline: the absolute-rate threshold is gone.
ref = K.load_reference()
assert not hasattr(K, "gate_threshold") and not hasattr(K, "GATE_FRAC")
RM = 124.4
fake = lambda a, v=0.2, k=0.06: {"mbon_approach_hz": a, "mbon_avoid_hz": v, "kc_active": k}  # noqa: E731
ok = K.headroom_row("rung_a", "dc2", fake(11.3), ref, RM)
assert ok["gate_pass"] and ok["reason"] is None and ok["phase"] == "headroom" and abs(ok["approach_ratio"] - 11.3 / ref["rung_a"]["baseline"]["dc2"]["mbon_approach_hz"]["mean"]) < 1e-9
assert "threshold_hz" not in ok
zero = K.headroom_row("rung_a", "dc2", fake(0.0), ref, RM)
assert not zero["gate_pass"] and zero["reason"] == "approach_zero"
assert K.headroom_row("rung_a", "dc2", fake(0.01), ref, RM)["gate_pass"]                      # small but nonzero baseline passes
sat = K.headroom_row("rung_a", "dc2", fake(RM), ref, RM)
assert not sat["gate_pass"] and sat["reason"] == "approach_at_r_max"
sv = K.headroom_row("rung_a", "dc2", fake(11.3, v=0.96 * RM), ref, RM)
assert not sv["gate_pass"] and sv["reason"] == "avoid_at_r_max"
assert K.headroom_row("rung3", "d", fake(6.0), ref, RM)["gate_pass"] and not K.headroom_row("rung3", "d", fake(0.0), ref, RM)["gate_pass"]
assert K.headroom_row("rung3", "d", fake(12.4), ref, RM)["approach_ratio"] == 12.4 / ref["rung3"]["baseline_approach_hz"]["d"]["mean"]
assert ok["ref_src_approach"] == "rung_a.baseline" and K.headroom_row("rung3", "d", fake(6.0), ref, RM)["ref_src_approach"] == "rung3.baseline_approach_hz"
json.dumps(ok)
print("13 toy: headroom gate zero baseline -> fail, at r_max -> fail, normal -> pass (no absolute threshold)", flush=True)

# ---------------------------------------------------------------- real C0 base
sim = K.base()
rs = sim.r
orn = C.odour_drive("dc2")[0]

# 3. Identity push: untouched Plastic writes back the baseline weights bit-exactly.
before = rs.run([(orn, C.ORN_HZ)], C.T_RUN)
P = PL.Plastic.real()
w_before = rs.W.data[P.offsets].copy()
P.push(sim)
assert np.array_equal(rs.W.data[P.offsets], w_before), "push changed baseline weights"
assert torch.equal(rs.run([(orn, C.ORN_HZ)], C.T_RUN), before), "push changed rates"
print("3 real: identity push is bit-exact (%d edges)" % P.n_edges, flush=True)

# 4. Adapter count round trip: C.trial == direct run within float32 rounding.
d = np.abs(C.trial(sim, "dc2") - before[0].cpu().numpy()).max()
assert d < 1e-4, d
print("4 real: C.trial via adapter == RateSim.run (max |diff| %.1e Hz)" % d, flush=True)

# 5. State carry through the adapter: 3 x 100 ms == 300 ms. State exact; rates to float32 summation
#    order (as rate/test_engine.py test 4).
p = [C.to_prob(*C.odour_drive("dc2"))]
st, acc = None, 0
for _ in range(3):
    c, st = sim.run_batch(p, [0], t_run=100.0, state=st, return_state=True)
    acc = acc + c
c300, st300 = sim.run_batch(p, [0], t_run=300.0, return_state=True)
assert torch.equal(st, st300), "carried state != uninterrupted state"
assert torch.allclose(acc, c300, rtol=1e-5, atol=1e-6)
print("5 real: adapter carries state exactly", flush=True)

# 6. Empty drive (persist.relax path).
e = sim.run_batch([(np.zeros(0, np.int64), np.zeros(0, np.float64))], [0], t_run=100.0)
assert tuple(e.shape) == (1, rs.n), e.shape
print("6 real: empty drive runs, shape %s" % (tuple(e.shape),), flush=True)

# 7. Lesion arm through condition.run_arm: rule off, weights never move.
P7, _ = K.rung_a(sim, ["lesion"], 5e-6, PL.LAM, io.StringIO(), n_train=2, n_probe=1)["lesion"]
assert not P7.dw.any(), "lesion arm moved weights"
print("7 real: lesion arm leaves dw all zero", flush=True)

# 8. bidir_mb structure only (values not read).
P8, r8 = K.bidir_mb(sim, "timed", io.StringIO(), cycles=1, f_ticks=4, b_ticks=4, every=3)
tags = [r["after"] for r in r8]
assert tags == [x for x in ("T0", "F1", "B1") for _ in C.STIM_SETS["odour"]], tags
n_us = {r["after"]: r["n_us"] for r in r8}
assert n_us["F1"] == n_us["B1"] == 1, n_us
assert {"avoid_index", "song", "w_mean_frac"} <= set(r8[0])
print("8 real: bidir_mb schedule T0/F1/B1, US count matched", flush=True)

# 11. Guard on real C0, no plasticity: lesion arm, 2 training steps. Probes are sighted at baseline and at
#     trial 2 (plus any unpaired training trial, same drive); nothing moves, so the engine is deterministic and every ratio is exactly 1.0, no abort.
lg = io.StringIO()
out11 = K.rung_a(sim, ["lesion"], 5e-6, PL.LAM, lg, n_train=2, n_probe=1)
assert out11["lesion"] is not None, "11 FAIL: guard aborted a no-plasticity arm"
g11 = [json.loads(x) for x in lg.getvalue().splitlines()]
g11 = [x for x in g11 if x.get("phase") == "guard"]
assert {x["song"] for x in g11} == set(C.STIM_SETS["odour"]) and len(g11) >= 2 * len(C.STIM_SETS["odour"]), len(g11)
# (aggregates only, no learning number). Real C0 odour probes have MBON06 = 0 Hz, so the absolute band is live:
# mbon06_ratio is then null and mbon06_hz must equal the reference exactly (0.0 here).
assert all(not x["breach"] and x["active_frac_ratio"] == 1.0 and x["mbon06_mode"] == "absolute"
           and x["mbon06_ratio"] is None and x["mbon06_hz"] == 0.0 for x in g11), g11
assert not any("label_hint" in x for x in g11)
print("11 real: no-plasticity arm, %d guard rows, active ratio exactly 1.0, MBON06 unchanged, no abort" % len(g11), flush=True)

# 12. Guard on a short bidir lesion arm (open-loop ticks carry state and run 100 ms: must NOT be checked
#     against the 300 ms fresh-state probe reference). Lesion = weights never move, so every probe repeats
#     the reference exactly: active ratio 1.0, one guard row per probe per test tag, no abort.
lg = io.StringIO()
K.guarded(sim, "bidir_lesion", C.STIM_SETS["odour"], lg,
          lambda: K.bidir_mb(sim, "timed", lg, cycles=1, f_ticks=4, b_ticks=4, every=3, lesion="all"))
rows12 = [json.loads(x) for x in lg.getvalue().splitlines()]
g12 = [x for x in rows12 if x.get("phase") == "guard"]
assert not any(x.get("phase") == "abort" for x in rows12), "12 FAIL: guard aborted a lesion bidir arm"
assert len(g12) == len(C.STIM_SETS["odour"]) * 3, len(g12)           # tags T0, F1, B1
assert all(x["active_frac_ratio"] == 1.0 and not x["breach"] for x in g12), g12
print("12 real: bidir lesion arm, %d guard rows (probes x tags), active ratio exactly 1.0" % len(g12), flush=True)

# 14. Real C0 headroom gate + latch rows (baseline only, no training): rows written, guard untouched.
lg = io.StringIO()
sim.arm_guard("x14", C.STIM_SETS["odour"], lg)
hr = {p: K.headroom(sim, ref, p, lg) for p in K.GATE_PROTOCOLS}
lt = {p: K.latch_check(sim, lg, p, "baseline", "baseline") for p in K.GATE_PROTOCOLS}
rows14 = [json.loads(x) for x in lg.getvalue().splitlines()]
assert not any(x["phase"] == "guard" for x in rows14) and not sim.ref, "14 FAIL: gate runs leaked into the guard"
assert [len(hr[p]) for p in K.GATE_PROTOCOLS] == [1, 2, 1] and sum(x["phase"] == "headroom" for x in rows14) == 4
assert all(isinstance(x["gate_pass"], bool) and "r_max_hz" in x for x in rows14 if x["phase"] == "headroom")
L = [x for x in rows14 if x["phase"] == "latch"]
assert len(L) == 4 and [x["song"] for x in lt["rung3"]] == ["dc2", "d"]
for x in L:
    assert x["when"] == "baseline" and x["protocol"] in K.GATE_PROTOCOLS and isinstance(x["latched"], bool)
    for k in ("n_latched", "n_latched_kc_paired", "n_kc_paired", "n_latched_mbon_approach", "n_latched_mbon_avoid",
              "approach_latched_rate_frac", "n_latched_ppl1", "n_latched_pam", "ppl1_hz_offset", "pam_hz_offset"):
        assert np.isfinite(x[k]) and x[k] >= 0, (k, x)
    assert 0.0 <= x["approach_latched_rate_frac"] <= 1.0 and x["n_kc_paired"] > 0
    assert x["latched"] == (x["n_latched_kc_paired"] > K.LATCH_KC_MIN or x["approach_latched_rate_frac"] >= K.LATCH_APPROACH_FRAC)
print("14 real: headroom gate %s; latch baseline %s" % ({p: [x["gate_pass"] for x in v] for p, v in hr.items()},
      [(x["song"], x["n_latched_kc_paired"], round(x["approach_latched_rate_frac"], 3), x["latched"]) for x in lt["rung3"]]), flush=True)

# 15. Science path: prologue (headroom + baseline latch) and end-of-arm latch land in the science log.
#     2-trial lesion arm, log only; no learning number is read or printed.
lg = io.StringIO()
K.science_prologue(sim, "rung_a", lg)
K.rung_a(sim, ["lesion"], 5e-6, PL.LAM, lg, n_train=2, n_probe=1, end_latch=("lesion",))
rows15 = [json.loads(x) for x in lg.getvalue().splitlines()]
ph = [x["phase"] for x in rows15]
assert ph.index("headroom") < ph.index("latch") < ph.index("train") if "train" in ph else ph.index("headroom") < ph.index("latch")
assert sum(x["phase"] == "headroom" for x in rows15) == 1
lat15 = [x for x in rows15 if x["phase"] == "latch"]
assert [(x["when"], x["arm"]) for x in lat15] == [("baseline", "baseline"), ("end", "lesion")], lat15
assert rows15.index(lat15[0]) < min(i for i, x in enumerate(rows15) if x["phase"] not in ("headroom", "latch", "guard"))
assert rows15.index(lat15[1]) > max(i for i, x in enumerate(rows15) if x["phase"] not in ("headroom", "latch", "guard"))
assert not any(x["phase"] == "abort" for x in rows15)
print("15 real: science log has headroom + baseline latch before the arm and end latch after it", flush=True)

# 16-23. --analyze labeller on synthetic rows (pure, no sim). Goldens: clear pass, 40% effect, guard abort, headroom
#     fail, zero dw, latch, needs-spiking, bidir pair, diagnostic precedence, file output.
import tempfile  # noqa: E402

A_MEAN = ref["rung_a"]["learn_dc2_approach_drop"]["mean"]   # paper-0 DC2 approach drop; test inputs are multiples of the reference, never literals
W_REF = ref["rung_a"]["learn_w_mean_frac_trial30"]["mean"]
R3_W_REF = ref["rung3"]["w_mean_frac"]["end_trainC"]["mean"]
NAIVE = ref["rung_a"]["naive_d_approach_drop"]["mean"]       # synthetic reversed-arm D approach drop = the paper-0 naive D approach drop
T_A = ref["rung_a"]["test_trial"]


APP_DC2 = ref["rung_a"]["baseline"]["dc2"]["mbon_approach_hz"]["mean"]   # paper-0 baseline approach-MBON rate per odour
APP_D = ref["rung_a"]["baseline"]["d"]["mbon_approach_hz"]["mean"]


def sub_exact(b, d):
    """b - d such that b - (b - d) == d exactly (the inclusive boundary golden needs the drop to round-trip)."""
    t = b - d
    for step in (0, 1, -1, 2, -2, 3, -3):
        c = t
        for _ in range(abs(step)):
            c = float(np.nextafter(c, np.inf if step > 0 else -np.inf))
        if b - c == d:
            return c
    return t


def syn_rung_a(learn=A_MEAN, lesion=0.0, shuffle=0.0, rev_d=NAIVE, w=W_REF, skip=(), app=1.0, avoid_rise=0.0):
    """learn/lesion/shuffle/rev_d are APPROACH drops (Hz); app = baseline approach as a multiple of paper 0's (the rate model's
    is 2.35x); avoid_rise = extra avoid-MBON Hz on the learn arm's DC2 test (moves the total avoid_index, not the approach)."""
    drops = {"learn": {"dc2": learn, "d": 1.0, "da1": 0.5}, "lesion": {"dc2": lesion}, "shuffle": {"dc2": shuffle},
             "reversed": {"d": rev_d}, "frozen": {}}
    rows = []
    for arm, sh in drops.items():
        if arm in skip:
            continue
        for s in C.STIM_SETS["odour"]:
            b, av0 = app * (APP_DC2 if s == "dc2" else APP_D), 0.2 * app
            rise = avoid_rise if (arm, s) == ("learn", "dc2") else 0.0
            d = sh.get(s, 0.0)
            rows.append({"arm": arm, "phase": "baseline", "trial": 0, "song": s, "avoid_index": av0 - b, "mbon_approach_hz": b, "mbon_avoid_hz": av0})
            rows.append({"arm": arm, "phase": "test", "trial": T_A, "song": s, "avoid_index": d + rise + av0 - b,
                         "mbon_approach_hz": sub_exact(b, d), "mbon_avoid_hz": av0 + rise})
        rows.append({"arm": arm, "phase": "train", "trial": T_A, "song": "dc2", "avoid_index": 0.0,
                     "w_mean_frac": w if arm == "learn" else 0.0})
    return rows


def syn_rung3(d=NAIVE * ref["rung3"]["approach_ratio_of_means"], w=R3_W_REF):
    """d = interfere-arm D approach drop (relax_test end minus trainC_test end)."""
    rows = [{"arm": "interfere", "phase": "relax_test", "trial": 55, "song": "d", "avoid_index": 7.0, "mbon_approach_hz": 99.0, "mbon_avoid_hz": 0.2},   # earlier test: ignored
            {"arm": "interfere", "phase": "relax_test", "trial": 60, "song": "d", "avoid_index": 2.0, "mbon_approach_hz": 40.0, "mbon_avoid_hz": 0.2},
            {"arm": "interfere", "phase": "trainC_test", "trial": 90, "song": "d", "avoid_index": 2.0 + d, "mbon_approach_hz": 40.0 - d, "mbon_avoid_hz": 0.2},
            {"arm": "interfere", "phase": "trainC", "trial": 90, "song": "d", "avoid_index": 0.0, "w_mean_frac": w}]
    return rows


def syn_bidir(rule, rises, drops, f1_floor_ok=True, w=W_REF, app=1.0):
    """rises[k-1] = F_k approach drop over the previous test, drops[k-1] = B_k approach recovery under F_k."""
    rows, cur = [], 0.0
    base = app * APP_DC2
    mk = lambda tag, cur, w: {"rule": rule, "after": tag, "song": "dc2", "avoid_index": cur, "w_mean_frac": w,  # noqa: E731
                              "mbon_approach_hz": base - cur, "mbon_avoid_hz": 0.2 * app}
    rows.append(mk("T0", cur, 0.0))
    for k, (r, d) in enumerate(zip(rises, drops), 1):
        cur += r
        rows.append(mk("F%d" % k, cur, w))
        cur -= d
        rows.append(mk("B%d" % k, cur, w))
    return rows


HEAD = lambda p, ok=True: {"arm": "headroom", "phase": "headroom", "protocol": p, "song": "dc2", "gate_pass": ok, "reason": None if ok else "approach_at_r_max"}  # noqa: E731
LATCH = lambda p, when, latched: {"arm": "baseline", "phase": "latch", "protocol": p, "song": "dc2", "when": when, "latched": latched,  # noqa: E731
                                  "n_latched_kc_paired": 120 if latched else 0, "approach_latched_rate_frac": 0.0}
ABORT = lambda arm: {"arm": arm, "phase": "abort", "label_hint": "FAIL-UNSTABLE", "song": "dc2"}  # noqa: E731
GOOD_B = dict(rises=[A_MEAN, A_MEAN, A_MEAN, A_MEAN], drops=[A_MEAN] * 4)
DEPR = dict(rises=[A_MEAN, 0.1 * A_MEAN, 0.1 * A_MEAN, 0.1 * A_MEAN], drops=[0.2 * A_MEAN] * 4)


def batch(rung_a=None, rung3=None, bt=None, bd=None, extra=()):
    L = {"rung_a": syn_rung_a() if rung_a is None else rung_a, "rung3": syn_rung3() if rung3 is None else rung3,
         "bidir_timed": syn_bidir("timed", **GOOD_B) if bt is None else bt,
         "bidir_depress": syn_bidir("depress", **DEPR) if bd is None else bd}
    for name, row in extra:
        L[name] = L[name] + [row]
    return K.label_protocols(L, ref)


def want(res, **kw):
    got = {p: (res[p]["label"], res[p]["diagnostic"]) for p in res}
    exp = {"rung_a": ("PASS", None), "rung3": ("PASS", None), "bidir": ("PASS", None)}
    exp.update(kw)
    assert got == exp, got


# 16 clear pass (and the exact-threshold boundary is inclusive)
want(batch())
want(batch(rung_a=syn_rung_a(learn=0.5 * A_MEAN)))
print("16 analyze: clear pass on all three protocols", flush=True)

# 17 a 40% effect fails rung A with NEEDS-SPIKING; a no-effect learn arm too; must beat BOTH controls
want(batch(rung_a=syn_rung_a(learn=0.4 * A_MEAN)), rung_a=("FAIL", "NEEDS-SPIKING"))
want(batch(rung_a=syn_rung_a(learn=A_MEAN, shuffle=A_MEAN * 1.1)), rung_a=("FAIL", "NEEDS-SPIKING"))
want(batch(rung_a=syn_rung_a(learn=A_MEAN, lesion=A_MEAN)), rung_a=("FAIL", "NEEDS-SPIKING"))
want(batch(rung_a=syn_rung_a(learn=-A_MEAN)), rung_a=("FAIL", "NEEDS-SPIKING"))
print("17 analyze: 40% effect / control tie / wrong sign -> FAIL NEEDS-SPIKING", flush=True)

# 18 guard abort: the aborted learn arm has no test row -> FAIL-UNSTABLE; an abort in an unneeded arm changes nothing
cut = [r for r in syn_rung_a() if not (r["arm"] == "learn" and r["phase"] in ("test", "train"))]
want(batch(rung_a=cut + [ABORT("learn")]), rung_a=("FAIL", "FAIL-UNSTABLE"))
want(batch(rung_a=syn_rung_a() + [ABORT("frozen")]))
want(batch(rung_a=syn_rung_a(skip=("reversed",)) + [ABORT("reversed")]), rung_a=("PASS", None), rung3=("FAIL", "FAIL-UNSTABLE"))
want(batch(bt=[r for r in syn_bidir("timed", **GOOD_B) if r["after"] in ("T0", "F1")] + [ABORT("bidir_timed")]), bidir=("FAIL", "FAIL-UNSTABLE"))
print("18 analyze: guard abort -> FAIL-UNSTABLE, only for needed arms", flush=True)

# 18b abort AFTER the scoring rows exist (breach on the d / da1 probe after dc2 was written) still voids the verdict
want(batch(rung_a=syn_rung_a() + [ABORT("learn")]), rung_a=("FAIL", "FAIL-UNSTABLE"))
want(batch(rung3=syn_rung3() + [ABORT("interfere")]), rung3=("FAIL", "FAIL-UNSTABLE"))
want(batch(rung_a=syn_rung_a() + [ABORT("reversed")]), rung3=("FAIL", "FAIL-UNSTABLE"))
want(batch(bt=syn_bidir("timed", **GOOD_B) + [ABORT("bidir_timed")]), bidir=("FAIL", "FAIL-UNSTABLE"))
want(batch(bd=syn_bidir("depress", **DEPR) + [ABORT("bidir_depress")]), bidir=("FAIL", "FAIL-UNSTABLE"))
print("18b analyze: abort after full scoring rows -> FAIL-UNSTABLE", flush=True)

# 19 headroom fail -> READOUT-CEILING, only that protocol, and it outranks every other diagnostic
want(batch(extra=[("rung_a", HEAD("rung_a", False))]), rung_a=("PASS", None))             # a PASS stays PASS
res = batch(rung_a=syn_rung_a(learn=0.4 * A_MEAN, w=0.0) + [ABORT("learn"), HEAD("rung_a", False), LATCH("rung_a", "baseline", True)])
assert (res["rung_a"]["label"], res["rung_a"]["diagnostic"]) == ("FAIL", "READOUT-CEILING") and res["rung3"]["label"] == "PASS"
want(batch(rung3=syn_rung3(d=0.0) + [HEAD("rung3", False)]), rung3=("FAIL", "READOUT-CEILING"))
print("19 analyze: headroom fail -> READOUT-CEILING (first in order)", flush=True)

# 20 zero dw -> NO-WEIGHT-CHANGE by magnitude; w just above 0.1x is not; outranks latch; abort outranks it
lo, hi = 0.05 * abs(W_REF), 0.2 * abs(W_REF)
want(batch(rung_a=syn_rung_a(learn=0.4 * A_MEAN, w=0.0)), rung_a=("FAIL", "NO-WEIGHT-CHANGE"))
want(batch(rung_a=syn_rung_a(learn=0.4 * A_MEAN, w=-lo)), rung_a=("FAIL", "NO-WEIGHT-CHANGE"))
want(batch(rung_a=syn_rung_a(learn=0.4 * A_MEAN, w=+lo)), rung_a=("FAIL", "NO-WEIGHT-CHANGE"))
want(batch(rung_a=syn_rung_a(learn=0.4 * A_MEAN, w=-hi)), rung_a=("FAIL", "NEEDS-SPIKING"))
want(batch(rung_a=syn_rung_a(learn=0.4 * A_MEAN, w=0.0) + [LATCH("rung_a", "baseline", True)]), rung_a=("FAIL", "NO-WEIGHT-CHANGE"))
want(batch(rung3=syn_rung3(d=0.0, w=0.0)), rung3=("FAIL", "NO-WEIGHT-CHANGE"))
want(batch(bt=syn_bidir("timed", **dict(GOOD_B, rises=[0.4 * A_MEAN] * 4), w=0.0)), bidir=("FAIL", "NO-WEIGHT-CHANGE"))
cut = [r for r in syn_rung_a(w=0.0) if not (r["arm"] == "learn" and r["phase"] == "test")]
want(batch(rung_a=cut + [ABORT("learn")]), rung_a=("FAIL", "FAIL-UNSTABLE"))
print("20 analyze: zero / tiny dw -> NO-WEIGHT-CHANGE (abs value), below latch, above abort", flush=True)

# 21 latch: baseline latched True -> LATCH-CONFOUNDED on a failing protocol; end rows never gate
bad = syn_rung_a(learn=0.4 * A_MEAN)
want(batch(rung_a=bad + [LATCH("rung_a", "baseline", True)]), rung_a=("FAIL", "LATCH-CONFOUNDED"))
want(batch(rung_a=bad + [LATCH("rung_a", "baseline", False), LATCH("rung_a", "end", True)]), rung_a=("FAIL", "NEEDS-SPIKING"))
want(batch(rung_a=syn_rung_a() + [LATCH("rung_a", "baseline", True)]))                    # a PASS stays PASS
want(batch(rung_a=bad + [LATCH("rung3", "baseline", True)]), rung_a=("FAIL", "NEEDS-SPIKING"))   # other protocol's latch
want(batch(extra=[("rung3", LATCH("rung3", "baseline", True))], rung3=syn_rung3(d=0.0)), rung3=("FAIL", "LATCH-CONFOUNDED"))
print("21 analyze: LATCH-CONFOUNDED reads baseline rows of its own protocol only", flush=True)

# 22 rung 3: ratio against the rung A reversed-arm D shift; same sign AND >= 0.5x the paper-0 ratio
r3ref = ref["rung3"]["approach_ratio_of_means"]
want(batch(rung3=syn_rung3(d=NAIVE * 0.51 * r3ref)))
want(batch(rung3=syn_rung3(d=NAIVE * 0.4 * r3ref)), rung3=("FAIL", "NEEDS-SPIKING"))
want(batch(rung3=syn_rung3(d=-NAIVE * r3ref)), rung3=("FAIL", "NEEDS-SPIKING"))
want(batch(rung_a=syn_rung_a(rev_d=-NAIVE)), rung3=("FAIL", "NEEDS-SPIKING"))              # mixed signs fail
want(batch(rung_a=syn_rung_a(rev_d=-NAIVE), rung3=syn_rung3(d=-NAIVE * r3ref)), rung3=("FAIL", "NEEDS-SPIKING"))  # both negative fail
tiny = 0.1 * NAIVE                                                                          # below the naive floor: ratio is fine, denominator is not
want(batch(rung_a=syn_rung_a(rev_d=tiny), rung3=syn_rung3(d=tiny * r3ref)), rung3=("FAIL", "NEEDS-SPIKING"))
want(batch(rung_a=syn_rung_a(rev_d=0.6 * NAIVE), rung3=syn_rung3(d=0.6 * NAIVE * r3ref)))   # above the floor
want(batch(rung_a=syn_rung_a(rev_d=0.0)), rung3=("FAIL", "NEEDS-SPIKING"))                 # zero denominator: no ratio
print("22 analyze: rung 3 ratio sign and size rule", flush=True)

# 23 bidir pair: sustained relearning beats depress -> PASS; same drop as depress / small F1 / 1 of 3 cycles -> FAIL
want(batch())
same = dict(GOOD_B, drops=DEPR["drops"])
want(batch(bt=syn_bidir("timed", **same)), bidir=("FAIL", "NEEDS-SPIKING"))                # drop not larger than depress
weak = dict(GOOD_B, rises=[0.4 * A_MEAN] * 4, drops=[0.4 * A_MEAN] * 4)
want(batch(bt=syn_bidir("timed", **weak)), bidir=("FAIL", "NEEDS-SPIKING"))                # F1 rise below the borrowed floor
one = dict(GOOD_B, rises=[A_MEAN, A_MEAN, 0.4 * A_MEAN, 0.4 * A_MEAN])
want(batch(bt=syn_bidir("timed", **one)), bidir=("FAIL", "NEEDS-SPIKING"))                 # 1 of 3 cycles relearns
two = dict(GOOD_B, rises=[A_MEAN, A_MEAN, A_MEAN, 0.4 * A_MEAN])
want(batch(bt=syn_bidir("timed", **two)))                                                   # 2 of 3 is enough
nodrop = dict(GOOD_B, drops=[0.0] * 4)                                                      # timed drop 0 still beats a negative depress drop
neg = dict(DEPR, drops=[-0.1 * A_MEAN] * 4)
want(batch(bt=syn_bidir("timed", **nodrop), bd=syn_bidir("depress", **neg)), bidir=("FAIL", "NEEDS-SPIKING"))
want(batch(bd=[]), bidir=("FAIL", "INCOMPLETE-DATA"))                                       # needs both rules
print("23 analyze: bidir size rule pair (pass / fail), needs both rules", flush=True)


# ---- file-level goldens (24-28, 32-38): 24 runs = c0 + c1s0..4 x 4 logs, each = meta row first, rows, done row last.
import hashlib  # noqa: E402
import contextlib  # noqa: E402
import random  # noqa: E402

STD = lambda log, **kw: dict({"mode": K.LOG_ARGS[log][0], "rule": K.LOG_ARGS[log][1] or "timed", "arms": "", **K.PREREG_PARAMS}, **kw)  # noqa: E731
DONE = {"phase": "done", "mode": "x", "seconds": 1.0}
FULL = {"rung_a": syn_rung_a(), "rung3": syn_rung3(), "bidir_timed": syn_bidir("timed", **GOOD_B), "bidir_depress": syn_bidir("depress", **DEPR)}


CODE = K.code_hashes()
CLEAN = ("abc1234", {"rate": False, "learn": False})    # golden-only analyzer state: the metas below say HEAD abc1234, clean


class Env:
    """Temp results dir + fake chunk-1 cell dir + prereg file. write() lays down a batch; m() builds one meta row."""
    def __init__(self, td):
        self.out, self.fit, self.prereg = os.path.join(td, "out"), os.path.join(td, "fit"), os.path.join(td, "PREREG.md")
        os.makedirs(self.out)
        os.makedirs(os.path.join(self.fit, "cells"))
        with open(self.prereg, "w") as f:
            f.write("prereg text")
        self.psha = K.sha256_file(self.prereg)
        self.refp = os.path.join(td, "reference.json")
        with open(self.refp, "w") as f:
            f.write("reference text")
        self.rsha = K.sha256_file(self.refp)
        for s in K.FIT_SEEDS:
            with open(K.fit_cell_path(s, self.fit), "w") as f:
                f.write("cell %d" % s)
        self.metas = {}

    def gains(self, base):
        if base == "c0":
            return None
        s = int(base[3:])
        return {"file": "cells/x", "sha256": K.sha256_file(K.fit_cell_path(s, self.fit)), "cap": 16.0, "fit_seed": s, "n_gains": 9, "n_active": 3}

    def m(self, log, b, **kw):
        d = {"phase": "meta", "prereg_sha256": self.psha, "reference_sha256": self.rsha, "git_head": "abc1234", "git_dirty_rate": False,
             "git_dirty_learn": False, "code_sha256": CODE, "args": STD(log), "base": b, "gains": self.gains(b), "chunk_params_loaded": [], "overwrite": False}
        return dict(d, **kw)

    def write(self, rows=None, metas=None, no_done=(), shuffle=None, skip=()):
        """rows: {(log, base): rows} overrides of FULL; metas: {(log, base): meta row}; no_done / skip: {(log, base)}."""
        for b in K.BASES:
            for n in K.LOGS:
                if (n, b) in skip:
                    continue
                r = list((rows or {}).get((n, b), FULL[n]))
                if shuffle:
                    shuffle.shuffle(r)
                r = [(metas or {}).get((n, b), self.m(n, b))] + r + ([] if (n, b) in no_done else [DONE])
                with open(os.path.join(self.out, "%s_%s.jsonl" % (n, b)), "w") as f:
                    f.write(chr(10).join(json.dumps(x) for x in r) + chr(10))

    def analyze(self, **kw):
        return K.analyze(self.out, ref, prereg=self.prereg, fit_dir=self.fit, ref_path=self.refp, **dict({"state": CLEAN}, **kw))

    def no_output(self):
        return not os.path.exists(os.path.join(self.out, "result.json")) and not os.path.exists(os.path.join(self.out, "RESULT.md"))


def refuses(env, needle, **kw):
    env.write(**kw)
    try:
        env.analyze()
    except SystemExit as e:
        assert needle in str(e), (needle, str(e))
    else:
        raise AssertionError("analyze accepted a batch that must refuse: %s" % needle)
    assert env.no_output(), "a refused batch wrote result files"


# 24 files: ONE combined analyze over all 24 runs writes result.json + RESULT.md (c0 section, c1 section per-seed table + arm label)
with tempfile.TemporaryDirectory() as td:
    e = Env(td)
    e.write(rows={("rung_a", "c0"): syn_rung_a() + [HEAD("rung_a"), LATCH("rung_a", "baseline", False), LATCH("rung_a", "end", True),
                                                      {"arm": "learn", "phase": "guard", "song": "dc2", "active_frac_ratio": 1.2, "mbon06_ratio": 0.9,
                                                       "mbon06_mode": "ratio", "breach": False}]})
    out = e.analyze()
    md = open(os.path.join(e.out, "RESULT.md")).read()
    js = json.load(open(os.path.join(e.out, "result.json")))
    assert sorted(f for f in os.listdir(e.out) if not f.endswith(".jsonl")) == ["RESULT.md", "result.json"]    # no per-base overwrite
    assert md.index("## Rule") < md.index("## c0 (arm i)") < md.index("## c1 (arm ii") < md.index("### Per-seed labels") < md.index("### Numbers (c1s0)")
    assert js["c0"]["protocols"]["rung_a"]["label"] == "PASS" and js["rule"] in md and len(js["c0"]["end_latch"]) == 1
    assert js["c0"]["guard"]["rung_a/learn"]["n"] == 1 and "- rung_a: PASS" in md and "- bidir: PASS" in md and js["c0"]["arm"] == "(i)"
    assert all(ord(c) < 128 for c in md), "RESULT.md must be ASCII"
    assert js["provenance"]["prereg_sha256"] == e.psha and js["provenance"]["reference_sha256"] == e.rsha
    assert e.psha in md and e.rsha in md and len(js["provenance"]["meta"]) == 24 and "bidir_depress_c1s4" in js["provenance"]["meta"]
    assert isinstance(js["analyzer"]["git_head"], str) and js["analyzer"]["git_head"] and set(js["analyzer"]) == {"git_head", "git_dirty_rate", "git_dirty_learn"}
    assert js["c1"]["arm"] == "(ii)" and sorted(js["c1"]["seeds"]) == ["0", "1", "2", "3", "4"]
    assert {p: v["label"] for p, v in js["c1"]["arm_labels"].items()} == {p: "PASS" for p in K.PROTOCOLS}
    assert "| seed | rung_a | rung3 | bidir |" in md and "| 4 | PASS | PASS | PASS |" in md and "- rung_a: PASS (5 of 5 seeds)" in md
with tempfile.TemporaryDirectory() as td:
    e = Env(td)
    try:
        e.analyze()
        raise AssertionError("24 FAIL: empty dir was analysed")
    except SystemExit as ex:
        assert "missing or incomplete" in str(ex) and "rung_a_c0" in str(ex) and "bidir_depress_c1s4" in str(ex)
    assert e.no_output()
print("24 analyze: one result.json + RESULT.md over 24 runs (c0 section, c1 per-seed table + arm label), provenance, git, ASCII; empty dir refuses", flush=True)

# 25 realistic interleaved stream through analyze() with files: guard rows from unpaired training, headroom + latch
#    rows, rung 3 with all three probes, bidir with three songs; shuffled order; all 24 runs; must PASS all three, both arms.
GUARD = lambda arm: {"arm": arm, "phase": "guard", "song": "d", "active_frac_ratio": 1.1, "mbon06_ratio": None,  # noqa: E731
                     "mbon06_mode": "absolute", "mbon06_hz": 0.3, "breach": False}
ra = syn_rung_a() + [GUARD(a) for a in C.ODOUR_ARMS for _ in range(3)] + [HEAD("rung_a"), LATCH("rung_a", "baseline", False), LATCH("rung_a", "end", True)]
r3 = syn_rung3() + [dict(r, song=s, avoid_index=50.0) for r in syn_rung3() if "w_mean_frac" not in r for s in ("dc2", "da1")]
r3 += [GUARD("interfere")] * 4 + [HEAD("rung3"), HEAD("rung3"), LATCH("rung3", "baseline", False)]


def three(rule, kw):
    rows = syn_bidir(rule, **kw)
    rows += [dict(r, song=s, avoid_index=-9.0 - i) for r in syn_bidir(rule, **kw) for i, s in enumerate(("d", "da1"))]
    return rows + [GUARD("bidir_" + rule)] * 5 + [HEAD("bidir"), LATCH("bidir", "baseline", False), LATCH("bidir", "end", False)]


REAL = {"rung_a": ra, "rung3": r3, "bidir_timed": three("timed", GOOD_B), "bidir_depress": three("depress", DEPR)}
with tempfile.TemporaryDirectory() as td:
    e = Env(td)
    e.write(rows={(n, b): REAL[n] for b in K.BASES for n in K.LOGS}, shuffle=random.Random(0))
    out = e.analyze()
    want_all = {p: ("PASS", None) for p in K.PROTOCOLS}
    for b, blk in [("c0", out["c0"])] + [("c1s" + s, v) for s, v in out["c1"]["seeds"].items()]:
        assert {p: (v["label"], v["diagnostic"]) for p, v in blk["protocols"].items()} == want_all, (b, blk["protocols"])
        assert len(blk["end_latch"]) == 3 and blk["guard"]["rung3/interfere"]["n"] == 4 and not any(g["aborted"] for g in blk["guard"].values())
    assert {p: v["label"] for p, v in out["c1"]["arm_labels"].items()} == {p: "PASS" for p in K.PROTOCOLS} and out["c0"]["protocols"]["rung_a"]["label"] == "PASS"
print("25 analyze: realistic shuffled 24-run stream (guard/headroom/latch rows, 3 songs) -> PASS x3 for arm (i) and arm (ii)", flush=True)

# 26 F1 prereg gate: missing / empty file -> SystemExit (not an assert); present -> meta row with hashes, git, args
with tempfile.TemporaryDirectory() as td:
    pr = os.path.join(td, "PREREG.md")
    try:
        K.require_prereg(pr)
        raise AssertionError("26 FAIL: missing prereg accepted")
    except SystemExit as ex:
        assert "prereg" in str(ex)
    open(pr, "w").close()
    try:
        K.require_prereg(pr)
        raise AssertionError("26 FAIL: empty prereg accepted")
    except SystemExit as ex:
        assert "prereg" in str(ex)
    saved = K.PREREG
    K.PREREG = pr                                   # the default path is what main() uses
    try:
        K.require_prereg()
        raise AssertionError("26 FAIL: empty PREREG accepted")
    except SystemExit:
        pass
    finally:
        K.PREREG = saved
    with open(pr, "w") as f:
        f.write("prereg text")
    m = K.meta_row({"mode": "rung_a", "eta": 5e-6}, overwrite=False, prereg=pr)
    assert m["phase"] == "meta" and m["prereg_sha256"] == hashlib.sha256(b"prereg text").hexdigest()
    assert m["reference_sha256"] == K.sha256_file(K.REF_PATH) and m["args"]["eta"] == 5e-6 and m["base"] == "c0"
    assert m["chunk_params_loaded"] == [] and m["overwrite"] is False and isinstance(m["git_head"], str) and m["git_head"]
    assert set(("git_dirty_rate", "git_dirty_learn")) <= set(m)
    assert m["code_sha256"] == K.code_hashes() and set(m["code_sha256"]) == set(K.CODE_FILES) and all(len(v) == 64 for v in m["code_sha256"].values())
assert K.chunk_params_loaded() == [] and sim.r.regime == "eln8"
print("26 F1: prereg missing/empty -> SystemExit; meta row carries hashes, git, args, chunk params (none)", flush=True)

# 27 F1 mixed hashes -> analyze refuses (prereg, then reference), writes nothing
for k, key in (("prereg", "prereg_sha256"), ("refh", "reference_sha256")):
    with tempfile.TemporaryDirectory() as td:
        e = Env(td)
        refuses(e, "different " + key, metas={("rung3", "c1s2"): e.m("rung3", "c1s2", **{key: "q" * 64})})
print("27 F1: mixed prereg / reference hashes -> analyze refuses", flush=True)

# 28 F3 missing done row -> names the run, prints no label or number, writes nothing
with tempfile.TemporaryDirectory() as td:
    e = Env(td)
    e.write(no_done=(("bidir_depress", "c0"),), skip=(("rung3", "c1s3"),))
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        try:
            e.analyze()
            raise AssertionError("28 FAIL: partial batch analysed")
        except SystemExit as ex:
            msg = str(ex)
    assert msg.endswith("bidir_depress_c0, rung3_c1s3") and "rung_a" not in msg and "bidir_timed" not in msg, msg
    assert not any(w in msg + buf.getvalue() for w in ("PASS", "FAIL", "NEEDS", "0.")), msg
    assert e.no_output()
print("28 F3: missing / incomplete runs (any of the 24) -> analyze refuses naming them, no label, no files", flush=True)

# 29 F3 existing log is never silently replaced; overwrite does
with tempfile.TemporaryDirectory() as td:
    lp = os.path.join(td, K.log_name("bidir", "timed"))
    assert os.path.basename(lp) == "bidir_timed_c0.jsonl" and K.log_name("rung_a", "timed") == "rung_a_c0.jsonl"
    with K.open_log(lp) as f:
        f.write("keep me")
    try:
        K.open_log(lp)
        raise AssertionError("29 FAIL: existing log opened without overwrite")
    except SystemExit as e:
        assert "--overwrite" in str(e)
    assert open(lp).read() == "keep me"
    with K.open_log(lp, overwrite=True) as f:
        f.write("new")
    assert open(lp).read() == "new" and K.done_row("rung_a", 0.0)["phase"] == "done"
    assert K.meta_row({}, overwrite=True, prereg=__file__)["overwrite"] is True
print("29 F3: existing log refused without overwrite; overwrite recorded in meta; done row shape", flush=True)

# 30. Real C0 + chunk-1 g*=16 seed-0 gains: identity push of Plastic.real() leaves the run unchanged vs
#     gains-only. No learning number is read or saved; only the max |diff| between two identical-weight runs.
from rate import chunk1 as C1  # noqa: E402
from rate import chunk1_fit as F1  # noqa: E402
from rate import suite as S  # noqa: E402
cell = json.load(open(os.path.join(_HERE, "results", "rate_chunk1_fit_c0", "cells", "cap016.000_s0.json")))
St1 = C1.sets()
tp = S.owned_pools(rs.pool.cpu().numpy(), St1["ct"], F1.OWN)
tab = rs.set_edge_gains(tp, cell["cap"])
rs.edge_x = torch.as_tensor(cell["edge_x"], dtype=rs.dtype, device=rs.device)
assert len(cell["edge_x"]) == len(tab["n_edges"]), "chunk-1 edge_x does not match this group table"
assert rs._full2frozen.shape[0] == rs.W.nnz and (rs._full2frozen[P.offsets] >= 0).all(), "KC->MBON edges must not be gained"
gains_only = rs.run([(orn, C.ORN_HZ)], C.T_RUN)
assert gains_only.any()
w_full = rs.W.data.copy()
dpre = rs.data_pre.clone()
P.push(sim)
assert torch.equal(rs.data_pre, dpre) and np.array_equal(rs.W.data, w_full), "identity push changed weights"
pushed = rs.run([(orn, C.ORN_HZ)], C.T_RUN)
dd = float((pushed - gains_only).abs().max())
assert torch.equal(pushed, gains_only) or dd <= 1e-6, dd
print("30 real: C0 + g*=16 seed-0 gains, identity push == gains-only (max |diff| %.1e Hz, %d gains)" % (dd, len(cell["edge_x"])), flush=True)

# 31. Arm (ii) base: c1s0 builds, records its gains, and a no-plasticity Fox-driven probe reproduces the
#     chunk-1 recorded per-component rates of the same seed (proves the gains loaded right). These are chunk-1
#     reward-path rates, not chunk-2 learning numbers. Tolerance 5e-3 Hz: same build, same device, same code
#     path as chunk1_fit.fit_cell's final C1.measure; the cell stores float64 of those rates.
base0_name, base0_gains = K.BASE_NAME, K.GAINS
try:
    c1sim = K.base("c1", 0)
    assert K.BASE_NAME == "c1s0" and K.log_name("rung_a", "timed") == "rung_a_c1s0.jsonl"
    assert K.log_name("bidir", "timed") == "bidir_timed_c1s0.jsonl"
    cell0, path0 = K.load_fit_cell(0)
    assert K.GAINS["cap"] == 16.0 and K.GAINS["fit_seed"] == 0 and K.GAINS["sha256"] == K.sha256_file(path0)
    assert K.GAINS["n_gains"] == len(cell0["edge_x"]) and 0 < K.GAINS["n_active"] <= K.GAINS["n_gains"]
    m = K.meta_row({}, prereg=__file__)
    assert m["base"] == "c1s0" and m["gains"] == K.GAINS and m["chunk_params_loaded"] == []
    _, D = C1.measure(c1sim.r, St1)
    diff = max(abs(D["FOX"][c] - cell0["D_comp"]["FOX"][c]) for c in cell0["D_comp"]["FOX"])
    assert diff <= 5e-3, diff
    print("31 real: c1s0 gains reloaded, no-plasticity Fox probe == chunk-1 cell (max |diff| %.2e Hz, %d gains, %d active, R %.3f vs %.3f Hz)"
          % (diff, K.GAINS["n_gains"], K.GAINS["n_active"], D["FOX"]["R"], cell0["D_comp"]["FOX"]["R"]), flush=True)
    for bad, why in (((("c1", None)), "needs --fit-seed"), (("c1", 7), "bad seed"), (("c0", 0), "seed on c0")):
        try:
            K.set_base(*bad)
            raise AssertionError("set_base%r accepted (%s)" % (bad, why))
        except SystemExit:
            pass
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        os.makedirs(os.path.join(td, "cells"))
        json.dump(dict(cell0, fpass=False), open(K.fit_cell_path(0, td), "w"))
        try:
            K.load_fit_cell(0, td)
            raise AssertionError("an F-failing seed was accepted")
        except SystemExit:
            pass
    print("31b: c1 needs a valid fit seed, c0 refuses one, an F-failing seed is refused", flush=True)
finally:
    K.BASE_NAME, K.GAINS = base0_name, base0_gains

# 32. Normalised threshold (pure labeller). Effects are divided by the brain's own baseline approach-MBON rate:
#     an absolute shift that passed the old rule fails when the baseline is 2.35x paper 0's, and the reverse.
#     (old rule: shift >= 0.5 x A_MEAN; new: shift / baseline >= 0.5 x A_MEAN / paper-0 baseline)
scaled = lambda app, **kw: dict(rung_a=syn_rung_a(app=app, rev_d=app * NAIVE, **kw), rung3=syn_rung3(d=app * NAIVE * r3ref))  # noqa: E731  (rung 3 kept passing)
want(batch(**scaled(2.35, learn=0.6 * A_MEAN)), rung_a=("FAIL", "NEEDS-SPIKING"))   # old PASS, normalised 0.26x -> FAIL
want(batch(**scaled(0.5, learn=0.4 * A_MEAN)))                                       # old FAIL, normalised 0.8x -> PASS
want(batch(**scaled(2.35, learn=1.2 * A_MEAN)))                                      # normalised 0.51x -> PASS
want(batch(**scaled(2.35, learn=1.17 * A_MEAN)), rung_a=("FAIL", "NEEDS-SPIKING"))   # normalised 0.498x -> FAIL
want(batch(rung_a=syn_rung_a(learn=0.5 * A_MEAN, app=1.0)))                                       # exact boundary is inclusive
nr = batch(**scaled(2.35, learn=1.2 * A_MEAN))["rung_a"]["numbers"]
assert abs(nr["norm_approach_drop"] - 1.2 * A_MEAN / (2.35 * APP_DC2)) < 1e-12 and abs(nr["threshold_norm"] - 0.5 * A_MEAN / APP_DC2) < 1e-12, nr
assert nr["approach_drop_hz"] == 1.2 * A_MEAN and abs(nr["dc2_avoid_index_shift"] - 1.2 * A_MEAN) < 1e-9   # total shift and avoid component reported
assert nr["dc2_avoid_component_shift_hz"] == 0.0
# rung 3 naive floor, normalised the same way (naive D shift / baseline approach D)
want(batch(rung_a=syn_rung_a(learn=2.35 * A_MEAN, rev_d=0.6 * NAIVE, app=2.35), rung3=syn_rung3(d=0.6 * NAIVE * r3ref)),
     rung3=("FAIL", "NEEDS-SPIKING"))                                                             # old floor ok, normalised 0.26x -> FAIL
want(batch(rung_a=syn_rung_a(learn=2.35 * A_MEAN, rev_d=1.2 * NAIVE, app=2.35), rung3=syn_rung3(d=1.2 * NAIVE * r3ref)))
want(batch(rung_a=syn_rung_a(learn=0.5 * A_MEAN, rev_d=0.4 * NAIVE, app=0.5), rung3=syn_rung3(d=0.4 * NAIVE * r3ref)))   # old floor fails, normalised 0.8x passes
# bidir: F_1 floor normalised with rung A's paper-0 effect
hi = dict(GOOD_B, rises=[1.2 * A_MEAN] * 4, drops=[1.2 * A_MEAN] * 4)
want(batch(bt=syn_bidir("timed", app=2.35, **hi)))
want(batch(bt=syn_bidir("timed", app=2.35, **GOOD_B)), bidir=("FAIL", "NEEDS-SPIKING"))           # rise A_MEAN / 2.35 baseline = 0.43x
lowb = dict(GOOD_B, rises=[0.4 * A_MEAN] * 4, drops=[0.4 * A_MEAN] * 4)
want(batch(bt=syn_bidir("timed", app=0.5, **lowb)))                                               # 0.4 absolute fails old floor; normalised 0.8x passes
# no baseline readout / zero baseline cannot pass or be scored
ZERO_L = lambda rows: [dict(r, mbon_approach_hz=0.0) if r.get("arm") == "learn" and r.get("song") == "dc2" and "mbon_approach_hz" in r else r for r in rows]  # noqa: E731
want(batch(rung_a=ZERO_L(syn_rung_a())), rung_a=("FAIL", "INCOMPLETE-DATA"))
want(batch(rung_a=ZERO_L(syn_rung_a()) + [HEAD("rung_a", False)]),
     rung_a=("FAIL", "READOUT-CEILING"))
# headroom gate: only a nonzero, unsaturated baseline (the absolute-rate threshold is gone)
assert K.headroom_row("rung_a", "dc2", fake(0.01), ref, RM)["gate_pass"] and "threshold_hz" not in K.headroom_row("rung_a", "dc2", fake(5.0), ref, RM)
print("32 normalised threshold: absolute-pass/normalised-fail and reverse, inclusive boundary, rung 3 naive floor, bidir F_1, zero baseline", flush=True)

# 33. Arm (ii) rule: PASS iff >= ceil(0.8 n) seeds PASS the protocol (4 of 5), else FAIL with per-seed labels listed.
assert [K.arm_need(n) for n in (3, 4, 5, 10)] == [3, 4, 4, 8]
ok_seed, bad_seed = batch(), batch(rung_a=syn_rung_a(learn=0.4 * A_MEAN))
a4 = K.arm_labels({s: (bad_seed if s == 2 else ok_seed) for s in K.FIT_SEEDS})["rung_a"]
assert a4["label"] == "PASS" and a4["n_pass"] == 4 and a4["need"] == 4 and a4["passing_seeds"] == [0, 1, 3, 4], a4
a3 = K.arm_labels({s: (bad_seed if s in (1, 2) else ok_seed) for s in K.FIT_SEEDS})
assert a3["rung_a"]["label"] == "FAIL" and a3["rung_a"]["n_pass"] == 3 and a3["rung3"]["label"] == "PASS" and a3["bidir"]["label"] == "PASS"
assert a3["rung_a"]["per_seed"][1] == {"label": "FAIL", "diagnostic": "NEEDS-SPIKING"} and a3["rung_a"]["per_seed"][0]["label"] == "PASS"
with tempfile.TemporaryDirectory() as td:
    e = Env(td)
    sick = lambda seeds: {("rung_a", "c1s%d" % s): syn_rung_a(learn=0.4 * A_MEAN) for s in seeds}  # noqa: E731
    e.write(rows=sick((3,)))
    o = e.analyze()
    assert o["c1"]["arm_labels"]["rung_a"]["label"] == "PASS" and o["c1"]["arm_labels"]["rung_a"]["n_pass"] == 4
    assert o["c0"]["protocols"]["rung_a"]["label"] == "PASS" and o["c1"]["seeds"]["3"]["protocols"]["rung_a"]["label"] == "FAIL"
    e.write(rows=sick((1, 3, 4)))
    o = e.analyze(reanalyze=True)
    md = open(os.path.join(e.out, "RESULT.md")).read()
    assert o["c1"]["arm_labels"]["rung_a"]["label"] == "FAIL" and o["c1"]["arm_labels"]["rung3"]["label"] == "PASS"
    assert "- rung_a: FAIL (2 of 5 seeds)" in md and "| 3 | FAIL (NEEDS-SPIKING) | PASS | PASS |" in md and o["c0"]["protocols"]["rung_a"]["label"] == "PASS"
print("33 arm (ii): 4/5 seeds PASS -> PASS, 3/5 -> FAIL with per-seed diagnostics; arm_need generalises (ceil 0.8 n)", flush=True)

# 34. Prereg on disk must be the one the logs were run under.
with tempfile.TemporaryDirectory() as td:
    e = Env(td)
    e.write()
    with open(e.prereg, "w") as f:
        f.write("prereg text, edited after the batch")
    try:
        e.analyze()
        raise AssertionError("34 FAIL: edited prereg accepted")
    except SystemExit as ex:
        assert "prereg on disk" in str(ex), str(ex)
    assert e.no_output()
    os.remove(e.prereg)
    try:
        e.analyze()
        raise AssertionError("34 FAIL: missing prereg accepted")
    except SystemExit as ex:
        assert "prereg" in str(ex)
print("34 provenance: prereg edited / deleted after the batch -> analyze refuses", flush=True)

# 35. A strict subset of the default arms (rung A, rung 3) refuses; the explicit full set and "" are fine.
with tempfile.TemporaryDirectory() as td:
    e = Env(td)
    for log, arms in (("rung_a", "learn,lesion"), ("rung3", "interfere"), ("rung_a", "learn,bogus")):
        refuses(e, "prereg parameters", metas={(log, "c1s1"): e.m(log, "c1s1", args=STD(log, arms=arms))})
    full_arms = {"rung_a": ",".join(C.ODOUR_ARMS), "rung3": ",".join(K.persist.ARMS)}
    e.write(metas={(n, "c1s1"): e.m(n, "c1s1", args=STD(n, arms=full_arms[n])) for n in ("rung_a", "rung3")})
    e.analyze()
print("35 provenance: --arms subset refuses, explicit full arm set accepted", flush=True)

# 36. Run parameters that differ from the prereg constants refuse (eta, lam, train, probe, relax), as does a wrong mode / rule.
with tempfile.TemporaryDirectory() as td:
    e = Env(td)
    for k, v in (("eta", 1e-5), ("lam", 0.02), ("train", 10), ("probe", 3), ("relax", 5)):
        refuses(e, "prereg parameters", metas={("bidir_timed", "c0"): e.m("bidir_timed", "c0", args=STD("bidir_timed", **{k: v}))})
    refuses(e, "prereg parameters", metas={("rung_a", "c0"): e.m("rung_a", "c0", args=dict(STD("rung_a"), eta=None))})
    refuses(e, "prereg parameters", metas={("rung_a", "c0"): e.m("rung_a", "c0", args=STD("rung_a", mode="rung3"))})
    refuses(e, "prereg parameters", metas={("bidir_timed", "c0"): e.m("bidir_timed", "c0", args=STD("bidir_timed", rule="depress"))})
    refuses(e, "prereg parameters", metas={("rung_a", "c0"): e.m("rung_a", "c0", args={})})
print("36 provenance: eta / lam / train / probe / relax override, wrong mode or rule, missing args -> refuses", flush=True)

# 37. Base and gains provenance: meta base = file base; one gains sha per c1 seed, equal to the cell on disk; c0 has none.
with tempfile.TemporaryDirectory() as td:
    e = Env(td)
    refuses(e, "base does not match", metas={("rung3", "c1s2"): e.m("rung3", "c1s2", base="c1s1")})
    g = dict(e.gains("c1s2"), sha256="f" * 64)
    refuses(e, "one gains sha256", metas={("bidir_depress", "c1s2"): e.m("bidir_depress", "c1s2", gains=g)})          # mixed across the 4 logs
    refuses(e, "one gains sha256", metas={("rung_a", "c1s2"): e.m("rung_a", "c1s2", gains=None)})
    refuses(e, "one gains sha256", metas={("rung_a", "c1s2"): e.m("rung_a", "c1s2", gains=dict(e.gains("c1s2"), fit_seed=3))})
    refuses(e, "carry gains", metas={("rung_a", "c0"): e.m("rung_a", "c0", gains=e.gains("c1s0"))})
    e.write()                                                                                                          # all four agree, but the cell changes
    with open(K.fit_cell_path(2, e.fit), "w") as f:
        f.write("regenerated cell")
    try:
        e.analyze()
        raise AssertionError("37 FAIL: regenerated cell accepted")
    except SystemExit as ex:
        assert "not the chunk-1 cell on disk" in str(ex), str(ex)
    assert e.no_output()
print("37 provenance: meta base, mixed / missing / wrong-seed gains sha, c0 with gains, cell regenerated on disk -> refuses", flush=True)

# 38. base() and the seed check read the same fit_dir (was: seed check read the default dir, cell load read fit_dir).
with tempfile.TemporaryDirectory() as td:
    os.makedirs(os.path.join(td, "cells"))
    cell_ok, _ = K.load_fit_cell(0)
    json.dump(dict(cell_ok, fpass=False), open(K.fit_cell_path(0, td), "w"))
    b0, g0 = K.BASE_NAME, K.GAINS
    try:
        for fn in (lambda: K.set_base("c1", 0, td), lambda: K.base("c1", 0, fit_dir=td)):
            try:
                fn()
                raise AssertionError("38 FAIL: fit_dir not used by the seed check")
            except SystemExit as ex:
                assert "did not pass" in str(ex), str(ex)
        K.set_base("c1", 0)                                                                                           # default dir still fine
        assert K.BASE_NAME == "c1s0"
    finally:
        K.BASE_NAME, K.GAINS = b0, g0
print("38 fit_dir: set_base / base refuse an F-failing seed found only in the given fit_dir", flush=True)


# 39. Approach-component gates (fix round 1). Plasticity acts on approach synapses, the rate model's avoid baseline is 10-25x paper 0's:
#     an avoid-only rise must not pass; an approach drop must.
AV0 = 26.496 / APP_DC2                                              # the reviewer's case: c0 baseline approach 26.5 Hz, avoid 5.27 -> 18.26 Hz
res = batch(**{"rung_a": syn_rung_a(learn=0.0, app=AV0, avoid_rise=18.26 - 5.27)})
assert res["rung_a"]["label"] == "FAIL" and res["rung_a"]["diagnostic"] == "NEEDS-SPIKING", res["rung_a"]
assert res["rung_a"]["numbers"]["dc2_avoid_index_shift"] > 12.9 and res["rung_a"]["numbers"]["approach_drop_hz"] == 0.0     # the old rule's shift would have passed
want(batch(rung_a=syn_rung_a(learn=0.0, avoid_rise=A_MEAN)), rung_a=("FAIL", "NEEDS-SPIKING"))                              # avoid-only at paper-0 scale
want(batch(rung_a=syn_rung_a(learn=A_MEAN, avoid_rise=-0.3)))                                                               # approach drop passes; avoid fall is reported only
want(batch(rung_a=syn_rung_a(learn=0.3 * A_MEAN, avoid_rise=A_MEAN)), rung_a=("FAIL", "NEEDS-SPIKING"))                      # mixed: big avoid rise cannot carry a small drop
want(batch(rung_a=syn_rung_a(learn=0.5 * A_MEAN, avoid_rise=-3.0)))                                                         # mirror: an avoid fall cannot sink a real approach drop
want(batch(rung_a=syn_rung_a(learn=A_MEAN, lesion=A_MEAN)), rung_a=("FAIL", "NEEDS-SPIKING"))                               # controls compared on the normalised approach drop
# rung 3 / bidir ignore avoid-only moves as well
r3_avoid = [dict(r, avoid_index=r["avoid_index"] + 30.0) if r.get("phase") == "trainC_test" else r for r in syn_rung3(d=0.0)]
want(batch(rung3=r3_avoid), rung3=("FAIL", "NEEDS-SPIKING"))
bid_avoid = [dict(r, avoid_index=r["avoid_index"] + 30.0) if r["after"] == "F1" else r for r in syn_bidir("timed", rises=[0.0] * 4, drops=[0.0] * 4)]
want(batch(bt=bid_avoid), bidir=("FAIL", "NEEDS-SPIKING"))
n3 = batch()["rung3"]["numbers"]
assert n3["trained_d_approach_drop_hz"] > 0 and "trained_d_avoid_index_shift" in n3 and "trained_d_avoid_component_shift_hz" in n3
nb = batch()["bidir"]["numbers"]
assert isinstance(nb["rise_1"], float) and "rise_1_avoid_index" in nb and "rise_1_avoid_component_hz" in nb
print("39 approach gates: avoid-only rise (5.27 -> 18.26 Hz) FAILS rung A / rung 3 / bidir, approach drop passes, avoid component reported", flush=True)

# 40. Provenance: reference.json on disk, mixed git HEAD, seed0, duplicate arms, stale result files.
with tempfile.TemporaryDirectory() as td:
    e = Env(td)
    e.write()
    with open(e.refp, "w") as f:
        f.write("reference text, regenerated after the batch")
    try:
        e.analyze()
        raise AssertionError("40 FAIL: regenerated reference accepted")
    except SystemExit as ex:
        assert "reference.json on disk" in str(ex), str(ex)
    assert e.no_output()
    with open(e.refp, "w") as f:
        f.write("reference text")
    e.analyze()
    assert not e.no_output()
    with open(e.prereg, "w") as f:                                   # a refusal (even with --reanalyze) leaves the earlier result untouched
        f.write("changed")
    before = K.sha256_file(os.path.join(e.out, "result.json"))
    try:
        e.analyze(reanalyze=True)
        raise AssertionError("40 FAIL: edited prereg accepted")
    except SystemExit:
        pass
    assert K.sha256_file(os.path.join(e.out, "result.json")) == before and not [f for f in os.listdir(e.out) if f.endswith(".tmp")]
with tempfile.TemporaryDirectory() as td:
    e = Env(td)
    refuses(e, "different git_head", metas={("rung3", "c1s4"): e.m("rung3", "c1s4", git_head="def5678")})
    refuses(e, "prereg parameters", metas={("rung_a", "c0"): e.m("rung_a", "c0", args=STD("rung_a", seed0=1))})
    dup = ",".join(list(C.ODOUR_ARMS) + [list(C.ODOUR_ARMS)[0]])
    refuses(e, "prereg parameters", metas={("rung_a", "c0"): e.m("rung_a", "c0", args=STD("rung_a", arms=dup))})
    e.write(metas={("rung_a", "c0"): e.m("rung_a", "c0", args=STD("rung_a", arms=",".join(sorted(C.ODOUR_ARMS))))})     # order does not matter
    e.analyze()
assert K.PREREG_PARAMS["seed0"] == 0
print("40 provenance: reference.json on disk, mixed git HEAD, seed0, duplicate arms, refusal leaves the earlier result alone", flush=True)


# 41. Final review I1: rule constants bound to the runs (one known HEAD = analyzer HEAD, clean, code hashes), I2: no silent reanalysis.
def refuses_kw(env, needle, **kw):
    try:
        env.analyze(**kw)
    except SystemExit as ex:
        assert needle in str(ex), (needle, str(ex))
    else:
        raise AssertionError("analyze accepted: %s" % needle)
    assert env.no_output(), "a refused batch wrote result files"


with tempfile.TemporaryDirectory() as td:
    e = Env(td)
    refuses(e, "different git_head", metas={("rung_a", "c0"): e.m("rung_a", "c0", git_head="def5678")})                       # one run differs
    for bad in ("unknown", "", None):
        refuses(e, "unknown git_head", metas={(n, b): e.m(n, b, git_head=bad) for b in K.BASES for n in K.LOGS})               # all 24 agree but unreadable
    refuses(e, "unknown git_head", metas={(n, b): {k: v for k, v in e.m(n, b).items() if k != "git_head"} for b in K.BASES for n in K.LOGS})
    e.write()
    refuses_kw(e, "analyzer HEAD", state=("fff9999", CLEAN[1]))                                                            # analyzer elsewhere
    refuses_kw(e, "analyzer (rate)", state=(CLEAN[0], {"rate": True, "learn": False}))                                     # analyzer dirty
    refuses_kw(e, "analyzer (learn)", state=(CLEAN[0], {"rate": False, "learn": None}))                                    # git unreadable
    refuses(e, "rung3_c1s1 (git_dirty_learn)", metas={("rung3", "c1s1"): e.m("rung3", "c1s1", git_dirty_learn=True)})
    refuses(e, "rung_a_c0 (git_dirty_rate)", metas={("rung_a", "c0"): e.m("rung_a", "c0", git_dirty_rate=True)})
    refuses(e, "git_dirty_rate", metas={("rung_a", "c0"): {k: v for k, v in e.m("rung_a", "c0").items() if k != "git_dirty_rate"}})
    refuses(e, "source changed", metas={("bidir_timed", "c1s3"): e.m("bidir_timed", "c1s3", code_sha256=dict(CODE, **{"learn/plastic.py": "0" * 64}))})
    refuses(e, "no complete code_sha256", metas={("rung_a", "c0"): {k: v for k, v in e.m("rung_a", "c0").items() if k != "code_sha256"}})
    refuses(e, "no complete code_sha256", metas={("rung_a", "c0"): e.m("rung_a", "c0", code_sha256={"rate/chunk2.py": CODE["rate/chunk2.py"]})})
    e.write()
    refuses_kw(e, "source changed", code_sha=dict(CODE, **{"rate/chunk2.py": "1" * 64}))                                   # a rule constant edited after the runs
    e.analyze()                                                                                                            # clean batch accepted
    # I2: a second analysis refuses unless reanalyze, and is then recorded
    for keep in ("result.json", "RESULT.md"):
        hold = os.path.join(e.out, keep)
        stash = open(hold, "rb").read()
        other = os.path.join(e.out, "RESULT.md" if keep == "result.json" else "result.json")
        os.remove(other)
        try:
            e.analyze()
            raise AssertionError("41 FAIL: existing %s did not refuse" % keep)
        except SystemExit as ex:
            assert "already holds a result" in str(ex) and "--reanalyze" in str(ex)
        assert open(hold, "rb").read() == stash and not os.path.exists(other)
        e.analyze(reanalyze=True)                                                                                          # restores the pair
    e.analyze(reanalyze=True)
    prev = K.sha256_file(os.path.join(e.out, "result.json"))
    o = e.analyze(reanalyze=True)
    js = json.load(open(os.path.join(e.out, "result.json")))
    md = open(os.path.join(e.out, "RESULT.md")).read()
    assert js["reanalyzed"] is True and js["previous_result_sha256"] == prev and o["previous_result_sha256"] == prev
    assert "REANALYZED: true; previous result.json sha256 " + prev in md
    assert K.sha256_file(os.path.join(e.out, "result.json")) != prev
    assert not [f for f in os.listdir(e.out) if f.endswith(".tmp")]
with tempfile.TemporaryDirectory() as td:
    e = Env(td)
    e.write()
    e.analyze()
    assert json.load(open(os.path.join(e.out, "result.json")))["reanalyzed"] is False
    assert json.load(open(os.path.join(e.out, "result.json")))["previous_result_sha256"] is None
print("41 final review I1/I2: mixed / unknown HEAD, dirty run or analyzer, code_sha mismatch refuse; existing result refuses without --reanalyze, records previous sha with it", flush=True)

# 42-46. LIF arm for bidir (prereg PREREGISTER_rate_chunk2_bidir_ref.md). Plumbing only: no learning number is printed or read.
from itertools import product  # noqa: E402
tick_seeds = {(s_, b_, t_): K.lif_tick_seed(s_, b_, t_) for s_, b_, t_ in product(K.LIF_SEEDS, range(8), range(200))}
assert len(set(tick_seeds.values())) == len(tick_seeds) and 0 not in tick_seeds.values() and max(tick_seeds.values()) < 2 ** 31
assert tick_seeds[(0, 0, 0)] != tick_seeds[(1, 0, 0)] and tick_seeds[(0, 0, 0)] == 1001000 and tick_seeds[(4, 7, 199)] == 5008199
print("42 lif: tick seeds distinct over 5 seeds x 8 blocks x 200 ticks, never 0 (the probe seed), formula 1000000*(S+1)+1000*(block+1)+tick", flush=True)

# 43. Refusals + meta row (cheap, before the sim build): lif is bidir-only, needs a seed, carries its own prereg sha / base / regime row.
for argv, why in ((["--arm", "lif"], "no --lif-seed"), (["--arm", "lif", "--lif-seed", "5"], "seed 5"),
                  (["--arm", "lif", "--lif-seed", "0", "--mode", "rung_a"], "rung_a"), (["--arm", "lif", "--lif-seed", "0", "--mode", "rung3"], "rung3"),
                  (["--arm", "lif", "--lif-seed", "0", "--gate"], "gate"), (["--arm", "c0", "--lif-seed", "0"], "lif-seed on c0")):
    sys.argv = ["chunk2.py"] + argv
    try:
        K.main()
        raise AssertionError("43 FAIL: %s was accepted" % why)
    except SystemExit as ex:
        assert not isinstance(ex.code, int), (why, ex.code)    # a message, not a clean exit
K.set_base("lif", lif_seed=3)
assert K.BASE_NAME == "lifs3" and K.log_name("bidir", "timed") == "bidir_timed_lifs3.jsonl" and K.OUT_REF.endswith("rate_chunk2_bidir_ref")
m43 = K.meta_row({"arm": "lif", "lif_seed": 3}, prereg=K.PREREG_REF, extra={"regime": "eln8"})
assert m43["prereg_sha256"] == K.sha256_file(K.PREREG_REF) and m43["base"] == "lifs3" and m43["lif"] == {"regime": "eln8"}
K.set_base("c0")
assert "lif" not in K.meta_row({"arm": "c0"}) and K.BASE_NAME == "c0"
print("43 lif: refuses non-bidir / gate / bad seed; meta row carries the new prereg sha, base lifs{S}, lif block; rate meta row unchanged", flush=True)

# 44-46. Real LIF build (paper 0's eln8, floor 5), one process, guard on, probes seed 0.
import time as _time  # noqa: E402
lif = K.base_lif()
assert lif.net.min_syn == 5 and lif.pn_kc_gain == 8.0 and isinstance(lif, K._GuardMixin) and not hasattr(lif, "r")
assert K.sim_r_max(lif) == 1000.0 / 2.2
rec = []
_rb = lif.run_batch


def _rec(drives, seeds, **kw):
    out = _rb(drives, seeds, **kw)
    counts = out[0] if kw.get("return_state") else out
    rec.append((list(seeds), counts[0].cpu().numpy().copy()))
    return out


lif.run_batch = _rec


def f_block(s_, n):
    """One F block of n ticks from baseline weights and a fresh Plastic; returns (per-tick (seeds, counts), final dw)."""
    del rec[:]
    P_ = PL.Plastic.real()
    P_.push(lif)
    K._block(lif, P_, n, "paired", "timed", K.ETA_BIDIR, K.EVERY, "none", lambda t: K.lif_tick_seed(s_, 0, t))
    return list(rec), P_.dw.copy()


# 44. Determinism (prereg precondition): seed 0, one full F block (200 ticks, 7 US), run twice -> identical counts and weights.
t44 = _time.time()
r1, dw1 = f_block(0, K.F_TICKS)
t44 = _time.time() - t44
r2, dw2 = f_block(0, K.F_TICKS)
assert len(r1) == len(r2) == K.F_TICKS
assert [x[0] for x in r1] == [[K.lif_tick_seed(0, 0, t)] for t in range(K.F_TICKS)], "ticks did not carry their own seed"
assert all(np.array_equal(a[1], b[1]) for a, b in zip(r1, r2)) and np.array_equal(dw1, dw2)
assert sum(int(x[1].sum()) for x in r1) > 0, "vacuous: no spikes"
print("44 lif: seed 0, one F block (200 ticks) twice -> per-tick counts and final dw identical (%d ticks, %d neurons; one block %.1f s wall)"
      % (len(r1), r1[0][1].shape[0], t44), flush=True)

# 45. A different S gives different tick seeds AND different noise (first 20 ticks, no US before tick 25).
ra, _ = f_block(0, 20)
rb, _ = f_block(1, 20)
assert [x[0] for x in ra] != [x[0] for x in rb] and not all(np.array_equal(a[1], b[1]) for a, b in zip(ra, rb))
print("45 lif: S=0 vs S=1 -> different tick seeds and different spike counts", flush=True)

# 46. Guard + bidir schedule on the LIF, lesion on (weights must not move): probes at seed 0 are sighted at T0 and every later tag
#     (ratio exactly 1.0 with no plasticity), the science path writes headroom/latch rows with the LIF r_max, no abort.
lif.run_batch = _rb
lg46 = io.StringIO()
res46 = K.guarded(lif, "bidir_timed", C.STIM_SETS["odour"], lg46,
                  lambda: K.bidir_mb(lif, "timed", lg46, cycles=1, f_ticks=30, b_ticks=30, lesion="all", lif_seed=0))
assert res46 is not None and not res46[0].dw.any()
g46 = [json.loads(x) for x in lg46.getvalue().splitlines() if '"guard"' in x]
assert len(g46) == 9 and all(not x["breach"] and x["active_frac_ratio"] == 1.0 for x in g46), g46
hr46 = K.headroom(lif, K.load_reference(), "bidir", io.StringIO())
assert hr46[0]["r_max_hz"] == 1000.0 / 2.2 and len(K.latch_check(lif, io.StringIO(), "bidir", "baseline", "baseline")) == 1
print("46 lif: guard works on GpuSim (%d guard rows, ratio exactly 1.0, no abort), lesion leaves dw zero, headroom + latch run with r_max 454.5 Hz" % len(g46), flush=True)
del lif
torch.cuda.empty_cache()

# 47-59. --analyze-bidir-ref (prereg PREREGISTER_rate_chunk2_bidir_ref.md): synthetic rows in temp dirs, no sim, no real learning number.
W_RUNG, SLOPE = 0.02, 30.0     # synthetic rung A: |w| at trial t = W_RUNG t/30, normalised DC2 approach drop = SLOPE |w| (so drop(|w|=0.01) = 0.3)
LIF_WANT = {"regime": "eln8", "eln_negate": True, "pn_kc_gain": 8.0, "min_syn": 5, "w_syn": 0.275, "brain": "data/brain_gpu.npz",
            "tick_seed_formula": K.LIF_TICK_SEED_FORMULA, "probe_seed": 0}


def pair(f1, ok=True, w=0.01):
    """(timed, depress) synthetic bidir rows with rise_1_norm = f1; ok = the timed rule relearns in all 3 scorable cycles."""
    r1 = f1 * APP_DC2
    t = syn_bidir("timed", [r1] * 4 if ok else [r1] + [0.1 * r1] * 3, [r1] * 4 if ok else [0.2 * r1] * 4, w=w)
    d = syn_bidir("depress", [r1] + [0.1 * r1] * 3, [0.2 * r1] * 4, w=w)
    return t, d


def rung_rows():
    rows = [{"arm": "learn", "phase": "baseline", "trial": 0, "song": "dc2", "mbon_approach_hz": APP_DC2}]
    for t in (5, 10, 15, 20, 25, 30):
        w = -W_RUNG * t / 30
        rows.append({"arm": "learn", "phase": "test", "trial": t, "song": "dc2", "mbon_approach_hz": APP_DC2 * (1 - SLOPE * abs(w))})
        rows.append({"arm": "learn", "phase": "train", "trial": t, "song": "dc2", "w_mean_frac": w})
    return rows


class RefEnv:
    """Temp dirs: out (LIF logs), rate (12 bidir + 6 rung_a logs + EVALUATION.md), src (paper-0 rows), prereg, reference file."""
    def __init__(self, td):
        self.out, self.rate, self.src = os.path.join(td, "out"), os.path.join(td, "rate"), os.path.join(td, "src")
        os.makedirs(self.out)
        os.makedirs(self.rate)
        os.makedirs(os.path.join(self.src, "results"))
        self.prereg, self.refp, self.eval = os.path.join(td, "PREREG.md"), os.path.join(td, "reference.json"), os.path.join(self.rate, "EVALUATION.md")
        for p, txt in ((self.prereg, "ref prereg"), (self.refp, "ref text")):
            with open(p, "w") as f:
                f.write(txt)
        self.psha, self.rsha = K.sha256_file(self.prereg), K.sha256_file(self.refp)
        self.ref = {"sources": {}}

    def lif_meta(self, rule, s):
        return {"phase": "meta", "prereg_sha256": self.psha, "reference_sha256": self.rsha, "git_head": "abc1234", "git_dirty_rate": False,
                "git_dirty_learn": False, "code_sha256": CODE, "data_sha256": {"data/brain_gpu.npz": "h"},
                "args": dict({"mode": "bidir", "rule": rule, "arm": "lif", "lif_seed": s, "arms": ""}, **K.PREREG_PARAMS),
                "base": "lifs%d" % s, "gains": None, "chunk_params_loaded": [], "overwrite": False, "lif": dict(LIF_WANT)}

    def put(self, path, rows):
        with open(path, "w") as f:
            f.write(chr(10).join(json.dumps(x) for x in rows) + chr(10))

    def write(self, lif=None, rate=None, no_done=(), extra=None, rate_extra=None):
        """lif: {seed: (f1, ok)} default all (0.3, True); rate: {base: (f1, ok)} default all (0.2, True); extra: {(rule, seed): [rows]} appended."""
        lif, rate, extra, rate_extra = lif or {}, rate or {}, extra or {}, rate_extra or {}
        for s in K.LIF_SEEDS:
            t, d = pair(*lif.get(s, (0.3, True)))
            for rule, rows in (("timed", t), ("depress", d)):
                tail = [] if (rule, s) in no_done else [DONE]
                self.put(os.path.join(self.out, K.lif_log_name(rule, s)), [self.lif_meta(rule, s)] + rows + extra.get((rule, s), []) + tail)
        listed = []
        for b in K.BASES:
            t, d = pair(*rate.get(b, (0.2, True)))
            for rule, rows in (("timed", t), ("depress", d)):
                n = "bidir_%s_%s.jsonl" % (rule, b)
                self.put(os.path.join(self.rate, n), [{"phase": "meta", "git_head": "r999999", "prereg_sha256": "p", "base": b,
                                                       "args": {"mode": "bidir", "rule": rule}}] + rows + rate_extra.get((rule, b), []) + [DONE])
                listed.append(n)
            self.put(os.path.join(self.rate, "rung_a_%s.jsonl" % b), rung_rows())
            listed.append("rung_a_%s.jsonl" % b)
        with open(self.eval, "w") as f:
            f.write("# eval" + chr(10) + chr(10) + chr(10).join("- %s %s" % (n, K.sha256_file(os.path.join(self.rate, n))) for n in listed) + chr(10))
        for s in K.LIF_SEEDS:
            p = os.path.join(self.src, "results", "condition_o1s%d.jsonl" % s)
            self.put(p, rung_rows())
            self.ref["sources"]["results/condition_o1s%d.jsonl" % s] = K.sha256_file(p)

    def rewrite_meta(self, rule, s, fn):
        p = os.path.join(self.out, K.lif_log_name(rule, s))
        rows = [json.loads(x) for x in open(p)]
        fn(rows[0])
        self.put(p, rows)

    def analyze(self, **kw):
        return K.analyze_bidir_ref(self.out, self.ref, rate_dir=self.rate, eval_md=self.eval, src_root=self.src, prereg=self.prereg,
                                   ref_path=self.refp, **dict({"state": CLEAN}, **kw))

    def no_output(self):
        return not any(os.path.exists(os.path.join(self.out, f)) for f in ("result.json", "RESULT.md"))


def run_ref(lif=None, rate=None, **kw):
    with tempfile.TemporaryDirectory() as td:
        e = RefEnv(td)
        e.write(lif, rate, **kw)
        return e.analyze()


def refuses_ref(needle, mutate=None, **kw):
    with tempfile.TemporaryDirectory() as td:
        e = RefEnv(td)
        e.write(**kw)
        if mutate:
            mutate(e)
        try:
            e.analyze()
        except SystemExit as ex:
            assert needle in str(ex), (needle, str(ex))
        else:
            raise AssertionError("analyze-bidir-ref accepted a batch that must refuse: %s" % needle)
        assert e.no_output(), "a refused batch wrote result files"


def near(a, b, tol=1e-9):
    return a is not None and abs(a - b) < tol


def append_to(path, text):
    with open(path, "a") as f:
        f.write(text)


# 47. LIF sanity rule: USABLE iff rise_1_norm > 0 in >= 4 of 5 seeds AND the timed rule relearns in >= 4 of 5; R, SD, n same sign reported.
r = run_ref(lif={0: (0.2, True), 1: (0.3, True), 2: (0.4, True), 3: (0.3, True), 4: (0.3, True)})
L = r["lif_reference"]
assert L["label"] == "USABLE" and near(L["R"], 0.3) and near(L["sd"], (0.02 / 4) ** 0.5) and L["n_positive"] == 5 and L["n_same_sign"] == 5 and L["n_relearn_seeds"] == 5
assert all(L["relearn_cycles"][str(s)] == 3 for s in K.LIF_SEEDS) and sorted(L["values"]) == ["0", "1", "2", "3", "4"]
r = run_ref(lif={4: (-0.1, True)})                                    # exactly 4 of 5 positive and 4 of 5 relearn (a negative rise_1 cannot relearn): USABLE
L = r["lif_reference"]
assert L["label"] == "USABLE" and L["n_positive"] == 4 and L["n_relearn_seeds"] == 4 and L["n_same_sign"] == 4 and near(L["R"], (4 * 0.3 - 0.1) / 5)
r = run_ref(lif={3: (0.3, False), 4: (0.3, False)})                   # all 5 positive but relearning in only 3 of 5: NO-REFERENCE
L = r["lif_reference"]
assert L["label"] == "NO-REFERENCE" and L["n_positive"] == 5 and L["n_relearn_seeds"] == 3
r = run_ref(lif={3: (-0.1, True), 4: (-0.1, True)})                   # positive in only 3 of 5: NO-REFERENCE
assert r["lif_reference"]["label"] == "NO-REFERENCE" and r["lif_reference"]["n_positive"] == 3
assert {v["rescored"]["label"] for v in r["rate"]["bases"].values()} == {"NO-REFERENCE"}
assert r["rate"]["arms"]["arm_i"]["label"] == "NO-REFERENCE" and r["rate"]["arms"]["arm_ii"]["label"] == "NO-REFERENCE"
assert all(v["rescored"]["b"] is True and v["rescored"]["a"] is None for v in r["rate"]["bases"].values())    # (b) still reported, (a) untestable
# a guard-aborted LIF seed is FAIL-UNSTABLE: not scored, listed; with 4 clean seeds the reference stays usable, with 3 it does not
ab = {("timed", 2): [{"arm": "bidir_timed", "phase": "abort", "label_hint": "FAIL-UNSTABLE"}]}
r = run_ref(extra=ab)
L = r["lif_reference"]
assert L["unstable_seeds"] == [2] and L["n_scored"] == 4 and L["label"] == "USABLE" and "2" not in L["values"] and L["n_relearn_seeds"] == 4
r = run_ref(extra={**ab, ("depress", 3): [{"arm": "bidir_depress", "phase": "abort", "label_hint": "FAIL-UNSTABLE"}]})
assert r["lif_reference"]["unstable_seeds"] == [2, 3] and r["lif_reference"]["label"] == "NO-REFERENCE"
print("47 bidir-ref: LIF sanity rule USABLE (5/5, 4/5) vs NO-REFERENCE (3/5 relearn, 3/5 positive, 2 aborted seeds); R, SD, n same sign; rate NO-REFERENCE with (b) still reported", flush=True)

# 48-51. Re-scored rate label against R = 0.3 (threshold 0.5 R = 0.15): PASS, FAIL (a), FAIL (b), FAIL both.
r = run_ref(rate={"c0": (0.16, True), "c1s0": (0.14, True), "c1s1": (0.2, False), "c1s2": (0.1, False), "c1s3": (0.15001, True)})
B = {b: v["rescored"] for b, v in r["rate"]["bases"].items()}
assert near(r["lif_reference"]["R"], 0.3) and r["label_name"] == "bidir vs LIF reference"
assert B["c0"]["label"] == "PASS" and B["c0"]["failing_part"] is None and B["c0"]["a"] is True and B["c0"]["b"] is True and near(B["c0"]["threshold"], 0.15)
assert B["c1s0"]["label"] == "FAIL" and B["c1s0"]["failing_part"] == "(a)" and B["c1s0"]["a"] is False and B["c1s0"]["b"] is True
assert B["c1s1"]["label"] == "FAIL" and B["c1s1"]["failing_part"] == "(b)" and B["c1s1"]["a"] is True and B["c1s1"]["b"] is False
assert B["c1s2"]["label"] == "FAIL" and B["c1s2"]["failing_part"] == "both"
assert B["c1s3"]["label"] == "PASS" and B["c1s4"]["label"] == "PASS"
assert r["rate"]["arms"]["arm_i"] == {"base": "c0", "label": "PASS", "failing": None}
# dose-matched secondary: rung A drop = 30 |w| at |w| = 0.01 -> 0.3; ratio = rise_1_norm / 0.3
D = r["rate"]["bases"]["c0"]["dose_matched_rung_a"]
assert D["status"] == "in-range" and near(D["value"], 0.3) and near(D["ratio"], 0.16 / 0.3) and near(D["w"], 0.01)
assert near(r["lif_reference"]["seeds"]["0"]["dose_matched_rung_a"]["ratio"], 1.0)
ser = r["rate"]["bases"]["c0"]["series"]["timed"]
assert set(ser) == {"avoid_mbon_hz", "pam_hz", "abs_w_mean_frac"} and near(ser["abs_w_mean_frac"]["F1"], 0.01) and near(ser["abs_w_mean_frac"]["T0"], 0.0)
print("48-51 bidir-ref: rate label PASS / FAIL (a) / FAIL (b) / FAIL both against R; threshold side checked at 0.15001; dose-matched ratio; per-block series", flush=True)

# 52. Arm (ii): PASS iff >= ceil(0.8 x 5) = 4 of 5 seeds PASS; 3 of 5 fails; arm (i) is c0 alone.
r = run_ref(rate={"c1s4": (0.1, True)})
a2 = r["rate"]["arms"]["arm_ii"]
assert a2["label"] == "PASS" and a2["n_pass"] == 4 and a2["need"] == 4 and a2["per_seed"]["c1s4"] == {"label": "FAIL", "failing": "(a)"}
r = run_ref(rate={"c1s3": (0.1, True), "c1s4": (0.2, False), "c0": (0.2, False)})
assert r["rate"]["arms"]["arm_ii"]["label"] == "FAIL" and r["rate"]["arms"]["arm_ii"]["n_pass"] == 3
assert r["rate"]["arms"]["arm_i"]["label"] == "FAIL" and r["rate"]["arms"]["arm_i"]["failing"] == "(b)"
print("52 bidir-ref: arm (ii) PASS at 4 of 5 seeds, FAIL at 3 of 5; arm (i) = c0", flush=True)

# 53. Rate-log sha256 against EVALUATION.md: edited log, missing log, unlisted log, edited rung_a log all refuse and write nothing.
refuses_ref("bidir_timed_c1s2.jsonl (sha256", mutate=lambda e: append_to(os.path.join(e.rate, "bidir_timed_c1s2.jsonl"), "{}" + chr(10)))
refuses_ref("bidir_depress_c0.jsonl (missing)", mutate=lambda e: os.remove(os.path.join(e.rate, "bidir_depress_c0.jsonl")))
refuses_ref("rung_a_c1s1.jsonl (sha256", mutate=lambda e: append_to(os.path.join(e.rate, "rung_a_c1s1.jsonl"), chr(10)))
refuses_ref("bidir_timed_c0.jsonl (not in", mutate=lambda e: open(e.eval, "w").write("- other.jsonl " + "0" * 64 + chr(10)))
print("53 bidir-ref: rate-log sha256 mismatch / missing / unlisted refuses, no result written", flush=True)

# 54. A missing LIF done row (or file) refuses, names the run, writes nothing.
refuses_ref("bidir_timed_lifs2", no_done={("timed", 2)})
refuses_ref("bidir_depress_lifs4", mutate=lambda e: os.remove(os.path.join(e.out, "bidir_depress_lifs4.jsonl")))
print("54 bidir-ref: missing LIF done row / file refuses and names the run", flush=True)

# 55. An existing result refuses without --reanalyze; with it the new result records the previous sha256.
with tempfile.TemporaryDirectory() as td:
    e = RefEnv(td)
    e.write()
    e.analyze()
    prev = K.sha256_file(os.path.join(e.out, "result.json"))
    assert os.path.exists(os.path.join(e.out, "RESULT.md"))
    try:
        e.analyze()
        raise AssertionError("55 FAIL: overwrote a result without reanalyze")
    except SystemExit as ex:
        assert "--reanalyze" in str(ex)
    assert K.sha256_file(os.path.join(e.out, "result.json")) == prev
    r2 = e.analyze(reanalyze=True)
    assert r2["reanalyzed"] is True and r2["previous_result_sha256"] == prev
    assert json.load(open(os.path.join(e.out, "result.json")))["previous_result_sha256"] == prev
    md = open(os.path.join(e.out, "RESULT.md")).read()
    assert md.index("## Rule") < md.index("## LIF reference") < md.index("## Re-scored rate label") and K.rule_text_ref() in md and "bidir vs LIF reference" in md
    assert "REANALYZED: true" in md
print("55 bidir-ref: existing result refuses without --reanalyze, records previous sha with it; RESULT.md = rule text first, then numbers", flush=True)

# 56. Paper-0 rung A rows must match reference.json sources.
refuses_ref("condition_o1s3.jsonl (sha256", mutate=lambda e: append_to(os.path.join(e.src, "results", "condition_o1s3.jsonl"), chr(10)))
refuses_ref("condition_o1s1.jsonl (not in reference.json", mutate=lambda e: e.ref["sources"].pop("results/condition_o1s1.jsonl"))
print("56 bidir-ref: paper-0 rung A rows checked against reference.json sources", flush=True)

# 57. LIF provenance as chunk 2 (HEAD = analyzer HEAD, clean, code hashes, prereg, parameters, lif block); the rate HEAD differs by design and is recorded.
with tempfile.TemporaryDirectory() as td:
    e = RefEnv(td)
    e.write()
    for kw, needle in (({"state": ("fffffff", {"rate": False, "learn": False})}, "analyzer HEAD"),
                       ({"state": ("abc1234", {"rate": True, "learn": False})}, "uncommitted"),
                       ({"code_sha": dict(CODE, **{"rate/chunk2.py": "0" * 64})}, "source changed")):
        try:
            e.analyze(**kw)
            raise AssertionError("57 FAIL: accepted %s" % needle)
        except SystemExit as ex:
            assert needle in str(ex), (needle, str(ex))
    assert e.no_output()
    r = e.analyze()
    assert r["provenance"]["git_head"] == "abc1234" and r["analyzer"]["git_head"] == "abc1234" and r["provenance"]["rate_logs"]["git_heads"] == ["r999999"]
refuses_ref("different git_head", mutate=lambda e: e.rewrite_meta("depress", 2, lambda m: m.update(git_head="def5678")))
refuses_ref("lif block", mutate=lambda e: e.rewrite_meta("timed", 0, lambda m: m["lif"].update(brain="data/other.npz")))
refuses_ref("prereg parameters", mutate=lambda e: e.rewrite_meta("timed", 0, lambda m: m["args"].update(eta=1e-5)))
refuses_ref("prereg parameters", mutate=lambda e: e.rewrite_meta("timed", 1, lambda m: m["args"].update(lif_seed=2)))
refuses_ref("prereg on disk", mutate=lambda e: open(e.prereg, "w").write("changed"))
print("57 bidir-ref: analyzer HEAD / dirty / code hash / mixed HEAD / lif block / parameters / prereg refuse; rate HEAD differs and is recorded", flush=True)

# 58. Dose-matched interpolation: linear between logged trials only; outside the range no value (origin-anchored shown apart).
pts = K.rung_a_curve(rung_rows())
assert len(pts) == 6 and near(pts[0][0], W_RUNG / 6) and near(pts[-1][1], SLOPE * W_RUNG)
assert near(K.dose_matched(pts, 0.01)["value"], 0.3) and near(K.dose_matched(pts, W_RUNG / 6)["value"], SLOPE * W_RUNG / 6)
lo, hi = K.dose_matched(pts, 0.001), K.dose_matched(pts, 0.05)
assert lo["value"] is None and lo["status"] == "below-range" and near(lo["origin_anchored"], 0.03)
assert hi["value"] is None and hi["status"] == "above-range" and K.dose_matched(pts, None)["status"] == "no-data"
print("58 bidir-ref: dose-matched rung A interpolation in range only, origin-anchored extension reported separately", flush=True)

# 59. Task-1 review minors: knobs at gpu_sim's own source defaults, resolved brain path asserted and recorded, LIF headroom ratio basis.
G_ = C.G
src_def = K.gpu_sim_source_defaults()
assert set(src_def) == set(K.LIF_DEFAULT_KNOBS) and all(getattr(G_, n) == v for n, v in src_def.items()), "gpu_sim does not sit at its source defaults"
assert src_def["GAP_COUPLE"] == 0.0 and src_def["APL_DELAYED"] is True and src_def["TYPE_W_SCALE"] == {} and src_def["SLOW_TYPES"] == ()
for name, bad in (("GAP_COUPLE", 0.5), ("KC_V_TH_DELTA", 1.0), ("APL_GRADED", True), ("NORM_TOTAL_TARGET", 1.0), ("TYPE_W_SCALE", {"KC": 2.0}),
                  ("SLOW_FRAC", 0.1), ("SFA_B_INC", 0.5)):
    keep = getattr(G_, name)
    setattr(G_, name, bad)
    try:
        K.base_lif()
        raise AssertionError("59 FAIL: base_lif accepted %s changed" % name)
    except AssertionError as ex:
        assert name in str(ex) and "source defaults" in str(ex), (name, str(ex))
    finally:
        setattr(G_, name, keep)
keep = G_.BRAIN
G_.BRAIN = os.path.join(_HERE, "data", "elsewhere.npz")
try:
    K.base_lif()
    raise AssertionError("59 FAIL: base_lif accepted another brain file")
except AssertionError as ex:
    assert "data/brain_gpu.npz" in str(ex) and "data/elsewhere.npz" in str(ex), str(ex)
finally:
    G_.BRAIN = keep
assert K.lif_meta()["brain"] == "data/brain_gpu.npz" == K.lif_brain_path()
hb = K.headroom_row("bidir", "dc2", fake(11.3), ref, 454.5, basis=K.LIF_RATIO_BASIS)
assert hb["ratio_basis"] == "vs paper-0 LIF seed-mean baseline" and "ratio_basis" not in K.headroom_row("bidir", "dc2", fake(11.3), ref, 124.4)
assert hr46[0]["ratio_basis"] == K.LIF_RATIO_BASIS
print("59 lif: base_lif refuses a changed gpu_sim knob or another brain file; meta records data/brain_gpu.npz; LIF headroom rows label ratio basis", flush=True)

# 60. CLI: --analyze-bidir-ref refuses to combine with the other analysis / run selectors (before any file is read).
for argv in (["--analyze-bidir-ref", "--analyze"], ["--analyze-bidir-ref", "--arm", "lif", "--lif-seed", "0"], ["--analyze-bidir-ref", "--fit-seed", "1"]):
    sys.argv = ["chunk2.py"] + argv
    try:
        K.main()
        raise AssertionError("60 FAIL: %s was accepted" % argv)
    except SystemExit as ex:
        assert not isinstance(ex.code, int), (argv, ex.code)
print("60 bidir-ref: CLI refuses --analyze-bidir-ref combined with --analyze / --arm / --fit-seed", flush=True)

# 61. Headroom + baseline/end latch rows are reported (result.json + RESULT.md) for every LIF seed and every rate base, never gated.
HR = lambda p, ok=True: {"arm": "headroom", "phase": "headroom", "protocol": p, "song": "dc2", "gate_pass": ok, "approach_ratio": 1.1}  # noqa: E731
LT = lambda when: {"arm": "baseline", "phase": "latch", "protocol": "bidir", "song": "dc2", "when": when, "latched": False, "n_latched_kc_paired": 0, "approach_latched_rate_frac": 0.0}  # noqa: E731
with tempfile.TemporaryDirectory() as td:
    e = RefEnv(td)
    e.write(extra={(rl, s): [HR("bidir"), LT("baseline"), LT("end")] for s in K.LIF_SEEDS for rl in K.RULES},
            rate_extra={(rl, b): [HR("bidir"), LT("baseline")] for b in K.BASES for rl in K.RULES})
    r = e.analyze()
    md = open(os.path.join(e.out, "RESULT.md")).read()
    for s in K.LIF_SEEDS:
        for rl in K.RULES:
            sr = r["lif_reference"]["seeds"][str(s)]["science_rows"][rl]
            assert len(sr["headroom"]) == 1 and [x["when"] for x in sr["latch"]] == ["baseline", "end"]
    for b in K.BASES:
        for rl in K.RULES:
            sr = r["rate"]["bases"][b]["science_rows"][rl]
            assert len(sr["headroom"]) == 1 and [x["when"] for x in sr["latch"]] == ["baseline"]
    assert "## Headroom and latch rows" in md and "LIF seed 4 depress latch end dc2" in md and "rate base c1s4 timed latch baseline dc2" in md
    assert r["lif_reference"]["label"] == "USABLE"    # reported only
print("61 bidir-ref: headroom + latch rows reported per LIF seed and rate base in result.json and RESULT.md, not gated", flush=True)

# 62. One relearn computation: the chunk-2 labeller (score_bidir) and the re-score (bidir_numbers) agree on cycles / n_relearn / rise_1 / rise_1_norm
#     on synthetic pairs (incl. a negative and a zero rise_1); the chunk-2 bidir labels on the 12 real rate logs are unchanged (booleans only).
for f1_, ok_ in ((0.3, True), (0.3, False), (-0.1, True), (0.0, True), (0.05, False)):
    t_, d_ = pair(f1_, ok_)
    passed_, num_ = K.score_bidir(t_, d_, ref)
    bn = K.bidir_numbers(t_, d_)
    assert num_["cycles"] == bn["cycles"] and num_["n_relearn"] == bn["n_relearn"] and num_["rise_1"] == bn["rise_1"] and num_["rise_1_norm"] == bn["rise_1_norm"]
    assert (bn["n_relearn"] == 0) if f1_ <= 0 else True
t_, d_ = pair(0.3, True)
assert K.score_bidir(t_[:-3], d_, ref)[0] is None and K.score_bidir(t_[:-3], d_, ref)[1]["missing"] == K.bidir_numbers(t_[:-3], d_)["missing"]
stored = json.load(open(os.path.join(K.OUT, "result.json")))
same = []
for b in K.BASES:
    rows_ = {rl: K.read_rows(os.path.join(K.OUT, "bidir_%s_%s.jsonl" % (rl, b))) for rl in K.RULES}
    passed_, num_ = K.score_bidir(rows_["timed"], rows_["depress"], ref)
    st = (stored["c0"] if b == "c0" else stored["c1"]["seeds"][b[3:]])["protocols"]["bidir"]
    same.append(passed_ == st["passed"] or (passed_ is False and st["passed"] is False))
    sn = st["numbers"]
    same.append(all(num_[k] == sn[k] for k in ("rise_1", "rise_1_norm", "n_relearn", "scorable", "baseline_approach_hz", "w_mean_frac")))
    same.append({str(k): v for k, v in num_["cycles"].items()} == sn["cycles"])
print("62 bidir-ref: chunk-2 labeller and re-score share one relearn computation (synthetic incl. negative rise_1); real chunk-2 bidir results unchanged: %s" % all(same), flush=True)
assert all(same) and len(same) == 3 * len(K.BASES)

# 63. Dose curve is kept in trial order; a non-monotone |w| over trials is flagged and gives no interpolated value; dose_row takes rise_1_norm as given.
pts_bad = [(0.003, 0.1), (0.006, 0.2), (0.005, 0.25), (0.012, 0.4)]
nm = K.dose_matched(pts_bad, 0.004)
assert nm["dose_curve_nonmonotone"] is True and nm["value"] is None and nm["status"] == "nonmonotone"
assert K.dose_matched(pts, 0.01)["dose_curve_nonmonotone"] is False and K.rung_a_curve(rung_rows()) == pts
dr_ = K.dose_row(0.15, 0.01, rung_rows())
assert near(dr_["ratio"], 0.5) and dr_["rise_1_norm"] == 0.15
print("63 bidir-ref: non-monotone dose curve flagged with no value; dose_row uses the given rise_1_norm", flush=True)

print("ok chunk2")
