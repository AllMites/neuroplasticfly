"""Slow synaptic channel (SLOW_FRAC / TAU_SLOW / SLOW_TYPES): off by default, and exact.

Checks, all on the real brain:
  1. ships default-off
  2. default path == counts saved BEFORE the channel existed (bit for bit)
  3. _body: h decays by exp(-DT/TAU_SLOW), survives a spike (g is reset, h is not)
  4. charge: one presynaptic spike delivers the same summed input with slow on or off
  5. carry with slow on: 3 x 100 ms carried == 1 x 300 ms, exactly
  6. resuming a state built under a different slow config is refused

--make-golden writes results/golden_preslow.npz. Run it ONCE, at the commit before the
channel was added (PREREGISTER_slow.md records its sha256).

Run: .venv/Scripts/python.exe regime/test_slow.py
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gpu_sim as G  # noqa: E402

GOLDEN = os.path.join(G._HERE, "results", "golden_preslow.npz")
CX = ("EPG", "PEN", "PEG")

meta = np.load(os.path.join(G._HERE, "data", "neuron_meta.npz"), allow_pickle=False)
ct = meta["cell_type"].astype(str)
orn = np.flatnonzero(ct == "ORN_DM4").astype(np.int64)
cx = np.flatnonzero(np.any([np.char.startswith(ct, p) for p in CX], axis=0)).astype(np.int64)
DRIVES = [(orn, np.full(len(orn), 40 * G.DT / 1000.0)),
          (cx, np.full(len(cx), 80 * G.DT / 1000.0))]
SEEDS = [0, 1]

if "--make-golden" in sys.argv:
    assert not hasattr(G, "SLOW_FRAC"), "golden must come from the pre-channel commit"
    counts = G.GpuSim(compile_=False).run_batch(DRIVES, SEEDS, t_run=300.0).cpu().numpy()
    np.savez_compressed(GOLDEN, counts=counts)
    print("wrote %s, %d spikes" % (GOLDEN, int(counts.sum())))
    sys.exit(0)

# 1 ---------------------------------------------------------------------------
assert G.SLOW_FRAC == 0.0, "the slow channel must ship default-off"

# 2 ---------------------------------------------------------------------------
sim = G.GpuSim(compile_=False)
assert sim.slow_cfg is None
got = sim.run_batch(DRIVES, SEEDS, t_run=300.0).cpu().numpy()
want = np.load(GOLDEN)["counts"]
assert np.array_equal(got, want), "default path changed: %d vs %d spikes" % (got.sum(), want.sum())
# The closed loop runs compiled, and compiled != eager bitwise (fusion reorders float
# ops: 100457 vs 100426 spikes at the pre-channel commit), so it gets its own golden.
got_c = G.GpuSim(compile_=True).run_batch(DRIVES, SEEDS, t_run=300.0).cpu().numpy()
want_c = np.load(GOLDEN.replace(".npz", "_compiled.npz"))["counts"]
assert np.array_equal(got_c, want_c), "compiled default path changed: %d vs %d" % (
    got_c.sum(), want_c.sum())

# 3 ---------------------------------------------------------------------------
dm = float(np.float32(np.exp(-G.DT / G.TAU_M)))
gn = float(np.float32(1.0 - np.exp(-G.DT / G.TAU_M)))
ds = float(np.float32(np.exp(-G.DT / G.TAU_SYN)))
da = float(np.float32(np.exp(-G.DT / G.TAU_A)))
dh = float(np.float32(np.exp(-G.DT / 100.0)))
z8 = torch.tensor(0, dtype=torch.int8)
one = lambda x: torch.full((1, 1), x)                                        # noqa: E731
args = lambda: (torch.zeros((1, 1), dtype=torch.float64), )                  # noqa: E731
v, g, h, a = one(G.V_REST), one(3.0), one(2.0), one(0.0)
r = torch.zeros((1, 1), dtype=torch.int8)
fire = torch.ones((1, 1), dtype=torch.bool)
nos = torch.zeros((1, 1), dtype=torch.bool)
v, g, r, a, spk, h = G._body(v, g, r, a, args()[0], fire, nos, G.V_REST, dm, gn, ds, da, 0.0,
                             G.V_TH, torch.tensor(G.V_RESET), torch.tensor(21, dtype=torch.int8),
                             z8, h, dh)
assert bool(spk[0, 0])
assert float(g[0, 0]) == 0.0, "g must still reset on spike"
assert abs(float(h[0, 0]) - 2.0 * dh) < 1e-6, "h must decay, not reset, on spike: %r" % float(h)
v, g, r, a, spk, h2 = G._body(v, g, r, a, args()[0], nos, nos, G.V_REST, dm, gn, ds, da, 0.0,
                              G.V_TH, torch.tensor(G.V_RESET), torch.tensor(21, dtype=torch.int8),
                              z8, h, dh)
assert abs(float(h2[0, 0]) - float(h[0, 0]) * dh) < 1e-6

# 4 ---------------------------------------------------------------------------
# Deliver one EPG spike through the real _deliver with and without the split and sum the
# input each post would integrate: fast g decays at TAU_SYN, slow h at TAU_SLOW. Charge
# = w * tau per unit weight, so charge-matching makes the two totals equal.
pre = torch.as_tensor(cx[:1], device=sim.device)
bcol = torch.zeros(1, dtype=torch.int64, device=sim.device)
g0 = torch.zeros((sim.net.n, 1), device=sim.device)
sim._deliver(g0, pre, bcol, 1)
q_off = float(g0.sum()) * G.TAU_SYN
# EPG only for 5: driven cells spike by Poisson alone (threshold crossing is masked), so
# driving all of EPG/PEN/PEG would leave no undriven slow post to feel h. Baseline taken
# here, before the globals flip: a slow-off sim refuses to run under slow-on globals.
epg = np.flatnonzero(np.char.startswith(ct, "EPG")).astype(np.int64)
d_epg = [(epg, np.full(len(epg), 80 * G.DT / 1000.0))] * 2
base = sim.run_batch(d_epg, SEEDS, t_run=300.0)
G.SLOW_FRAC, G.SLOW_TYPES, G.TAU_SLOW = 0.5, CX, 100.0
try:
    sim_on = G.GpuSim(compile_=False)
    g1 = torch.zeros_like(g0)
    h1 = torch.zeros_like(g0)
    sim_on._deliver(g1, pre, bcol, 1, h1)
    assert float(h1.sum()) > 0, "no slow input delivered onto CX posts"
    assert bool((h1[torch.as_tensor(np.setdiff1d(np.arange(sim.net.n), cx),
                                     device=sim.device)] == 0).all()), "slow input off-population"
    q_on = float(g1.sum()) * G.TAU_SYN + float(h1.sum()) * G.TAU_SLOW
    assert abs(q_on - q_off) <= 1e-3 * abs(q_off), (q_on, q_off)

    # 5 -----------------------------------------------------------------------
    whole, _ = sim_on.run_batch(d_epg, SEEDS, t_run=300.0, return_state=True)
    st, parts = None, []
    for _ in range(3):
        c, st = sim_on.run_batch(d_epg, SEEDS, t_run=100.0, state=st, return_state=True)
        parts.append(c)
    assert torch.equal(whole, sum(parts)), "slow-on carry is not exact"
    assert not torch.equal(whole, base), "slow on changed nothing"

    # 6 -----------------------------------------------------------------------
    G.SLOW_FRAC = 0.25
    sim_q = G.GpuSim(compile_=False)
    try:
        sim_q.run_batch(d_epg, SEEDS, t_run=10.0, state=st)
        raise SystemExit("FAIL: resumed a state with a different slow config")
    except AssertionError:
        pass
finally:
    G.SLOW_FRAC, G.SLOW_TYPES, G.TAU_SLOW = 0.0, (), 100.0
print("ok slow: default-off, default path bit-identical to golden, h decays and survives spikes, "
      "charge matched (%.4g vs %.4g), carry exact, config checked on resume" % (q_on, q_off))
