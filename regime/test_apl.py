"""Graded APL: its outgoing weights are delivered in proportion to depolarisation."""
import os, sys
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gpu_sim as G

assert G.APL_GRADED is False, "graded APL must ship default-off until Task 7"
sim = G.GpuSim()
assert sim.apl_idx.tolist() == [17917, 67229], sim.apl_idx.tolist()
assert sim.apl_tgt.numel() == sim.apl_w.numel()
assert sim.apl_tgt.numel() == 2763 + 2761, sim.apl_tgt.numel()
assert float(sim.apl_w.sum()) < 0.0, "APL output must be inhibitory"

# activation: 0 at rest, 1 at APL_SCALE above rest, clamped above that
v = torch.tensor([[G.V_REST], [G.V_REST + G.APL_SCALE]], device=sim.device)
act = sim._apl_activation(v)
assert abs(float(act[0, 0]) - 0.0) < 1e-6, float(act[0, 0])
assert abs(float(act[1, 0]) - 1.0) < 1e-6, float(act[1, 0])
v2 = torch.tensor([[G.V_REST - 5.0], [G.V_REST + 100.0]], device=sim.device)
act2 = sim._apl_activation(v2)
assert float(act2[0, 0]) == 0.0 and float(act2[1, 0]) == 1.0, "activation not clamped to [0,1]"

# delivery scales linearly and lands on APL's targets only
g = torch.zeros((sim.net.n, 1), dtype=torch.float32, device=sim.device)
sim._deliver_apl(g, torch.full((2, 1), 1.0, device=sim.device))
full = g.clone()
assert float(full.sum()) < 0.0
assert float(full.sum()) < 0.0
# rate equivalence: act=1 delivers the charge a spiking APL at APL_MAX_HZ would,
# NOT one full row per 0.1 ms step (which would be a ~10 kHz APL).
want = float(sim.apl_w.sum()) * G.APL_MAX_HZ * G.DT / 1000.0
assert abs(float(full.sum()) - want) < 1e-3 * abs(want), (float(full.sum()), want)
g.zero_()
sim._deliver_apl(g, torch.full((2, 1), 0.5, device=sim.device))
assert torch.allclose(g, full * 0.5, atol=1e-5), "delivery is not linear in activation"
touched = (full != 0).squeeze(1).nonzero().squeeze(1)
assert set(touched.tolist()) <= set(sim.apl_tgt.tolist()), "delivered outside APL targets"

# left and right APL share targets, so the scatter has duplicate indices and
# must be deterministic: same input, same output, every time.
g.zero_(); sim._deliver_apl(g, torch.full((2, 1), 1.0, device=sim.device))
once = g.clone()
for _ in range(5):
    g.zero_(); sim._deliver_apl(g, torch.full((2, 1), 1.0, device=sim.device))
    assert torch.equal(g, once), "APL scatter is not bit-reproducible"
assert sim.apl_tgt.unique().numel() < sim.apl_tgt.numel(), \
    "expected shared targets between left and right APL; if this ever fails the " \
    "determinism guard in _deliver_apl is no longer load-bearing, not that it is wrong"

# ---------------------------------------------------------------- divisive form
assert G.APL_DIVISIVE is False, "divisive APL must ship default-off"
inh = -full                                   # magnitude the subtractive form adds
G.APL_DIVISIVE = True
try:
    # It is MULTIPLICATIVE on drive: g must come back as g / (1 + inh/S),
    # exactly, and a zero g must stay zero (an offset would not).
    g.zero_()
    sim._deliver_apl(g, torch.full((2, 1), 1.0, device=sim.device))
    assert float(g.abs().sum()) == 0.0, "divisive APL added an offset to a zero drive"

    base = torch.linspace(-2.0, 10.0, sim.net.n, device=sim.device).unsqueeze(1)
    g = base.clone()
    sim._deliver_apl(g, torch.full((2, 1), 1.0, device=sim.device))
    want = base / (1.0 + inh / G.APL_DIV_SCALE)
    assert torch.allclose(g, want, atol=1e-6), "not g / (1 + inh/S)"
    # ... and it scales, it does not offset: doubling the drive doubles the result.
    g2 = base.clone() * 2.0
    sim._deliver_apl(g2, torch.full((2, 1), 1.0, device=sim.device))
    assert torch.allclose(g2, g * 2.0, atol=1e-5), "divisive APL is not homogeneous in g"
    # untouched neurons keep their drive bit for bit
    untouched = (inh == 0).squeeze(1)
    assert torch.equal(g[untouched], base[untouched]), "divided outside APL targets"

    # same shared-target scatter, so the same determinism requirement
    g = base.clone(); sim._deliver_apl(g, torch.full((2, 1), 1.0, device=sim.device))
    once = g.clone()
    for _ in range(5):
        g = base.clone()
        sim._deliver_apl(g, torch.full((2, 1), 1.0, device=sim.device))
        assert torch.equal(g, once), "divisive APL scatter is not bit-reproducible"

    # stronger means smaller: halving APL_DIV_SCALE deepens the division
    s0 = G.APL_DIV_SCALE
    G.APL_DIV_SCALE = s0 / 2.0
    g = base.clone(); sim._deliver_apl(g, torch.full((2, 1), 1.0, device=sim.device))
    hit = (inh > 0).squeeze(1) & (base.squeeze(1) > 0)
    assert bool((g[hit] < once[hit] - 1e-9).all()), "smaller APL_DIV_SCALE is not stronger"
    G.APL_DIV_SCALE = s0
finally:
    G.APL_DIVISIVE = False

print("ok apl: indices, activation curve, linear inhibitory delivery, "
      "divisive form, determinism")
