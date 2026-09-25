"""Homeostatic normalisation of incoming input, and the one interaction that
would silently undo it on 18,674 edges.

The scale is per POSTSYNAPTIC neuron over TOTAL input magnitude and is applied
to every one of that neuron's incoming edges, so the convergence tail flattens
while the neuron's E/I ratio is preserved exactly. An earlier version scaled
excitation alone; that inverted E/I instead of flattening the tail and silenced
the brain one hop from the driven ORNs, so the ratio assertion below is the
assertion this test exists for.
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gpu_sim as G
from regime import weights as W
from learn.plastic import Plastic

assert G.NORM_TOTAL_TARGET == 0.0, "normalisation must ship default-off until Task 7"

sim = G.GpuSim()
assert sim.norm_scale.shape == (sim.net.n,), sim.norm_scale.shape
assert sim.norm_scale.dtype == np.float32, sim.norm_scale.dtype
assert (sim.norm_scale == 1.0).all(), "scale must be all ones while the target is 0.0"

# the scale maths, checked against regime/weights.py on the real matrix
ip, ix, wd, N = sim.net.W_indptr, sim.net.W_indices, sim.net.W_data, sim.net.n
exc0, inh0, _ = W.input_totals(ip, ix, (wd * np.float32(G.W_SYN)).astype(np.float32), N)
tot = exc0 - inh0                      # summed |weight| in mV; inh sums are negative
target = 29.15
hit = tot > 0
want = np.ones(N, np.float32)
want[hit] = (target / tot[hit]).astype(np.float32)
got = G.GpuSim._input_scale(sim.net, target)
assert np.allclose(got, want, rtol=1e-5), "scale is not target / summed total input"
assert (got[~hit] == 1.0).all(), "neurons with no input at all must be left alone"
assert hit.sum() > 100000, int(hit.sum())

# applying it hits the target, and scales inhibition by the SAME factor
vals = (wd * np.float32(G.W_SYN)).astype(np.float32) * got[ix]
exc2, inh2, _ = W.input_totals(ip, ix, vals.astype(np.float32), N)
assert np.allclose((exc2 - inh2)[hit], target, rtol=1e-4), \
    ((exc2 - inh2)[hit].min(), (exc2 - inh2)[hit].max())
assert np.allclose(inh2[hit], inh0[hit] * got[hit], rtol=1e-4), \
    "inhibition was not scaled by the same per-neuron factor"
assert not np.allclose(inh2, inh0), "inhibition was left unscaled - E/I would invert"

# THE POINT: each neuron's E/I ratio survives untouched.
both = hit & (exc0 > 0) & (inh0 < 0)
assert both.sum() > 50000, int(both.sum())
assert np.allclose(exc2[both] / -inh2[both], exc0[both] / -inh0[both], rtol=1e-4), \
    "E/I ratio changed - normalisation is not ratio-preserving"

# and the tail is what actually flattened
assert tot[hit].max() / np.median(tot[hit]) > 10.0, "no tail to flatten in the input"

# THE REGRESSION. With normalisation on, an untrained Plastic (dw = 0) must push
# values identical to what normalisation already wrote. Before the signed_values
# fix this reverts 18,674 edges to w0 * W_SYN.
p = Plastic.real()
assert p.n_edges == 18674, p.n_edges
G.NORM_TOTAL_TARGET = target
try:
    sim2 = G.GpuSim()
    off = np.asarray(p.offsets, np.int64)
    before = sim2.data.cpu().numpy()[off].copy()
    p.push(sim2)
    after = sim2.data.cpu().numpy()[off]
    n_bad = int((np.abs(before - after) > 1e-6).sum())
    assert n_bad == 0, "push() reverted %d normalised edges" % n_bad
    assert not np.allclose(before, p.signed_values(G.W_SYN)), \
        "normalisation did not change the plastic edges at all - the test is vacuous"
    # the 5 negative plastic edges are scaled too, not special-cased
    neg = p.sign < 0
    assert neg.sum() == 5, int(neg.sum())
    raw_neg = (p.w() * p.sign * np.float32(G.W_SYN))[neg]
    assert not np.allclose(before[neg], raw_neg), \
        "the negative plastic edges were skipped by the scale"

    # THE OTHER PATH OUT. sim.data is not the only copy of the weights:
    # graded APL keeps a private one in apl_w that bypasses the step-loop
    # gather entirely, so normalisation has to reach it too or APL's 5,524
    # outgoing edges are the one unnormalised pathway in the brain - and they
    # land on the Kenyon cells, which sit in the tail that gets scaled hardest.
    ip2, ix2 = sim2.net.W_indptr, sim2.net.W_indices
    aidx = sim2.apl_idx.cpu().numpy()
    assert aidx.size == 2, aidx
    offs_apl = np.concatenate([np.arange(ip2[i], ip2[i + 1]) for i in aidx])
    from_data = sim2.data.cpu().numpy()[offs_apl]
    from_apl = sim2.apl_w.cpu().numpy()
    assert from_apl.shape == from_data.shape, (from_apl.shape, from_data.shape)
    assert np.allclose(from_apl, from_data, rtol=1e-5, atol=1e-7), \
        "apl_w bypassed normalisation: max rel dev %.3g" % float(
            np.abs(from_apl - from_data).max() / max(np.abs(from_data).max(), 1e-9))
    # and the check is not vacuous: normalisation really did move these edges
    raw_apl = np.concatenate(
        [sim2.net.W_data[ip2[i]:ip2[i + 1]] for i in aidx]) * np.float32(G.W_SYN)
    assert not np.allclose(from_apl, raw_apl), \
        "normalisation left APL's outgoing edges unchanged - test is vacuous"
finally:
    G.NORM_TOTAL_TARGET = 0.0
print("ok norm: scale maths, E/I ratio preserved, default off, survives Plastic.push")
