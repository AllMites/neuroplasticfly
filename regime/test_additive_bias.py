"""run_batch(bias=): additive input that keeps synaptic input, unlike a drive (clamp).

(a) off == baseline, bit-exact: bias=None and an all-zero bias through the bias
    arithmetic both reproduce the no-bias counts and state exactly.
(b) a bias on one neuron raises its rate, and silencing its presynaptic partners still
    changes that rate with the bias on (its synaptic input is still in the sum). The
    same neuron under a Poisson drive (the clamp) gives the SAME count with or without
    its partners: the clamp discards synaptic input.
(c) bias_mv_for_hz: with every partner silenced the cell is isolated, and the bias
    for 40 Hz gives 40 Hz (+-1 spike in 300 ms).

Run: .venv/Scripts/python.exe tools/gpu_slot.py -- .venv/Scripts/python.exe regime/test_additive_bias.py
(TEST_DEVICE=cpu runs it without the GPU, ~1 min.)
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gpu_sim as G  # noqa: E402

T = 300.0
meta = np.load(os.path.join(G._HERE, "data", "neuron_meta.npz"), allow_pickle=False)
ct = meta["cell_type"].astype(str)
stim = np.flatnonzero(ct == "ORN_DM4").astype(np.int64)
drive = [(stim, np.full(stim.size, 150 * G.DT / 1000.0))]
sim = G.GpuSim(compile_=False, device=os.environ.get("TEST_DEVICE", "cuda"))
print("sim built", flush=True)


def run(drives=drive, **kw):
    c, st = sim.run_batch(drives, seeds=[0], t_run=T, return_state=True, **kw)
    return c.cpu().numpy()[0], st


def same(x, y):
    (cx, sx), (cy, sy) = x, y
    return np.array_equal(cx, cy) and all(
        np.array_equal(sx[k].cpu().numpy(), sy[k].cpu().numpy()) for k in "vgra")


# (a) ------------------------------------------------------------ off is untouched
base = run()
assert same(base, run(bias=None)), "bias=None changed the run"
assert same(base, run(bias=[(np.array([0], np.int64), 0.0)])), \
    "zero bias through the bias branch changed the run"
assert same(base, run(bias=np.zeros(sim.net.n, np.float32))), "dense zero bias changed the run"
print("(a) off == baseline, zero bias == baseline: bit-exact", flush=True)

# (b) ------------------------------------------------------------ add, not replace
c0 = base[0]
ip, ix = sim.net.W_indptr, sim.net.W_indices
is_stim = np.zeros(sim.net.n, bool)
is_stim[stim] = True
excl = is_stim.copy()
excl[sim.apl_idx.cpu().numpy()] = True
excl[sim.kc_idx.cpu().numpy()] = True        # APL targets: graded APL is not
excl[sim.apl_tgt.cpu().numpy()] = True       # removable by silence, so not isolable
excl[sim.gap_idx.cpu().numpy()] = True


def partners(t):
    off = np.flatnonzero(ix == t)
    pre = np.unique(np.searchsorted(ip, off, side="right") - 1).astype(np.int64)
    return pre[pre != t]                     # an autapse must not silence the target


tgt = None
for t in np.argsort(-c0, kind="stable"):
    if not 3 <= c0[t] <= 30 or excl[t]:     # 10-100 Hz: off the refractory ceiling
        continue
    pre = partners(t)
    if pre.size and not is_stim[pre].any():
        tgt, pre_t = int(t), pre
        break
assert tgt is not None, "no active second-order neuron to test on"
print("target %d %s: %d spikes baseline, %d presynaptic partners"
      % (tgt, ct[tgt], c0[tgt], pre_t.size), flush=True)

B_MV = 5.0                                   # subthreshold alone (threshold is 7 mV)
one = [(np.array([tgt], np.int64), B_MV)]
cb, _ = run(bias=one)
c0s, _ = run(silence=pre_t)
cbs, _ = run(bias=one, silence=pre_t)
print("bias off: %d intact, %d partners silenced | bias %.1f mV: %d intact, %d silenced"
      % (c0[tgt], c0s[tgt], B_MV, cb[tgt], cbs[tgt]), flush=True)
assert cb[tgt] > c0[tgt], "bias did not raise the target's rate"
assert cbs[tgt] != cb[tgt], "with bias on, silencing the partners changed nothing: input dropped"

hz = max(cb[tgt] / (T / 1000.0), 1.0)
cl = [(np.concatenate([stim, [tgt]]).astype(np.int64),
       np.concatenate([drive[0][1], [hz * G.DT / 1000.0]]))]
cc, _ = run(cl)
ccs, _ = run(cl, silence=pre_t)
print("clamp %.0f Hz: %d intact, %d silenced" % (hz, cc[tgt], ccs[tgt]), flush=True)
assert cc[tgt] == ccs[tgt], "clamp responded to its partners; the contrast is not what it claims"
print("(b) bias keeps synaptic input (%d vs %d); clamp ignores it (%d vs %d)"
      % (cb[tgt], cbs[tgt], cc[tgt], ccs[tgt]), flush=True)

# (c) ------------------------------------------------------------ Hz conversion
mv = float(G.bias_mv_for_hz(40.0))
cf, _ = run(bias=[(np.array([tgt], np.int64), mv)], silence=pre_t)
print("(c) isolated cell, bias %.3f mV for 40 Hz: %d spikes in %.0f ms (expect %d)"
      % (mv, cf[tgt], T, round(40 * T / 1000.0)), flush=True)
assert abs(cf[tgt] - 40 * T / 1000.0) <= 1, "bias_mv_for_hz off by more than one spike"

# guard: a bias on a Poisson-driven neuron is refused
try:
    run(bias=[(stim[:1], 1.0)])
    raise SystemExit("FAIL: bias on a stimulated neuron was accepted")
except AssertionError as e:
    assert "Poisson-stimulated" in str(e), str(e)
print("ok additive bias", flush=True)
