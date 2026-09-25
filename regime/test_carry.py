"""Brain state carry-over: chaining ticks must equal never stopping.

The closed loop ticks the brain every few tens of ms. Without carry-over each tick
restarts from V_REST, so the brain reboots several times a second and re-runs its
ignition transient every time. This checks the carry-over is COMPLETE -- exact
equality, not "close enough", because the failure modes are all silent:

  - dropping the `pending` / `apl_pending` delay lines deletes every in-flight spike
    at each tick boundary, which reads as a slightly quieter brain, not as a bug;
  - restarting the counter RNG at step 0 replays identical Poisson draws every tick,
    a periodic artefact at exactly the tick frequency.

Run: .venv/Scripts/python.exe regime/test_carry.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gpu_sim as G  # noqa: E402

meta = np.load(os.path.join(G._HERE, "data", "neuron_meta.npz"), allow_pickle=False)
ct = meta["cell_type"].astype(str)
idx = np.flatnonzero(ct == "ORN_DM4").astype(np.int64)
drive = [(idx, np.full(len(idx), 40 * G.DT / 1000.0))]

sim = G.GpuSim(compile_=False)


def counts(**kw):
    return sim.run_batch(drive, seeds=[0], **kw)


# 1. Default path untouched: no state arguments behaves exactly as before, and asking
#    for the state back does not change the counts it returns.
plain = counts(t_run=300.0).cpu().numpy()
same, st300 = counts(t_run=300.0, return_state=True)
assert np.array_equal(plain, same.cpu().numpy()), \
    "return_state changed the counts of an otherwise identical call"

# 2. The point of the whole exercise: 3 x 100 ms carried == 1 x 300 ms, exactly.
total = np.zeros_like(plain)
st = None
for _ in range(3):
    c, st = counts(t_run=100.0, state=st, return_state=True)
    total += c.cpu().numpy()
assert np.array_equal(total, plain), (
    "chained ticks differ from one continuous run: max |diff| %d over %d neurons"
    % (np.abs(total - plain).max(), (total != plain).sum()))

# 3. Negative test: the check above must actually be able to fail. Resume with the
#    delay line thrown away and the result has to diverge, otherwise check 2 proves
#    nothing about `pending`.
_, st_a = counts(t_run=100.0, return_state=True)
broken = dict(st_a)
empty = st_a["pending"][0]
broken["pending"] = [(empty[0][:0], empty[1][:0]) for _ in st_a["pending"]]
c_broken, _ = counts(t_run=100.0, state=broken, return_state=True)
c_good, _ = counts(t_run=100.0, state=st_a, return_state=True)
assert not np.array_equal(c_broken.cpu().numpy(), c_good.cpu().numpy()), \
    "dropping the synaptic delay line changed nothing; check 2 cannot detect it"

# 4. The counter RNG must advance across ticks, not replay. Two consecutive carried
#    ticks under a CONSTANT drive should not produce identical spike counts.
_, s1 = counts(t_run=100.0, return_state=True)
t1, s2 = counts(t_run=100.0, state=s1, return_state=True)
t2, _ = counts(t_run=100.0, state=s2, return_state=True)
assert not np.array_equal(t1.cpu().numpy(), t2.cpu().numpy()), \
    "two consecutive ticks are bit-identical; the Poisson stream is restarting"
assert s2["step0"] == 2000, "step offset did not advance: %r" % s2["step0"]

# 5. A state from a different batch width must be refused, not broadcast.
_, s_b1 = counts(t_run=50.0, return_state=True)
try:
    sim.run_batch(drive * 2, seeds=[0, 1], t_run=50.0, state=s_b1)
    raise SystemExit("FAIL: mismatched batch width was accepted")
except AssertionError as e:
    assert "batch width" in str(e), str(e)

print("ok carry: default unchanged, 3x100ms == 1x300ms exactly, delay line required, "
      "RNG advances, batch width checked")
