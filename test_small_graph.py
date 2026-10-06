"""Small-graph engine test on CPU (issue #4): gpu_sim.GpuSim and rate.engine.RateSim on a
synthetic circuit (tools/small_graph.py), no data/ and no GPU needed.

Checks the engines' wiring, not the fly: rest is silent, a drive propagates only along
edges, inhibition has a sign, runs are bit-reproducible and batch-invariant, a carried
state equals an uninterrupted run, silencing gates output, and the rate engine settles on
its analytic fixed point.

Run: python test_small_graph.py
"""
import atexit
import os
import shutil
import sys
import tempfile

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "tools"))
import gpu_sim as G  # noqa: E402
import small_graph as SG  # noqa: E402
from rate.engine import RateSim  # noqa: E402

tmp = tempfile.mkdtemp(prefix="small_graph_")
atexit.register(shutil.rmtree, tmp, True)
path = SG.build(tmp)
G._HERE = tmp                   # gpu_sim reads <_HERE>/data/neuron_meta.npz; see small_graph.py

# ------------------------------------------------------------------ spiking engine
net = G.Brain(path)
assert net.n == SG.N_V783 and net.n_edges == len(SG.edges())
assert np.array_equal(np.sort(net.select("cell_type=IN")), SG.IN)
sim = G.GpuSim(n=net, device="cpu", compile_=False)
none = (np.empty(0, np.int64), np.empty(0, np.float64))
d_in = G.drive_of_selector("cell_type=IN", 300.0, n=net)
d_inh = (np.append(d_in[0], SG.INH), np.append(d_in[1], 300.0 * G.DT / 1000.0))
T = 100.0


def run(drives, seeds, **kw):
    out = sim.run_batch(drives, seeds=seeds, t_run=T, **kw)
    if isinstance(out, tuple):
        return (out[0].cpu().numpy(),) + out[1:]
    return out.cpu().numpy()


# 1. Rest: no drive, no spikes anywhere.
assert run([none], [0]).sum() == 0, "spikes at rest"

# 2. Propagation: IN drives RELAY drives OUT; nothing outside the circuit fires.
c = run([d_in, d_inh], [0, 0])
assert (c[0, SG.IN] > 0).all() and (c[0, SG.RELAY] > 0).all() and (c[0, SG.OUT] > 0).all(), \
    c[0, :SG.N_WIRED]
assert c[:, SG.N_WIRED:].sum() == 0, "an unconnected neuron fired"

# 3. Inhibition has a sign. Same seed in both columns, so the IN spike trains are identical:
#    driving INH must lower OUT 0..3 (its targets) and leave OUT 4..7 bit-identical.
assert c[1, SG.INH] > c[0, SG.INH]
assert (c[1, SG.OUT[:4]] < c[0, SG.OUT[:4]]).all(), (c[0, SG.OUT], c[1, SG.OUT])
assert np.array_equal(c[1, SG.OUT[4:]], c[0, SG.OUT[4:]])

# 4. Reproducible and batch-invariant: a position's counts do not depend on its batch.
assert np.array_equal(run([d_in, d_inh], [0, 0]), c), "rerun differs"
assert np.array_equal(run([d_in], [0])[0], c[0]), "counts depend on batch composition"
assert not np.array_equal(run([d_in], [1])[0], c[0]), "seed does not change the Poisson stream"

# 5. Carry: two carried halves == one run, exactly (regime/test_carry.py on the real brain).
T = 50.0
a, st = run([d_in], [0], return_state=True)
b, _ = run([d_in], [0], state=st, return_state=True)
assert np.array_equal(a + b, c[:1]), "carried 2 x 50 ms != 100 ms"
T = 100.0

# 6. Silence gates output: silenced RELAYs never fire, so OUT and INH get nothing.
s = run([d_in], [0], silence=SG.RELAY)
assert s[0, SG.RELAY].sum() == 0 and s[0, SG.OUT].sum() == 0 and s[0, SG.INH] == 0
assert np.array_equal(s[0, SG.IN], c[0, SG.IN])
print("ok gpu_sim (cpu, %d wired of %d neurons): rest, propagation, inhibition sign, "
      "determinism, batch invariance, carry, silence" % (SG.N_WIRED, net.n))

# ------------------------------------------------------------------ rate engine
r = RateSim(0.01, path=path, device="cpu", dtype=torch.float64)
assert r.n == SG.N_V783 and r.W.nnz == len(SG.edges())
drive = [(SG.IN, 40.0)]

rest = r.run([(np.empty(0, np.int64), 0.0)], t_run=100.0)
assert torch.count_nonzero(rest) == 0, "rate engine not silent at rest"

# Analytic fixed point (gain 1, bias 0): v = w_scale * W^T r.
relay = 0.01 * SG.W_IN_RELAY * 40.0
inh = 0.01 * SG.W_RELAY_INH * len(SG.RELAY) * relay
out_inh = 0.01 * (SG.W_RELAY_OUT * relay + SG.W_INH_OUT * inh)
out_free = 0.01 * SG.W_RELAY_OUT * relay
_, v = r.run(drive, t_run=3000.0, return_state=True)
v = v[:, 0]
want = {"RELAY": (SG.RELAY, relay), "INH": ([SG.INH], inh), "OUT 0..3": (SG.OUT[:4], out_inh),
        "OUT 4..7": (SG.OUT[4:], out_free)}
for k, (idx, x) in want.items():
    assert torch.allclose(v[idx], torch.full((len(idx),), x, dtype=v.dtype), atol=1e-6), (k, v[idx], x)
assert torch.count_nonzero(v[SG.N_WIRED:]) == 0

# Determinism and carry, exact.
a = r.run(drive, t_run=300.0)
assert torch.equal(a, r.run(drive, t_run=300.0))
_, s1 = r.run(drive, t_run=150.0, return_state=True)
_, s2 = r.run(drive, t_run=150.0, state=s1, return_state=True)
_, s300 = r.run(drive, t_run=300.0, return_state=True)
assert torch.equal(s2, s300), "rate carry != uninterrupted"
print("ok rate engine (cpu): rest, analytic fixed point, determinism, carry")
