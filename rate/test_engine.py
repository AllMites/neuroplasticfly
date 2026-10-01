"""Rate engine goldens (PRD path 2, phase 1). ORN->PN circuit only.

Deliberately touches NO sugar/bitter/Fox/FDA/PAM neuron: the first sugar->PAM number this
engine produces IS the phase-3 unfitted-control result and must come after its prereg
(simulation-experiment-validity rule 12).

Exact equality where the failure is silent (determinism, carry, drive clamp); tolerances
only where arithmetic genuinely differs (analytic fixed point, finite differences).

Run: .venv/Scripts/python.exe tools/gpu_slot.py -- .venv/Scripts/python.exe rate/test_engine.py
"""
import os
import sys
import time

import numpy as np
import scipy.sparse as sp
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gpu_sim as G  # noqa: E402
from rate.engine import RateSim  # noqa: E402

# 1. Toy analytic (CPU, float64): 0 -(+2)-> 1 -(-1)-> 2, neuron 0 driven at 10 Hz,
#    bias 30 on neuron 2. Fixed point: v1 = 0.1*2*10 = 2, v2 = 0.1*(-1)*2 + 30 = 29.8.
toy = sp.csr_matrix((np.array([2.0, -1.0]), np.array([1, 2]), np.array([0, 1, 2, 2])),
                    shape=(3, 3))
t = RateSim(0.1, device="cpu", dtype=torch.float64, csr=toy, pool=np.arange(3))
t.bias[2] = 30.0
_, vt = t.run([(np.array([0]), 10.0)], t_run=2000.0, return_state=True)
assert torch.allclose(vt[1:, 0], torch.tensor([2.0, 29.8], dtype=torch.float64), atol=1e-6), vt

# Whole brain, ORN_DM4 at 40 Hz (as regime/test_carry.py); PN = DM4_adPN with most input.
meta = np.load(os.path.join(G._HERE, "data", "neuron_meta.npz"), allow_pickle=False)
ct = meta["cell_type"].astype(str)
orn = np.flatnonzero(ct == "ORN_DM4").astype(np.int64)
sim = RateSim(1.0)
rho = sim.spectral_bound()
sim.w_scale = 0.5 / rho
syn = np.asarray(sim.W[orn].sum(0)).ravel()
pns = np.flatnonzero(ct == "DM4_adPN")
pn = int(pns[np.argmax(syn[pns])])
drive = [(orn, 40.0)]
print("rho(|W|) = %.1f synapses; test w_scale = 0.5/rho = %.3e" % (rho, sim.w_scale), flush=True)

# 2. Rest: no drive, bias 0 -> exactly silent.
rest = sim.run([(np.array([], dtype=np.int64), 0.0)], t_run=100.0)
assert torch.count_nonzero(rest) == 0, "brain not silent at rest"

# 3. Determinism: bit-identical reruns.
a = sim.run(drive, t_run=300.0)
b = sim.run(drive, t_run=300.0)
assert torch.equal(a, b), "two identical runs differ (non-deterministic SpMM?)"

# 4. Carry: 150 + 150 ms carried == 300 ms, exactly.
r1, st = sim.run(drive, t_run=150.0, return_state=True)
r2 = sim.run(drive, t_run=150.0, state=st)
_, st300 = sim.run(drive, t_run=300.0, return_state=True)
_, st2 = sim.run(drive, t_run=150.0, state=st, return_state=True)
assert torch.equal(st2, st300), "carried state != uninterrupted state"
# state is the exact guard; averaged rates differ only by float32 summation order
# (measured: max 3.6e-5 Hz at 16.7 Hz, rel 2e-6)
assert torch.allclose((r1 + r2) / 2, a, rtol=1e-5, atol=1e-6), "carried rates != uninterrupted rates"

# 5. Drive clamps output exactly.
assert torch.all(a[0, orn] == 40.0), a[0, orn]
assert a[0, pn] > 0, "DM4 PN got no rate from its ORNs"

# 6. Silence gates output only.
s_rates, s_v = sim.run(drive, t_run=300.0, silence=np.array([pn]), return_state=True)
assert s_rates[0, pn] == 0 and s_v[pn, 0] > 0, (s_rates[0, pn], s_v[pn, 0])
try:
    sim.run(drive, t_run=10.0, silence=orn[:1])
    raise SystemExit("FAIL: silence overlapping drive was accepted")
except AssertionError:
    pass

# 7. Gradient (CPU float64, 50 ms): autograd d(PN rate)/d(bias of PN pool) == central FD.
#    CPU because autograd keeps an [nnz] product per step (~50 MB) and VRAM is shared.
g64 = RateSim(0.5 / rho, device="cpu", dtype=torch.float64)
p = int(g64.pool[pn])
g64.set_trainable("bias")
g64.run(drive, t_run=50.0)[0, pn].backward()
auto = float(g64.bias.grad[p])
g64.bias.requires_grad_(False)
eps = 1e-3
with torch.no_grad():
    g64.bias[p] += eps
    up = float(g64.run(drive, t_run=50.0)[0, pn])
    g64.bias[p] -= 2 * eps
    dn = float(g64.run(drive, t_run=50.0)[0, pn])
    g64.bias[p] += eps
fd = (up - dn) / (2 * eps)
assert abs(auto - fd) <= 1e-4 * abs(fd), (auto, fd)
print("grad: autograd %.8f  fd %.8f" % (auto, fd), flush=True)
del g64

# 8. Stability: finite at 0.5/rho (guaranteed side); 2/rho recorded, not asserted.
assert torch.isfinite(a).all()
print("max rate at 0.5/rho: %.3f Hz" % float(a.max()), flush=True)
sim.w_scale = 2.0 / rho
hot = sim.run(drive, t_run=300.0)
print("max rate at 2/rho: %.3e Hz (n > 100 Hz: %d)"
      % (float(hot.max()), int((hot > 100).sum())), flush=True)
sim.w_scale = 0.5 / rho

# 9. Toy step: one gradient step on the PN pool bias raises the PN rate.
sim.set_trainable("bias")
before = sim.run(drive, t_run=50.0)[0, pn]
before.backward()
with torch.no_grad():
    sim.bias += 1.0 * sim.bias.grad
sim.bias.requires_grad_(False)
with torch.no_grad():
    after = sim.run(drive, t_run=50.0)[0, pn]
assert after > before, (float(before), float(after))

# 10. Timing: ms wall per simulated second.
for B in (1, 8):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        sim.run(drive * B, t_run=1000.0)
    torch.cuda.synchronize()
    print("B=%d: %.0f ms per simulated s" % (B, 1000 * (time.perf_counter() - t0)), flush=True)

# 11. Edge gains, toy analytic: gain g on edge 0->1 gives v1 = 0.1*2*g*10 at the fixed point.
cap = 18.287
t2 = RateSim(0.1, device="cpu", dtype=torch.float64, csr=toy, pool=np.arange(3))
t2.bias[2] = 30.0
tab = t2.set_edge_gains([1], cap)
assert len(tab["src_pool"]) == 1 and tab["src_pool"][0] == 0 and tab["tgt_pool"][0] == 1, tab
_, v = t2.run([(np.array([0]), 10.0)], t_run=2000.0, return_state=True)
assert torch.allclose(v[1:, 0], torch.tensor([2.0, 29.8], dtype=torch.float64), atol=1e-6), v
t2.edge_x[0] = float(np.log(3.0 / (cap - 3.0)))  # gain 3
_, v = t2.run([(np.array([0]), 10.0)], t_run=2000.0, return_state=True)
assert torch.allclose(v[1:, 0], torch.tensor([6.0, 29.4], dtype=torch.float64), atol=1e-6), v

# 12. Whole brain (CPU float64, 50 ms): gains at 1 change nothing; d(PN rate)/d(edge_x) == FD.
e64 = RateSim(0.5 / rho, device="cpu", dtype=torch.float64)
ref = e64.run(drive, t_run=50.0)
pn_pool = int(e64.pool[pn])
tab = e64.set_edge_gains([pn_pool], cap)
assert torch.allclose(e64.run(drive, t_run=50.0), ref, rtol=1e-12, atol=0), "gain 1 != no gain"
k = int(np.flatnonzero((tab["src_pool"] == int(e64.pool[orn[0]])) & (tab["tgt_pool"] == pn_pool))[0])
e64.edge_x.requires_grad_(True)
e64.run(drive, t_run=50.0)[0, pn].backward()
auto = float(e64.edge_x.grad[k])
e64.edge_x.requires_grad_(False)
with torch.no_grad():
    e64.edge_x[k] += eps
    up = float(e64.run(drive, t_run=50.0)[0, pn])
    e64.edge_x[k] -= 2 * eps
    dn = float(e64.run(drive, t_run=50.0)[0, pn])
    e64.edge_x[k] += eps
fd = (up - dn) / (2 * eps)
assert auto != 0 and abs(auto - fd) <= 1e-4 * abs(fd), (auto, fd)
print("edge grad: autograd %.8f  fd %.8f  (%d groups into the PN pool)" % (auto, fd, len(tab["n_edges"])),
      flush=True)
del e64

# 13. GPU: edge-gain gradients bit-identical across reruns; peak memory of a 1 s backward.
gsim = RateSim(0.5 / rho)
gsim.set_edge_gains([int(gsim.pool[pn])], cap)
grads = []
for _ in range(2):
    gsim.edge_x.grad = None
    gsim.edge_x.requires_grad_(True)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    gsim.run(drive, t_run=1000.0)[0, pn].backward()
    torch.cuda.synchronize()
    grads.append(gsim.edge_x.grad.clone())
assert torch.equal(grads[0], grads[1]), "edge gradients not bit-reproducible"
print("1 s forward+backward: %.0f ms, peak GPU %.2f GB" % (1000 * (time.perf_counter() - t0),
      torch.cuda.max_memory_allocated() / 1e9), flush=True)

# 14. Surrogate gradient: forward bit-identical; below threshold the gradient is the formula,
#     not relu's 0. Toy: neuron 1 driven below threshold by bias -5 (input 2 -> v1 = -3).
del gsim
t3 = RateSim(0.1, device="cpu", dtype=torch.float64, csr=toy, pool=np.arange(3))
t3.bias[1] = -5.0
plain = t3.run([(np.array([0]), 10.0)], t_run=500.0)
t3.surrogate_beta = 35.0
t3.set_trainable("bias")
sur = t3.run([(np.array([0]), 10.0)], t_run=500.0)
assert torch.equal(sur.detach(), plain), "surrogate changed the forward pass"
sur[0, 1].backward()
g1 = float(t3.bias.grad[1])
assert g1 > 0, "no gradient below threshold with the surrogate"
t3.bias.requires_grad_(False)
t3.surrogate_beta = None
t3.bias.grad = None
t3.set_trainable("bias")
t3.run([(np.array([0]), 10.0)], t_run=500.0)[0, 1].backward()
assert float(t3.bias.grad[1]) == 0.0, "plain relu should give exactly 0 below threshold"
print("surrogate: forward identical, sub-threshold grad %.4f (plain relu 0)" % g1, flush=True)
# mask excluding neuron 1 -> its sub-threshold slope is exactly 0 again
t3.bias.requires_grad_(False)
t3.bias.grad = None
t3.surrogate_beta, t3.surrogate_mask = 35.0, np.array([2])
t3.set_trainable("bias")
t3.run([(np.array([0]), 10.0)], t_run=500.0)[0, 1].backward()
assert float(t3.bias.grad[1]) == 0.0, "surrogate leaked outside its mask"

print("ok rate engine")
