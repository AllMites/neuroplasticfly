"""Checks for chunk 0 (PREREGISTER_rate_chunk0_base.md + amendment 1). Plain asserts; needs the GPU.

Run: .venv/Scripts/python.exe tools/gpu_slot.py -- .venv/Scripts/python.exe rate/test_chunk0.py
"""
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _HERE)
import gpu_sim as G  # noqa: E402
from learn import condition as C  # noqa: E402
from rate.engine import RateSim, lif_fi  # noqa: E402

# 1. lif_fi is the Shiu LIF f-I curve, 0 at or below threshold.
f = lambda dv: 1000.0 / (G.T_REFR + G.TAU_M * np.log(dv / (dv - 7.0)))  # noqa: E731
v = np.array([7.5, 10, 20, 50, 200.0])
assert np.allclose(lif_fi(torch.tensor(v, dtype=torch.float64)).numpy(), f(v), rtol=1e-6)
assert (lif_fi(torch.tensor([-5.0, 0.0, 7.0, 7 + 1e-9], dtype=torch.float64)).numpy() == 0).all()
print("lif_fi: matches the f-I formula, 0 at/below threshold", flush=True)

# 2. eln8 rate W == gpu_sim's eln8 device weights / W_SYN, edge by edge (amendment 1, change 2).
rs = RateSim(1.0, regime="eln8")
G.ELN_NEGATE, G.PN_KC_GAIN = C.REGIMES["eln8"]
lif = G.GpuSim(compile_=False)
dev = lif.data.detach().cpu().double().numpy().ravel() / G.W_SYN
assert rs.W.nnz == dev.size, (rs.W.nnz, dev.size)
assert np.array_equal(rs.W.indptr, lif.net.W_indptr) and np.array_equal(rs.W.indices, lif.net.W_indices)
err = np.abs(rs.W.data - dev).max()
assert err < 1e-4, err
n_neg = int((np.asarray(RateSim(1.0).W.data) > 0).sum() - (rs.W.data > 0).sum())
print("eln8 W: %d edges equal to gpu_sim within float32 rounding (max |diff| %.1e syn); %d edges flipped to negative"
      % (dev.size, err, n_neg), flush=True)
del lif
G.ELN_NEGATE, G.PN_KC_GAIN = C.REGIMES["stock"]

# 3. lif family at rest: silent (bias 0, below threshold everywhere).
rs.transfer, rs.w_scale = "lif", G.W_SYN * G.TAU_SYN / 1000.0
r = rs.run([(np.zeros(0, np.int64), 0.0)], 50.0)
assert float(r.max()) == 0.0, float(r.max())
print("lif family: silent at rest", flush=True)

print("ok chunk0")
