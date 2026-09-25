"""silence= kwarg: empty set is a no-op; silenced Fox never spikes under sugar."""
import os, sys
import numpy as np
import torch
_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _HERE); sys.path.insert(0, os.path.join(_HERE, "flywatch"))
import gpu_sim as G
from learn import condition as C

meta = np.load(os.path.join(_HERE, "data", "neuron_meta.npz"), allow_pickle=False)
# Fox = CB0525. The CB* FlyWire labels live in `cell_type`; `hemibrain_type`
# reads "none" for every one of them in this matrix (checked 2026-09-23).
ct = meta["cell_type"].astype(str)
fox = np.flatnonzero(ct == "CB0525").astype(np.int64)
assert len(fox) == 2, fox

sim = G.GpuSim()
drive = C.to_prob(C.SUGAR, np.full(len(C.SUGAR), C.GRN_HZ, np.float32))

# rng="counter" (the default) derives the Poisson stream from seed + step index,
# so two seed-0 calls are bit-identical and an empty lesion must not move a count.
c0 = sim.run_batch([drive], [0], t_run=C.T_RUN)[0].cpu().numpy()
c1 = sim.run_batch([drive], [0], t_run=C.T_RUN, silence=np.empty(0, np.int64))[0].cpu().numpy()
assert np.array_equal(c0, c1), "empty silence changed counts"

# If Fox is already silent unsilenced the next assert is vacuous, so log the
# unsilenced counts either way - a vacuous pass must be visible in the log.
c2 = sim.run_batch([drive], [0], t_run=C.T_RUN, silence=fox)[0].cpu().numpy()
assert c2[fox].sum() == 0, c2[fox]

# torch tensor already on device: same path, same result.
c3 = sim.run_batch([drive], [0], t_run=C.T_RUN,
                   silence=torch.as_tensor(fox, device=sim.device))[0].cpu().numpy()
assert np.array_equal(c2, c3), "tensor silence differs from ndarray silence"

# Overlap with the Poisson-driven set is refused, not silently half-applied.
try:
    sim.run_batch([drive], [0], t_run=C.T_RUN, silence=C.SUGAR[:1])
    raise SystemExit("overlap not refused")
except AssertionError:
    pass

print("ok silence: empty no-op, Fox %s silent under sugar (unsilenced counts %s), "
      "tensor == ndarray, overlap refused"
      % (fox.tolist(), c0[fox].tolist()))
