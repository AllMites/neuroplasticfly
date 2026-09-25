"""Adaptation current: decays, accumulates on spikes, and is off by default."""
import os, sys
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gpu_sim as G

assert G.SFA_B_INC == 0.0, "SFA must ship default-off until Task 7"
decay_a = float(np.float32(np.exp(-G.DT / G.TAU_A)))
z8 = torch.tensor(0, dtype=torch.int8)
one = torch.tensor(1.0)

# one neuron, one batch column, no input, forced to spike by a Poisson mask
v = torch.full((1, 1), G.V_REST); g = torch.zeros((1, 1)); a = torch.zeros((1, 1))
r = torch.zeros((1, 1), dtype=torch.int8); prob = torch.zeros((1, 1), dtype=torch.float64)
no_spike = torch.zeros((1, 1), dtype=torch.bool)
spike_now = torch.ones((1, 1), dtype=torch.bool)
v, g, r, a, spk = G._body(v, g, r, a, prob, spike_now, no_spike, G.V_REST,
                          float(np.float32(np.exp(-G.DT / G.TAU_M))),
                          float(np.float32(1.0 - np.exp(-G.DT / G.TAU_M))),
                          float(np.float32(np.exp(-G.DT / G.TAU_SYN))), decay_a, 0.5,
                          G.V_TH, torch.tensor(G.V_RESET), torch.tensor(21, dtype=torch.int8), z8)
assert bool(spk[0, 0]), "forced spike did not fire"
assert abs(float(a[0, 0]) - 0.5) < 1e-6, float(a[0, 0])   # incremented, not yet decayed

# next step, no spike: decays by exactly decay_a
no_poisson = torch.zeros((1, 1), dtype=torch.bool)
v, g, r, a2, spk = G._body(v, g, r, a, prob, no_poisson, no_spike, G.V_REST,
                           float(np.float32(np.exp(-G.DT / G.TAU_M))),
                           float(np.float32(1.0 - np.exp(-G.DT / G.TAU_M))),
                           float(np.float32(np.exp(-G.DT / G.TAU_SYN))), decay_a, 0.5,
                           G.V_TH, torch.tensor(G.V_RESET), torch.tensor(21, dtype=torch.int8), z8)
assert abs(float(a2[0, 0]) - 0.5 * decay_a) < 1e-6, float(a2[0, 0])

# no_spike mask suppresses threshold crossing
v3 = torch.full((1, 1), G.V_TH + 5.0); a3 = torch.zeros((1, 1))
g3 = torch.zeros((1, 1)); r3 = torch.zeros((1, 1), dtype=torch.int8)
_, _, _, _, spk3 = G._body(v3, g3, r3, a3, prob, no_poisson, torch.ones((1, 1), dtype=torch.bool),
                           G.V_REST, float(np.float32(np.exp(-G.DT / G.TAU_M))),
                           float(np.float32(1.0 - np.exp(-G.DT / G.TAU_M))),
                           float(np.float32(np.exp(-G.DT / G.TAU_SYN))), decay_a, 0.0,
                           G.V_TH, torch.tensor(G.V_RESET), torch.tensor(21, dtype=torch.int8), z8)
assert not bool(spk3[0, 0]), "no_spike mask did not suppress the spike"
print("ok sfa: increment, decay, no_spike mask, default off")
