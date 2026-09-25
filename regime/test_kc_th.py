"""KC threshold offset: off by default, Kenyon-cell only, and the sign points the right way."""
import os, sys
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gpu_sim as G

assert G.KC_V_TH_DELTA == 0.0, "KC threshold offset must ship default-off"

decay_m = float(np.float32(np.exp(-G.DT / G.TAU_M)))
gain = float(np.float32(1.0 - np.exp(-G.DT / G.TAU_M)))
decay_s = float(np.float32(np.exp(-G.DT / G.TAU_SYN)))
decay_a = float(np.float32(np.exp(-G.DT / G.TAU_A)))
z8 = torch.tensor(0, dtype=torch.int8)
v_reset = torch.tensor(G.V_RESET)
refr_m1 = torch.tensor(21, dtype=torch.int8)


def body(v, v_th):
    """One _body step on a [n, 1] state with no input and no Poisson spike."""
    n = v.shape[0]
    g = torch.zeros((n, 1)); a = torch.zeros((n, 1))
    r = torch.zeros((n, 1), dtype=torch.int8)
    prob = torch.zeros((n, 1), dtype=torch.float64)
    no_spike = torch.zeros((n, 1), dtype=torch.bool)
    no_poisson = torch.zeros((n, 1), dtype=torch.bool)
    return G._body(v, g, r, a, prob, no_poisson, no_spike, G.V_REST, decay_m, gain,
                   decay_s, decay_a, 0.0, v_th, v_reset, refr_m1, z8)


# 1. Direction and broadcasting in one check. Two neurons, one at the stock threshold
#    and one offset 3 mV lower. Both sit at a voltage between the two thresholds, so
#    only the offset cell may fire. Decay pulls v toward V_REST each step, so start
#    above the midpoint by enough that vn still lands between the two thresholds.
v_th_vec = torch.tensor([[G.V_TH], [G.V_TH - 3.0]])
start = G.V_TH - 1.0                      # -46: below stock -45, above offset -48
v = torch.full((2, 1), start)
_, _, _, _, spk = body(v, v_th_vec)
vn = G.V_REST + (start - G.V_REST) * decay_m
assert G.V_TH - 3.0 < vn <= G.V_TH, "test setup: vn %.3f not between the thresholds" % vn
assert not bool(spk[0, 0]), "stock-threshold neuron fired below V_TH"
assert bool(spk[1, 0]), "offset neuron did not fire; NEGATIVE delta must mean MORE excitable"

# 2. A [n, 1] tensor of the stock threshold must behave exactly like the float.
v = torch.full((3, 1), G.V_TH + 2.0)
_, _, _, _, spk_scalar = body(v.clone(), G.V_TH)
_, _, _, _, spk_tensor = body(v.clone(), torch.full((3, 1), float(G.V_TH)))
assert torch.equal(spk_scalar, spk_tensor), "broadcast v_th differs from the scalar path"

# 3. The offset reaches Kenyon cells and nothing else. Build the vector the way
#    run_batch does rather than trusting the index alone.
meta = np.load(os.path.join(G._HERE, "data", "neuron_meta.npz"), allow_pickle=False)
kc = np.flatnonzero(meta["cell_class"].astype(str) == "Kenyon_Cell")
assert len(kc) > 0, "no Kenyon cells in neuron_meta.npz"
n = len(meta["cell_class"])
delta = -3.0
v_th_full = torch.full((n, 1), float(G.V_TH))
v_th_full[torch.as_tensor(kc.astype(np.int64))] = G.V_TH + delta
assert torch.allclose(v_th_full[kc], torch.tensor(G.V_TH + delta)), "KC rows not offset"
mask = np.ones(n, bool); mask[kc] = False
assert torch.allclose(v_th_full[torch.as_tensor(np.flatnonzero(mask))],
                      torch.tensor(float(G.V_TH))), "offset leaked to non-KC neurons"

# 4. The Shiu constants are untouched by any of this.
assert (G.DT, G.V_REST, G.V_RESET, G.V_TH, G.TAU_M, G.TAU_SYN, G.T_REFR, G.DELAY,
        G.W_SYN) == (0.1, -52.0, -52.0, -45.0, 20.0, 5.0, 2.2, 1.8, 0.275), \
    "a Shiu constant changed; a port that changes one of these is not a port"

print("ok kc_th: default off, direction, broadcast-equals-scalar, KC-only, Shiu frozen")

# 5. PN_KC_GAIN: off by default, and when on it touches ALPN->KC edges and nothing else.
assert G.PN_KC_GAIN == 1.0, "PN->KC gain must ship default-off"
_base = G.GpuSim(compile_=False)
_d0 = _base.data.clone()
G.PN_KC_GAIN = 2.0
_gain = G.GpuSim(compile_=False)
G.PN_KC_GAIN = 1.0
_changed = (_gain.data != _d0).nonzero().flatten().cpu().numpy()
_cc = meta["cell_class"].astype(str)
_ip, _ix = _base.net.W_indptr, _base.net.W_indices
_pre = np.repeat(np.arange(len(_ip) - 1), np.diff(_ip))
assert _changed.size, "PN_KC_GAIN = 2.0 changed no edges"
assert (_cc[_pre[_changed]] == "ALPN").all(), "gain touched a non-ALPN presynaptic row"
assert (_cc[_ix[_changed]] == "Kenyon_Cell").all(), "gain touched a non-KC target"
_expect = np.flatnonzero((_cc[_pre] == "ALPN") & (_cc[_ix] == "Kenyon_Cell"))
assert _changed.size == _expect.size, \
    "gain hit %d edges, expected %d ALPN->KC" % (_changed.size, _expect.size)
print("ok pn_kc_gain: default off, ALPN->KC only, %d edges" % _changed.size)

# 6. PN_KC_GAIN is construction-time. Changing it on a live sim must fail loudly rather
#    than silently returning the baseline -- the trap is that KC_V_TH_DELTA, its
#    documented companion sitting next to it in the constants block, IS read per run.
_sim = G.GpuSim(compile_=False)
G.PN_KC_GAIN = 8.0
try:
    _sim.run_batch([(np.zeros(0, np.int64), np.zeros(0, np.float64))], seeds=[0], t_run=1.0)
    raise SystemExit("FAIL: stale PN_KC_GAIN did not raise; it silently ran the baseline")
except AssertionError as _e:
    assert "baked into" in str(_e), str(_e)
finally:
    G.PN_KC_GAIN = 1.0
print("ok pn_kc_gain_staleness: changing it on a live sim raises instead of misleading")
