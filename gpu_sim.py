"""Batched GPU port of the flypoke LIF integrator. flypoke stays the oracle.

WHAT IS REAL HERE: the dynamics and every constant are Shiu et al. 2024 as
implemented by `flypoke.sim.run_trial`; this file re-expresses the same loop over
a batch dimension so B chess positions integrate at once on the RTX 5070 Ti.
AUTHORED: nothing about the model. Only the memory layout, the batching, and the
counter-based RNG used in throughput mode.

WHY: measured 2026-09-15, the CPU bottleneck is not the connectome (~8M sparse
scatter-adds per sim) but the dense per-step state update - 139,248 neurons x
3,000 steps x ~10 array passes = ~16.7 GB of memory traffic per 300 ms sim. That
is bandwidth-bound, which is what a GPU is for, and batching also amortises the
sparse scatter across positions.

Ordering inside a step is load-bearing and mirrors `run_trial` line for line:
  1. deliver spikes that left the presynaptic neuron `delay_steps` ago
  2. active = not refractory
  3. v = where(active, v_rest + (v - v_rest)*decay_m + g*gain, v)
  4. g *= decay_s                       (after the v update, before the reset)
  5. fired = (v > v_th) & active & ~is_stim
  6. append Poisson spikes on stimulated neurons (no refractory gate)
  7. v[spk] = v_reset; g[spk] = 0; refractory restart for spk

DIFFERENCES FROM THE CPU PATH, all measured in `verify_gpu.py` and written up in
`.claude/PRPs/reports/gpu-sim-port.md`:
  * float32 reduction order in the synaptic scatter differs from scipy's
    `W[arriving].sum(axis=0)`, so the two engines are NOT bit-identical even on
    an identical random stream. They diverge in the last ulp and the chaotic
    dynamics amplify that into different individual spike times.
  * `rng="counter"` (the throughput default) uses a counter-based hash keyed by
    (seed, step, neuron) instead of numpy's PCG64. It is reproducible and
    batch-invariant; it is not flypoke's stream.
  * `rng="numpy"` reproduces flypoke's PCG64 stream exactly - measured: every
    stimulated neuron's spike count is identical to flypoke's on all 32 checked
    positions, 2,567,198 Poisson spikes either way. The cost is materialising
    the whole stimulus spike train, so it is the verifier's mode, not
    production's. It is what isolates the float32 difference above from the
    random-stream difference.
  * the synaptic scatter runs under `use_deterministic_algorithms` (see
    `_deliver`). Without it the same batch run twice gave different spike
    counts, which `verify_gpu.py` caught.

Refractory state is stored as an int8 countdown rather than flypoke's int64
`refr_until` absolute step. Identical semantics (a neuron that fires at step s is
inactive for s+1..s+21 and active again at s+22), one quarter of the traffic.
"""
import os
import time
from collections import deque

import numpy as np
import torch

BRAIN = os.environ.get("FLYCHESS_BRAIN", os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "data", "brain_gpu.npz"))

# encode.py reads its cache path at import time, and the container default
# (/app/data) does not exist on the host. Point it at the same frozen file.
_HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("FLYCHESS_VISUAL_MAP",
                      os.path.join(_HERE, "data", "visual_map.npz"))
os.environ.setdefault("FLYCHESS_LIVE_MASK",
                      os.path.join(_HERE, "data", "live_mask.npz"))

# Shiu et al. 2024 constants, copied from flypoke.sim.Params. Not tunable: a port
# that changes one of these is not a port.
DT = 0.1            # ms
V_REST = -52.0      # mV
V_RESET = -52.0     # mV
V_TH = -45.0        # mV
TAU_M = 20.0        # ms
TAU_SYN = 5.0       # ms
T_REFR = 2.2        # ms
DELAY = 1.8         # ms
W_SYN = 0.275       # mV per synapse

# AUTHORED, not Shiu. Spike-frequency adaptation, a graded APL and homeostatic
# normalisation, added 2026-09-19 to stop brain-wide runaway. All three default
# OFF (SFA_B_INC = 0.0, APL_GRADED = False, NORM_TOTAL_TARGET = 0.0) so that
# every result before that date reproduces exactly; Task 7 of the regime plan
# flips them. APL_DELAYED is not a mechanism switch - it selects between the
# biological form of graded APL and a zero-delay variant kept for comparison.
# See docs/superpowers/specs/2026-09-19-regime-stability-design.md.
TAU_A = 100.0       # ms, adaptation decay
# AUTHORED, not Shiu (PREREGISTER_slow.md). A slow excitatory synaptic channel `h`:
# SLOW_FRAC of every EXCITATORY weight whose postsynaptic cell_type starts with one of
# SLOW_TYPES is delivered into h instead of g. h decays at TAU_SLOW and, unlike g, is NOT
# reset when its neuron spikes -- the Shiu port zeroes g on every spike, so a slow channel
# that reset too would be erased by the very activity it is meant to sustain.
# CHARGE-MATCHED: the slow share is scaled by TAU_SYN / TAU_SLOW, so one spike delivers
# the same total input with the channel on or off and only its time course changes.
# Without that, SLOW_FRAC = 0.5 at 100 ms would inject ~10x more net excitation and any
# persistence found would just be "more drive". Phenomenological: fly excitation is
# mostly cholinergic, so this is "NMDA-like", never "NMDA". Off by default (0.0); the
# default path is bit-identical to before it existed (regime/test_slow.py).
TAU_SLOW = 100.0    # ms
SLOW_FRAC = 0.0
SLOW_TYPES = ()     # cell_type prefixes, e.g. ("EPG", "PEN", "PEG")
SFA_B_INC = 0.0     # mV-equivalent added to the adaptation current per spike
APL_GRADED = False  # replace APL's spiking output with a graded rate
APL_SCALE = 7.0     # mV above rest at which graded APL output saturates
APL_MAX_HZ = 450.0  # spiking APL's refractory ceiling; act=1 delivers the same
                    # charge per second that a spiking APL at this rate would
APL_DELAYED = True  # route graded APL's output through the same DELAY (1.8 ms,
                    # 18 steps) the spiking path uses. Default True because that
                    # is the biology and the mechanism being replaced; False
                    # gives the zero-delay (0.1 ms) variant, kept reachable only
                    # so the two can be compared. Loop delay is the dominant
                    # parameter for whether negative feedback damps, divides or
                    # oscillates, and damping runaway is this mechanism's job.
APL_DIVISIVE = False # AUTHORED. Selects the KIND of inhibition graded APL
                    # delivers, not the amount. False (default) keeps the
                    # SUBTRACTIVE form: APL's weights are scatter-ADDED into g,
                    # so every target loses the same offset regardless of how
                    # much drive it already carries. True switches to the
                    # DIVISIVE form g_j <- g_j / (1 + inh_j / APL_DIV_SCALE),
                    # where inh_j is the magnitude of APL's summed inhibitory
                    # input to neuron j at that step - the standard account of
                    # Kenyon-cell sparse coding, in which APL scales a cell's
                    # drive rather than offsetting it, so a cell receiving twice
                    # the drive of its neighbour stays ahead of it afterwards.
                    # Only reachable with APL_GRADED = True; the spiking path
                    # has no per-step inhibition magnitude to divide by.
APL_DIV_SCALE = 11.75 # mV of per-step APL inhibition at which the divisor is 2.
                    # AUTHORED. Units are the same as `inh_j`: summed |w| over
                    # APL's synapses onto j, times the APL_MAX_HZ rate-
                    # equivalence factor, per 0.1 ms step. SMALLER is STRONGER.
                    # 11.75 = 50 x 0.235, where 0.235 mV/step is the MEASURED
                    # median inhibition a Kenyon cell receives from a fully
                    # activated (act=1) graded APL on this matrix. The 50x is
                    # not cosmetic: g decays by exp(-DT/TAU_SYN) = 0.9802 a
                    # step, so the steady-state drive under a constant divisor
                    # r is proportional to 1/(1 - 0.9802 + r), and r = 0.02 -
                    # i.e. inh/S = 1/50 - is exactly the point where a
                    # saturated APL halves a median KC's steady-state drive.
                    # So the default is "one unit of biologically-plausible
                    # divisive gain control", and the sweep moves off it.
KC_V_TH_DELTA = 0.0 # AUTHORED. mV added to the spike threshold of Kenyon cells only.
                    # NEGATIVE is MORE EXCITABLE: V_TH is -45 and V_REST is -52, so
                    # -3.0 puts a KC's threshold at -48, 4 mV above rest instead of 7.
                    # 0.0 = off, and the scalar V_TH path is used unchanged, so every
                    # prior result reproduces bit for bit.
                    #
                    # This exists because ELN_NEGATE separates the odour code (PN
                    # Jaccard 0.961 -> 0.057) but starves the mushroom body: 2.3% of KCs
                    # active at 800 Hz ORN, 2 of 8 channels silent. ORN drive scales
                    # recruitment roughly logarithmically and APL gain cannot reach the
                    # merge at all - both measured out in
                    # docs/superpowers/option-a/kc_sparsity_gate_2026-09-20.md. That
                    # leaves the Kenyon cell itself as the only place left to buy
                    # recruitment, and there is room to pay: ELN_NEGATE's odour Jaccard
                    # is 0.016 against a 0.3 target.
                    #
                    # V_TH itself is a Shiu constant and is not touched. Real Kenyon
                    # cells are high-threshold sparse-firing cells while the port gives
                    # every neuron in the brain the same V_TH, so a KC-specific offset
                    # is arguably more faithful - but it is AUTHORED either way and is
                    # disclosed alongside the fly-hash graft and the VNC stand-in.
TYPE_W_SCALE = {}   # AUTHORED. {cell_type: multiplier} on every OUTGOING device weight of
                    # that type ({} = off, default path untouched). -1.0 flips a sign,
                    # 0.0 removes the type's chemical output. Exists for transmitters
                    # whose sign is receptor-dependent (glutamate: GluCl vs iGluR), first
                    # ExR6 (PREREGISTER_exr6.md). Baked at construction like PN_KC_GAIN.
PN_KC_GAIN = 1.0    # AUTHORED. Multiplier on ALPN -> Kenyon-cell synaptic weights.
                    # 1.0 = off. The companion to KC_V_TH_DELTA and the one to prefer:
                    # the threshold lever only reaches the 5-10% band at delta -7.0,
                    # which puts V_TH exactly on V_REST and leaves a Kenyon cell with no
                    # threshold margin at all (at -7.5 the brain free-runs). Raising the
                    # drive into the cell buys the same recruitment without a degenerate
                    # cell, and PN->KC synaptic strength is a real parameter that the
                    # connectome gives as a count, not a conductance.
NORM_TOTAL_TARGET = 0.0 # mV of summed incoming TOTAL input MAGNITUDE per neuron
                        # (excitatory plus |inhibitory|); 0.0 = off. One factor per
                        # postsynaptic neuron scales ALL of its incoming edges, so the
                        # heavy convergence tail flattens while each neuron's E/I ratio
                        # is preserved exactly.
                        # AUTHORED. 29.15 = 106 synapses times W_SYN. 106 is the
                        # brain-wide median of summed incoming |weight| measured on this
                        # matrix (p95 1016, p99 2878 - the same 27x tail P0.2 found in
                        # excitation alone), so the median neuron is unchanged and only
                        # the tail moves. Not a published figure.
                        # An earlier version normalised EXCITATION only, using 13.75 mV
                        # = 50 synapses * W_SYN. That derivation is void: scaling
                        # excitation alone does not flatten the tail, it inverts E/I.
                        # Measured, 41% of neurons carried more summed inhibition
                        # (median -12.9 mV, p5 -120.7 mV) than the whole 13.75 mV
                        # excitatory budget, and the brain went silent one hop from the
                        # driven ORNs at every rate.
ELN_NEGATE = False  # AUTHORED BRIDGE, not physiology. Negate the outgoing weights (on
                    # the DEVICE copy only; self.net.W_data is untouched) of
                    # the 44 antennal-lobe excitatory local neurons whose cholinergic
                    # identity is immuno-confirmed (known_nt = acetylcholine, Shang et
                    # al. 2007; FlyWire v783 top_nt is noise for them, median conf
                    # 0.31). Their chemical broadcast onto ALPNs is what merges every
                    # odour into one attractor; in the fly that lateral excitation is
                    # electrical (Yaksi & Wilson 2010), which this LIF cannot express
                    # yet. Negation is the only measured working point (ALPN Jaccard
                    # 0.94 -> 0.20 on two 8-glomerulus odours) and is a HACK: it adds
                    # inhibition, it does not model a gap junction. Superseded by
                    # GAP_COUPLE when phase 2 of
                    # .claude/PRPs/prds/eln-electrical-coupling.prd.md lands.
                    # Selection comes from data/nt_conf.npz (regime/nt_conf.py).
GAP_COUPLE = 0.0    # AUTHORED. Electrical (gap-junction) coupling from the 44
                    # immuno-confirmed cholinergic AL eLNs onto ALPNs, replacing the
                    # chemical delivery of those 3,802 edges. 0.0 = off (chemical path
                    # unchanged, bit-identical to every earlier result). Units: per-step
                    # fraction per synapse; each step every coupled pair (i eLN, j PN)
                    # exchanges dv_j += k (v_i - v_j), dv_i += k (v_j - v_i) with
                    # k = GAP_COUPLE * synapse_count, applied only to non-refractory
                    # ends, straight into v (NOT into g: a per-step current into g is
                    # the 1/DT rate-equivalence trap _deliver_apl documents). Summed
                    # per-neuron coefficient is asserted <= 1 at init: that keeps the
                    # explicit step stable and makes the update a convex combination,
                    # so coupling alone can never lift a PN above the highest coupled
                    # eLN voltage (sub-threshold by construction). Coefficients come
                    # from synapse COUNTS in the exported matrix, so W_SYN, homeostatic
                    # normalisation and ELN_NEGATE do not touch them. The stability
                    # bound is numerical only; physiological steady-state coupling
                    # coefficients (0.05-0.2) correspond to a summed per-step
                    # coefficient S ~ 0.0005-0.001 (c = S / (S + DT/TAU_M)), far below
                    # 1. Per-target summed counts on this matrix: median 80, p90 399,
                    # max 1958 (24x median), so one scalar over-couples the heaviest
                    # PNs long before it moves the median; the working point and
                    # per-target normalisation are phase 3 questions. The kernel
                    # decays v by decay_m in the same step, so the delivered
                    # coefficient is k*decay_m (~0.995k). Physiology:
                    # Yaksi & Wilson 2010 and Huang et al. 2010 (shakB-dependent,
                    # transmits both polarities). Supersedes ELN_NEGATE. See
                    # .claude/PRPs/prds/eln-electrical-coupling.prd.md phase 2.
GAP_NORM = False    # AUTHORED. With GAP_COUPLE > 0, divide each gap coefficient by
                    # its TARGET's summed synapse count over the gap edges, so every
                    # coupled PN carries a summed per-step coefficient of exactly
                    # GAP_COUPLE (which then reads as S; physiological S ~ 0.0005-0.001).
                    # Motivation: per-target summed counts run median 80 / max 1958,
                    # so one scalar over-couples the heaviest PNs 24x before it moves
                    # the median (alln-findings s8). The eLN (source) side is not
                    # normalised; the per-neuron <= 1 stability assert still applies.

_brain = None


# ---------------------------------------------------------------- host network
class Brain:
    """Host-side stand-in for `flypoke.data.Network`, arrays only.

    `export_brain.py` resolved every selector string in the container with the
    real annotation table, so `select()` is a lookup, not a reimplementation of
    flypoke's query language. Anything not exported raises instead of guessing.
    """

    def __init__(self, path=BRAIN):
        blob = np.load(path, allow_pickle=False)
        self.n = int(blob["n_neurons"])
        self.min_syn = int(blob["min_syn"])
        self.W_indptr = blob["W_indptr"]
        self.W_indices = blob["W_indices"]
        self.W_data = blob["W_data"]
        # corrected v783 build (2026-09-19) records its own edge count; the
        # 2026-09-16 bad build (3,532,411 edges) has no such field.
        self.n_edges = int(blob["n_edges"]) if "n_edges" in blob.files else 3532411
        self.pool_group_id = blob["pool_group_id"]
        self.pool_sizes = blob["pool_sizes"]
        self.pool_names = [str(x) for x in blob["pool_names"]]
        names = [str(x) for x in blob["selector_names"]]
        self._sel = {s: blob["selector_%04d" % k] for k, s in enumerate(names)}

    def select(self, selector):
        if selector not in self._sel:
            raise KeyError("selector %r was not exported; add it to "
                           "export_brain.SELECTORS and re-run it" % selector)
        return self._sel[selector]


def _install_stubs():
    """Let `encode` and `reservoir` import on a host with no flypoke/pandas.

    Only the pieces those two modules touch at import time or inside the drive
    builder are provided. `flypoke.sim.run` deliberately raises: if anything on
    the host path ever calls it, that is a bug worth crashing on, not a silent
    fallback to the CPU engine.
    """
    import sys
    import types

    try:                                     # inside the container, use the real thing
        import flypoke.sim                   # noqa: F401
    except ImportError:
        pkg = types.ModuleType("flypoke")
        sim = types.ModuleType("flypoke.sim")

        class Stimulus:
            def __init__(self, indices, rate):
                self.indices, self.rate = indices, rate

        class Params:
            def __init__(self, **kw):
                self.__dict__.update(kw)

        def _no(*a, **k):
            raise RuntimeError("flypoke is not installed on the host; the GPU "
                               "path must not fall back to the CPU engine")

        sim.Stimulus, sim.Params, sim.run, sim.run_trial = Stimulus, Params, _no, _no
        pkg.sim = sim
        sys.modules["flypoke"], sys.modules["flypoke.sim"] = pkg, sim

    if "behaviors" not in sys.modules:
        try:
            import behaviors                 # noqa: F401
        except ImportError:
            beh = types.ModuleType("behaviors")
            beh.net = lambda: brain()        # late-bound: the .npz loads on first use
            sys.modules["behaviors"] = beh


# Installed at import time, not inside brain(): `import reservoir` anywhere -
# verify_gpu, precompute_gpu, a REPL - has to work without the caller knowing to
# warm the network first, and a missing stub shows up as a confusing
# ModuleNotFoundError halfway through a long run.


_install_stubs()


def brain():
    global _brain
    if _brain is None:
        _brain = Brain()
        import reservoir as R
        # Hand reservoir the pooling table the container computed, so `pool`,
        # `views` and `dn` run their real code on the real groups without pandas.
        R._pool = (_brain.pool_group_id, _brain.pool_sizes, _brain.pool_names)
    return _brain


def drive_of(board, n=None):
    """flypoke's (stim_idx, stim_prob) for one board, in flypoke's exact order.

    `encode.stimuli` groups neurons by rate rounded to 0.1 Hz and emits one
    Stimulus per group sorted by rate; `run_trial` then concatenates them in that
    order. The order is irrelevant to the dynamics but fixes which Poisson draw
    lands on which neuron, so reproducing flypoke's RNG stream requires it.
    """
    import encode
    n = n or brain()
    stims = encode.stimuli(board, n)
    if not stims:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)
    idx = np.concatenate([s.indices for s in stims])
    prob = np.concatenate([np.full(len(s.indices), s.rate * DT / 1000.0)
                           for s in stims])
    return idx, prob


def drive_of_selector(selector, rate, n=None):
    """The flypet behaviour form: one selector driven at one rate."""
    n = n or brain()
    if selector is None or rate <= 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)
    idx = np.asarray(n.select(selector), dtype=np.int64)
    return idx, np.full(len(idx), rate * DT / 1000.0)


# ---------------------------------------------------------------- counter RNG
# A 32-bit counter-based hash of (seed, step, neuron). Two murmur3 finalisers
# over a mixed counter. It replaces numpy's PCG64 in throughput mode for two
# reasons: a shared torch.Generator would make a position's Poisson stream depend
# on its column and on the batch size (no batch-invariance), and a counter-based
# draw can be computed inside the fused kernel from the neuron's own index, which
# is what removes ~35 kernel launches and a device sync from every step.
# `verify_gpu.py` measures that it gives the same rate distribution as PCG64.
def _w32(x):
    """Wrap a python int into torch's signed int32 range (two's complement)."""
    x &= 0xFFFFFFFF
    return x - (1 << 32) if x >= (1 << 31) else x


_M1, _M2 = _w32(0x85EBCA6B), _w32(0xC2B2AE35)
_S1, _S2 = 0x9E3779B1, _w32(0x85EBCA77)


def _srl32(x, k):
    """Logical (not arithmetic) right shift on torch's signed int32."""
    return torch.bitwise_right_shift(x, k) & ((1 << (32 - k)) - 1)


def _mix32(h):
    h = h ^ _srl32(h, 16)
    h = h * _M1
    h = h ^ _srl32(h, 13)
    h = h * _M2
    return h ^ _srl32(h, 16)


def _uniform(seed32, step_const, neuron):
    """[0,1) float32 from (seed[1,B], wrapped step scalar, neuron[N,1])."""
    h = seed32 + step_const + neuron * _S2
    h = _mix32(_mix32(h))
    return _srl32(h, 8).to(torch.float32) * (1.0 / (1 << 24))


def _numpy_poisson(idx_list, prob_list, seeds, n_steps, S, device):
    """flypoke's exact PCG64 stream, materialised as [n_steps, S, B] bool.

    `run_trial` calls `rng.random(len(stim_idx))` once per step from a generator
    seeded with `seed`, so replaying the same calls in the same order reproduces
    the stimulus spike train bit for bit. Costs n_steps x n_stim float64 draws
    per position, which is why this mode is for verification, not production.
    """
    B = len(idx_list)
    out = np.zeros((n_steps, S, B), dtype=bool)
    for b in range(B):
        k = len(idx_list[b])
        if not k:
            continue
        rng = np.random.default_rng(int(seeds[b]))
        p = prob_list[b]
        col = np.empty((n_steps, k), dtype=bool)
        for step in range(n_steps):
            col[step] = rng.random(k) < p
        out[:, :k, b] = col
    return torch.from_numpy(out).to(device)


def bias_mv_for_hz(hz):
    """run_batch `bias` (mV) that makes an ISOLATED cell (no synaptic input) fire at
    `hz`: inverts the LIF rate 1000 / (T_REFR + TAU_M ln(I / (I - (V_TH - V_REST)))).
    For putting a drive's Hz and a bias on one scale; in the network the realised rate
    also carries the cell's synaptic input, which is the point of `bias`. Continuous-
    time, so off by up to one DT step per interval."""
    hz = np.asarray(hz, dtype=np.float64)
    isi = 1000.0 / hz - T_REFR
    assert np.all(isi > 0), "rate above the refractory ceiling %.0f Hz" % (1000.0 / T_REFR)
    return (V_TH - V_REST) / (1.0 - np.exp(-isi / TAU_M))


# ---------------------------------------------------------------- the integrator
def _body(v, g, r, a, prob, poisson, no_spike, v_rest, decay_m, gain, decay_s,
          decay_a, b_inc, v_th, v_reset, refr_m1, zero_i8, h=None, decay_h=None,
          bias=None):
    """Steps 2-7 of run_trial over the whole [N, B] state, in one pass.

    `prob` is the per-step Poisson probability (rate * dt / 1000) and doubles as
    flypoke's `is_stim`: every stimulated neuron has a strictly positive rate, so
    `prob > 0` is exactly the `is_stim` mask and no second array is needed.

    `a` is the AUTHORED adaptation current, subtracted from the synaptic drive
    and incremented on each spike. With b_inc = 0 it stays zero for every step
    and `(g - a)` is `g`, so the arithmetic is unchanged from the Shiu port.
    `no_spike` is the AUTHORED mask for neurons whose output is delivered by
    some other mechanism (graded APL); it only gates threshold crossing.

    `h` is the AUTHORED slow channel (SLOW_FRAC). None on the default path, which is
    then exactly the arithmetic it always was; given, it adds to the drive, decays at
    decay_h, is NOT reset on spike, and is returned as a sixth value.

    `bias` is the AUTHORED additive input (run_batch's `bias`): mV, [N, 1] or [N, B],
    summed into the drive next to g, so synaptic input is kept (unlike a Poisson
    stimulus, whose `prob > 0` masks threshold crossing). None = the branches below,
    untouched.
    """
    active = r == zero_i8
    if bias is not None:
        drive = (g - a if h is None else g + h - a) + bias
        vn = torch.where(active, v_rest + (v - v_rest) * decay_m + drive * gain, v)
        if h is not None:
            h = h * decay_h
    elif h is None:
        vn = torch.where(active, v_rest + (v - v_rest) * decay_m + (g - a) * gain, v)
    else:
        vn = torch.where(active, v_rest + (v - v_rest) * decay_m + (g + h - a) * gain, v)
        h = h * decay_h
    g = g * decay_s
    a = a * decay_a
    fired = (vn > v_th) & active & (prob <= 0.0) & ~no_spike
    spk = fired | poisson
    v = torch.where(spk, v_reset, vn)
    g = torch.where(spk, torch.zeros((), dtype=g.dtype, device=g.device), g)
    a = torch.where(spk, a + b_inc, a)
    r = torch.where(spk, refr_m1, torch.clamp(r - 1, min=0))
    if h is None:
        return v, g, r, a, spk
    return v, g, r, a, spk, h


def _step_counter(v, g, r, a, prob, seed32, step_const, neuron, no_spike, v_rest,
                  decay_m, gain, decay_s, decay_a, b_inc, v_th, v_reset, refr_m1,
                  zero_i8, h=None, decay_h=None, bias=None):
    """Throughput path: the Poisson draw is computed in-register, not stored."""
    poisson = _uniform(seed32, step_const, neuron) < prob
    return _body(v, g, r, a, prob, poisson, no_spike, v_rest, decay_m, gain,
                 decay_s, decay_a, b_inc, v_th, v_reset, refr_m1, zero_i8, h, decay_h,
                 bias)


def _step_mask(v, g, r, a, prob, poisson, no_spike, v_rest, decay_m, gain,
               decay_s, decay_a, b_inc, v_th, v_reset, refr_m1, zero_i8, h=None,
               decay_h=None, bias=None):
    """Verification path: the Poisson mask came from numpy's PCG64 stream."""
    return _body(v, g, r, a, prob, poisson, no_spike, v_rest, decay_m, gain,
                 decay_s, decay_a, b_inc, v_th, v_reset, refr_m1, zero_i8, h, decay_h,
                 bias)


_compiled = {}


def _kernel(name, fn, compile_=True):
    if not compile_:
        return fn
    if name not in _compiled:
        try:
            _compiled[name] = torch.compile(fn, dynamic=False)
        except Exception as exc:              # no triton -> say so, run eager
            print("torch.compile unavailable (%s); running eager" % exc)
            _compiled[name] = fn
    return _compiled[name]


def _eln_idx(net):
    """The 44 immuno-confirmed cholinergic AL local neurons, from data/nt_conf.npz.

    cell_class == ALLN, known_nt == acetylcholine, and every out-edge positive in
    W_data (sign is per neuron; the assert below is the guard). Count and cell-type
    histogram are asserted so a rebuilt nt_conf.npz cannot silently change the set.
    """
    path = os.path.join(_HERE, "data", "nt_conf.npz")
    assert os.path.exists(path), \
        "data/nt_conf.npz missing; build it with regime/nt_conf.py"
    nt = np.load(path, allow_pickle=False)
    meta = np.load(os.path.join(_HERE, "data", "neuron_meta.npz"), allow_pickle=False)
    cc = meta["cell_class"].astype(str)
    ct = meta["cell_type"].astype(str)
    known = nt["known_nt"].astype(str)
    assert len(known) == net.n, "nt_conf rows %d != brain neurons %d" % (len(known), net.n)
    idx = np.flatnonzero((cc == "ALLN") & (known == "acetylcholine")).astype(np.int64)
    # Set identity first: a rebuilt nt_conf.npz must fail with "expected 44, got N"
    # rather than with a per-neuron sign message from a set that is already wrong.
    assert len(idx) == 44, "expected 44 known eLNs, got %d" % len(idx)
    hist = {t: int((ct[idx] == t).sum()) for t in ("lLN1_bc", "lLN2X03", "lLN2T_b", "lLN2T_c")}
    assert hist == {"lLN1_bc": 30, "lLN2X03": 6, "lLN2T_b": 4, "lLN2T_c": 4}, hist
    ip, w = net.W_indptr, net.W_data
    for i in idx:
        assert (w[ip[i]:ip[i + 1]] > 0).all(), "eLN %d has a non-positive out-edge" % i
    return idx


def _gap_edges(net):
    """Flat CSR offsets of the 44 known eLNs' edges onto ALPNs, with pre, post, count.

    Returns (offsets, pre, post, count) as int64/int64/int64/float32 host arrays.
    3,802 edges on the v783 export; asserted so a re-export cannot change the set
    silently. Counts are the raw W_data synapse counts (positive by _eln_idx's guard).
    """
    eln = _eln_idx(net)
    meta = np.load(os.path.join(_HERE, "data", "neuron_meta.npz"), allow_pickle=False)
    is_pn = meta["cell_class"].astype(str) == "ALPN"
    assert len(is_pn) == net.n, \
        "neuron_meta rows %d != brain neurons %d" % (len(is_pn), net.n)
    ip, ix, w = net.W_indptr, net.W_indices, net.W_data
    offs, pre = [], []
    for i in eln:
        o = np.arange(ip[i], ip[i + 1])
        o = o[is_pn[ix[o]]]
        offs.append(o); pre.append(np.full(len(o), i, np.int64))
    offs = np.concatenate(offs).astype(np.int64)
    pre = np.concatenate(pre)
    post = ix[offs].astype(np.int64)
    count = w[offs].astype(np.float32)
    assert len(offs) == 3802, "expected 3802 eLN->ALPN edges, got %d" % len(offs)
    assert (count > 0).all()
    return offs, pre, post, count


class GpuSim:
    """Holds W on the GPU across batches; one instance per process."""

    def __init__(self, n=None, device="cuda", compile_=True, deterministic=True):
        self.net = n or brain()
        self.deterministic = deterministic
        self.device = torch.device(device)
        N = self.net.n
        assert N == 139248, "unexpected neuron count %d" % N
        assert len(self.net.W_data) == self.net.n_edges, \
            "nnz %d != n_edges %d in npz" % (len(self.net.W_data), self.net.n_edges)
        # Presynaptic-major CSR: run_trial gathers the ROWS of the spiking
        # neurons, so rows must stay presynaptic. (A postsynaptic-major SpMM
        # would touch all 3.5M edges every step instead of the ~2,700 that
        # actually carry a spike.)
        self.indptr = torch.from_numpy(self.net.W_indptr).to(self.device)
        self.indices = torch.from_numpy(
            self.net.W_indices.astype(np.int64)).to(self.device)
        # flypoke scales the matrix once per trial: W = W * np.float32(w_syn).
        self.data = torch.from_numpy(
            (self.net.W_data * np.float32(W_SYN)).astype(np.float32)).to(self.device)
        self.compile_ = compile_
        self.group_id = torch.from_numpy(
            self.net.pool_group_id.astype(np.int64)).to(self.device)
        self.group_sizes = torch.from_numpy(
            self.net.pool_sizes.astype(np.float64)).to(self.device)

        # AUTHORED: homeostatic normalisation of incoming excitatory weight.
        # The scale is KEPT, not just applied, because learn/plastic.py rewrites
        # 18,674 of these edges from w0 and has to reapply it; see
        # Plastic.signed_values. It is float32[N] on the host, not a device
        # tensor: nothing in the step loop reads it. It is indexed by the
        # POSTsynaptic neuron, which is what W_indices holds.
        #
        # The gap edges are resolved HERE, above the normalisation, because the
        # 3,802 edges GAP_COUPLE zeroes must not count toward any PN's incoming
        # budget; see _input_scale's `exclude`. The offsets are kept and reused
        # by the gap block at the bottom of __init__ rather than recomputed.
        #
        # The mutual-exclusion guard depends only on the two constants, so it is
        # hoisted here, ahead of the resolution: an incompatible configuration
        # should fail before it does any work, not 80 lines later.
        assert not (ELN_NEGATE and GAP_COUPLE > 0.0), \
            "ELN_NEGATE and GAP_COUPLE together would negate the 44's non-ALPN " \
            "edges while zeroing their ALPN edges: half hack, half physiology; " \
            "pick one"
        gap_edges = _gap_edges(self.net) if GAP_COUPLE > 0.0 else None
        self.norm_scale = self._input_scale(
            self.net, NORM_TOTAL_TARGET,
            exclude=None if gap_edges is None else gap_edges[0])
        if NORM_TOTAL_TARGET > 0.0:
            vals = (self.net.W_data * np.float32(W_SYN)).astype(np.float32)
            vals = vals * self.norm_scale[self.net.W_indices]
            self.data = torch.from_numpy(vals.astype(np.float32)).to(self.device)

        # AUTHORED: graded APL. CSR rows are presynaptic, so APL's outgoing
        # edges are two contiguous slices. Cached once; the weights already
        # carry W_SYN so _deliver_apl matches _deliver's units.
        #
        # This block MUST sit below the normalisation block. `apl_w` is a
        # private copy of APL's rows that bypasses `self.data` entirely, so it
        # has to carry `norm_scale[post]` itself or graded APL would be the one
        # pathway in the brain arriving unnormalised. Measured when it did:
        # 3.9% of neurons take scale < 0.1, Kenyon cells - APL's principal
        # targets - sit in that high-convergence tail, so their excitation was
        # cut up to 10x while APL's inhibition onto them was not. That is a 10x
        # relative over-inhibition, and it silenced the combined arm outright.
        meta = np.load(os.path.join(_HERE, "data", "neuron_meta.npz"),
                       allow_pickle=False)
        ct = meta["cell_type"].astype(str)
        # neuron_meta row order is assumed to align with the brain npz. If it
        # ever does not, apl_tgt/apl_w are silently wrong rather than loud.
        assert len(ct) == self.net.n, \
            "neuron_meta rows %d != brain neurons %d" % (len(ct), self.net.n)
        apl = np.flatnonzero(ct == "APL").astype(np.int64)
        ip, ix, w = self.net.W_indptr, self.net.W_indices, self.net.W_data
        if apl.size:
            tgt = np.concatenate([ix[ip[i]:ip[i + 1]] for i in apl]).astype(np.int64)
            wts = np.concatenate([w[ip[i]:ip[i + 1]] for i in apl]) * np.float32(W_SYN)
            wts = wts * self.norm_scale[tgt]
            row = np.concatenate([np.full(ip[i + 1] - ip[i], k, np.int64)
                                  for k, i in enumerate(apl)])
        else:
            # A brain with no APL is constructable; np.concatenate([]) is not.
            # Only APL_GRADED = True actually needs these to be non-empty.
            tgt = np.zeros(0, np.int64)
            wts = np.zeros(0, np.float32)
            row = np.zeros(0, np.int64)
        self.apl_idx = torch.as_tensor(apl, device=self.device)
        self.apl_tgt = torch.as_tensor(tgt, device=self.device)
        self.apl_w = torch.as_tensor(wts.astype(np.float32), device=self.device)
        self.apl_row = torch.as_tensor(row, device=self.device)

        # AUTHORED: KC_V_TH_DELTA's target set. Cached as an index rather than a
        # threshold vector because run_batch needs it shaped [N, 1] against that
        # run's dtype and device, and because at the default delta of 0.0 no vector
        # is built at all. Sits below the `len(ct) == self.net.n` assert above, so a
        # misaligned neuron_meta.npz fails there rather than silently offsetting the
        # wrong rows here.
        self.kc_idx = torch.as_tensor(
            np.flatnonzero(meta["cell_class"].astype(str) == "Kenyon_Cell").astype(np.int64),
            device=self.device)

        # AUTHORED BRIDGE (ELN_NEGATE). Negate the 44 known eLNs' rows in the device
        # weight tensor only; self.net.W_data stays the exported matrix, so every
        # host-side reader (regime/reach.py, weights.py, Plastic.real) still sees v783.
        # Plastic edges are KC->MBON rows, these are ALLN rows: disjoint, so
        # Plastic.push cannot revert this (asserted in regime/test_eln.py).
        #
        # Like the apl_w block, this MUST sit below the normalisation block: that
        # block REASSIGNS self.data when NORM_TOTAL_TARGET > 0, so a negation
        # applied above it would be silently thrown away - and only on the
        # normalised path, which is exactly where it would go unnoticed.
        #
        # self.eln_idx is EMPTY on the default path: it means "the negated eLNs",
        # not "the eLNs". Call _eln_idx() for the set itself.
        self.eln_idx = torch.zeros(0, dtype=torch.int64, device=self.device)
        if ELN_NEGATE:
            eln = _eln_idx(self.net)
            r = np.concatenate([np.arange(ip[i], ip[i + 1]) for i in eln]).astype(np.int64)
            rt = torch.as_tensor(r, device=self.device)
            self.data[rt] *= -1.0          # row ranges are disjoint, so in-place is safe
            self.eln_idx = torch.as_tensor(eln, device=self.device)

        # AUTHORED (PN_KC_GAIN). Scale the ALPN -> Kenyon-cell edges. The companion to
        # KC_V_TH_DELTA and the one to prefer: the threshold lever reaches the 5-10%
        # band only at delta -7.0, where V_TH lands exactly on V_REST and a Kenyon cell
        # has no threshold margin left at all (measured; -7.5 makes the whole brain
        # free-run). Raising the drive INTO the cell buys the same recruitment without
        # a degenerate cell.
        #
        # Sits BELOW the normalisation block, for the reason the apl_w block documents:
        # NORM_TOTAL_TARGET > 0 REASSIGNS self.data, so a scale applied above it is
        # silently discarded, and only on the normalised path. Sits below the negation
        # block too - ALPN rows and eLN rows are disjoint, but the file's rule is that
        # every edit to self.data stays above every cache that reads it.
        if PN_KC_GAIN != 1.0:
            cc_all = meta["cell_class"].astype(str)
            pn_rows = np.flatnonzero(cc_all == "ALPN").astype(np.int64)
            is_kc = cc_all == "Kenyon_Cell"
            # CSR rows are PRESYNAPTIC and W_indices are POSTSYNAPTIC, so this is
            # "edges leaving an ALPN that land on a Kenyon cell".
            sel = np.concatenate([np.arange(ip[i], ip[i + 1]) for i in pn_rows]) \
                if pn_rows.size else np.zeros(0, np.int64)
            sel = sel[is_kc[ix[sel]]].astype(np.int64)
            assert sel.size, "PN_KC_GAIN set but no ALPN->KC edges found"
            self.data[torch.as_tensor(sel, device=self.device)] *= float(PN_KC_GAIN)
        # Unlike KC_V_TH_DELTA, which run_batch reads every call, this one is baked into
        # self.data here and cannot change afterwards. The two are documented as
        # companions and sit next to each other in the constants block, so the obvious
        # thing to try -- set the global on an existing sim and run again -- silently
        # returns the gain-1.0 baseline. Record what was baked in and let run_batch
        # refuse rather than quietly mislead.
        self.pn_kc_gain = PN_KC_GAIN

        # AUTHORED (TYPE_W_SCALE). Same placement rule as the two blocks above: below the
        # normalisation block, which reassigns self.data. Rows are PRESYNAPTIC, so a row
        # range is "everything this neuron sends". Plastic KC->MBON rows are untouched
        # unless someone scales Kenyon cells, which is refused.
        self.type_w_scale = dict(TYPE_W_SCALE)
        if TYPE_W_SCALE:
            ct_all = meta["cell_type"].astype(str)
            for t, k in TYPE_W_SCALE.items():
                assert t not in ("KC", "Kenyon_Cell") and not t.startswith("KC"),                     "TYPE_W_SCALE on Kenyon cells would fight Plastic.push"
                rows = np.flatnonzero(ct_all == t)
                assert rows.size, "TYPE_W_SCALE: no neuron of cell_type %r" % t
                sel = np.concatenate([np.arange(ip[i], ip[i + 1]) for i in rows]).astype(np.int64)
                self.data[torch.as_tensor(sel, device=self.device)] *= float(k)

        # AUTHORED (SLOW_FRAC). Which posts take the slow share. Baked at construction
        # like PN_KC_GAIN, and run_batch refuses a changed config for the same reason.
        self.slow_cfg = None
        if SLOW_FRAC > 0.0:
            assert SLOW_TYPES, "SLOW_FRAC > 0 with no SLOW_TYPES selects nothing"
            ct = meta["cell_type"].astype(str)
            sel = np.any([np.char.startswith(ct, p) for p in SLOW_TYPES], axis=0)
            assert sel.any(), "SLOW_TYPES %r match no neuron" % (SLOW_TYPES,)
            assert not sel[self.apl_idx.cpu().numpy()].any(), \
                "APL in SLOW_TYPES; graded APL has its own delivery path"
            self.slow_post = torch.as_tensor(sel, device=self.device)
            self.slow_cfg = (float(SLOW_FRAC), float(TAU_SLOW), tuple(SLOW_TYPES))

        # AUTHORED (GAP_COUPLE). Electrical delivery cache for the 44 known eLNs'
        # edges onto ALPNs. Coefficients derive from synapse COUNTS in the exported
        # matrix, so they are independent of W_SYN, normalisation and ELN_NEGATE;
        # the chemical entries for the same edges are zeroed in self.data so
        # _deliver never double-delivers. Sits BELOW the negation block: zeroing must
        # win over negation, and (per the apl_w rule) every edit to self.data stays
        # above every cache that reads it. Do not overload self.eln_idx here; it
        # means "the negated eLNs" and is empty unless ELN_NEGATE.
        #
        # These edges are also EXCLUDED from the homeostatic normalisation
        # totals (`gap_edges`, resolved above the normalisation block): they
        # are zeroed here, so counting them as incoming chemical budget would
        # silently shrink everything a gap-target PN actually receives. Measured
        # without the exclusion: the 337 target ALPNs lost a median 11.6% and up
        # to 36% of their budget to edges delivering nothing.
        assert GAP_COUPLE >= 0.0, "GAP_COUPLE must be >= 0"
        assert not (GAP_NORM and GAP_COUPLE == 0.0), \
            "GAP_NORM needs GAP_COUPLE > 0; alone it is a no-op that would be " \
            "reported as a mechanism"
        # The ELN_NEGATE mutual-exclusion guard fires above, next to the gap
        # offset resolution, before either mechanism has touched self.data.
        z = torch.zeros(0, dtype=torch.int64, device=self.device)
        self.gap_pre, self.gap_post, self.gap_idx = z, z.clone(), z.clone()
        self.gap_k = torch.zeros(0, dtype=torch.float32, device=self.device)
        self.gap_offsets = np.zeros(0, np.int64)
        if GAP_COUPLE > 0.0:
            offs, pre, post, count = gap_edges     # resolved above the norm block
            k = count * np.float32(GAP_COUPLE)
            if GAP_NORM:
                s_t = np.zeros(self.net.n, np.float64); np.add.at(s_t, post, count)
                k = k / s_t[post]
            k = k.astype(np.float32)
            # One neuron can be both an end and a target of other pairs, so the
            # bound is on its TOTAL coefficient, not on the larger of the two
            # halves: the two scatters land in the same v on the same step.
            s_post = np.zeros(self.net.n, np.float64); np.add.at(s_post, post, k)
            s_pre = np.zeros(self.net.n, np.float64); np.add.at(s_pre, pre, k)
            smax = float((s_post + s_pre).max())
            assert smax <= 1.0, \
                "GAP_COUPLE=%g gives a summed per-neuron coefficient of %.3f > 1 " \
                "(unstable explicit step); the largest stable value on this matrix " \
                "is %.4g" % (GAP_COUPLE, smax, GAP_COUPLE / smax)
            self.gap_offsets = offs
            self.data[torch.as_tensor(offs, device=self.device)] = 0.0
            self.gap_pre = torch.as_tensor(pre, device=self.device)
            self.gap_post = torch.as_tensor(post, device=self.device)
            self.gap_k = torch.as_tensor(k, device=self.device)
            # Both directions of every pair in ONE deterministic scatter: the
            # post half takes +d, the pre half -d. Two index_add_ calls under
            # the deterministic flag cost roughly twice what one does, and the
            # flag is what makes this scatter reproducible at all.
            self.gap_idx = torch.cat([self.gap_post, self.gap_pre])

    # ------------------------------------------------------------ scatter
    def _deliver(self, g, pre, bcol, B, h=None):
        """g[post, b] += sum over spiking pre of W[pre, post].

        Two presynaptic spikes can land on the same postsynaptic neuron in the
        same step, and CUDA's default float `index_add_` resolves that with
        atomics, so the summation order - and therefore the last bit of g -
        varies run to run. Measured: that is enough to flip a threshold
        crossing occasionally, and `verify_gpu.py` caught the same batch giving
        different spike counts twice. So the scatter runs under
        `use_deterministic_algorithms`, which costs ~0.54 ms a step at B=256
        (42 -> 33 sims/s). A shard you cannot reproduce is not worth 9 sims/s.

        The flag is toggled around this one call rather than set globally: the
        integer `counts` accumulation is order-independent already and pays
        0.19 ms a step under the deterministic flag for nothing.
        """
        start = self.indptr[pre]
        cnt = self.indptr[pre + 1] - start
        total = int(cnt.sum().item())
        if total == 0:
            return
        csum = torch.cumsum(cnt, 0) - cnt
        ar = torch.arange(total, device=self.device)
        offs = ar - torch.repeat_interleave(csum, cnt, output_size=total) \
            + torch.repeat_interleave(start, cnt, output_size=total)
        post = self.indices[offs]
        w = self.data[offs]
        bb = torch.repeat_interleave(bcol, cnt, output_size=total)
        if h is None:
            if self.deterministic:
                torch.use_deterministic_algorithms(True)
                try:
                    g.view(-1).index_add_(0, post * B + bb, w)
                finally:
                    torch.use_deterministic_algorithms(False)
            else:
                g.view(-1).index_add_(0, post * B + bb, w)
            return
        # AUTHORED (SLOW_FRAC): excitatory edges onto slow posts split; inhibition stays
        # fast. The slow share is charge-matched by TAU_SYN / TAU_SLOW.
        ws = torch.where(self.slow_post[post] & (w > 0), w * np.float32(self.slow_cfg[0]),
                         torch.zeros((), dtype=w.dtype, device=w.device))
        w = w - ws
        ws = ws * np.float32(TAU_SYN / self.slow_cfg[1])
        flat = post * B + bb
        torch.use_deterministic_algorithms(bool(self.deterministic))
        try:
            g.view(-1).index_add_(0, flat, w)
            h.view(-1).index_add_(0, flat, ws)
        finally:
            torch.use_deterministic_algorithms(False)

    # ------------------------------------------------------------ plasticity
    def set_plastic(self, offsets, values):
        """Overwrite weights at flat CSR offsets. `values` are already signed,
        already multiplied by W_SYN (mV). learn/plastic.py owns the bookkeeping."""
        self.data[torch.as_tensor(offsets, device=self.device, dtype=torch.int64)] = \
            torch.as_tensor(values, device=self.device, dtype=torch.float32)

    @staticmethod
    def _input_scale(net, target, exclude=None):
        """Per-neuron factor bringing summed incoming TOTAL input magnitude to
        `target` mV. Applied to every incoming edge of that neuron, excitatory
        and inhibitory alike, so the E/I ratio is preserved exactly and only the
        convergence tail moves.

        CSR rows are PRESYNAPTIC and W_indices are POSTSYNAPTIC, so incoming
        input is a scatter-add over W_indices - the same indexing as
        regime/weights.py:input_totals, but over |weight| rather than the
        positive entries alone. Neurons with no input at all get 1.0: there is
        nothing to normalise and target/0 would put a NaN in the weight tensor.

        `exclude` is an optional array of flat CSR offsets whose |w| is left out
        of the totals. It exists for GAP_COUPLE: those edges are zeroed in
        self.data and delivered electrically instead, so counting them here
        would budget a target PN for chemical input it never receives and shrink
        everything that actually arrives. Measured on the v783 matrix, the 337
        gap-target ALPNs would lose a median 11.6% and up to 36% of their
        incoming budget to edges that deliver nothing. Default None, so the
        normalisation path with GAP_COUPLE off is byte-for-byte what it was.
        """
        scale = np.ones(net.n, np.float32)
        if target <= 0.0:
            return scale
        w = np.abs(net.W_data)
        if exclude is not None and len(exclude):
            w = w.copy()
            w[np.asarray(exclude, dtype=np.int64)] = 0.0
        tot = np.zeros(net.n, np.float64)
        np.add.at(tot, net.W_indices.astype(np.int64), w)
        tot *= float(W_SYN)
        hit = tot > 0
        scale[hit] = (float(target) / tot[hit]).astype(np.float32)
        return scale

    def _apl_activation(self, v_apl):
        """v_apl: [n_apl, B] -> graded output in [0, 1]. Real APL is non-spiking."""
        return torch.clamp((v_apl - V_REST) / APL_SCALE, 0.0, 1.0)

    def _deliver_apl(self, g, act):
        """Add APL's outgoing inhibition, scaled by `act` [n_apl, B], into g.

        RATE EQUIVALENCE. Spiking APL delivered its full inhibitory row once per
        spike, through the delay line, so at f Hz it delivered `f * W` of charge
        per second. Graded APL delivers every step, with no delay and no
        refractory gap, so an unscaled `act * W` would deliver `act * W / DT`
        per second - at DT = 0.1 ms that is 10,000 * act * W, i.e. act=1 would
        be the equivalent of a 10 kHz APL, ~22x the ~450 Hz refractory ceiling
        of the mechanism it replaces. The `APL_MAX_HZ * DT / 1000` factor below
        fixes that: act=1 now delivers exactly the charge per second that a
        spiking APL firing at APL_MAX_HZ would, so `act` reads as "fraction of
        the maximum rate the spiking model could reach" and the graded and
        spiking arms are directly comparable. The factor lives here, not in
        `_apl_activation`, because the per-step units belong at the point of
        delivery and the activation should stay a clean [0,1] voltage curve.

        Left and right APL share postsynaptic targets, so this scatter has
        duplicate indices within a single call and CUDA's float `index_add_`
        would resolve them with atomics in a run-varying order - the exact
        nondeterminism `_deliver` documents. Same guard, same reason.

        FORM. With APL_DIVISIVE the same per-target quantity is computed by the
        same scatter, the same activation and the same rate-equivalence factor,
        and then applied as a DIVISOR on the target's synaptic drive instead of
        an offset into it: `g_j <- g_j / (1 + inh_j / APL_DIV_SCALE)`, with
        `inh_j` the magnitude (a positive number) of what the subtractive form
        would have added. Subtractive lowers every target by the same amount,
        which thins a population without reordering it; divisive scales, so the
        ratio between two targets' drives survives the inhibition. The scatter
        lands in a scratch buffer rather than in `g` so the division sees the
        WHOLE step's inhibition at once - dividing incrementally per synapse
        would compound and would not be the stated form.

        The divisor is applied to `g` signed, not to its positive part: a
        neuron whose net drive is already negative gets that negative drive
        scaled toward zero. That is a real asymmetry of the canonical form, and
        it is left in rather than special-cased because `g` is one pooled
        conductance here, not separated E and I channels, so there is no
        principled place to split it. It is small for Kenyon cells, whose `g`
        under this form carries PN excitation and no APL offset.
        """
        rate_eq = np.float32(APL_MAX_HZ * DT / 1000.0)
        contrib = self.apl_w.unsqueeze(1) * act[self.apl_row] * float(rate_eq)
        target = torch.zeros_like(g) if APL_DIVISIVE else g
        if self.deterministic:
            torch.use_deterministic_algorithms(True)
            try:
                target.index_add_(0, self.apl_tgt, contrib)
            finally:
                torch.use_deterministic_algorithms(False)
        else:
            target.index_add_(0, self.apl_tgt, contrib)
        if APL_DIVISIVE:
            inh = target.neg_().clamp_(min=0.0)
            g.div_(inh.div_(float(APL_DIV_SCALE)).add_(1.0))

    def _deliver_gap(self, v, r):
        """v[j, b] += sum_i k_ij (v[i, b] - v[j, b]) and the mirror onto i, for the
        cached eLN->ALPN pairs, on non-refractory ends only.

        O(E*B), never O(N*B): every tensor here is [E, B] or [2E, B] over the
        3,802 cached edges, and the result scatters straight into v. There is no
        dense [N, B] intermediate to allocate or zero - the earlier form built
        one per step and paid for 139,248 rows to move 381.

        `d` is computed from v BEFORE anything is added, and the += onto both
        ends reads that one snapshot, so the two directions of a pair see the
        same pre-update voltages however the scatter orders itself.

        The refractory mask is applied PER EDGE END, not per neuron: the [2E, B]
        contribution is multiplied by `r == 0` gathered at the receiving index,
        so a refractory end takes nothing while its non-refractory partner still
        gives. Same semantics as masking a dense dv, one gather instead of a
        dense where().

        Deterministic scatter for the same reason _deliver gives: eLNs share
        ALPN targets, so gap_idx has duplicate indices within one call and
        CUDA's float index_add_ would resolve them with atomics in a run-varying
        order. On the `deterministic=False` branch those atomics now add each
        edge's contribution into v at the ~52 mV scale, where float32 spacing is
        3.8e-6 mV, rather than into a dv buffer that starts at zero: the
        per-edge headroom at small GAP_COUPLE is correspondingly smaller than
        the dense form had, and a contribution below that spacing is lost.

        A SPIKING eLN transmits nothing through this path. V_RESET == V_REST, so
        a neuron that just fired sits at exactly the voltage the coupling
        measures from, and it is refractory for the next 21 steps anyway. Only
        the sub-threshold ramp couples, which is what a gap junction does and is
        the whole point of the primitive.
        """
        if self.gap_k.numel() == 0:
            return
        d = (v.index_select(0, self.gap_pre)
             - v.index_select(0, self.gap_post)).mul_(self.gap_k.view(-1, 1))
        both = torch.cat([d, -d], 0).mul_(
            (r.index_select(0, self.gap_idx) == 0).to(v.dtype))
        if self.deterministic:
            torch.use_deterministic_algorithms(True)
            try:
                v.index_add_(0, self.gap_idx, both)
            finally:
                torch.use_deterministic_algorithms(False)
        else:
            v.index_add_(0, self.gap_idx, both)

    # ------------------------------------------------------------ main loop
    def run_batch(self, drives, seeds, t_run=300.0, rng="counter", progress=None,
                  state=None, return_state=False, silence=None, bias=None):
        """drives: list of B (stim_idx, stim_prob) pairs -> counts int32 [B, N].

        Returns spike COUNTS, not rates, so the caller can divide exactly the way
        `SpikeRecord.rates` does (integer counts / float64 seconds) and land on
        bit-identical rates for identical spike trains.

        AUTHORED (closed loop). `state` resumes a previous call's integrator and
        `return_state` hands it back, so a sensorimotor loop can tick the brain
        repeatedly without it restarting from rest every time. Chaining is EXACT:
        three carried 100 ms calls give the same counts as one 300 ms call
        (regime/test_carry.py). Both default off and the untouched path is
        byte-for-byte what it was.

        There are SIX pieces of state, not four. `pending` and `apl_pending` are
        the 1.8 ms synaptic delay lines; dropping them silently deletes every
        in-flight spike at each tick boundary, which looks like a slightly quieter
        brain rather than like a bug. And the counter RNG derives its stream from
        the step index, so a resumed call must continue counting: restarting at 0
        replays identical Poisson draws every tick, which is a periodic artefact
        at exactly the tick frequency -- the worst possible confound for a
        rhythmic locomotion experiment.

        `silence`: int64 index array/tensor; those neurons never cross threshold
        (AUTHORED lesion, the silencing of Christie et al. 2026). It gates
        OUTPUT only - a silenced cell still receives synaptic input, exactly as
        in an optogenetic/genetic silencing experiment. Must not overlap the
        Poisson-driven set: `no_spike` masks `fired`, not `spk = fired |
        poisson`, so a stimulated-and-silenced neuron would still spike.

        `bias`: AUTHORED additive input (2026-09-23, docs/superpowers/cx_bridge/
        additive_bias_2026-09-23.md). A drive, in the `drives` sense, REPLACES a
        neuron's output with a Poisson train and discards its synaptic input
        (`prob > 0` masks threshold crossing). `bias` ADDS to the drive term next to
        g instead, so the neuron keeps integrating its synaptic input. Units: mV of
        steady-state depolarisation (the same units as g; a constant bias I alone
        settles v at V_REST + I; threshold is 7 mV above rest). bias_mv_for_hz()
        converts a drive's Hz into the bias giving that rate in an isolated cell.
        Either a list of B (idx, mV) pairs, one per drive (duplicates sum), or a
        dense float array/tensor of shape (N,) (every position) or (N, B). Constant
        within a call; vary it per window by chaining calls with `state`. None = off,
        and the off path is bit-identical to before (regime/test_additive_bias.py).
        Must not overlap the Poisson-driven set, where it would do nothing.
        """
        dev = self.device
        N, B = self.net.n, len(drives)
        n_steps = int(round(t_run / DT))
        step0 = 0
        if state is not None:
            assert state["b"] == B, (
                "resuming a state of batch width %d into a call with %d drives"
                % (state["b"], B))
            assert state["apl_graded"] == bool(APL_GRADED and APL_DELAYED), (
                "APL_GRADED/APL_DELAYED changed since this state was produced; its "
                "apl_pending delay line no longer matches the configuration")
            assert rng == "counter", (
                "state carry-over is only implemented for the counter RNG; %r "
                "regenerates its draws from (seeds, n_steps) with no step offset, so "
                "every resumed tick would replay the same Poisson stream" % rng)
            step0 = state["step0"]
            assert state.get("slow") == self.slow_cfg, (
                "resuming a state built under slow config %r into a sim with %r"
                % (state.get("slow"), self.slow_cfg))
        delay_steps = max(1, int(round(DELAY / DT)))
        refr_steps = int(round(T_REFR / DT))
        decay_m = float(np.float32(np.exp(-DT / TAU_M)))
        decay_s = float(np.float32(np.exp(-DT / TAU_SYN)))
        decay_a = float(np.float32(np.exp(-DT / TAU_A)))
        gain = float(np.float32(1.0 - np.float32(np.exp(-DT / TAU_M))))

        idx_list = [np.asarray(d[0], dtype=np.int64) for d in drives]
        prob_list = [np.asarray(d[1], dtype=np.float64) for d in drives]
        S = max((len(i) for i in idx_list), default=0)

        # Dense per-neuron Poisson probability. It carries flypoke's `is_stim`
        # for free (prob > 0), which is why the fused kernel needs no second
        # mask. Asserted rather than assumed: a zero-rate Stimulus would make a
        # stimulated neuron look unstimulated and silently change the dynamics.
        prob_np = np.zeros((N, B), dtype=np.float32)
        for b, (ii, pp) in enumerate(zip(idx_list, prob_list)):
            if len(ii):
                if not (pp > 0).all():
                    raise ValueError("stimulus with rate <= 0 in position %d; "
                                     "prob > 0 can no longer stand in for "
                                     "is_stim" % b)
                prob_np[ii, b] = pp
        prob = torch.from_numpy(prob_np).to(dev)

        bx = {}                    # kernel kwargs; empty on the default path
        if bias is not None:
            if torch.is_tensor(bias) or isinstance(bias, np.ndarray):
                bias_t = torch.as_tensor(bias, dtype=torch.float32, device=dev)
                if bias_t.dim() == 1:
                    bias_t = bias_t.view(-1, 1)
                assert tuple(bias_t.shape) in ((N, 1), (N, B)), \
                    "dense bias must be (N,) or (N, B); got %r" % (tuple(bias_t.shape),)
            else:
                assert len(bias) == B, "bias: %d (idx, mV) pairs for %d drives" % (
                    len(bias), B)
                bias_np = np.zeros((N, B), dtype=np.float32)
                for b, (ii, mv) in enumerate(bias):
                    ii = np.asarray(ii, dtype=np.int64)
                    np.add.at(bias_np[:, b], ii,
                              np.broadcast_to(np.asarray(mv, np.float32), ii.shape))
                bias_t = torch.from_numpy(bias_np).to(dev)
            assert bool(torch.isfinite(bias_t).all()), "bias has a non-finite entry"
            assert not bool(((bias_t != 0) & (prob > 0)).any()), \
                "bias on a Poisson-stimulated neuron does nothing: prob > 0 masks " \
                "threshold crossing"
            bx["bias"] = bias_t

        seeds = np.asarray(seeds, dtype=np.int64)
        seed32 = torch.from_numpy(
            (seeds & 0xFFFFFFFF).astype(np.uint32).astype(np.int32)).to(dev).view(1, B)
        neuron = torch.arange(N, dtype=torch.int32, device=dev).view(N, 1)
        # The per-step RNG counter lives on the device as a table, not as a
        # python int argument: dynamo specialises on int arguments, so passing
        # the step number by value recompiled the fused kernel every step until
        # it hit the recompile limit and fell back to eager (measured 4.2 sims/s
        # instead of 21). This costs one index into a [n_steps] tensor.
        # step0 is 0 on a fresh call, so this is unchanged there; on a resumed call
        # the stream continues rather than replaying the same draws every tick.
        step_consts = torch.from_numpy(
            np.array([_w32((step0 + t) * _S1) for t in range(n_steps)],
                     dtype=np.int32)).to(dev)
        poisson_buf = None
        if rng == "numpy":
            stim_neuron = np.zeros((max(S, 1), B), dtype=np.int64)
            for b, ii in enumerate(idx_list):
                stim_neuron[:len(ii), b] = ii
            stim_neuron = torch.from_numpy(stim_neuron).to(dev)
            hits_all = _numpy_poisson(idx_list, prob_list, seeds, n_steps,
                                      max(S, 1), dev)
            poisson_buf = torch.zeros((N, B), dtype=torch.bool, device=dev)
        elif rng != "counter":
            raise ValueError("rng must be 'counter' or 'numpy'")

        if state is None:
            v = torch.full((N, B), V_REST, dtype=torch.float32, device=dev)
            g = torch.zeros((N, B), dtype=torch.float32, device=dev)
            r = torch.zeros((N, B), dtype=torch.int8, device=dev)
            a = torch.zeros((N, B), dtype=torch.float32, device=dev)
        else:
            v, g, r, a = (state["v"].clone(), state["g"].clone(),
                          state["r"].clone(), state["a"].clone())
        h = decay_h = None
        if self.slow_cfg is not None:
            h = (torch.zeros((N, B), dtype=torch.float32, device=dev) if state is None
                 else state["h"].clone())
            decay_h = float(np.float32(np.exp(-DT / self.slow_cfg[1])))
        no_spike = torch.zeros((N, 1), dtype=torch.bool, device=dev)
        if silence is not None and len(silence):
            sil = (silence.to(device=dev, dtype=torch.int64)
                   if torch.is_tensor(silence)
                   else torch.as_tensor(np.asarray(silence, dtype=np.int64), device=dev))
            no_spike[sil] = True
            assert not bool((prob[sil] > 0).any()), \
                "silenced neuron is Poisson-stimulated; no_spike masks fired, not poisson"
        # AUTHORED: the divisive form is a property of the graded delivery path.
        # Silently ignoring it under a spiking APL is how an arm gets reported
        # as divisive when it ran subtractive.
        assert not (APL_DIVISIVE and not APL_GRADED), \
            "APL_DIVISIVE requires APL_GRADED; the spiking path has no per-step " \
            "inhibition magnitude to divide by"
        if APL_GRADED:
            no_spike[self.apl_idx] = True
            # `no_spike` masks `fired`, not `spk = fired | poisson`. An APL in a
            # stimulus set would still spike via Poisson, enter `pending`, and
            # have its full row delivered by _deliver ON TOP of _deliver_apl -
            # double-counted inhibition with no error. Refuse it instead.
            assert not bool((prob[self.apl_idx] > 0).any()), \
                "APL is Poisson-stimulated while APL_GRADED is on; its output " \
                "would be delivered twice"
        counts = torch.zeros(N * B, dtype=torch.int32, device=dev)
        refr_m1 = torch.tensor(refr_steps - 1, dtype=torch.int8, device=dev)
        zero_i8 = torch.tensor(0, dtype=torch.int8, device=dev)
        v_reset_t = torch.tensor(V_RESET, dtype=torch.float32, device=dev)
        # AUTHORED (KC_V_TH_DELTA). A plain float on the default path, so the
        # arithmetic and the compiled graph are exactly what they were. When the
        # offset is live, a [N, 1] tensor: _body compares it against vn's [N, B] and
        # broadcasting does the rest, so the kernel needs no change. Both step call
        # sites below must take this and not V_TH - the mask path is the verification
        # path, and a mechanism that silently skips it is the worst place to lose one.
        assert self.pn_kc_gain == PN_KC_GAIN, (
            "PN_KC_GAIN is %r now but this GpuSim was built with %r. It is baked into "
            "the weights at construction, unlike KC_V_TH_DELTA which is read here every "
            "call, so changing it on an existing sim does nothing. Build a new GpuSim."
            % (PN_KC_GAIN, self.pn_kc_gain))
        assert self.type_w_scale == TYPE_W_SCALE, (
            "TYPE_W_SCALE is %r now but this GpuSim was built with %r; it is baked into "
            "the weights. Build a new GpuSim." % (TYPE_W_SCALE, self.type_w_scale))
        cur_slow = ((float(SLOW_FRAC), float(TAU_SLOW), tuple(SLOW_TYPES))
                    if SLOW_FRAC > 0.0 else None)
        assert cur_slow == self.slow_cfg, (
            "slow config is %r now but this GpuSim was built with %r. Build a new GpuSim."
            % (cur_slow, self.slow_cfg))
        v_th_t = V_TH
        if KC_V_TH_DELTA != 0.0:
            v_th_t = torch.full((N, 1), V_TH, dtype=torch.float32, device=dev)
            v_th_t[self.kc_idx] = V_TH + KC_V_TH_DELTA
        ones = torch.ones(1 << 14, dtype=torch.int32, device=dev)

        empty = torch.empty(0, dtype=torch.int64, device=dev)
        # Cloned, like v/g/r/a: a returned state may be resumed more than once (the
        # carry test branches two runs off one), and sharing the delay-line tensors
        # would make any future in-place write to them corrupt the other branch.
        pending = (deque([(empty, empty) for _ in range(delay_steps)])
                   if state is None
                   else deque((i.clone(), v_.clone()) for i, v_ in state["pending"]))
        # AUTHORED: graded APL's own delay line. The spiking path routes every
        # contribution through `pending`, so APL->KC inhibition arrived 1.8 ms
        # (18 steps) after the spike. Reading v[apl] in the same step would
        # shorten that loop 18x, and an 18x shorter loop is a different
        # dynamical system, not a faster version of the same one.
        apl_pending = None
        if APL_GRADED and APL_DELAYED:
            apl_pending = (deque(
                [torch.zeros((self.apl_idx.numel(), B), dtype=torch.float32,
                             device=dev) for _ in range(delay_steps)])
                if state is None
                else deque(t.clone() for t in state["apl_pending"]))
        v_apl_max = V_REST + APL_SCALE
        step_fn = _kernel("counter" if rng == "counter" else "mask",
                          _step_counter if rng == "counter" else _step_mask,
                          self.compile_)

        for step in range(n_steps):
            pre, bcol = pending.popleft()
            if pre.numel():
                self._deliver(g, pre, bcol, B, h)
            if APL_GRADED:
                act = self._apl_activation(v[self.apl_idx])
                if apl_pending is not None:
                    apl_pending.append(act)
                    act = apl_pending.popleft()
                self._deliver_apl(g, act)
            if self.gap_k.numel():
                self._deliver_gap(v, r)

            if rng == "counter" and h is not None:
                v, g, r, a, spk, h = step_fn(v, g, r, a, prob, seed32,
                                             step_consts[step], neuron, no_spike,
                                             V_REST, decay_m, gain, decay_s,
                                             decay_a, SFA_B_INC, v_th_t, v_reset_t,
                                             refr_m1, zero_i8, h, decay_h, **bx)
            elif rng == "counter":
                v, g, r, a, spk = step_fn(v, g, r, a, prob, seed32,
                                          step_consts[step], neuron, no_spike,
                                          V_REST, decay_m, gain, decay_s,
                                          decay_a, SFA_B_INC, v_th_t, v_reset_t,
                                          refr_m1, zero_i8, **bx)
            else:
                poisson_buf.zero_()
                if S:
                    p_s, p_b = hits_all[step].nonzero(as_tuple=True)
                    if p_s.numel():
                        poisson_buf.view(-1)[stim_neuron[p_s, p_b] * B + p_b] = True
                if h is not None:
                    v, g, r, a, spk, h = step_fn(v, g, r, a, prob, poisson_buf,
                                                 no_spike, V_REST, decay_m, gain,
                                                 decay_s, decay_a, SFA_B_INC, v_th_t,
                                                 v_reset_t, refr_m1, zero_i8, h, decay_h, **bx)
                else:
                    v, g, r, a, spk = step_fn(v, g, r, a, prob, poisson_buf,
                                              no_spike, V_REST, decay_m, gain,
                                              decay_s, decay_a, SFA_B_INC, v_th_t,
                                              v_reset_t, refr_m1, zero_i8, **bx)

            if APL_GRADED:
                # A non-spiking APL never fires, so it never resets and never
                # enters refractory: nothing bounds its membrane. The spiking
                # APL it replaces was ceilinged at V_TH by its own reset, and
                # _apl_activation clamps the OUTPUT but not the STATE, so a
                # transient to V_REST+150 would leave act pinned at 1.0 for
                # TAU_M*ln(150/APL_SCALE) ~ 61 ms after its input stopped - a
                # fifth of a trial of hysteresis in the mechanism whose job is
                # to remove hysteresis. Clamp the state at saturation, where
                # any higher voltage is behaviourally identical anyway.
                v[self.apl_idx] = v[self.apl_idx].clamp(max=v_apl_max)

            s_pre, s_b = spk.nonzero(as_tuple=True)
            if s_pre.numel():
                flat = s_pre * B + s_b
                if flat.numel() > ones.numel():
                    ones = torch.ones(flat.numel(), dtype=torch.int32, device=dev)
                counts.index_add_(0, flat, ones[:flat.numel()])
            pending.append((s_pre, s_b))
            if progress and step % 500 == 0:
                progress(step, n_steps)

        out = counts.view(N, B).T.contiguous()
        if not return_state:
            return out
        return out, {"v": v, "g": g, "r": r, "a": a,
                     "pending": list(pending),
                     "apl_pending": None if apl_pending is None else list(apl_pending),
                     "step0": step0 + n_steps, "b": B,
                     "apl_graded": bool(APL_GRADED and APL_DELAYED),
                     **({"h": h, "slow": self.slow_cfg} if h is not None else {})}

    # ------------------------------------------------------------ readouts
    def views_of(self, counts, t_run=300.0):
        """Whatever `reservoir.views` currently keeps, from GPU spike counts.

        The readout is deliberately NOT reimplemented on the GPU. It changed
        twice while this port was being written (cell_type means -> pooled +
        proj + subset -> pooled + live), and a second copy of it is precisely
        how two engines drift apart without anyone noticing. So the rates come
        back to the host and go through the same `reservoir.views` the CPU
        workers call: whatever keys it returns are the keys a GPU shard gets.

        The division matches `SpikeRecord.rates` exactly - integer counts over
        float64 seconds - so the rate vector handed to `views` is bit-identical
        to what the CPU path would produce from the same spike train.

        Measured: ~1.4 ms a position, against ~24 ms of GPU integration at
        B=256. Not worth the risk of a divergent second implementation.
        """
        import reservoir as R
        rates = counts.cpu().numpy().astype(np.int64) / (t_run / 1000.0)
        rows = [R.views(rates[i], self.net) for i in range(len(rates))]
        return {k: np.stack([r[k] for r in rows]) for k in rows[0]}

    def pool_gpu(self, counts, t_run=300.0):
        """GPU float64 `reservoir.pool`, kept only as verifier evidence.

        Not used by the shard writer. It exists so `verify_gpu.py` can show
        that a GPU-side reduction of the rate vector lands on the same float16
        as numpy's `bincount`, i.e. that the rate vector itself is right and
        not merely the CPU code being reused.
        """
        rates = counts.to(torch.float64) / (t_run / 1000.0)
        pooled = torch.zeros((rates.shape[0], len(self.net.pool_sizes)),
                             dtype=torch.float64, device=self.device)
        pooled.index_add_(1, self.group_id, rates)
        return (pooled / self.group_sizes).to(torch.float16).cpu().numpy()


# ---------------------------------------------------------------- convenience
def simulate_fens(fens, t_run=300.0, batch=256, rng="counter", sim=None,
                  seeds=None, compile_=True):
    """FENs -> dict of the three views, one row per FEN, in input order."""
    import chess
    import reservoir as R
    sim = sim or GpuSim(compile_=compile_)
    out = {}
    for lo in range(0, len(fens), batch):
        chunk = fens[lo:lo + batch]
        boards = [chess.Board(f) for f in chunk]
        drives = [drive_of(b, sim.net) for b in boards]
        sd = ([R.seed_for(b) for b in boards] if seeds is None
              else seeds[lo:lo + batch])
        counts = sim.run_batch(drives, sd, t_run=t_run, rng=rng)
        for k, v in sim.views_of(counts, t_run).items():
            out.setdefault(k, []).append(v)
    return {k: np.concatenate(v) for k, v in out.items()}


# ---------------------------------------------------------------- benchmark
def bench(sizes=(1, 64, 256, 512, 1024), t_run=300.0, compile_=True,
          out="results/gpu_bench.json", modes=(True, False)):
    import json

    import chess
    import pyarrow.parquet as pq

    import reservoir as R

    n = brain()
    table = pq.read_table("data/positions.parquet", columns=["fen"])
    fens = table.column("fen").to_pylist()[:max(sizes)]
    boards = [chess.Board(f) for f in fens]

    t0 = time.time()
    drives = [drive_of(b, n) for b in boards]
    drive_secs = (time.time() - t0) / len(boards)
    seeds = [R.seed_for(b) for b in boards]
    print("drive build: %.1f ms/position (CPU, single core)" % (1e3 * drive_secs))

    sim = GpuSim(compile_=compile_)
    all_rows = {}
    for det in modes:
        sim.deterministic = det
        rows = []
        counts = None
        for B in sizes:
            del counts                       # else the previous B's [B, N] state
            counts = None                    # inflates this B's peak-VRAM figure
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            base = torch.cuda.memory_allocated()   # subtracted below: the peak
            # counter is process-wide, so anything still resident from an
            # earlier sweep would otherwise be charged to this batch size
            # Warm at the SAME B: torch.compile specialises on shape, so warming
            # at another batch size buys a compile it then throws away.
            sim.run_batch(drives[:B], seeds[:B], t_run=20.0)
            torch.cuda.synchronize()
            t0 = time.time()
            counts = sim.run_batch(drives[:B], seeds[:B], t_run=t_run)
            torch.cuda.synchronize()
            sim_s = time.time() - t0
            t1 = time.time()
            sim.views_of(counts, t_run)
            view_s = time.time() - t1
            peak = (torch.cuda.max_memory_allocated() - base) / 2**30
            rows.append({"batch": B, "deterministic_scatter": det,
                         "sim_seconds": round(sim_s, 3),
                         "view_seconds": round(view_s, 3),
                         "sims_per_s_gpu_only": round(B / sim_s, 2),
                         "sims_per_s_end_to_end": round(
                             B / (sim_s + view_s + B * drive_secs), 2),
                         "peak_vram_gb": round(peak, 2),
                         "mean_spikes_per_sim": round(
                             int(counts.sum().item()) / B, 1)})
            print("det=%-5s B=%-5d %6.2f sims/s gpu  %6.2f end-to-end  "
                  "%.2f GB peak  %.0f spikes/sim"
                  % (det, B, rows[-1]["sims_per_s_gpu_only"],
                     rows[-1]["sims_per_s_end_to_end"], peak,
                     rows[-1]["mean_spikes_per_sim"]))
        all_rows["deterministic" if det else "nondeterministic"] = rows

    blob = {"device": torch.cuda.get_device_name(0),
            "deterministic_scatter": True,
            "note": ("sims_per_s_end_to_end includes the CPU drive build; "
                     "B=1 is SLOWER than one CPU worker because a single "
                     "position is launch-latency bound, not bandwidth bound"),
            "torch": torch.__version__,
            "t_run_ms": t_run,
            "compiled": bool(compile_ and _compiled.get("counter") not in
                             (None, _step_counter)),
            "drive_build_ms_per_position": round(1e3 * drive_secs, 2),
            "cpu_reference_sims_per_s": 7.66,
            "rows": all_rows.get("deterministic", []),
            "rows_by_mode": all_rows}
    attach_evidence(blob)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(blob, open(out, "w"), indent=1)
    print("wrote", out)
    return blob


def attach_evidence(blob, verify="results/gpu_verify.json",
                    cpu_dir="data/reservoir_v2", gpu_dir="data/reservoir_gpu"):
    """Fold the agreement evidence into the throughput file.

    A sims/s number on its own is the dangerous half of this port, so the file
    that records the speed also records what it was measured against.
    """
    import json
    if os.path.exists(verify):
        v = json.load(open(verify))
        blob["agreement_vs_flypoke"] = {
            k: v[k] for k in
            ("drive_identical", "deterministic_same_seed_twice",
             "batch_invariant", "gpu_pool_equals_reservoir_pool", "views",
             "numpy_rng", "counter_rng", "cpu_seed_noise",
             "stimulated_neuron_rate_error", "pooled", "dn_worst_case",
             "behaviours", "all_ok")
            if k in v}
        blob["agreement_source"] = verify
    if os.path.exists(os.path.join(gpu_dir, "shard_0000.npz")):
        import precompute_gpu
        blob["shard_interop_vs_flypoke"] = (
            precompute_gpu.compare_to_reference(gpu_dir))
        if os.path.exists(os.path.join(cpu_dir, "shard_0000.npz")):
            blob["shard_interop_vs_cpu_shard"] = (
                precompute_gpu.compare_to_cpu(cpu_dir, gpu_dir))
            blob["shard_interop_vs_cpu_shard"]["cpu_dir"] = cpu_dir
    return blob


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--sizes", type=str, default="1,64,256,512,1024")
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--t-run", type=float, default=300.0)
    args = ap.parse_args()
    if args.bench:
        bench(sizes=tuple(int(x) for x in args.sizes.split(",")),
              t_run=args.t_run, compile_=not args.no_compile)
    else:
        print(__doc__)
