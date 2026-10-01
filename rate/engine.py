"""Differentiable whole-brain rate model on the same v783 W as gpu_sim (path 2, PRD phase 1).

Lappalainen et al. 2024 dynamics, per cell-type pool p(i):

    tau_p dv_i/dt = -v_i + gain_p * w_scale * sum_j W_ji r_j + bias_p,    r = relu(v)

v and r are in Hz; W is the signed synapse count from brain_gpu.npz; w_scale (Hz per
synapse-Hz) is the ONE global knob, deliberately a required argument: its science value
is chosen in the phase-3 prereg, not here. Parameters are per pool_group_id (8,865 pools),
never per synapse.

Semantics copied from gpu_sim.run_batch so the two models answer the same questions:
  - a drive REPLACES the driven neuron's output with its rate (its input is ignored);
  - `silence` gates OUTPUT only: a silenced neuron still integrates its input;
  - state carry is exact (the only state is v: no delay line, no RNG).

Deterministic ODE, forward Euler at DT_MS. No noise: add it when a prereg needs trials.
"""
import os
import sys

import numpy as np
import scipy.sparse as sp
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gpu_sim as G  # noqa: E402  (BRAIN path only)

DT_MS = 1.0         # ms, Euler step; dt/tau = 0.05 at the default tau
TAU0_MS = 20.0      # ms, AUTHORED default time constant (= TAU_M of the LIF)


class _SpMM(torch.autograd.Function):
    """y = W^T r over a post-major CSR, with a DETERMINISTIC backward over the pre-major CSR.

    Autograd through segment_reduce would keep the [nnz, B] product of every step (~43 MB each,
    43 GB for a 1 s run); this keeps nothing per step. Both directions are segment_reduce (no
    atomics), so gradients are bit-reproducible too."""

    @staticmethod
    def forward(ctx, r, data, indices, indptr, data_pre, indices_pre, indptr_pre):
        ctx.save_for_backward(data_pre, indices_pre, indptr_pre)
        return torch.segment_reduce(data * r[indices], "sum", offsets=indptr, axis=0)

    @staticmethod
    def backward(ctx, g):
        data_pre, indices_pre, indptr_pre = ctx.saved_tensors
        gr = torch.segment_reduce(data_pre * g[indices_pre], "sum", offsets=indptr_pre, axis=0)
        return gr, None, None, None, None, None, None


class _SurrogateReLU(torch.autograd.Function):
    """Forward: exactly relu. Backward: true slope 1 above threshold (v > 0); below it the
    fast-sigmoid tail (1 + |v|/beta)^-2 (SuperSpike, Zenke & Ganguli 2018) instead of relu's 0.
    Only the optimiser sees the difference; the simulated model is unchanged. `mask` [N, 1] limits
    the tail to chosen neurons (the chunk being fitted); everywhere else the slope stays relu's exact
    0 below threshold, because a surrogate on the whole silent brain turns the backward into the
    linearised full recurrence, which is unstable (gradient 1.7e36 at init, amendment 2)."""

    @staticmethod
    def forward(ctx, v, beta, mask):
        ctx.save_for_backward(v, mask)
        ctx.beta = beta
        return torch.relu(v)

    @staticmethod
    def backward(ctx, g):
        v, mask = ctx.saved_tensors
        tail = torch.where(mask, (1.0 + v.abs() / ctx.beta) ** -2, torch.zeros_like(v))
        return g * torch.where(v > 0, torch.ones_like(v), tail), None, None


def eln8_edits(csr):
    """The two authored edits of gpu_sim's eln8 regime, on a pre-major CSR copy, in gpu_sim's order:
    negate every out-edge of the 44 known eLNs (gpu_sim._eln_idx), then scale ALPN -> Kenyon-cell edges
    by 8. Checked edge-by-edge against gpu_sim's device weights in rate/test_chunk0.py."""
    from types import SimpleNamespace
    csr = csr.copy()
    ip = csr.indptr
    for i in G._eln_idx(SimpleNamespace(n=csr.shape[0], W_indptr=ip, W_data=csr.data)):
        csr.data[ip[i]:ip[i + 1]] *= -1
    cc = np.load(os.path.join(G._HERE, "data", "neuron_meta.npz"), allow_pickle=False)["cell_class"].astype(str)
    is_kc = cc == "Kenyon_Cell"
    for i in np.flatnonzero(cc == "ALPN"):
        seg = slice(ip[i], ip[i + 1])
        csr.data[seg] = np.where(is_kc[csr.indices[seg]], csr.data[seg] * 8.0, csr.data[seg])
    return csr


LIF_TH_MV = G.V_TH - G.V_REST  # 7 mV: threshold of the LIF, above rest


def lif_fi(v):
    """Exact Shiu LIF f-I curve (Hz) of a mean depolarisation v above rest (mV); 0 at or below
    threshold. v is masked before the log, so the curve and its gradient stay finite everywhere."""
    th = LIF_TH_MV
    ok = v > th + 1e-6
    vs = torch.where(ok, v, torch.full_like(v, th + 1.0))
    f = 1000.0 / (G.T_REFR + G.TAU_M * torch.log(vs / (vs - th)))
    return torch.where(ok, f, torch.zeros_like(v))


class RateSim:
    def __init__(self, w_scale, path=G.BRAIN, device="cuda", dtype=torch.float32,
                 csr=None, pool=None, regime="stock"):
        """csr/pool: optional pre-major scipy CSR + pool ids (toy circuits in tests).
        regime: "stock" or "eln8" (gpu_sim's authored edits, see eln8_edits; real brain only)."""
        if csr is None:
            blob = np.load(path, allow_pickle=False)
            n = int(blob["n_neurons"])
            # rows = PRESYNAPTIC (H32: reading this as CSC silently transposes W)
            csr = sp.csr_matrix((blob["W_data"], blob["W_indices"], blob["W_indptr"]),
                                shape=(n, n))
            pool = blob["pool_group_id"]
            if regime == "eln8":
                csr = eln8_edits(csr.astype(np.float64))
        assert regime in ("stock", "eln8"), regime
        self.regime = regime
        self.n = csr.shape[0]
        self.W = csr
        self.w_scale = float(w_scale)
        self.device, self.dtype = device, dtype
        self._full_nnz = csr.nnz
        self._set_frozen(csr)
        self.edge_x = None  # trainable edge-type gains, see set_edge_gains
        self._full2frozen = None  # full pre-major offset -> frozen pre-major offset (-1: gained edge); None = identity
        self.pool = torch.as_tensor(np.asarray(pool), dtype=torch.int64, device=device)
        n_pools = int(self.pool.max()) + 1
        f = dict(device=device, dtype=dtype)
        self.log_tau = torch.full((n_pools,), float(np.log(TAU0_MS)), **f)
        self.bias = torch.zeros(n_pools, **f)
        self.gain = torch.ones(n_pools, **f)
        self.r_max = float("inf")  # optional output ceiling (Hz); inf = plain ReLU
        self.transfer = "lin"  # "lin": r = relu(v), v in Hz; "lif": r = lif_fi(v), v in mV (chunk 0)
        self.surrogate_beta = None  # Hz; set -> surrogate gradient below threshold (fits only)
        self.surrogate_mask = None  # neuron index array the surrogate applies to (None = all)

    PARAMS = ("log_tau", "bias", "gain")

    def set_trainable(self, *names, pools=None):
        """Turn on grad for `names`. pools: the current chunk's owned pool ids; the gradient
        outside them is zeroed, so earlier chunks' params (and unowned ones) stay frozen."""
        for h in getattr(self, "_hooks", []):
            h.remove()
        self._hooks = []
        mask = None
        if pools is not None:
            mask = torch.zeros_like(self.bias)
            mask[torch.as_tensor(np.asarray(pools, dtype=np.int64), device=self.device)] = 1
        for k in names:
            p = getattr(self, k)
            p.requires_grad_(True)
            if mask is not None:
                self._hooks.append(p.register_hook(lambda g, m=mask: g * m))

    def _set_frozen(self, W):
        """W: pre-major scipy CSR of the edges NOT carried by trainable edge gains."""
        # segment_reduce over CSR rows, NOT torch.sparse.mm: cuSPARSE SpMM is not
        # bit-deterministic on this GPU, and use_deterministic_algorithms(True) does not
        # fix it (no error raised, reruns still differ; checked 2026-09-24).
        t = lambda x, dt=torch.int64: torch.as_tensor(x, dtype=dt, device=self.device)  # noqa: E731
        WT = W.T.tocsr()  # post-major: input_i = sum_j W_ji r_j = (WT @ r)_i
        self.indptr, self.indices = t(WT.indptr), t(WT.indices)
        self.data = t(WT.data, self.dtype)[:, None]
        W = W.tocsr()
        self.indptr_pre, self.indices_pre = t(W.indptr), t(W.indices)
        self.data_pre = t(W.data, self.dtype)[:, None]
        # post-major position of every pre-major flat offset (set_plastic). Tag each edge with its
        # pre-major rank and push it through the SAME transpose as WT. float64: nnz ~2.7M, and
        # float32 loses integers above 2^24.
        tag = sp.csr_matrix((np.arange(1, W.nnz + 1, dtype=np.float64), W.indices, W.indptr), shape=W.shape)
        pre_of_post = tag.T.tocsr().data.astype(np.int64) - 1
        pre2post = np.empty_like(pre_of_post)
        pre2post[pre_of_post] = np.arange(len(pre_of_post))
        self._pre2post, self._pre_nnz = t(pre2post), W.nnz

    def set_plastic(self, offsets, values):
        """Overwrite edges at flat PRE-MAJOR offsets of brain_gpu.npz (learn/plastic.py's offsets).
        values: signed synapse counts (NOT x W_SYN; w_scale is applied in run). Updates both CSR
        copies and self.W. self.W is always the FULL matrix, so `offsets` index it directly; with edge
        gains active the CSR copies hold only the un-gained edges, so the offsets are remapped there
        and an offset that lands on a gained edge is refused (its weight lives in t_base * gain)."""
        assert self.W.nnz == self._full_nnz, "W changed shape since _set_frozen"
        assert self.data_pre.shape[0] == self._pre_nnz, "frozen CSR changed since _set_frozen"
        off_np = np.asarray(offsets, np.int64)
        if self._full2frozen is None:
            froz_np = off_np
        else:
            froz_np = self._full2frozen[off_np]
            assert (froz_np >= 0).all(), "plastic edge carried by an edge gain (%d of %d offsets)" % (
                int((froz_np < 0).sum()), len(off_np))
        froz = torch.as_tensor(froz_np, device=self.device)
        val = torch.as_tensor(np.asarray(values), device=self.device, dtype=self.dtype)[:, None]
        self.data_pre[froz] = val
        self.data[self._pre2post[froz]] = val
        self.W.data[off_np] = np.asarray(values, self.W.dtype)

    def set_edge_gains(self, target_pools, cap):
        """One trainable gain per (source pool, target pool) edge type, for every edge INTO
        `target_pools` (Lappalainen's per-edge-type scale factors, D48). gain = cap*sigmoid(x),
        so it lies in [0, cap]; x starts at gain 1. Returns the group table (src, tgt, edges, syn)."""
        pool = self.pool.cpu().numpy()
        C = self.W.tocoo()
        m = np.isin(pool[C.col], np.asarray(target_pools))
        pair = pool[C.row[m]].astype(np.int64) * (int(pool.max()) + 1) + pool[C.col[m]]
        keys, grp = np.unique(pair, return_inverse=True)
        n = self.n
        # Tag every kept edge with its FULL pre-major offset + 1 (self.W is canonical CSR, so tocoo()
        # order == flat offset order) and push the tags through the same build + eliminate_zeros as the
        # data, so full2frozen cannot drift from the frozen CSR.
        assert self.W.has_canonical_format, "set_edge_gains needs a canonical (sorted, no duplicate) W"
        tag = np.where(m | (C.data == 0), 0, np.arange(1, len(C.data) + 1)).astype(np.float64)
        frozen = sp.csr_matrix((tag, (C.row, C.col)), shape=(n, n))
        frozen.eliminate_zeros()
        full2frozen = np.full(len(C.data), -1, np.int64)
        kept = frozen.data.astype(np.int64) - 1
        full2frozen[kept] = np.arange(len(kept))
        frozen.data = C.data[kept]
        self._set_frozen(frozen)
        self._full2frozen = full2frozen
        order = np.lexsort((C.row[m], C.col[m]))  # post-major, deterministic order
        tgt, src = C.col[m][order], C.row[m][order]
        t = lambda x, dt=torch.int64: torch.as_tensor(x, dtype=dt, device=self.device)  # noqa: E731
        self.t_indptr = t(np.searchsorted(tgt, np.arange(n + 1)))
        self.t_src, self.t_grp = t(src), t(grp[order])
        self.t_base = t(C.data[m][order], self.dtype)[:, None]
        self.edge_cap = float(cap)
        self.edge_x = torch.full((len(keys),), float(np.log(1.0 / (cap - 1.0))),
                                 device=self.device, dtype=self.dtype)
        P = int(pool.max()) + 1
        return {"src_pool": keys // P, "tgt_pool": keys % P,
                "n_edges": np.bincount(grp, minlength=len(keys)),
                "n_syn": np.bincount(grp, weights=np.abs(C.data[m]), minlength=len(keys))}

    def edge_gain(self):
        return self.edge_cap * torch.sigmoid(self.edge_x)

    def get_params(self, pools):
        pools = np.asarray(pools, dtype=np.int64)
        return {k: getattr(self, k).detach()[pools].cpu().numpy() for k in self.PARAMS}

    def load_params(self, pools, vals):
        pools = torch.as_tensor(np.asarray(pools, dtype=np.int64), device=self.device)
        with torch.no_grad():
            for k in self.PARAMS:
                getattr(self, k)[pools] = torch.as_tensor(vals[k], device=self.device,
                                                          dtype=self.dtype)

    def spectral_bound(self, iters=100):
        """Perron root of |W| (power iteration): an upper bound on rho(W), synapse units."""
        A = abs(self.W).T.tocsr().astype(np.float64)
        x = np.ones(self.n)
        lam = 0.0
        for _ in range(iters):
            y = A @ x
            lam = float(np.linalg.norm(y) / np.linalg.norm(x))
            x = y / np.linalg.norm(y)
        return lam

    def run(self, drives, t_run, silence=None, state=None, return_state=False, record=None):
        """drives: list of B (idx, hz) pairs -> mean output rates [B, N] over t_run (ms).

        record: index array -> also traces [T, B, len(record)].
        return_state: also v [N, B]; pass it back as `state` to continue exactly.
        """
        N, B = self.n, len(drives)
        f = dict(device=self.device, dtype=self.dtype)
        clamp = torch.zeros(N, B, dtype=torch.bool, device=self.device)
        hz = torch.zeros(N, B, **f)
        driven = set()
        for b, (idx, rate) in enumerate(drives):
            idx = torch.as_tensor(np.asarray(idx, dtype=np.int64), device=self.device)
            clamp[idx, b] = True
            hz[idx, b] = torch.as_tensor(rate, **f)
            driven.update(idx.tolist())
        out = torch.ones(N, 1, **f)
        if silence is not None:
            silence = np.asarray(silence, dtype=np.int64)
            assert not driven.intersection(silence.tolist()), \
                "silence overlaps the driven set; a driven neuron cannot be silenced"
            out[torch.as_tensor(silence, device=self.device)] = 0.0
        v = torch.zeros(N, B, **f) if state is None else state
        tau = torch.exp(self.log_tau)[self.pool][:, None]
        gain = self.gain[self.pool][:, None]
        bias = self.bias[self.pool][:, None]
        steps = int(round(t_run / DT_MS))
        eg = None if self.edge_x is None else self.edge_gain()[:, None]
        smask = torch.ones(N, 1, dtype=torch.bool, device=self.device)
        if self.surrogate_mask is not None:
            smask = torch.zeros_like(smask)
            smask[torch.as_tensor(np.asarray(self.surrogate_mask, dtype=np.int64), device=self.device)] = True
        acc = torch.zeros(N, B, **f)
        trace = []
        for _ in range(steps):
            if self.transfer == "lif":
                assert self.surrogate_beta is None, "surrogate is defined for the lin transfer only"
                rv = lif_fi(v)
            else:
                rv = torch.relu(v) if self.surrogate_beta is None else _SurrogateReLU.apply(
                    v, self.surrogate_beta, smask)
            r = torch.where(clamp, hz, rv.clamp(max=self.r_max)) * out
            syn = _SpMM.apply(r, self.data, self.indices, self.indptr,
                              self.data_pre, self.indices_pre, self.indptr_pre)
            if self.edge_x is not None:
                syn = syn + torch.segment_reduce(
                    self.t_base * eg[self.t_grp] * r[self.t_src], "sum",
                    offsets=self.t_indptr, axis=0)
            inp = self.w_scale * syn
            v = v + DT_MS / tau * (-v + gain * inp + bias)
            acc = acc + r
            if record is not None:
                trace.append(r[record].T)
        rates = (acc / steps).T
        res = [rates]
        if record is not None:
            res.append(torch.stack(trace))
        if return_state:
            res.append(v)
        return res[0] if len(res) == 1 else tuple(res)
