"""The only synapses that change: KC->MBON, dopamine-gated depression.

Rule (Bennett, Philippides, Nowotny 2021; Hige et al. 2015), once per trial:
    e[edge]  = KC_rate[pre]                          eligibility (Hz)
    d[edge]  = DAN_rate pooled by the valence that opposes the MBON's valence
               (punish -> approach MBONs, reward -> avoid MBONs)          (Hz)
    dw       = -eta * e * d * w0                     depression only, scaled by w0
    w        = clip(w + dw, W_MIN*w0, w0) ; w += lam * (w0 - w)   floor, slow recovery

AUTHORED: eta, lam, W_MIN, the compartment table. REAL: the 18,674 edges, their
baseline counts, the KC/MBON/DAN identities, every rate that feeds the rule.
State = dw (w - w0) per edge, saved with a log; w0 is recomputed from the npz.
"""
import hashlib
import json
import os

import numpy as np

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
META = os.path.join(_HERE, "data", "neuron_meta.npz")
BRAIN = os.environ.get("FLYCHESS_BRAIN", os.path.join(_HERE, "data", "brain_gpu.npz"))
OFFSETS = os.path.join(_HERE, "data", "plastic_offsets.npz")
TABLE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "compartments.json")

ETA = 1e-3          # per Hz*Hz per trial
LAM = 0.01          # recovery fraction per trial
W_MIN = 0.1         # floor as fraction of baseline
OPPOSES = {"approach": "punish", "avoid": "reward"}


def _edge_fp(nnz, mbon_types):
    """Binds an edge set to the matrix nnz and the MBON type list that selected it."""
    h = hashlib.sha256("|".join(sorted(mbon_types)).encode()).hexdigest()[:16]
    return "%d:%s" % (nnz, h)


class Plastic:
    def __init__(self, w0, kc_of_edge, mbon_of_edge, mbon_valence_of_edge,
                 mbon_comp_of_edge, dan_idx, dan_comp, n_neurons, offsets=None, sign=None,
                 edge_fp=None):
        self.w0 = w0.astype(np.float32)
        self.dw = np.zeros_like(self.w0)
        self.kc_of_edge = kc_of_edge
        self.mbon_of_edge = mbon_of_edge
        self.mbon_valence_of_edge = mbon_valence_of_edge
        self.mbon_comp_of_edge = mbon_comp_of_edge
        self.dan_idx = dan_idx            # {"punish": int64[], "reward": int64[]}
        self.dan_comp = dan_comp          # {"punish": str[], "reward": str[]} parallel to dan_idx
        self.n_neurons = n_neurons
        self.offsets = offsets            # flat CSR offsets (real only)
        self.sign = np.ones_like(self.w0) if sign is None else sign.astype(np.float32)
        self.edge_fp = edge_fp            # which matrix + MBON list these edges came from
        self.n_edges = len(self.w0)
        # A clipped edge does not rest at W_MIN*w0: update() applies recovery in the same
        # call, so it settles at W_MIN*w0 + lam*(1-W_MIN)*w0. Testing w() <= W_MIN*w0 is
        # unreachable for every edge with w0 > 0, which is why n_at_floor read 0 while
        # min_frac sat at -(1-W_MIN)*(1-lam). update() rewrites this with the lam it used;
        # the default covers state loaded from disk (lam is not saved).
        self.floor = (W_MIN + LAM * (1.0 - W_MIN)) * self.w0

    # ------------------------------------------------------------ construction
    @classmethod
    def real(cls, brain=BRAIN):
        meta = np.load(META, allow_pickle=False)
        ct = meta["cell_type"].astype(str); cc = meta["cell_class"].astype(str)
        table = json.load(open(TABLE))
        b = np.load(brain, allow_pickle=False)
        ip, ix, w = b["W_indptr"], b["W_indices"], b["W_data"]
        n = len(ip) - 1
        fp = _edge_fp(len(w), table["mbon"])
        rows = np.repeat(np.arange(n), np.diff(ip))
        # One cache per matrix: the flat offsets differ between matrices with different
        # nnz, so keying the filename by nnz lets them alternate without manual deletes.
        offsets_path = OFFSETS.replace(".npz", "_%d.npz" % len(w))
        if os.path.exists(offsets_path):
            z = np.load(offsets_path)
            assert str(z["edge_fp"]) == fp,                 "stale %s (matrix nnz or MBON type list changed) "                 "- delete it to rebuild" % offsets_path
            off = z["offsets"]
        else:
            kc = cc == "Kenyon_Cell"; mb = np.isin(ct, list(table["mbon"]))
            off = np.flatnonzero(kc[rows] & mb[ix]).astype(np.int64)
            np.savez(offsets_path, offsets=off, edge_fp=fp)
        kc_e, mb_e = rows[off], ix[off]
        val = np.array([table["mbon"][ct[i]]["valence"] for i in mb_e])
        comp = np.array([table["mbon"][ct[i]]["compartment"] for i in mb_e])
        dan_idx, dan_comp = {}, {}
        for v in ("punish", "reward"):
            types = [k for k, r in table["dan"].items() if r["valence"] == v]
            idx = np.flatnonzero(np.isin(ct, types)).astype(np.int64)
            assert len(idx) > 0,                 "no %s DAN found - check the dan valences in learn/compartments.json" % v
            dan_idx[v] = idx; dan_comp[v] = np.array([table["dan"][ct[i]]["compartment"] for i in idx])
        w0 = np.abs(w[off])
        assert (w0 > 0).all(), "zero-weight KC->MBON edge: sign 0 would pin it forever"
        return cls(w0, kc_e, mb_e, val, comp, dan_idx, dan_comp, n, off,
                   sign=np.sign(w[off]), edge_fp=fp)

    @classmethod
    def toy(cls):
        # neurons 0-2 KC, 3 approach MBON, 4 avoid MBON, 5 punish DAN, 6 reward DAN
        kc = np.array([0, 1, 2, 0, 1, 2]); mb = np.array([3, 3, 3, 4, 4, 4])
        val = np.array(["approach"] * 3 + ["avoid"] * 3); comp = np.array(["a"] * 3 + ["b"] * 3)
        return cls(np.full(6, 8.0, np.float32), kc, mb, val, comp,
                   {"punish": np.array([5]), "reward": np.array([6])},
                   {"punish": np.array(["a"]), "reward": np.array(["b"])}, 7)

    # ------------------------------------------------------------ rule
    def w(self):
        return self.w0 + self.dw

    def update(self, rates, eta=ETA, lam=LAM, lesion="none", compartment=False):
        """rates: float[n_neurons] Hz for the trial just run. Returns dw applied this call."""
        rates = np.asarray(rates, np.float32)
        e = rates[self.kc_of_edge]
        d = np.zeros(self.n_edges, np.float32)
        for mv, dv in OPPOSES.items():
            if lesion in (dv, "all"):
                continue
            sel = self.mbon_valence_of_edge == mv
            # 2-char prefix match leaves some MBONs with an empty DAN set: MBON18/MBON19
            # (compartment "a2") miss PPL103 because its compartment reads "a'2a2" (prefix
            # "a'"), and MBON09 ("g3b'1") / MBON10 ("b'1") match no punish DAN at all. Those
            # edges get d = 0 and never depress. Unused by the default protocol.
            if compartment:
                for c in np.unique(self.mbon_comp_of_edge[sel]):
                    dans = self.dan_idx[dv][np.char.startswith(self.dan_comp[dv].astype(str), c[:2])]
                    m = sel & (self.mbon_comp_of_edge == c)
                    d[m] = rates[dans].mean() if len(dans) else 0.0
            else:
                d[sel] = rates[self.dan_idx[dv]].mean()
        step = -eta * e * d * self.w0
        w_new = np.clip(self.w() + step, W_MIN * self.w0, self.w0)
        w_new = w_new + lam * (self.w0 - w_new)
        self.floor = (W_MIN + lam * (1.0 - W_MIN)) * self.w0   # resting floor for this lam
        applied = w_new - self.w()
        self.dw = (w_new - self.w0).astype(np.float32)
        return applied

    # ------------------------------------------------------------ sim binding
    def signed_values(self, w_syn, scale=None):
        """Values to write into sim.data. w0 holds magnitudes; sign carries the raw
        matrix's polarity - 5 of the 18,674 edges are negative in v783 (KCg-m onto
        MBON01/05/09/11), so it must be reapplied or push() would flip them.

        `scale` is gpu_sim's per-neuron homeostatic normalisation factor
        (float32[n_neurons], all ones when NORM_TOTAL_TARGET is 0.0), indexed by the
        POSTsynaptic MBON. Without it push() would rewrite these 18,674 edges from
        w0 and silently revert them to unnormalised values after every training
        trial. It applies to EVERY edge, including the 5 negative ones: the scale
        is per-postsynaptic-neuron over total input magnitude and preserves that
        neuron's E/I ratio, so singling out the excitatory edges would break the
        very thing it preserves.
        """
        v = self.w() * self.sign * np.float32(w_syn)
        if scale is not None:
            v = v * np.asarray(scale, np.float32)[self.mbon_of_edge]
        return v.astype(np.float32)

    def push(self, sim):
        import gpu_sim as G
        sim.set_plastic(self.offsets,
                        self.signed_values(G.W_SYN, getattr(sim, "norm_scale", None)))

    # ------------------------------------------------------------ state
    def sha(self):
        return hashlib.sha256(self.dw.tobytes()).hexdigest()[:16]

    def save(self, d):
        os.makedirs(d, exist_ok=True)
        np.save(os.path.join(d, "dw.npy"), self.dw)
        json.dump({"n_edges": self.n_edges, "sha": self.sha(), "edge_fp": self.edge_fp},
                  open(os.path.join(d, "state.json"), "w"))

    def load(self, d):
        dw = np.load(os.path.join(d, "dw.npy")); meta = json.load(open(os.path.join(d, "state.json")))
        assert len(dw) == self.n_edges == meta["n_edges"], "state/edge mismatch"
        assert hashlib.sha256(dw.tobytes()).hexdigest()[:16] == meta["sha"], "state sha mismatch"
        assert meta.get("edge_fp") == self.edge_fp, "state belongs to a different edge set"
        self.dw = dw.astype(np.float32)

    def summary(self):
        f = self.dw / self.w0
        return {"mean_frac": float(f.mean()), "min_frac": float(f.min()),
                "n_at_floor": int((self.w() <= self.floor + 1e-5 * self.w0).sum()), "sha": self.sha()}
