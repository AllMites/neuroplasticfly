"""The PAM/PPL1 floor-1 patch: only target columns changed, raw counts, signs and metadata intact."""
import os, sys
import numpy as np
import scipy.sparse as sp
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gpu_sim as G

NT_SIGN = {"acetylcholine": 1.0, "dopamine": 1.0, "serotonin": 1.0,
           "octopamine": 1.0, "gaba": -1.0, "glutamate": -1.0}
DATA = os.path.join(G._HERE, "data")
ORIG = os.path.join(DATA, "brain_gpu.npz")
PATCHED = os.path.join(DATA, "brain_gpu_danfloor1.npz")

a = np.load(ORIG, allow_pickle=False)
b = np.load(PATCHED, allow_pickle=False)
n = int(a["n_neurons"])
meta = np.load(os.path.join(DATA, "neuron_meta.npz"), allow_pickle=False)
ct = meta["cell_type"].astype(str)
top_nt = np.load(os.path.join(DATA, "nt_conf.npz"), allow_pickle=False)["top_nt"].astype(str)
is_pam = np.char.startswith(ct, "PAM")
is_ppl1 = np.char.startswith(ct, "PPL1")
tmask = is_pam | is_ppl1
floor = int(b["dan_floor"])
assert str(b["dan_floor_targets"][0]) == "pam,ppl1", str(b["dan_floor_targets"])
assert floor == 1, "expected floor 1, got %d" % floor

A = sp.csr_matrix((a["W_data"], a["W_indices"], a["W_indptr"]), shape=(n, n))
B = sp.csr_matrix((b["W_data"], b["W_indices"], b["W_indptr"]), shape=(n, n))

# 1. The nnz/n_edges contract gpu_sim asserts on load.
assert int(b["n_edges"]) == len(b["W_data"]), \
    "n_edges %d != len(W_data) %d" % (int(b["n_edges"]), len(b["W_data"]))
assert int(a["n_edges"]) == len(a["W_data"])

# 2. Every non-target postsynaptic column is untouched. Zeroing the target
#    columns in both and comparing is stronger than comparing row counts: it
#    catches a value change as well as an added or dropped edge.
def zero_targets(M):
    """Same matrix with every edge onto a target neuron dropped."""
    rows = np.repeat(np.arange(n, dtype=np.int64), np.diff(M.indptr.astype(np.int64)))
    keep = ~tmask[M.indices]
    out = sp.csr_matrix((M.data[keep], (rows[keep], M.indices[keep].astype(np.int64))),
                        shape=(n, n), dtype=np.float32)
    out.sort_indices()
    return out


A0, B0 = zero_targets(A), zero_targets(B)
assert (A0 != B0).nnz == 0, \
    "%d non-target entries differ between the two matrices" % (A0 != B0).nnz
print("ok non-target columns byte-identical (%d edges)" % A0.nnz)

# 3. Every edge onto a target is a raw synapse count at or above the floor, with
#    the presynaptic neuron's neurotransmitter sign (unlisted NT -> excitatory).
pre_b = np.repeat(np.arange(n, dtype=np.int64), np.diff(B.indptr.astype(np.int64)))
tsel = tmask[B.indices]
tw, tpre = B.data[tsel], pre_b[tsel]
assert (np.abs(tw) >= floor).all(), "%d target edges below floor %d" % (
    int((np.abs(tw) < floor).sum()), floor)
assert np.array_equal(np.abs(tw), np.round(np.abs(tw))), \
    "target weights are not exact integer synapse counts"
want = np.array([NT_SIGN.get(s, 1.0) for s in top_nt], np.float32)[tpre]
assert np.array_equal(np.sign(tw), want), \
    "%d target edges carry the wrong sign for their presynaptic NT" % \
    int((np.sign(tw) != want).sum())
print("ok %d target edges: |w| >= %d, integer, NT sign" % (len(tw), floor))

# 4. Sign-per-neuron: a presynaptic neuron that had out-edges before must not
#    emit both signs now. Dale's law is what the sign rule encodes; a mapping
#    slip in the patch would show up here as a mixed-sign row.
pre_a = np.repeat(np.arange(n, dtype=np.int64), np.diff(A.indptr.astype(np.int64)))
had = np.zeros(n, bool)
had[pre_a] = True
sgn_a = np.zeros(n, np.float32)
sgn_a[pre_a] = np.sign(A.data)
mixed = np.flatnonzero(np.bincount(pre_b, weights=(np.sign(B.data) != sgn_a[pre_b]),
                                   minlength=n) * had)
assert mixed.size == 0, \
    "%d presynaptic neurons flipped or mixed sign, e.g. %s" % (mixed.size, mixed[:5])
print("ok sign-per-neuron invariant held for %d presynaptic neurons" % int(had.sum()))

# 5. The in-degrees the patch claims.
deg = np.bincount(B.indices, minlength=n)
for name, m, want_deg in (("PAM", is_pam, 274.8), ("PPL1", is_ppl1, 1576.8)):
    got = float(deg[m].mean())
    assert abs(got - want_deg) <= 0.02 * want_deg, \
        "%s mean in-degree %.1f, expected %.1f +-2%%" % (name, got, want_deg)
    print("ok %s mean in-degree %.1f (expected %.1f)" % (name, got, want_deg))

# 6. Everything that is not the matrix is carried over untouched.
for k in a.files:
    if k.startswith("W_") or k == "n_edges":
        continue
    assert k in b.files, "patched npz dropped key %r" % k
    assert np.array_equal(a[k], b[k]), "key %r changed" % k
assert set(b.files) - set(a.files) == {"dan_floor", "dan_floor_targets"}, \
    "unexpected new keys %s" % (set(b.files) - set(a.files))
print("ok selector_*/pool_*/n_neurons/min_syn carried over (min_syn %d)"
      % int(b["min_syn"]))

# 7. The real loader. This is the point of the test: gpu_sim's own asserts
#    (139248 neurons, nnz == n_edges, exactly 3802 eLN->ALPN edges) have to pass
#    on the patched file, or no later run can use it.
import torch
dev = "cuda" if torch.cuda.is_available() else "cpu"
sim = G.GpuSim(n=G.Brain(PATCHED), device=dev, compile_=(dev == "cuda"))
assert sim.net.n_edges == int(b["n_edges"])
print("ok GpuSim constructed on %s from the patched npz" % dev)

print("ok")
