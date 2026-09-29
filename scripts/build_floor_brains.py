"""Build data/brain_gpu_min1.npz: the same connectome with a 1-synapse floor instead of 5.

The canonical brain (data/brain_gpu.npz, scripts/build_data.py) keeps an edge only if it
has >= 5 synapses. Shiu et al. 2024 and Christie et al. 2026 use no floor. This script
builds the floor-1 matrix for the Christie protocol-matched check (learn/christie_match.py).

One flypoke build_network(1) from the feather (the same loader as build_data.py). Every
non-W key (selectors, pooling, n_neurons) is copied from data/brain_gpu.npz unchanged;
min_syn and n_edges are set to the floor's.

Golden, asserted BEFORE anything is written: masking the floor-1 build at |w| >= 5 gives
data/brain_gpu.npz W_indptr, W_indices and W_data bit-exact INCLUDING sign. A mismatch
means masking does not reproduce a build at that floor, and nothing is written.

Never touches data/brain_gpu.npz or data/CHECKSUMS.json. Needs scripts/build_data.py first.

Run: python scripts/build_floor_brains.py
"""
import os
import time
from pathlib import Path

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXT = os.path.join(HERE, "data", "ext")
DATA = os.path.join(HERE, "data")
CANON = os.path.join(DATA, "brain_gpu.npz")
NNZ = {1: 15090883, 5: 2700429}     # edge census per floor, v783
W_KEYS = ("W_indptr", "W_indices", "W_data", "n_edges", "min_syn")


def floor1():
    from flypoke.data import build_network
    n = build_network(1, data_dir=Path(EXT), use_cache=True)
    W = n.W.tocsr()
    W.sort_indices()
    assert W.nnz == NNZ[1], "floor-1 nnz %d != census %d" % (W.nnz, NNZ[1])
    return W


def mask(W, k):
    M = W.copy()
    M.data[np.abs(M.data) < k] = 0.0
    M.eliminate_zeros()
    M.sort_indices()
    return M


def arrays(M):
    return (M.indptr.astype(np.int64), M.indices.astype(np.int32), M.data.astype(np.float32))


def golden(W, canon):
    ip, ix, w = arrays(mask(W, 5))
    for name, a in (("W_indptr", ip), ("W_indices", ix), ("W_data", w)):
        b = canon[name]
        assert a.dtype == b.dtype and a.shape == b.shape, "%s dtype/shape differ" % name
        assert np.array_equal(a, b), "GOLDEN FAIL: mask(5) %s != brain_gpu.npz" % name
    assert int(canon["n_edges"]) == len(w) == NNZ[5]
    assert (np.sign(w) == np.sign(canon["W_data"])).all()
    print("golden ok: mask(5) == brain_gpu.npz bit-exact (indptr, indices, signed data), nnz %d" % len(w))


def main():
    if not os.path.exists(CANON):
        raise SystemExit("missing %s\n  run: python scripts/build_data.py" % CANON)
    t0 = time.time()
    canon = np.load(CANON, allow_pickle=False)
    W = floor1()
    print("floor-1 build nnz %d (%.1f s)" % (W.nnz, time.time() - t0))
    golden(W, canon)
    ip, ix, w = arrays(mask(W, 1))     # as the private build did; a no-op on integer counts
    assert len(w) == NNZ[1]
    blob = {key: canon[key] for key in canon.files if key not in W_KEYS}
    blob.update(W_indptr=ip, W_indices=ix, W_data=w, n_edges=np.int64(len(w)), min_syn=np.int64(1))
    path = os.path.join(DATA, "brain_gpu_min1.npz")
    tmp = path + ".tmp.npz"
    np.savez(tmp, **blob)
    os.replace(tmp, path)
    print("wrote %s nnz %d synapses %.2f M (%.1f s)" % (path, len(w), float(np.abs(w).sum()) / 1e6,
                                                        time.time() - t0))


if __name__ == "__main__":
    main()
