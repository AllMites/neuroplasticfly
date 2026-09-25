"""Freeze everything gpu_sim.py needs out of the container into one .npz.

The connectome, the annotation TSV, flypoke and pandas all live in the `sim`
container; the CUDA torch lives on the Windows host. Rather than duplicate the
loader (and risk the two engines disagreeing about which neuron is index 4711),
this script runs INSIDE the container, resolves every selector string and the
pooling table with the real `flypoke.data.Network`, and writes the answers next
to the shards. `gpu_sim.py` then reads only arrays - no pandas, no flypoke.

Exported (measured 2026-09-16): N=139248, W nnz=3,532,411, 8,865 pooling groups.

Run: docker compose run --rm sim python -u export_brain.py
"""
import os

import numpy as np

import behaviors as B
import encode
import reservoir as R

OUT = os.environ.get("FLYCHESS_BRAIN", "/app/data/brain_gpu.npz")

# Every selector string either engine ever passes to Network.select(). Resolved
# here once so the host never needs the annotation table. Keep this list in sync
# with encode.py / reservoir.py / behaviors.py if their selectors change; the
# host loader raises on an unknown selector rather than guessing.
SELECTORS = [
    encode.SUGAR, encode.BITTER, encode.GROOMING, encode.JOHNSTON,
    R.INGESTION, "super_class=central", "super_class=descending",
    "cell_type=LC4|LPLC2", "cell_type=DNp04|DNp01", "cell_type=DNg29|DNg84",
]
SELECTORS += ["cell_type=%s,side=%s" % (ct, side)
              for ct in R.DN_TYPES for side in ("left", "right")]


def orn_selectors(n):
    """Per-side ORN class selectors; regime/probe.py drives one channel alone."""
    types = n.neurons.cell_type.astype("string")
    is_orn = types.str.startswith("ORN_").fillna(False).to_numpy()
    return ["cell_type=%s,side=%s" % (ct, side)
            for ct in sorted(set(types[is_orn].tolist()))
            for side in ("left", "right")]


def main():
    n = B.net()
    selectors = SELECTORS + orn_selectors(n)
    W = n.W.tocsr()
    W.sort_indices()
    group_id, sizes, names = R.pooling(n)

    blob = {
        "n_neurons": np.int64(n.n),
        "min_syn": np.int64(n.min_syn),
        "W_indptr": W.indptr.astype(np.int64),
        "W_indices": W.indices.astype(np.int32),
        "W_data": W.data.astype(np.float32),
        "n_edges": np.int64(W.nnz),
        "pool_group_id": group_id.astype(np.int32),
        "pool_sizes": sizes.astype(np.float32),
        "pool_names": np.array(names),
        "selector_names": np.array(sorted(set(selectors))),
    }
    for k, sel in enumerate(blob["selector_names"]):
        blob["selector_%04d" % k] = n.select(str(sel)).astype(np.int64)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.savez(OUT, **blob)                       # uncompressed: load speed > 29 MB
    print("wrote %s" % OUT)
    print("  N=%d  nnz=%d  groups=%d  selectors=%d"
          % (n.n, W.nnz, len(names), len(blob["selector_names"])))
    print("  W rows presynaptic, cols postsynaptic, dtype=%s, sorted_indices=%s"
          % (W.data.dtype, W.has_sorted_indices))


if __name__ == "__main__":
    main()
