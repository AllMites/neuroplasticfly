"""Build data/{brain_gpu.npz, neuron_meta.npz, root_ids_sorted.npy, nt_conf.npz} from
data/ext/, on the host, with flypoke.data.build_network -- the same loader the original
container build used. Replaces the container-only export_brain.py + readout_analysis
.export_meta(); both of those stay in the tree as the record of how the first build ran.

Falsifiers, all checked before anything reaches its final name:
  1. N == 139248, and the TSV's root_id sort IS the neuron index (root_ids_sorted).
  2. W.nnz == 2700429  (the corrected 2026-09-19 build; the bad 2026-09-16 one had
     3532411 because it summed duplicate neuropil rows differently).
  3. Content hashes equal data/CHECKSUMS.json (scripts/checksum_data.py --verify).

Run: python scripts/build_data.py [--no-cache] [--update-checksums]
     --no-cache          ignore data/ext/cache/W_min5.npz and re-aggregate the feather
     --update-checksums  republish CHECKSUMS.json instead of verifying. Only after a
                         deliberate, understood change to the inputs or the loader.
"""
import os
import sys
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXT = os.path.join(HERE, "data", "ext")
DATA = os.path.join(HERE, "data")
os.environ.setdefault("FLYPOKE_DATA", EXT)       # encode._soma_xyz reads the TSV from here
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pathlib import Path                          # noqa: E402

import numpy as np                                # noqa: E402

N_EXPECT = 139248
NNZ_EXPECT = 2700429

# ponytail: copied from export_brain.py (which imports `behaviors`, a container-only
# module). Keep in sync: if encode.py / reservoir.py grow a selector, both lists move.
import encode                                     # noqa: E402
import reservoir as R                             # noqa: E402

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


def _save(writer, path, **arrays):
    tmp = path + ".tmp.npz" if path.endswith(".npz") else path + ".tmp.npy"
    writer(tmp, **arrays) if arrays else writer(tmp)
    os.replace(tmp, path)


def build(use_cache=True):
    from flypoke.data import build_network

    feather = os.path.join(EXT, "proofread_connections_783.feather")
    if not os.path.exists(feather):
        raise SystemExit("missing %s\n  run: python scripts/fetch_data.py" % feather)

    t0 = time.time()
    n = build_network(5, data_dir=Path(EXT), use_cache=use_cache)
    print("network: N=%d  nnz=%d  (%.1f s)" % (n.n, n.W.nnz, time.time() - t0))

    # -- falsifier 1: the neuron index is the root_id sort of the TSV
    root_ids = n.neurons.root_id.to_numpy(np.int64)
    if n.n != N_EXPECT:
        raise SystemExit(
            "N=%d, expected %d. The annotation TSV is unversioned upstream and has "
            "changed. Use the commit recorded in data/ext/annotations_provenance.json, "
            "or rebuild every downstream artefact deliberately." % (n.n, N_EXPECT))
    if not np.array_equal(root_ids, np.sort(root_ids)):
        raise SystemExit("root_id column is not sorted; flypoke's index assumption broke")

    # -- falsifier 2: this is the corrected connectome
    W = n.W.tocsr()
    W.sort_indices()
    if W.nnz != NNZ_EXPECT:
        raise SystemExit(
            "W.nnz=%d, expected %d. The 2026-09-16 bad build had 3532411; do not ship a "
            "brain until you know which rule changed (top_nt NaN sign, duplicate "
            "summing, min_syn)." % (W.nnz, NNZ_EXPECT))

    group_id, sizes, names = R.pooling(n)           # module-level cache: call once
    selectors = sorted(set(SELECTORS + orn_selectors(n)))

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
        "selector_names": np.array(selectors),
    }
    for k, sel in enumerate(blob["selector_names"]):
        blob["selector_%04d" % k] = n.select(str(sel)).astype(np.int64)

    os.makedirs(DATA, exist_ok=True)
    _save(np.savez, os.path.join(DATA, "brain_gpu.npz"), **blob)  # uncompressed: load speed
    print("wrote brain_gpu.npz  groups=%d  selectors=%d" % (len(names), len(selectors)))

    tmp = os.path.join(DATA, "root_ids_sorted.npy.tmp.npy")
    np.save(tmp, root_ids)
    os.replace(tmp, os.path.join(DATA, "root_ids_sorted.npy"))
    print("wrote root_ids_sorted.npy")

    # -- neuron_meta: readout_analysis.export_meta's body, plus hemibrain_type (the
    #    column the 2026-09-19 build dropped and tools/add_hemibrain_type.py patched in)
    df = n.neurons
    xyz = encode._soma_xyz(n)

    def col(name):
        if name not in df.columns:
            return np.array(["missing"] * n.n)
        return df[name].astype("string").fillna("none").to_numpy().astype(str)

    _save(np.savez, os.path.join(DATA, "neuron_meta.npz"),
          group_id=group_id.astype(np.int32),
          group_sizes=sizes.astype(np.int32),
          group_names=np.array([str(x) for x in names]),
          xyz=xyz.astype(np.float32),
          side=col("side"), super_class=col("super_class"), cell_class=col("cell_class"),
          cell_sub_class=col("cell_sub_class"), cell_type=col("cell_type"),
          hemibrain_type=col("hemibrain_type"),
          columns=np.array([str(c) for c in df.columns]))
    print("wrote neuron_meta.npz")

    # -- nt_conf reads neuron_meta.npz back and asserts row-for-row alignment itself
    import regime.nt_conf as NT
    NT.main()
    return n


def main(argv):
    n = build(use_cache="--no-cache" not in argv)
    import checksum_data as CS
    if "--update-checksums" in argv:
        CS.main(["--write"])
        print("checksums REPUBLISHED (you asked for it)")
        return 0
    rc = CS.main(["--verify"])
    if rc:
        print("\nThe rebuild is not the published brain. Diff before trusting either side:\n"
              "  W_data edge by edge, then top_nt NaN handling, then min_syn.")
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
