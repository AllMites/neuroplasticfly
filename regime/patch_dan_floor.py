"""Restore floor-1 input pairs onto PAM/PPL1 in a copy of brain_gpu.npz.
READ ONLY on brain_gpu.npz. Writes data/brain_gpu_danfloor1.npz.
Run: .venv/Scripts/python.exe regime/patch_dan_floor.py [--floor 1] [--targets pam,ppl1]
"""
import argparse
import os

import numpy as np
import pyarrow.feather as pf
import scipy.sparse as sp

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BRAIN = os.path.join(HERE, "data", "brain_gpu.npz")
FEATHER = os.path.join(HERE, "data", "ext", "proofread_connections_783.feather")
OUT = os.path.join(HERE, "data", "brain_gpu_danfloor1.npz")

# Sign of a neuron's output edges, from its top neurotransmitter. This
# reproduces the loader that built brain_gpu.npz; anything unlisted (e.g.
# 'none') is excitatory, which is why the .get default is +1.0.
NT_SIGN = {"acetylcholine": 1.0, "dopamine": 1.0, "serotonin": 1.0,
           "octopamine": 1.0, "gaba": -1.0, "glutamate": -1.0}


def target_mask(cell_type, spec):
    """Boolean mask over neurons for --targets.

    Only pam,ppl1 exists. An all-dan variant was cut rather than shipped
    untested: the diff table below names PAM and PPL1 explicitly, so a wider
    target set would silently report on a subset of what it patched.
    """
    assert spec == "pam,ppl1", "only --targets pam,ppl1 is supported, got %r" % spec
    return (np.char.startswith(cell_type, "PAM")
            | np.char.startswith(cell_type, "PPL1"))


def raw_pairs(root_ids):
    """Raw pairs as (pre_idx, post_idx, syn_count, n_unmapped).

    n_unmapped is the number of feather rows dropped because one of their root
    ids is absent from the annotation table.
    """
    tb = pf.read_table(FEATHER,
                       columns=["pre_pt_root_id", "post_pt_root_id", "syn_count"])
    pre = tb.column("pre_pt_root_id").to_numpy()
    post = tb.column("post_pt_root_id").to_numpy()
    sc = tb.column("syn_count").to_numpy().astype(np.int64)
    del tb
    n = len(root_ids)
    ip = np.searchsorted(root_ids, pre)
    jp = np.searchsorted(root_ids, post)
    # range-check BEFORE the equality test: searchsorted returns n for anything
    # past the end, which would index out of bounds.
    ok = (ip < n) & (jp < n)
    ok &= (root_ids[np.minimum(ip, n - 1)] == pre)
    ok &= (root_ids[np.minimum(jp, n - 1)] == post)
    n_unmapped = int((~ok).sum())
    ip, jp, sc = ip[ok], jp[ok], sc[ok]
    key = ip.astype(np.int64) * n + jp
    uk, inv = np.unique(key, return_inverse=True)
    w = np.bincount(inv, sc).astype(np.int64)
    return ((uk // n).astype(np.int64), (uk % n).astype(np.int64), w, n_unmapped)


def csr_to_coo(indptr, indices, data, n):
    rows = np.repeat(np.arange(n, dtype=np.int64), np.diff(indptr.astype(np.int64)))
    return rows, indices.astype(np.int64), data.astype(np.float32)


def hop3_reach(csr, seeds, n):
    """Boolean 3-step reachability from `seeds` over the (presynaptic-major) CSR.

    Accumulation is float32, NOT int8. scipy accumulates the matvec in the
    matrix dtype, so an int8 pattern silently wraps any column whose
    contributor count is a multiple of 256 back to zero, dropping real
    frontier nodes (measured: 26667 int8 vs 26673 float64 at hop 3).
    """
    pat = sp.csr_matrix((np.ones(csr.nnz, np.float32), csr.indices, csr.indptr),
                        shape=(n, n))
    front = np.zeros(n, bool)
    front[seeds] = True
    seen = front.copy()
    for _ in range(3):
        front = (pat.T.dot(front.astype(np.float32)) > 0) & ~seen
        seen |= front
    return seen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--floor", type=int, default=1)
    ap.add_argument("--targets", default="pam,ppl1", choices=["pam,ppl1"])
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    assert a.floor <= 5, (
        "floor %d is above the shipped matrix's own >=5 floor, so the subset "
        "check below cannot verify it against brain_gpu.npz" % a.floor)

    src = np.load(BRAIN, allow_pickle=False)
    blob = {k: src[k] for k in src.files}
    n = int(blob["n_neurons"])

    meta = np.load(os.path.join(HERE, "data", "neuron_meta.npz"), allow_pickle=False)
    ct = meta["cell_type"].astype(str)
    cc = meta["cell_class"].astype(str)
    top_nt = np.load(os.path.join(HERE, "data", "nt_conf.npz"),
                     allow_pickle=False)["top_nt"].astype(str)
    root_ids = np.load(os.path.join(HERE, "data", "root_ids_sorted.npy"))
    assert (len(ct) == n and len(cc) == n and len(top_nt) == n
            and len(root_ids) == n)

    tmask = target_mask(ct, a.targets)
    is_pam = np.char.startswith(ct, "PAM")
    is_ppl1 = np.char.startswith(ct, "PPL1")
    print("targets %s: %d neurons (PAM %d, PPL1 %d)"
          % (a.targets, int(tmask.sum()), int(is_pam.sum()), int(is_ppl1.sum())))

    print("reading %s ..." % os.path.basename(FEATHER))
    pi, pj, pw, n_unmapped = raw_pairs(root_ids)
    print("raw pairs mapped: %d" % len(pi))
    print("feather rows dropped (root id not in annotations): %d" % n_unmapped)
    has_raw = np.zeros(n, bool)
    has_raw[pj] = True
    print("target neurons with zero raw inputs: %d of %d"
          % (int((tmask & ~has_raw).sum()), int(tmask.sum())))

    keep = tmask[pj] & (pw >= a.floor)
    npre, npost, ncnt = pi[keep], pj[keep], pw[keep]
    sign = np.array([NT_SIGN.get(s, 1.0) for s in top_nt], np.float32)
    ndata = (ncnt.astype(np.float32) * sign[npre]).astype(np.float32)

    rows, cols, data = csr_to_coo(blob["W_indptr"], blob["W_indices"],
                                  blob["W_data"], n)
    old_t = tmask[cols]
    n_removed = int(old_t.sum())

    # (d) Proof that the shipped npz really is raw-counts-at-floor-5: every old
    # edge onto a target must appear in the new set with the same value.
    okey = rows[old_t].astype(np.int64) * n + cols[old_t]
    nkey = npre * n + npost
    order = np.argsort(nkey, kind="stable")
    pos = np.searchsorted(nkey[order], okey)
    inb = pos < len(nkey)
    hit = np.zeros(len(okey), bool)
    hit[inb] = nkey[order][np.minimum(pos, len(nkey) - 1)][inb] == okey[inb]
    missing = int((~hit).sum())
    vmis = 0
    if missing == 0:
        vmis = int((ndata[order][pos] != data[old_t]).sum())
    if missing or vmis:
        raise SystemExit("BLOCKED: %d old target edges absent from the raw set, "
                         "%d with a different value; brain_gpu.npz is not raw "
                         "counts at floor 5" % (missing, vmis))
    print("subset check ok: all %d old target edges match raw counts" % n_removed)

    rows = np.concatenate([rows[~old_t], npre])
    cols = np.concatenate([cols[~old_t], npost])
    data = np.concatenate([data[~old_t], ndata])
    W = sp.csr_matrix((data, (rows, cols)), shape=(n, n), dtype=np.float32)
    assert W.nnz == len(data),         "duplicate (pre,post) after concat: %d entries -> %d nnz" % (len(data), W.nnz)
    W.sort_indices()

    old = sp.csr_matrix((blob["W_data"], blob["W_indices"],
                         blob["W_indptr"]), shape=(n, n))

    blob["W_indptr"] = W.indptr.astype(np.int64)
    blob["W_indices"] = W.indices.astype(np.int32)
    blob["W_data"] = W.data.astype(np.float32)
    blob["n_edges"] = np.int64(W.nnz)
    blob["dan_floor"] = np.int64(a.floor)
    blob["dan_floor_targets"] = np.array([a.targets])
    np.savez(a.out, **blob)
    print("wrote %s" % a.out)

    # ------------------------------------------------------------ diff table
    def stats(mask, mat):
        indeg = np.bincount(mat.indices, minlength=n)[mask]
        syn = np.bincount(mat.indices, weights=np.abs(mat.data),
                          minlength=n)[mask]
        return float(indeg.mean()), float(syn.mean()), int(indeg.sum())

    print("")
    print("group  pairs_before pairs_after  in-deg before -> after   syn/neuron before -> after")
    for name, m in (("PAM", is_pam), ("PPL1", is_ppl1)):
        d0, s0, p0 = stats(m, old)
        d1, s1, p1 = stats(m, W)
        print("%-6s %12d %11d  %13.1f -> %-9.1f %13.1f -> %.1f"
              % (name, p0, p1, d0, d1, s0, s1))
    print("nnz %d -> %d" % (old.nnz, W.nnz))
    print("edges removed %d, added %d" % (n_removed, len(npre)))
    for name, m in (("PAM", is_pam), ("PPL1", is_ppl1)):
        print("  %s: removed %d, added %d"
              % (name, int(np.count_nonzero(m[old.indices])),
                 int(np.count_nonzero(m[npost]))))

    # KC->PAM is the bulk of what comes back. By design, not filtered.
    is_kc = cc == "Kenyon_Cell"
    kc_new = is_kc[npre] & is_pam[npost]
    print("KC->PAM restored: %d pairs, %d syn (expected bulk; NOT filtered)"
          % (int(kc_new.sum()), int(ncnt[kc_new].sum())))

    # hop-2: sugar-GRN direct targets -> PAM pair count, floor 5 vs floor 1
    sugar = np.flatnonzero(meta["cell_sub_class"].astype(str) == "sugar/water")
    print("sugar GRNs: %d" % len(sugar))
    for label, mat in (("floor 5 (orig)", old), ("floor 1 (patched)", W)):
        hit1 = np.zeros(n, bool)
        for s in sugar:
            hit1[mat.indices[mat.indptr[s]:mat.indptr[s + 1]]] = True
        step2 = 0
        for s in np.flatnonzero(hit1):
            tgt = mat.indices[mat.indptr[s]:mat.indptr[s + 1]]
            step2 += int(np.count_nonzero(is_pam[tgt]))
        print("hop-2 sugar-GRN-target -> PAM pairs, %s: %d (via %d hop-1 neurons)"
              % (label, step2, int(hit1.sum())))

    seen = hop3_reach(W, sugar, n)
    print("sugar GRNs reach %d/%d PAM in 3 hops"
          % (int((seen & is_pam).sum()), int(is_pam.sum())))


if __name__ == "__main__":
    main()
