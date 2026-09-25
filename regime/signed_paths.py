"""Signed-path analysis sugar GRN -> PAM on the DAN-floor-patched matrix (L1),
plus the KC->PAM-stripped variant npz (L3 structural half).

READ ONLY on every existing npz. Writes:
  results/signed_paths.json
  data/brain_gpu_danfloor1_nokc.npz   (only with --write-nokc; refuses to overwrite)

Run:
  .venv/Scripts/python.exe regime/signed_paths.py --write-nokc

Conventions copied from regime/patch_dan_floor.py so the numbers are comparable:
 - CSR is presynaptic-major: W[pre, post]; forward propagation is W.T @ v.
 - Edge sign is the PRESYNAPTIC neuron's top_nt (NT_SIGN, default +1, incl. 'none').
 - index <-> root_id via data/root_ids_sorted.npy (annotations join on it).

Why "signed net" is computed by vector propagation rather than a sparse matrix
power: W is 139248^2 with 2.8M nnz; W.T @ W densifies far past memory.
Propagating a signed vector and an absolute-value vector separately gives the
same two aggregates (sum of signed path products, sum of |path products|) in
O(nnz) per hop; excitatory / inhibitory path mass then split out as
(abs +/- signed)/2. This counts every walk of exactly k steps, weighted by the
product of synapse counts along it - a structural quantity, not a rate.
"""
import argparse
import csv as _csv
import json
import os

import numpy as np
import scipy.sparse as sp

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(HERE, "data", "brain_gpu.npz")
PATCHED = os.path.join(HERE, "data", "brain_gpu_danfloor1.npz")
NOKC = os.path.join(HERE, "data", "brain_gpu_danfloor1_nokc.npz")
ANN = os.path.join(HERE, "data", "ext", "Supplemental_file1_neuron_annotations.tsv")
OUT = os.path.join(HERE, "results", "signed_paths.json")

NT_SIGN = {"acetylcholine": 1.0, "dopamine": 1.0, "serotonin": 1.0,
           "octopamine": 1.0, "gaba": -1.0, "glutamate": -1.0}


def load_csr(path):
    b = np.load(path, allow_pickle=False)
    n = int(b["n_neurons"])
    W = sp.csr_matrix((b["W_data"], b["W_indices"], b["W_indptr"]), shape=(n, n))
    return b, W, n


def hemibrain_type(root_ids):
    """hemibrain_type per neuron index, '' where the annotation is empty.

    neuron_meta.npz does not carry hemibrain_type - PAM01..PAM15 live only in
    the annotations tsv - so it is joined here on root_id.
    """
    hb = {}
    with open(ANN, newline="", encoding="utf-8") as fh:
        for r in _csv.DictReader(fh, delimiter="\t"):
            v = (r.get("hemibrain_type") or "").strip()
            if v:
                hb[int(r["root_id"])] = v
    return np.array([hb.get(int(x), "") for x in root_ids], dtype=object)


def coo_rows(M, n):
    return np.repeat(np.arange(n, dtype=np.int64), np.diff(M.indptr.astype(np.int64)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write-nokc", action="store_true")
    a = ap.parse_args()

    meta = np.load(os.path.join(HERE, "data", "neuron_meta.npz"), allow_pickle=False)
    ct = meta["cell_type"].astype(str)
    cc = meta["cell_class"].astype(str)
    csc = meta["cell_sub_class"].astype(str)
    top_nt = np.load(os.path.join(HERE, "data", "nt_conf.npz"),
                     allow_pickle=False)["top_nt"].astype(str)
    root_ids = np.load(os.path.join(HERE, "data", "root_ids_sorted.npy"))

    bp, W, n = load_csr(PATCHED)
    bb, W0, n0 = load_csr(BASE)
    assert n == n0
    is_pam = np.char.startswith(ct, "PAM")
    is_ppl1 = np.char.startswith(ct, "PPL1")
    is_kc = cc == "Kenyon_Cell"
    sugar = np.flatnonzero(csc == "sugar/water").astype(np.int64)
    pam = np.flatnonzero(is_pam).astype(np.int64)

    res = {"brain_patched": os.path.abspath(PATCHED),
           "n_edges_patched": int(W.nnz), "n_edges_base": int(W0.nnz),
           "n_sugar": int(len(sugar)), "n_pam": int(len(pam))}
    print("patched nnz %d, base nnz %d, sugar GRNs %d, PAM %d"
          % (W.nnz, W0.nnz, len(sugar), len(pam)))

    # ---------------------------------------------------- restored edge set
    # Present in the patched matrix, absent from the base one. Keys fit int64
    # (139248^2 ~ 1.9e10).
    rows_all = coo_rows(W, n)
    kp = rows_all * n + W.indices.astype(np.int64)
    kb = coo_rows(W0, n) * n + W0.indices.astype(np.int64)
    added = ~np.isin(kp, kb, assume_unique=True)
    add_idx = np.flatnonzero(added)
    add_pre, add_post, add_w = rows_all[added], W.indices[added].astype(np.int64), W.data[added]
    print("restored (added) edges: %d" % len(add_idx))
    assert np.all(is_pam[add_post] | is_ppl1[add_post]), "added edges land outside PAM/PPL1"

    ap_m = is_pam[add_post]
    a_pre, a_post, a_w = add_pre[ap_m], add_post[ap_m], add_w[ap_m]
    a_kc = is_kc[a_pre]
    a_syn = np.abs(a_w)
    a_inh = a_w < 0

    def frac(mask):
        tot = float(a_syn[mask].sum())
        inh = float(a_syn[mask & a_inh].sum())
        return {"pairs": int(mask.sum()), "syn": tot, "syn_inhib": inh,
                "frac_syn_inhib": (inh / tot if tot else 0.0),
                "pairs_inhib": int((mask & a_inh).sum()),
                "frac_pairs_inhib": float((mask & a_inh).sum()) / max(int(mask.sum()), 1)}

    res["restored_pam"] = {"all": frac(np.ones(len(a_w), bool)),
                           "kc": frac(a_kc), "non_kc": frac(~a_kc)}
    for k, v in res["restored_pam"].items():
        print("restored PAM input [%7s]: %6d pairs, %9.0f syn, inhib syn frac %.4f"
              % (k, v["pairs"], v["syn"], v["frac_syn_inhib"]))

    per = np.bincount(a_post, weights=a_syn, minlength=n)[pam]
    per_nk = np.bincount(a_post[~a_kc], weights=a_syn[~a_kc], minlength=n)[pam]
    res["restored_per_pam"] = {
        "syn_mean": float(per.mean()), "syn_max": float(per.max()),
        "syn_zero_count": int((per == 0).sum()),
        "non_kc_syn_mean": float(per_nk.mean()), "non_kc_syn_max": float(per_nk.max()),
        "non_kc_syn_zero_count": int((per_nk == 0).sum())}
    print("per-PAM restored syn: mean %.1f max %.0f zeros %d | non-KC mean %.1f max %.0f zeros %d"
          % (per.mean(), per.max(), (per == 0).sum(),
             per_nk.mean(), per_nk.max(), (per_nk == 0).sum()))

    nt_rows = {}
    for s in np.unique(top_nt[a_pre[~a_kc]]):
        m = (~a_kc) & (top_nt[a_pre] == s)
        nt_rows[str(s)] = {"pairs": int(m.sum()), "syn": float(a_syn[m].sum()),
                           "sign": NT_SIGN.get(str(s), 1.0)}
    res["restored_pam_non_kc_by_nt"] = nt_rows
    print("")
    print("| restored non-KC -> PAM, presyn top_nt | sign | pairs | syn |")
    for s, v in sorted(nt_rows.items(), key=lambda t: -t[1]["syn"]):
        print("| %-20s | %+.0f | %6d | %9.0f |" % (s, v["sign"], v["pairs"], v["syn"]))

    # ------------------------------------------------------ signed hop sums
    Wa = sp.csr_matrix((np.abs(W.data), W.indices, W.indptr), shape=(n, n))
    v0 = np.zeros(n, np.float64)
    v0[sugar] = 1.0
    vs1, va1 = W.T.dot(v0), Wa.T.dot(v0)
    vs2, va2 = W.T.dot(vs1), Wa.T.dot(va1)
    vs3, va3 = W.T.dot(vs2), Wa.T.dot(va2)

    def split(sgn, absv):
        ex = 0.5 * (absv + sgn)
        ih = 0.5 * (absv - sgn)
        return {"signed_net": float(sgn), "abs_total": float(absv),
                "excit": float(ex), "inhib": float(ih),
                "frac_inhib": float(ih / absv) if absv else 0.0}

    res["hops"] = {"hop1": split(vs1[pam].sum(), va1[pam].sum()),
                   "hop2": split(vs2[pam].sum(), va2[pam].sum()),
                   "hop3": split(vs3[pam].sum(), va3[pam].sum())}
    print("")
    for k in ("hop1", "hop2", "hop3"):
        v = res["hops"][k]
        print("%s sugar->PAM: signed_net %+.4e  abs %.4e  frac_inhib %.4f"
              % (k, v["signed_net"], v["abs_total"], v["frac_inhib"]))

    # ------------------------------------- intermediate breakdown (hop2/hop3)
    # Contribution of the LAST intermediate m to the hop-k total at PAM is
    #   v_{k-1}[m] * sum_{p in PAM} W[m, p].
    colsel = is_pam[W.indices]
    to_pam_s = np.bincount(rows_all[colsel], weights=W.data[colsel], minlength=n)
    to_pam_a = np.bincount(rows_all[colsel], weights=np.abs(W.data[colsel]), minlength=n)

    hb = hemibrain_type(root_ids)

    def top_types(vprev_s, vprev_a, label, k=25):
        cs = vprev_s * to_pam_s
        ca = vprev_a * to_pam_a
        idx = np.flatnonzero(ca != 0)
        rows = {}
        for i in idx:
            key = "%s|%s|%s" % (cc[i], ct[i], hb[i])
            r = rows.setdefault(key, [0.0, 0.0, 0])
            r[0] += float(cs[i])
            r[1] += float(ca[i])
            r[2] += 1
        out = sorted(rows.items(), key=lambda t: -t[1][1])[:k]
        print("")
        print("%s intermediates (top %d by |contribution|), %d contributing neurons"
              % (label, k, len(idx)))
        print("| cell_class | cell_type | hemibrain_type | n | signed | abs | frac_inhib |")
        recs = []
        for key, (s, av, cnt) in out:
            c_, t_, h_ = key.split("|")
            fi = 0.5 * (av - s) / av if av else 0.0
            print("| %s | %s | %s | %d | %+.3e | %.3e | %.3f |" % (c_, t_, h_, cnt, s, av, fi))
            recs.append({"cell_class": c_, "cell_type": t_, "hemibrain_type": h_,
                         "n": cnt, "signed": s, "abs": av, "frac_inhib": fi})
        return recs, cs, ca

    res["hop2_intermediates"], cs2, ca2 = top_types(vs1, va1, "hop-2")
    res["hop3_intermediates"], cs3, ca3 = top_types(vs2, va2, "hop-3 (last hop before PAM)")

    ord2 = np.argsort(-np.abs(ca2))
    top2 = [int(i) for i in ord2[:200] if ca2[i] != 0]
    res["hop2_top_neurons"] = top2
    res["hop2_top_types"] = sorted({str(ct[i]) for i in top2})

    # plan's guessed US carriers: restored non-KC presynaptic partners of PAM
    # with no cell_class / cell_type label ("none"-class SMP/CRE).
    presyn_pam = np.unique(a_pre[~a_kc])
    unl = [int(i) for i in presyn_pam
           if str(cc[i]).strip() in ("", "none") or str(ct[i]).strip() in ("", "none")]
    res["pam_presyn_non_kc_n"] = int(len(presyn_pam))
    res["pam_presyn_unlabelled_n"] = len(unl)
    res["pam_presyn_unlabelled_idx"] = unl
    print("")
    print("restored non-KC PAM presynaptic partners: %d neurons, unlabelled class/type: %d"
          % (len(presyn_pam), len(unl)))

    # ------------------------------------------------------------ nokc npz
    if a.write_nokc:
        assert not os.path.exists(NOKC), "refusing to overwrite %s" % NOKC
        drop = np.zeros(W.nnz, bool)
        drop[add_idx[ap_m][a_kc]] = True
        keep = ~drop
        M = sp.csr_matrix((W.data[keep], (rows_all[keep], W.indices[keep].astype(np.int64))),
                          shape=(n, n), dtype=np.float32)
        M.sort_indices()
        blob = {k: bp[k] for k in bp.files}
        blob["W_indptr"] = M.indptr.astype(np.int64)
        blob["W_indices"] = M.indices.astype(np.int32)
        blob["W_data"] = M.data.astype(np.float32)
        blob["n_edges"] = np.int64(M.nnz)
        blob["dan_floor_targets"] = np.array(["pam,ppl1;restored KC->PAM stripped"])
        np.savez(NOKC, **blob)
        print("wrote %s  nnz %d -> %d (dropped %d restored KC->PAM edges)"
              % (NOKC, W.nnz, M.nnz, int(drop.sum())))
        res["nokc"] = {"path": os.path.abspath(NOKC), "n_edges": int(M.nnz),
                       "dropped": int(drop.sum())}

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=1)
    print("")
    print("wrote %s" % OUT)


if __name__ == "__main__":
    main()
