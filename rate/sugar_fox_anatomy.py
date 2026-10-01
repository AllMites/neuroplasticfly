"""Sugar GRN -> Fox (CB0525) route anatomy on FlyWire v783, at synapse floors 1 and 5.

Descriptive connectome metadata, no simulation, no label. Pair counts are summed over
neuropils and then floored, the same convention as the brain build (regime/patch_dan_floor.py
raw_pairs). Influence uses input fractions: F[i, j] = w_ij / (total input to j at that floor),
signed by the presynaptic top_nt (GABA, Glu -> -1; everything else +1, as NT_SIGN).
k-hop influence S -> Fox = sum over sugar GRNs and both Fox cells of (F^k)[S, Fox], averaged
over the two Fox cells. References on the same scale: FDA-I -> PAM01-15 (1 hop) and the
ORN_DM4 -> DM4 PN control (1 hop).

Run: .venv/Scripts/python.exe rate/sugar_fox_anatomy.py  (~1 min, CPU). Writes
rate/sugar_fox_anatomy.json.
"""
import json
import os

import numpy as np
import pandas as pd
import pyarrow.feather as pf
import scipy.sparse as sp

HERE = os.path.dirname(os.path.abspath(__file__))
EXT = os.path.join(HERE, "..", "data", "ext")
FOX = ["CB0525"]
FDA_I = ["CB0546", "CB0272", "CB3199"]  # same as learn/christie_sweep.py FDA_I
NEG = {"gaba", "glutamate"}


def load():
    a = pd.read_csv(os.path.join(EXT, "Supplemental_file1_neuron_annotations.tsv"), sep="\t",
                    usecols=["root_id", "cell_sub_class", "cell_type", "top_nt", "super_class"],
                    low_memory=False).sort_values("root_id").reset_index(drop=True)
    tb = pf.read_table(os.path.join(EXT, "proofread_connections_783.feather"),
                       columns=["pre_pt_root_id", "post_pt_root_id", "syn_count"])
    rid = a.root_id.values
    n = len(rid)
    pre, post = tb.column(0).to_numpy(), tb.column(1).to_numpy()
    sc = tb.column(2).to_numpy().astype(np.int64)
    i, j = np.searchsorted(rid, pre), np.searchsorted(rid, post)
    ok = (i < n) & (j < n)
    ok &= (rid[np.minimum(i, n - 1)] == pre) & (rid[np.minimum(j, n - 1)] == post)
    W = sp.coo_matrix((sc[ok], (i[ok], j[ok])), shape=(n, n)).tocsr()  # sums duplicate pairs
    W.sum_duplicates()
    return a, W


def idx(a, col, vals):
    return np.flatnonzero(a[col].astype(str).isin(vals).values)


def analyse(a, W, floor):
    W = W.copy()
    W.data[W.data < floor] = 0
    W.eliminate_zeros()
    sign = np.where(a.top_nt.astype(str).str.lower().isin(NEG).values, -1.0, 1.0)
    tot_in = np.asarray(W.sum(axis=0)).ravel().astype(float)
    F = sp.diags(sign) @ W.astype(float) @ sp.diags(1.0 / np.maximum(tot_in, 1))
    F = F.tocsr()
    S = idx(a, "cell_sub_class", ["sugar/water"])
    fox = idx(a, "cell_type", FOX)
    assert len(S) == 129 and len(fox) == 2, (len(S), len(fox))
    Wc = W.tocsc()

    # direct and 2-hop routes S -> X -> Fox, per interneuron type
    s_out = np.asarray(W[S].sum(axis=0)).ravel()           # syn from all sugar GRNs onto each X
    x_fox = np.asarray(Wc[:, fox].sum(axis=1)).ravel()      # syn from each X onto Fox
    mids = np.flatnonzero((s_out > 0) & (x_fox > 0))
    mids = mids[~np.isin(mids, S)]
    rows = pd.DataFrame({"type": a.cell_type.astype(str).values[mids],
                         "super_class": a.super_class.astype(str).values[mids],
                         "nt": a.top_nt.astype(str).values[mids],
                         "syn_from_sugar": s_out[mids], "syn_to_fox": x_fox[mids],
                         "sugar_share_of_X_input": s_out[mids] / np.maximum(tot_in[mids], 1),
                         "X_share_of_fox_input": x_fox[mids] / tot_in[fox].sum()})
    by_type = (rows.groupby(["type", "super_class", "nt"], as_index=False)
               .agg(n=("type", "size"), syn_from_sugar=("syn_from_sugar", "sum"),
                    syn_to_fox=("syn_to_fox", "sum"),
                    sugar_share_of_X_input=("sugar_share_of_X_input", "mean"),
                    X_share_of_fox_input=("X_share_of_fox_input", "sum"))
               .sort_values("X_share_of_fox_input", ascending=False))

    # k-hop signed influence S -> Fox (mean over the two Fox cells), absolute too
    v = np.zeros(W.shape[0]); v[S] = 1.0
    va = v.copy()
    Fa = abs(F)
    hops = {}
    for k in range(1, 5):
        v = F.T @ v
        va = Fa.T @ va
        hops[k] = {"signed": float(v[fox].mean()), "abs": float(va[fox].mean())}

    fda = idx(a, "cell_type", FDA_I)
    pam = np.flatnonzero(a.cell_type.astype(str).str.match(r"PAM(0[1-9]|1[0-5])").values)
    orn = idx(a, "cell_type", ["ORN_DM4"])
    pn = np.flatnonzero(a.cell_type.astype(str).str.startswith("DM4_").values
                        & (a.super_class.astype(str) != "sensory").values)

    def share(src, dst):  # mean over dst of (syn from src / total input)
        s = np.asarray(Wc[:, dst][src].sum(axis=0)).ravel()
        return float(np.mean(s / np.maximum(tot_in[dst], 1)))

    return {
        "floor": floor,
        "fox_total_input_syn": int(tot_in[fox].sum()),
        "fox_n_input_partners": int(Wc[:, fox].getnnz()),
        "direct_sugar_to_fox_syn": int(np.asarray(W[S][:, fox].sum())),
        "n_two_hop_interneurons": int(len(mids)),
        "two_hop_share_of_fox_input": float(x_fox[mids].sum() / tot_in[fox].sum()),
        "top_two_hop_types": by_type.head(15).round(4).to_dict("records"),
        "influence_sugar_to_fox_by_hop": hops,
        "ref_fdaI_share_of_pam_input": share(fda, pam),
        "ref_ornDM4_share_of_dm4pn_input": share(orn, pn),
        "ref_n": {"fda_i": int(len(fda)), "pam": int(len(pam)), "orn_dm4": int(len(orn)),
                  "dm4_pn": int(len(pn))},
    }


if __name__ == "__main__":
    a, W = load()
    out = {f: analyse(a, W, f) for f in (1, 5)}
    json.dump(out, open(os.path.join(HERE, "sugar_fox_anatomy.json"), "w"), indent=1)
    for f, r in out.items():
        print("floor %d: Fox input %d syn / %d partners; direct sugar->Fox %d syn; "
              "%d 2-hop interneurons carry %.2f%% of Fox input"
              % (f, r["fox_total_input_syn"], r["fox_n_input_partners"],
                 r["direct_sugar_to_fox_syn"], r["n_two_hop_interneurons"],
                 100 * r["two_hop_share_of_fox_input"]))
        print("   influence by hop:", {k: "%.2e/%.2e" % (h["signed"], h["abs"])
                                        for k, h in r["influence_sugar_to_fox_by_hop"].items()})
        print("   refs: FDA-I share of PAM input %.4f, ORN_DM4 share of DM4 PN input %.4f %s"
              % (r["ref_fdaI_share_of_pam_input"], r["ref_ornDM4_share_of_dm4pn_input"],
                 r["ref_n"]))
        for t in r["top_two_hop_types"][:8]:
            print("   ", t)
