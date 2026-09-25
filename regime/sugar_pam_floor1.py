"""Track 1: brain-wide floor-1 convergence, sugar GRN -> rewarding PAM subtypes.

Built straight from data/ext/proofread_connections_783.feather (every proofread
edge, syn_count >= 1, neuropil rows summed per pre/post pair). No npz, no sim,
nothing our build thresholded away.

Question (signed_paths_2026-09-21.md, "Next live lever"): is the sugar -> PAM
channel missing from v783, or real and carried by convergence we cut elsewhere?

Falsifier: < ~100 sugar-driven synapses per rewarding PAM subtype (PAM02 /
PAM04 / PAM11 = beta'2a / gamma4 / gamma3) -> sugar->PAM is absent from v783 as
reconstructed; close reward-through-anatomy.

Run:
  .venv/Scripts/python.exe regime/sugar_pam_floor1.py
"""
import csv as _csv
import json
import os

import numpy as np
import pyarrow.feather as feather
import scipy.sparse as sp

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATHER = os.path.join(HERE, "data", "ext", "proofread_connections_783.feather")
ANN = os.path.join(HERE, "data", "ext", "Supplemental_file1_neuron_annotations.tsv")
OUT = os.path.join(HERE, "results", "sugar_pam_floor1.json")

# Literature reward carriers. PAM02 = beta'2a, PAM04 = gamma4(?), PAM11 = gamma3;
# alpha1 (PAM11/PAM12 in some tables) reported too - all 15 are printed below so
# the choice of three cannot hide a hit.
REWARD_SUBTYPES = ("PAM02", "PAM04", "PAM11")
EDGE_FLOOR = 10  # "carries >= 10 synapses onto them"


def main():
    meta = np.load(os.path.join(HERE, "data", "neuron_meta.npz"), allow_pickle=False)
    ct = meta["cell_type"].astype(str)
    csc = meta["cell_sub_class"].astype(str)
    root_ids = np.load(os.path.join(HERE, "data", "root_ids_sorted.npy"))
    n = len(root_ids)

    # hemibrain_type now lives in the meta (tools/add_hemibrain_type.py, 2026-09-22),
    # where missing is the string "none". The TSV re-parse stays as a fallback so this
    # script still runs against an older meta; the two agree by construction, the
    # backfill asserts cell_type row-for-row against the same TSV before writing.
    if "hemibrain_type" in meta:
        hb = meta["hemibrain_type"].astype(str)
        hb = np.where(hb == "none", "", hb)
    else:
        hb_map = {}
        with open(ANN, newline="", encoding="utf-8") as fh:
            for r in _csv.DictReader(fh, delimiter="\t"):
                v = (r.get("hemibrain_type") or "").strip()
                if v:
                    hb_map[int(r["root_id"])] = v
        hb = np.array([hb_map.get(int(x), "") for x in root_ids], dtype=object)

    t = feather.read_table(FEATHER, columns=["pre_pt_root_id", "post_pt_root_id", "syn_count"],
                           memory_map=True)
    pre_r = t.column(0).to_numpy(zero_copy_only=False)
    post_r = t.column(1).to_numpy(zero_copy_only=False)
    syn = t.column(2).to_numpy(zero_copy_only=False).astype(np.float64)
    del t
    print("feather rows %d, raw syn %.0f" % (len(syn), syn.sum()))

    # root_id -> index; drop edges whose endpoints are outside our neuron set.
    pre = np.searchsorted(root_ids, pre_r)
    post = np.searchsorted(root_ids, post_r)
    ok = ((pre < n) & (post < n))
    ok[ok] &= (root_ids[pre[ok]] == pre_r[ok]) & (root_ids[post[ok]] == post_r[ok])
    pre, post, syn = pre[ok].astype(np.int64), post[ok].astype(np.int64), syn[ok]
    print("mapped edges %d (%.4f of rows), syn %.0f"
          % (len(syn), len(syn) / len(ok), syn.sum()))

    # Sum the per-neuropil rows into one pre->post weight, floor 1 (i.e. all).
    W = sp.coo_matrix((syn, (pre, post)), shape=(n, n)).tocsr()
    W.sum_duplicates()
    print("floor-1 CSR: %d pairs, %.0f syn" % (W.nnz, W.data.sum()))

    seed = np.flatnonzero(csc == "sugar/water").astype(np.int64)
    is_pam = np.char.startswith(ct, "PAM")
    pam = np.flatnonzero(is_pam).astype(np.int64)
    print("sugar GRNs %d, PAM %d" % (len(seed), len(pam)))

    res = {"feather": os.path.abspath(FEATHER), "n_pairs_floor1": int(W.nnz),
           "syn_total": float(W.data.sum()), "n_sugar": int(len(seed)),
           "n_pam": int(len(pam)), "edge_floor": EDGE_FLOOR,
           "reward_subtypes": list(REWARD_SUBTYPES)}

    # Forward reach. h[k] = boolean mask of neurons receiving >= 1 syn from h[k-1].
    v = np.zeros(n, bool)
    v[seed] = True
    sugar_in = W.T.dot(v.astype(np.float64))   # syn each neuron gets FROM sugar GRNs
    hops = []
    cur = v
    for k in (1, 2, 3):
        nxt = (W.T.dot(cur.astype(np.float64)) > 0)
        hops.append(nxt)
        print("hop-%d reach: %d neurons" % (k, int(nxt.sum())))
        cur = nxt
    res["reach"] = {"hop%d" % (i + 1): int(h.sum()) for i, h in enumerate(hops)}

    # Per-subtype convergence. A "hop-k carrier" is a neuron in the hop-(k-1)
    # reach set with an edge onto a member of the subtype.
    Wc = W.tocsc()

    def subtype_report(mask_sub, label):
        tgt = np.flatnonzero(mask_sub)
        if len(tgt) == 0:
            return None
        sub = Wc[:, tgt].tocsr()
        in_syn = np.asarray(sub.sum(axis=1)).ravel()   # per presyn neuron -> subtype
        row = {"label": label, "n_neurons": int(len(tgt)),
               "total_input_syn": float(in_syn.sum())}
        for k, carriers in ((2, hops[0]), (3, hops[1])):
            m = carriers & (in_syn > 0)
            m10 = carriers & (in_syn >= EDGE_FLOOR)
            row["hop%d" % k] = {
                "carriers": int(m.sum()), "syn": float(in_syn[m].sum()),
                "carriers_ge%d" % EDGE_FLOOR: int(m10.sum()),
                "syn_ge%d" % EDGE_FLOOR: float(in_syn[m10].sum())}
        # Bottleneck mass: a hop-2 carrier can only relay what sugar delivers TO
        # it. min(sugar -> carrier, carrier -> subtype) per carrier, summed.
        # Raw hop-2 syn overstates the channel whenever a strongly PAM-projecting
        # neuron happens to pick up 1-2 sugar synapses (CB0032 -> PAM11 is 112
        # syn off a single sugar synapse).
        m = hops[0] & (in_syn > 0)
        row["hop2"]["bottleneck_syn"] = float(
            np.minimum(sugar_in[m], in_syn[m]).sum())
        strong = hops[0] & (in_syn > 0) & (sugar_in >= 5)
        row["hop2"]["carriers_sugar_in_ge5"] = int(strong.sum())
        row["hop2"]["syn_from_sugar_in_ge5"] = float(in_syn[strong].sum())
        return row

    rows = []
    for i in range(1, 16):
        key = "PAM%02d" % i
        m = np.array([str(x).startswith(key) for x in hb], dtype=bool)
        r = subtype_report(m, key)
        if r:
            rows.append(r)
    rows.append(subtype_report(is_pam, "ALL_PAM"))
    res["subtypes"] = [r for r in rows if r]

    print("")
    print("| subtype | n | total in-syn | hop2 carriers | hop2 syn | bottleneck | carriers sugar_in>=5 | their syn | hop3 carriers | hop3 syn |")
    for r in res["subtypes"]:
        h2, h3 = r["hop2"], r["hop3"]
        print("| %-7s | %4d | %10.0f | %5d | %8.0f | %8.0f | %4d | %8.0f | %5d | %9.0f |"
              % (r["label"], r["n_neurons"], r["total_input_syn"],
                 h2["carriers"], h2["syn"], h2["bottleneck_syn"],
                 h2["carriers_sugar_in_ge5"], h2["syn_from_sugar_in_ge5"],
                 h3["carriers"], h3["syn"]))

    verdict = {}
    for r in res["subtypes"]:
        if r["label"] in REWARD_SUBTYPES:
            verdict[r["label"]] = {"hop2_syn": r["hop2"]["syn"],
                                   "hop2_bottleneck_syn": r["hop2"]["bottleneck_syn"],
                                   "hop3_syn": r["hop3"]["syn"]}
    worst2 = max((v["hop2_bottleneck_syn"] for v in verdict.values()), default=0.0)
    res["verdict"] = {"per_subtype": verdict, "max_hop2_syn": worst2,
                      "closed": bool(worst2 < 100.0)}
    print("")
    print("FALSIFIER: max hop-2 bottleneck syn onto a rewarding subtype = %.0f "
          "(threshold 100) -> %s" % (worst2, "CLOSE reward-through-anatomy"
                                     if worst2 < 100 else "WIDEN to global floor-1"))

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    print("wrote %s" % OUT)


if __name__ == "__main__":
    main()
