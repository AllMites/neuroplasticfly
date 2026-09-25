"""Why does DA3 reach zero Kenyon cells and DA2 only five?

Open question carried since the odour work, item 3 of "Do next" in
the 2026-09-22 project notes. Two candidate explanations, and the
whole point of this script is that they look identical from inside the simulator:

  FLOORED  the DA3 ALPN -> KC synapses exist in the proofread connectome but sit
           below brain_gpu.npz's min_syn edge floor (measured: 5), so they were
           dropped when the matrix was exported. A pipeline artifact. Fixable, or
           at minimum disclosable.
  ABSENT   they are not in the raw data either. A connectome gap in the same
           family as sugar -> PAM (docs/superpowers/reward-path/) and the APL
           in-degree bug, and it gets the same treatment: nothing is claimed about
           the animal without a positive control.

So every number is computed TWICE: once from brain_gpu.npz (what the simulator
actually runs, floor 5) and once from data/ext/proofread_connections_783.feather
at floor 1 (every proofread synapse). The difference between the two columns IS
the answer.

Structural only - no GpuSim, no CUDA. ~2 min, dominated by the 852 MB feather.

Run: .venv/Scripts/python.exe regime/alpn_kc_degree.py
"""
import json
import os
import sys

import numpy as np
import scipy.sparse as sp
from pyarrow import feather

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

BRAIN = os.environ.get("FLYCHESS_BRAIN", os.path.join(HERE, "data", "brain_gpu.npz"))
FEATHER = os.path.join(HERE, "data", "ext", "proofread_connections_783.feather")
OUT = os.path.join(HERE, "results", "alpn_kc_degree.json")

# Channels called out in the handoff, plus the two the CS gate uses as the
# working reference (gate_cs_channels.py: DC2 0.0603, D 0.0551 kc_active at gain 8).
FOCUS = ("DA3", "DA2", "DC2", "D", "DA1")


def raw_matrix(root_ids):
    """Every proofread synapse, floor 1, summed over neuropils. As sugar_pam_floor1."""
    n = len(root_ids)
    t = feather.read_table(FEATHER, columns=["pre_pt_root_id", "post_pt_root_id", "syn_count"],
                           memory_map=True)
    pre_r = t.column(0).to_numpy(zero_copy_only=False)
    post_r = t.column(1).to_numpy(zero_copy_only=False)
    syn = t.column(2).to_numpy(zero_copy_only=False).astype(np.float64)
    del t
    pre = np.searchsorted(root_ids, pre_r)
    post = np.searchsorted(root_ids, post_r)
    ok = (pre < n) & (post < n)
    ok[ok] &= (root_ids[pre[ok]] == pre_r[ok]) & (root_ids[post[ok]] == post_r[ok])
    print("feather rows %d, mapped %d, syn %.0f" % (len(syn), int(ok.sum()), syn[ok].sum()))
    return sp.coo_matrix((syn[ok], (pre[ok].astype(np.int64), post[ok].astype(np.int64))),
                         shape=(n, n)).tocsr()


def hop(W, src, dst, min_syn=0):
    """(n distinct dst reached, total synapse mass) from src onto dst."""
    if len(src) == 0 or len(dst) == 0:
        return 0, 0.0
    sub = W[src][:, dst]
    if min_syn:
        sub = sub.multiply(sub >= min_syn)
    reached = int((np.asarray(sub.sum(axis=0)).ravel() > 0).sum())
    return reached, float(sub.sum())


def main():
    meta = np.load(os.path.join(HERE, "data", "neuron_meta.npz"), allow_pickle=False)
    ct = meta["cell_type"].astype(str)
    cc = meta["cell_class"].astype(str)
    root_ids = np.load(os.path.join(HERE, "data", "root_ids_sorted.npy"))

    kc = np.flatnonzero(cc == "Kenyon_Cell").astype(np.int64)
    alpn = np.flatnonzero(cc == "ALPN").astype(np.int64)
    # ALPN cell_type is glomerulus-prefixed: DA3_adPN, DC2_adPN, DA4l_adPN.
    alpn_glom = np.array([s.split("_")[0] for s in ct[alpn]])
    # ...except where the ORN naming is finer than the PN naming. The VM6 receptor
    # classes are ORN_VM6l / ORN_VM6m / ORN_VM6v and they all project to the single
    # PN type VM6_adPN. Taking the prefix literally reported all three as having no
    # projection neuron at all, which is a naming artifact, not anatomy: VM6_adPN
    # receives 589 + 901 + 1339 = 2829 synapses from them (above the 2502 median) and
    # reaches 335 Kenyon cells. Caught 2026-09-22, after it had already reached a doc.
    # Explicit rather than a prefix rule, because a prefix rule would fold DA1/DA2/DA3
    # into "D", which is itself a real and separate glomerulus.
    ALIAS = {"VM6l": "VM6", "VM6m": "VM6", "VM6v": "VM6"}
    orn_types = sorted(set(s for s in ct if s.startswith("ORN_")))
    gloms = sorted(set(s[len("ORN_"):] for s in orn_types))
    print("%d KC, %d ALPN, %d ORN glomeruli" % (len(kc), len(alpn), len(gloms)))
    # Surface naming mismatches instead of silently reporting them as missing anatomy.
    # The VP*/CB*/M/MZ/Z labels are thermo/hygro and multiglomerular PNs with no single
    # ORN_ channel and are expected here; anything olfactory in this list is a bug.
    unmatched = sorted(set(alpn_glom) - {ALIAS.get(g, g) for g in gloms})
    print("ALPN labels with no ORN_ channel (expected: VP*/CB*/M/MZ/Z only): %s"
          % " ".join(unmatched))

    b = np.load(BRAIN)
    min_syn = int(b["min_syn"])
    Wsim = sp.csr_matrix((b["W_data"], b["W_indices"], b["W_indptr"]),
                         shape=(int(b["n_neurons"]), int(b["n_neurons"])))
    print("brain_gpu.npz: min_syn %d, nnz %d" % (min_syn, Wsim.nnz))
    Wraw = raw_matrix(root_ids)

    rows = []
    for g in gloms:
        orn = np.flatnonzero(ct == "ORN_" + g).astype(np.int64)
        pn = alpn[alpn_glom == ALIAS.get(g, g)]
        sim_pn_kc_n, sim_pn_kc_s = hop(Wsim, pn, kc)
        raw_pn_kc_n, raw_pn_kc_s = hop(Wraw, pn, kc)
        raw_at_floor_n, _ = hop(Wraw, pn, kc, min_syn=min_syn)
        sim_o_p_n, sim_o_p_s = hop(Wsim, orn, pn)
        raw_o_p_n, raw_o_p_s = hop(Wraw, orn, pn)
        rows.append({
            "glomerulus": g, "n_orn": len(orn), "n_alpn": len(pn),
            "sim_orn_alpn": sim_o_p_n, "raw_orn_alpn": raw_o_p_n,
            # Hop-1 MASS, not just whether the pair is connected. This is the column
            # that answers the DA3 question: DA3's 4 ALPNs are all contacted by its
            # ORNs and still receive 80 synapses between them, against DC2's 2356.
            "sim_orn_alpn_syn": sim_o_p_s, "raw_orn_alpn_syn": raw_o_p_s,
            "sim_orn_alpn_syn_per_pn": sim_o_p_s / len(pn) if len(pn) else 0.0,
            "sim_kc_n": sim_pn_kc_n, "sim_kc_syn": sim_pn_kc_s,
            "raw_kc_n": raw_pn_kc_n, "raw_kc_syn": raw_pn_kc_s,
            "raw_kc_n_at_floor": raw_at_floor_n,
        })

    rows.sort(key=lambda r: r["sim_orn_alpn_syn_per_pn"])
    print("\n| glom | ORN | ALPN | ORN->ALPN syn | per PN | ALPN->KC n | raw n | raw n >=%d | ALPN->KC syn |"
          % min_syn)
    for r in rows:
        mark = " <<<" if r["glomerulus"] in FOCUS else ""
        print("| %-5s | %3d | %2d | %8.0f | %6.0f | %6d | %5d | %5d | %9.0f |%s"
              % (r["glomerulus"], r["n_orn"], r["n_alpn"], r["sim_orn_alpn_syn"],
                 r["sim_orn_alpn_syn_per_pn"], r["sim_kc_n"], r["raw_kc_n"],
                 r["raw_kc_n_at_floor"], r["sim_kc_syn"], mark))

    # ---- export fidelity. The simulated matrix should be exactly the raw one
    # thresholded at min_syn. If it is not, the export dropped something for a
    # reason nobody wrote down, and every "missing wire" result below is suspect.
    # This is the APL-in-degree check generalised over all 53 channels.
    drift = [r for r in rows if r["sim_kc_n"] != r["raw_kc_n_at_floor"]]
    print("\nexport fidelity: sim == raw thresholded at min_syn=%d for %d/%d glomeruli"
          % (min_syn, len(rows) - len(drift), len(rows)))
    for r in drift:
        print("  DRIFT %-5s sim %d != raw>=%d %d" % (r["glomerulus"], r["sim_kc_n"],
                                                     min_syn, r["raw_kc_n_at_floor"]))

    # ---- the pre-registered split, decided by the numbers rather than by reading the table
    with_pn = [r for r in rows if r["n_alpn"] > 0]
    median_kc = float(np.median([r["sim_kc_n"] for r in with_pn]))
    median_h1 = float(np.median([r["sim_orn_alpn_syn_per_pn"] for r in with_pn]))
    # TOTAL hop-1 mass is the discriminator, not the per-PN average. DA1 sits at
    # 0.32x the median per PN and still ignites 600 KCs, because it has 17 ALPNs
    # carrying 2940 synapses between them. DA3 has 4 ALPNs carrying 80.
    median_h1_total = float(np.median([r["sim_orn_alpn_syn"] for r in with_pn]))
    verdicts = {}
    for r in rows:
        if r["n_alpn"] == 0:
            v = "NO_ALPN"      # no projection neuron for this glomerulus at all
        elif r["raw_kc_n"] == 0:
            v = "ABSENT"       # no ALPN->KC synapse at floor 1 either: a connectome gap
        elif r["sim_kc_n"] == 0:
            v = "FLOORED"      # the synapses exist below min_syn and the export cut them
        elif r["sim_orn_alpn_syn"] < 0.25 * median_h1_total:
            # The ORNs contact their ALPNs and carry almost nothing. DA3: 4/4 pairs
            # connected, 80 synapses total. This is where DA3's zero KCs come from --
            # its PNs never fire, so the KC step is never reached. Same SHAPE as the
            # sugar -> PAM result: the wire is nominally present and carries no mass.
            v = "STARVED_HOP1"
        elif r["sim_kc_n"] < 0.35 * median_kc:
            v = "THIN_HOP2"    # PNs fire, but the ALPN->KC fan-out is far below typical
        else:
            v = "ok"
        verdicts[r["glomerulus"]] = v
    print("\nmedians over the %d glomeruli with ALPNs: hop-1 %.0f syn total (%.0f per PN), "
          "hop-2 %.0f KC" % (len(with_pn), median_h1_total, median_h1, median_kc))
    print("NOTE: structural only. Whether a starved channel actually fails to ignite is a")
    print("      separate measurement: learn/gate_cs_channels.py --probe ORN_DA3,ORN_DA2")
    print("verdicts for the focus channels:")
    for g in FOCUS:
        r = next(x for x in rows if x["glomerulus"] == g)
        print("  %-5s %-13s hop1 %6.0f syn (%.2fx) | hop2 %4d KC (%.2fx) | %2d ALPN"
              % (g, verdicts[g], r["sim_orn_alpn_syn"],
                 r["sim_orn_alpn_syn"] / median_h1_total,
                 r["sim_kc_n"], r["sim_kc_n"] / median_kc, r["n_alpn"]))
    for v in ("ABSENT", "FLOORED", "STARVED_HOP1", "THIN_HOP2", "NO_ALPN"):
        names = sorted(g for g, x in verdicts.items() if x == v)
        print("  %-13s %2d  %s" % (v, len(names), " ".join(names)))

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump({"min_syn": min_syn, "brain": BRAIN, "median_kc_fanout": median_kc,
               "median_hop1_syn_per_pn": median_h1,
               "median_hop1_syn_total": median_h1_total,
               "export_drift": [r["glomerulus"] for r in drift],
               "rows": rows, "verdicts": verdicts}, open(OUT, "w"), indent=1)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
