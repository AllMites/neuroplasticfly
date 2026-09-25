"""Christie et al. 2026 reconciliation: sugar GRN -> Fox -> FDA -> PAM, hop by hop.

Christie et al. 2026 (Curr Biol, PMC12869359) report that sugar GRN activation
recruits PAM-DANs through Fox (CB0525) and 11 "FDA" ascending pairs, in a
Shiu-2024 Brian2 LIF on the FlyWire connectome. flychess measured PAM silent
under the same 150 Hz sugar drive. This runs sugar alone at several rates with
each hop silenced in turn and reads every hop, so the disagreement can be
located rather than asserted.

Readouts are reported under BOTH criteria: Christie's (a PAM counts as
responsive if its mean rate is nonzero) and flychess's usual >= 1 Hz cut.

Run:
  .venv/Scripts/python.exe learn/christie_sweep.py --seeds 5 --tag v783stock
"""
import argparse
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _HERE); sys.path.insert(0, os.path.join(_HERE, "flywatch"))
import gpu_sim as G
from learn import condition as C
from learn import plastic as PL

# Christie's cell sets, by FlyWire cell_type. NOT by `hemibrain_type`: that
# column reads "none" for every CB* label in data/neuron_meta.npz (2026-09-23),
# so resolving there silently returns the empty set.
FOX = ["CB0525"]
# FDA-I: the ascending pairs that synapse DIRECTLY onto PAM (Christie Data S2A,
# Fox -> FDA synapse counts: CB0546 106, CB0272 55, CB3199 10).
FDA_I = ["CB0546", "CB0272", "CB3199"]
# FDA-II: the rest (CB0233 217, DNp62 57, then CB0337, CB1514, CB1025, CB3470,
# CB3573). Christie: silencing CB0233 alone, FDA-II, or FDA-all kills the PAM
# response; silencing FDA-I does not.
FDA_II = ["CB0233", "DNp62", "CB0337", "CB1514", "CB1025", "CB3470", "CB3573"]


def type_index(meta_ct, types):
    """int64 indices of every neuron whose cell_type is in `types`."""
    idx = np.flatnonzero(np.isin(meta_ct, types)).astype(np.int64)
    assert len(idx), "no neuron matches %s" % (types,)
    return idx


def fox_to_syn(net, pre, post):
    """Raw synapse count on pre -> post edges. CSR rows are PRESYNAPTIC and
    W_indices are POSTSYNAPTIC (gpu_sim.py:660); W_data holds signed raw counts."""
    ip, ix, w = net.W_indptr, net.W_indices, net.W_data
    post_set = np.zeros(net.n, dtype=bool); post_set[post] = True
    tot, n_edge = 0.0, 0
    for p in pre:
        s, e = ip[p], ip[p + 1]
        m = post_set[ix[s:e].astype(np.int64)]
        tot += float(np.abs(w[s:e][m]).sum()); n_edge += int(m.sum())
    return tot, n_edge


def summarise(rates, idx):
    r = rates[idx]
    return {"n": int(len(idx)), "mean": float(r.mean()), "max": float(r.max()),
            "n_silent": int((r < 1.0).sum()), "frac_silent": float((r < 1.0).mean()),
            # Christie's criterion: responsive = nonzero mean rate.
            "n_responsive": int((r > 0).sum())}


def agg(rows, group, key):
    return float(np.mean([r["groups"][group][key] for r in rows]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--hz", default="100,150,200")
    ap.add_argument("--t-run", type=float, default=C.T_RUN)
    a = ap.parse_args()
    hz_list = [float(x) for x in a.hz.split(",")]

    t0 = time.time()
    # Sets resolved BEFORE the brain load, so a bad label fails in a second
    # rather than after a minute of matrix build (us_path_sweep.py:57).
    meta = np.load(os.path.join(_HERE, "data", "neuron_meta.npz"), allow_pickle=False)
    ct = meta["cell_type"].astype(str)
    fox = type_index(ct, FOX)
    assert len(fox) == 2, "Fox (CB0525) should be one bilateral pair, got %d" % len(fox)
    fda_i, fda_ii = type_index(ct, FDA_I), type_index(ct, FDA_II)
    fda_all = np.union1d(fda_i, fda_ii)
    pam = np.flatnonzero(np.char.startswith(ct, "PAM")).astype(np.int64)
    pam_subs = {t: np.flatnonzero(ct == t).astype(np.int64)
                for t in sorted(set(ct[pam]))}
    print("sets: fox=%d fdaI=%d fdaII=%d pam=%d" % (len(fox), len(fda_i), len(fda_ii), len(pam)))
    for t in FOX + FDA_I + FDA_II:
        print("  %-8s n=%d" % (t, int((ct == t).sum())))

    sim = G.GpuSim(); P = PL.Plastic.real()

    # Sanity against the paper before spending GPU time: Fox -> CB0233 is 217
    # synapses in Christie Data S2A. If this matrix has 0, the path is not in it
    # and every number below would be a measurement of its absence, not of the
    # physiology.
    cb0233 = type_index(ct, ["CB0233"])
    syn, n_edge = fox_to_syn(sim.net, fox, cb0233)
    print("Fox -> CB0233: %.0f synapses over %d edges (Christie Data S2A: 217)"
          % (syn, n_edge))
    assert n_edge > 0, "Fox -> CB0233 absent from this matrix; the run would be moot"

    groups = {"sugar_grn": C.SUGAR, "fox": fox, "fdaI": fda_i, "fdaII": fda_ii,
              "pam_all": pam}
    groups.update({"pam_" + t: i for t, i in pam_subs.items()})

    conds = [("base", 0.0, None)]
    for hz in hz_list:
        conds += [("sugar", hz, None), ("sugar-fox", hz, fox),
                  ("sugar-fdaI", hz, fda_i), ("sugar-fdaII", hz, fda_ii),
                  ("sugar-fdaAll", hz, fda_all)]

    rows = []
    for cond, hz, sil in conds:
        for seed in range(a.seeds):
            if hz > 0:
                idx, rate = C.SUGAR, np.full(len(C.SUGAR), hz, np.float32)
            else:
                idx, rate = np.zeros(0, np.int64), np.zeros(0, np.float32)
            counts = sim.run_batch([C.to_prob(idx, rate)], [seed],
                                   t_run=a.t_run, silence=sil)
            rates = counts[0].cpu().numpy().astype(np.float32) / (a.t_run / 1000.0)
            r = C.readout(rates, P)
            rows.append({"cond": cond, "hz": hz, "seed": seed,
                         "groups": {k: summarise(rates, v) for k, v in groups.items()},
                         "kc_active": r["kc_active"], "central_active": r["central_active"]})
        print("%-13s hz=%-5.0f done (%.1f s)" % (cond, hz, time.time() - t0))

    out = {"brain": os.path.abspath(G.BRAIN), "n_edges": int(sim.net.n_edges),
           "seeds": a.seeds, "hz": hz_list, "t_run": a.t_run,
           "sets": {"fox": len(fox), "fdaI": len(fda_i), "fdaII": len(fda_ii),
                    "pam": len(pam)},
           "set_types": {"fox": FOX, "fdaI": FDA_I, "fdaII": FDA_II},
           "fox_to_cb0233_syn": syn, "fox_to_cb0233_edges": n_edge,
           "rows": rows}
    os.makedirs(C.RESULTS, exist_ok=True)
    path = os.path.join(C.RESULTS, "christie_sweep_%s.json" % a.tag)
    json.dump(out, open(path, "w"), indent=1)

    subs = sorted(pam_subs)[:4]
    print("")
    print("brain=%s  n_edges=%d  seeds=%d  t_run=%.0f ms" % (G.BRAIN, out["n_edges"], a.seeds, a.t_run))
    head = ("| cond         |  hz | fox Hz | fdaI Hz | fdaII Hz | PAM resp/%d | PAM >=1Hz | PAM Hz | PAM max |"
            % len(pam)) + "".join(" %-8s |" % s for s in subs)
    print(head)
    print("|" + "-" * (len(head) - 2) + "|")
    for cond, hz, _ in conds:
        rs = [r for r in rows if r["cond"] == cond and r["hz"] == hz]
        line = ("| %-12s | %3.0f | %6.2f | %7.2f | %8.2f | %11.1f | %9.1f | %6.3f | %7.2f |"
                % (cond, hz, agg(rs, "fox", "mean"), agg(rs, "fdaI", "mean"),
                   agg(rs, "fdaII", "mean"), agg(rs, "pam_all", "n_responsive"),
                   len(pam) - agg(rs, "pam_all", "n_silent"),
                   agg(rs, "pam_all", "mean"), agg(rs, "pam_all", "max")))
        print(line + "".join(" %8.2f |" % agg(rs, "pam_" + s, "mean") for s in subs))
    print("")
    print("PAM resp = Christie criterion (rate > 0). PAM >=1Hz = flychess cut.")
    print("wrote %s  (%.1f s)" % (path, time.time() - t0))


if __name__ == "__main__":
    main()
