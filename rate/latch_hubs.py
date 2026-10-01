"""Paper-1 phase 3 (descriptive): overlap of the post-offset latch with Li et al. 2026 hub types and their partners.
Data: results/rate_chunk0/{stock/,}lif_ref.npz rate_off [53 odours, 10 x 20 ms bins, N] (seed-mean Hz, after 300 ms 40 Hz drive).
No simulation. Run: .venv/Scripts/python.exe rate/latch_hubs.py"""
import json
import os
import numpy as np
from math import lgamma, exp, log

H = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(H, "results")
HUB_TYPES = ["KCab", "L2", "LPLC2", "LC9", "PFNv"]   # PRD list (Li et al. preprint was 403 to WebFetch)
HZ, MIN_LATCH_N, FLOOR = 1.0, 100, 5
RULE = ("Rule (fixed before numbers): latched = seed-mean rate > 1 Hz over the last 100 ms (bins 5-9) of the 200 ms off period; "
        "an odour latches if > 100 neurons do. Test set = hub neurons (cell_type in %s) UNION their direct pre/post partners "
        "(synapse floor 5, brain_gpu.npz). Per odour: enrichment = P(in set | latched) / P(in set | brain), hypergeometric upper-tail p. "
        "ENRICHED if, in BOTH LIF regimes, median ratio >= 1.5 and p < 0.01 in >= 80%% of latching odours; "
        "NOT-ENRICHED if median ratio <= 1.2 in both; otherwise INCONCLUSIVE." % HUB_TYPES)


def hyp_sf(k, K, n, N):
    """P(X >= k), X~Hypergeom(N pop, K successes, n draws); exact log-space sum."""
    def lc(a, b): return lgamma(a + 1) - lgamma(b + 1) - lgamma(a - b + 1)
    top = min(K, n); tot = 0.0; base = lc(N, n)
    terms = [lc(K, i) + lc(N - K, n - i) - base for i in range(k, top + 1)]
    if not terms: return 0.0
    m = max(terms)
    return min(1.0, exp(m) * sum(exp(t - m) for t in terms))


def main():
    meta = np.load(os.path.join(H, "data", "neuron_meta.npz"), allow_pickle=False)
    ct, cc = meta["cell_type"].astype(str), meta["cell_class"].astype(str)
    N = len(ct)
    b = np.load(os.path.join(H, "data", "brain_gpu.npz"))
    assert int(b["min_syn"]) == FLOOR
    ip, ix = b["W_indptr"], b["W_indices"]
    pre = np.repeat(np.arange(N), np.diff(ip))              # rows = pre (CSR, pre-major)
    hub = np.isin(ct, HUB_TYPES)
    post_p = np.zeros(N, bool); post_p[ix[hub[pre]]] = True          # targets of hubs
    pre_p = np.zeros(N, bool); pre_p[pre[hub[ix]]] = True            # sources onto hubs
    partner = (post_p | pre_p) & ~hub
    tset = hub | partner
    kc = cc == "Kenyon_Cell"
    mb6 = ct == "MBON06"
    print("hub n=%d, partners n=%d, set=%.3f of brain, MBON06 n=%d" % (hub.sum(), partner.sum(), tset.mean(), mb6.sum()))
    out = {"rule": RULE, "hub_types": HUB_TYPES, "hub_source": "PRD list (Li et al. full text returned HTTP 403; not verified against preprint)",
           "floor": FLOOR, "N": N, "n_hub": int(hub.sum()), "n_partner": int(partner.sum()), "base_frac_set": float(tset.mean()),
           "hub_counts": {t: int((ct == t).sum()) for t in HUB_TYPES}, "regimes": {},
           "rate_C0": "NOT ANALYSED: only counts stored (cells/*.json, result.json); per-neuron latch needs a forward rate run (not launched)"}
    md = []
    verdicts = {}
    for reg, path in [("stock", "stock/lif_ref.npz"), ("eln8", "lif_ref.npz")]:
        z = np.load(os.path.join(R, "rate_chunk0", path), allow_pickle=False)
        off, names = z["rate_off"], z["names"].astype(str)
        rows = []
        for k, nm in enumerate(names):
            lat = off[k, 5:].mean(0) > HZ
            n = int(lat.sum())
            if n <= MIN_LATCH_N: continue
            ov = int((lat & tset).sum()); K = int(tset.sum())
            ratio = (ov / n) / (K / N)
            nk = lat & ~kc; nnk = int(nk.sum())
            ratio_nk = ((nk & tset).sum() / nnk) / ((tset & ~kc).sum() / (~kc).sum()) if nnk else float("nan")
            rows.append({"odour": nm, "n_latched": n, "kc": int((lat & kc).sum()), "in_set": ov, "frac": ov / n, "ratio": ratio,
                         "p": hyp_sf(ov, K, n, N), "ratio_nonKC": float(ratio_nk),
                         "hub_latched": int((lat & hub).sum()), "hub_frac_latched": float((lat & hub).sum() / hub.sum()),
                         "hub_KCab_latched": int((lat & (ct == "KCab")).sum()), "partner_latched": int((lat & partner).sum()),
                         "mbon06_latched": int((lat & mb6).sum()), "mbon06_on_rate_off_max": float(off[k, 5:].mean(0)[mb6].max())})
        nl = len(rows); rat = np.array([r["ratio"] for r in rows]); ps = np.array([r["p"] for r in rows])
        s = {"n_odours": len(names), "n_latching": nl, "median_ratio": float(np.median(rat)), "frac_p_lt_001": float((ps < .01).mean()),
             "median_ratio_nonKC": float(np.nanmedian([r["ratio_nonKC"] for r in rows])),
             "median_n_latched": float(np.median([r["n_latched"] for r in rows])),
             "median_kc_share": float(np.median([r["kc"] / r["n_latched"] for r in rows])),
             "median_hub_frac_latched": float(np.median([r["hub_frac_latched"] for r in rows])),
             "median_hub_latched": float(np.median([r["hub_latched"] for r in rows])),
             "median_partner_latched": float(np.median([r["partner_latched"] for r in rows])),
             "mbon06_latched_in_odours": int(sum(r["mbon06_latched"] > 0 for r in rows)),
             "base_kc_frac": float(kc.mean())}
        verdicts[reg] = ("E" if s["median_ratio"] >= 1.5 and s["frac_p_lt_001"] >= .8 else
                         "N" if s["median_ratio"] <= 1.2 else "I")
        out["regimes"][reg] = {"summary": s, "per_odour": rows}
        md.append("| %s | %d/53 | %.0f | %.2f | %.2f | %.2f | %.0f%% | %.0f%% | %.3f | %d | %d/%d |" % (
            reg, nl, s["median_n_latched"], s["median_ratio"], s["median_ratio_nonKC"], s["median_kc_share"],
            s["frac_p_lt_001"] * 100, s["median_hub_frac_latched"] * 100, tset.mean(), s["median_partner_latched"],
            s["mbon06_latched_in_odours"], nl))
    v = "ENRICHED" if all(x == "E" for x in verdicts.values()) else "NOT-ENRICHED" if all(x == "N" for x in verdicts.values()) else "INCONCLUSIVE"
    out["verdicts_per_regime"], out["verdict"] = verdicts, v
    json.dump(out, open(os.path.join(R, "rate_latch_hubs", "result.json"), "w"), indent=1)
    hdr = ("| regime | odours latching | median n latched | enrichment (hub+partner) | enrichment excl. KC | KC share of latched | odours p<0.01 | hub neurons latched (median % of hubs) | base frac in set | median partners latched | MBON06 latched (odours) |\n"
           "|---|---|---|---|---|---|---|---|---|---|---|\n")
    text = ("# RESULT: latch vs Li et al. hub overlap (paper-1 phase 3, descriptive)\n\n%s\n\n"
            "Hub types (PRD list; Li et al. text not accessible, HTTP 403): %s; counts %s. Hub neurons %d, partners %d (floor 5), "
            "set = %.1f%% of brain.\nRate C0 not analysed: only latch counts are stored, no per-neuron arrays (would need a forward rate run).\n\n"
            "%s%s\n\n**VERDICT: %s** (per regime: %s)\n") % (RULE, ", ".join(HUB_TYPES), out["hub_counts"], hub.sum(), partner.sum(),
                                                          tset.mean() * 100, hdr, "\n".join(md), v, verdicts)
    open(os.path.join(R, "rate_latch_hubs", "RESULT.md"), "w").write(text)
    print(text)


if __name__ == "__main__":
    main()
