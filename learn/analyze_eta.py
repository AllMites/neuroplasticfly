"""Eta sweep: does a lower learning rate buy a learning CURVE instead of a step?

Reads the four condition_*.json/.jsonl runs that differ only in eta and prints,
per eta, the learn-arm avoid-index shift and the trial-by-trial weight state.
Evidence for docs/superpowers/condition-v1/eta_sweep_notes.md.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import learn.plastic as PL  # noqa: E402  (needs HERE on the path)
RESULTS = os.path.join(HERE, "results")
STATES = os.path.join(HERE, "brain_state")
RUNS = [("3e-5", "eta3e-5"), ("1e-4", "eta1e-4"), ("3e-4", "eta3e-4"), ("1e-3", "v2neg")]


def shift(curve, song):
    """Final probe minus baseline probe, in Hz of avoid index."""
    d = curve[song]
    t = sorted(d, key=int)
    return d[t[-1]][0] - d[t[0]][0], d[t[0]][0], d[t[-1]][0]


def train_rows(state):
    with open(os.path.join(RESULTS, "condition_%s.jsonl" % state)) as f:
        return [r for r in map(json.loads, f)
                if r["arm"] == "learn" and r["phase"] == "train" and "w_mean_frac" in r]


def main():
    print("%-6s %8s %8s %8s %8s %8s" % ("eta", "misery", "skyhigh", "mozart", "mean_dw", "at_floor"))
    curves = {}
    for eta, state in RUNS:
        c = json.load(open(os.path.join(RESULTS, "condition_%s.json" % state)))["curves"]["learn"]
        prov = json.load(open(os.path.join(STATES, state, "provenance.json")))
        s = prov["summary"]
        curves[eta] = c
        print("%-6s %8.3f %8.3f %8.3f %8.4f %8d" % (
            eta, shift(c, "misery")[0], shift(c, "skyhigh")[0], shift(c, "mozart")[0],
            s["mean_frac"], s["n_at_floor"]))

    print("\nper-trial mean weight fraction (learn arm, both songs interleaved)")
    print("%-6s %s" % ("eta", "  ".join("t%02d" % k for k in range(1, 11))))
    for eta, state in RUNS:
        rows = train_rows(state)
        by_trial = {}
        for r in rows:
            by_trial.setdefault(r["trial"], []).append(r["w_mean_frac"])
        seq = [np.mean(by_trial[k]) for k in sorted(by_trial)[:10]]
        print("%-6s %s" % (eta, "  ".join("%.4f" % v for v in seq)))

    print("\nfraction of the final shift already present at the first probe (trial 5)")
    for eta in curves:
        for song in ("misery", "skyhigh"):
            d = curves[eta][song]
            t = sorted(d, key=int)
            base, first, last = d[t[0]][0], d[t[1]][0], d[t[-1]][0]
            total = last - base
            print("  eta %-5s %-8s %+.3f of %+.3f = %.0f%%" % (
                eta, song, first - base, total, 100.0 * (first - base) / total if total else float("nan")))

    print("\nwhich edges moved at all, and how far, as a fraction of w0")
    P = PL.Plastic.real()
    w0 = P.w0
    ref = None
    for eta, state in RUNS:
        dw = np.load(os.path.join(STATES, state, "dw.npy"))
        idx = np.flatnonzero(dw)
        if ref is None:
            ref = idx
        f = dw[idx] / w0[idx]
        print("  eta %-5s moved %d of %d (%.1f%%), same set as eta 3e-5: %s, "
              "frac of w0 min %.4f med %.4f max %.4f, at floor %d" % (
                  eta, len(idx), len(dw), 100.0 * len(idx) / len(dw),
                  np.array_equal(idx, ref), f.min(), np.median(f), f.max(),
                  int((f < -0.89).sum())))

    print("\nwhy the other %d edges never move" % (len(w0) - len(ref)))
    moved = np.zeros(P.n_edges, bool); moved[ref] = True
    for mv, dv in PL.OPPOSES.items():
        sel = P.mbon_valence_of_edge == mv
        dc = set(c[:2] for c in P.dan_comp[dv])
        has = np.array([c[:2] in dc for c in P.mbon_comp_of_edge[sel]])
        print("  %-9s edges %6d, with a live %s DAN %6d (%.0f%%), moved %6d" % (
            mv, sel.sum(), dv, has.sum(), 100.0 * has.mean(), moved[sel].sum()))
    print("  distinct KCs among movers %d of %d (the fly-hash graft drives 5%% per song)" % (
        len(set(P.kc_of_edge[moved].tolist())), len(set(P.kc_of_edge.tolist()))))
    print("  distinct MBONs among movers %d of %d" % (
        len(set(P.mbon_of_edge[moved].tolist())), len(set(P.mbon_of_edge.tolist()))))


if __name__ == "__main__":
    main()
