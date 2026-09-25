"""Seed table + population descending-neuron readout for condition runs.

Reads results/condition_<state>.json (avoid_index curves) and results/rates_<state>/
(probe rate vectors saved by `condition.py --dump`). Two outputs:

1. Seed table: avoid_index shift (last probe - baseline) per arm x stimulus x seed, mean, sd.
2. Population DN readout: for every DN cell type, dS = (trained - baseline) for the PAIRED
   stimulus minus the same for the UNPAIRED one. Which those are is read from the run json
   ("probes"), so this works for --cs song (misery/skyhigh) and --cs odour (da1/dc2) alike. Then Pearson r between arms (learn vs reversed should be negative,
   learn vs lesion ~0, learn vs shuffle ~0) and between seeds (learn vs learn should be
   positive). Same test TheMrRaGe/flybrain uses (learned_dn.py), applied to our runs.

Usage: python learn/analyze_seeds.py v1s0 v1s1 v1s2 v1s3 v1s4
       python learn/analyze_seeds.py o1s0 o1s1 o1s2 o1s3 o1s4
"""
import glob
import json
import os
import re
import sys

import numpy as np

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(_HERE, "results")
meta = np.load(os.path.join(_HERE, "data", "neuron_meta.npz"), allow_pickle=True)
ct = meta["cell_type"].astype(str)
DN_TYPES = sorted(t for t in set(ct) if t.startswith("DN"))
DN_IDX = {t: np.flatnonzero(ct == t) for t in DN_TYPES}


def seed_table(states):
    print("\n== avoid_index shift (last probe - baseline), Hz ==")
    rows = {}
    for st in states:
        cv = json.load(open(os.path.join(RESULTS, "condition_%s.json" % st)))["curves"]
        for arm, sd in cv.items():
            for song, d in sd.items():
                t = {int(k): v for k, v in d.items()}
                rows.setdefault((arm, song), {})[st] = t[max(t)][0] - t[0][0]
    for (arm, song), d in sorted(rows.items()):
        v = np.array(list(d.values()))
        print("%-9s %-8s n=%d mean %+7.3f sd %6.3f  [%s]" % (
            arm, song, len(v), v.mean(), v.std(ddof=1) if len(v) > 1 else 0.0,
            " ".join("%+.2f" % x for x in v)))
    return rows


def probes_of(state):
    """(paired, unpaired) stimulus names for this run.

    Song runs written before --cs existed have no "probes" key; they were always
    (misery, skyhigh, mozart), so that is the fallback and old results reproduce.
    Odour runs carry ("da1", "dc2", "d")."""
    try:
        pr = json.load(open(os.path.join(RESULTS, "condition_%s.json" % state))).get("probes")
    except (OSError, ValueError):
        pr = None
    return tuple(pr[:2]) if pr else ("misery", "skyhigh")


def dn_profile(state, arm):
    """dS per DN type: (paired trained - paired baseline) - (unpaired trained - unpaired baseline).
    Also returns per-type z of each stimulus's shift against its baseline probe sd."""
    d = os.path.join(RESULTS, "rates_%s" % state)
    paired, unpaired = probes_of(state)
    out = {}
    for song in (paired, unpaired):
        fs = sorted(glob.glob(os.path.join(d, "rates_%s_%s_t*.npy" % (arm, song))))
        if len(fs) < 2:
            return None
        base = np.load(fs[0]).astype(np.float32); last = np.load(fs[-1]).astype(np.float32)
        b = np.stack([base[:, DN_IDX[t]].mean(1) for t in DN_TYPES], 1)   # probes x types
        l = np.stack([last[:, DN_IDX[t]].mean(1) for t in DN_TYPES], 1)
        out[song] = (l.mean(0) - b.mean(0), b.std(0) + 1e-3)
    dS = out[paired][0] - out[unpaired][0]
    z = np.maximum(np.abs(out[paired][0]) / out[paired][1],
                   np.abs(out[unpaired][0]) / out[unpaired][1])
    return dS, z


def population(states):
    print("\n== population DN readout, %d DN types ==" % len(DN_TYPES))
    prof = {}
    for st in states:
        for arm in ("learn", "lesion", "shuffle", "reversed", "frozen"):
            p = dn_profile(st, arm)
            if p is not None:
                prof[(st, arm)] = p
    if not prof:
        print("no rate dumps found"); return
    r = lambda a, b: float(np.corrcoef(a, b)[0, 1])
    for (st, arm), (dS, z) in sorted(prof.items()):
        print("%-6s %-9s |dS| mean %.3f  n(|z|>3) %3d / %d" % (st, arm, np.abs(dS).mean(), (z > 3).sum(), len(z)))
    print("\nr(dS) between arms, same seed:")
    for st in states:
        L = prof.get((st, "learn"))
        if L is None:
            continue
        for arm in ("reversed", "lesion", "shuffle", "frozen"):
            if (st, arm) in prof:
                print("  %s learn vs %-9s r = %+.3f" % (st, arm, r(L[0], prof[(st, arm)][0])))
    print("\nr(dS) learn vs learn across seeds:")
    ls = [(st, prof[(st, "learn")][0]) for st in states if (st, "learn") in prof]
    for i in range(len(ls)):
        for j in range(i + 1, len(ls)):
            print("  %s vs %s r = %+.3f" % (ls[i][0], ls[j][0], r(ls[i][1], ls[j][1])))
    # Probe seeds differ between baseline and the last test block, so every arm of a seed
    # shares the same seed noise; the lesion arm (US given, rule off) is that noise alone.
    # Subtracting it leaves the plasticity effect E per arm.
    print("\nlesion-referenced effect E = dS(arm) - dS(lesion):")
    Es = {}
    for st in states:
        if (st, "lesion") not in prof:
            continue
        for arm in ("learn", "reversed", "shuffle"):
            if (st, arm) in prof:
                Es[(st, arm)] = prof[(st, arm)][0] - prof[(st, "lesion")][0]
    for (st, arm), E in sorted(Es.items()):
        line = "  %-6s %-9s |E| mean %.3f  n(|E|>1Hz) %3d" % (st, arm, np.abs(E).mean(), (np.abs(E) > 1).sum())
        if arm != "learn" and (st, "learn") in Es:
            line += "  r(E, E_learn) = %+.3f" % r(E, Es[(st, "learn")])
        print(line)
    El = [(st, Es[(st, "learn")]) for st in states if (st, "learn") in Es]
    for i in range(len(El)):
        for j in range(i + 1, len(El)):
            print("  E_learn %s vs %s r = %+.3f" % (El[i][0], El[j][0], r(El[i][1], El[j][1])))
    if len(El) > 1:
        m = np.mean([x[1] for x in El], 0); sd = np.std([x[1] for x in El], 0, ddof=1) + 1e-6
        hit = np.abs(m) / sd > 3
        print("  DN types with |mean E| > 3 sd across seeds: %d / %d" % (hit.sum(), len(m)))
        print("  top-10 DN types by |mean E| (Hz, mean +- sd):")
        for i in np.argsort(-np.abs(m))[:10]:
            print("    %-10s %+.2f +- %.2f" % (DN_TYPES[i], m[i], sd[i]))
    if len(ls) > 1:
        m = np.mean([x[1] for x in ls], 0)
        top = np.argsort(-np.abs(m))[:10]
        print("\ntop-10 DN types by |mean dS| across seeds (Hz):")
        for i in top:
            print("  %-10s %+.2f" % (DN_TYPES[i], m[i]))


if __name__ == "__main__":
    states = sys.argv[1:] or sorted(re.sub(r".*condition_(v1s\d+)\.json$", r"\1", f)
                                    for f in glob.glob(os.path.join(RESULTS, "condition_v1s*.json")))
    seed_table(states)
    population(states)
