"""docs/figures/rung_a.png: the rung A learning curve, learn arm, mean over 5 seeds.

Reads results/condition_o1s{0..4}.json (the published run,
docs/superpowers/option-a/odour_conditioning_2026-09-21.md) and plots the avoid index
against training trial for the paired, unpaired and never-paired odour. Shaded band is
the across-seed sd -- five noise replicates of ONE brain, not five flies.

Run: python scripts/plot_rung_a.py [--states o1s0,o1s1,o1s2,o1s3,o1s4]
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "docs", "figures", "rung_a.png")
LABEL = {"dc2": "ORN_DC2  paired with punishment",
         "d": "ORN_D  unpaired",
         "da1": "ORN_DA1  never paired"}
COLOR = {"dc2": "#c1272d", "d": "#2b6cb0", "da1": "#777777"}


def curves(states):
    """{stim: (trials, mean over seeds, sd over seeds)} for the learn arm."""
    per = {}
    for st in states:
        path = os.path.join(HERE, "results", "condition_%s.json" % st)
        if not os.path.exists(path):
            raise SystemExit("missing %s; run the seeds first (see the rung A doc)" % path)
        for stim, d in json.load(open(path))["curves"]["learn"].items():
            t = sorted(int(k) for k in d)
            per.setdefault(stim, []).append((t, [d[str(k)][0] for k in t]))
    out = {}
    for stim, runs in per.items():
        t = runs[0][0]
        assert all(r[0] == t for r in runs), "seeds disagree on the trial grid"
        v = np.array([r[1] for r in runs], float)
        out[stim] = (np.array(t), v.mean(0), v.std(0, ddof=1) if len(v) > 1 else v[0] * 0)
    return out


def main(argv):
    states = ["o1s%d" % i for i in range(5)]
    for a in argv:
        if a.startswith("--states="):
            states = a.split("=", 1)[1].split(",")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cv = curves(states)
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    for stim in ("dc2", "d", "da1"):
        if stim not in cv:
            continue
        t, m, s = cv[stim]
        ax.plot(t, m, marker="o", ms=3.5, lw=2, color=COLOR[stim], label=LABEL[stim])
        ax.fill_between(t, m - s, m + s, color=COLOR[stim], alpha=0.15, lw=0)
    ax.set_xlabel("training trials")
    ax.set_ylabel("avoid index (Hz)")
    ax.set_title("Odour conditioning through the real antennal lobe (%d seeds, one brain)"
                 % len(states), fontsize=10)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT, dpi=150)
    last = {k: v[1][-1] - v[1][0] for k, v in cv.items()}
    print("wrote %s" % OUT)
    print("  shift last-baseline: " + "  ".join("%s %+.2f Hz" % (k, v) for k, v in sorted(last.items())))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
