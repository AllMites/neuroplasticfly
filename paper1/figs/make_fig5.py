"""Fig 5 + Fig S1: relearning (paper 1, R4). Reads raw logs + result files; no simulation. Prints every plotted number.

Fig 5
  a  approach response to DC2 at each test (T0, F1, B1 ... F4, B4), timing rule vs weaken-only rule,
     rate model (bare) vs spiking model (mean +- SD over 5 noise replicates), normalised to each run's T0
  b  the same in absolute Hz (comparison scale)
  c  normalised drop vs weight change |w_mean_frac|: specificity-test curves over trials (rate bare; spiking
     5 replicates) and the first relearning block (F1) of every run
Fig S1
  avoid-MBON and PAM rate at each test, timing rule, bare rate model vs the 5 versions with reward gains
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("NEUROPLASTICFLY_ROOT") or os.path.normpath(os.path.join(HERE, "..", ".."))
FC = ROOT
sys.path.insert(0, FC)
sys.path.insert(0, os.path.join(FC, "rate"))
import chunk2 as C2  # noqa: E402

RES = os.path.join(FC, "results")
TAGS = ["T0"] + ["%s%d" % (p, k) for k in range(1, 5) for p in "FB"]
C_RATE, C_LIF = "#0072B2", "#D55E00"

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "svg.fonttype": "none",
})


def rows(p):
    return C2.read_rows(os.path.join(RES, p))


def curve(r):
    return np.array([C2._field(r, "mbon_approach_hz", "dc2", after=t) for t in TAGS], dtype=float)


def fig5():
    ref = json.load(open(os.path.join(RES, "rate_chunk2_bidir_ref", "result.json")))
    rate = {rule: curve(rows("rate_chunk2/bidir_%s_c0.jsonl" % rule)) for rule in ("timed", "depress")}
    lif = {rule: np.array([curve(rows("rate_chunk2_bidir_ref/bidir_%s_lifs%d.jsonl" % (rule, s))) for s in range(5)])
           for rule in ("timed", "depress")}

    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.5), gridspec_kw={"wspace": 0.5})
    x = np.arange(len(TAGS))
    ls = {"timed": "-", "depress": "--"}
    lab = {"timed": "timing rule", "depress": "weaken-only rule"}
    for rule in ("timed", "depress"):
        rn = rate[rule] / rate[rule][0]
        ln = lif[rule] / lif[rule][:, :1]
        axs[0].plot(x, rn, ls[rule], color=C_RATE, marker="o", ms=2.5, label="rate, " + lab[rule])
        axs[0].errorbar(x, ln.mean(0), yerr=ln.std(0, ddof=1), fmt=ls[rule], color=C_LIF, marker="o", ms=2.5,
                        capsize=1.5, label="spiking, " + lab[rule])
        axs[1].plot(x, rate[rule], ls[rule], color=C_RATE, marker="o", ms=2.5)
        axs[1].errorbar(x, lif[rule].mean(0), yerr=lif[rule].std(0, ddof=1), fmt=ls[rule], color=C_LIF, marker="o",
                        ms=2.5, capsize=1.5)
        print("a  %-7s rate norm %s" % (rule, " ".join("%.3f" % v for v in rn)))
        print("   %-7s LIF  norm %s" % (rule, " ".join("%.3f" % v for v in ln.mean(0))))
        print("b  %-7s rate Hz %s | LIF Hz %s" % (rule, " ".join("%.1f" % v for v in rate[rule]),
                                                  " ".join("%.1f" % v for v in lif[rule].mean(0))))
    for ax in axs[:2]:
        ax.set_xticks(x, TAGS, rotation=90)
        for k in range(1, 5):
            ax.axvspan(2 * k - 1 - 0.5, 2 * k - 1 + 0.5, color="0.93", zorder=0)
    axs[0].set_ylabel("approach response to DC2\n(fraction of first test)")
    axs[0].set_ylim(0, 1.1)
    axs[0].legend(frameon=False, fontsize=5.5, loc="upper center", bbox_to_anchor=(1.2, -0.3), ncol=2)
    axs[0].set_title("a  normalised", loc="left", fontweight="bold")
    axs[1].set_ylabel("approach response to DC2 (Hz)")
    axs[1].set_ylim(0, None)
    axs[1].set_title("b  absolute", loc="left", fontweight="bold")

    ax = axs[2]
    rc = C2.rung_a_curve(rows("rate_chunk2/rung_a_c0.jsonl"))
    ax.plot([p[0] for p in rc], [p[1] for p in rc], "-", color=C_RATE, marker="s", ms=2.5, label="rate, specificity test")
    print("c  rate specificity curve (|w|, drop): " + " ".join("(%.4f, %.3f)" % p for p in rc))
    for s in range(5):
        lc = C2.rung_a_curve(rows("condition_o1s%d.jsonl" % s))
        ax.plot([p[0] for p in lc], [p[1] for p in lc], "-", color=C_LIF, marker="s", ms=2, alpha=0.6,
                label="spiking, specificity test" if s == 0 else None)
        print("   LIF s%d specificity curve: " % s + " ".join("(%.4f, %.3f)" % p for p in lc))
    lw = [ref["lif_reference"]["seeds"][str(s)]["abs_w_mean_f1"] for s in range(5)]
    lr = [ref["lif_reference"]["seeds"][str(s)]["numbers"]["rise_1_norm"] for s in range(5)]
    rw = [ref["rate"]["bases"][b]["abs_w_mean_f1"] for b in C2.BASES]
    rr = [ref["rate"]["bases"][b]["numbers"]["rise_1_norm"] for b in C2.BASES]
    ax.scatter(lw, lr, marker="*", s=40, color=C_LIF, edgecolor="k", linewidths=0.3, zorder=4,
               label="spiking, first relearning block")
    ax.scatter(rw, rr, marker="*", s=40, color=C_RATE, edgecolor="k", linewidths=0.3, zorder=4,
               label="rate, first relearning block")
    print("   F1 LIF (|w|, drop): " + " ".join("(%.4f, %.3f)" % p for p in zip(lw, lr)))
    print("   F1 rate (|w|, drop) %s: " % list(C2.BASES) + " ".join("(%.4f, %.3f)" % p for p in zip(rw, rr)))
    ax.set_xlabel(r"mean weight change $|\Delta w / w_0|$")
    ax.set_ylabel("drop in approach response\n(fraction of baseline)")
    ax.set_xlim(0, None)
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, fontsize=5.5, loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=1)
    ax.set_title("c  drop vs weight change", loc="left", fontweight="bold")
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(HERE, "fig5." + ext), dpi=300, bbox_inches="tight")


def figS1():
    ref = json.load(open(os.path.join(RES, "rate_chunk2_bidir_ref", "result.json")))
    fig, axs = plt.subplots(1, 2, figsize=(5.2, 2.3), gridspec_kw={"wspace": 0.45})
    x = np.arange(len(TAGS))
    for i, (key, ylab) in enumerate((("avoid_mbon_hz", "avoid MBONs (Hz)"), ("pam_hz", "PAM dopamine neurons (Hz)"))):
        ax = axs[i]
        for b in C2.BASES:
            ser = ref["rate"]["bases"][b]["series"]["timed"][key]
            y = [ser.get(t) for t in TAGS]
            bare = b == "c0"
            ax.plot(x, y, "-", color="k" if bare else "#CC79A7", lw=1.2 if bare else 0.8, marker="o", ms=2,
                    label=("bare rate model" if bare else ("with reward gains (5 versions)" if b == "c1s0" else None)))
            print("S1 %-12s %-5s %s" % (key, b, " ".join("%.2f" % v for v in y)))
        ax.set_xticks(x, TAGS, rotation=90)
        ax.set_ylabel(ylab)
        ax.set_ylim(0, None)
    axs[0].legend(frameon=False, fontsize=6)
    axs[0].set_title("a", loc="left", fontweight="bold")
    axs[1].set_title("b", loc="left", fontweight="bold")
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(HERE, "figS1." + ext), dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    fig5()
    figS1()
