"""Fig 3: sugar does not reach the reward neurons (paper 1, R2).

Reads only committed result files; no simulation. Prints (ASCII) every plotted number.
  a  compartment rates under Fox drive vs real sugar: unfitted rate model, fitted (cap 16, 5 starting points)
  b  route from sugar-sensing neurons to Fox: relay cell types, share of Fox input, GABA marked (floor 5)
  c  influence of sugar on Fox by route length (exactly k synapses), signed and unsigned, vs DM4 ORN -> PN
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("NEUROPLASTICFLY_ROOT") or os.path.normpath(os.path.join(HERE, "..", ".."))
RES = os.path.join(ROOT, "results")
ANAT = os.path.join(ROOT, "rate", "sugar_fox_anatomy.json")

REWARD = [("g4", "γ4"), ("g5", "γ5"), ("b2", "β2"), ("b'2", "β′2")]
PUNISH = [("g1", "γ1"), ("g2", "γ2")]
C_FOX, C_SUG, C_UNF = "#E69F00", "#0072B2", "#999999"   # Okabe-Ito

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "svg.fonttype": "none",
})


def load():
    unf = json.load(open(os.path.join(RES, "rate_chunk1_control_c0", "result.json")))["arms"]["C0"]["D_comp"]
    cells = [json.load(open(os.path.join(RES, "rate_chunk1_fit_c0", "cells", "cap016.000_s%d.json" % s)))
             for s in range(5)]
    assert all(c["cap"] == 16.0 and c["fpass"] for c in cells)
    anat = json.load(open(ANAT))["5"]
    return unf, cells, anat


def main():
    unf, cells, anat = load()
    comps = REWARD + PUNISH
    fox = np.array([[c["D_comp"]["FOX"][k] for k, _ in comps] for c in cells])
    sug = np.array([[c["D_comp"]["SUGAR"][k] for k, _ in comps] for c in cells])
    unf_fox = np.array([unf["FOX"][k] for k, _ in comps])
    print("a  compartment  unfitted-Fox  fitted-Fox(min-max)  fitted-sugar(max)")
    for i, (k, lab) in enumerate(comps):
        print("   %-4s %.3f  %.3f-%.3f  %.4f" % (k, unf_fox[i], fox[:, i].min(), fox[:, i].max(), sug[:, i].max()))
    print("   max sugar over reward compartments: %.4f Hz (text: below 0.002 Hz)" % sug[:, :4].max())

    fig = plt.figure(figsize=(7.2, 3.0))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.5, 1.1, 0.9], wspace=0.55)

    # a
    ax = fig.add_subplot(gs[0])
    x = np.arange(len(comps))
    w = 0.26
    ax.bar(x - w, unf_fox, w, color=C_UNF, label="Fox drive, unfitted (all 0 Hz)")
    ax.bar(x, fox.mean(0), w, color=C_FOX, label="Fox drive, fitted")
    ax.bar(x + w, sug.mean(0), w, color=C_SUG, label="real sugar, fitted")
    rng = np.random.default_rng(0)
    for i in range(len(comps)):
        ax.scatter(x[i] + rng.uniform(-0.06, 0.06, 5), fox[:, i], s=6, color="k", zorder=3, linewidths=0)
        ax.scatter(x[i] + w + rng.uniform(-0.06, 0.06, 5), sug[:, i], s=6, color="k", zorder=3, linewidths=0)
    ax.hlines(1.0, -0.5, len(REWARD) - 0.5, colors="k", linestyles=":", linewidth=0.8)
    ax.text(-0.45, 1.12, "target 1 Hz", fontsize=6, ha="left", va="bottom")
    ax.axvline(len(REWARD) - 0.5, color="0.7", linewidth=0.6)
    ax.text(1.5, 7.3, "reward (PAM)", ha="center", fontsize=7)
    ax.text(4.5, 7.3, "punishment (PPL1),\nnot scored", ha="center", fontsize=6.5)
    ax.set_xticks(x, [lab for _, lab in comps])
    ax.set_ylim(0, 8.2)
    ax.set_ylabel("compartment rate (Hz)")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=6)
    ax.set_title("a", loc="left", fontweight="bold")

    # b
    ax = fig.add_subplot(gs[1])
    top = sorted(anat["top_two_hop_types"], key=lambda t: t["X_share_of_fox_input"])
    names = [t["type"] for t in top]
    share = [100 * t["X_share_of_fox_input"] for t in top]
    cols = ["#CC79A7" if t["nt"] == "gaba" else "#56B4E9" for t in top]
    y = np.arange(len(top))
    ax.barh(y, share, color=cols)
    for yi, t in zip(y, top):
        ax.text(100 * t["X_share_of_fox_input"] + 0.05, yi, "%d%% sugar" % round(100 * t["sugar_share_of_X_input"]),
                va="center", fontsize=5.5)
    ax.set_yticks(y, names, fontsize=6)
    ax.set_xlabel("share of Fox input (%)")
    ax.set_xlim(0, 2.8)
    ax.text(0.02, 1.02, "direct sugar → Fox: %d synapses (floor %d)" % (anat["direct_sugar_to_fox_syn"], anat["floor"]),
            transform=ax.transAxes, fontsize=6)
    ax.scatter([], [], marker="s", color="#CC79A7", label="GABA")
    ax.scatter([], [], marker="s", color="#56B4E9", label="acetylcholine")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, fontsize=6)
    ax.set_title("b", loc="left", fontweight="bold", x=-0.35)
    print("b  relays (type, %% of Fox input, %% sugar of its input, nt):")
    for t in reversed(top):
        print("   %-10s %.2f  %.1f  %s" % (t["type"], 100 * t["X_share_of_fox_input"], 100 * t["sugar_share_of_X_input"], t["nt"]))
    print("   direct %d, two-hop interneurons %d carry %.1f%% of Fox input (%d synapses)" % (
        anat["direct_sugar_to_fox_syn"], anat["n_two_hop_interneurons"], 100 * anat["two_hop_share_of_fox_input"],
        anat["fox_total_input_syn"]))

    # c
    ax = fig.add_subplot(gs[2])
    hops = anat["influence_sugar_to_fox_by_hop"]
    ks = sorted(hops, key=int)
    signed = [100 * hops[k]["signed"] for k in ks]
    absv = [100 * hops[k]["abs"] for k in ks]
    xk = np.arange(len(ks))
    ax.bar(xk - 0.18, absv, 0.36, color="0.75", label="unsigned")
    ax.bar(xk + 0.18, signed, 0.36, color=C_SUG, label="net of inhibition")
    ref = 100 * anat["ref_ornDM4_share_of_dm4pn_input"]
    ax.set_yscale("log")
    ax.set_ylim(0.01, 150)
    ax.axhline(ref, color="k", linestyle="--", linewidth=0.8)
    ax.text(-0.4, ref * 1.2, "DM4 receptor → PN,\n1 synapse: %.0f%%" % ref, fontsize=5.5, ha="left", va="bottom")
    ax.set_xticks(xk, ["%s" % k for k in ks])
    ax.set_xlabel("route length (synapses)")
    ax.set_ylabel("sugar influence on Fox (%)")
    ax.legend(frameon=False, loc="upper center", fontsize=6, bbox_to_anchor=(0.5, -0.22), ncol=1)
    ax.set_title("c", loc="left", fontweight="bold")
    print("c  hop signed%% abs%%: " + ", ".join("%s %.2f/%.2f" % (k, s, a) for k, s, a in zip(ks, signed, absv)))
    print("   max signed %.2f%%, max abs %.2f%% (text: at most 0.9%% / 2.2%%); DM4 ref %.1f%%" % (max(signed), max(absv), ref))

    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(HERE, "fig3." + ext), dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
