"""Fig 4: odour learning and accumulation survive the swap (paper 1, R3).

Reads results/rate_chunk2/{result.json, reference.json}; no simulation. Prints every plotted number.
  a  specificity test: normalised drop of the approach-MBON response to DC2 after 30 trials;
     rate model (bare + 5 versions with reward gains) with its controls, beside the spiking-model reference
  b  accumulation test: drop for D on a trained brain / drop on a naive brain, rate vs spiking
Spiking-model bars = the preregistered ratio-of-means references; orange dots = per-replicate values derived from the
per-seed lists in reference.json (drop_s / baseline_s; trained-D drop_s / naive-D drop_s, paired by seed index).
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("NEUROPLASTICFLY_ROOT") or os.path.normpath(os.path.join(HERE, "..", ".."))
RES = os.path.join(ROOT, "results", "rate_chunk2")
C_RATE, C_LIF, C_CTRL = "#0072B2", "#D55E00", "#999999"

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "svg.fonttype": "none",
})


def main():
    res = json.load(open(os.path.join(RES, "result.json")))
    ref = json.load(open(os.path.join(RES, "reference.json")))
    c0 = res["c0"]["protocols"]
    c1 = [res["c1"]["seeds"][str(s)]["protocols"] for s in range(5)]
    ra, r3 = c0["rung_a"]["numbers"], c0["rung3"]["numbers"]

    lif_a = ref["rung_a"]["learn_dc2_approach_drop_norm"]
    lif_3 = ref["rung3"]["approach_ratio_of_means"]
    rr = ref["rung_a"]
    lif_a_s = [d / b for d, b in zip(rr["learn_dc2_approach_drop"]["per_seed"], rr["baseline"]["dc2"]["mbon_approach_hz"]["per_seed"])]
    lif_3_s = [t / v for t, v in zip(ref["rung3"]["trained_brain_d_approach_drop"]["per_seed"], rr["naive_d_approach_drop"]["per_seed"])]
    print("   LIF per-replicate specificity %s; accumulation %s" % (["%.3f" % v for v in lif_a_s], ["%.3f" % v for v in lif_3_s]))
    gain_a = [p["rung_a"]["numbers"]["norm_approach_drop"] for p in c1]
    gain_3 = [p["rung3"]["numbers"]["ratio"] for p in c1]
    print("a  rate bare %.4f, gains %s, lesion %.4f, shuffle %.4f; LIF %.4f; bar %.4f" % (
        ra["norm_approach_drop"], ["%.4f" % g for g in gain_a], ra["lesion_norm_approach_drop"],
        ra["shuffle_norm_approach_drop"], lif_a, ra["threshold_norm"]))
    print("   rate DC2 %.1f -> %.1f Hz; unpaired D drop %.2f Hz" % (
        ra["baseline_approach_hz"], ra["baseline_approach_hz"] - ra["approach_drop_hz"], ra["d_approach_drop_hz"]))
    print("b  rate bare %.4f (%.2f / %.2f Hz), gains %s; LIF %.4f; bar %.4f" % (
        r3["ratio"], r3["trained_d_approach_drop_hz"], r3["naive_d_approach_drop_hz"], ["%.4f" % g for g in gain_3],
        lif_3, r3["threshold_ratio"]))

    fig, axs = plt.subplots(1, 2, figsize=(5.2, 2.4), gridspec_kw={"width_ratios": [1.6, 1], "wspace": 0.5})
    rng = np.random.default_rng(0)

    ax = axs[0]
    labs = ["spiking\nmodel", "rate\nmodel", "rate,\nrule off", "rate,\nshuffled KCs"]
    vals = [lif_a, ra["norm_approach_drop"], ra["lesion_norm_approach_drop"], ra["shuffle_norm_approach_drop"]]
    cols = [C_LIF, C_RATE, C_CTRL, C_CTRL]
    ax.bar(range(4), vals, 0.6, color=cols)
    ax.scatter(1 + rng.uniform(-0.12, 0.12, 5), gain_a, s=8, color="k", zorder=3, linewidths=0,
               label="with reward gains (5 versions)")
    ax.scatter(0 + rng.uniform(-0.12, 0.12, 5), lif_a_s, s=8, color="k", marker="^", zorder=3, linewidths=0,
               label="spiking, single replicates")
    ax.axhline(ra["threshold_norm"], color="k", linestyle=":", linewidth=0.8)
    ax.text(3.45, ra["threshold_norm"] + 0.02, "bar: half of spiking", fontsize=6, ha="right")
    ax.text(2, 0.02, "0", ha="center", fontsize=6)
    ax.set_xticks(range(4), labs)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("drop in approach response to DC2\n(fraction of baseline)")
    ax.legend(frameon=False, loc="upper right", fontsize=6)
    ax.set_title("a  specificity", loc="left", fontweight="bold")

    ax = axs[1]
    ax.bar([0, 1], [lif_3, r3["ratio"]], 0.6, color=[C_LIF, C_RATE])
    ax.scatter(1 + rng.uniform(-0.12, 0.12, 5), gain_3, s=8, color="k", zorder=3, linewidths=0)
    ax.scatter(0 + rng.uniform(-0.12, 0.12, 5), lif_3_s, s=8, color="k", marker="^", zorder=3, linewidths=0)
    ax.axhline(r3["threshold_ratio"], color="k", linestyle=":", linewidth=0.8)
    ax.text(1.45, r3["threshold_ratio"] + 0.02, "bar", fontsize=6, ha="right")
    ax.set_xticks([0, 1], ["spiking\nmodel", "rate\nmodel"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("second odour learned\n(trained / naive drop)")
    ax.set_title("b  accumulation", loc="left", fontweight="bold")

    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(HERE, "fig4." + ext), dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
