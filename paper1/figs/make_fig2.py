"""Fig 2: calibration (paper 1, R1). Reads fig2_r.json (compute_fig2_r.py), results/rate_chunk0/result.json,
rate/regress/latch_ref.json. No simulation. Prints every plotted number.
  a  correlation with the spiking model per held-out odour, bar 0.8 x self-agreement C
  b  fraction of KCs active per held-out odour, rate vs spiking
  c  neurons still firing 500-1000 ms after the DM4 odour ends: rate model vs each spiking replicate
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("NEUROPLASTICFLY_ROOT") or os.path.normpath(os.path.join(HERE, "..", ".."))
FC = ROOT
C_RATE, C_LIF = "#0072B2", "#D55E00"

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "svg.fonttype": "none",
})


def main():
    fr = json.load(open(os.path.join(HERE, "fig2_r.json")))
    res = json.load(open(os.path.join(FC, "results", "rate_chunk0", "result.json")))
    m = res["families"]["lin"]["metrics"]
    lat_rate = res["families"]["lin"]["latch"]["ORN_DM4"]["n_above"]
    lat_lif = json.load(open(os.path.join(FC, "rate", "regress", "latch_ref.json")))["conditions"]["ORN_DM4"]["n_per_seed"]
    r = np.array(fr["r"])
    print("a  per-odour r min %.3f median %.3f max %.3f; bar %.3f; C %.3f" % (np.nanmin(r), fr["median"], np.nanmax(r), fr["bar"], fr["C"]))
    kc_m, kc_l = np.array(m["L3_kc_model"]), np.array(m["L3_kc_lif"])
    print("b  KC frac median rate %.4f vs spiking %.4f (n=%d)" % (np.median(kc_m), np.median(kc_l), len(kc_m)))
    print("c  DM4 after-offset: rate %d; spiking replicates %s (max %d)" % (lat_rate, lat_lif, max(lat_lif)))

    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.4), gridspec_kw={"width_ratios": [1.5, 1, 0.9], "wspace": 0.5})
    ax = axs[0]
    o = np.argsort(r)
    ax.scatter(np.arange(len(r)), r[o], s=10, color=C_RATE, zorder=3)
    ax.axhline(fr["bar"], color="k", linestyle=":", linewidth=0.8)
    ax.axhline(fr["C"], color="0.5", linestyle="--", linewidth=0.8)
    ax.axhline(fr["median"], color=C_RATE, linewidth=0.6)
    ax.text(len(r) - 0.5, fr["bar"] - 0.03, "bar (0.8 × self-agreement)", fontsize=5.5, ha="right", va="top")
    ax.text(len(r) - 0.5, fr["C"] + 0.01, "spiking self-agreement", fontsize=5.5, ha="right")
    ax.text(0, fr["median"] + 0.01, "median %.3f" % fr["median"], fontsize=5.5, color=C_RATE)
    ax.set_xticks([])
    ax.set_xlabel("held-out odours (25), sorted")
    ax.set_ylabel("correlation with spiking model")
    ax.set_ylim(-0.05, 1.0)
    lo = int(np.nanargmin(r))
    ax.annotate("DA3: only its 30 driven\nreceptor neurons fire\nin the spiking model", (0, r[lo]), (3, 0.12), fontsize=5.5,
                arrowprops=dict(arrowstyle="-", lw=0.5))
    print("   above bar: %d of %d; lowest: %s" % ((r >= fr["bar"]).sum(), len(r), sorted(zip(r.round(3), fr["names"]))[:3]))
    ax.set_title("a", loc="left", fontweight="bold")

    ax = axs[1]
    lim = max(kc_m.max(), kc_l.max()) * 1.1
    ax.plot([0, lim], [0, lim], color="0.7", linewidth=0.6)
    ax.scatter(kc_l, kc_m, s=10, color=C_RATE)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("KCs active, spiking model")
    ax.set_ylabel("KCs active, rate model")
    ax.set_title("b", loc="left", fontweight="bold")

    ax = axs[2]
    ax.bar(range(5), lat_lif, 0.6, color=C_LIF)
    ax.bar(5.5, lat_rate, 0.6, color=C_RATE)
    ax.set_xticks(list(range(5)) + [5.5], ["s%d" % s for s in range(5)] + ["rate"], fontsize=6)
    ax.set_xlabel("spiking replicates")
    ax.set_ylabel("neurons firing after\nthe DM4 odour ends")
    ax.set_title("c", loc="left", fontweight="bold")
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(HERE, "fig2." + ext), dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
