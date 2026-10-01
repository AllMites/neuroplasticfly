"""Fig 1: the swap (paper 1). Illustrative single-neuron traces from each model's equations + study pipeline.
  a  one LIF neuron (parameters of Shiu et al. 2024 / paper 0) and one rate neuron (chunk-0 calibration:
     tau 17 ms, cap 124.5 Hz) given the same 200 ms input step; input scales are illustrative
  b  pipeline: same wiring -> calibrate -> reward fit -> learning tests (+ spiking relearning reference)
  c  how to read a result
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
C_RATE, C_LIF = "#0072B2", "#D55E00"
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "svg.fonttype": "none",
})

V_REST, V_TH, TAU_M, T_REF = -52.0, -45.0, 20.0, 2.2      # Shiu et al. 2024 (paper 0 M1)
TAU_R, R_MAX = 17.0, 124.5                                 # chunk-0 calibrated rate neuron
DT, T_END, ON, OFF = 0.1, 300.0, 50.0, 250.0


def lif(drive_mv):
    t = np.arange(0, T_END, DT)
    v, ref, vs, spikes = V_REST, 0.0, [], []
    for ti in t:
        inp = drive_mv if ON <= ti < OFF else 0.0
        if ref > 0:
            ref -= DT
        else:
            v += DT / TAU_M * (-(v - V_REST) + inp)
        if v >= V_TH:
            spikes.append(ti)
            vs.append(0.0)          # drawn spike
            v, ref = V_REST, T_REF
        else:
            vs.append(v)
    return t, np.array(vs), spikes


def rate(drive_hz):
    t = np.arange(0, T_END, DT)
    v, rs = 0.0, []
    for ti in t:
        inp = drive_hz if ON <= ti < OFF else 0.0
        v += DT / TAU_R * (-v + inp)
        rs.append(min(max(v, 0.0), R_MAX))
    return t, np.array(rs)


def box(ax, x, y, w, h, text, fc):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.03", fc=fc, ec="0.3", lw=0.6))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=5.8)


def main():
    fig = plt.figure(figsize=(7.2, 3.3))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1.5, 1], height_ratios=[1, 1], hspace=0.55, wspace=0.35)

    t, v, spikes = lif(9.0)
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(t, v, color=C_LIF, lw=0.6)
    ax.axhline(V_TH, color="0.6", ls=":", lw=0.6)
    ax.text(T_END, V_TH + 1, "threshold", fontsize=5.5, ha="right", color="0.4")
    ax.axvspan(ON, OFF, color="0.93", zorder=0)
    ax.set_ylabel("voltage (mV)")
    ax.set_xticks([])
    ax.set_title("a  LIF neuron: all-or-none spikes", loc="left", fontweight="bold")
    print("a  LIF spikes in 200 ms step: %d (%.0f Hz)" % (len(spikes), len(spikes) / 0.2))

    t, r = rate(len(spikes) / 0.2)
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(t, r, color=C_RATE, lw=1.0)
    ax.axvspan(ON, OFF, color="0.93", zorder=0)
    ax.set_ylabel("firing rate (Hz)")
    ax.set_xlabel("time (ms); grey = same input")
    ax.set_title("rate neuron: smooth rate, no spikes", loc="left", fontsize=8)
    print("   rate neuron steady output %.1f Hz" % r[int((OFF - 1) / DT)])

    ax = fig.add_subplot(gs[:, 1])
    ax.axis("off")
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.02, 1.04)
    ax.set_title("b  study design", loc="left", fontweight="bold")
    box(ax, 0.0, 0.84, 1.0, 0.12, "same wiring: FlyWire v783, floor 5,\nsign flip of 44 local neurons, PN-to-KC gain 8", "#F0F0F0")
    box(ax, 0.0, 0.64, 1.0, 0.12, "calibrate the rate model to the LIF model\n4 global numbers, checked on 25 held-out odours", "#DCEBF5")
    box(ax, 0.0, 0.44, 1.0, 0.12, "reward fit: gains downstream of Fox\nreal sugar held out as the test", "#FBE3CC")
    box(ax, 0.0, 0.18, 1.0, 0.18, "learning tests of paper 0, code and constants unchanged\nspecificity, accumulation, relearning\nbare rate model and with the reward gains", "#E3F1E0")
    box(ax, 0.0, 0.0, 1.0, 0.11, "spiking model on the same relearning schedule\n(reference at the same dose)", "#F7E1EC")
    for y0, y1 in ((0.84, 0.76), (0.64, 0.56), (0.44, 0.36)):
        ax.annotate("", (0.5, y1), (0.5, y0), arrowprops=dict(arrowstyle="->", lw=0.7))
    ax.text(0.5, 1.0, "every step preregistered", ha="center", fontsize=6, style="italic")

    ax = fig.add_subplot(gs[:, 2])
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("c  reading a result", loc="left", fontweight="bold")
    box(ax, 0.02, 0.58, 0.96, 0.3, "survives the swap\n↓\ncomes from the wiring\nand the plasticity rule", "#E3F1E0")
    box(ax, 0.02, 0.18, 0.96, 0.3, "changes with the swap\n↓\ndepends on the\nneuron model", "#FBE3CC")
    ax.text(0.5, 0.05, "within this model, not a living fly", ha="center", fontsize=6, style="italic")
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(HERE, "fig1." + ext), dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
