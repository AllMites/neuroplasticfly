"""Rung 4a: after the stimulus is switched OFF, does anything outlast the passive decay?

WHAT THE MUSHROOM BODY CANNOT DO. Rung 3 measured an accumulating brain, but the
accumulation lives entirely in synaptic weights and every update needs a US trial.
Between trials the brain holds nothing at all: results/persist_p1s0.jsonl, all 90
relax rows, zero drive ->

    {"phase": "relax", "central_active": 0.0, "kc_active": 0.0, "dn_DNp01": 0.0}

So the capability the MB does not have is carrying state in ACTIVITY. That is what
this script measures, and it is the only rung-4 question that can be asked before
any new machinery exists.

THE TRAP THIS SCRIPT IS BUILT AROUND. Rung 3's forgetting curve turned out to be
PL.LAM read back out (0.99^30 = 0.7397 predicted, 0.740 measured, zero variance).
Rung 4 has the identical trap: the brain is silent from rest because the neuron
model guarantees it. V_REST == V_RESET == -52.0, V_TH == -45.0, and gpu_sim has no
background or noise current anywhere. A leaky integrate-and-fire neuron with no
input decays to rest and never spikes. "The simulated brain has no spontaneous
activity" is NOT a finding -- it is V_REST.

The question that is not a restatement of a constant:

    after a stimulus is switched off, does activity outlast the PASSIVE ceiling?

The ceiling is known and small. TAU_M = 20 ms, TAU_SYN = 5 ms, DELAY = 1.8 ms,
T_REFR = 2.2 ms. Five membrane time constants is 100 ms. A spike at t_off + 100 ms
cannot be leak; it has to have gone round a loop. That is a property of the
connectome, not of a number we chose.

FALSIFIERS, fixed before the run:
  0. Positive control. The ON arm (drive never removed) must keep firing for the
     whole window. If it does not, the state carry is broken and nothing else here
     means anything. regime/test_carry.py is the independent check on the same
     mechanism (3 x 100 ms carried == 1 x 300 ms, exactly).
  1. PASS, recurrence exists: at least one population still spiking at
     t_off + 100 ms, reproducing across seeds.
  2. FAIL, no recurrence: everything silent by 100 ms. The claim is then
     "the connectome as simulated supports no persistent activity", which is a
     real bounded result about the model and the honest end of rung 4.
  3. Not claimable either way: anything about persistent activity in the animal.

Both arms share ONE 300 ms ignition phase at the same seed, so the comparison is
matched spike-for-spike up to the offset.

Run: .venv/Scripts/python.exe regime/offset_persistence.py --seeds 3
"""
import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "flywatch"))
import gpu_sim as G
from learn import condition as C

OUT_FMT = os.path.join(_HERE, "results", "offset_persistence_%s.json")

# Central-complex populations by cell_type prefix, plus the mushroom-body and output
# populations the earlier rungs already report. Counts confirmed present in
# neuron_meta.npz on 2026-09-22: EPG 47, PEN 42, PEG 20, ER 278, FB 593, FC 235,
# FS 293, PFL 50, ExR 26, EL 17.
CX_PREFIX = ["EPG", "PEG", "PEN", "ER", "ExR", "EL", "FB", "FC", "FS", "PFL"]


def populations(meta):
    ct = meta["cell_type"].astype(str)
    cc = meta["cell_class"].astype(str)
    sc = meta["super_class"].astype(str)
    pops = {}
    for p in CX_PREFIX:
        idx = np.flatnonzero(np.char.startswith(ct, p))
        if len(idx):
            pops["cx:" + p] = idx
    for k in ("Kenyon_Cell", "MBON", "ALPN", "DAN"):
        idx = np.flatnonzero(cc == k)
        if len(idx):
            pops["mb:" + k] = idx
    for k, idx in C.DN.items():
        if len(idx):
            pops["dn:" + k] = idx
    pops["all:central"] = np.flatnonzero(sc == "central")
    pops["all:brain"] = np.arange(len(ct))
    return pops


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--cs", default="dc2", choices=sorted(C.ODOURS))
    ap.add_argument("--regime", default="eln8", choices=sorted(C.REGIMES))
    ap.add_argument("--on-ms", type=float, default=300.0, help="ignition phase before offset")
    ap.add_argument("--off-ms", type=float, default=500.0, help="window measured after offset")
    ap.add_argument("--bin-ms", type=float, default=20.0)
    ap.add_argument("--drive", default="", help="drive these cell_types directly at --hz "
                    "instead of the odour CS, e.g. --drive EPG,PEN. Exists because the odour "
                    "CS reaches no central-complex population at all (measured 2026-09-22), "
                    "so asking whether the CX persists needs the CX to be lit first.")
    ap.add_argument("--hz", type=float, default=C.ORN_HZ)
    ap.add_argument("--tag", default="dc2")
    a = ap.parse_args()

    # Set before construction: baked into the device weights by GpuSim.__init__.
    G.ELN_NEGATE, G.PN_KC_GAIN = C.REGIMES[a.regime]
    passive_ms = 5.0 * G.TAU_M
    print("regime %s (ELN_NEGATE=%s PN_KC_GAIN=%.1f), cs %s, on %.0f ms, off %.0f ms, bin %.0f ms"
          % (a.regime, G.ELN_NEGATE, G.PN_KC_GAIN, a.cs, a.on_ms, a.off_ms, a.bin_ms), flush=True)
    print("passive ceiling: 5 * TAU_M = %.0f ms (TAU_SYN %.0f, DELAY %.1f, T_REFR %.1f)"
          % (passive_ms, G.TAU_SYN, G.DELAY, G.T_REFR), flush=True)

    meta = np.load(os.path.join(_HERE, "data", "neuron_meta.npz"), allow_pickle=False)
    pops = populations(meta)
    print("%d populations: %s" % (len(pops), ", ".join("%s(%d)" % (k, len(v))
                                                       for k, v in pops.items())), flush=True)

    sim = G.GpuSim(compile_=False)
    if a.drive:
        ct = meta["cell_type"].astype(str)
        want = [s for s in a.drive.split(",") if s.strip()]
        on_idx = np.flatnonzero(np.isin(ct, want) | np.any(
            [np.char.startswith(ct, w) for w in want], axis=0)).astype(np.int64)
        assert on_idx.size, "no neuron matches --drive %s" % a.drive
        on_hz = np.full(len(on_idx), a.hz, np.float32)
        print("direct drive: %s -> %d neurons at %.0f Hz" % (a.drive, len(on_idx), a.hz))
    else:
        on_idx, on_hz = C.odour_drive(a.cs)
    on = C.to_prob(on_idx, on_hz)
    off = (np.zeros(0, np.int64), np.zeros(0, np.float64))

    # Batch layout: 2 arms x seeds. Arm 0 keeps the drive (positive control), arm 1
    # loses it at t_off. The ignition phase is identical for both -- same drive, same
    # seed - so any later difference is the offset and nothing else.
    seeds = [s for s in range(a.seeds) for _ in range(2)]
    arms = ["on", "off"] * a.seeds
    ign = sim.run_batch([on] * len(seeds), seeds, t_run=a.on_ms, return_state=True)
    ign_counts, st = ign
    ign_counts = ign_counts.cpu().numpy()
    print("ignition done; KC active frac %.4f"
          % float((ign_counts[1][pops["mb:Kenyon_Cell"]] > 0).mean()), flush=True)

    post = [on if arm == "on" else off for arm in arms]
    n_bins = int(round(a.off_ms / a.bin_ms))
    bins = []
    for k in range(n_bins):
        c, st = sim.run_batch(post, seeds, t_run=a.bin_ms, state=st, return_state=True)
        bins.append(c.cpu().numpy())
    counts = np.stack(bins)                       # [bin, batch, N]
    t_ms = (np.arange(n_bins) + 1) * a.bin_ms     # right edge of each bin, ms after offset

    # ---- per population, per arm: rate per bin and the last bin carrying any spike
    #
    # A population that never fired during ignition cannot be scored for persistence:
    # "silent after the offset" and "never switched on" are the same row of zeros and
    # completely different claims. The odour CS reaches the antennal lobe and the
    # mushroom body; on the measured run it reaches no central-complex population at
    # all, so the CX is NOT_ENGAGED here rather than non-persistent.
    res = {}
    for name, idx in pops.items():
        ign_hz = ign_counts[:, idx].sum(axis=1) / (len(idx) * a.on_ms / 1000.0)
        per_arm = {"ignition_hz": float(ign_hz.mean()),
                   "engaged": bool(ign_hz.mean() > 0)}
        for arm in ("on", "off"):
            cols = [i for i, x in enumerate(arms) if x == arm]
            hz = counts[:, cols][:, :, idx].sum(axis=2) / (len(idx) * a.bin_ms / 1000.0)  # [bin, seed]
            last = []
            for s in range(hz.shape[1]):
                nz = np.flatnonzero(hz[:, s] > 0)
                last.append(float(t_ms[nz[-1]]) if len(nz) else 0.0)
            per_arm[arm] = {"hz_mean": [float(x) for x in hz.mean(axis=1)],
                            "last_spike_ms": last,
                            "hz_at_ceiling": [float(x) for x in
                                              hz[int(passive_ms / a.bin_ms) - 1]]}
        res[name] = {"n": int(len(idx)), **per_arm}

    # ---- report
    print("\nlast spike after offset, per population (ms; %.0f ms is the passive ceiling)"
          % passive_ms)
    print("| population | n | ignition Hz | ON last | OFF last (per seed) | OFF Hz at %.0f ms |"
          % passive_ms)
    survivors, not_engaged = [], []
    for name, r in res.items():
        off_last = r["off"]["last_spike_ms"]
        ceil_hz = r["off"]["hz_at_ceiling"]
        print("| %-18s | %6d | %11.3f | %5.0f | %-18s | %8.3f |%s"
              % (name, r["n"], r["ignition_hz"], max(r["on"]["last_spike_ms"]),
                 " ".join("%.0f" % x for x in off_last), float(np.mean(ceil_hz)),
                 "" if r["engaged"] else "  NOT_ENGAGED"))
        if not r["engaged"]:
            not_engaged.append(name)
        elif all(x > 0 for x in ceil_hz):
            survivors.append(name)
    if not_engaged:
        print("\nNOT_ENGAGED (never fired during ignition; persistence is not defined for")
        print("them and they are excluded from the verdict): %s" % ", ".join(not_engaged))

    # falsifier 0: the positive control
    on_ok = [name for name, r in res.items()
             if r["n"] > 50 and min(r["on"]["last_spike_ms"]) >= a.off_ms - a.bin_ms]
    print("\npositive control: %d populations still firing at the end of the ON arm" % len(on_ok))
    if not on_ok:
        print("CONTROL FAILED - the drive-on arm died too. The state carry or the drive is")
        print("broken; nothing below is interpretable. Cross-check regime/test_carry.py.")

    engaged = [n for n, r in res.items() if r["engaged"]]
    verdict = "PASS" if survivors else ("FAIL" if engaged else "UNDECIDED")
    print("\nRUNG 4a: %s  (%d/%d populations engaged by this CS)"
          % (verdict, len(engaged), len(res)))
    if survivors:
        print("populations still spiking at t_off + %.0f ms in EVERY seed: %s"
              % (passive_ms, ", ".join(survivors)))
        print("That is beyond five membrane time constants, so it is not leak. Recurrence.")
    elif engaged:
        print("Every ENGAGED population is silent by t_off + %.0f ms in at least one seed."
              % passive_ms)
        print("The connectome as simulated carries no activity past its passive decay, on the")
        print("pathway this stimulus reaches: no working memory, nothing to hold a bump.")
        print("Two things this does NOT say. (a) Nothing about the animal - gpu_sim has no")
        print("background current at all, so silence from rest is V_REST, not a measurement.")
        print("(b) Nothing about the %d NOT_ENGAGED populations, which this CS never lit."
              % len(not_engaged))
        cx_engaged = [n for n in engaged if n.startswith("cx:")]
        if not cx_engaged:
            print("NO central-complex population was engaged at all, so the CX is untested")
            print("here. Re-run with --drive EPG,PEN,PEG to light it directly first.")
        else:
            print("The central complex WAS engaged (%s) and died with everything else, so the"
                  % ", ".join(cx_engaged))
            print("result covers the CX too: there is no bump to hold. cx_bump.py is moot.")

    out = OUT_FMT % a.tag
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump({"args": vars(a), "passive_ceiling_ms": passive_ms, "t_ms": t_ms.tolist(),
               "constants": {"TAU_M": G.TAU_M, "TAU_SYN": G.TAU_SYN, "V_REST": G.V_REST,
                             "V_TH": G.V_TH, "DELAY": G.DELAY, "T_REFR": G.T_REFR},
               "verdict": verdict, "survivors": survivors, "not_engaged": not_engaged,
               "populations": res},
              open(out, "w"), indent=1)
    print("wrote", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
