"""Conditioning protocol on the whole-brain fly. Two CS modes.

--cs song (default, unchanged): one trial = 300 ms run_batch with song JO drive
(gain 3, as reel 3) + song KC fingerprint (authored) + optional US.

--cs odour: the CS is a single ORN channel at 80 Hz through the real antennal
lobe - no graft, no authored KC fingerprint. CS pair DC2 + D (not DA1, which is
innately valenced - see ODOURS below). Intended with --regime eln8
(ELN_NEGATE + PN_KC_GAIN 8.0), the operating point measured in
docs/superpowers/option-a/kc_excitability_2026-09-20.md and gated per channel by
learn/gate_cs_channels.py. Punishment only: sugar does not
drive PAM in this matrix (measured with silencing controls against Christie et
al. 2026 in docs/superpowers/reward-path/christie_reconciliation_2026-09-23.md -
the path IS present anatomically and dies at FDA-I -> PAM, 0/307 PAMs firing),
so the US is PPL1 and that is disclosed rather than worked around.

US tries the real gustatory path first (sugar GRNs -> PAM, bitter GRNs -> PPL1);
if PAM/PPL1 do not rise above baseline it falls back to driving the DANs
directly, and says so in the log. For odour runs pass --us-path dan explicitly:
the auto check spends two trials proving what is already closed.

Usage:
  python learn/condition.py --state v1 --train 30
  python learn/condition.py --state v1 --check          # acceptance assert on the saved curves
  python learn/condition.py --state o1s0 --cs odour --regime eln8       --arms learn,lesion,shuffle --us-path dan --eta 5e-6 --dump
"""
import argparse
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _HERE); sys.path.insert(0, os.path.join(_HERE, "flywatch"))
import gpu_sim as G
from learn import fingerprint as F
from learn import plastic as PL

SONGS = {   # name: (wav, offset s)  -- same clips as reel 3 (HANDOFF-reel3.md)
    "misery": ("flywatch/raw/misery.wav", 0.0),
    "skyhigh": ("flywatch/raw/skyhigh.wav", 55.0),
    "mozart": ("flywatch/raw/mozart.wav", 85.0),
}
# Odour CS: one ORN channel each, driven at ORN_HZ through the real AL.
#
# The CS pair is DC2 + D, NOT DA1, per the reel-5 PRD decision of 2026-09-20:
# DA1 is the cVA pheromone glomerulus (Or67d; Kurtovic, Widmer & Dickson 2007).
# cVA carries innate valence and drives approach/avoidance without learning, so
# a shift on DA1 could be innate rather than learned and the shuffle arm would
# not separate the two. DC2 (0.0597) and D (0.0551) are general-odorant
# channels with nearly matched recruitment, both inside the 5-10% band.
#
# DA1 stays in the probe set as the third, never-paired channel. That is not
# decoration: if DA1 moves in the learn arm as much as the paired DC2 does, the
# shift is not CS-specific and the result is drift, not learning.
ODOURS = {"dc2": "ORN_DC2", "d": "ORN_D", "da1": "ORN_DA1"}
ORN_HZ = 80.0
# (paired, unpaired, never-paired). analyze_seeds reads these back out of the
# run json, so the population readout does not hardcode song names.
STIM_SETS = {"song": ("misery", "skyhigh", "mozart"),
             "odour": ("dc2", "d", "da1")}
REGIMES = {   # name: (ELN_NEGATE, PN_KC_GAIN). "stock" must stay a no-op.
    "stock": (False, 1.0),
    "eln8": (True, 8.0),
    # eln6 exists to test whether the Track 2 / rung 3 results are about the
    # connectome or about the authored gain. PN_KC_GAIN is the single largest
    # authored number in the regime, so it gets its own falsifier.
    "eln6": (True, 6.0),
}
AUDIO_GAIN = 3.0
GRN_HZ = 150.0
DAN_HZ = 100.0
T_RUN = 300.0
RESULTS = os.path.join(_HERE, "results")
STATES = os.path.join(_HERE, "brain_state")

meta = np.load(os.path.join(_HERE, "data", "neuron_meta.npz"), allow_pickle=True)
ct = meta["cell_type"].astype(str); cc = meta["cell_class"].astype(str)
csc = meta["cell_sub_class"].astype(str)
cen = meta["super_class"].astype(str) == "central"
SUGAR = np.flatnonzero(csc == "sugar/water").astype(np.int64)
BITTER = np.flatnonzero(csc == "bitter").astype(np.int64)
DN = {k: np.flatnonzero(ct == k) for k in ["DNp01", "DNg29", "DNg84", "DNa02", "DNp04"]}


def _sha256(path):
    import hashlib
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def song_drive(name, bands_cache={}):
    from sim_song import audio_drives
    if name not in bands_cache:
        wav, off = SONGS[name]
        bands_cache[name] = F.song_bands(os.path.join(_HERE, wav), 3.0, off)
    b = bands_cache[name]
    jo_idx, jo_hz = audio_drives(b[None, :], meta)[0]
    k_idx, k_hz = F.kc_drive(b)
    return np.concatenate([jo_idx, k_idx]), np.concatenate([jo_hz * AUDIO_GAIN, k_hz])


def odour_drive(name):
    idx = np.flatnonzero(ct == ODOURS[name]).astype(np.int64)
    assert idx.size, "odour channel %s (%s) not in this matrix" % (name, ODOURS[name])
    return idx, np.full(len(idx), ORN_HZ, np.float32)


def cs_drive(name):
    return odour_drive(name) if name in ODOURS else song_drive(name)


def us_drive(us, path, _dan_cache={}):
    """us: None|'punish'|'reward'; path: 'grn'|'dan'."""
    if us is None:
        return np.zeros(0, np.int64), np.zeros(0, np.float32)
    if path == "grn":
        idx = BITTER if us == "punish" else SUGAR
        return idx, np.full(len(idx), GRN_HZ, np.float32)
    if us not in _dan_cache:
        idx = PL.Plastic.real().dan_idx[us]
        _dan_cache[us] = (idx, np.full(len(idx), DAN_HZ, np.float32))
    return _dan_cache[us]


def to_prob(idx, hz):
    return idx.astype(np.int64), (hz * G.DT / 1000.0).astype(np.float64)


def trial(sim, song, us=None, us_path="grn", seed=0):
    i1, h1 = cs_drive(song); i2, h2 = us_drive(us, us_path)
    idx, hz = np.concatenate([i1, i2]), np.concatenate([h1, h2])
    counts = sim.run_batch([to_prob(idx, hz)], [seed], t_run=T_RUN)
    return counts[0].cpu().numpy().astype(np.float32) / (T_RUN / 1000.0)


def readout(rates, P):
    appr = np.unique(P.mbon_of_edge[P.mbon_valence_of_edge == "approach"])
    avoid = np.unique(P.mbon_of_edge[P.mbon_valence_of_edge == "avoid"])
    r = {"avoid_index": float(rates[avoid].mean() - rates[appr].mean()),
         "mbon_approach_hz": float(rates[appr].mean()), "mbon_avoid_hz": float(rates[avoid].mean()),
         "pam_hz": float(rates[P.dan_idx["reward"]].mean()), "ppl1_hz": float(rates[P.dan_idx["punish"]].mean()),
         "kc_active": float((rates[cc == "Kenyon_Cell"] > 1).mean()), "central_active": float((rates[cen] > 1).mean())}
    for k, v in DN.items():
        r["dn_" + k] = float(rates[v].mean())
    return r


def check_us_path(sim, P, cs="misery"):
    """Real gustatory path or direct DAN drive? Decided by measurement, logged."""
    base = readout(trial(sim, cs), P)
    out = {}
    for us, dan in (("punish", "ppl1_hz"), ("reward", "pam_hz")):
        r = readout(trial(sim, cs, us=us, us_path="grn"), P)
        out[us] = {"grn": r[dan], "baseline": base[dan], "ok": r[dan] > base[dan] + 1.0}
    path = "grn" if all(v["ok"] for v in out.values()) else "dan"
    return path, out


# ---------------------------------------------------------------- protocol
ARMS = {
    "learn":    {"lesion": "none", "pair": {"misery": "punish", "skyhigh": "reward"}},
    "lesion":   {"lesion": "all",  "pair": {"misery": "punish", "skyhigh": "reward"}},
    "reversed": {"lesion": "none", "pair": {"misery": "reward", "skyhigh": "punish"}},
    "frozen":   {"lesion": "none", "pair": {"misery": None, "skyhigh": None}},
    # Same wiring, same US, same rule, but the eligibility trace is read from a permuted KC
    # for every edge, so depression lands on random KC->MBON synapses instead of the ones
    # the song activated. Pairing-specificity control (flytris / fly-hero "shuffled").
    # ponytail: permuting w0 instead would be the other standard shuffle; add if asked.
    "shuffle":  {"lesion": "none", "pair": {"misery": "punish", "skyhigh": "reward"}, "shuffle": True},
}

# Odour arms are PUNISHMENT-ONLY. There is no reward arm because there is no
# reward path: sugar -> PAM is 30 synapses at its widest on the brain-wide
# floor-1 edge list (reward-path/floor1_convergence_2026-09-21.md). dc2 is
# paired, d is the unpaired control odour, "reversed" swaps which one is
# paired, and da1 is probed but never paired in any arm.
ODOUR_ARMS = {
    "learn":    {"lesion": "none", "pair": {"dc2": "punish", "d": None}},
    "lesion":   {"lesion": "all",  "pair": {"dc2": "punish", "d": None}},
    "reversed": {"lesion": "none", "pair": {"dc2": None, "d": "punish"}},
    "frozen":   {"lesion": "none", "pair": {"dc2": None, "d": None}},
    "shuffle":  {"lesion": "none", "pair": {"dc2": "punish", "d": None}, "shuffle": True},
}


def run_arm(sim, name, arm, n_train, n_probe, us_path, eta, lam, log, seed0=0, dump=None,
            probes=("misery", "skyhigh", "mozart")):
    P = PL.Plastic.real(); P.push(sim)
    if arm.get("shuffle"):
        P.kc_of_edge = np.random.default_rng(seed0 + 12345).permutation(P.kc_of_edge)
    rows = []
    def probe(phase, k):
        for s in probes:
            vecs = []
            for j in range(n_probe):
                rates = trial(sim, s, seed=seed0 + 1000 * k + j); vecs.append(rates)
                r = readout(rates, P)
                r.update(arm=name, phase=phase, trial=k, song=s, us=None, seed0=seed0)
                rows.append(r); log.write(json.dumps(r) + "\n")
            if dump:   # full rate vectors, for population (all-DN) readouts offline
                np.save(os.path.join(dump, "rates_%s_%s_t%02d.npy" % (name, s, k)),
                        np.stack(vecs).astype(np.float16))
    probe("baseline", 0)
    for k in range(1, n_train + 1):
        for s in arm["pair"]:
            us = arm["pair"][s]
            rates = trial(sim, s, us=us, us_path=us_path, seed=seed0 + 7 * k)
            if us is not None:
                P.update(rates, eta=eta, lam=lam, lesion=arm["lesion"]); P.push(sim)
            r = readout(rates, P); r.update(arm=name, phase="train", trial=k, song=s, us=us, seed0=seed0, **{"w_" + a: b for a, b in P.summary().items()})
            rows.append(r); log.write(json.dumps(r) + "\n")
        if k % 5 == 0 or k == n_train:
            probe("test", k)
    return P, rows


def curves(rows):
    """phase=='test' or 'baseline' avoid_index per (arm, song, trial)."""
    out = {}
    for r in rows:
        if r["phase"] in ("baseline", "test"):
            out.setdefault(r["arm"], {}).setdefault(r["song"], {}).setdefault(r["trial"], []).append(r["avoid_index"])
    return {a: {s: {int(t): [float(np.mean(v)), float(np.std(v))] for t, v in d.items()} for s, d in sd.items()} for a, sd in out.items()}


def plot(cv, path):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(cv), figsize=(4 * len(cv), 3), sharey=True)
    for ax, (arm, sd) in zip(np.atleast_1d(axes), cv.items()):
        for song, d in sd.items():
            t = sorted(d); m = [d[k][0] for k in t]; s = [d[k][1] for k in t]
            ax.errorbar(t, m, yerr=s, label=song, marker="o")
        ax.set_title(arm); ax.set_xlabel("training trials"); ax.set_ylabel("avoid index (Hz)")
    np.atleast_1d(axes)[0].legend() if len(cv) else None
    fig.tight_layout(); fig.savefig(path, dpi=120)


def check(cv):
    """Acceptance: learn arm moves misery up and skyhigh down beyond baseline SD; lesion arm does not."""
    norm = lambda sd: {s: {int(t): v for t, v in d.items()} for s, d in sd.items()}
    L = norm(cv["learn"]); Z = norm(cv["lesion"])
    last = max(L["misery"]); sd = max(L["misery"][0][1], L["skyhigh"][0][1], 1e-3)
    d_mis = L["misery"][last][0] - L["misery"][0][0]
    d_sky = L["skyhigh"][last][0] - L["skyhigh"][0][0]
    z_mis = abs(Z["misery"][max(Z["misery"])][0] - Z["misery"][0][0])
    print("learn: misery %+.3f skyhigh %+.3f (baseline sd %.3f); lesion: misery %+.3f" % (d_mis, d_sky, sd, z_mis))
    assert d_mis > sd, "misery avoidance did not rise"
    assert d_sky < -sd, "skyhigh avoidance did not fall"
    assert z_mis < sd, "lesion arm learned"
    print("ACCEPT")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required=True)
    ap.add_argument("--train", type=int, default=30)
    ap.add_argument("--probe", type=int, default=5)
    ap.add_argument("--arms", default="learn,lesion,reversed,frozen")
    ap.add_argument("--eta", type=float, default=PL.ETA)
    ap.add_argument("--lam", type=float, default=PL.LAM)
    ap.add_argument("--us-path", default="auto", choices=["auto", "grn", "dan"])
    ap.add_argument("--cs", default="song", choices=sorted(STIM_SETS),
                    help="song = JO drive + authored KC fingerprint (default, reproduces v1); "
                         "odour = one ORN channel at 80 Hz through the real AL")
    ap.add_argument("--regime", default="stock", choices=sorted(REGIMES),
                    help="eln8 = ELN_NEGATE + PN_KC_GAIN 8.0, the measured odour operating point. "
                         "Both are baked into the weights at GpuSim construction.")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--seed0", type=int, default=0, help="offsets every trial seed; v1 used 0")
    ap.add_argument("--dump", action="store_true", help="save probe rate vectors to results/rates_<state>/")
    ap.add_argument("--w-syn", type=float, default=None,
                    help="override gpu_sim.W_SYN (mV/synapse) before GpuSim(); default None = "
                         "untouched (0.275). W_SYN x min_syn grid H3 (PREREGISTER_wsyn_grid.md). "
                         "The brain (min_syn) comes from FLYCHESS_BRAIN.")
    a = ap.parse_args()
    out_json = os.path.join(RESULTS, "condition_%s.json" % a.state)
    if a.check:
        check(json.load(open(out_json))["curves"]); return
    probes = STIM_SETS[a.cs]
    arm_table = ODOUR_ARMS if a.cs == "odour" else ARMS
    # Set before construction: ELN_NEGATE and PN_KC_GAIN are baked into the device
    # weights by GpuSim.__init__ (gpu_sim asserts on a later mismatch), unlike
    # KC_V_TH_DELTA which run_batch reads every call. Stock stays a no-op so every
    # song result reproduces bit for bit.
    G.ELN_NEGATE, G.PN_KC_GAIN = REGIMES[a.regime]
    print("cs %s, regime %s (ELN_NEGATE=%s PN_KC_GAIN=%.1f), probes %s"
          % (a.cs, a.regime, G.ELN_NEGATE, G.PN_KC_GAIN, ",".join(probes)), flush=True)
    if a.w_syn is not None:
        G.W_SYN = a.w_syn   # baked into sim.data at construction; Plastic.push reads it too
    sim = G.GpuSim(); P0 = PL.Plastic.real()
    if os.environ.get("FLYCHESS_PERTURB_WSHA"):   # validation batch (PREREGISTER_validation.md): assert the loaded copy
        import hashlib
        _got = hashlib.sha256(np.ascontiguousarray(sim.net.W_data, np.float32).tobytes()).hexdigest()
        assert _got == os.environ["FLYCHESS_PERTURB_WSHA"], "PERTURBATION HASH MISMATCH %s" % _got
        print("perturbation hash OK %s" % _got[:12], flush=True)
    if a.w_syn is not None:
        assert G.W_SYN == a.w_syn, "W_SYN not applied"
    print("brain %s min_syn %d n_edges %d W_SYN %.4f"
          % (G.BRAIN, sim.net.min_syn, sim.net.n_edges, G.W_SYN), flush=True)
    us_path, us_detail = ((a.us_path, None) if a.us_path != "auto"
                          else check_us_path(sim, P0, cs=probes[0]))
    print("US path", us_path, us_detail, flush=True)
    os.makedirs(RESULTS, exist_ok=True); rows = []; t0 = time.time()
    dump = None
    if a.dump:
        dump = os.path.join(RESULTS, "rates_%s" % a.state); os.makedirs(dump, exist_ok=True)
    with open(os.path.join(RESULTS, "condition_%s.jsonl" % a.state), "w") as log:
        for name in a.arms.split(","):
            P, r = run_arm(sim, name, arm_table[name], a.train, a.probe, us_path, a.eta, a.lam, log,
                           seed0=a.seed0, dump=dump, probes=probes)
            rows += r
            if name == "learn":
                P.save(os.path.join(STATES, a.state))
                # flywatch/sim.py applies FLYWATCH_GAIN inside audio_drives, so the effective
                # CS gain is FLYWATCH_GAIN * AUDIO_GAIN; recording only the latter hides a rerun
                # under a different environment.
                json.dump({"song_pairs": arm_table[name]["pair"], "cs": a.cs,
                           "regime": a.regime, "probes": list(probes), "us_path": us_path, "eta": a.eta, "lam": a.lam, "seed0": a.seed0,
                           "n_train": a.train, "kc_hz": F.KC_HZ, "audio_gain": AUDIO_GAIN, "brain": os.path.relpath(G.BRAIN, _HERE).replace("\\", "/"),
                           "flywatch_gain": float(os.environ.get("FLYWATCH_GAIN", "1.0")),
                           # Only the clips that are actually present: the song CS needs
                           # flywatch/raw/*.wav, which is commercial audio and is not
                           # distributed. On --cs odour nothing here is used at all.
                           "song_sha256": {k: _sha256(os.path.join(_HERE, v[0]))
                                           for k, v in SONGS.items()
                                           if os.path.exists(os.path.join(_HERE, v[0]))},
                           "summary": P.summary()}, open(os.path.join(STATES, a.state, "provenance.json"), "w"), indent=1)
            print("arm %s done %.0fs" % (name, time.time() - t0), flush=True)
    cv = curves(rows)
    json.dump({"curves": cv, "probes": list(probes), "cs": a.cs, "regime": a.regime,
               "us_path": us_path, "us_detail": us_detail, "args": vars(a),
               "w_syn": float(G.W_SYN), "min_syn": int(sim.net.min_syn),
               "n_edges": int(sim.net.n_edges), "brain": os.path.relpath(G.BRAIN, _HERE),
               "w_sha": os.environ.get("FLYCHESS_PERTURB_WSHA")},
              open(out_json, "w"), indent=1)
    plot(cv, out_json.replace(".json", ".png"))
    print("wrote", out_json)


if __name__ == "__main__":
    main()
