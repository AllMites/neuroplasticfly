"""Chunk-2 bridge driver: paper 0's odour-learning protocols on the rate model (paper 1, PRD phase 1).

The protocol code is paper 0's, unchanged: learn/condition.run_arm (rung A) and learn/persist.run_arm
(rung 3) run on the frozen C0 base through rate/as_gpusim.RateAsGpuSim, so any bridge difference is the
neuron model, not a re-implementation. bidir_mb is the one new schedule (plan D1): loop/cycles.py's
F -> T -> B -> T cycles replayed open-loop with no body, 100 ms ticks, read at the MB (avoid index).
  F  cue DC2 on every tick, US every `every` ticks with the cue live       (protocol "paired")
  B  US every `every` ticks, cue withheld on the US tick and the one before (protocol "backward")
  T  weights frozen, fresh state, one C.trial per odour probe
Open loop has no arrivals, so F's US is periodic; same `every` in F and B makes the US count match.

Phase 1 is plumbing only: --smoke prints no learning number. Science runs need the phase-2 prereg.

Run: .venv/Scripts/python.exe tools/gpu_slot.py -- .venv/Scripts/python.exe rate/chunk2.py --smoke
"""
import argparse
import ast
import hashlib
import io
import json
import math
import os
import re
import subprocess
import sys
import time

import numpy as np
import torch

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _HERE)
from learn import condition as C  # noqa: E402
from learn import persist  # noqa: E402
from learn import plastic as PL  # noqa: E402
from rate import suite as S  # noqa: E402
from rate.as_gpusim import RateAsGpuSim  # noqa: E402

PREREG = os.path.join(_HERE, "PREREGISTER_rate_chunk2_bridge.md")
OUT = os.path.join(_HERE, "results", "rate_chunk2")
# loop/protocol.py US_HZ, ETA and loop/cycles.py:29 schedule (copied: protocol imports the body)
US_HZ, ETA_BIDIR, TICK_MS = 60.0, 5e-6, 100.0
CYCLES, F_TICKS, B_TICKS, EVERY = 4, 200, 200, 25
# prereg: the run parameters every science log must carry in its meta row args (analyze refuses any other value)
PREREG_PARAMS = {"train": 30, "relax": 30, "probe": 1, "eta": 5e-6, "lam": PL.LAM, "seed0": 0}


# Stability guard (user 2026-10-01; Li et al. 2026: MBON06 + KCab are essential for boundedness).
ACTIVE_HZ, ACTIVE_MAX = 1.0, 2.0          # whole-brain active = rate > 1 Hz; abort above 2x the reference
MBON06_BAND = (0.5, 2.0)                  # abort outside [0.5x, 2x] of the reference MBON06 mean rate
MBON06_MIN_REF = 0.5                      # reference below this: a ratio is meaningless -> absolute band
MBON06_ABS_BAND = (0.0, 1.0)              # Hz, used instead when the reference MBON06 < MBON06_MIN_REF


class GuardAbort(Exception):
    def __init__(self, info):
        super().__init__("FAIL-UNSTABLE %s" % (info,))
        self.info = info


def _mbon06_idx():
    ct = np.load(os.path.join(_HERE, "data", "neuron_meta.npz"), allow_pickle=True)["cell_type"].astype(str)
    idx = np.where(ct == "MBON06")[0]
    assert len(idx) == 2, "expected 2 MBON06 neurons, found %d" % len(idx)
    return idx


def _key(idx, p):
    return np.asarray(idx, np.int64).tobytes() + np.asarray(p, np.float64).tobytes()


class _GuardMixin:
    """The stability guard, engine-agnostic: mixed in front of RateAsGpuSim (rate arms) or gpu_sim.GpuSim (LIF arm).
    Call _guard_init(mbon06) after the engine's own __init__.

    Aborts an arm when learning destabilises the brain.

    arm_guard(arm, songs, log) resets the guard for one arm. The probe drives are those C.trial builds
    with us=None for `songs`. The first sighting of each (before any plasticity: asserted through
    set_plastic) is the reference; every later run of the same drive is compared with it, a guard row
    is logged, and a breach raises GuardAbort. Other drives (training with a US, relax) are not checked.
    The reference rows carry ratios of exactly 1.0. Drives are keyed on what run_batch receives
    (idx + per-step probability), which is what condition.to_prob makes from (idx, hz).
    """

    def _guard_init(self, mbon06=None):
        self.mbon06 = _mbon06_idx() if mbon06 is None else np.asarray(mbon06, np.int64)
        self.songs = None

    def arm_guard(self, arm, songs, log):
        self.arm, self.log, self.ref, self.w0, self.moved = arm, log, {}, None, False
        self.songs = {}
        for s in songs:
            (i1, h1), (i2, h2) = C.cs_drive(s), C.us_drive(None, "grn")
            self.songs[_key(*C.to_prob(np.concatenate([i1, i2]), np.concatenate([h1, h2])))] = s

    def set_plastic(self, offsets, values):
        super().set_plastic(offsets, values)
        if self.songs is not None:
            v = np.array(values, np.float32)
            if self.w0 is None:
                self.w0 = v
            elif not np.array_equal(v, self.w0):
                self.moved = True

    def run_batch(self, drives, seeds, t_run=300.0, state=None, return_state=False, **kw):
        out = super().run_batch(drives, seeds, t_run=t_run, state=state, return_state=return_state, **kw)
        if self.songs is None or state is not None or t_run != C.T_RUN:
            return out    # only C.trial-shaped runs (fresh state, T_RUN) are comparable with the reference
        counts = out[0] if return_state else out
        for b, (i, p) in enumerate(drives):
            key = _key(i, p)
            if key in self.songs:
                self._check(self.songs[key], counts[b].cpu().numpy().astype(np.float64) / (t_run / 1000.0), key)
        return out

    def _check(self, song, rates, key):
        m = {"active_frac": float((rates > ACTIVE_HZ).mean()), "mbon06_hz": float(rates[self.mbon06].mean())}
        if key not in self.ref:
            assert not self.moved, "first sighting of probe %s came after plasticity moved weights" % song
            self.ref[key] = m
        ref = self.ref[key]
        absolute = ref["mbon06_hz"] < MBON06_MIN_REF
        a_ratio = m["active_frac"] / ref["active_frac"] if ref["active_frac"] > 0 else (1.0 if m["active_frac"] == 0 else None)
        m_val = m["mbon06_hz"] if absolute else m["mbon06_hz"] / ref["mbon06_hz"]
        lo, hi = MBON06_ABS_BAND if absolute else MBON06_BAND
        why = []
        if a_ratio is None or a_ratio > ACTIVE_MAX:
            why.append("active_frac")
        if not lo <= m_val <= hi:
            why.append("mbon06")
        row = {"arm": self.arm, "phase": "guard", "song": song, "active_frac_ratio": a_ratio,
               "mbon06_ratio": None if absolute else m_val, "mbon06_mode": "absolute" if absolute else "ratio",
               "active_frac": m["active_frac"], "mbon06_hz": m["mbon06_hz"], "breach": bool(why)}
        self.log.write(json.dumps(row) + "\n")
        if why:
            raise GuardAbort(dict(row, breached=why, ref_active_frac=ref["active_frac"], ref_mbon06_hz=ref["mbon06_hz"]))


class GuardedSim(_GuardMixin, RateAsGpuSim):
    def __init__(self, rsim, mbon06=None):
        RateAsGpuSim.__init__(self, rsim)
        self._guard_init(mbon06)


def guarded(sim, arm, songs, log, fn):
    """Run fn() under the guard for one arm; on a breach write the FAIL-UNSTABLE row and return None."""
    if isinstance(sim, _GuardMixin):
        sim.arm_guard(arm, songs, log)
    try:
        return fn()
    except GuardAbort as e:
        log.write(json.dumps({**e.info, "phase": "abort", "label_hint": "FAIL-UNSTABLE"}) + "\n")
        return None


# Headroom gate + latch check (plan task 3). Baseline rates only: no training, no plasticity.
REF_PATH = os.path.join(OUT, "reference.json")
R_MAX_FRAC = 0.95                         # a readout MBON group mean at >= this x r_max is saturated
# protocol -> probe odours read at baseline. Bidir borrows rung A's paper-0 normalised effect (same cue, same MBON readout).
GATE_PROTOCOLS = {"rung_a": ("dc2",), "rung3": ("dc2", "d"), "bidir": ("dc2",)}
# prereg: LATCHED iff any paired-odour KC is still > 1 Hz after offset, or latched approach MBONs carried
# >= LATCH_APPROACH_FRAC of the approach-MBON rate during the odour trial. Whole-brain n_latched is info only
# (C0 latches ~740 odour-independent central-complex neurons, rate/regress/latch_ref.json).
LATCH_KC_MIN = 0                          # prereg: latched iff n_latched_kc_paired > LATCH_KC_MIN
LATCH_APPROACH_FRAC = 0.5                 # prereg: latched iff approach_latched_rate_frac >= this
LATCH_OFF_MS, LATCH_WIN_MS = 200.0, 100.0 # empty-drive window after the 300 ms odour trial; last 100 ms read
LATCH_END_ARMS = ("learn", "interfere")   # arms that get an end-of-arm (trained brain) latch check; bidir always


def load_reference(path=REF_PATH):
    with open(path) as f:
        return json.load(f)


def _ratio(x, y):
    return x / y if y else None


def headroom_row(protocol, song, r, ref, r_max, arm="headroom", basis=None):
    """Pure gate decision for one probe. r = C.readout-shaped dict (baseline). PASS iff the baseline approach MBON
    mean is > 0 AND neither MBON group mean is at >= 0.95 r_max (effects are normalised by that baseline)."""
    b = ref["rung_a"]["baseline"][song]
    r3 = protocol == "rung3"
    b_app = ref["rung3"]["baseline_approach_hz"][song]["mean"] if r3 else b["mbon_approach_hz"]["mean"]
    ceil = R_MAX_FRAC * r_max
    why = []
    if not r["mbon_approach_hz"] > 0:
        why.append("approach_zero")
    if r["mbon_approach_hz"] >= ceil:
        why.append("approach_at_r_max")
    if r["mbon_avoid_hz"] >= ceil:
        why.append("avoid_at_r_max")
    return {"arm": arm, "phase": "headroom", "protocol": protocol, "song": song,
            "mbon_approach_hz": r["mbon_approach_hz"], "mbon_avoid_hz": r["mbon_avoid_hz"], "kc_active": r["kc_active"],
            "ref_src_approach": "rung3.baseline_approach_hz" if r3 else "rung_a.baseline",
            "ref_src_avoid_kc": "rung_a.baseline",
            "approach_ratio": _ratio(r["mbon_approach_hz"], b_app),
            "avoid_ratio": _ratio(r["mbon_avoid_hz"], b["mbon_avoid_hz"]["mean"]),
            "kc_active_ratio": _ratio(r["kc_active"], b["kc_active"]["mean"]),
            "r_max_hz": r_max, "gate_pass": not why, "reason": ",".join(why) or None,
            **({} if basis is None else {"ratio_basis": basis})}


class _Unguarded:
    """Suspend the GuardedSim probe check: these runs are baseline-only and must not become guard references."""
    def __init__(self, sim):
        self.sim = sim

    def __enter__(self):
        self.songs = getattr(self.sim, "songs", None)
        if self.songs is not None:
            self.sim.songs = None

    def __exit__(self, *a):
        if self.songs is not None:
            self.sim.songs = self.songs


def sim_r_max(sim):
    """Output ceiling (Hz): the rate engine's r_max, or the LIF refractory ceiling 1000 / T_REFR."""
    return float(sim.r.r_max) if hasattr(sim, "r") else 1000.0 / C.G.T_REFR


def headroom(sim, ref, protocol, log, P=None):
    """Baseline probe readout on the protocol's paired odour(s); one `phase="headroom"` row each, logged.
    Returns the rows; overall pass = all(r["gate_pass"]). A FAIL is logged, not raised: the batch still runs."""
    P = PL.Plastic.real() if P is None else P
    rows = []
    with _Unguarded(sim):
        PL.Plastic.real().push(sim)      # baseline weights (identity push, golden 3) even after a learn arm
        for s in GATE_PROTOCOLS[protocol]:
            rows.append(headroom_row(protocol, s, C.readout(C.trial(sim, s), P), ref, sim_r_max(sim),
                                     basis=None if hasattr(sim, "r") else LIF_RATIO_BASIS))
    for r in rows:
        log.write(json.dumps(r) + "\n")
    return rows


def latch_row(protocol, song, when, arm, P, odour, post):
    """Pure latch summary from per-neuron rates (Hz): odour = during the odour trial, post = last
    LATCH_WIN_MS of the empty offset. P supplies the MBON / DAN index sets."""
    hot = post > S.LATCH_HZ
    appr = np.unique(P.mbon_of_edge[P.mbon_valence_of_edge == "approach"])
    avoid = np.unique(P.mbon_of_edge[P.mbon_valence_of_edge == "avoid"])
    ppl1, pam = P.dan_idx["punish"], P.dan_idx["reward"]
    kc_paired = (C.cc == "Kenyon_Cell") & (odour > S.LATCH_HZ)
    a_tot = float(odour[appr].sum())
    a_frac = float(odour[appr][hot[appr]].sum() / a_tot) if a_tot > 0 else 0.0
    n_kc = int((kc_paired & hot).sum())
    return {"arm": arm, "phase": "latch", "protocol": protocol, "song": song, "when": when,
            "n_latched": int(hot.sum()), "n_latched_kc_paired": n_kc, "n_kc_paired": int(kc_paired.sum()),
            "n_latched_mbon_approach": int(hot[appr].sum()), "n_latched_mbon_avoid": int(hot[avoid].sum()),
            "approach_latched_rate_frac": a_frac, "n_latched_ppl1": int(hot[ppl1].sum()), "n_latched_pam": int(hot[pam].sum()),
            "ppl1_hz_offset": float(post[ppl1].mean()), "pam_hz_offset": float(post[pam].mean()),
            "latched": bool(n_kc > LATCH_KC_MIN or a_frac >= LATCH_APPROACH_FRAC),
            "off_ms": LATCH_OFF_MS, "win_ms": LATCH_WIN_MS}


def latch_check(sim, log, protocol, when, arm, P=None):
    """Post-offset latch: odour trial (C.T_RUN ms, no US), then LATCH_OFF_MS of empty drive with state carried,
    rates read over the last LATCH_WIN_MS. Mirrors rate/suite.latch_check (on -> empty carry -> trailing
    window, S.LATCH_HZ = 1 Hz) with shorter windows. One row per probe odour of the protocol. Runs on whatever
    weights the sim holds (baseline or trained); unguarded."""
    P = PL.Plastic.real() if P is None else P
    empty = (np.zeros(0, np.int64), np.zeros(0, np.float64))
    rows = []
    with _Unguarded(sim):
        for s in GATE_PROTOCOLS[protocol]:
            (i1, h1), (i2, h2) = C.cs_drive(s), C.us_drive(None, "grn")
            drv = C.to_prob(np.concatenate([i1, i2]), np.concatenate([h1, h2]))
            c0, st = sim.run_batch([drv], [0], t_run=C.T_RUN, return_state=True)
            _, st = sim.run_batch([empty], [0], t_run=LATCH_OFF_MS - LATCH_WIN_MS, state=st, return_state=True)
            c1 = sim.run_batch([empty], [0], t_run=LATCH_WIN_MS, state=st)
            rows.append(latch_row(protocol, s, when, arm, P,
                                  c0[0].cpu().numpy().astype(np.float64) / (C.T_RUN / 1000.0),
                                  c1[0].cpu().numpy().astype(np.float64) / (LATCH_WIN_MS / 1000.0)))
    for r in rows:
        log.write(json.dumps(r) + "\n")
    return rows


def science_prologue(sim, protocol, log):
    """Once per protocol, on baseline weights, into the science log: headroom rows + baseline latch rows."""
    return headroom(sim, load_reference(), protocol, log), latch_check(sim, log, protocol, "baseline", "baseline")


BASE_NAME = "c0"
GAINS = None        # arm (ii) only: {"file", "sha256", "cap", "fit_seed", "n_gains", "n_active"}; None on c0
FIT_DIR = os.path.join(_HERE, "results", "rate_chunk1_fit_c0")
FIT_CAP = 16.0      # chunk-1 g* (result.json); the sigmoid maps edge_x to a gain only at the cap it was fitted with
FIT_SEEDS = (0, 1, 2, 3, 4)


def chunk_params_loaded():
    """Names of the registered suite chunks whose params S.build loads into any base (none today)."""
    return [c["name"] for c in S.CHUNKS if c["params"] is not None]


def fit_cell_path(seed, fit_dir=None):
    return os.path.join(FIT_DIR if fit_dir is None else fit_dir, "cells", "cap%07.3f_s%d.json" % (FIT_CAP, seed))


def load_fit_cell(seed, fit_dir=None):
    """(cell, path) of the chunk-1 fit at g*. Refuses (SystemExit) a bad seed or a seed that did not F-pass."""
    if seed not in FIT_SEEDS:
        raise SystemExit("--fit-seed must be one of %s (got %r)" % (list(FIT_SEEDS), seed))
    path = fit_cell_path(seed, fit_dir)
    with open(path) as f:
        cell = json.load(f)
    if cell["cap"] != FIT_CAP or cell["seed"] != seed:
        raise SystemExit("%s holds cap %r seed %r, not cap %g seed %d" % (path, cell["cap"], cell["seed"], FIT_CAP, seed))
    if not cell["fpass"]:
        raise SystemExit("chunk-1 seed %d did not pass F1-F5 at cap %g; arm (ii) refuses it" % (seed, FIT_CAP))
    return cell, path


def set_base(arm="c0", fit_seed=None, fit_dir=None, lif_seed=None):
    """Select the base for this process: name (drives log names), gains record (drives the meta row)."""
    global BASE_NAME, GAINS
    if arm == "lif":
        if lif_seed not in LIF_SEEDS or fit_seed is not None:
            raise SystemExit("--arm lif needs --lif-seed 0-4 and no --fit-seed")
        BASE_NAME, GAINS = "lifs%d" % lif_seed, None
        return BASE_NAME
    if lif_seed is not None:
        raise SystemExit("--lif-seed only applies to --arm lif")
    if arm == "c0":
        if fit_seed is not None:
            raise SystemExit("--fit-seed only applies to --arm c1")
        BASE_NAME, GAINS = "c0", None
    elif arm == "c1":
        if fit_seed is None:
            raise SystemExit("--arm c1 needs --fit-seed 0-4")
        load_fit_cell(fit_seed, fit_dir)
        BASE_NAME, GAINS = "c1s%d" % fit_seed, None    # GAINS is filled by base() once the gains are loaded
    else:
        raise SystemExit("unknown arm %r" % arm)
    return BASE_NAME


def base(arm=None, fit_seed=None, fit_dir=None):
    """The frozen C0 brain (arm c0) or C0 plus the chunk-1 reward-path edge gains at g* (arm c1, one fit seed).
    Gains are set before any plasticity (set_edge_gains resets edge_x to gain 1.0, so edge_x is assigned after)."""
    if arm is not None:
        set_base(arm, fit_seed, fit_dir)
    loaded = chunk_params_loaded()
    if loaded:    # a real raise, not an assert: the C0 base must stay the frozen C0 brain
        raise RuntimeError("S.build would load chunk params %s into the chunk-2 base %s" % (loaded, BASE_NAME))
    sim = GuardedSim(S.build(base="c0"))
    assert sim.r.regime == "eln8", "paper 0 ran eln8; the C0 base must too"
    if BASE_NAME != "c0":
        global GAINS
        from rate import chunk1 as C1    # noqa: E402  (heavy import kept off the c0 path)
        from rate import chunk1_fit as F1    # noqa: E402
        seed = int(BASE_NAME[3:])
        cell, path = load_fit_cell(seed, fit_dir)
        rs = sim.r
        tpools = S.owned_pools(rs.pool.cpu().numpy(), C1.sets()["ct"], F1.OWN)
        tab = rs.set_edge_gains(tpools, cell["cap"])
        assert len(cell["edge_x"]) == len(tab["n_edges"]), "chunk-1 edge_x does not match this group table"
        rs.edge_x = torch.as_tensor(cell["edge_x"], dtype=rs.dtype, device=rs.device)
        g = np.asarray(cell["gains"], np.float64)
        assert np.allclose(rs.edge_gain().detach().cpu().double().numpy(), g, rtol=1e-4, atol=1e-6), "reloaded gains differ from the cell's"
        GAINS = {"file": os.path.relpath(path, _HERE).replace(os.sep, "/"), "sha256": sha256_file(path), "cap": cell["cap"],
                 "fit_seed": seed, "n_gains": len(g), "n_active": int((np.abs(np.log(np.maximum(g, 1e-12))) > 0.05).sum())}
    return sim


def rung_a(sim, arms, eta, lam, log, n_train=30, n_probe=1, end_latch=None):
    pr = C.STIM_SETS["odour"]
    out = {}
    for name in arms:
        out[name] = guarded(sim, name, pr, log, lambda: C.run_arm(sim, name, C.ODOUR_ARMS[name], n_train, n_probe,
                                                                  "dan", eta, lam, log, probes=pr))
        if out[name] is not None and name in (end_latch or ()):
            latch_check(sim, log, "rung_a", "end", name)
    return out


def rung3(sim, arms, a, log, end_latch=None):
    out = {}
    for name in arms:
        out[name] = guarded(sim, name, persist.PROBES, log, lambda: persist.run_arm(sim, name, persist.ARMS[name], a, log))
        if out[name] is not None and name in (end_latch or ()):
            latch_check(sim, log, "rung3", "end", name)
    return out


def _block(sim, P, n, mode, rule, eta, every, lesion, tick_seed=None):
    """One F ("paired") or B ("backward") block of n ticks, state carried. Returns US count.
    tick_seed(t) -> counter-RNG seed of tick t (LIF arm); None = seed 0 every tick (rate engine ignores seeds)."""
    assert mode in ("paired", "backward") and every >= 3, (mode, every)
    P.reset_lag()
    cue = C.odour_drive("dc2")
    us = (P.dan_idx["punish"], np.full(len(P.dan_idx["punish"]), US_HZ, np.float32))
    st, n_us = None, 0
    for t in range(n):
        us_tick = bool(t) and t % every == 0
        cue_on = not (mode == "backward" and (us_tick or (t + 1) % every == 0))
        parts = ([cue] if cue_on else []) + ([us] if us_tick else [])
        idx = np.concatenate([p[0] for p in parts]) if parts else np.zeros(0, np.int64)
        hz = np.concatenate([p[1] for p in parts]) if parts else np.zeros(0, np.float32)
        counts, st = sim.run_batch([C.to_prob(idx, hz)], [0 if tick_seed is None else tick_seed(t)], t_run=TICK_MS, state=st, return_state=True)
        rates = counts[0].cpu().numpy().astype(np.float32) / (TICK_MS / 1000.0)
        n_us += us_tick
        if rule == "depress" and us_tick:
            P.update(rates, eta=eta, lesion=lesion)
            P.push(sim)
        elif rule == "timed":
            P.update_timed(rates, eta, lam=PL.LAM if us_tick else 0.0, lesion=lesion)
            P.push(sim)
    return n_us


def bidir_mb(sim, rule, log, eta=ETA_BIDIR, cycles=CYCLES, f_ticks=F_TICKS, b_ticks=B_TICKS,
             every=EVERY, lesion="none", lif_seed=None):
    """lif_seed (0-4): LIF arm; every 100 ms tick gets its own counter seed (lif_tick_seed). Probes keep C.trial's seed 0."""
    assert rule in ("depress", "timed"), rule
    P = PL.Plastic.real()
    P.push(sim)
    rows = []

    def test(tag, n_us):
        for s in C.STIM_SETS["odour"]:
            r = C.readout(C.trial(sim, s), P)
            r.update(rule=rule, after=tag, song=s, n_us=n_us, **{"w_" + a: b for a, b in P.summary().items()})
            rows.append(r)
            log.write(json.dumps(r) + "\n")

    def seeds(block):
        return None if lif_seed is None else (lambda t: lif_tick_seed(lif_seed, block, t))

    test("T0", None)
    for k in range(1, cycles + 1):
        test("F%d" % k, _block(sim, P, f_ticks, "paired", rule, eta, every, lesion, seeds(2 * k - 2)))
        test("B%d" % k, _block(sim, P, b_ticks, "backward", rule, eta, every, lesion, seeds(2 * k - 1)))
    return P, rows


# LIF reference arm (PREREGISTER_rate_chunk2_bidir_ref.md): paper 0's GpuSim, eln8, floor 5, default W_SYN.
PREREG_REF = os.path.join(_HERE, "PREREGISTER_rate_chunk2_bidir_ref.md")
OUT_REF = os.path.join(_HERE, "results", "rate_chunk2_bidir_ref")
LIF_SEEDS = (0, 1, 2, 3, 4)
LIF_MIN_SYN, LIF_REGIME = 5, "eln8"
LIF_TICK_SEED_FORMULA = "1000000*(S+1) + 1000*(block+1) + tick   (block 0..7 = F1,B1,..,F4,B4; tick 0..199; probes seed 0)"


def lif_tick_seed(s, block, tick):
    """Counter-RNG seed of tick `tick` of block `block` in LIF seed s. Distinct over (s, block, tick), never 0
    (0 is every T probe's seed), below 2**31. Formula: LIF_TICK_SEED_FORMULA."""
    assert s in LIF_SEEDS and 0 <= block < 999 and 0 <= tick < 1000, (s, block, tick)
    return 1000000 * (s + 1) + 1000 * (block + 1) + tick


class GuardedGpuSim(_GuardMixin, C.G.GpuSim):
    def __init__(self, mbon06=None):
        C.G.GpuSim.__init__(self)
        self._guard_init(mbon06)


LIF_RATIO_BASIS = "vs paper-0 LIF seed-mean baseline"    # headroom ratios of the LIF arm divide by reference.json (paper 0's LIF)
LIF_BRAIN = "data/brain_gpu.npz"
# Module knobs that must sit at the values gpu_sim.py itself defines (read from its source, not typed here).
LIF_DEFAULT_KNOBS = ("KC_V_TH_DELTA", "APL_GRADED", "APL_SCALE", "APL_MAX_HZ", "APL_DELAYED", "APL_DIVISIVE", "APL_DIV_SCALE",
                     "NORM_TOTAL_TARGET", "GAP_COUPLE", "GAP_NORM", "TYPE_W_SCALE", "SLOW_FRAC", "SLOW_TYPES", "SFA_B_INC")


def gpu_sim_source_defaults(names=LIF_DEFAULT_KNOBS):
    """{name: literal} of the module-level assignments in gpu_sim.py's source (ast), so a runtime edit of the module
    cannot move the reference."""
    with open(os.path.join(_HERE, "gpu_sim.py"), encoding="utf-8") as f:
        tree = ast.parse(f.read())
    out = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id in names:
            out[node.targets[0].id] = ast.literal_eval(node.value)
    missing = [n for n in names if n not in out]
    assert not missing, "gpu_sim.py defines no literal default for %s" % missing
    return out


def lif_brain_path():
    """The brain file gpu_sim resolved (FLYCHESS_BRAIN or its default), relative to the repo with forward slashes."""
    return os.path.relpath(os.path.abspath(C.G.BRAIN), _HERE).replace(os.sep, "/")


def base_lif():
    """Paper 0's model exactly (learn/condition.py main): ELN_NEGATE + PN_KC_GAIN set BEFORE construction, default W_SYN,
    every other gpu_sim knob at its source default, the frozen brain file."""
    G = C.G
    G.ELN_NEGATE, G.PN_KC_GAIN = C.REGIMES[LIF_REGIME]
    off = {n: getattr(G, n) for n, v in gpu_sim_source_defaults().items() if getattr(G, n) != v}
    assert not off, "gpu_sim knobs differ from their source defaults: %s" % off
    assert lif_brain_path() == LIF_BRAIN, "LIF reference needs %s, gpu_sim resolved %s (FLYCHESS_BRAIN?)" % (LIF_BRAIN, lif_brain_path())
    sim = GuardedGpuSim()
    assert sim.net.min_syn == LIF_MIN_SYN, "LIF reference needs synapse floor %d, brain has %d" % (LIF_MIN_SYN, sim.net.min_syn)
    assert sim.pn_kc_gain == 8.0 and G.ELN_NEGATE and G.W_SYN == 0.275, "not paper 0's eln8 regime / default W_SYN"
    return sim


def lif_meta():
    G = C.G
    return {"regime": LIF_REGIME, "eln_negate": bool(G.ELN_NEGATE), "pn_kc_gain": float(G.PN_KC_GAIN), "min_syn": LIF_MIN_SYN,
            "w_syn": float(G.W_SYN), "brain": lif_brain_path(), "tick_seed_formula": LIF_TICK_SEED_FORMULA, "probe_seed": 0}


# Provenance (final review F1-F3). Every science log: meta row first, done row last; analyze binds to both.
def sha256_file(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def require_prereg(path=None):
    """SystemExit unless the prereg file exists and is non-empty (an explicit raise: survives python -O)."""
    path = PREREG if path is None else path
    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        raise SystemExit("chunk-2 science runs need a non-empty %s (phase 2 prereg); not found or empty" % path)
    return path


def git_state():
    """(short HEAD or "unknown", dirty flags for rate/ and learn/ (None if git failed))."""
    def run(*args):
        return subprocess.run(("git",) + args, cwd=_HERE, capture_output=True, text=True, timeout=30, check=True).stdout
    try:
        head = run("rev-parse", "HEAD").strip() or "unknown"
    except Exception:
        head = "unknown"
    dirty = {}
    for d in ("rate", "learn"):
        try:
            # ponytail: gpu_sim.py (eln8 edits, imported by condition) rides on the "rate" flag
            paths = (d, "gpu_sim.py") if d == "rate" else (d,)
            dirty[d] = bool(run("status", "--porcelain", "--", *paths).strip())
        except Exception:
            dirty[d] = None
    return head, dirty


# Source files whose text is the rule and the science path; hashed into every meta row, re-checked by analyze (final review I1).
CODE_FILES = ("rate/chunk2.py", "rate/engine.py", "learn/plastic.py", "learn/condition.py", "learn/persist.py",
              "gpu_sim.py", "rate/suite.py", "rate/as_gpusim.py", "rate/chunk1.py", "rate/chunk1_fit.py",
              "rate/regress/c0_base.json")
# Connectome inputs (gitignored): recorded in every meta row and reported by analyze, not refused on.
DATA_FILES = ("data/brain_gpu.npz", "data/neuron_meta.npz")


def code_hashes():
    """{relative path: sha256 of the file on disk, CRLF normalised to LF} for CODE_FILES (core.autocrlf is on here,
    so a checkout can flip line endings; normalising keeps the hash reproducible from a clean clone)."""
    def lf_sha(f):
        with open(os.path.join(_HERE, *f.split("/")), "rb") as fh:
            return hashlib.sha256(fh.read().replace(bytes([13, 10]), bytes([10]))).hexdigest()
    return {f: lf_sha(f) for f in CODE_FILES}


def meta_row(args, overwrite=False, prereg=None, ref_path=None, extra=None):
    head, dirty = git_state()
    row = {"phase": "meta", "prereg_sha256": sha256_file(require_prereg(prereg)),
            "reference_sha256": sha256_file(REF_PATH if ref_path is None else ref_path),
            "git_head": head, "git_dirty_rate": dirty["rate"], "git_dirty_learn": dirty["learn"], "code_sha256": code_hashes(),
            "data_sha256": {f: sha256_file(os.path.join(_HERE, *f.split("/"))) for f in DATA_FILES},
            "args": args, "base": BASE_NAME, "gains": GAINS, "chunk_params_loaded": chunk_params_loaded(),
            "overwrite": bool(overwrite)}
    if extra:    # LIF arm only: rate meta rows stay byte-identical
        row["lif"] = extra
    return row


def log_name(mode, rule):
    return "%s_%s.jsonl" % (mode if mode != "bidir" else "bidir_" + rule, BASE_NAME)


def open_log(path, overwrite=False):
    """Open a science log for writing; an existing file is never replaced unless overwrite."""
    if os.path.exists(path) and not overwrite:
        raise SystemExit("%s exists; refusing to overwrite a log (pass --overwrite to replace it)" % path)
    return open(path, "w")


def done_row(mode, t0):
    return {"phase": "done", "mode": mode, "seconds": round(time.time() - t0, 1)}


# --analyze labeller (plan task 4; two arms task 3). Pure over jsonl rows; every constant below is part of the rule the prereg quotes.
PASS_FRAC = 0.5          # prereg: rung A / rung 3 PASS needs >= PASS_FRAC x the paper-0 NORMALISED approach-component effect (approach-MBON rate drop / baseline approach-MBON rate) / ratio
NO_DW_FRAC = 0.1         # prereg: NO-WEIGHT-CHANGE iff |w_mean_frac| < NO_DW_FRAC x |paper-0 value| (paper-0 value is negative)
BIDIR_RELEARN_FRAC = 0.5 # prereg: bidir cycle k relearns iff F_k rise >= this x the F_1 rise ...
BIDIR_MIN_RELEARN = 2    # prereg: ... and PASS iff at least this many of the scorable cycles k = 2..CYCLES relearn
BIDIR_F1_MIN_FRAC = 0.5  # prereg: bidir has no MB-level paper-0 number; it borrows rung A's normalised DC2 approach-component effect: the normalised F_1 approach drop must be >= this x that
RUNG3_NAIVE_MIN_FRAC = 0.5  # prereg (controller ruling, prereg confirms): the normalised approach-component drop of naive D (rung A reversed) must be >= this x the paper-0 normalised naive D drop
ARM_PASS_FRAC = 0.8      # prereg: an arm (ii) protocol label is PASS iff >= ceil(ARM_PASS_FRAC x n) of the n fitted seeds PASS it (4 of 5)
BIDIR_SONG = "dc2"       # prereg: the bidir cue / readout odour
DIAG_ORDER = ("READOUT-CEILING", "FAIL-UNSTABLE", "NO-WEIGHT-CHANGE", "LATCH-CONFOUNDED", "NEEDS-SPIKING")
RESULT_JSON, RESULT_MD = "result.json", "RESULT.md"
PROTOCOLS = ("rung_a", "rung3", "bidir")
# protocol -> [(log, arms whose abort row voids the verdict)]; rung 3's denominator is rung A's reversed arm
NEEDED = {"rung_a": [("rung_a", ("learn", "lesion", "shuffle"))],
          "rung3": [("rung3", ("interfere",)), ("rung_a", ("reversed",))],
          "bidir": [("bidir_timed", ("bidir_timed",)), ("bidir_depress", ("bidir_depress",))]}
LOGS = ("rung_a", "rung3", "bidir_timed", "bidir_depress")
# one base per run set: arm (i) = c0, arm (ii) = c1s0..c1s4 (analyze needs all of them, LOGS each: 24 runs)
BASES = ("c0",) + tuple("c1s%d" % s for s in FIT_SEEDS)
# log -> (meta args mode, meta args rule or None)
LOG_ARGS = {"rung_a": ("rung_a", None), "rung3": ("rung3", None), "bidir_timed": ("bidir", "timed"), "bidir_depress": ("bidir", "depress")}


def arm_need(n):
    """Seeds that must PASS for an arm (ii) label: ceil(ARM_PASS_FRAC x n); round() guards float noise (0.8 x 5)."""
    return math.ceil(round(ARM_PASS_FRAC * n, 9))


def rule_text():
    n = len(FIT_SEEDS)
    return "\n".join([
        "RULE (fixed in code before the batch; thresholds are fractions of reference.json, paper 0):",
        "Approach component: every gate uses the approach-MBON mean rate, because plasticity acts only on approach synapses in both models "
        "(the rate model's avoid baseline is 10-25x paper 0's, so a total avoid_index shift can be moved by avoid MBONs). approach drop = "
        "baseline (trial-0, baseline weights) approach-MBON mean minus the approach-MBON mean at the test; positive = learning. "
        "Normalised drop = approach drop / the baseline approach-MBON mean of the same arm and odour. The paper-0 normalised drop is its "
        "mean approach drop divided by its mean baseline approach (reference.json rung_a.learn_dc2_approach_drop_norm, naive_d_approach_drop_norm, "
        "rung3.approach_ratio_of_means). The total avoid_index shift and the avoid-MBON component are reported, never gated.",
        "Headroom gate (baseline weights, each probe odour): PASS iff the baseline approach-MBON mean is > 0 and neither MBON group "
        "mean is >= %g x r_max. No absolute rate threshold." % R_MAX_FRAC,
        "rung A: approach drop = learn-arm DC2 approach at trial 0 minus at the test trial (reference test_trial). "
        "PASS iff drop > 0 and normalised drop >= %g x the paper-0 normalised DC2 approach drop and the learn normalised drop exceeds both the lesion "
        "and the shuffle arm normalised DC2 drops. D and DA1 are reported, not gated." % PASS_FRAC,
        "rung 3: ratio = D approach drop of the interfere arm (relax_test end minus trainC_test end) / D approach drop of the rung A reversed "
        "arm (D paired with punishment on a naive brain; trial 0 minus test trial). PASS iff both drops are > 0 (same direction as paper 0; two negative drops fail), "
        "the rate naive D normalised approach drop >= %g x the paper-0 naive D normalised approach drop (a near-zero denominator cannot inflate the ratio) and "
        "ratio >= %g x the paper-0 approach-drop ratio." % (RUNG3_NAIVE_MIN_FRAC, PASS_FRAC),
        "bidir (timed rule scored against the depress rule; both must be in the batch): learning level = minus the DC2 approach-MBON mean; "
        "rise_k = level at F_k minus at B_(k-1) (T0 for k = 1) (an approach drop); drop_j = level at F_j minus at B_j. Cycle k = 2..%d relearns iff rise_k >= %g x rise_1 and "
        "timed drop_(k-1) > 0 and timed drop_(k-1) > depress drop_(k-1). PASS iff rise_1 / baseline approach DC2 (timed T0) >= %g x the rung A "
        "paper-0 normalised DC2 approach drop (borrowed, no MB-level bidir reference exists) and at least %d cycles relearn."
        % (CYCLES, BIDIR_RELEARN_FRAC, BIDIR_F1_MIN_FRAC, BIDIR_MIN_RELEARN),
        "Arms: arm (i) = base c0 (the frozen C0 brain); its protocol labels are its verdicts. Arm (ii) = bases c1s0..c1s%d (C0 plus the chunk-1 "
        "reward-path gains at cap %g, one base per fitted seed). Each seed is labelled per protocol by the rules above on its own baseline. "
        "An arm (ii) protocol label is PASS iff at least ceil(%g x n) = %d of the n = %d seeds PASS that protocol, else FAIL with the per-seed "
        "labels and diagnostics listed. The batch is 4 runs x %d bases = %d runs; every run needs its done row." % (
            FIT_SEEDS[-1], FIT_CAP, ARM_PASS_FRAC, arm_need(n), n, len(BASES), 4 * len(BASES)),
        "Diagnostic for a failing protocol (or failing seed), first match wins: " + " > ".join(DIAG_ORDER) + ". "
        "READOUT-CEILING = any headroom row of the protocol failed its gate (a saturated or a zero baseline); FAIL-UNSTABLE = a guard abort row in an arm the "
        "verdict needs, even when its scoring rows exist (an abort voids a PASS); NO-WEIGHT-CHANGE = |w_mean_frac| < %g x |paper-0 value| (rung A learn arm at the test trial; rung 3 "
        "interfere arm end of trainC; bidir timed rule, largest |w_mean_frac| over its test rows, rung A value borrowed); "
        "LATCH-CONFOUNDED = a BASELINE latch row of the protocol has latched True (end rows are reported, never gated); "
        "NEEDS-SPIKING = none of the above. A failing protocol with missing rows and neither an abort row nor a failed "
        "headroom row to explain them is INCOMPLETE-DATA (a pipeline fault, not a finding)." % NO_DW_FRAC])


def _mean(v):
    return float(np.mean(v)) if len(v) else None


def _field(rows, key, song, arm=None, **kw):
    """Mean of `key` over the readout rows matching arm/song and every key=value in kw (None if no row)."""
    return _mean([r[key] for r in rows if r.get("song") == song and (arm is None or r.get("arm") == arm)
                  and all(r.get(k) == v for k, v in kw.items()) and key in r])


def _avoid(rows, song, arm=None, **kw):
    return _field(rows, "avoid_index", song, arm, **kw)


def _sub(a, b):
    return None if a is None or b is None else a - b


def _norm(shift, baseline_approach):
    """Effect / baseline approach-MBON rate; None when either is missing or the baseline is not > 0."""
    return None if shift is None or baseline_approach is None or not baseline_approach > 0 else shift / baseline_approach


def ref_norm_effect(ref):
    """Paper-0 rung A normalised APPROACH drop on DC2: mean (A0 - A30) / mean A0 over the 5 seeds (learn arm)."""
    return ref["rung_a"]["learn_dc2_approach_drop_norm"]


def ref_norm_naive(ref):
    """Paper-0 naive D normalised approach drop: rung A reversed arm, mean (A0 - A30) / mean A0 on D."""
    return ref["rung_a"]["naive_d_approach_drop_norm"]


def _last_trial(rows, arm, phase):
    t = [r["trial"] for r in rows if r.get("arm") == arm and r.get("phase") == phase]
    return max(t) if t else None


def _w_frac(rows, arm, phase, trial):
    v = [r["w_mean_frac"] for r in rows if r.get("arm") == arm and r.get("phase") == phase and r.get("trial") == trial and "w_mean_frac" in r]
    return v[-1] if v else None


def _shift(rows, arm, song, T):
    """Total avoid_index shift (reported, not gated)."""
    return _sub(_avoid(rows, song, arm, phase="test", trial=T), _avoid(rows, song, arm, phase="baseline", trial=0))


def _app(rows, arm, song, phase, trial, key="mbon_approach_hz"):
    return _field(rows, key, song, arm, phase=phase, trial=trial)


def _app_drop(rows, arm, song, T):
    """Approach-MBON drop: baseline (trial 0) minus test trial T. Positive = learning (the plasticity acts on approach synapses)."""
    return _sub(_app(rows, arm, song, "baseline", 0), _app(rows, arm, song, "test", T))


def _avoid_comp(rows, arm, song, T):
    """Avoid-MBON component shift (test minus baseline; reported, not gated)."""
    return _sub(_app(rows, arm, song, "test", T, "mbon_avoid_hz"), _app(rows, arm, song, "baseline", 0, "mbon_avoid_hz"))


def score_rung_a(rows, ref):
    T = ref["rung_a"]["test_trial"]
    sh = {a: {s: _shift(rows, a, s, T) for s in C.STIM_SETS["odour"]} for a in C.ODOUR_ARMS}
    dr = {a: {s: _app_drop(rows, a, s, T) for s in C.STIM_SETS["odour"]} for a in C.ODOUR_ARMS}
    a0 = {a: _app(rows, a, "dc2", "baseline", 0) for a in ("learn", "lesion", "shuffle")}
    N = {a: _norm(dr[a]["dc2"], a0[a]) for a in a0}
    thr = PASS_FRAC * ref_norm_effect(ref)
    passed = None if None in N.values() else bool(dr["learn"]["dc2"] > 0 and N["learn"] >= thr and N["learn"] > N["lesion"] and N["learn"] > N["shuffle"])
    return passed, {"test_trial": T, "baseline_approach_hz": a0["learn"], "approach_drop_hz": dr["learn"]["dc2"],
                    "norm_approach_drop": N["learn"], "threshold_norm": thr,
                    "lesion_norm_approach_drop": N["lesion"], "shuffle_norm_approach_drop": N["shuffle"],
                    "d_approach_drop_hz": dr["learn"]["d"], "reversed_d_approach_drop_hz": dr["reversed"]["d"],
                    "dc2_avoid_index_shift": sh["learn"]["dc2"], "dc2_avoid_component_shift_hz": _avoid_comp(rows, "learn", "dc2", T),
                    "lesion_dc2_avoid_index_shift": sh["lesion"]["dc2"], "shuffle_dc2_avoid_index_shift": sh["shuffle"]["dc2"],
                    "d_avoid_index_shift": sh["learn"]["d"], "da1_avoid_index_shift": sh["learn"]["da1"],
                    "reversed_d_avoid_index_shift": sh["reversed"]["d"], "shifts": sh,
                    "w_mean_frac": _w_frac(rows, "learn", "train", T),
                    "w_mean_frac_ref": ref["rung_a"]["learn_w_mean_frac_trial30"]["mean"]}


def score_rung3(rows3, rows_a, ref):
    T = ref["rung_a"]["test_trial"]
    tc, tr = _last_trial(rows3, "interfere", "trainC_test"), _last_trial(rows3, "interfere", "relax_test")
    ok = tc is not None and tr is not None
    d_tr = None if not ok else _sub(_app(rows3, "interfere", "d", "relax_test", tr), _app(rows3, "interfere", "d", "trainC_test", tc))
    d_ai = None if not ok else _sub(_avoid(rows3, "d", "interfere", phase="trainC_test", trial=tc), _avoid(rows3, "d", "interfere", phase="relax_test", trial=tr))
    d_av = None if not ok else _sub(_app(rows3, "interfere", "d", "trainC_test", tc, "mbon_avoid_hz"), _app(rows3, "interfere", "d", "relax_test", tr, "mbon_avoid_hz"))
    naive = _app_drop(rows_a, "reversed", "d", T)
    app_d = _app(rows_a, "reversed", "d", "baseline", 0)
    naive_norm, nmin = _norm(naive, app_d), RUNG3_NAIVE_MIN_FRAC * ref_norm_naive(ref)
    ratio = None if d_tr is None or naive is None else _ratio(d_tr, naive)
    ref_ratio = ref["rung3"]["approach_ratio_of_means"]
    passed = None if d_tr is None or naive is None or naive_norm is None else bool(
        d_tr > 0 and naive > 0 and naive_norm >= nmin and ratio is not None and ratio >= PASS_FRAC * ref_ratio)
    tw = _last_trial(rows3, "interfere", "trainC")
    return passed, {"trained_d_approach_drop_hz": d_tr, "naive_d_approach_drop_hz": naive, "baseline_approach_d_hz": app_d,
                    "naive_norm_approach_drop": naive_norm, "ratio": ratio, "ref_ratio": ref_ratio,
                    "threshold_ratio": PASS_FRAC * ref_ratio, "naive_min_norm": nmin,
                    "trained_d_avoid_index_shift": d_ai, "trained_d_avoid_component_shift_hz": d_av,
                    "naive_d_avoid_index_shift": _shift(rows_a, "reversed", "d", T),
                    "naive_d_avoid_component_shift_hz": _avoid_comp(rows_a, "reversed", "d", T),
                    "w_mean_frac": None if tw is None else _w_frac(rows3, "interfere", "trainC", tw),
                    "w_mean_frac_ref": ref["rung3"]["w_mean_frac"]["end_trainC"]["mean"]}


def _bidir_curve(rows, key="mbon_approach_hz", sign=-1.0):
    """{tag: sign x DC2 mean of `key`} for T0, F1, B1, ...; default = minus the approach rate, so a rise = learning."""
    return {t: (lambda v: None if v is None else sign * v)(_field(rows, key, BIDIR_SONG, after=t))
            for t in {r.get("after") for r in rows if r.get("after")}}


def score_bidir(timed, depress, ref):
    thr = BIDIR_F1_MIN_FRAC * ref_norm_effect(ref)
    app = _field(timed, "mbon_approach_hz", BIDIR_SONG, after="T0")
    ai, av = _bidir_curve(timed, "avoid_index", 1.0), _bidir_curve(timed, "mbon_avoid_hz", 1.0)
    wf = [r["w_mean_frac"] for r in timed if "w_mean_frac" in r]
    base = {"f1_min_rise_norm": thr, "baseline_approach_hz": app, "w_mean_frac": max(wf, key=abs) if wf else None,
            "w_mean_frac_ref": ref["rung_a"]["learn_w_mean_frac_trial30"]["mean"], "cycles": {},
            "rise_1_avoid_index": _sub(ai.get("F1"), ai.get("T0")), "rise_1_avoid_component_hz": _sub(av.get("F1"), av.get("T0"))}
    nums = bidir_numbers(timed, depress)    # the one relearn computation, shared with the bidir-ref re-score
    if not nums["complete"]:
        return None, dict(base, missing=nums["missing"])
    n1 = nums["rise_1_norm"]
    base.update(cycles=nums["cycles"], rise_1=nums["rise_1"], rise_1_norm=n1, n_relearn=nums["n_relearn"], scorable=CYCLES - 1)
    return (None if n1 is None else bool(n1 >= thr and nums["n_relearn"] >= BIDIR_MIN_RELEARN)), base


def _diagnose(proto, passed, num, logs):
    """(label, diagnostic). Diagnostic only for a failing protocol; first match wins (DIAG_ORDER)."""
    aborts = [r for lg, arms in NEEDED[proto] for r in logs[lg] if r.get("phase") == "abort" and r.get("arm") in arms]
    if passed and not aborts:
        return "PASS", None
    mine = logs["bidir_timed"] + logs["bidir_depress"] if proto == "bidir" else logs[proto]
    heads = [r for r in mine if r.get("phase") == "headroom" and r.get("protocol") == proto]
    lat = [r for r in mine if r.get("phase") == "latch" and r.get("protocol") == proto and r.get("when") == "baseline"]
    ceiling = any(not h["gate_pass"] for h in heads)
    if passed is None and not aborts and not ceiling:
        return "FAIL", "INCOMPLETE-DATA"
    w, wref = num.get("w_mean_frac"), num.get("w_mean_frac_ref")
    hit = {"READOUT-CEILING": ceiling, "FAIL-UNSTABLE": bool(aborts),
           "NO-WEIGHT-CHANGE": w is not None and abs(w) < NO_DW_FRAC * abs(wref),
           "LATCH-CONFOUNDED": any(r["latched"] for r in lat), "NEEDS-SPIKING": True}
    return "FAIL", next(d for d in DIAG_ORDER if hit[d])


def label_protocols(logs, ref):
    """logs = {name: rows} for LOGS (missing = []). Returns {protocol: {label, diagnostic, passed, numbers}}."""
    logs = {k: list(logs.get(k, [])) for k in LOGS}
    scored = {"rung_a": score_rung_a(logs["rung_a"], ref), "rung3": score_rung3(logs["rung3"], logs["rung_a"], ref),
              "bidir": score_bidir(logs["bidir_timed"], logs["bidir_depress"], ref)}
    out = {}
    for p, (passed, num) in scored.items():
        label, diag = _diagnose(p, passed, num, logs)
        out[p] = {"label": label, "diagnostic": diag, "passed": None if passed is None else label == "PASS", "numbers": num}
    return out


def arm_labels(per_seed):
    """per_seed = {seed: label_protocols result}. Arm (ii) label per protocol: PASS iff >= arm_need(n) seeds PASS."""
    n = len(per_seed)
    need = arm_need(n)
    out = {}
    for p in PROTOCOLS:
        passing = [s for s in sorted(per_seed) if per_seed[s][p]["label"] == "PASS"]
        out[p] = {"label": "PASS" if len(passing) >= need else "FAIL", "n_pass": len(passing), "n_seeds": n, "need": need,
                  "passing_seeds": passing,
                  "per_seed": {s: {"label": per_seed[s][p]["label"], "diagnostic": per_seed[s][p]["diagnostic"]} for s in sorted(per_seed)}}
    return out


def summaries(logs):
    """End-of-arm latch rows (reported, never gated) and per-arm guard summary."""
    rows = [(n, r) for n, rs in logs.items() for r in rs]
    end = [dict(r, log=n) for n, r in rows if r.get("phase") == "latch" and r.get("when") == "end"]
    guard = {}
    new = lambda: {"n": 0, "max_active_ratio": None, "min_mbon06_ratio": None, "max_mbon06_ratio": None,  # noqa: E731
                   "breaches": 0, "aborted": False}
    for n, r in rows:
        if r.get("phase") == "guard":
            g = guard.setdefault("%s/%s" % (n, r["arm"]), new())
            g["n"] += 1
            g["breaches"] += bool(r["breach"])
            for k, v, f in (("max_active_ratio", r["active_frac_ratio"], max), ("min_mbon06_ratio", r["mbon06_ratio"], min),
                            ("max_mbon06_ratio", r["mbon06_ratio"], max)):
                if v is not None:
                    g[k] = v if g[k] is None else f(g[k], v)
        elif r.get("phase") == "abort":
            guard.setdefault("%s/%s" % (n, r["arm"]), new())["aborted"] = True
    return end, guard


def read_logs(out_dir, base_name):
    """{log: rows} of one base (missing file = [])."""
    logs = {}
    for n in LOGS:
        p = os.path.join(out_dir, "%s_%s.jsonl" % (n, base_name))
        if os.path.exists(p):
            with open(p) as f:
                logs[n] = [json.loads(x) for x in f if x.strip()]
        else:
            logs[n] = []
    return logs


def _arms_ok(log, args):
    """True iff the recorded --arms is the default (empty) or exactly the protocol's default arm set. Bidir has no arm list."""
    if log not in ("rung_a", "rung3"):
        return True
    default = set(C.ODOUR_ARMS if log == "rung_a" else persist.ARMS)
    given = args.get("arms")
    return given == "" or (isinstance(given, str) and sorted(given.split(",")) == sorted(default))


def check_provenance(all_logs, prereg=None, fit_dir=None, ref_path=None, state=None, code_sha=None):
    """Refuse (SystemExit) unless every run of every base carries a done row (names the missing runs only: no label, no
    number is computed here); then require one meta row per run whose base matches its file, identical prereg / reference
    hashes, a prereg hash equal to the file on disk, the prereg run parameters and default arm sets, and (arm ii) one
    gains sha256 per seed equal to the chunk-1 cell on disk, one known git_head equal to the analyzer's HEAD, no dirty
    rate/ or learn/ (runs or analyzer), and code_sha256 equal to CODE_FILES on disk. state / code_sha: golden-only overrides
    ((head, dirty) and the disk hashes), not reachable from the CLI. all_logs = {base: {log: rows}}.
    Returns {"prereg_sha256", "reference_sha256", "meta": {"<log>_<base>": meta row}}."""
    runs = [(b, n) for b in BASES for n in LOGS]
    bad = ["%s_%s" % (n, b) for b, n in runs if not any(r.get("phase") == "done" for r in all_logs[b][n])]
    if bad:
        raise SystemExit("analyze refused: missing or incomplete runs (no done row): %s" % ", ".join(bad))
    meta = {}
    for b, n in runs:
        m = [r for r in all_logs[b][n] if r.get("phase") == "meta"]
        if len(m) != 1:
            raise SystemExit("analyze refused: log %s_%s has %d meta rows (need exactly 1)" % (n, b, len(m)))
        meta[(n, b)] = m[0]
    wrong = ["%s_%s (base %r)" % (n, b, m.get("base")) for (n, b), m in meta.items() if m.get("base") != b]
    if wrong:
        raise SystemExit("analyze refused: meta base does not match the file: %s" % ", ".join(wrong))
    for k in ("prereg_sha256", "reference_sha256"):
        if len({m.get(k) for m in meta.values()}) != 1:
            raise SystemExit("analyze refused: logs carry different %s: %s" % (k, {"%s_%s" % nb: (m.get(k) or "none")[:12] for nb, m in meta.items()}))
    want = next(iter(meta.values()))["prereg_sha256"]
    have = sha256_file(require_prereg(prereg))
    if have != want:
        raise SystemExit("analyze refused: prereg on disk (sha256 %s) is not the one the logs were run under (%s)" % (have[:12], str(want)[:12]))
    have = sha256_file(REF_PATH if ref_path is None else ref_path)
    want_ref = next(iter(meta.values()))["reference_sha256"]
    if have != want_ref:
        raise SystemExit("analyze refused: reference.json on disk (sha256 %s) is not the one the logs were run under (%s)" % (have[:12], str(want_ref)[:12]))
    heads = {m.get("git_head") for m in meta.values()}
    if len(heads) != 1:
        raise SystemExit("analyze refused: the 24 runs carry different git_head: %s" % {"%s_%s" % nb: m.get("git_head") for nb, m in meta.items()})
    run_head = next(iter(heads))
    if not isinstance(run_head, str) or not re.fullmatch(r"[0-9a-f]{7,40}", run_head):
        raise SystemExit("analyze refused: the runs carry an unknown git_head (%r); science runs need a readable commit" % (run_head,))
    a_head, a_dirty = git_state() if state is None else state
    if a_head != run_head:
        raise SystemExit("analyze refused: analyzer HEAD %s is not the commit the runs were made at (%s)" % (a_head, run_head))
    dirty = ["%s_%s (%s)" % (n, b, k) for (n, b), m in meta.items() for k in ("git_dirty_rate", "git_dirty_learn") if m.get(k) is not False]
    dirty += ["analyzer (%s)" % k for k in ("rate", "learn") if a_dirty.get(k) is not False]
    if dirty:
        raise SystemExit("analyze refused: uncommitted changes (or git unreadable) in rate/ or learn/ for: %s" % ", ".join(dirty))
    disk = code_hashes() if code_sha is None else code_sha
    for (n, b), m in meta.items():
        cs = m.get("code_sha256")
        if not isinstance(cs, dict) or set(cs) != set(CODE_FILES):
            raise SystemExit("analyze refused: %s_%s meta has no complete code_sha256 (%s)" % (n, b, ", ".join(CODE_FILES)))
        off = [f for f in CODE_FILES if cs[f] != disk[f]]
        if off:
            raise SystemExit("analyze refused: source changed since the runs: %s_%s code_sha256 differs on disk for %s" % (n, b, ", ".join(off)))
    for (n, b), m in meta.items():
        args = m.get("args") or {}
        off = [k for k, v in PREREG_PARAMS.items() if args.get(k) != v]
        if off or not _arms_ok(n, args) or (args.get("mode"), args.get("rule") if LOG_ARGS[n][1] else None) != LOG_ARGS[n]:
            raise SystemExit("analyze refused: %s_%s was not run with the prereg parameters %s / default arms / its own mode (args: %s)" % (
                n, b, PREREG_PARAMS, {k: args.get(k) for k in ("mode", "rule", "arms", *PREREG_PARAMS)}))
    for b in BASES:
        gains = {n: meta[(n, b)].get("gains") for n in LOGS}
        if b == "c0":
            if any(g is not None for g in gains.values()):
                raise SystemExit("analyze refused: base c0 logs carry gains")
            continue
        seed = int(b[3:])
        shas = {(g or {}).get("sha256") for g in gains.values()}
        if len(shas) != 1 or None in shas or any(g.get("fit_seed") != seed for g in gains.values()):
            raise SystemExit("analyze refused: base %s logs do not carry one gains sha256 for seed %d: %s" % (
                b, seed, {n: ((g or {}).get("sha256") or "none")[:12] for n, g in gains.items()}))
        cell = sha256_file(fit_cell_path(seed, fit_dir))
        if shas != {cell}:
            raise SystemExit("analyze refused: base %s gains sha256 %s is not the chunk-1 cell on disk (%s)" % (b, next(iter(shas))[:12], cell[:12]))
    return {"prereg_sha256": want, "reference_sha256": next(iter(meta.values()))["reference_sha256"],
            "meta": {"%s_%s" % nb: m for nb, m in meta.items()}}


def _f(x):
    return "n/a" if x is None else "%.4g" % x


def _lab(v):
    return "%s%s" % (v["label"], " (%s)" % v["diagnostic"] if v.get("diagnostic") else "")


def _base_md(base, res, end, guard):
    md = ["### Numbers (%s)" % base, ""]
    for p, v in res.items():
        md.append("- %s: %s" % (p, ", ".join("%s=%s" % (k, _f(x)) for k, x in v["numbers"].items()
                                              if x is None or isinstance(x, (int, float)))))
    md += ["", "### Labels (%s)" % base, ""] + ["- %s: %s" % (p, _lab(v)) for p, v in res.items()]
    md += ["", "### End-of-arm latch (%s; reported, never gated)" % base, ""]
    md += ["- %s %s %s %s: latched=%s kc_paired=%d appr_frac=%s" % (
        r["log"], r["protocol"], r["arm"], r["song"], r["latched"], r["n_latched_kc_paired"],
        _f(r["approach_latched_rate_frac"])) for r in end] or ["- none"]
    md += ["", "### Guard summary (%s)" % base, ""]
    md += ["- %s: rows %d, max active ratio %s, MBON06 ratio [%s, %s], breaches %d, aborted %s" % (
        k, g["n"], _f(g["max_active_ratio"]), _f(g["min_mbon06_ratio"]), _f(g["max_mbon06_ratio"]), g["breaches"], g["aborted"])
        for k, g in sorted(guard.items())] or ["- no guard rows"]
    return md + [""]


def analyze(out_dir, ref, prereg=None, fit_dir=None, ref_path=None, reanalyze=False, state=None, code_sha=None):
    """Read all 24 batch logs (c0 + c1s0..4 x LOGS), refuse partial, mixed-provenance, dirty, code-changed or off-prereg
    batches, label every base, then write ONE result.json + RESULT.md (c0 section, c1 section) into out_dir. Returns the result dict.
    An existing result.json / RESULT.md refuses unless reanalyze, which records reanalyzed + the previous result.json sha256.
    Both files are written to temp names and renamed, so a refusal or crash never leaves a partial result.
    The base is passed explicitly (read_logs / label_protocols are pure); no module global is read or set.
    state / code_sha: golden-only overrides, see check_provenance."""
    rj, rm = os.path.join(out_dir, RESULT_JSON), os.path.join(out_dir, RESULT_MD)
    prev_sha = None
    if os.path.exists(rj) or os.path.exists(rm):
        if not reanalyze:
            raise SystemExit("analyze refused: %s already holds a result (result.json / RESULT.md); pass --reanalyze to label again (recorded in the new result)" % out_dir)
        prev_sha = sha256_file(rj) if os.path.exists(rj) else None
    all_logs = {b: read_logs(out_dir, b) for b in BASES}
    prov = check_provenance(all_logs, prereg, fit_dir, ref_path, state, code_sha)
    head, dirty = git_state() if state is None else state
    res = {b: label_protocols(all_logs[b], ref) for b in BASES}
    seeds = {int(b[3:]): res[b] for b in BASES if b != "c0"}
    arm2 = arm_labels(seeds)
    summ = {b: summaries(all_logs[b]) for b in BASES}
    per_base = lambda b: {"protocols": res[b], "end_latch": summ[b][0], "guard": summ[b][1],  # noqa: E731
                          "logs_present": {n: len(r) for n, r in all_logs[b].items()}}
    result = {"analyzer": {"git_head": head, "git_dirty_rate": dirty["rate"], "git_dirty_learn": dirty["learn"]},
              "reanalyzed": bool(reanalyze), "previous_result_sha256": prev_sha,
              "provenance": prov, "rule": rule_text(),
              "c0": dict(per_base("c0"), arm="(i)"),
              "c1": {"arm": "(ii)", "arm_labels": arm2, "seeds": {str(s): per_base("c1s%d" % s) for s in sorted(seeds)}}}
    md = ["# Chunk-2 result", "", "## Provenance", "",
          "- prereg sha256: %s" % prov["prereg_sha256"], "- reference.json sha256: %s" % prov["reference_sha256"],
          "- analyzer: git %s dirty rate=%s learn=%s" % (head, dirty["rate"], dirty["learn"]),
          "- data sha256 (distinct across runs; reported, not refused): %s" % sorted({json.dumps(m.get("data_sha256"), sort_keys=True) for m in prov["meta"].values()})]
    if reanalyze:
        md.append("- REANALYZED: true; previous result.json sha256 %s" % prev_sha)
    md += ["- %s: git %s dirty rate=%s learn=%s overwrite=%s%s" % (n, m["git_head"], m["git_dirty_rate"], m["git_dirty_learn"], m["overwrite"],
                                                               "" if not m.get("gains") else " gains %s" % m["gains"]["sha256"][:12])
           for n, m in prov["meta"].items()]
    md += ["", "## Rule", "", rule_text(), "", "## c0 (arm i)", ""] + _base_md("c0", res["c0"], *summ["c0"])
    md += ["## c1 (arm ii: C0 + chunk-1 gains, %d fitted seeds)" % len(seeds), "", "### Arm label (PASS iff >= %d of %d seeds PASS)" % (arm_need(len(seeds)), len(seeds)), ""]
    md += ["- %s: %s (%d of %d seeds)" % (p, v["label"], v["n_pass"], v["n_seeds"]) for p, v in arm2.items()]
    md += ["", "### Per-seed labels", "", "| seed | " + " | ".join(PROTOCOLS) + " |", "|---|" + "---|" * len(PROTOCOLS)]
    md += ["| %d | %s |" % (s, " | ".join(_lab(res["c1s%d" % s][p]) for p in PROTOCOLS)) for s in sorted(seeds)] + [""]
    for s in sorted(seeds):
        md += _base_md("c1s%d" % s, res["c1s%d" % s], *summ["c1s%d" % s])
    with open(rj + ".tmp", "w") as f:
        json.dump(result, f, indent=1)
    with open(rm + ".tmp", "w") as f:
        f.write("\n".join(md) + "\n")
    os.replace(rj + ".tmp", rj)
    os.replace(rm + ".tmp", rm)
    return result


# ---------------------------------------------------------------------------------------------------------------
# --analyze-bidir-ref (PREREGISTER_rate_chunk2_bidir_ref.md): LIF MB-level reference R + re-score of the 12 rate bidir logs.
# Pure functions over jsonl rows; every constant below is part of the rule text the result quotes.
REF_FRAC = 0.5            # prereg: rate PASS (a) needs rise_1_norm >= REF_FRAC x R
LIF_MIN_POSITIVE = 4      # prereg: LIF reference USABLE needs R > 0 in >= this many of the 5 seeds ...
LIF_MIN_RELEARN_SEEDS = 4 # ... and the LIF timed rule relearning (>= BIDIR_MIN_RELEARN of 3 cycles) in >= this many seeds
REF_LABEL = "bidir vs LIF reference"
EVAL_MD = os.path.join(OUT, "EVALUATION.md")
PAPER0_RUNG_A = tuple("results/condition_o1s%d.jsonl" % s for s in LIF_SEEDS)
RULES = ("timed", "depress")


def rule_text_ref():
    return "\n".join([
        "RULE (PREREGISTER_rate_chunk2_bidir_ref.md; thresholds fixed in code before any LIF run):",
        "Quantities (identical to the chunk-2 bidir rule): A = DC2 approach-MBON mean at a T probe; level = -A; rise_k = level(F_k) - level(B_(k-1)) "
        "(T0 for k = 1); drop_j = level(F_j) - level(B_j); rise_1_norm = rise_1 / A(T0) of the same run. Cycle k = 2..%d relearns iff rise_1 > 0 and "
        "rise_k >= %g x rise_1 and timed drop_(k-1) > 0 and timed drop_(k-1) > depress drop_(k-1) (depress paired with timed by seed for the LIF, "
        "by base for the rate model; the rise_1 > 0 clause is chunk 2's, unchanged)." % (CYCLES, BIDIR_RELEARN_FRAC),
        "Reference: a LIF seed is SCORED iff neither its timed nor its depress run has a guard abort and both runs have a complete T/F/B curve; any abort or "
        "incomplete curve in either rule's run voids that seed (FAIL-UNSTABLE / incomplete: not positive, not relearning). R = mean over the SCORED seeds of the "
        "timed rise_1_norm (n scored reported; SD is the sample SD). LIF sanity rule, thresholds absolute out of the %d seeds: the reference is USABLE iff rise_1_norm > 0 "
        "in at least %d seeds and R > 0 and the LIF timed rule relearns (>= %d of %d scorable cycles) in at least %d seeds. Otherwise the label is NO-REFERENCE: "
        "the rate bidir is untestable at the MB level." % (len(LIF_SEEDS), LIF_MIN_POSITIVE, BIDIR_MIN_RELEARN, CYCLES - 1, LIF_MIN_RELEARN_SEEDS),
        "Re-scored rate label (%s), per base c0, c1s0..c1s%d: PASS iff (a) rate timed rise_1_norm >= %g x R and (b) >= %d of the %d scorable cycles "
        "relearn. Otherwise FAIL naming the failing part ((a), (b) or both). Arm (i) = base c0. Arm (ii) = bases c1s0..c1s%d: PASS iff at least "
        "ceil(%g x n) = %d of the n = %d seeds PASS, else FAIL. The chunk-2 bidir label (FAIL) is not changed by this result."
        % (REF_LABEL, FIT_SEEDS[-1], REF_FRAC, BIDIR_MIN_RELEARN, CYCLES - 1, FIT_SEEDS[-1], ARM_PASS_FRAC, arm_need(len(FIT_SEEDS)), len(FIT_SEEDS)),
        "Secondary (reported, never gated): (1) dose-matched rung A: the rung-A learn-arm normalised DC2 approach drop, interpolated linearly between the "
        "logged test trials 5..30 as a function of |w_mean_frac| (curve in trial order; if |w| is not non-decreasing over the trials it is flagged "
        "dose_curve_nonmonotone and gives no value), at the |w_mean_frac| the timed rule reached at F1, and rise_1_norm / that drop "
        "(rate: each base's rung_a log; LIF: paper-0 rung A rows condition_o1s0-4 paired by seed). A dose outside the logged range gives no value (an "
        "origin-anchored value, 0 weight change = 0 drop, is shown separately and is not part of the prereg). (2) avoid-MBON component and PAM rate per "
        "block (DC2 probe, both rules). (3) |w_mean_frac| per block, both rules. (4) The headroom and baseline / end latch rows of every LIF run and rate log."])


def read_rows(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(x) for x in f if x.strip()]


def eval_log_hashes(md_path=EVAL_MD):
    """{log file name: sha256} from the '- name.jsonl <sha256>' lines of results/rate_chunk2/EVALUATION.md."""
    with open(md_path, encoding="utf-8") as f:
        return {m.group(1): m.group(2) for m in re.finditer(r"^- (\S+\.jsonl) ([0-9a-f]{64})\s*$", f.read(), re.M)}


def verify_rate_logs(rate_dir=OUT, md_path=EVAL_MD):
    """Refuse (SystemExit) unless every rate bidir (and rung_a, for the dose-matched secondary) log of every base is listed in
    EVALUATION.md, exists, and matches its sha256. Returns {file name: sha256}."""
    listed = eval_log_hashes(md_path)
    names = ["%s_%s.jsonl" % (n, b) for b in BASES for n in ("bidir_timed", "bidir_depress", "rung_a")]
    bad = []
    for n in names:
        p = os.path.join(rate_dir, n)
        if n not in listed:
            bad.append("%s (not in %s)" % (n, os.path.basename(md_path)))
        elif not os.path.isfile(p):
            bad.append("%s (missing)" % n)
        elif sha256_file(p) != listed[n]:
            bad.append("%s (sha256 %s != listed %s)" % (n, sha256_file(p)[:12], listed[n][:12]))
    if bad:
        raise SystemExit("analyze-bidir-ref refused: rate logs do not match the EVALUATION.md sha256 list: %s" % "; ".join(bad))
    return {n: listed[n] for n in names}


def verify_sources(ref, root=_HERE):
    """Refuse unless the paper-0 rung-A jsonl rows (condition_o1s0-4) match the sha256 recorded in reference.json sources."""
    src, bad = ref.get("sources") or {}, []
    for k in PAPER0_RUNG_A:
        p = os.path.join(root, *k.split("/"))
        if k not in src:
            bad.append("%s (not in reference.json sources)" % k)
        elif not os.path.isfile(p):
            bad.append("%s (missing)" % k)
        elif sha256_file(p) != src[k]:
            bad.append("%s (sha256 differs from reference.json)" % k)
    if bad:
        raise SystemExit("analyze-bidir-ref refused: paper-0 rung A rows do not match reference.json: %s" % "; ".join(bad))
    return {k: src[k] for k in PAPER0_RUNG_A}


def lif_log_name(rule, s):
    return "bidir_%s_lifs%d.jsonl" % (rule, s)


def check_lif_provenance(lif_logs, prereg=None, ref_path=None, state=None, code_sha=None):
    """Refuse (SystemExit) unless all 10 LIF runs (rule x seed) carry a done row, exactly one meta row with the right base, one prereg sha
    (= PREREG_REF on disk), one reference.json sha (= disk), one known git_head equal to the analyzer's, clean rate/ learn/ (runs and analyzer),
    code_sha256 equal to CODE_FILES on disk, the prereg run parameters / mode / rule / arm / seed, and the LIF block (regime, floor 5, W_SYN, frozen
    brain file, tick-seed formula, probe seed). lif_logs = {(rule, seed): rows}. state / code_sha: golden-only overrides."""
    runs = [(r, s) for s in LIF_SEEDS for r in RULES]
    tag = lambda k: lif_log_name(*k)[:-6]    # noqa: E731
    bad = [tag(k) for k in runs if not any(r.get("phase") == "done" for r in lif_logs.get(k, []))]
    if bad:
        raise SystemExit("analyze-bidir-ref refused: missing or incomplete LIF runs (no done row): %s" % ", ".join(bad))
    meta = {}
    for k in runs:
        m = [r for r in lif_logs[k] if r.get("phase") == "meta"]
        if len(m) != 1:
            raise SystemExit("analyze-bidir-ref refused: LIF log %s has %d meta rows (need exactly 1)" % (tag(k), len(m)))
        meta[k] = m[0]
    wrong = ["%s (base %r)" % (tag(k), m.get("base")) for k, m in meta.items() if m.get("base") != "lifs%d" % k[1]]
    if wrong:
        raise SystemExit("analyze-bidir-ref refused: LIF meta base does not match the file: %s" % ", ".join(wrong))
    for key in ("prereg_sha256", "reference_sha256"):
        if len({m.get(key) for m in meta.values()}) != 1:
            raise SystemExit("analyze-bidir-ref refused: LIF logs carry different %s" % key)
    want = next(iter(meta.values()))["prereg_sha256"]
    have = sha256_file(require_prereg(PREREG_REF if prereg is None else prereg))
    if have != want:
        raise SystemExit("analyze-bidir-ref refused: prereg on disk (sha256 %s) is not the one the LIF runs were run under (%s)" % (have[:12], str(want)[:12]))
    want_ref = next(iter(meta.values()))["reference_sha256"]
    have = sha256_file(REF_PATH if ref_path is None else ref_path)
    if have != want_ref:
        raise SystemExit("analyze-bidir-ref refused: reference.json on disk (sha256 %s) is not the one the LIF runs used (%s)" % (have[:12], str(want_ref)[:12]))
    heads = {m.get("git_head") for m in meta.values()}
    if len(heads) != 1:
        raise SystemExit("analyze-bidir-ref refused: the 10 LIF runs carry different git_head: %s" % {tag(k): m.get("git_head") for k, m in meta.items()})
    run_head = next(iter(heads))
    if not isinstance(run_head, str) or not re.fullmatch(r"[0-9a-f]{7,40}", run_head):
        raise SystemExit("analyze-bidir-ref refused: the LIF runs carry an unknown git_head (%r)" % (run_head,))
    a_head, a_dirty = git_state() if state is None else state
    if a_head != run_head:
        raise SystemExit("analyze-bidir-ref refused: analyzer HEAD %s is not the commit the LIF runs were made at (%s)" % (a_head, run_head))
    dirty = ["%s (%s)" % (tag(k), d) for k, m in meta.items() for d in ("git_dirty_rate", "git_dirty_learn") if m.get(d) is not False]
    dirty += ["analyzer (%s)" % d for d in ("rate", "learn") if a_dirty.get(d) is not False]
    if dirty:
        raise SystemExit("analyze-bidir-ref refused: uncommitted changes (or git unreadable) in rate/ or learn/ for: %s" % ", ".join(dirty))
    disk = code_hashes() if code_sha is None else code_sha
    for k, m in meta.items():
        cs = m.get("code_sha256")
        if not isinstance(cs, dict) or set(cs) != set(CODE_FILES):
            raise SystemExit("analyze-bidir-ref refused: %s meta has no complete code_sha256" % tag(k))
        off = [f for f in CODE_FILES if cs[f] != disk[f]]
        if off:
            raise SystemExit("analyze-bidir-ref refused: source changed since the LIF runs: %s code_sha256 differs on disk for %s" % (tag(k), ", ".join(off)))
    want_lif = {"regime": LIF_REGIME, "eln_negate": True, "pn_kc_gain": 8.0, "min_syn": LIF_MIN_SYN, "w_syn": 0.275, "brain": LIF_BRAIN,
                "tick_seed_formula": LIF_TICK_SEED_FORMULA, "probe_seed": 0}
    for (rule, s), m in meta.items():
        args = m.get("args") or {}
        off = [x for x, v in PREREG_PARAMS.items() if args.get(x) != v]
        if off or (args.get("mode"), args.get("rule"), args.get("arm"), args.get("lif_seed")) != ("bidir", rule, "lif", s):
            raise SystemExit("analyze-bidir-ref refused: %s was not run with the prereg parameters %s / bidir / its own rule, arm lif, seed (args: %s)" % (
                tag((rule, s)), PREREG_PARAMS, {x: args.get(x) for x in ("mode", "rule", "arm", "lif_seed", *PREREG_PARAMS)}))
        if m.get("lif") != want_lif or m.get("gains") is not None:
            raise SystemExit("analyze-bidir-ref refused: %s lif block is not the prereg model %s (got %s)" % (tag((rule, s)), want_lif, m.get("lif")))
    return {"prereg_sha256": want, "reference_sha256": want_ref, "git_head": run_head,
            "data_sha256": sorted({json.dumps(m.get("data_sha256"), sort_keys=True) for m in meta.values()}),
            "overwrite": {tag(k): bool(m.get("overwrite")) for k, m in meta.items()}}


def check_rate_meta(rate_logs):
    """Record (never refuse on HEAD) the rate bidir logs' provenance: one meta row, a done row, base and rule as named. rate_logs = {(rule, base): rows}."""
    heads, pre = set(), set()
    for (rule, b), rows in rate_logs.items():
        nm = "bidir_%s_%s" % (rule, b)
        m = [r for r in rows if r.get("phase") == "meta"]
        if len(m) != 1 or not any(r.get("phase") == "done" for r in rows):
            raise SystemExit("analyze-bidir-ref refused: rate log %s needs one meta row and a done row" % nm)
        if m[0].get("base") != b or (m[0].get("args") or {}).get("rule") != rule or (m[0].get("args") or {}).get("mode") != "bidir":
            raise SystemExit("analyze-bidir-ref refused: rate log %s meta does not name its base / rule / mode" % nm)
        heads.add(m[0].get("git_head"))
        pre.add(m[0].get("prereg_sha256"))
    return {"git_heads": sorted(map(str, heads)), "prereg_sha256": sorted(map(str, pre))}


def bidir_numbers(timed, depress):
    """Per-run bidir quantities (pure). complete = every T/F/B tag present in both rules. rise_1 / rise_1_norm only need timed T0 and F1.
    Relearning counts only on a complete pair of curves and follows chunk 2's rule (rise_1 > 0, rise_k >= 0.5 rise_1, timed drop > 0 and > depress drop)."""
    ct, cd = _bidir_curve(timed), _bidir_curve(depress)
    tags = ["T0"] + ["%s%d" % (p, k) for k in range(1, CYCLES + 1) for p in "FB"]
    missing = [t for t in tags if ct.get(t) is None or cd.get(t) is None]
    app0 = _field(timed, "mbon_approach_hz", BIDIR_SONG, after="T0")
    r1 = _sub(ct.get("F1"), ct.get("T0"))
    out = {"complete": not missing, "missing": missing, "baseline_approach_hz": app0, "rise_1": r1, "rise_1_norm": _norm(r1, app0),
           "cycles": {}, "n_relearn": None, "scorable": CYCLES - 1, "relearns": None}
    if missing:
        return out
    rise = lambda c, k: c["F%d" % k] - c["T0" if k == 1 else "B%d" % (k - 1)]  # noqa: E731
    drop = lambda c, j: c["F%d" % j] - c["B%d" % j]  # noqa: E731
    n = 0
    for k in range(2, CYCLES + 1):
        ok = bool(r1 > 0 and rise(ct, k) >= BIDIR_RELEARN_FRAC * r1 and drop(ct, k - 1) > 0 and drop(ct, k - 1) > drop(cd, k - 1))
        out["cycles"][k] = {"rise": rise(ct, k), "timed_drop_prev": drop(ct, k - 1), "depress_drop_prev": drop(cd, k - 1), "relearns": ok}
        n += ok
    out.update(n_relearn=n, relearns=bool(n >= BIDIR_MIN_RELEARN))
    return out


def _w_f1(rows):
    v = [r["w_mean_frac"] for r in rows if r.get("after") == "F1" and r.get("song") == BIDIR_SONG and "w_mean_frac" in r]
    return abs(v[-1]) if v else None


def rung_a_curve(rows):
    """[(|w_mean_frac| at train trial t, normalised DC2 approach drop at test trial t)] over the learn arm's logged test trials, in trial order."""
    a0 = _app(rows, "learn", "dc2", "baseline", 0)
    pts = []
    for t in sorted({r["trial"] for r in rows if r.get("arm") == "learn" and r.get("phase") == "test" and r.get("song") == "dc2"}):
        d, w = _norm(_sub(a0, _app(rows, "learn", "dc2", "test", t)), a0), _w_frac(rows, "learn", "train", t)
        if d is not None and w is not None:
            pts.append((abs(w), d))
    return pts


def dose_matched(pts, w):
    """Linear interpolation of the rung-A normalised drop at |w| (prereg: only between the logged trials). pts is in trial order; if |w| is not
    non-decreasing over the trials the curve is flagged (dose_curve_nonmonotone) and no value is given. Outside the logged range value is None;
    below it an origin-anchored (0, 0) extension is shown separately (not part of the prereg)."""
    if w is None or len(pts) < 2:
        return {"w": w, "value": None, "status": "no-data", "origin_anchored": None, "dose_curve_nonmonotone": False}
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    if any(b < a for a, b in zip(xs, xs[1:])):
        return {"w": w, "value": None, "status": "nonmonotone", "origin_anchored": None, "dose_curve_nonmonotone": True, "range": [min(xs), max(xs)]}
    r = {"dose_curve_nonmonotone": False, "range": [xs[0], xs[-1]]}
    if w < xs[0]:
        return dict(r, w=w, value=None, status="below-range", origin_anchored=ys[0] * w / xs[0] if xs[0] > 0 else None)
    if w > xs[-1]:
        return dict(r, w=w, value=None, status="above-range", origin_anchored=None)
    return dict(r, w=w, value=float(np.interp(w, xs, ys)), status="in-range", origin_anchored=None)


def dose_row(n1, w, rung_rows):
    """n1 = rise_1_norm from bidir_numbers; w = |w_mean_frac| at the timed F1."""
    d = dose_matched(rung_a_curve(rung_rows), w)
    d["rise_1_norm"] = n1
    d["ratio"] = _ratio(n1, d["value"]) if n1 is not None and d["value"] else None
    d["ratio_origin_anchored"] = _ratio(n1, d["origin_anchored"]) if n1 is not None and d["origin_anchored"] else None
    return d


def science_rows(rows):
    """Headroom and latch rows of one run (reported, never gated)."""
    return {"headroom": [r for r in rows if r.get("phase") == "headroom"], "latch": [r for r in rows if r.get("phase") == "latch"]}


def _tag_means(rows, key, f=lambda v: v):
    tags = []
    for r in rows:
        if r.get("after") and r["after"] not in tags:
            tags.append(r["after"])
    return {t: (lambda v: None if v is None else f(v))(_field(rows, key, BIDIR_SONG, after=t)) for t in tags}


def block_series(rows):
    """Secondary (2)+(3): per T-probe tag, DC2 avoid-MBON rate, PAM rate, |w_mean_frac| (None when absent)."""
    return {"avoid_mbon_hz": _tag_means(rows, "mbon_avoid_hz"), "pam_hz": _tag_means(rows, "pam_hz"),
            "abs_w_mean_frac": _tag_means(rows, "w_mean_frac", abs)}


def _unstable(*row_sets):
    return any(r.get("phase") == "abort" for rows in row_sets for r in rows)


def lif_reference(per_seed):
    """per_seed = {seed: {"nums": bidir_numbers, "unstable": bool}}. Returns the reference summary with the sanity rule applied."""
    scored = {s: v["nums"]["rise_1_norm"] for s, v in per_seed.items() if v["nums"]["complete"] and not v["unstable"] and v["nums"]["rise_1_norm"] is not None}
    vals = list(scored.values())
    R = float(np.mean(vals)) if vals else None
    sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else None
    n_pos = sum(v > 0 for v in vals)
    n_same = None if R is None else sum(v != 0 and (v > 0) == (R > 0) for v in vals)
    relearn = {s: bool(v["nums"]["relearns"]) and not v["unstable"] for s, v in per_seed.items()}
    n_rl = sum(relearn.values())
    usable = bool(R is not None and R > 0 and n_pos >= LIF_MIN_POSITIVE and n_rl >= LIF_MIN_RELEARN_SEEDS)
    return {"R": R, "sd": sd, "values": {str(s): v for s, v in sorted(scored.items())}, "n_scored": len(vals), "n_positive": n_pos,
            "n_same_sign": n_same, "n_seeds": len(per_seed), "relearn_cycles": {str(s): per_seed[s]["nums"]["n_relearn"] for s in sorted(per_seed)},
            "relearns": {str(s): relearn[s] for s in sorted(relearn)}, "n_relearn_seeds": n_rl,
            "unstable_seeds": [s for s in sorted(per_seed) if per_seed[s]["unstable"]],
            "label": "USABLE" if usable else "NO-REFERENCE", "usable": usable}


def rescore_base(nums, R, usable, unstable=False):
    """Re-scored rate label for one base (pure). NO-REFERENCE when the LIF reference is unusable; otherwise PASS iff (a) and (b)."""
    n1 = nums["rise_1_norm"]
    a = bool(usable and n1 is not None and n1 >= REF_FRAC * R)
    b = bool(nums["relearns"])
    out = {"rise_1_norm": n1, "threshold": REF_FRAC * R if usable else None, "a": a if usable else None, "b": b,
           "n_relearn": nums["n_relearn"], "complete": nums["complete"], "unstable": unstable}
    if not usable:
        return dict(out, label="NO-REFERENCE", failing=None)
    failing = (["(a)"] if not a else []) + (["(b)"] if not b else []) + (["(incomplete)"] if not nums["complete"] else []) + (["(unstable)"] if unstable else [])
    return dict(out, label="FAIL" if failing else "PASS", failing=failing or None)


def fail_part(failing):
    if not failing:
        return None
    return "both" if "(a)" in failing and "(b)" in failing else " ".join(failing)


def rate_arms(per_base, usable):
    """Arm (i) = c0; arm (ii) = c1s0..4 PASS iff >= arm_need(n) pass. NO-REFERENCE propagates."""
    seeds = [b for b in BASES if b != "c0"]
    n_pass = sum(per_base[b]["label"] == "PASS" for b in seeds)
    need = arm_need(len(seeds))
    l2 = "NO-REFERENCE" if not usable else ("PASS" if n_pass >= need else "FAIL")
    return {"arm_i": {"base": "c0", "label": per_base["c0"]["label"], "failing": fail_part(per_base["c0"]["failing"])},
            "arm_ii": {"label": l2, "n_pass": n_pass, "n_seeds": len(seeds), "need": need,
                       "per_seed": {b: {"label": per_base[b]["label"], "failing": fail_part(per_base[b]["failing"])} for b in seeds}}}


def analyze_bidir_ref(out_dir, ref, rate_dir=OUT, eval_md=EVAL_MD, src_root=_HERE, prereg=None, ref_path=None, reanalyze=False,
                      state=None, code_sha=None):
    """One analysis pass over the 10 LIF logs (out_dir) and the 12 rate bidir logs (rate_dir). Refuses an existing result without reanalyze,
    a missing LIF done row, any provenance mismatch, and any rate / paper-0 log whose sha256 differs from the recorded one. Writes result.json +
    RESULT.md atomically. state / code_sha: golden-only overrides."""
    rj, rm = os.path.join(out_dir, RESULT_JSON), os.path.join(out_dir, RESULT_MD)
    prev_sha = None
    if os.path.exists(rj) or os.path.exists(rm):
        if not reanalyze:
            raise SystemExit("analyze-bidir-ref refused: %s already holds a result; pass --reanalyze to label again (recorded in the new result)" % out_dir)
        prev_sha = sha256_file(rj) if os.path.exists(rj) else None
    lif_logs = {(r, s): read_rows(os.path.join(out_dir, lif_log_name(r, s))) for s in LIF_SEEDS for r in RULES}
    prov = check_lif_provenance(lif_logs, prereg, ref_path, state, code_sha)
    rate_sha = verify_rate_logs(rate_dir, eval_md)
    src_sha = verify_sources(ref, src_root)
    rate_logs = {(r, b): read_rows(os.path.join(rate_dir, "bidir_%s_%s.jsonl" % (r, b))) for b in BASES for r in RULES}
    rate_prov = check_rate_meta(rate_logs)
    head, dirty = git_state() if state is None else state

    lif_ps = {s: {"nums": bidir_numbers(lif_logs[("timed", s)], lif_logs[("depress", s)]),
                  "unstable": _unstable(lif_logs[("timed", s)], lif_logs[("depress", s)])} for s in LIF_SEEDS}
    lref = lif_reference(lif_ps)
    p0 = {s: read_rows(os.path.join(src_root, *PAPER0_RUNG_A[s].split("/"))) for s in LIF_SEEDS}
    lif_out = {}
    for s in LIF_SEEDS:
        t = lif_logs[("timed", s)]
        lif_out[str(s)] = {"numbers": lif_ps[s]["nums"], "unstable": lif_ps[s]["unstable"], "abs_w_mean_f1": _w_f1(t),
                           "dose_matched_rung_a": dose_row(lif_ps[s]["nums"]["rise_1_norm"], _w_f1(t), p0[s]),
                           "science_rows": {r: science_rows(lif_logs[(r, s)]) for r in RULES},
                           "series": {r: block_series(lif_logs[(r, s)]) for r in RULES},
                           "guard": summaries({"%s_lifs%d" % (r, s): lif_logs[(r, s)] for r in RULES})[1],
                           "headroom_pass": {r: [x["gate_pass"] for x in lif_logs[(r, s)] if x.get("phase") == "headroom"] for r in RULES}}
    per_base, rate_out = {}, {}
    for b in BASES:
        t, d = rate_logs[("timed", b)], rate_logs[("depress", b)]
        nums, unst = bidir_numbers(t, d), _unstable(t, d)
        per_base[b] = rescore_base(nums, lref["R"], lref["usable"], unst)
        rate_out[b] = {"numbers": nums, "rescored": dict(per_base[b], failing_part=fail_part(per_base[b]["failing"])), "abs_w_mean_f1": _w_f1(t),
                       "dose_matched_rung_a": dose_row(nums["rise_1_norm"], _w_f1(t), read_rows(os.path.join(rate_dir, "rung_a_%s.jsonl" % b))),
                       "science_rows": {r: science_rows(rate_logs[(r, b)]) for r in RULES},
                       "series": {r: block_series(rate_logs[(r, b)]) for r in RULES}}
    result = {"label_name": REF_LABEL, "analyzer": {"git_head": head, "git_dirty_rate": dirty["rate"], "git_dirty_learn": dirty["learn"]},
              "reanalyzed": bool(reanalyze), "previous_result_sha256": prev_sha, "rule": rule_text_ref(),
              "provenance": dict(prov, rate_log_sha256=rate_sha, paper0_rung_a_sha256=src_sha, rate_logs=rate_prov),
              "lif_reference": dict(lref, seeds=lif_out), "rate": {"arms": rate_arms(per_base, lref["usable"]), "bases": rate_out}}
    with open(rj + ".tmp", "w") as f:
        json.dump(result, f, indent=1)
    with open(rm + ".tmp", "w") as f:
        f.write(ref_md(result) + "\n")
    os.replace(rj + ".tmp", rj)
    os.replace(rm + ".tmp", rm)
    return result


def _dm(d):
    if d["value"] is not None:
        return "%s (%s)" % (_f(d["value"]), d["status"])
    return "n/a (%s%s)" % (d["status"], "" if d.get("origin_anchored") is None else "; origin-anchored %s" % _f(d["origin_anchored"]))


def ref_md(res):
    L, prov, rate = res["lif_reference"], res["provenance"], res["rate"]
    a1, a2 = rate["arms"]["arm_i"], rate["arms"]["arm_ii"]
    md = ["# Chunk-2 bidir vs LIF reference", "", "## Rule", "", res["rule"], "", "## Provenance", "",
          "- prereg sha256 (LIF runs): %s" % prov["prereg_sha256"], "- reference.json sha256: %s" % prov["reference_sha256"],
          "- LIF runs git %s; analyzer git %s dirty rate=%s learn=%s" % (prov["git_head"], res["analyzer"]["git_head"], res["analyzer"]["git_dirty_rate"], res["analyzer"]["git_dirty_learn"]),
          "- rate bidir logs git (by design different): %s" % ", ".join(prov["rate_logs"]["git_heads"]),
          "- rate log sha256 verified against EVALUATION.md: %d logs; paper-0 rung A rows verified against reference.json: %d" % (len(prov["rate_log_sha256"]), len(prov["paper0_rung_a_sha256"])),
          "- LIF data sha256 (reported): %s" % prov["data_sha256"]]
    if res["reanalyzed"]:
        md.append("- REANALYZED: true; previous result.json sha256 %s" % res["previous_result_sha256"])
    md += ["", "## LIF reference: %s" % L["label"], "",
           "- R = %s, SD = %s, scored %d of %d seeds; rise_1_norm > 0 in %d; same sign as R in %s; timed rule relearns in %d seeds" % (
               _f(L["R"]), _f(L["sd"]), L["n_scored"], L["n_seeds"], L["n_positive"], L["n_same_sign"], L["n_relearn_seeds"]),
           "- unstable seeds: %s" % (L["unstable_seeds"] or "none"), "",
           "| seed | rise_1_norm | relearn cycles (of %d) | relearns | abs w F1 | dose-matched rung A drop | ratio |" % (CYCLES - 1), "|---|---|---|---|---|---|---|"]
    for s, v in L["seeds"].items():
        n = v["numbers"]
        md.append("| %s | %s | %s | %s | %s | %s | %s |" % (s, _f(n["rise_1_norm"]), n["n_relearn"], n["relearns"], _f(v["abs_w_mean_f1"]),
                                                          _dm(v["dose_matched_rung_a"]), _f(v["dose_matched_rung_a"]["ratio"])))
    md += ["", "## Re-scored rate label (%s)" % REF_LABEL, "",
           "- arm (i) c0: %s%s" % (a1["label"], "" if not a1["failing"] else " (failing %s)" % a1["failing"]),
           "- arm (ii): %s (%d of %d seeds PASS, need %d)" % (a2["label"], a2["n_pass"], a2["n_seeds"], a2["need"]), "",
           "| base | rise_1_norm | threshold %gxR | (a) | relearn cycles (of %d) | (b) | label | failing | dose-matched rung A drop | ratio |" % (REF_FRAC, CYCLES - 1),
           "|---|---|---|---|---|---|---|---|---|---|"]
    for b, v in rate["bases"].items():
        r = v["rescored"]
        md.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (b, _f(r["rise_1_norm"]), _f(r["threshold"]), r["a"], r["n_relearn"], r["b"], r["label"],
                                                                     r["failing_part"] or "-", _dm(v["dose_matched_rung_a"]), _f(v["dose_matched_rung_a"]["ratio"])))
    md += ["", "## Headroom and latch rows (reported, never gated)", ""]
    for who, items in (("LIF seed", L["seeds"]), ("rate base", rate["bases"])):
        for k, v in items.items():
            for r in RULES:
                sr = v["science_rows"][r]
                md += ["- %s %s %s headroom %s: gate_pass=%s approach_ratio=%s" % (who, k, r, h.get("song"), h.get("gate_pass"), _f(h.get("approach_ratio"))) for h in sr["headroom"]]
                md += ["- %s %s %s latch %s %s: latched=%s kc_paired=%s appr_frac=%s" % (who, k, r, x.get("when"), x.get("song"), x.get("latched"),
                                                                                         x.get("n_latched_kc_paired"), _f(x.get("approach_latched_rate_frac"))) for x in sr["latch"]]
    md += ["", "## Secondary (reported, never gated): per block, DC2 probe", ""]
    for who, items in (("LIF seed", L["seeds"]), ("rate base", rate["bases"])):
        for k, v in items.items():
            for r in RULES:
                for q, ser in v["series"][r].items():
                    md.append("- %s %s %s %s: %s" % (who, k, r, q, ", ".join("%s=%s" % (t, _f(x)) for t, x in ser.items())))
    return "\n".join(md)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--arm", choices=("c0", "c1", "lif"), default="c0", help="c1 = C0 + chunk-1 gains at g*=16 (needs --fit-seed); lif = paper 0 spiking model, bidir only (needs --lif-seed)")
    ap.add_argument("--fit-seed", type=int, default=None, help="chunk-1 fit seed 0-4 (arm c1)")
    ap.add_argument("--lif-seed", type=int, default=None, help="LIF noise seed 0-4 (arm lif)")
    ap.add_argument("--analyze", action="store_true", help="label the batch logs in results/rate_chunk2 (CPU only)")
    ap.add_argument("--reanalyze", action="store_true", help="with --analyze / --analyze-bidir-ref: replace an existing result (recorded in it)")
    ap.add_argument("--analyze-bidir-ref", action="store_true", help="LIF MB-level bidir reference + re-score of the rate bidir logs (CPU only)")
    ap.add_argument("--gate", action="store_true", help="headroom gate + latch check at baseline; no training")
    ap.add_argument("--mode", choices=("rung_a", "rung3", "bidir"), default="rung_a")
    ap.add_argument("--overwrite", action="store_true", help="replace an existing science log (recorded in its meta row)")
    ap.add_argument("--arms", default="")
    ap.add_argument("--rule", choices=("depress", "timed"), default="timed")
    ap.add_argument("--train", type=int, default=30)
    ap.add_argument("--relax", type=int, default=30)
    ap.add_argument("--probe", type=int, default=1)
    ap.add_argument("--eta", type=float, default=5e-6)
    ap.add_argument("--lam", type=float, default=PL.LAM)
    ap.add_argument("--seed0", type=int, default=0)
    a = ap.parse_args()
    t0 = time.time()
    if a.analyze and a.analyze_bidir_ref:
        raise SystemExit("pick one of --analyze / --analyze-bidir-ref")
    if a.analyze:    # one analysis over all 24 runs (c0 + c1s0..4); no per-base selection
        if a.arm != "c0" or a.fit_seed is not None:
            raise SystemExit("--analyze covers all bases at once; do not pass --arm / --fit-seed")
        res = analyze(OUT, load_reference(), reanalyze=a.reanalyze)
        for p, v in res["c0"]["protocols"].items():
            print("c0  %-7s %s %s" % (p, v["label"], v["diagnostic"] or ""))
        for p, v in res["c1"]["arm_labels"].items():
            print("c1  %-7s %s (%d of %d seeds)" % (p, v["label"], v["n_pass"], v["n_seeds"]))
        return
    if a.analyze_bidir_ref:
        if a.analyze or a.arm != "c0" or a.fit_seed is not None or a.lif_seed is not None:
            raise SystemExit("--analyze-bidir-ref covers all runs at once; do not combine it with --analyze / --arm / --fit-seed / --lif-seed")
        res = analyze_bidir_ref(OUT_REF, load_reference(), reanalyze=a.reanalyze)
        L = res["lif_reference"]
        print("%s: LIF reference %s (R %s, %d of %d seeds positive, %d relearn)" % (REF_LABEL, L["label"], _f(L["R"]), L["n_positive"], L["n_seeds"], L["n_relearn_seeds"]))
        arms = res["rate"]["arms"]
        print("rate arm (i) c0 %s; arm (ii) %s (%d of %d)" % (arms["arm_i"]["label"], arms["arm_ii"]["label"], arms["arm_ii"]["n_pass"], arms["arm_ii"]["n_seeds"]))
        return
    if a.reanalyze:
        raise SystemExit("--reanalyze only applies to --analyze / --analyze-bidir-ref")
    set_base(a.arm, a.fit_seed, lif_seed=a.lif_seed)    # before any log name is formed
    lif = a.arm == "lif"
    out = OUT_REF if lif else OUT
    if lif and (a.gate or (a.mode != "bidir" and not a.smoke)):
        raise SystemExit("--arm lif runs --mode bidir only (no --gate, rung_a, rung3)")
    if not (a.smoke or a.gate):    # refuse before the slow build: prereg missing/empty, or a log that would be overwritten
        require_prereg(PREREG_REF if lif else None)
        if not a.overwrite:
            p = os.path.join(out, log_name(a.mode, a.rule))
            if os.path.exists(p):
                open_log(p)
    sim = base_lif() if lif else base()
    print("base %s%s" % (BASE_NAME, "" if GAINS is None else "  gains %d (%d active) cap %g" % (GAINS["n_gains"], GAINS["n_active"], GAINS["cap"])))
    if lif:
        print("lif regime %s min_syn %d W_SYN %g tick seed: %s" % (LIF_REGIME, sim.net.min_syn, C.G.W_SYN, LIF_TICK_SEED_FORMULA))
    if a.smoke and lif:
        # Tiny open-loop bidir with the lesion on: weights must not move; no learning number is printed.
        P, _ = bidir_mb(sim, "timed", io.StringIO(), cycles=1, f_ticks=30, b_ticks=30, lesion="all", lif_seed=a.lif_seed)
        print("smoke ok, weights moved: %s  (%.0f s)" % (bool(P.dw.any()), time.time() - t0))
        assert not P.dw.any(), "lesion moved weights"
        return
    if a.smoke:
        # No science number before the prereg: the log goes nowhere, only the weight check prints.
        P, _ = rung_a(sim, ["lesion"], a.eta, a.lam, io.StringIO(), n_train=2, n_probe=1)["lesion"]
        print("w_mean_frac %s  (%.0f s)" % (P.summary()["mean_frac"], time.time() - t0))
        assert not P.dw.any(), "lesion arm moved weights"
        return
    if a.gate:
        os.makedirs(OUT, exist_ok=True)
        with open(os.path.join(OUT, "gate_%s.jsonl" % BASE_NAME), "w") as log:
            for pr in GATE_PROTOCOLS:
                rows, lat = science_prologue(sim, pr, log)
                for r in rows:
                    print("%-6s %-4s approach %.3f Hz (%.3fx) avoid %.3f (%.3fx) kc %.4f gate %s %s" % (
                        pr, r["song"], r["mbon_approach_hz"], r["approach_ratio"] or -1, r["mbon_avoid_hz"],
                        r["avoid_ratio"] or -1, r["kc_active"],
                        "PASS" if r["gate_pass"] else "FAIL", r["reason"] or ""))
                for r in lat:
                    print("latch  %-6s %-4s n %d kc_paired %d/%d appr_frac %.3f appr %d avoid %d ppl1 %d pam %d ppl1_hz %.3f pam_hz %.3f latched %s" % (
                        pr, r["song"], r["n_latched"], r["n_latched_kc_paired"], r["n_kc_paired"],
                        r["approach_latched_rate_frac"], r["n_latched_mbon_approach"], r["n_latched_mbon_avoid"],
                        r["n_latched_ppl1"], r["n_latched_pam"], r["ppl1_hz_offset"], r["pam_hz_offset"], r["latched"]))
        return
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, log_name(a.mode, a.rule))
    with open_log(path, a.overwrite) as log:
        log.write(json.dumps(meta_row(vars(a), a.overwrite, prereg=PREREG_REF if lif else None, extra=lif_meta() if lif else None)) + "\n")
        log.flush()
        science_prologue(sim, a.mode, log)
        if a.mode == "rung_a":
            rung_a(sim, (a.arms or ",".join(C.ODOUR_ARMS)).split(","), a.eta, a.lam, log, a.train, a.probe, LATCH_END_ARMS)
        elif a.mode == "rung3":
            rung3(sim, (a.arms or ",".join(persist.ARMS)).split(","), a, log, LATCH_END_ARMS)
        else:
            arm = "bidir_" + a.rule
            if guarded(sim, arm, C.STIM_SETS["odour"], log, lambda: bidir_mb(sim, a.rule, log, lif_seed=a.lif_seed if lif else None)) is not None:
                latch_check(sim, log, "bidir", "end", arm)
        log.write(json.dumps(done_row(a.mode, t0)) + "\n")
    print("done %s %.0f s" % (a.mode, time.time() - t0))


if __name__ == "__main__":
    main()
