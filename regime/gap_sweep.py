"""GAP_COUPLE / GAP_NORM sweep against plan criterion 1 (phase 3).

Nine arms, each a fresh GpuSim, on the standard two-odour multi-glomerular probe
(`regime/alln_multiglom.A` and `.B`, 2 shared glomeruli):

  base          everything off (the shipped regime)
  delete        GAP_COUPLE = kmax*1e-6: the same 3,802 chemical edges zeroed,
                coupling ~nil. The attribution control - it separates "deleting
                the eLN broadcast" from "coupling electrically instead".
  negate        ELN_NEGATE = True, the phase-1 bridge (alln-findings s7).
  scalar xF     GAP_COUPLE = kmax*F for F in 0.003 / 0.01 / 0.03.
  norm S=..     GAP_NORM = True, GAP_COUPLE = S in 0.0003 / 0.001 / 0.003, so
                every coupled PN carries a summed per-step coefficient of
                exactly S (physiological S ~ 0.0005-0.001).

`kmax` is derived here the way `regime/test_gap.py` does (1 / max per-neuron
summed synapse count over the gap edges); it is never hardcoded. The arm list is
a parameter of `main`, so a later phase can sweep a different one.

Verdict per arm, at SCORE_HZ, against `regime.layers.CRIT1` (the plan Success
Metrics, one copy):
  ALPN Jaccard <= 0.35 (mean over seeds)
  KC Jaccard <= 0.20 (mean over seeds)
  KC active 2-10% of the KC population, BOTH odours, EVERY seed
  central 1-5% of the central population (mean over seeds and odours)
  IGNITION_HZ central < 8% (max over seeds and odours) - no ignition

Working point: among arms passing all five, the lowest ALPN Jaccard; ties broken
by KC active closest to 5%. If none passes, the negative result is the
deliverable: which metric fails for the best arm, and the failure mode - `dead`
(KC below the floor), `saturated` (KC above the ceiling) or `merged` (a Jaccard
above threshold). An arm with no live KCs at all is not ranked: `jac` returns 0.0
on an empty union, so a silenced layer would otherwise look like perfect
separation.

Every run writes the table to `--out` (first line: the gpu_sim constants, the
commit and the date) and the per-row numbers to `<out>.json`, which `--rescore`
re-scores without touching the GPU. The text table rounds Jaccards to 2 dp and
cannot be re-scored; the JSON is the substrate.

Run: .venv/Scripts/python.exe regime/gap_sweep.py [--hz 60 120] [--seeds 0 2 4]
     [--out results/gap_sweep.txt]
     .venv/Scripts/python.exe regime/gap_sweep.py --rescore results/gap_sweep.txt.json
"""
import argparse
import datetime
import json
import os
import subprocess
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _HERE)
import gpu_sim as G
from regime.alln_multiglom import A, B, SHOW, row_cells, run
from regime.layers import CRIT1, LAYERS, jac
from regime.measure import set_consts

_M = dict(LAYERS)
ALPN, KC, CEN = _M["ALPN"], _M["KC"], _M["central"]
N_KC, N_CEN = int(KC.sum()), int(CEN.sum())

# The two rates the verdict is defined on. Criterion 1 scores separation and
# liveness at one rate and ignition at a higher one; they are named here so the
# scoring code never carries a bare 60 or 120.
SCORE_HZ = 60
IGNITION_HZ = 120

# Restored after every arm. Nothing else moves: APL, SFA and normalisation stay
# at their shipped defaults for the whole sweep.
DEFAULTS = {"GAP_COUPLE": 0.0, "GAP_NORM": False, "ELN_NEGATE": False}

# Everything a later reader needs to know the sweep ran on the brain it claims.
PROV_CONSTS = ["SFA_B_INC", "TAU_A", "APL_GRADED", "APL_SCALE", "APL_MAX_HZ",
               "APL_DIVISIVE", "NORM_TOTAL_TARGET", "ELN_NEGATE", "GAP_COUPLE",
               "GAP_NORM"]

_METRICS = ["alpn_j", "kc_j", "kc_act", "central", "cen120"]
SKIPPED = {}     # arm label -> the assert message that skipped it
_OUT = None


def emit(s):
    print(s, flush=True)
    if _OUT is not None:
        _OUT.write(s + "\n")
        _OUT.flush()


def provenance():
    try:
        head = subprocess.check_output(["git", "describe", "--always", "--dirty"],
                                       cwd=_HERE).decode().strip()
    except Exception as e:                       # not a checkout, or no git
        head = "unknown (%s)" % type(e).__name__
    return {"consts": {n: getattr(G, n) for n in PROV_CONSTS},
            "commit": head,
            "date": datetime.date.today().isoformat(),
            "score_hz": SCORE_HZ, "ignition_hz": IGNITION_HZ}


def kmax():
    """1 / max per-neuron summed synapse count over the gap edges, as test_gap."""
    b = G.brain()
    _, pre, post, count = G._gap_edges(b)
    s = np.zeros(b.n, np.float64)
    np.add.at(s, post, count.astype(np.float64))
    np.add.at(s, pre, count.astype(np.float64))
    return 1.0 / s.max()


def arms(km):
    out = [("base", {}),
           ("delete", {"GAP_COUPLE": km * 1e-6}),
           ("negate", {"ELN_NEGATE": True})]
    out += [("scalar x%g" % f, {"GAP_COUPLE": km * f}) for f in (0.003, 0.01, 0.03)]
    out += [("norm S=%g" % s, {"GAP_NORM": True, "GAP_COUPLE": s})
            for s in (0.0003, 0.001, 0.003)]
    return out


def run_arm(label, consts, rates, seeds):
    """One fresh GpuSim, every (hz, seed) on it.

    Returns `(rows, skip_reason)`. An arm whose constants trip a gpu_sim assert
    (the eLN-side <= 1 stability bound is the expected one at large normalised S)
    is SKIPPED, not fatal - and the reason is carried back so the verdict table
    shows the arm rather than silently dropping it.
    """
    rows = []
    try:
        set_consts(**DEFAULTS)
        set_consts(**consts)
        try:
            sim = G.GpuSim()
        except AssertionError as e:
            emit("%-13s SKIPPED: %s" % (label, e))
            return rows, str(e)
        try:
            for hz in rates:
                for seed in seeds:
                    act = run(sim, float(hz), seed)
                    rows.append({"arm": label, "hz": hz, "seed": seed,
                                 "alpn_j": jac(act[0, ALPN], act[1, ALPN]),
                                 "kc_j": jac(act[0, KC], act[1, KC]),
                                 "kc": (float(act[0, KC].sum()) / N_KC,
                                        float(act[1, KC].sum()) / N_KC),
                                 "cen": (float(act[0, CEN].sum()) / N_CEN,
                                         float(act[1, CEN].sum()) / N_CEN)})
                    emit("%-13s %5d %s  seed%d" % (
                        label, hz, " ".join("%16s" % c for c in row_cells(act)),
                        seed))
        finally:
            del sim
            torch.cuda.empty_cache()
    finally:
        set_consts(**DEFAULTS)
    return rows, None


def verdict(rows):
    """Metric -> (value, pass_bool_or_None). None = not measured in this run."""
    score = [r for r in rows if r["hz"] == SCORE_HZ]
    ign = [r for r in rows if r["hz"] == IGNITION_HZ]
    v = {}
    if score:
        aj = float(np.mean([r["alpn_j"] for r in score]))
        kj = float(np.mean([r["kc_j"] for r in score]))
        kc = [f for r in score for f in r["kc"]]
        cen = float(np.mean([f for r in score for f in r["cen"]]))
        lo, hi = CRIT1["kc_act"]
        clo, chi = CRIT1["central"]
        v["alpn_j"] = (aj, aj <= CRIT1["alpn_j"])
        v["kc_j"] = (kj, kj <= CRIT1["kc_j"])
        v["kc_act"] = ((min(kc), max(kc)), min(kc) >= lo and max(kc) <= hi)
        v["central"] = (cen, clo <= cen <= chi)
    else:
        for m in ("alpn_j", "kc_j", "kc_act", "central"):
            v[m] = (None, None)
    if ign:
        c = max(f for r in ign for f in r["cen"])
        v["cen120"] = (c, c < CRIT1["cen120"])
    else:
        v["cen120"] = (None, None)
    return v


def _band_dist(metric, val):
    """How far outside its band this metric sits, as a fraction of the bound.

    Used ONLY to rank failing arms. Ranking on the raw Jaccard would reward a
    silenced layer, whose empty union makes `jac` return 0.0.
    """
    if metric in ("alpn_j", "kc_j", "cen120"):
        return (val - CRIT1[metric]) / CRIT1[metric]
    lo, hi = CRIT1[metric]
    v0, v1 = val if isinstance(val, (tuple, list)) else (val, val)
    return max(0.0, (lo - v0) / lo) + max(0.0, (v1 - hi) / hi)


def _dist(v):
    return sum(_band_dist(m, v[m][0]) for m in _METRICS if v[m][1] is False)


def _mode(v):
    """dead / saturated / merged, from which metric failed and on which side."""
    tags = []
    if v["kc_act"][1] is False:
        lo, hi = CRIT1["kc_act"]
        if v["kc_act"][0][0] < lo:
            tags.append("dead")
        if v["kc_act"][0][1] > hi:
            tags.append("saturated")
    if v["alpn_j"][1] is False or v["kc_j"][1] is False:
        tags.append("merged")
    return "+".join(tags) or "-"


def _state(v):
    miss = [m for m in _METRICS if v[m][1] is False]
    na = [m for m in _METRICS if v[m][1] is None]
    if na:
        s = "INCOMPLETE(%s)" % ",".join(na)
        return s + (" MISS(%s)" % ",".join(miss) if miss else "")
    return "MISS(%s)" % ",".join(miss) if miss else "PASS"


def _fmt(name, val):
    if val is None:
        return "%11s" % "n/a"
    if name == "kc_act":
        return "%11s" % ("%.3f-%.3f" % val)
    return "%11.3f" % val


def table(verdicts):
    """`verdicts` is a list of (label, verdict_dict_or_None); None = SKIPPED."""
    lo, hi = CRIT1["kc_act"]
    clo, chi = CRIT1["central"]
    emit("")
    emit("VERDICT (%d Hz unless stated; ALPN J <= %.2f, KC J <= %.2f, KC act "
         "%.2f-%.2f both odours every seed, central %.2f-%.2f, %d Hz central "
         "< %.2f)" % (SCORE_HZ, CRIT1["alpn_j"], CRIT1["kc_j"], lo, hi, clo,
                      chi, IGNITION_HZ, CRIT1["cen120"]))
    emit("%-13s %11s %11s %11s %11s %11s  %s" % (
        "arm", "ALPN J", "KC J", "KC act", "central",
        "%dHz cen" % IGNITION_HZ, "verdict"))
    for label, v in verdicts:
        if v is None:
            emit("%-13s %s  SKIPPED(%s)" % (
                label, " ".join("%11s" % "-" for _ in _METRICS),
                SKIPPED.get(label, "no reason recorded")))
            continue
        emit("%-13s %s  %s" % (label, " ".join(
            _fmt(m, v[m][0]) for m in _METRICS), _state(v)))


def pick(verdicts):
    """Design decision 5, plus the honest cases the plan did not enumerate."""
    emit("")
    live = [(l, v) for l, v in verdicts if v is not None]
    if not live:
        emit("NO VERDICT: every arm was skipped")
        return
    full = [(l, v) for l, v in live if all(v[m][1] is True for m in _METRICS)]
    if full:
        best = min(full, key=lambda lv: (round(lv[1]["alpn_j"][0], 3),
                                         abs(np.mean(lv[1]["kc_act"][0]) - 0.05)))
        emit("WORKING POINT: %s (ALPN J %.3f, KC J %.3f, KC act %.3f-%.3f, "
             "central %.3f, %d Hz central %.3f)"
             % (best[0], best[1]["alpn_j"][0], best[1]["kc_j"][0],
                best[1]["kc_act"][0][0], best[1]["kc_act"][0][1],
                best[1]["central"][0], IGNITION_HZ, best[1]["cen120"][0]))
        return
    # An arm that failed nothing it was measured on has not been judged: say so
    # rather than calling a partial run a negative result.
    clean = [(l, v) for l, v in live
             if not any(v[m][1] is False for m in _METRICS)]
    if clean:
        l, v = clean[0]
        na = [m for m in _METRICS if v[m][1] is None]
        emit("INCOMPLETE: %s passes all measured metrics; %s not measured"
             % (l, ",".join(na)))
        return
    scored = [(l, v) for l, v in live if v["alpn_j"][0] is not None]
    bad = [l for l, v in scored if v["kc_act"][0][0] == 0.0]
    for l in bad:
        emit("NOT RANKED: %s has no active KCs for one odour; jac() returns 0.0 "
             "on an empty union, so its separation numbers are not evidence"
             % l)
    scored = [(l, v) for l, v in scored if l not in bad]
    if not scored:
        emit("NO ARM PASSES; no arm is rankable")
        return
    # Fewest missed metrics first, then least far outside the bands. Never the
    # raw Jaccard: see _band_dist.
    l, v = min(scored, key=lambda lv: (
        sum(1 for m in _METRICS if lv[1][m][1] is False), _dist(lv[1])))
    miss = [m for m in _METRICS if v[m][1] is False]
    na = [m for m in _METRICS if v[m][1] is None]
    if na:
        emit("INCOMPLETE: best %s fails on %s (failure mode %s); %s not "
             "measured, so no arm can be judged PASS"
             % (l, ",".join(miss), _mode(v), ",".join(na)))
        return
    emit("NO ARM PASSES; best %s fails on %s; failure mode %s"
         % (l, ",".join(miss), _mode(v)))


def score(rows, skipped, order):
    verdicts = []
    for label in order:
        if label in skipped:
            SKIPPED[label] = skipped[label]
            verdicts.append((label, None))
            continue
        rs = [r for r in rows if r["arm"] == label]
        if rs:
            verdicts.append((label, verdict(rs)))
    table(verdicts)
    pick(verdicts)


def rescore(path):
    with open(path) as fh:
        blob = json.load(fh)
    for r in blob["rows"]:                       # JSON has no tuples
        r["kc"], r["cen"] = tuple(r["kc"]), tuple(r["cen"])
    emit("rescored from %s" % path)
    emit("provenance %s" % json.dumps(blob["provenance"], sort_keys=True))
    score(blob["rows"], blob.get("skipped", {}), blob["order"])


def main(rates=(SCORE_HZ, IGNITION_HZ), seeds=(0, 2, 4),
         out="results/gap_sweep.txt", arm_list=None):
    global _OUT
    assert SCORE_HZ in rates, \
        "SCORE_HZ=%d is not in --hz %s; criterion 1 is defined at that rate, " \
        "so there would be nothing to score" % (SCORE_HZ, list(rates))
    path = out if os.path.isabs(out) else os.path.join(_HERE, out)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    _OUT = open(path, "w")
    try:
        prov = provenance()
        emit("provenance %s" % json.dumps(prov, sort_keys=True))
        if IGNITION_HZ not in rates:
            emit("NOTE: IGNITION_HZ=%d is not in --hz %s, so the no-ignition "
                 "metric is not measured and no arm can be judged PASS"
                 % (IGNITION_HZ, list(rates)))
        km = kmax()
        emit("kmax %.6g   KC n %d   central n %d   rates %s   seeds %s"
             % (km, N_KC, N_CEN, list(rates), list(seeds)))
        emit("%-13s %5s %s  %s" % ("arm", "hz",
                                   " ".join("%16s" % n for n in SHOW), "seed"))
        al = arm_list if arm_list is not None else arms(km)
        rows, skipped, order = [], {}, []
        for label, consts in al:
            order.append(label)
            rs, why = run_arm(label, consts, rates, seeds)
            rows += rs
            if why is not None:
                skipped[label] = why
        with open(path + ".json", "w") as fh:
            json.dump({"provenance": prov, "kmax": km, "rates": list(rates),
                       "seeds": list(seeds), "order": order,
                       "skipped": skipped, "rows": rows}, fh, indent=1)
        score(rows, skipped, order)
    finally:
        _OUT.close()
        _OUT = None


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--hz", type=int, nargs="+", default=[SCORE_HZ, IGNITION_HZ])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 2, 4])
    p.add_argument("--out", default="results/gap_sweep.txt")
    p.add_argument("--rescore", help="a <out>.json from an earlier run; "
                                     "re-scores it without touching the GPU")
    a = p.parse_args()
    if a.rescore:
        rescore(a.rescore)
    else:
        main(tuple(a.hz), tuple(a.seeds), a.out)
