"""Signal locator: where along a pathway does a stimulus signal get lost?

Stimulus groups in, per-stage slope + label out; the label rules come from a preregistration file.
One API for the shape every per-experiment probe in this repository shares (regime/layers.py,
regime/alln_probe*.py, snr_trials.py, where_locality_dies.py): groups of stimuli -> rates per stage
(cell groups along a pathway) -> one statistic per stage -> the stage where it collapses.
Those scripts are kept as run; this module does not change them. docs/probes.md has a worked example.

Inputs are plain numpy arrays, so it runs on CPU without the brain:
  x       [G]     signed stimulus contrast per group, e.g. left-minus-right drive in Hz (-10, 0, +10),
                  or 0 / 1 for "odour A" vs "odour B"
  stages  {name: y [G, S]}  one readout per stage, per group and seed, e.g. left-minus-right mean rate
                  (readout() builds it from rates [G, S, N]); simulate() is the optional GPU path

Per stage: per-seed OLS slope of y on x, its mean, SEM across seeds, t = slope / SEM, and the
variance SNR of snr_trials.decompose (signal across groups / noise across seeds).

Label rules: a ```locator block in a PREREGISTER_*.md (format in docs/PREREGISTER_TEMPLATE.md):
  stages: ORN > uPN > LHN | KC > DNa02   pathway order; "a | b" = parallel branches, either may carry it
  t_min: 2                               a stage carries the signal if |slope| >= t_min * SEM ...
  silent_hz: 0.1                         ... and it is not silent (mean rate below this) ...
  k: 4                                   ... and (optional) >= k seeds have the slope's sign (prereg.holds)
Labels: "CARRIED TO <last>", "LOST AT <stage>" (+ " (SILENT)" when every cell group there is silent),
"INSTRUMENT FAIL (<first>)" when the stimulated stage itself does not carry it.
"""
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import prereg as P  # noqa: E402  (holds, sha_lf)

DEFAULTS = {"t_min": 2.0, "silent_hz": 0.1, "k": None}
_TYPES = {"stages": str, "t_min": float, "silent_hz": float, "k": int}


def read_rules(path):
    """Parse the ```locator block of a preregistration -> rules dict. Unknown keys are refused."""
    text = open(path, encoding="utf-8").read()
    m = re.search(r"^```locator[ \t]*\r?\n(.*?)^```", text, re.S | re.M)
    if not m:
        raise ValueError("%s has no ```locator block" % path)
    rules = dict(DEFAULTS)
    for line in m.group(1).splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        key, sep, val = (s.strip() for s in line.partition(":"))
        if not sep or key not in _TYPES:
            raise ValueError("%s: unknown locator rule %r (keys: %s)" % (path, line, ", ".join(_TYPES)))
        rules[key] = _TYPES[key](val)
    if "stages" not in rules:
        raise ValueError("%s: the locator block needs a 'stages:' line" % path)
    rules["stages"] = [[s.strip() for s in step.split("|")] for step in rules["stages"].split(">")]
    rules["prereg"], rules["prereg_sha256"] = path, P.sha_lf(path)
    return rules


def slope(y, x):
    """y [G, S], x [G] -> per-seed OLS slope of y on x: {slope, sem, t, per_seed}."""
    y, x = np.asarray(y, np.float64), np.asarray(x, np.float64)
    assert y.ndim == 2 and y.shape[0] == x.size, "y must be [groups, seeds] with one x per group"
    xc = x - x.mean()
    per = xc @ (y - y.mean(0)) / (xc @ xc)
    m = float(per.mean())
    sem = float(per.std(ddof=1) / np.sqrt(per.size)) if per.size > 1 else float("nan")
    t = m / sem if sem > 0 else (0.0 if m == 0 else float(np.copysign(np.inf, m)))
    return {"slope": m, "sem": sem, "t": t, "per_seed": per.tolist()}


def snr(y):
    """Variance SNR of y [G, S] (snr_trials.decompose on one readout): signal across groups / noise across seeds."""
    y = np.asarray(y, np.float64)
    noise = float(y.var(axis=1, ddof=1).mean())
    signal = float(y.mean(axis=1).var(ddof=1) - noise / y.shape[1])
    return signal / noise if noise > 0 else float("inf")


def readout(rates, plus, minus=None):
    """rates [G, S, N] Hz -> (y [G, S], mean_hz). y = mean over `plus` cells minus mean over `minus` cells
    (left minus right for a side signal; minus=None for an identity signal)."""
    rates = np.asarray(rates, np.float64)
    ix = lambda a: np.flatnonzero(a) if np.asarray(a).dtype == bool else np.asarray(a).ravel()
    cells = plus = ix(plus)
    y = rates[..., plus].mean(-1)
    if minus is not None:
        minus = ix(minus)
        y = y - rates[..., minus].mean(-1)
        cells = np.concatenate([plus, minus])
    return y, float(rates[..., cells].mean())


def carries(r, rules):
    """One stage's result -> does it carry the signal under the preregistered rules?"""
    if r.get("silent"):
        return False
    ok = abs(r["slope"]) >= rules["t_min"] * r["sem"] and r["slope"] != 0
    if ok and rules.get("k") is not None:
        ok = P.holds([np.sign(p) == np.sign(r["slope"]) for p in r["per_seed"]], rules["k"])
    return bool(ok)


def locate(stages, x, rules, mean_hz=None):
    """stages {name: y [G, S]}, x [G], rules (dict from read_rules, or a preregistration path),
    mean_hz {name: Hz} (optional, for the silence check) -> {"label", "lost_at", "stages", "rules"}."""
    if isinstance(rules, str):
        rules = read_rules(rules)
    mean_hz = mean_hz or {}
    res = {}
    for name, y in stages.items():
        r = slope(y, x)
        r["snr"] = snr(y)
        r["mean_hz"] = mean_hz.get(name)
        r["silent"] = r["mean_hz"] is not None and r["mean_hz"] < rules["silent_hz"]
        r["carries"] = carries(r, rules)
        res[name] = r
    missing = [s for step in rules["stages"] for s in step if s not in res]
    if missing:
        raise KeyError("preregistered stages without data: %s" % ", ".join(missing))
    first = rules["stages"][0]
    if first[0] in res and res[first[0]]["slope"]:
        for r in res.values():
            r["ratio_to_first"] = r["slope"] / res[first[0]]["slope"]
    label, lost = "CARRIED TO " + " | ".join(rules["stages"][-1]), None
    for i, step in enumerate(rules["stages"]):
        if any(res[s]["carries"] for s in step):
            continue
        lost = " | ".join(step)
        silent = " (SILENT)" if all(res[s]["silent"] for s in step) else ""
        label = ("INSTRUMENT FAIL (%s)" if i == 0 else "LOST AT %s") % (lost + silent)
        break
    return {"label": label, "lost_at": lost, "stages": res, "rules": rules}


def table(out):
    """Plain-ASCII per-stage table of a locate() result."""
    rows = ["%-14s %9s %9s %7s %8s %8s  %s" % ("stage", "slope", "sem", "t", "snr", "meanHz", "carries")]
    for name, r in out["stages"].items():
        hz = "-" if r["mean_hz"] is None else "%.2f" % r["mean_hz"]
        rows.append("%-14s %+9.4f %9.4f %+7.1f %8.3f %8s  %s%s" % (
            name, r["slope"], r["sem"], r["t"], r["snr"], hz, "yes" if r["carries"] else "no",
            " SILENT" if r["silent"] else ""))
    rows.append("LABEL: " + out["label"])
    return "\n".join(rows)


def simulate(sim, drives, seeds, t_run=2000.0, warm=200.0):
    """Optional GPU path (needs data/ and a gpu_sim.GpuSim): one (stim_idx, stim_prob) drive per stimulus
    group, each run at every seed -> rates [G, S, N] in Hz. The same seed list across groups gives common
    random numbers, as regime/probe.py does. `warm` ms run first and discarded, the state carried on."""
    batch = [d for d in drives for _ in seeds]
    bseeds = [s for _ in drives for s in seeds]
    state = None
    if warm:
        _, state = sim.run_batch(batch, bseeds, t_run=warm, state=None, return_state=True)
    counts, _ = sim.run_batch(batch, bseeds, t_run=t_run, state=state, return_state=True)
    rates = counts.cpu().numpy().astype(np.float64) / (t_run / 1000.0)
    return rates.reshape(len(drives), len(seeds), -1)
