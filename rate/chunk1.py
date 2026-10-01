"""Chunk 1 (sugar -> PAM): readout sets, the 11-row battery, and the measurement.

Moved out of chunk1_control.py (975464e) unchanged, so the regression suite (rate/suite.py) and the
control run share one definition. Sets: rate/chunk1_targets.md. Rules: PREREGISTER_rate_chunk1_control.md.
"""
import os

import numpy as np
import torch

from learn.christie_sweep import FOX, FDA_I, FDA_II  # same sets as the LIF (rule 14)

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HZ = 150.0
T_RUN = 1000.0
UP, NONE = 1.0, 0.5
COMP = {"g4": ["PAM07", "PAM08"], "g5": ["PAM01", "PAM15"], "b2": ["PAM04", "PAM03"],
        "b'2": ["PAM02", "PAM05", "PAM06", "PAM03", "PAM15"], "g3": ["PAM12"],
        "b1": ["PAM09", "PAM10"], "b'1": ["PAM13", "PAM14"], "g1": ["PPL101"], "g2": ["PPL102"]}
R_COMP = ["g4", "g5", "b2", "b'2"]
COMP["R"] = sorted({t for c in R_COMP for t in COMP[c]})
COMP["g12"] = ["PPL101", "PPL102"]
TYPES = ["PAM%02d" % i for i in range(1, 16)] + ["PPL10%d" % i for i in range(1, 7)]
CONDS = ["BASE", "FOX", "SUGAR", "BITTER", "SUGAR_FOXSIL"]


def lif_constants():
    """Prereg arm A: least-squares slope of the Shiu LIF f-I curve through threshold, 0-200 Hz."""
    tm, tr, th = 20.0, 2.2, 7.0
    f = lambda dv: 1000.0 / (tr + tm * np.log(dv / (dv - th)))  # noqa: E731
    lo, hi = th + 1e-9, th + 200.0
    for _ in range(200):
        m = (lo + hi) / 2
        lo, hi = (m, hi) if f(m) < 200 else (lo, m)
    dv = np.linspace(th + 1e-6, hi, 20001)
    x = dv - th
    k = float((x * f(dv)).sum() / (x * x).sum())
    return 0.275 * 0.005 * k, -th * k, 1000.0 / tr


def status(d):
    return "up" if d >= UP else ("none" if d < NONE else "ambiguous")


def battery(D):
    """D[cond][comp] -> {row: (passed, detail)}. Prereg rows F1-F5, H1-H6."""
    r = {}
    for i, c in enumerate(R_COMP, 1):
        r["F%d" % i] = (status(D["FOX"][c]) == "up", "%s FOX %.3f" % (c, D["FOX"][c]))
    r["F5"] = (status(D["FOX"]["b1"]) == "none", "b1 FOX %.3f" % D["FOX"]["b1"])
    r["H1"] = (all(status(D["SUGAR"][c]) == "up" for c in ("g4", "g5")),
               "g4 %.3f g5 %.3f" % (D["SUGAR"]["g4"], D["SUGAR"]["g5"]))
    r["H2"] = (all(status(D["SUGAR"][c]) == "up" for c in ("b2", "b'2")),
               "b2 %.3f b'2 %.3f" % (D["SUGAR"]["b2"], D["SUGAR"]["b'2"]))
    r["H3"] = (not any(status(D["BITTER"][c]) == "up" for c in R_COMP),
               " ".join("%s %.3f" % (c, D["BITTER"][c]) for c in R_COMP))
    s, sil = D["SUGAR"]["R"], D["SUGAR_FOXSIL"]["R"]
    r["H4"] = ((False, "NOT TESTABLE: D_R(SUGAR) %.3f < %.1f" % (s, UP)) if s < UP else
               (sil <= 0.5 * s, "D_R SUGAR %.3f, Fox silenced %.3f" % (s, sil)))
    r["H5"] = (status(D["FOX"]["g3"]) == "none", "g3 FOX %.3f" % D["FOX"]["g3"])
    b, g = D["BITTER"]["g12"], D["SUGAR"]["g12"]
    r["H6"] = (b - g >= UP and b >= UP, "g1ug2 BITTER %.3f SUGAR %.3f" % (b, g))
    return r


def selftest():
    ok = {c: 5.0 for c in COMP}
    D = {"FOX": dict(ok, b1=0.0, g3=0.0), "SUGAR": dict(ok, g12=0.0),
         "BITTER": dict({c: 0.0 for c in COMP}, g12=5.0), "SUGAR_FOXSIL": dict(ok, R=1.0)}
    assert all(p for p, _ in battery(D).values()), battery(D)
    D["SUGAR"] = dict(D["SUGAR"], R=0.7)
    assert battery(D)["H4"][0] is False and "NOT TESTABLE" in battery(D)["H4"][1]
    D["FOX"] = dict(D["FOX"], b1=0.7)  # ambiguous fails "none"
    assert battery(D)["F5"][0] is False


def sets():
    meta = np.load(os.path.join(_HERE, "data", "neuron_meta.npz"), allow_pickle=False)
    ct = meta["cell_type"].astype(str)
    csc = meta["cell_sub_class"].astype(str)
    fox = np.flatnonzero(np.isin(ct, FOX)).astype(np.int64)
    S = {"ct": ct,
         "sugar": np.flatnonzero(csc == "sugar/water").astype(np.int64),
         "bitter": np.flatnonzero(csc == "bitter").astype(np.int64),
         "fox": fox,
         "comp": {c: np.flatnonzero(np.isin(ct, ts)) for c, ts in COMP.items()},
         "type": {t: np.flatnonzero(ct == t) for t in TYPES},
         "fda": {"fox": fox, "fdaI": np.flatnonzero(np.isin(ct, FDA_I)),
                 "fdaII": np.flatnonzero(np.isin(ct, FDA_II))}}
    assert len(fox) == 2, "G0: Fox (CB0525) should be 2 neurons, got %d" % len(fox)
    for name, s in [("sugar", S["sugar"]), ("bitter", S["bitter"])] + list(S["comp"].items()):
        assert len(s), "G2: empty set %s" % name
    assert not np.intersect1d(S["sugar"], fox).size, "G2: Fox silence overlaps sugar drive"
    return S


def measure(sim, S):
    """Five prereg conditions on `sim` as configured -> (rates per cond [N], D[cond][comp])."""
    empty = np.zeros(0, np.int64)
    with torch.no_grad():
        r4 = sim.run([(empty, 0.0), (S["fox"], HZ), (S["sugar"], HZ), (S["bitter"], HZ)], T_RUN)
        r1 = sim.run([(S["sugar"], HZ)], T_RUN, silence=S["fox"])
    rates = dict(zip(CONDS, [x.cpu().double().numpy() for x in list(r4) + [r1[0]]]))
    D = {k: {c: float((v[i] - rates["BASE"][i]).mean()) for c, i in S["comp"].items()}
         for k, v in rates.items() if k != "BASE"}
    return rates, D
