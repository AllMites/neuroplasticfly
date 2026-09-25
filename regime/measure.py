"""Criterion-1 evidence: sparsity and odour separation under each mechanism.

Six arms isolate the three AUTHORED mechanisms - off (the shipped regime, the
control), sfa, apl, norm - plus `sfa+apl` (the two gain-control mechanisms) and
`all`. Each arm sets the gpu_sim constants in-process, builds a fresh GpuSim
(normalisation is baked into the weight tensor in __init__, so a sim cannot be
reused across arms) and runs the same ORN sweep on both channels.

Three things this differs from the plan text in `docs/superpowers/plans/
2026-09-19-regime-stability-p0-p1.md`:

1. The normalisation constant is `NORM_TOTAL_TARGET` (29.15), not
   `NORM_EXC_TARGET` (13.75). Both the name and the value in the plan are void.
   Setting an unknown attribute on a module does NOT raise, so the plan's code
   would have run the norm arm with normalisation OFF and reported a confident
   fake result. `set_consts` asserts the attribute already exists before writing it.
2. `APL_SCALE` is the depolarisation at which graded APL saturates, so LARGER
   means WEAKER inhibition. The plan's sweep (3.0/7.0/14.0) walks the wrong way
   and cannot reach a working point. The sweep here goes DOWN from 7.0.
3. Every graded-APL run reports the mean and max of the activation actually
   observed, because `APL_SCALE` and `APL_MAX_HZ` are entangled: a run whose
   activation sits at 0.05 is not a test of the mechanism at full strength.
   `_Sim` instruments `_deliver_apl` locally; gpu_sim.py is not touched.

Run: .venv/Scripts/python.exe regime/measure.py
"""
import json
import os
import sys

import torch

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _HERE)
import gpu_sim as G
from regime import probe as PR

SELECTORS = ["cell_type=ORN_DA1,side=left", "cell_type=ORN_DM4,side=left"]
NORM = 29.15          # AUTHORED: 106 synapses (brain-wide median of summed
                      # incoming |weight|) * W_SYN. See gpu_sim.NORM_TOTAL_TARGET.
# (SFA_B_INC, APL_GRADED, NORM_TOTAL_TARGET)
ARMS = {"off":     (0.0, False, 0.0),
        "sfa":     (0.5, False, 0.0),
        "apl":     (0.0, True,  0.0),
        "norm":    (0.0, False, NORM),
        "sfa+apl": (0.5, True,  0.0),
        "all":     (0.5, True,  NORM)}
RATES = (20, 60, 120)

# Criterion-1 thresholds.
#
# Sparse and silenced look alike in a KC-active number and are opposite in
# meaning, so the discriminator here is NOT the KC number: it is whether the
# rest of the brain is still alive. `central_active` near zero means the drive
# never propagated and the mechanism reached its low KC count by killing the
# network - a FAILURE of criterion 1, not a pass. A low KC count with central
# still carrying activity is sparsity, which is what criterion 1 wants.
#
# DEAD is on central_active; KC_DEAD is the separate case of a live brain whose
# mushroom-body output has been extinguished, which is also not a pass.
DEAD = 0.02      # central_active below this on both channels -> network dead
KC_DEAD = 0.002  # kc_active below this with a live brain -> MB output dead
IGN_LO = 0.26
JAC_OK = 0.90


def set_consts(**kw):
    """Set gpu_sim module constants, refusing to invent new ones."""
    for name, val in kw.items():
        assert hasattr(G, name), "gpu_sim has no constant %r - typo or rename" % name
        setattr(G, name, val)


_set = set_consts   # historic name, kept so nothing that imports it breaks


class _Sim(G.GpuSim):
    """GpuSim that records the graded-APL activation it actually delivered."""

    def reset_act(self):
        z = torch.zeros((), dtype=torch.float64, device=self.device)
        self._act_sum = z.clone()
        self._act_max = z.clone()
        self._act_n = 0

    def _deliver_apl(self, g, act):
        self._act_sum = self._act_sum + act.sum().double()
        self._act_max = torch.maximum(self._act_max, act.max().double())
        self._act_n += act.numel()
        return super()._deliver_apl(g, act)

    def act_stats(self):
        if not getattr(self, "_act_n", 0):
            return (None, None)
        return (float(self._act_sum) / self._act_n, float(self._act_max))


def classify(rep):
    """ignited / silenced / KC-silent / in between / mixed, from one report.

    `mixed` is the honest label for a row where the two channels disagree - at
    20 Hz DA1 is below its 46.5 Hz ignition threshold while DM4 (1.15 Hz) is
    well above it, so every arm shows one dead channel and one live one there.
    It also catches a sparse row whose Jaccard has not moved.
    """
    kc = rep["kc_active"]
    cen = rep["central_active"]
    jac = rep["jac"]
    if max(cen) < DEAD:
        return "silenced"
    if max(kc) < KC_DEAD:
        return "KC-silent"
    if min(kc) >= IGN_LO and jac >= JAC_OK:
        return "ignited"
    if min(kc) >= KC_DEAD and max(kc) < IGN_LO and jac < JAC_OK:
        return "in between"
    return "mixed"


def run_arm(b_inc, graded, norm_target, rates=RATES, seed0=0,
            apl_scale=None, apl_max_hz=None, tau_a=None):
    _set(SFA_B_INC=float(b_inc), APL_GRADED=bool(graded),
         NORM_TOTAL_TARGET=float(norm_target))
    if apl_scale is not None:
        _set(APL_SCALE=float(apl_scale))
    if apl_max_hz is not None:
        _set(APL_MAX_HZ=float(apl_max_hz))
    if tau_a is not None:
        _set(TAU_A=float(tau_a))
    sim = _Sim()
    # Verify the norm arm is actually normalising. A silent no-op here is the
    # exact failure the plan's stale constant name would have produced.
    on = bool((sim.norm_scale != 1.0).any())
    assert on == (float(norm_target) > 0.0), \
        "norm_scale on=%s but NORM_TOTAL_TARGET=%r" % (on, norm_target)
    out = []
    for hz in rates:
        sim.reset_act()
        _, rep = PR.measure(sim, SELECTORS, hz, seed0=seed0)
        rep["jac"] = PR.pair_jaccard(rep)
        rep["apl_act_mean"], rep["apl_act_max"] = sim.act_stats()
        rep["seed0"] = seed0
        rep["const"] = {"SFA_B_INC": G.SFA_B_INC, "TAU_A": G.TAU_A,
                        "APL_GRADED": G.APL_GRADED, "APL_SCALE": G.APL_SCALE,
                        "APL_MAX_HZ": G.APL_MAX_HZ,
                        "NORM_TOTAL_TARGET": G.NORM_TOTAL_TARGET,
                        "ELN_NEGATE": G.ELN_NEGATE,
                        "GAP_COUPLE": G.GAP_COUPLE, "GAP_NORM": G.GAP_NORM}
        rep["verdict"] = classify(rep)
        out.append(rep)
        print(line(rep), flush=True)
    del sim
    torch.cuda.empty_cache()
    return out


def line(rep):
    am, ax = rep["apl_act_mean"], rep["apl_act_max"]
    act = "  act   -  /  -  " if am is None else "  act %.3f/%.3f" % (am, ax)
    return ("  hz %3d  KC %.4f/%.4f  KCHz %6.2f/%6.2f  cen %.4f/%.4f"
            "  APLHz %6.1f/%6.1f  jac %.3f%s  %s" % (
                rep["hz"], rep["kc_active"][0], rep["kc_active"][1],
                rep["kc_hz"][0], rep["kc_hz"][1],
                rep["central_active"][0], rep["central_active"][1],
                rep["apl_hz"][0], rep["apl_hz"][1], rep["jac"], act,
                rep["verdict"]))


def write(res):
    os.makedirs(os.path.join(_HERE, "results"), exist_ok=True)
    with open(os.path.join(_HERE, "results", "regime_p1.json"), "w") as fh:
        json.dump(res, fh, indent=1)


def main():
    defaults = {"SFA_B_INC": G.SFA_B_INC, "TAU_A": G.TAU_A,
                "APL_GRADED": G.APL_GRADED, "APL_SCALE": G.APL_SCALE,
                "APL_MAX_HZ": G.APL_MAX_HZ,
                "NORM_TOTAL_TARGET": G.NORM_TOTAL_TARGET,
                "ELN_NEGATE": G.ELN_NEGATE,
                "GAP_COUPLE": G.GAP_COUPLE, "GAP_NORM": G.GAP_NORM}
    res = {"gpu_sim_defaults": defaults, "arms": {}, "sfa_sweep": {},
           "apl_sweep": {}, "aplhz_sweep": {}, "seeds": {}}
    for name, (b_inc, graded, norm_target) in ARMS.items():
        print("arm %-8s SFA_B_INC %s  APL_GRADED %s  NORM_TOTAL_TARGET %s"
              % (name, b_inc, graded, norm_target), flush=True)
        res["arms"][name] = run_arm(b_inc, graded, norm_target)
        write(res)

    # SFA has never been run ON at any value. Same attention as the APL sweep.
    for b in (1.0, 2.0, 4.0):
        key = "sfa SFA_B_INC=%s" % b
        print(key, flush=True)
        res["sfa_sweep"][key] = run_arm(b, False, 0.0)
        write(res)

    # APL_SCALE: SMALLER is STRONGER (it is the depolarisation at which the
    # activation saturates). 7.0 is the default and measures near baseline.
    for s in (3.0, 1.5, 0.75, 0.35):
        key = "apl APL_SCALE=%s" % s
        print(key, flush=True)
        res["apl_sweep"][key] = run_arm(0.0, True, 0.0, apl_scale=s)
        _set(APL_SCALE=defaults["APL_SCALE"])
        write(res)

    # The other half of the entanglement. At APL_SCALE 7.0 the measured mean
    # activation is already ~0.85 of the [0,1] ceiling, so the whole APL_SCALE
    # axis is worth at most ~1.2x more inhibition. APL_MAX_HZ is the axis that
    # actually spans baseline to the silence seen at ~22x over-strength.
    for f in (600.0, 750.0, 900.0, 1100.0, 1350.0, 1800.0, 3600.0, 7200.0):
        key = "apl APL_MAX_HZ=%s" % f
        print(key, flush=True)
        res["aplhz_sweep"][key] = run_arm(0.0, True, 0.0, apl_max_hz=f)
        _set(APL_MAX_HZ=defaults["APL_MAX_HZ"])
        write(res)

    # The more promising combination: `all` is silenced by normalisation, so the
    # only live pairing is sfa+apl.
    for f in (600.0, 900.0, 1350.0):
        key = "sfa+apl APL_MAX_HZ=%s" % f
        print(key, flush=True)
        res["aplhz_sweep"][key] = run_arm(0.5, True, 0.0, apl_max_hz=f)
        _set(APL_MAX_HZ=defaults["APL_MAX_HZ"])
        write(res)

    # `in between` is the first such observation in this investigation, so any
    # configuration that shows it gets three more seeds.
    for key, rows in list(res["aplhz_sweep"].items()):
        if not any(r["verdict"] == "in between" for r in rows):
            continue
        c = rows[0]["const"]
        print("seeds", key, flush=True)
        res["seeds"][key] = [r for r in rows if r["hz"] in (60, 120)] + \
            confirm_seeds(key, rates=(60, 120), b_inc=c["SFA_B_INC"],
                          graded=c["APL_GRADED"], norm_target=c["NORM_TOTAL_TARGET"],
                          apl_max_hz=c["APL_MAX_HZ"])
        _set(APL_MAX_HZ=defaults["APL_MAX_HZ"])
        write(res)
    print("done")


def confirm_seeds(label, seeds=(2, 4, 6), rates=(60,), **kw):
    """Re-run one configuration at further seeds. Used when a row lands
    'in between' - the first such observation in this investigation - to check
    it is not a one-draw artifact."""
    out = []
    for s in seeds:
        print("%s seed0=%d" % (label, s), flush=True)
        out += run_arm(rates=rates, seed0=s, **kw)
    return out


if __name__ == "__main__":
    main()
