"""Rung 3: persistence, decay and interference on ONE brain carrying its weights.

Three phases, run back to back without ever rebuilding the plastic state:

  A  train DC2 with punishment          (30 trials)  - the association forms
  B  relax: no stimulus, no US, rule ON (30 trials)  - recovery only, measures DECAY
  C  train D with punishment            (30 trials)  - measures INTERFERENCE with A

Phase B is what makes decay measurable at all. `Plastic.update` applies the
recovery term `lam * (w0 - w)` unconditionally and the depression term is
`-eta * e * d * w0`, so on a trial with no US the punish DANs sit at baseline,
d ~ 0, and only recovery runs. condition.py calls update ONLY on US trials
(`if us is not None`), so in Track 2 weights could never decay on their own.
Here the relax phase calls update with no drive and no US on purpose.

Arms:
  retain     A, B        - decay alone, the reference for phase C
  interfere  A, B, C     - decay plus interference
  lesion     A, B, C with the rule off - seed noise alone

The fourth control needs no run: Track 2's `reversed` arm already trained D on a
NAIVE brain and moved DC2 by +1.74 Hz. Any DC2 movement during phase C has to be
read against that, not against zero.

FALSIFIERS, fixed before the run:
  0. Harness. Phase A must reproduce Track 2: DC2 +11.05 +- 0.20 Hz at t30. If it
     does not, stop - the harness is broken, not the science.
  1. Persistence. After phase B, DC2 must retain > 50% of its end-of-A shift. If
     an untrained interval returns it to baseline, weights do not persist and the
     accumulating-brain claim dies here.
  2. Accumulation. Phase C must still form the second association: D reaches a
     shift comparable to Track 2's naive +11.29. If D cannot learn on an
     already-trained brain, the brain is saturated, not accumulating - and that
     is the result that matters most.
  3. Interference. DC2 loss during C, compared with the loss over an equal-length
     B. Greater -> real interference. Equal -> the two associations are
     independent. Either is a result; neither is a failure.
  4. lesion ~ 0 throughout.

Run:
  .venv/Scripts/python.exe learn/persist.py --state p1s0 --seed0 0
"""
import argparse
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "flywatch"))
import gpu_sim as G
from learn import condition as C
from learn import plastic as PL

RESULTS = os.path.join(_HERE, "results")
STATES = os.path.join(_HERE, "brain_state")

CS_A = "dc2"          # trained in phase A
CS_C = "d"            # trained in phase C
PROBES = ("dc2", "d", "da1")
US = "punish"         # the only US the anatomy supports; injected at the DANs

ARMS = {
    "retain":    {"lesion": "none", "phase_c": False},
    "interfere": {"lesion": "none", "phase_c": True},
    "lesion":    {"lesion": "all",  "phase_c": True},
}


def run_arm(sim, name, arm, a, log):
    P = PL.Plastic.real()
    P.push(sim)
    rows = []

    def probe(phase, k):
        for s in PROBES:
            for j in range(a.probe):
                rates = C.trial(sim, s, seed=a.seed0 + 1000 * k + j)
                r = C.readout(rates, P)
                r.update(arm=name, phase=phase, trial=k, song=s, us=None, seed0=a.seed0)
                rows.append(r)
                log.write(json.dumps(r) + "\n")

    def train(cs, k0, n, phase):
        for k in range(1, n + 1):
            rates = C.trial(sim, cs, us=US, us_path="dan", seed=a.seed0 + 7 * (k0 + k))
            P.update(rates, eta=a.eta, lam=a.lam, lesion=arm["lesion"])
            P.push(sim)
            r = C.readout(rates, P)
            r.update(arm=name, phase=phase, trial=k0 + k, song=cs, us=US, seed0=a.seed0,
                     **{"w_" + x: y for x, y in P.summary().items()})
            rows.append(r)
            log.write(json.dumps(r) + "\n")
            if k % 5 == 0 or k == n:
                probe(phase + "_test", k0 + k)

    def relax(k0, n):
        """No stimulus, no US, rule ON. Depression term is ~0, recovery term runs."""
        for k in range(1, n + 1):
            # Empty drive: the brain runs on its own activity for 300 ms.
            counts = sim.run_batch([(np.zeros(0, np.int64), np.zeros(0, np.float64))],
                                   [a.seed0 + 7 * (k0 + k)], t_run=C.T_RUN)
            rates = counts[0].cpu().numpy().astype(np.float32) / (C.T_RUN / 1000.0)
            P.update(rates, eta=a.eta, lam=a.lam, lesion=arm["lesion"])
            P.push(sim)
            r = C.readout(rates, P)
            r.update(arm=name, phase="relax", trial=k0 + k, song=None, us=None, seed0=a.seed0,
                     **{"w_" + x: y for x, y in P.summary().items()})
            rows.append(r)
            log.write(json.dumps(r) + "\n")
            if k % 5 == 0 or k == n:
                probe("relax_test", k0 + k)

    probe("baseline", 0)
    train(CS_A, 0, a.train, "trainA")
    relax(a.train, a.relax)
    if arm["phase_c"]:
        train(CS_C, a.train + a.relax, a.train, "trainC")
    return P, rows


def curves(rows):
    """Mean avoid_index per (arm, stimulus, trial) over probe phases only."""
    out = {}
    for r in rows:
        if r["phase"].endswith("test") or r["phase"] == "baseline":
            out.setdefault(r["arm"], {}).setdefault(r["song"], {}).setdefault(
                r["trial"], []).append(r["avoid_index"])
    return {a: {s: {int(t): [float(np.mean(v)), float(np.std(v))] for t, v in d.items()}
                for s, d in sd.items()} for a, sd in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required=True)
    ap.add_argument("--train", type=int, default=30)
    ap.add_argument("--relax", type=int, default=30, help="length of the untrained interval")
    ap.add_argument("--probe", type=int, default=3)
    ap.add_argument("--arms", default="retain,interfere,lesion")
    ap.add_argument("--eta", type=float, default=5e-6)
    ap.add_argument("--lam", type=float, default=PL.LAM)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--regime", default="eln8", choices=sorted(C.REGIMES))
    a = ap.parse_args()

    G.ELN_NEGATE, G.PN_KC_GAIN = C.REGIMES[a.regime]
    print("regime %s (ELN_NEGATE=%s PN_KC_GAIN=%.1f), A=%s C=%s, train %d relax %d"
          % (a.regime, G.ELN_NEGATE, G.PN_KC_GAIN, CS_A, CS_C, a.train, a.relax), flush=True)
    sim = G.GpuSim()
    os.makedirs(RESULTS, exist_ok=True)
    rows, t0 = [], time.time()
    with open(os.path.join(RESULTS, "persist_%s.jsonl" % a.state), "w") as log:
        for name in a.arms.split(","):
            P, r = run_arm(sim, name, ARMS[name], a, log)
            rows += r
            print("arm %s done %.0fs" % (name, time.time() - t0), flush=True)
    out = os.path.join(RESULTS, "persist_%s.json" % a.state)
    json.dump({"curves": curves(rows), "cs_a": CS_A, "cs_c": CS_C, "probes": list(PROBES),
               "args": vars(a)}, open(out, "w"), indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()
