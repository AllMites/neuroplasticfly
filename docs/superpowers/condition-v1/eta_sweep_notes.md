# Eta sweep: the step is structural, not a learning-rate choice (2026-09-20)

Condition v1 shipped at eta 1e-3 with a 30-trial curve that is a step: the
weights are already at their final value by the first probe. The obvious next
experiment was a lower eta. It was run at 3e-5, 1e-4 and 3e-4 against the
shipped 1e-3, and it does not buy a learning curve.

Reproduce: `python learn/analyze_eta.py`. Raw: `results/analyze_eta_2026-09-20.txt`,
`results/condition_eta*.json`, `brain_state/eta*/`, `brain_state/v2neg/` (= eta 1e-3).
All four runs are seed0 = 0, 30 training trials, 5 probes, lam 0.01, W_MIN 0.1,
us_path `dan`, same corrected connectome, same song hashes. Only eta differs.

## 1. The same 1,366 edges move at every eta

| eta | edges moved | same set | frac of w0: min / med / max | at floor |
|---|---|---|---|---|
| 3e-5 | 1,366 of 18,674 (7.3%) | yes | -0.891 / -0.882 / -0.010 | 445 |
| 1e-4 | 1,366 (7.3%) | yes | -0.891 / -0.882 / -0.035 | 445 |
| 3e-4 | 1,366 (7.3%) | yes | -0.891 / -0.882 / -0.107 | 445 |
| 1e-3 | 1,366 (7.3%) | yes | -0.891 / -0.882 / -0.392 | 445 |

A 33x change in eta moves the *set* not at all and the *median* mover not at
all: -0.882 of baseline in every run, i.e. essentially pinned at the W_MIN
floor. Only the max — the least-depressed edge in the tail — responds to eta.
Whatever eta is, the plastic layer ends in the same place.

## 2. Why 17,308 edges never move

Not the learning rate, and only partly the compartment table:

| MBON valence | edges | with a live opposing DAN | moved |
|---|---|---|---|
| approach | 10,158 | 5,756 (57%) | 486 |
| avoid | 8,516 | 8,516 (100%) | 880 |

The 2-char compartment prefix match kills 43% of the approach side (MBON09,
MBON10, MBON18, MBON19 — already documented in `learn/plastic.py`). But 14,282
edges have a live DAN path and still never move. The real bottleneck is
presynaptic: only **501 of 5,066 KCs** ever fire during the songs, because the
CS is a fly-hash graft with `SPARSITY = 0.05` per song. Three songs at 5% with
overlap is ~10% of the KC population, and the plastic layer can only touch the
edges leaving them. 41 of 47 MBONs are reached.

**The plastic layer's capacity is set by the graft, not by the rule.** Eta
cannot reach an edge whose presynaptic KC is silent.

## 3. The published "saturates on trial 1" line is right only for eta >= 3e-4

Mean weight fraction per training trial, learn arm:

| eta | t01 | t02 | t03 | t04 | t05 | t06 | t07 | t08 |
|---|---|---|---|---|---|---|---|---|
| 3e-5 | -0.0057 | -0.0129 | -0.0202 | -0.0273 | -0.0340 | -0.0400 | -0.0434 | -0.0442 |
| 1e-4 | -0.0191 | -0.0384 | -0.0440 | -0.0444 | -0.0444 | -0.0444 | -0.0445 | -0.0445 |
| 3e-4 | -0.0337 | -0.0445 | -0.0446 | -0.0447 | -0.0448 | -0.0449 | -0.0450 | -0.0452 |
| 1e-3 | -0.0343 | -0.0452 | -0.0456 | -0.0460 | -0.0463 | -0.0467 | -0.0470 | -0.0474 |

At 3e-5 the weights *are* graded: a roughly linear approach over ~7 trials
before the asymptote. At 1e-4 it is 3 trials. At 3e-4 and above it is 1 trial.

That graded window is invisible in the published figure because **the probe
schedule is every 5 trials**. By the trial-5 probe, 96-109% of the final
behavioural shift is already there at every eta including 3e-5. Part of the
"step" is a sampling artefact, not saturation.

Correct the claim in an internal project note: "weights saturate on
training trial 1" holds at the shipped eta, not at every eta. The honest line
is that no eta in this range produces a curve the current probe schedule can
resolve, and that at the one eta where the weights are graded the behaviour is
already at asymptote by the first probe anyway.

## 4. Lower eta makes the effect bigger and generalisation worse

Avoid-index shift, last probe minus baseline, learn arm:

| eta | misery (trained, punish) | skyhigh (trained, reward) | mozart (untrained) |
|---|---|---|---|
| 3e-5 | +6.928 | -5.354 | **+7.688** |
| 1e-4 | +6.743 | -5.354 | +7.577 |
| 3e-4 | +6.373 | -5.354 | +7.096 |
| 1e-3 | +6.077 | -5.354 | +5.568 |

Two things to carry into any claim:

- The skyhigh shift is **-5.354 at every eta, to four significant figures**.
  It is set entirely by which edges reach the floor, and that set is
  eta-invariant. It is not a learned magnitude.
- Untrained Mozart shifts **more than either trained song** at eta 3e-5
  (+7.688 vs +6.928). The known broad-generalisation problem gets worse as eta
  falls. Lowering eta is not a free improvement.

## Verdict

Eta is not the lever, and the sweep is closed. The levers that remain, in the
order they would actually change the result:

1. **Probe every trial** at eta 3e-5. Cheap, and it is the only way to show a
   curve rather than a step. It does not change what the brain does, only what
   the figure can see.
2. **Raise KC recruitment** — more than 501 KCs in play. The graft's `SPARSITY`
   is the knob, but raising it trades against the sparse-coding claim; the fly
   is 5-10%, and 5% is what is authored.
3. **Fix the compartment prefix match** to recover the 4,402 approach-side
   edges that currently have no DAN. Bounded, mechanical, disclosed already.
4. **Raise W_MIN or lower lam** so the movers do not all end in the same place.
   The floor, not eta, is what makes every run identical.

None of these change the option-A blocker, which is the ignition regime, not
this rule.
