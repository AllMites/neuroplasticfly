# RESULT: latch check. NEGATIVE CONTROL FAILED: the unfitted base model latches

Prereg `c9f0357`. Check implemented in rate/suite.py (latch_check, a global invariant in run_suite).
Toy tests pass (a feed-forward chain returns to rest; a self-exciting neuron latches). 2026-09-25 ~07:50 Manila.

## Result

The unfitted arm-A base (suite.build(): w_scale 6.880548e-3, bias -35.03 Hz, r_max 454.5 Hz) FAILS
the check. Per the prereg this means "stop and report". The positive control and the chunk-1 g* fits
were NOT run.

| condition | neurons > 1 Hz in the last 500 ms after offset | max rate | in R |
|---|---|---|---|
| FOX 150 Hz | 0 | 0.0 | 0 |
| SUGAR 150 Hz | 411 | 454.5 | 0 |
| BITTER 150 Hz | 0 | 0.0 | 0 |
| ORN_DM4 40 Hz | 7,411 | 454.5 | 28 |

## Diagnostic (after the fail, to describe it; changes no label)

- ORN_DM4 during the stimulus: 7,878 neurons > 1 Hz (5.7 % of the brain), including 2,198 Kenyon cells,
  501 ME, 495 ME>LA, 343 LHLN, 314 ALPN, 236 ALLN. After offset, per 100 ms: 7498, 7410, 7409,
  7410, 7411, ... flat. A self-sustained state that never decays.
  - Super-class of the latched set: central 5,889; optic 1,261; descending 137.
- SUGAR during the stimulus: 642 > 1 Hz. After offset: 506, 430, ..., 411, flat.
  - Latched set: central 247, descending 86, 27 brain motor neurons.
- Only 1 neuron sat at r_max DURING either stimulus, so the ceiling does not create the latch; the
  latched attractor reaches the ceiling afterwards.

## CORRECTION 2026-09-25 ~09:30 (see results/rate_chunk0/STOP.md)

Point 1 below is WRONG. The stock LIF latches too: after ORN_DM4 40 Hz (1 s on, then off) 5,316 LIF
neurons (1,713 KCs) still fire 500-1000 ms later; 39 of 53 single-glomerulus odours latch it. Even the
eln8 LIF keeps 700 on. The rate base inherits an all-or-none latched state from the model it copies;
rung 4's silence was a weak drive (eln8, DC2). The check's premise must be restated relative to the LIF.

## What it means (original text, point 1 superseded)

1. The arm-A rate reduction of the paper-0 LIF is dynamically wrong at its derived scale. The
   paper-0 LIF is silent 40-60 ms after offset (rung 4) and keeps Kenyon-cell coding sparse. This base
   drives about 2,200 KCs by one glomerulus and holds them on forever. The noise-free linear reduction
   at w_scale ~19x the |W| stability bound has recurrent attractors that the LIF does not.
2. Earlier path-2 results were measured on this base:
   - Engine goldens: unaffected (they use the contracting scale 0.5/rho).
   - Chunk-1 control and fit: valid as procedures (every condition runs from rest, BASE = 0), but the
     base they rest on is not physiological. Their conclusions ("the block is at FDA -> PAM / at
     GRN -> Fox") are conditional on it.
   - Loop test: its NEEDS-REST-OF-BRAIN latch is almost certainly this base attractor, not something
     the fit created.
3. The suite is now RED on the base model (test_suite.py step 6 fails; kept red on purpose as the
   blocker). No chunk should be trained on top of a base that fails the invariant.

## Next (the user's call)

The base must pass the latch check before any further chunk. Options are in the session report
(recommended: a "chunk 0" that calibrates the rate base to the paper-0 LIF's own responses, preregistered).
