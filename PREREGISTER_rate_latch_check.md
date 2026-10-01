# PREREGISTER: latch check (return-to-rest invariant) for the rate-model regression suite

Written and committed 2026-09-25 07:40 Manila, before any latch-check run. Requested by the user:
"Add a latch check first", ahead of the next training chunk. Motivation: the ignition loop test
(`01af94a`, D51) found fitted states that stay on after the drive is removed. Any chunk that carries
state across trials (chunk 3: learn A, learn B, test A) is invalid on a model that latches.

## The check (a model invariant; every chunk must pass it, independent of records)

For each stimulus condition, all conditions batched in one call:
1. Drive for T_ON = 1000 ms (the chunk-1 battery window), from rest.
2. Remove the drive: no drive for T_OFF = 1000 ms, state carried exactly (engine carry is exact).
3. Readout: the mean rate of every neuron over the LAST 500 ms of T_OFF.

- **PASS** if no neuron's mean rate in that window is above 1.0 Hz, in every condition.
- **FAIL (LATCH)** otherwise. Report per condition: the number of neurons above 1.0 Hz, the max rate,
  and how many of them are in the reward set R.

Conditions (fixed): FOX 150 Hz, SUGAR 150 Hz, BITTER 150 Hz (the chunk-1 stimuli), and ORN_DM4 40 Hz
(an odour stimulus, from the engine goldens).

Why these numbers: T_OFF = 1000 ms is 50 membrane time constants (tau 20 ms), long enough for any
non-self-sustained activity to decay to about e^-25. 1.0 Hz is the battery's "up" threshold. The
paper-0 LIF is silent 40-60 ms after offset (rung 4), so this check is far more lenient than the
LIF's own behaviour.

## Models checked

- **Negative control:** the unfitted arm-A base (suite.build()). Expected PASS. If it FAILS, the
  check or the engine is wrong; stop and report.
- **Positive control:** the 5 loop-test fits (pass-2, cap 2; stored edge_x). They latched on the Fox
  down-sweep, so expected FAIL for FOX. If they PASS, the check is wrong; stop and report.
- **Result of interest:** the 5 primary chunk-1 fits at g* = 8 (seeds 0-4).
  - Their cells store gains, not raw edge_x, so x is rebuilt as logit(clip(g/cap, 1e-7, 1 - 1e-7)).
  - **Gate G-rep:** the rebuilt model must reproduce each cell's stored FOX D for all 4 R
    compartments and b1 within 0.01 Hz. If it does not, that seed is reported NOT REPRODUCED and
    left out.

## Labels

- Per model: PASS or LATCH(conditions).
- Summary for the primary g* fits: NO-LATCH (5/5 PASS), LATCHES (5/5 LATCH), or MIXED(n/5).

## What it changes

- The check joins the regression suite as a GLOBAL invariant (run_suite is RED if the model
  latches), alongside the per-chunk records.
- If the primary g* fits LATCH, that goes on the record for chunk 1 (they would fail the new
  invariant). How future fits must satisfy it (as a loss constraint or as a filter) is decided in
  that chunk's prereg, not here.

## Not in scope

Finding the latch loop; noise; other drive rates or durations.
