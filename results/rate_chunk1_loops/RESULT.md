# RESULT: ignition loop test

Prereg `0f0b2f7` Part 2; script `4a69198`. Fits tested: for each seed, the pass-2 fit at the lowest cap
where >= 1 reward compartment is up. That is cap 2 for all 5 seeds; each has 2 compartments up
(these fits do not pass F). Analysed once.

## Labels: BISTABLE (5/5), NEEDS-REST-OF-BRAIN (5/5)

| seed | max abs(up - down) | turns on (up-sweep) | still on (down-sweep) at | D_R at 150 Hz up / down | C4 n_up | C5 silenced |
|---|---|---|---|---|---|---|
| 0 | 4.7 Hz | 150 Hz | 0 Hz | 1.2 / 4.7 | 2 | 1367 |
| 1 | 4.5 | 160 | 0 | 0.6 / 4.5 | 2 | 1368 |
| 2 | 5.6 | 150 | 0 | 1.5 / 5.5 | 2 | 1370 |
| 3 | 6.1 | 160 | 0 | 0.0 / 6.1 | 2 | 1369 |
| 4 | 5.1 | 160 | 0 | 0.0 / 5.0 | 2 | 1370 |

- Cuts C1 (PAM->PAM), C2 (PAM->FDA), C3 (FDA->FDA) and C4 (all three) never stop ignition. C5
  (output-silence the ~1,370 active non-owned neurons) always does.

## Descriptive: the ignited state LATCHES

Once ignited on the up-sweep (Fox 150-160 Hz), the reward compartments stay on all the way back to Fox
= 0 Hz on the down-sweep. The activity is self-sustained and never returns to baseline within the sweep.
- **Mechanism:** with no external drive left, only recurrence can hold it. The owned-pool loops are
  not needed (C4), and silencing the active rest of the brain ends it (C5). Taken together, a
  sustaining loop runs through non-owned neurons. C5 alone could not separate loop from relay; the
  persistence at 0 Hz is what points to a loop. The specific loop is NOT identified.
- **Ignition sits near the test drive:** onset is at 150-160 Hz, right at the 150 Hz used for fitting
  and testing. These cap-2 fits are marginal.

## Caveats (the artefact caveat in the prereg applies in full)

- The model has no noise, is linear above threshold and has a hard 454.5 Hz ceiling. A latched,
  never-resetting state is at least as likely to be a property of this rate model as of the fly.
  Real DANs return to baseline after a stimulus.
- These are cap-2 pass-2 fits (POST-HOC optimiser), not the g* = 8 fits.
- In the paper-0 LIF, the brain is silent 40-60 ms after stimulus offset (rung 4). The latch here
  shows the rate model and the LIF differ sharply in recurrent regime. Any memory or persistence
  claim built on the rate model must address this first.

## Consequence for path 2

The fitted rate model can enter a self-sustained latched state through recurrence outside chunk 1. It
does not show up in the chunk-1 battery (BASE stays 0; the battery runs each condition from rest), but
it will matter for any chunk that runs trials in sequence with state carried over (chunk 3: learn A,
then B, retention). Before chunk 3, a reset / no-latch check belongs in the regression suite. That is a
scope call for the user.
