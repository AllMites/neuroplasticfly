<!-- AUTHORITATIVE copy of the condition v1 run notes. results/condition_v1_notes.md is a stale working file (gitignored) and is now a subset of this one -- do not read back from it. results/condition_v1.json and .jsonl remain untracked on disk only. -->

# condition v1 -- run notes

## Configuration
- Command: `python learn/condition.py --state v1 --train 30 --probe 5`
- Arms: learn, lesion, reversed, frozen. 30 training trials/arm, 5 probes/song.
- **US path: `dan`** (direct DAN drive at 100 Hz). Auto-detection rejected the real
  gustatory path at seed 0 (150 Hz on sugar GRNs left PAM at baseline; 150 Hz on bitter
  GRNs moved PPL1 to 0.83 Hz, under the +1.0 threshold). See the disclosure section below
  for the 10-seed sweep behind those two numbers.
- eta = 1e-3 (PL.ETA default), lam = 0.01 (PL.LAM default).
- Brain matrix recorded in provenance as `gpu_sim.BRAIN`.
- Wall time: 1444 s (learn 397 s, lesion 725 s, reversed 1105 s, frozen 1444 s cumulative).

## eta values tried
Smoke run (`--state smoke --train 3 --probe 2 --arms learn,lesion`, 98 s) at eta 1e-3 gave
w_mean_frac -0.046 and w_n_at_floor 0, so eta was not adjusted at the time. That reasoning was
wrong: both of those are SATURATED values (see *Learning-rule parameters* below), and the
3-trial run agreeing with the 30-trial run to two decimals was evidence of saturation being
misread as evidence of stability.

A sweep has since been run: three further runs (`--arms learn,lesion --train 30 --probe 5`)
at eta 3e-4, 1e-4 and 3e-5, alongside the original v1 run at eta 1e-3. v1 itself was not
touched. Artifacts on disk (gitignored): `results/condition_eta{3e-4,1e-4,3e-5}.{json,jsonl,png}`,
`brain_state/eta{3e-4,1e-4,3e-5}/`.

| eta | saturation trial (w_min_frac reaches -0.891) | check | misery shift | skyhigh shift |
|-----|----------------------------------------------|-------|-------------:|---------------:|
| 1e-3 (v1) | 1 | ACCEPT | +6.077 | -5.354 |
| 3e-4 | 1 | ACCEPT | +6.373 | -5.354 |
| 1e-4 | 1 | ACCEPT | +6.743 | -5.354 |
| 3e-5 | 5 | ACCEPT | +6.928 | -5.354 |

All four pass acceptance; the lesion control is identical at all of them (+0.437).

**Finding 1 -- weight saturation is eta-sensitive, and 3e-5 fixes it.** The update is
fractional (`step = -eta * kc_rate * dan_rate * w0`), and the DAN side is clamped at
DAN_HZ=100 by the direct-injection US path, so a single trial clips any edge whose
presynaptic KC fires above `0.9/(100*eta)` Hz: 9 Hz at eta 1e-3, 30 Hz at 3e-4, 90 Hz at
1e-4, 300 Hz at 3e-5. The last is above the fastest KC in a trial, which is why 3e-5 is
the first eta where w_min_frac takes 5 trials instead of 1 to reach -0.891. Its
trajectory: -0.286, -0.439, -0.641, -0.813, -0.891, then flat.

**Finding 2 -- the behavioural step is NOT about eta.** The skyhigh avoid_index probe
sequence is bit-identical across 1e-3, 3e-4 and 1e-4 at every probe block, at full float
precision -- a 33x change in learning rate moves it by exactly zero. Only 3e-5 perturbs
it, only at the trial-5 probe (-9.484 vs -9.687), and it rejoins the identical trace from
trial 10 onward. At 3e-5 the weights are demonstrably graded over trials 1-4 while 97% of
the misery shift has already landed by the first probe. Two causes: the hard floor
combined with the same ~5% KC fingerprint being recruited on every trial (making trial 2
onward a no-op once that fingerprint is saturated), and the probe cadence, hardcoded to
`k % 5 == 0` at `learn/condition.py:137`, which never samples during the ramp.

Conclusion: the step shape is a limit of the rule's architecture, not a hyperparameter. A
graded curve would need a soft or multiplicative bound instead of the hard clip, and a
recruited KC set that varies from trial to trial.

Cheapest next experiment (suggestion, not a commitment): eta 1e-5 (clip threshold ~900 Hz,
guaranteed unsaturated across all 30 trials) plus a probe every trial for the first five --
a one-line change to the `k % 5 == 0` cadence. That would separate "the behaviour genuinely
steps" from "we never measured during the ramp."

## Acceptance check
```
learn: misery +6.077 skyhigh -5.354 (baseline sd 0.939); lesion: misery +0.437
ACCEPT
```

## Step 4 -- descending-neuron readout (learn arm, baseline vs trial 30)
```
dn_DNp01     misery   baseline 52.00 -> trained 51.67  (sd 2.45)
dn_DNp01     skyhigh  baseline 60.33 -> trained 59.00  (sd 1.25)
dn_DNg29     misery   baseline 99.00 -> trained 97.33  (sd 2.00)
dn_DNg29     skyhigh  baseline 106.67 -> trained 102.33  (sd 1.05)
dn_DNg84     misery   baseline 2.00 -> trained 0.00  (sd 2.45)
dn_DNg84     skyhigh  baseline 6.67 -> trained 10.67  (sd 3.80)
dn_DNa02     misery   baseline 2.67 -> trained 0.00  (sd 3.43)
dn_DNa02     skyhigh  baseline 8.00 -> trained 17.00  (sd 5.91)
dn_DNp04     misery   baseline 0.00 -> trained 0.00  (sd 0.00)
dn_DNp04     skyhigh  baseline 0.00 -> trained 0.00  (sd 0.00)
avoid_index  misery   baseline -6.42 -> trained -0.34  (sd 0.50)
avoid_index  skyhigh  baseline -3.84 -> trained -9.19  (sd 0.94)
```

Shift in SD units (delta / that song's baseline sd):

| DN | misery | skyhigh | song separation (skyhigh - misery) |
|----|--------|---------|------------------------------------|
| DNp01 | -0.13 | -1.06 | -0.93 |
| **DNg29** | **-0.84** | **-4.12** | **-3.29** |
| DNg84 | -0.82 | +1.05 | +1.87 |
| DNa02 | -0.78 | +1.52 | +2.30 |
| DNp04 | 0.00 | 0.00 | 0.00 |

## Chosen on-screen behaviour: DNg29
Largest song-specific shift in SD units on both readings -- the biggest single shift
(-4.12 SD for skyhigh) and the biggest separation between the two songs (3.29 SD).
It also has a high, stable baseline (99-107 Hz), so it is readable on screen without
hitting a floor, unlike DNg84/DNa02 (baseline 2-8 Hz) and DNp04 (identically 0 Hz,
never recruited).

Runner-up: **DNa02**, the only DN that moves in *opposite* directions for the two songs
(-0.78 SD misery, +1.52 SD skyhigh, 2.30 SD separation). More visually striking if a
sign flip is wanted, but its baseline is near-silent and its SDs are large relative to
the shift.

## Step 6 -- generalisation to Mozart (novel song, probed but never trained)

Avoid-index shift = trial-30 avoid_index minus trial-0 avoid_index, learn arm.
Jaccard = KC-fingerprint overlap between that song and the named trained song.

| song    | avoid-index shift | jac(misery) | jac(skyhigh) |
|---------|-------------------:|------------:|-------------:|
| misery  | +6.077              | 1.00        | 0.00         |
| skyhigh | -5.354              | 0.00        | 1.00         |
| mozart  | +5.568              | 0.39        | 0.01         |

Mozart's shift is positive, same sign as misery, and misery is the song it overlaps
more with (jac 0.39 vs 0.01) -- so the sign follows the higher-overlap trained song.
But the magnitude test is mixed: 5.568 is smaller than misery's 6.077, but *larger*
than skyhigh's 5.354 -- so it is not smaller than *both* trained songs, only one of
them. Reporting as-is, not massaged: partial support for generalisation-by-overlap,
not a clean confirmation.

## Disclosure: US path did not use the real gustatory pathway
Auto-detection tried the real path first, at seed 0, and it failed to reach threshold:
150 Hz on 65 bitter GRNs moved mean PPL1 only 0.00 -> 0.83 Hz (needed baseline + 1.0);
150 Hz on 129 sugar GRNs left mean PAM at exactly 0.00 Hz. The run therefore used
**US path `dan`**: punishment/reward were injected directly into PPL1/PAM, bypassing
gustatory input entirely. Any claim about "tasting" punishment or reward is not
supported -- only "DANs were driven directly" is.

### 10-seed sweep of the same check (run after the fact)
The check was later re-run across 10 seeds. It did **not** change the v1 run, which
stands exactly as recorded; it only tells us how solid each half of that one-shot
measurement was.

| arm | population | mean (Hz) | SD | range | seeds over threshold |
|-----|-----------|-----------|----|-------|----------------------|
| punish | PPL1 under 150 Hz bitter drive | 1.0000 | 0.4757 | 0.2778 -- 1.9444 | 4/10 |
| reward | PAM under 150 Hz sugar drive | 0.0239 | 0.0514 | max 0.1412 | 0/10 |
| reward | PAM at baseline (no drive) | 0.0206 | 0.0453 | max 0.1303 | -- |

Decision tally: `dan` in 10/10 seeds.

The two arms are not equally solid. **Reward is the decisive, robust half**: PAM under
sugar drive is indistinguishable from PAM at baseline -- the same small nonzero values
turn up in both conditions, so they are background, not drive. It misses the
`baseline + 1.0 Hz` threshold by roughly a full Hz, about 20 SD. Sugar GRNs do not
reach PAM at all.

**Punish is seed-dependent.** Bitter GRNs *do* weakly reach PPL1, at around 1 Hz, but
with large trial-to-trial variance and a distribution centred on the threshold itself.
Seed 0's 0.83 Hz was one draw from that distribution, not a stable measurement, and it
must not be quoted as a precise fact.

The decision survives this anyway, because the acceptance rule is an `all()` over both
arms: the real path is used only if punish *and* reward both clear threshold. Reward
fails in every seed by a wide margin, so it alone forces `us_path = "dan"` no matter
which way the punish draw lands. The outcome is robust even though one of its two
inputs is not.

Honest one-liner: bitter GRNs do weakly reach PPL1, at about 1 Hz with large variance;
sugar GRNs do not reach PAM at all.

## Learning-rule parameters
- eta = 1e-3 (PL.ETA default) for v1. Now swept against 3e-4, 1e-4 and 3e-5 (see *eta
  values tried* above); all four pass ACCEPT and the behavioural readout is unchanged
  except at 3e-5, where saturation is delayed to trial 5.
- lam = 0.01 (PL.LAM default).
- W_MIN = 0.1 (floor as fraction of baseline weight, learn/plastic.py).
- Final weight state (v1): mean_frac -0.054, min_frac -0.891, n_at_floor 0. **`n_at_floor 0`
  does not mean no edge hit the floor.** `update()` clips to `W_MIN*w0` and then applies
  recovery in the same call, so a floored edge rests slightly above the clip point and the
  `<= W_MIN*w0` test can never match; the metric could not fire. (A code fix is in flight.)
  What the log actually shows is `w_min_frac = -0.891` -- exactly the floor-plus-recovery
  resting value -- on the FIRST training trial and on every trial after it.
- **The weights saturate immediately, and "30 training trials" is misleading.**
  `w_mean_frac` reaches its asymptote by trial 2 of 30 (-0.045) and only creeps to -0.054
  over the remaining 28. The probe curves are flat after the first probe block:

| song | avoid_index by probe |
|---------|----------------------|
| misery  | -6.421, +0.109, -0.100, -0.308, -0.345, -0.345, -0.345 |
| skyhigh | -3.837, -9.687, -9.857, -9.024, -9.604, -9.268, -9.190 |

  Essentially all of the learning happens within the first 5 training trials. This is a
  step, not a learning curve, and 28 of the 30 trials change almost nothing. Do not present
  the trial count or the curve shape as gradual learning.

## Chosen descending neuron: DNg29 (recap)
DNg29 was chosen for the on-screen readout: largest song-specific shift in SD units
(-4.12 SD skyhigh) and largest separation between songs (3.29 SD), on a high stable
baseline (99-107 Hz) that does not sit near a floor or ceiling.

## INTERPRETATION CAVEAT -- what avoid_index actually measures
The plasticity rule is depression-only and gates by opposing valence: each trial type can
only weaken the KC drive onto the MBON population on its own valence side. An earlier version
of this section quoted avoid-MBONs at **0.185 Hz** against **26.4 Hz** for approach-MBONs and
concluded that both songs move avoid_index by one mechanism, approach-side depression. Both
halves were wrong. Those two rates came from a one-off smoke trial that had a 100 Hz direct
DAN injection running, which drives the MBONs hard; they are not the protocol baseline. The
learn-arm baseline probe, and trial 0 -> trial 30:

| song    | approach MBON Hz | avoid MBON Hz |
|---------|-----------------:|--------------:|
| misery  | 6.64 -> 0.34 | 0.22 -> 0.00 |
| skyhigh | 6.87 -> 9.26 | 3.04 -> 0.07 |

**The two songs move the index by different mechanisms.** Misery's shift is approach-side
silencing, and it is near-total rather than graded: the approach-MBON population goes
6.64 -> 0.34 Hz, roughly 95% silenced. Sky High's -5.354 is majority avoid-side: its
avoid-MBONs fall 2.97 Hz, which is exactly the reward -> avoid-MBON depression the rule was
designed to produce, while its approach-MBONs *rise* 2.39 Hz -- a network effect the
depression-only argument does not predict and does not explain.

The bottom line is unchanged: do not overstate this as "the fly learned to avoid Misery",
because avoidance drive did not go up for either song. What must stop is the claim that both
songs share one mechanism.

Separately: DNg29's shift is the SAME SIGN for both songs (misery -0.84 SD,
skyhigh -4.12 SD) -- a magnitude difference, not a valence flip. DNa02 is the only DN
in the readout table that moves in opposite directions per song (misery -0.78 SD,
skyhigh +1.52 SD), but its baseline is near-silent (2-8 Hz) and its SDs are large
relative to the shift, so it is noisier evidence and was not chosen as the primary
readout.
