# Rung 3: the brain holds two odour associations at once. It forgets at exactly the rate we told it to (2026-09-21)

Rung 3 of `fly/.claude/PRPs/plans/neuroplas-roadmap-2026-09-21.md`. Three phases on ONE
brain, plastic weights never rebuilt between them:

| phase | what | trials |
|---|---|---|
| A | train DC2 with punishment | 30 |
| B | relax: no stimulus, no US, rule ON | 30 |
| C | train D with punishment | 30 |

Arms: `retain` (A, B), `interfere` (A, B, C), `lesion` (all three, rule off). 5 seeds,
eta 5e-6, `ELN_NEGATE` + `PN_KC_GAIN = 8.0`, US = punish at PPL1. `learn/persist.py`,
~25 min/seed.

The fourth control needed no run: Track 2's `reversed` arm already trained D on a **naive**
brain and moved DC2 by +1.74 Hz. Phase C's effect on DC2 is read against that.

## Why phase B needed a new script

`Plastic.update` applies the recovery term `lam * (w0 - w)` unconditionally, but
`condition.py` calls update **only on US trials** (`if us is not None`). In Track 2 there
was therefore no code path by which a weight could decay. `learn/persist.py` calls update
through the relax phase with no drive and no US: the punish DANs sit at 0.00 Hz, the
depression term `-eta * e * d * w0` goes to zero, and recovery runs alone.

## Falsifier results

### 0. Harness — PASS

Phase A reproduces Track 2: DC2 shift **+10.955 +- 0.267** against Track 2's +11.05 +- 0.20.
`retain` and `interfere` are bit-identical through phases A and B, as they must be.

### 1. Persistence — PASS at the readout, but the readout overstates it

| measure | A_end | B_end | retention |
|---|---|---|---|
| avoid-index shift (Hz) | +10.955 | +10.760 | **98.2% +- 0.9** |
| plastic `w_mean_frac` | -0.01972 | -0.01459 | **74.0%** |

**The honest number is 74%, not 98%.** Approach-MBON drive on the DC2 probe sits at
0.314 Hz at A_end and only lifts to 0.460 Hz by B_end, against an 11.2 Hz naive baseline.
A 26% recovery of weight moves the readout by 0.15 Hz because the readout is pinned near
its floor — the same ceiling effect flagged in `odour_conditioning_2026-09-21.md`, seen
from the other side. Report both numbers or report the weight one; never the 98% alone.

Also: `n_at_floor` goes 185.4 -> 0.0 within the first half of the relax phase. The floored
edges lift off immediately; what persists is the bulk, not the saturated tail.

### 2. Accumulation — PASS, at 85% of naive capacity

Phase C on an already-trained brain still forms the second association:

| | D shift across phase C |
|---|---|
| interfere arm | **+9.655 +- 0.451** |
| Track 2, D trained on a naive brain | +11.29 +- 0.21 |
| lesion arm | -0.344 +- 0.174 |

85.5% of naive. **Both associations are present simultaneously at C_end**, on approach-MBON
drive (Hz, naive baselines 11.195 for DC2 and ~12.4 for D):

| stimulus | baseline | A_end | B_end | C_end |
|---|---|---|---|---|
| dc2 | 11.195 | 0.314 | 0.460 | **1.556** |
| d | ~12.4 | 11.111 | 11.042 | **1.103** |

Total depression roughly doubles rather than being overwritten: `w_mean_frac` -0.01972
(A_end) -> -0.01459 (B_end) -> **-0.02791** (C_end). Two associations stacked, not one
replacing the other. This is the first real accumulating-brain measurement.

### 3. Interference — real, and small

| | DC2 change |
|---|---|
| decay alone, over 30 relax trials (`retain`) | -0.195 +- 0.104 |
| over 30 trials of training D (`interfere`) | **-1.120 +- 0.147** |
| training D on a naive brain (Track 2 `reversed`) | +1.74 |

Training the second odour costs the first ~5.7x more than an equal stretch of idle time,
and it flips the sign relative to the naive case. But the absolute loss is 1.12 Hz out of a
10.96 Hz association: DC2 still retains **88%** of its readout shift at C_end. The two
associations interfere; they do not compete for the same substrate in any strong sense.

### 4. Lesion — PASS

t0 -> C_end: dc2 -0.628, d -0.366, da1 -0.366. Flat within noise across all three stimuli
and all three phases.

## The caveat that matters most: the forgetting curve is an authored constant

`lam = 0.01` per trial. Over 30 relax trials that predicts `0.99^30 = 0.7397` weight
retention. Measured: **0.740**. An exact match.

**We did not discover a forgetting rate. We set one, and then measured it back.** The
persistence result says the machinery carries weights across phases and that the decay
behaves as specified — it does not say anything about how a fly forgets. Any copy claiming
"the fly's memory faded over time" would be describing `PL.LAM`. The non-trivial findings
here are accumulation (85% second-association capacity, both memories co-present) and
interference (5.7x decay, sign flip), neither of which follows from `lam`.

## Other caveats

1. **The relax interval is a fully silent brain** (`central_active` 0.0000, no drive at
   all), not a spontaneously active one. Recovery is per-trial and time-based so the
   measurement holds, but it is "time passed", not "the fly lived a while".
2. **Retention at the readout (98%) overstates retention at the weights (74%).** See above.
3. Everything from `odour_conditioning_2026-09-21.md` carries: five noise replicates of ONE
   brain not five flies; the whole effect is approach-MBON suppression with avoid MBONs
   flat, so the claim is "stops finding it attractive"; `PN_KC_GAIN = 8.0` and `ELN_NEGATE`
   are authored; the US is injected at the DANs because the connectome does not carry taste
   to the reward neurons.
4. Interference was measured at one interval length (30 trials) and one training length.
   Whether the 5.7x ratio holds at other spacings is untested.

## Does rung 3 depend on PN_KC_GAIN? No, and the decay constant is provably gain-independent

Repeated at `PN_KC_GAIN = 6.0`, 3 seeds (`--regime eln6`, states `p6s0-2`), everything else
identical.

| | gain 8 (n=5) | gain 6 (n=3) |
|---|---|---|
| F0 phase A, dc2 shift (Hz) | +10.955 +- 0.267 | +5.080 +- 0.140 |
| F1 readout retention over B | 98.2% +- 0.9 | 98.6% +- 1.0 |
| **F1 WEIGHT retention over B** | **74.0% +- 0.0** | **74.0% +- 0.0** |
| *`lam` prediction, 0.99^30* | *74.0%* | *74.0%* |
| F2 phase C, d shift (Hz) | +9.655 +- 0.451 | +2.580 +- 0.145 |
| F3 dc2 decay over B (Hz) | -0.195 +- 0.104 | -0.072 +- 0.050 |
| F3 dc2 change over C (Hz) | -1.120 +- 0.147 | -0.557 +- 0.039 |
| **F3 interference / decay** | **5.7x** | **7.7x** |
| F4 lesion dc2 t0 -> C_end (Hz) | -0.628 +- 0.444 | -0.466 +- 0.192 |

**Weight retention is 74.0% with zero variance at both gains.** That is the strongest possible
confirmation of the caveat above: the forgetting curve is `lam` and nothing else. It does not
move when the regime's largest authored parameter changes by 25%, because it never depended on
the regime at all.

Both memories are still co-present at C_end at gain 6 — approach-MBON drive 0.894 Hz (DC2) and
0.971 Hz (D) against naive baselines of 5.530 and 4.317. Interference is real at both gains and
the ratio is, if anything, larger at gain 6.

**One thing does shift with the gain: accumulation capacity.** Normalised as the fraction of
remaining approach drive that phase C consumes, the second association reaches **90.0% at gain 8
but 72.7% at gain 6**. Less recruitment leaves less room for a second memory. The "85.5% of
naive" figure above cannot be recomputed at gain 6 because the gain-6 Track 2 run had no
`reversed` arm, so there is no naive-D reference at that gain — the normalised comparison is
used instead, and it is not the same statistic.

Caveat on this check: 3 seeds at gain 6 against 5 at gain 8, one gain step, one direction.

## Reproduce

```
for s in 0 1 2 3 4; do
  .venv/Scripts/python.exe learn/persist.py --state p1s$s --seed0 $s \
    --arms retain,interfere,lesion --train 30 --relax 30 --probe 3 --eta 5e-6
done
```

```
# the PN_KC_GAIN falsifier for rung 3
for s in 0 1 2; do
  .venv/Scripts/python.exe learn/persist.py --state p6s$s --seed0 $s --regime eln6     --arms retain,interfere,lesion --train 30 --relax 30 --probe 3 --eta 5e-6
done
```

Raw: `results/persist_p1s*.json{,l}` (gain 8), `results/persist_p6s*.json{,l}` (gain 6). Both
gitignored.

## What is now open

- ~~`PN_KC_GAIN = 8.0` is load-bearing and still untested.~~ **TESTED at gain 6.0 for both
  Track 2 (2026-09-21) and rung 3 (2026-09-22). Both survive.** See the section below and
  `odour_conditioning_2026-09-21.md`.
- Spacing: does the interference ratio hold at longer and shorter intervals?
- A decay measurement that is not just `lam` would need an activity-dependent recovery
  mechanism, which does not exist in `plastic.py` and is not a small change.
- Rung 4 (CX / recurrence) can now be specced from these results rather than from a
  literature list, as the roadmap asked.
