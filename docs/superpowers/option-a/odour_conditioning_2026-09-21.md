# First odour-CS conditioning through the real antennal lobe: the paired odour loses its attractiveness, the unpaired one does not (2026-09-21)

Track 2 of `fly/.claude/PRPs/plans/neuroplas-roadmap-2026-09-21.md`, rung A. Every
learning result before this one used the song graft — JO drive plus an *authored* KC
fingerprint pinned at 5% active. This one drives 20 ORNs of a single glomerulus at 80 Hz
and lets the real antennal lobe and mushroom body produce the KC code.

**Result: the falsifier passes on every clause.** The paired odour's avoid index moves
+11.05 Hz (sd 0.195, 5/5 seeds same sign); the unpaired odour moves +1.43; the never-paired
odour +0.56; the lesion arm -0.73; the shuffle arm +0.54, i.e. 4.9% of learn. Swapping
which odour is paired swaps the result cleanly.

## Setup

`learn/condition.py --cs odour --regime eln8 --arms learn,lesion,shuffle,reversed
--us-path dan --eta 5e-6 --train 30 --probe 5 --seed0 <s> --dump`, seeds 0-4,
~22 min/seed on one RTX 5070 Ti.

- **CS**: one ORN channel at 80 Hz, 300 ms. Paired **ORN_DC2** (20 ORNs), unpaired
  **ORN_D** (31), never-paired **ORN_DA1** (126).
- **Regime**: `ELN_NEGATE` + `PN_KC_GAIN = 8.0`, stock `V_TH`. Both baked into the device
  weights at construction.
- **US**: punish at PPL1, injected at the DANs. Disclosed, and now *closed* rather than
  merely unavailable — see `reward-path/floor1_convergence_2026-09-21.md`. There is no
  reward arm because there is no reward path.
- **eta** 5e-6, no sweep (measured out).

### The CS pair is DC2 + D, and DA1 is deliberately excluded

The roadmap line said "DA1 vs DC2". That contradicted the project-plan decision of
2026-09-20 and was wrong: DA1 is the cVA pheromone glomerulus (Or67d; Kurtovic, Widmer &
Dickson 2007). cVA carries innate valence and drives approach/avoidance without learning,
so a shift on DA1 could be innate rather than learned and the shuffle arm would not
separate the two. Caught mid-run; the first run was killed and both stale docs corrected
in place.

DA1 stays as the third, **never-paired** probe channel, which turns the exclusion into a
measurement: if DA1 moved as much as DC2 the effect would be drift, not learning.

### Gate, run before any learning number

`learn/gate_cs_channels.py` — a script with a non-zero exit, not a judgement call. This is
the narrowed criterion that `kc_excitability_2026-09-20.md` asked to be recorded *in
advance* rather than reinterpreted after seeing a result.

| channel | role | ORNs | KC frac | KC n | Jaccard vs DC2 |
|---|---|---|---|---|---|
| ORN_DC2 | CS | 20 | 0.0603 | 312 | — |
| ORN_D | CS | 31 | 0.0551 | 285 | 0.0566 |
| ORN_DA1 | probe-only | 126 | 0.0894 | 463 | 0.0238 |

Both CS channels inside the 5-10% band, Jaccard 0.057 against a 0.3 target, 0 silent
channels, 0 spontaneous KC firing at zero drive. **PASS.**

## Result: avoid-index shift, last probe minus baseline (Hz, 5 seeds)

| arm | **dc2** (paired) | **d** (unpaired) | **da1** (never paired) |
|---|---|---|---|
| **learn** | **+11.05 +- 0.20** | +1.43 +- 0.49 | +0.56 +- 0.09 |
| lesion | -0.73 +- 0.41 | +0.29 +- 0.28 | +0.29 +- 0.07 |
| shuffle | +0.54 +- 0.51 | +0.40 +- 0.33 | +0.49 +- 0.03 |
| reversed | +1.74 +- 0.25 | **+11.29 +- 0.21** | +0.67 +- 0.10 |

Ratios: unpaired / paired **12.9%**, never-paired / paired **5.0%**, shuffle / learn
**4.9%**, lesion / learn **-6.6%**. The reversed arm is the mirror image — whichever odour
carries the US gets the +11, and its unpaired partner gets 15.4% of it.

### It is a learning curve, not a step

learn arm, avoid index by training trial (mean over seeds, Hz):

| stimulus | t0 | t5 | t10 | t15 | t20 | t25 | t30 |
|---|---|---|---|---|---|---|---|
| **dc2 (paired)** | -11.06 | -6.67 | -3.27 | -1.08 | -0.17 | -0.17 | **-0.01** |
| d (unpaired) | -11.82 | -11.39 | -10.77 | -11.00 | -10.61 | -10.45 | -10.39 |
| da1 (never paired) | -2.22 | -1.91 | -1.86 | -1.78 | -1.86 | -3.01 | -1.66 |

Monotone, saturating by ~t20. Lesion and shuffle curves are flat to within noise across
all three stimuli.

### The mechanism is approach-MBON suppression, and the readout has a ceiling

learn arm, MBON population rates (Hz, mean over seeds):

| stimulus | phase | approach | avoid |
|---|---|---|---|
| dc2 | baseline | 11.278 | 0.215 |
| dc2 | trained | **0.322** | 0.311 |
| d | baseline | 12.391 | 0.570 |
| d | trained | 10.979 | 0.585 |

The entire +11.05 is approach-MBON drive collapsing from 11.3 Hz to 0.3 Hz. Avoid MBONs do
not move. **So the honest claim is "the fly stops treating the punished odour as
attractive", not "the fly learns to avoid it"** — the same asymmetry the earlier DN
sensitivity work found, and it follows from reward being silent.

It also means the readout is **ceiling-limited**: approach drive cannot go below zero, so
+11.05 is close to the whole available range and this run measures *that* learning happens,
not *how much*. Plastic weights end at mean_frac -0.0198 with ~185 edges at the floor.

## Population DN readout (474 DN types)

Lesion-referenced effect `E = dS(arm) - dS(lesion)`, where `dS` is the paired-minus-unpaired
shift per DN type. The lesion arm gets the same US and the same drive with the rule off, so
subtracting it removes shared seed noise.

- **E_learn reproduces across seeds: r = +0.57 to +0.89** on all 10 pairs. There is a
  stable learned descending-neuron signature.
- **E_reversed anti-correlates with E_learn on every seed: r = -0.31 to -0.78.** Swapping
  the paired odour flips the population signature, as it must.
- 5 of 474 DN types exceed |mean E| > 3 sd across seeds. Largest: DNb05 -6.00 +- 1.75,
  DNge053 -5.93 +- 2.42, DNge037 -5.80 +- 1.02, DNbe007 -5.53 +- 1.50, DNp31 -3.40 +- 0.76.

## Caveats that must reach any copy

1. **The shuffle control separates at the MBON readout but NOT at the DN population
   readout.** shuffle/learn is 4.9% on avoid index, but `r(E_shuffle, E_learn)` is **+0.66
   to +0.94** on every seed, with |E| of the same order as learn's (0.15-0.21 vs 0.16).
   Whatever the DN-level effect is, permuting the eligibility trace does not remove it, so
   the *population* signature is not demonstrated to be pairing-specific. Only the
   MBON-level result is. Do not present the DN signature as a learned code without saying
   this. Related to the known earlier finding that shuffle conditions ~1/5 as much rather
   than being inert.
2. **Five seeds are five noise replicates of ONE brain, not five flies.**
3. **The US is injected at the dopaminergic neurons.** The connectome as published does not
   carry taste to the reward neurons; punishment through PPL1 is what the anatomy supports.
4. **`PN_KC_GAIN = 8.0` and `ELN_NEGATE` are authored parameters, not measurements.** The
   connectome gives synapse counts, not conductances, and `W_SYN` is a single global
   mV-per-synapse constant. **Tested 2026-09-21 at gain 6.0 — the result survives; see
   "Does the result depend on PN_KC_GAIN?" below.**
5. **The unpaired odour is not inert**: +1.43 Hz, 12.9% of the paired shift. Generalisation
   or drift; not separated here.
6. **Effect size is ceiling-limited** (see above). "It learned" is supported; "it learned
   this much" is not.
7. Two of eight tested ORN channels (DA3, DA2) recruit no Kenyon cells at any setting
   tested — a pre-existing defect, still unexplained, possibly another connectome gap.

## Does the result depend on PN_KC_GAIN? No — the gain sets the dynamic range, not the learning

`PN_KC_GAIN = 8.0` is the single largest authored number in the regime, so it got its own
falsifier: **if the paired/unpaired separation collapses at gain 6.0, Track 2 and rung 3 are
findings about the parameter rather than about the connectome.** One seed (seed 0), arms
learn / lesion / shuffle, everything else identical. Gate re-run at gain 6 first: DC2 0.0547,
D 0.0504, Jaccard 0.0604, 0 silent, 0 spontaneous — PASS, though D now sits on the band's
lower edge.

| | gain 8 | gain 6 |
|---|---|---|
| dc2 (paired) shift | +10.90 | +5.10 |
| d (unpaired) / paired | 13.9% | **-0.2%** |
| da1 (never paired) / paired | 4.9% | 3.6% |
| shuffle / learn | 5.3% | **-2.7%** |
| lesion / learn | -10.1% | -12.2% |

The absolute shift halves, but so does the room to move in. Naive approach-MBON drive on DC2
is 11.126 Hz at gain 8 and 5.540 Hz at gain 6, and training drives it to 0.299 and 0.437
respectively — **97.3% and 92.1% of the available range**. Learning is essentially as
complete at gain 6; there is simply less of it to consume.

Every control is at least as clean at gain 6 as at gain 8. **The result is not an artifact of
the gain.**

Two things worth keeping from this:

- **The higher gain makes the code LESS specific.** Generalisation to the unpaired odour is
  13.9% at gain 8 and -0.2% at gain 6. The 12.9% generalisation reported above is therefore a
  recruitment artifact of the gain — more Kenyon cells recruited, more cells shared between
  channels — not a property of the odour code. Caveat 5 should be read in that light.
- **Naive approach drive scales supralinearly with the gain**: 2.01x for a 1.33x gain change.
  A spike-threshold nonlinearity, not a scaling factor.

Caveat on this check: **one seed at gain 6 against five at gain 8.** It is a falsifier that did
not fire, not a five-seed replication.

Raw: `results/condition_g6s0.json{,l}`, `results/gate_cs_channels_g6.json`.

## Reproduce

```
.venv/Scripts/python.exe learn/gate_cs_channels.py --gain 8
for s in 0 1 2 3 4; do
  .venv/Scripts/python.exe learn/condition.py --state o1s$s --cs odour --regime eln8 \
    --arms learn,lesion,shuffle,reversed --us-path dan --eta 5e-6 \
    --train 30 --probe 5 --seed0 $s --dump
done
.venv/Scripts/python.exe learn/analyze_seeds.py o1s0 o1s1 o1s2 o1s3 o1s4

# the PN_KC_GAIN falsifier
.venv/Scripts/python.exe learn/gate_cs_channels.py --gain 6
.venv/Scripts/python.exe learn/condition.py --state g6s0 --cs odour --regime eln6   --arms learn,lesion,shuffle --us-path dan --eta 5e-6 --train 30 --probe 5 --seed0 0 --dump
```

Raw: `results/condition_o1s*.json{,l}`, `results/rates_o1s*/`, `results/gate_cs_channels.json`
(all gitignored).

## What this unblocks

Rung 3, persistence / interference, is now runnable and needs no new mechanism: condition
DC2, then D, test DC2 retention; add a no-stimulus interval for decay; measure interference
as loss of the first association per trial of the second. Plastic weights already persist in
`brain_state/`. That is the first real accumulating-brain measurement.
