# Option A gate: the MB has no sparse regime reachable by ORN drive (2026-09-20)

The gate before any option-A work was: get KC active fraction into 5-10% and
odour Jaccard under 0.3. This measures the first half on the **corrected**
connectome. It fails, and it fails structurally.

## The previous measurement was on the bad build

`results/kc_sparsity_probe.json` was written 2026-09-19 11:28. The corrected
`data/brain_gpu.npz` was written 2026-09-19 17:24. Every number in that file is
from the bad build, and its own output says so: **APL fired at 0.0 Hz** at every
drive, the documented bad-build defect (APL in-degree 0 vs 2,843). The old file
is kept at `results/kc_sparsity_probe_badbuild_2026-09-19.json`.

Re-run on the corrected matrix (`python kc_sparsity_probe.py`):

| | bad build | corrected |
|---|---|---|
| KC active, 1ch DA1 @ 40 Hz | 0.854 | **0.329** |
| KC mean Hz | 57.6 | 11.0 |
| APL mean Hz | 0.0 | 326.7 |
| central active | 0.272 | 0.157 |
| KC active, 40ch @ 150 Hz | 0.888 | 0.563 |

The correction restored APL feedback inhibition and roughly halved KC
recruitment. 33% is still 3-6x above the fly's 5-10%.

## The gap between 20 and 40 Hz is a cliff, not a ramp

The coarse sweep showed DA1 silent at 20 Hz and 33% at 40 Hz, so the gap was
swept at 2 Hz steps (`kc_sparsity_fine.py`, `results/kc_sparsity_fine.json`):

| ORN_DA1 Hz | 20 | 22 | 24 | 26 | **28** | 30 | 32 | 40 |
|---|---|---|---|---|---|---|---|---|
| KC active frac | 0.000 | 0.000 | 0.000 | 0.000 | **0.304** | 0.329 | 0.331 | 0.331 |
| KC mean Hz | 0.0 | 0.0 | 0.0 | 0.0 | 2.2 | 9.1 | 11.3 | 11.1 |
| APL Hz | 0.0 | 0.0 | 0.0 | 0.0 | 70.0 | 268.3 | 330.0 | 323.3 |

ORN_DM4 is already ignited at 20 Hz (0.327) and flat to 40 Hz. Different
channels have different thresholds; none has an intermediate plateau.

**KC active fraction is bistable: 0% or ~33%, with the transition inside 2 Hz.**
Even the single point on the edge (DA1 @ 28 Hz) is already at 0.304 fraction —
what is lower there is the *rate* (2.2 Hz) and the APL response (70 Hz), not
how many cells are recruited.

## What this means for option A

Drive strength is not the knob. No ORN rate, on one channel or forty, puts the
mushroom body in a 5-10% regime; the network either does not ignite or ignites
to a third of the KC population. Tuning the stimulus cannot pass this gate.

The remaining levers are structural, and each one is an authored change to be
disclosed:

1. **APL gain.** APL already fires at 320-420 Hz, which is not a plausible rate
   for a broad slow inhibitor. The ignited state may be APL's own saturation.
   Cheapest test: scale APL output and re-run this sweep.
2. **KC spike threshold / input scaling.** Raises the bar per KC directly, and
   is the standard way sparse coding is enforced in MB models.
3. **Short-term depression** on PN->KC, already on the roadmap as a later rung.

Odour Jaccard on the corrected matrix was 0.94 in the 2026-09-19 report and has
not been re-measured since; the second half of the gate is still open. It is
not worth measuring until one of the three levers above moves the fraction,
because a 33%-active MB cannot discriminate whatever the Jaccard says.

Scripts: `kc_sparsity_probe.py` (coarse, 5 channel sets x 7 rates),
`kc_sparsity_fine.py` (2 Hz steps across the cliff, 2 channels).

---

# Part 2: APL gain, and where the merge actually is (2026-09-20, same day)

Part 1 ended with three proposed structural levers. APL gain was the cheapest,
so it was run first. It works on sparsity and does nothing for odour identity,
and finding out why located the real fault.

Scripts: `apl_sparsity_sweep.py`, `apl_jaccard.py`, `eln_drive_sweep.py`.
Raw: `results/apl_sparsity_sweep.json`, `results/apl_jaccard.json`,
`results/eln_drive_sweep.json`.

No new mechanism was written. `gpu_sim` already carries a graded APL
(`APL_GRADED`) in a subtractive or a divisive form (`APL_DIVISIVE`,
`APL_DIV_SCALE`, smaller is stronger), all off by default, added 2026-09-19 and
documented as the standard account of Kenyon-cell sparse coding. This turns
them on.

## 1. Divisive APL does give a sparse regime, smoothly

1ch ORN_DA1 at 40 Hz, KC active fraction by `APL_DIV_SCALE`:

| config | 11.75 (1x) | 8.0 | 6.0 | 5.0 | 4.5 | 4.0 | 3.5 | 3.0 |
|---|---|---|---|---|---|---|---|---|
| KC active frac | 0.402 | 0.229 | **0.099** | **0.046** | 0.027 | 0.014 | 0.005 | 0.002 |
| KC mean Hz | 9.8 | 4.4 | 1.6 | 0.7 | 0.4 | 0.2 | 0.1 | 0.0 |

Monotone and graded, no cliff — the opposite of drive strength, which was
bistable. The 5-10% band sits at `APL_DIV_SCALE` about 5-6, and KC rates there
are 0.7-1.6 Hz, which is the right order for the fly. The subtractive form
barely moves anything (0.284 vs 0.330 baseline), and divisive at the authored
1x makes the MB *less* sparse than baseline, not more.

**Sparsity, the first half of the gate, is reachable.**

## 2. It buys nothing, because the codes do not separate

Eight single ORN channels at 80 Hz, mean pairwise Jaccard of the active-KC
sets. Target < 0.3.

| config | KC frac | KC Jaccard (mean / min / max) | PN frac | PN Jaccard |
|---|---|---|---|---|
| baseline spiking | 0.330 | 0.964 / 0.947 / 0.983 | 0.488 | 0.961 |
| divisive 8.0 | 0.229 | 0.947 / 0.931 / 0.970 | 0.470 | 0.960 |
| divisive 6.0 | 0.101 | 0.915 / 0.881 / 0.953 | 0.463 | 0.957 |
| divisive 5.0 | 0.047 | 0.892 / 0.841 / 0.962 | 0.463 | 0.955 |
| divisive 4.5 | 0.027 | 0.898 / 0.828 / 0.957 | 0.462 | 0.959 |

Taking the MB from 33% to 5% active moves the Jaccard from 0.96 to 0.89. Every
odour lights nearly the same Kenyon cells; divisive APL just lights fewer of
them. It subsamples a merged code.

**Measurement trap, recorded because it nearly got written up as a pass.** The
first run drove the channels at 40 Hz, where 4 of the 8 are below their own
ignition threshold. A silent channel scores Jaccard 0 against everything, and
the mean came out at 0.25 — under the gate, for the wrong reason. Only the max
(0.94) gave it away. `apl_jaccard.py` now drives at 80 Hz, scores ignited pairs
only, and reports the silent count and the max alongside the mean.

## 3. The merge is upstream of the mushroom body

The PN column above is the answer: **PN Jaccard is 0.955-0.961 under every APL
setting.** APL is an MB-intrinsic neuron and cannot reach it. The KC code is
merged because its input is already merged, so no APL configuration can fix it.

`ELN_NEGATE` — the one regime known to un-merge the antennal lobe, from the
regime plan closed earlier the same day — confirms this directly:

| | PN frac | PN Jaccard | KC frac | KC Jaccard | MBON |
|---|---|---|---|---|---|
| baseline | 0.488 | 0.961 | 0.330 | 0.964 | 31.0 |
| ELN_NEGATE | 0.019 | **0.057** | 0.007 | **0.000** | 4.6 |
| ELN_NEGATE + divisive 6.0 | 0.015 | 0.036 | 0.0002 | n/a (7 of 8 silent) | 0.0 |

The AL eLN chemical broadcast is the merge. Remove it and odour identity comes
back at the PN layer and is inherited by the KCs.

## 4. Under ELN_NEGATE the MB is under-driven, and drive scales it too slowly

Eight channels, `ELN_NEGATE` on, ORN rate swept (`eln_drive_sweep.py`):

| ORN Hz | 80 | 120 | 200 | 300 | 500 | 800 |
|---|---|---|---|---|---|---|
| channels silent (of 8) | 4 | 3 | 2 | 2 | 2 | 2 |
| KC active frac | 0.007 | 0.009 | 0.013 | 0.017 | 0.019 | **0.023** |
| KC Jaccard mean (max) | 0.000 (0.000) | 0.002 (0.014) | 0.005 (0.039) | 0.007 (0.027) | 0.010 (0.031) | 0.016 (0.035) |
| MBON active | 4.6 | 7.0 | 7.6 | 8.5 | 11.4 | 14.9 |

Separation is not the problem here — 0.016 against a 0.3 target, twenty times
better than required. The problem is recruitment: a 10x rate increase buys
0.7% -> 2.3%, roughly logarithmic, and 800 Hz on an ORN is already far outside
anything physiological. Two channels never ignite at all.

## Gate verdict

Split, and the two halves want opposite things:

- **Default regime**: MB active (33%) but odour-blind (Jaccard 0.96). Divisive
  APL at `APL_DIV_SCALE` 5-6 fixes the fraction and leaves the blindness.
- **ELN_NEGATE**: odours cleanly separated (Jaccard 0.016) but the MB is
  under-driven (2.3% at an absurd 800 Hz, 2 of 8 channels dead).

ELN_NEGATE is much the better starting point: being under-active is the
opposite of the original failure and the far easier end to fix. The remaining
gap is excitatory gain into the KCs, and the lever is deliberately *not* ORN
rate, which has now been shown to scale too slowly.

Untried, in order of cost:

1. **KC excitability under ELN_NEGATE** — lower the KC spike threshold or raise
   PN->KC weight until the fraction reaches 5-10% at a physiological ORN rate.
   The Jaccard headroom (0.016 vs 0.3) says there is a lot of room to spend.
2. **Why 2 of 8 channels never ignite** under ELN_NEGATE at any rate. Possibly
   the same class of connectome gap as the APL in-degree bug; worth one look
   before tuning around it.
3. `NORM_TOTAL_TARGET`, the third switch added alongside APL_GRADED, untouched
   here.

Do not spend more time on APL gain or on ORN drive. Both are measured out.
