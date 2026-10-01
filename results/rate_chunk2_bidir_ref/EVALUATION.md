# bidir vs LIF reference: independent evaluation (Opus, 2026-10-01) + log hashes

## Recompute (own script from raw jsonl, not the analyzer): all match
- LIF A(T0) = 11.954 Hz in every seed (probe seed 0 + identical baseline weights = the same 104 spikes over 29 approach MBONs x 300 ms). rise_1 = 39, 40, 42, 42, 43 spikes, so rise_1_norm = 0.3750, 0.3846, 0.4038, 0.4038, 0.4135: match. Relearn 3/3 cycles in every seed: match (timed drop 16-27 spikes vs depress -1 to 6 per cycle).
- R = 0.39615, sample SD = 0.01580, 5/5 positive, 5/5 relearn, USABLE: match.
- Rate rise_1_norm c0 0.13067, c1s0-4 0.13735, 0.13770, 0.13742, 0.13671, 0.13747; threshold 0.19808; (a) False on all 6; (b) 3/3 cycles on all 6; label FAIL (a) on all 6: match. Arm (i) FAIL (a); arm (ii) 0/5 FAIL: match.
- Dose-matched secondary: LIF F1 |w_mean_frac| 0.0040-0.0045 sits at paper-0 rung A trial 5 (0.0042-0.0044), so 3 seeds are below-range by under 10%; the 2 in-range give ratio 1.01-1.02. Rate 0.82 (c0), 0.53-0.55 (c1): match.

## Provenance: clean
- 10 LIF logs: one meta row first, done row last, 27 guard rows, 0 breaches, 0 aborts, overwrite False, args = prereg params / bidir / own rule / arm lif / own seed, lif block = prereg model, gains None. prereg sha256, code_sha256 (11 files, LF-normalised) and data_sha256 identical across the 10 and equal to disk; reference.json and paper-0 condition_o1s0-4 match; 30 rate-dir logs match EVALUATION.md hashes; rate logs at 2fe2f2d, clean, no aborts.
- Timeline: prereg 40c920b 08:06; code 08:30-09:03; A1 57af1c1 09:06:31; first run started 09:08:02; runs strictly sequential to 13:13:28; run 10 started 12:48:50, after the parallel commit 66b4f44 (12:39:28); A2 c89244f 13:36:37; result written 13:36:44. err.log = 10 torch-inductor cubin cache warnings, one per process start, nothing else.
- A2 is legitimate and provenance-only. H1 is an ancestor of H2; H1..H2 = one line of data/CHECKSUMS.json, which no file on the science path reads (only build/verify scripts). The wrapper changes only (1) run 10's git_head, in memory, to H1, and (2) the analyzer head, passed as H1 with the real dirty flags (clean). Every other check (prereg, code and data sha, args, lif block, rate and paper-0 hashes, done rows) runs unmodified, and the scoring is untouched. The read_rows monkeypatch touches only rows of bidir_depress_lifs4.jsonl. reanalyzed = False.
- The early glance (seed-4 timed T0/F1/B1, about 13:05) could not change the label: thresholds and scoring were hash-pinned in code before the runs, A2 touches no scoring, and dropping seed 4 entirely still gives R = 0.392 and FAIL (a) on all bases. Not verifiable from logs: the pre-launch determinism check.

## Defects that look like findings
1. **FAIL (a) is not slower learning.** At F1 the rate model has MORE weight change than the LIF: |w_mean_frac| is 0.0060 on c0 (1.4x the LIF mean of 0.0043) and 0.0088-0.0092 on c1 (2.1x). The gap sits in how weight change maps to MBON rate. Normalised drop per unit |w| at F1 is 92 for the LIF and 22 for c0 (0.24x). In absolute Hz the LIF drops 4.7 Hz and c0 3.5 Hz, and the rate baseline is 2.2x higher (26.5 vs 12.0 Hz). The 3.0x gap in rise_1_norm is about 2.2x from the denominator and 1.4x from the absolute drop.
2. **The outcome was already in the rung A logs.** At matched |w| of about 0.0044 (rung A trial 5), paper-0 LIF gives a normalised drop of 0.37-0.38 and rate rung A gives 0.105 (3.6x lower). Rate rung A is near-linear (24-30 per unit |w|). LIF rung A is steep at first and then saturates (88 falling to 49; 0.97 by trial 30). Rung A passed only because it was scored at trial 30, where the LIF saturates. Scored at trial 5 it would also fail 0.5x. The LIF bidir run reproduces rung A at matched dose (ratio 1.0), so the bidir FAIL (a) is the same low-dose gain gap, not something specific to bidir. The mechanism (thresholded LIF near its operating point versus the rate model's higher, near-linear operating point) is plausible but NOT tested here.
3. **The dose metric is not specific to approach synapses.** w_mean_frac averages all KC-to-MBON edges. On c1 it includes the avoid-side collapse (avoid MBON 5.27 down to 0.2 Hz, tonic PAM 1.4 Hz), so the c1 "dose" overstates the approach dose. This is why c1's dose-matched ratio (0.54) is lower than c0's (0.82). It affects the secondary result only.
4. **Readout coarseness and low SD.** LIF readouts are quantised at 1/104 of A(T0) (0.0096 in rise_1_norm). SD 0.016 is about 1.6 spikes. All probes use seed 0, so probe noise is frozen and the SD reflects training-tick noise only. That makes it an underestimate of trial-to-trial spread, but it does not threaten the label: the margin to the threshold is about 12 SD, and the rate values are noise-free. Seeds are decorrelated: the tick seeds are distinct by construction, the counter hash is seed + step*S1 + neuron*S2 put through two murmur finalisers (checked: 0 same-neuron (seed, step) collisions over all 8,000 tick seeds and the seed-0 probes, assuming step0 carries within a block), and the weight hashes at F1 differ across all 10 runs. Rung A trial 5 with different probe seeds (0.37-0.38) agrees with R, so R is not a quirk of the probe-0 draw.
5. **Not limiting:** no latch (LIF kc_paired 0, approach latched fraction 0); headroom gate passes (LIF 1.06); no weights at floor or ceiling; LIF approach F4 3.3-3.7 Hz, still above 0. The LIF avoid MBONs are effectively silent (0.37 Hz = 1 spike per probe, constant), so the LIF reference measures the approach side only, and so does the rule.

## Verdicts
- LIF reference USABLE: **CONFIRMED.**
- Rate bidir vs LIF reference, arm (i) FAIL (a), arm (ii) FAIL 0/5 (a): **LABEL-CORRECT-BUT-MISLEADING.** The numbers and the rule are right. The prereg's reading of FAIL (a) as "learns less per block at the same dose ... learning speed" is contradicted by its own secondary data: the rate model changes its weights more per block, and its lower first-block effect is a lower weight-to-output gain at low dose. That gain gap already exists in rung A. Relearning (b) passes on every base.
- The chunk-2 bidir FAIL stands unchanged, as preregistered.

## Paper wording (may be used)
We ran paper 0's spiking model through the same open-loop mushroom-body schedule with five noise seeds. Its first training block lowered the approach output by 40% of baseline, and the timed rule re-learned in every scorable cycle on every seed. Measured against half of that reference, the rate model failed on all six bases, with a first-block drop of about 13%, while still re-learning in every cycle. The rate model's weights changed more per block than the spiking model's, so this is not slower plasticity. The same weight change moves the rate model's approach output less, and the same gap is present early in rung A, where the spiking model's response saturates by trial 30 but the rate model's does not. We therefore report that the rate model re-learns like the spiking model but has a lower weight-to-output gain at small weight changes.

Can say: the LIF shows MB-level open-loop relearning under this schedule; the rate model's relearning (rule b) matches it; the first-block gap is a gain difference at low dose, consistent across both protocols. Cannot say: that the rate model learns more slowly; that spiking is needed for relearning; anything about closed-loop behaviour; that the gain gap comes from thresholding (untested); that bidir "survives the neuron-model swap" (the label is FAIL).

## Log sha256 (logs gitignored; archived by hash)

- bidir_depress_lifs0.jsonl 76e3f77fbaf6ea481a6bac189f8a4d15e027f3bed96107b58861813b95b75b1a
- bidir_depress_lifs1.jsonl 113558d8ff537fbe692a1913062cecab6518369f854a3fc9f21359a1ce9563be
- bidir_depress_lifs2.jsonl 491383af2108ba77c83371d5d67e3222bfbc546f9125dce8f46d9d3e4971d403
- bidir_depress_lifs3.jsonl 5619131f9b06499d6e5e89542be990add411c9c1244c5068273013cf97e006ae
- bidir_depress_lifs4.jsonl d68587a9a945c32fc81a3c71013be77bc650c1878085807159dbe374c2d4cc40
- bidir_timed_lifs0.jsonl ba4e29a33ab78d7a98d23e1f2b394cc5903036a1da8a3272be02348da8cbd1d7
- bidir_timed_lifs1.jsonl f063fefeba1ea19ebe19cbb9e281cab562b2937a0fdb63f6d7e42e0d2994442b
- bidir_timed_lifs2.jsonl a280c125b14ae7bee306b4f5e80f93e40242dc3f932f4514f476c577a58bd11b
- bidir_timed_lifs3.jsonl af7299c11b542cf62fd53b5811db165b30ea4d01f12db3e150f8e0df4fc85192
- bidir_timed_lifs4.jsonl 918aa6444c95e7c12fa0952225e6d89c4a5c0415c8797d9ab8a652aa7bd5ce1a
