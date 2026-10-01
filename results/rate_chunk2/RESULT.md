# Chunk-2 result

## Provenance

- prereg sha256: 1d4ae3e00cdd14415deb6a5706935ccf42f3962e6eb4402cde62c96e6d08711d
- reference.json sha256: 51f00736c27fdef3d9a08b83b98e55bd954ff39104051253e7466992088e669f
- analyzer: git 2fe2f2ddafedd1372988b4d33ffb8a7d12b512b2 dirty rate=False learn=False
- data sha256 (distinct across runs; reported, not refused): ['{"data/brain_gpu.npz": "b6308a8f51c0b68d90c78d337c88418a7ccaf10778a37a6de06b7e3e8653f281", "data/neuron_meta.npz": "7d02505d61ce7a5dd673f3f94db29edd35843a0f6cc7cd9f6d5b9b067b276228"}']
- rung_a_c0: git 2fe2f2ddafedd1372988b4d33ffb8a7d12b512b2 dirty rate=False learn=False overwrite=False
- rung3_c0: git 2fe2f2ddafedd1372988b4d33ffb8a7d12b512b2 dirty rate=False learn=False overwrite=False
- bidir_timed_c0: git 2fe2f2ddafedd1372988b4d33ffb8a7d12b512b2 dirty rate=False learn=False overwrite=False
- bidir_depress_c0: git 2fe2f2ddafedd1372988b4d33ffb8a7d12b512b2 dirty rate=False learn=False overwrite=False
- rung_a_c1s0: git 2fe2f2ddafedd1372988b4d33ffb8a7d12b512b2 dirty rate=False learn=False overwrite=False gains dcd5e129a906
- rung3_c1s0: git 2fe2f2ddafedd1372988b4d33ffb8a7d12b512b2 dirty rate=False learn=False overwrite=False gains dcd5e129a906
- bidir_timed_c1s0: git 2fe2f2ddafedd1372988b4d33ffb8a7d12b512b2 dirty rate=False learn=False overwrite=False gains dcd5e129a906
- bidir_depress_c1s0: git 2fe2f2ddafedd1372988b4d33ffb8a7d12b512b2 dirty rate=False learn=False overwrite=False gains dcd5e129a906
- rung_a_c1s1: git 2fe2f2ddafedd1372988b4d33ffb8a7d12b512b2 dirty rate=False learn=False overwrite=False gains 889b058078bd
- rung3_c1s1: git 2fe2f2ddafedd1372988b4d33ffb8a7d12b512b2 dirty rate=False learn=False overwrite=False gains 889b058078bd
- bidir_timed_c1s1: git 2fe2f2ddafedd1372988b4d33ffb8a7d12b512b2 dirty rate=False learn=False overwrite=False gains 889b058078bd
- bidir_depress_c1s1: git 2fe2f2ddafedd1372988b4d33ffb8a7d12b512b2 dirty rate=False learn=False overwrite=False gains 889b058078bd
- rung_a_c1s2: git 2fe2f2ddafedd1372988b4d33ffb8a7d12b512b2 dirty rate=False learn=False overwrite=False gains 2a183cc7174e
- rung3_c1s2: git 2fe2f2ddafedd1372988b4d33ffb8a7d12b512b2 dirty rate=False learn=False overwrite=False gains 2a183cc7174e
- bidir_timed_c1s2: git 2fe2f2ddafedd1372988b4d33ffb8a7d12b512b2 dirty rate=False learn=False overwrite=False gains 2a183cc7174e
- bidir_depress_c1s2: git 2fe2f2ddafedd1372988b4d33ffb8a7d12b512b2 dirty rate=False learn=False overwrite=False gains 2a183cc7174e
- rung_a_c1s3: git 2fe2f2ddafedd1372988b4d33ffb8a7d12b512b2 dirty rate=False learn=False overwrite=False gains 7150ef0b211c
- rung3_c1s3: git 2fe2f2ddafedd1372988b4d33ffb8a7d12b512b2 dirty rate=False learn=False overwrite=False gains 7150ef0b211c
- bidir_timed_c1s3: git 2fe2f2ddafedd1372988b4d33ffb8a7d12b512b2 dirty rate=False learn=False overwrite=False gains 7150ef0b211c
- bidir_depress_c1s3: git 2fe2f2ddafedd1372988b4d33ffb8a7d12b512b2 dirty rate=False learn=False overwrite=False gains 7150ef0b211c
- rung_a_c1s4: git 2fe2f2ddafedd1372988b4d33ffb8a7d12b512b2 dirty rate=False learn=False overwrite=False gains 79fe609d312b
- rung3_c1s4: git 2fe2f2ddafedd1372988b4d33ffb8a7d12b512b2 dirty rate=False learn=False overwrite=False gains 79fe609d312b
- bidir_timed_c1s4: git 2fe2f2ddafedd1372988b4d33ffb8a7d12b512b2 dirty rate=False learn=False overwrite=False gains 79fe609d312b
- bidir_depress_c1s4: git 2fe2f2ddafedd1372988b4d33ffb8a7d12b512b2 dirty rate=False learn=False overwrite=False gains 79fe609d312b

## Rule

RULE (fixed in code before the batch; thresholds are fractions of reference.json, paper 0):
Approach component: every gate uses the approach-MBON mean rate, because plasticity acts only on approach synapses in both models (the rate model's avoid baseline is 10-25x paper 0's, so a total avoid_index shift can be moved by avoid MBONs). approach drop = baseline (trial-0, baseline weights) approach-MBON mean minus the approach-MBON mean at the test; positive = learning. Normalised drop = approach drop / the baseline approach-MBON mean of the same arm and odour. The paper-0 normalised drop is its mean approach drop divided by its mean baseline approach (reference.json rung_a.learn_dc2_approach_drop_norm, naive_d_approach_drop_norm, rung3.approach_ratio_of_means). The total avoid_index shift and the avoid-MBON component are reported, never gated.
Headroom gate (baseline weights, each probe odour): PASS iff the baseline approach-MBON mean is > 0 and neither MBON group mean is >= 0.95 x r_max. No absolute rate threshold.
rung A: approach drop = learn-arm DC2 approach at trial 0 minus at the test trial (reference test_trial). PASS iff drop > 0 and normalised drop >= 0.5 x the paper-0 normalised DC2 approach drop and the learn normalised drop exceeds both the lesion and the shuffle arm normalised DC2 drops. D and DA1 are reported, not gated.
rung 3: ratio = D approach drop of the interfere arm (relax_test end minus trainC_test end) / D approach drop of the rung A reversed arm (D paired with punishment on a naive brain; trial 0 minus test trial). PASS iff both drops are > 0 (same direction as paper 0; two negative drops fail), the rate naive D normalised approach drop >= 0.5 x the paper-0 naive D normalised approach drop (a near-zero denominator cannot inflate the ratio) and ratio >= 0.5 x the paper-0 approach-drop ratio.
bidir (timed rule scored against the depress rule; both must be in the batch): learning level = minus the DC2 approach-MBON mean; rise_k = level at F_k minus at B_(k-1) (T0 for k = 1) (an approach drop); drop_j = level at F_j minus at B_j. Cycle k = 2..4 relearns iff rise_k >= 0.5 x rise_1 and timed drop_(k-1) > 0 and timed drop_(k-1) > depress drop_(k-1). PASS iff rise_1 / baseline approach DC2 (timed T0) >= 0.5 x the rung A paper-0 normalised DC2 approach drop (borrowed, no MB-level bidir reference exists) and at least 2 cycles relearn.
Arms: arm (i) = base c0 (the frozen C0 brain); its protocol labels are its verdicts. Arm (ii) = bases c1s0..c1s4 (C0 plus the chunk-1 reward-path gains at cap 16, one base per fitted seed). Each seed is labelled per protocol by the rules above on its own baseline. An arm (ii) protocol label is PASS iff at least ceil(0.8 x n) = 4 of the n = 5 seeds PASS that protocol, else FAIL with the per-seed labels and diagnostics listed. The batch is 4 runs x 6 bases = 24 runs; every run needs its done row.
Diagnostic for a failing protocol (or failing seed), first match wins: READOUT-CEILING > FAIL-UNSTABLE > NO-WEIGHT-CHANGE > LATCH-CONFOUNDED > NEEDS-SPIKING. READOUT-CEILING = any headroom row of the protocol failed its gate (a saturated or a zero baseline); FAIL-UNSTABLE = a guard abort row in an arm the verdict needs, even when its scoring rows exist (an abort voids a PASS); NO-WEIGHT-CHANGE = |w_mean_frac| < 0.1 x |paper-0 value| (rung A learn arm at the test trial; rung 3 interfere arm end of trainC; bidir timed rule, largest |w_mean_frac| over its test rows, rung A value borrowed); LATCH-CONFOUNDED = a BASELINE latch row of the protocol has latched True (end rows are reported, never gated); NEEDS-SPIKING = none of the above. A failing protocol with missing rows and neither an abort row nor a failed headroom row to explain them is INCOMPLETE-DATA (a pipeline fault, not a finding).

## c0 (arm i)

### Numbers (c0)

- rung_a: test_trial=30, baseline_approach_hz=26.5, approach_drop_hz=15.23, norm_approach_drop=0.5748, threshold_norm=0.4857, lesion_norm_approach_drop=0, shuffle_norm_approach_drop=0.01623, d_approach_drop_hz=0.8916, reversed_d_approach_drop_hz=14.65, dc2_avoid_index_shift=16.08, dc2_avoid_component_shift_hz=0.8478, lesion_dc2_avoid_index_shift=0, shuffle_dc2_avoid_index_shift=0.4372, d_avoid_index_shift=0.9839, da1_avoid_index_shift=0.2168, reversed_d_avoid_index_shift=16.11, w_mean_frac=-0.01988, w_mean_frac_ref=-0.01972
- rung3: trained_d_approach_drop_hz=14.19, naive_d_approach_drop_hz=14.65, baseline_approach_d_hz=24.85, naive_norm_approach_drop=0.5898, ratio=0.9685, ref_ratio=0.8581, threshold_ratio=0.4291, naive_min_norm=0.4673, trained_d_avoid_index_shift=15.67, trained_d_avoid_component_shift_hz=1.481, naive_d_avoid_index_shift=16.11, naive_d_avoid_component_shift_hz=1.452, w_mean_frac=-0.02927, w_mean_frac_ref=-0.02791
- bidir: f1_min_rise_norm=0.4857, baseline_approach_hz=26.5, w_mean_frac=-0.01912, w_mean_frac_ref=-0.01972, rise_1_avoid_index=3.424, rise_1_avoid_component_hz=-0.038, rise_1=3.462, rise_1_norm=0.1307, n_relearn=3, scorable=3

### Labels (c0)

- rung_a: PASS
- rung3: PASS
- bidir: FAIL (NEEDS-SPIKING)

### End-of-arm latch (c0; reported, never gated)

- rung_a rung_a learn dc2: latched=False kc_paired=0 appr_frac=0.3532
- rung3 rung3 interfere dc2: latched=False kc_paired=0 appr_frac=0.3028
- rung3 rung3 interfere d: latched=False kc_paired=0 appr_frac=0.349
- bidir_timed bidir bidir_timed dc2: latched=False kc_paired=0 appr_frac=0.2885
- bidir_depress bidir bidir_depress dc2: latched=False kc_paired=0 appr_frac=0.3122

### Guard summary (c0)

- bidir_depress/bidir_depress: rows 27, max active ratio 1.01, MBON06 ratio [n/a, n/a], breaches 0, aborted False
- bidir_timed/bidir_timed: rows 27, max active ratio 1.006, MBON06 ratio [n/a, n/a], breaches 0, aborted False
- rung3/interfere: rows 57, max active ratio 1.008, MBON06 ratio [n/a, n/a], breaches 0, aborted False
- rung3/lesion: rows 57, max active ratio 1, MBON06 ratio [n/a, n/a], breaches 0, aborted False
- rung3/retain: rows 39, max active ratio 1.008, MBON06 ratio [n/a, n/a], breaches 0, aborted False
- rung_a/frozen: rows 81, max active ratio 1, MBON06 ratio [n/a, n/a], breaches 0, aborted False
- rung_a/learn: rows 51, max active ratio 1.008, MBON06 ratio [n/a, n/a], breaches 0, aborted False
- rung_a/lesion: rows 51, max active ratio 1, MBON06 ratio [n/a, n/a], breaches 0, aborted False
- rung_a/reversed: rows 51, max active ratio 1, MBON06 ratio [n/a, n/a], breaches 0, aborted False
- rung_a/shuffle: rows 51, max active ratio 1.002, MBON06 ratio [n/a, n/a], breaches 0, aborted False

## c1 (arm ii: C0 + chunk-1 gains, 5 fitted seeds)

### Arm label (PASS iff >= 4 of 5 seeds PASS)

- rung_a: PASS (5 of 5 seeds)
- rung3: PASS (5 of 5 seeds)
- bidir: FAIL (0 of 5 seeds)

### Per-seed labels

| seed | rung_a | rung3 | bidir |
|---|---|---|---|
| 0 | PASS | PASS | FAIL (NEEDS-SPIKING) |
| 1 | PASS | PASS | FAIL (NEEDS-SPIKING) |
| 2 | PASS | PASS | FAIL (NEEDS-SPIKING) |
| 3 | PASS | PASS | FAIL (NEEDS-SPIKING) |
| 4 | PASS | PASS | FAIL (NEEDS-SPIKING) |

### Numbers (c1s0)

- rung_a: test_trial=30, baseline_approach_hz=26.5, approach_drop_hz=15.24, norm_approach_drop=0.5752, threshold_norm=0.4857, lesion_norm_approach_drop=0, shuffle_norm_approach_drop=0.01624, d_approach_drop_hz=0.8946, reversed_d_approach_drop_hz=14.67, dc2_avoid_index_shift=16.01, dc2_avoid_component_shift_hz=0.7726, lesion_dc2_avoid_index_shift=0, shuffle_dc2_avoid_index_shift=0.4371, d_avoid_index_shift=0.9953, da1_avoid_index_shift=0.2168, reversed_d_avoid_index_shift=15.95, w_mean_frac=-0.02016, w_mean_frac_ref=-0.01972
- rung3: trained_d_approach_drop_hz=14.2, naive_d_approach_drop_hz=14.67, baseline_approach_d_hz=24.85, naive_norm_approach_drop=0.5902, ratio=0.9684, ref_ratio=0.8581, threshold_ratio=0.4291, naive_min_norm=0.4673, trained_d_avoid_index_shift=15.5, trained_d_avoid_component_shift_hz=1.295, naive_d_avoid_index_shift=15.95, naive_d_avoid_component_shift_hz=1.283, w_mean_frac=-0.02976, w_mean_frac_ref=-0.02791
- bidir: f1_min_rise_norm=0.4857, baseline_approach_hz=26.5, w_mean_frac=-0.02987, w_mean_frac_ref=-0.01972, rise_1_avoid_index=3.253, rise_1_avoid_component_hz=-0.3864, rise_1=3.64, rise_1_norm=0.1373, n_relearn=3, scorable=3

### Labels (c1s0)

- rung_a: PASS
- rung3: PASS
- bidir: FAIL (NEEDS-SPIKING)

### End-of-arm latch (c1s0; reported, never gated)

- rung_a rung_a learn dc2: latched=False kc_paired=0 appr_frac=0.3532
- rung3 rung3 interfere dc2: latched=False kc_paired=0 appr_frac=0.3029
- rung3 rung3 interfere d: latched=False kc_paired=0 appr_frac=0.3482
- bidir_timed bidir bidir_timed dc2: latched=False kc_paired=0 appr_frac=0.1977
- bidir_depress bidir bidir_depress dc2: latched=False kc_paired=0 appr_frac=0.3119

### Guard summary (c1s0)

- bidir_depress/bidir_depress: rows 27, max active ratio 1.009, MBON06 ratio [n/a, n/a], breaches 0, aborted False
- bidir_timed/bidir_timed: rows 27, max active ratio 1, MBON06 ratio [n/a, n/a], breaches 0, aborted False
- rung3/interfere: rows 57, max active ratio 1.009, MBON06 ratio [n/a, n/a], breaches 0, aborted False
- rung3/lesion: rows 57, max active ratio 1, MBON06 ratio [n/a, n/a], breaches 0, aborted False
- rung3/retain: rows 39, max active ratio 1.008, MBON06 ratio [n/a, n/a], breaches 0, aborted False
- rung_a/frozen: rows 81, max active ratio 1, MBON06 ratio [n/a, n/a], breaches 0, aborted False
- rung_a/learn: rows 51, max active ratio 1.008, MBON06 ratio [n/a, n/a], breaches 0, aborted False
- rung_a/lesion: rows 51, max active ratio 1, MBON06 ratio [n/a, n/a], breaches 0, aborted False
- rung_a/reversed: rows 51, max active ratio 1.001, MBON06 ratio [n/a, n/a], breaches 0, aborted False
- rung_a/shuffle: rows 51, max active ratio 1.002, MBON06 ratio [n/a, n/a], breaches 0, aborted False

### Numbers (c1s1)

- rung_a: test_trial=30, baseline_approach_hz=26.5, approach_drop_hz=15.24, norm_approach_drop=0.5752, threshold_norm=0.4857, lesion_norm_approach_drop=0, shuffle_norm_approach_drop=0.01624, d_approach_drop_hz=0.8948, reversed_d_approach_drop_hz=14.67, dc2_avoid_index_shift=16.01, dc2_avoid_component_shift_hz=0.7702, lesion_dc2_avoid_index_shift=0, shuffle_dc2_avoid_index_shift=0.4371, d_avoid_index_shift=1.004, da1_avoid_index_shift=0.2168, reversed_d_avoid_index_shift=15.94, w_mean_frac=-0.02017, w_mean_frac_ref=-0.01972
- rung3: trained_d_approach_drop_hz=14.2, naive_d_approach_drop_hz=14.67, baseline_approach_d_hz=24.85, naive_norm_approach_drop=0.5902, ratio=0.9684, ref_ratio=0.8581, threshold_ratio=0.4291, naive_min_norm=0.4673, trained_d_avoid_index_shift=15.48, trained_d_avoid_component_shift_hz=1.279, naive_d_avoid_index_shift=15.94, naive_d_avoid_component_shift_hz=1.274, w_mean_frac=-0.02978, w_mean_frac_ref=-0.02791
- bidir: f1_min_rise_norm=0.4857, baseline_approach_hz=26.5, w_mean_frac=-0.03008, w_mean_frac_ref=-0.01972, rise_1_avoid_index=3.246, rise_1_avoid_component_hz=-0.4034, rise_1=3.649, rise_1_norm=0.1377, n_relearn=3, scorable=3

### Labels (c1s1)

- rung_a: PASS
- rung3: PASS
- bidir: FAIL (NEEDS-SPIKING)

### End-of-arm latch (c1s1; reported, never gated)

- rung_a rung_a learn dc2: latched=False kc_paired=0 appr_frac=0.3532
- rung3 rung3 interfere dc2: latched=False kc_paired=0 appr_frac=0.3029
- rung3 rung3 interfere d: latched=False kc_paired=0 appr_frac=0.3482
- bidir_timed bidir bidir_timed dc2: latched=False kc_paired=0 appr_frac=0.1982
- bidir_depress bidir bidir_depress dc2: latched=False kc_paired=0 appr_frac=0.3119

### Guard summary (c1s1)

- bidir_depress/bidir_depress: rows 27, max active ratio 1.008, MBON06 ratio [n/a, n/a], breaches 0, aborted False
- bidir_timed/bidir_timed: rows 27, max active ratio 1, MBON06 ratio [n/a, n/a], breaches 0, aborted False
- rung3/interfere: rows 57, max active ratio 1.007, MBON06 ratio [n/a, n/a], breaches 0, aborted False
- rung3/lesion: rows 57, max active ratio 1, MBON06 ratio [n/a, n/a], breaches 0, aborted False
- rung3/retain: rows 39, max active ratio 1.006, MBON06 ratio [n/a, n/a], breaches 0, aborted False
- rung_a/frozen: rows 81, max active ratio 1, MBON06 ratio [n/a, n/a], breaches 0, aborted False
- rung_a/learn: rows 51, max active ratio 1.006, MBON06 ratio [n/a, n/a], breaches 0, aborted False
- rung_a/lesion: rows 51, max active ratio 1, MBON06 ratio [n/a, n/a], breaches 0, aborted False
- rung_a/reversed: rows 51, max active ratio 1.001, MBON06 ratio [n/a, n/a], breaches 0, aborted False
- rung_a/shuffle: rows 51, max active ratio 1.002, MBON06 ratio [n/a, n/a], breaches 0, aborted False

### Numbers (c1s2)

- rung_a: test_trial=30, baseline_approach_hz=26.5, approach_drop_hz=15.24, norm_approach_drop=0.5752, threshold_norm=0.4857, lesion_norm_approach_drop=0, shuffle_norm_approach_drop=0.01624, d_approach_drop_hz=0.8947, reversed_d_approach_drop_hz=14.67, dc2_avoid_index_shift=16.01, dc2_avoid_component_shift_hz=0.7708, lesion_dc2_avoid_index_shift=0, shuffle_dc2_avoid_index_shift=0.4371, d_avoid_index_shift=0.9974, da1_avoid_index_shift=0.2168, reversed_d_avoid_index_shift=15.94, w_mean_frac=-0.02017, w_mean_frac_ref=-0.01972
- rung3: trained_d_approach_drop_hz=14.2, naive_d_approach_drop_hz=14.67, baseline_approach_d_hz=24.85, naive_norm_approach_drop=0.5902, ratio=0.9684, ref_ratio=0.8581, threshold_ratio=0.4291, naive_min_norm=0.4673, trained_d_avoid_index_shift=15.49, trained_d_avoid_component_shift_hz=1.288, naive_d_avoid_index_shift=15.94, naive_d_avoid_component_shift_hz=1.277, w_mean_frac=-0.02977, w_mean_frac_ref=-0.02791
- bidir: f1_min_rise_norm=0.4857, baseline_approach_hz=26.5, w_mean_frac=-0.02987, w_mean_frac_ref=-0.01972, rise_1_avoid_index=3.25, rise_1_avoid_component_hz=-0.3913, rise_1=3.641, rise_1_norm=0.1374, n_relearn=3, scorable=3

### Labels (c1s2)

- rung_a: PASS
- rung3: PASS
- bidir: FAIL (NEEDS-SPIKING)

### End-of-arm latch (c1s2; reported, never gated)

- rung_a rung_a learn dc2: latched=False kc_paired=0 appr_frac=0.3532
- rung3 rung3 interfere dc2: latched=False kc_paired=0 appr_frac=0.3029
- rung3 rung3 interfere d: latched=False kc_paired=0 appr_frac=0.3482
- bidir_timed bidir bidir_timed dc2: latched=False kc_paired=0 appr_frac=0.198
- bidir_depress bidir bidir_depress dc2: latched=False kc_paired=0 appr_frac=0.3119

### Guard summary (c1s2)

- bidir_depress/bidir_depress: rows 27, max active ratio 1.009, MBON06 ratio [n/a, n/a], breaches 0, aborted False
- bidir_timed/bidir_timed: rows 27, max active ratio 1, MBON06 ratio [n/a, n/a], breaches 0, aborted False
- rung3/interfere: rows 57, max active ratio 1.009, MBON06 ratio [n/a, n/a], breaches 0, aborted False
- rung3/lesion: rows 57, max active ratio 1, MBON06 ratio [n/a, n/a], breaches 0, aborted False
- rung3/retain: rows 39, max active ratio 1.008, MBON06 ratio [n/a, n/a], breaches 0, aborted False
- rung_a/frozen: rows 81, max active ratio 1, MBON06 ratio [n/a, n/a], breaches 0, aborted False
- rung_a/learn: rows 51, max active ratio 1.008, MBON06 ratio [n/a, n/a], breaches 0, aborted False
- rung_a/lesion: rows 51, max active ratio 1, MBON06 ratio [n/a, n/a], breaches 0, aborted False
- rung_a/reversed: rows 51, max active ratio 1.001, MBON06 ratio [n/a, n/a], breaches 0, aborted False
- rung_a/shuffle: rows 51, max active ratio 1.002, MBON06 ratio [n/a, n/a], breaches 0, aborted False

### Numbers (c1s3)

- rung_a: test_trial=30, baseline_approach_hz=26.5, approach_drop_hz=15.24, norm_approach_drop=0.5751, threshold_norm=0.4857, lesion_norm_approach_drop=0, shuffle_norm_approach_drop=0.01624, d_approach_drop_hz=0.894, reversed_d_approach_drop_hz=14.67, dc2_avoid_index_shift=16.02, dc2_avoid_component_shift_hz=0.7789, lesion_dc2_avoid_index_shift=0, shuffle_dc2_avoid_index_shift=0.4371, d_avoid_index_shift=0.9849, da1_avoid_index_shift=0.2168, reversed_d_avoid_index_shift=15.97, w_mean_frac=-0.02014, w_mean_frac_ref=-0.01972
- rung3: trained_d_approach_drop_hz=14.2, naive_d_approach_drop_hz=14.67, baseline_approach_d_hz=24.85, naive_norm_approach_drop=0.5901, ratio=0.9684, ref_ratio=0.8581, threshold_ratio=0.4291, naive_min_norm=0.4673, trained_d_avoid_index_shift=15.52, trained_d_avoid_component_shift_hz=1.321, naive_d_avoid_index_shift=15.97, naive_d_avoid_component_shift_hz=1.3, w_mean_frac=-0.02971, w_mean_frac_ref=-0.02791
- bidir: f1_min_rise_norm=0.4857, baseline_approach_hz=26.5, w_mean_frac=-0.02934, w_mean_frac_ref=-0.01972, rise_1_avoid_index=3.279, rise_1_avoid_component_hz=-0.3439, rise_1=3.623, rise_1_norm=0.1367, n_relearn=3, scorable=3

### Labels (c1s3)

- rung_a: PASS
- rung3: PASS
- bidir: FAIL (NEEDS-SPIKING)

### End-of-arm latch (c1s3; reported, never gated)

- rung_a rung_a learn dc2: latched=False kc_paired=0 appr_frac=0.3532
- rung3 rung3 interfere dc2: latched=False kc_paired=0 appr_frac=0.3029
- rung3 rung3 interfere d: latched=False kc_paired=0 appr_frac=0.3483
- bidir_timed bidir bidir_timed dc2: latched=False kc_paired=0 appr_frac=0.1957
- bidir_depress bidir bidir_depress dc2: latched=False kc_paired=0 appr_frac=0.3119

### Guard summary (c1s3)

- bidir_depress/bidir_depress: rows 27, max active ratio 1.008, MBON06 ratio [n/a, n/a], breaches 0, aborted False
- bidir_timed/bidir_timed: rows 27, max active ratio 1, MBON06 ratio [n/a, n/a], breaches 0, aborted False
- rung3/interfere: rows 57, max active ratio 1.006, MBON06 ratio [n/a, n/a], breaches 0, aborted False
- rung3/lesion: rows 57, max active ratio 1, MBON06 ratio [n/a, n/a], breaches 0, aborted False
- rung3/retain: rows 39, max active ratio 1.005, MBON06 ratio [n/a, n/a], breaches 0, aborted False
- rung_a/frozen: rows 81, max active ratio 1, MBON06 ratio [n/a, n/a], breaches 0, aborted False
- rung_a/learn: rows 51, max active ratio 1.005, MBON06 ratio [n/a, n/a], breaches 0, aborted False
- rung_a/lesion: rows 51, max active ratio 1, MBON06 ratio [n/a, n/a], breaches 0, aborted False
- rung_a/reversed: rows 51, max active ratio 1.001, MBON06 ratio [n/a, n/a], breaches 0, aborted False
- rung_a/shuffle: rows 51, max active ratio 1.002, MBON06 ratio [n/a, n/a], breaches 0, aborted False

### Numbers (c1s4)

- rung_a: test_trial=30, baseline_approach_hz=26.5, approach_drop_hz=15.24, norm_approach_drop=0.5752, threshold_norm=0.4857, lesion_norm_approach_drop=0, shuffle_norm_approach_drop=0.01624, d_approach_drop_hz=0.8943, reversed_d_approach_drop_hz=14.67, dc2_avoid_index_shift=16.01, dc2_avoid_component_shift_hz=0.7712, lesion_dc2_avoid_index_shift=0, shuffle_dc2_avoid_index_shift=0.4371, d_avoid_index_shift=0.9851, da1_avoid_index_shift=0.2168, reversed_d_avoid_index_shift=15.94, w_mean_frac=-0.02017, w_mean_frac_ref=-0.01972
- rung3: trained_d_approach_drop_hz=14.2, naive_d_approach_drop_hz=14.67, baseline_approach_d_hz=24.85, naive_norm_approach_drop=0.5902, ratio=0.9684, ref_ratio=0.8581, threshold_ratio=0.4291, naive_min_norm=0.4673, trained_d_avoid_index_shift=15.5, trained_d_avoid_component_shift_hz=1.298, naive_d_avoid_index_shift=15.94, naive_d_avoid_component_shift_hz=1.277, w_mean_frac=-0.02977, w_mean_frac_ref=-0.02791
- bidir: f1_min_rise_norm=0.4857, baseline_approach_hz=26.5, w_mean_frac=-0.02987, w_mean_frac_ref=-0.01972, rise_1_avoid_index=3.251, rise_1_avoid_component_hz=-0.3921, rise_1=3.643, rise_1_norm=0.1375, n_relearn=3, scorable=3

### Labels (c1s4)

- rung_a: PASS
- rung3: PASS
- bidir: FAIL (NEEDS-SPIKING)

### End-of-arm latch (c1s4; reported, never gated)

- rung_a rung_a learn dc2: latched=False kc_paired=0 appr_frac=0.3533
- rung3 rung3 interfere dc2: latched=False kc_paired=0 appr_frac=0.303
- rung3 rung3 interfere d: latched=False kc_paired=0 appr_frac=0.3483
- bidir_timed bidir bidir_timed dc2: latched=False kc_paired=0 appr_frac=0.1971
- bidir_depress bidir bidir_depress dc2: latched=False kc_paired=0 appr_frac=0.312

### Guard summary (c1s4)

- bidir_depress/bidir_depress: rows 27, max active ratio 1.009, MBON06 ratio [n/a, n/a], breaches 0, aborted False
- bidir_timed/bidir_timed: rows 27, max active ratio 1, MBON06 ratio [n/a, n/a], breaches 0, aborted False
- rung3/interfere: rows 57, max active ratio 1.009, MBON06 ratio [n/a, n/a], breaches 0, aborted False
- rung3/lesion: rows 57, max active ratio 1, MBON06 ratio [n/a, n/a], breaches 0, aborted False
- rung3/retain: rows 39, max active ratio 1.008, MBON06 ratio [n/a, n/a], breaches 0, aborted False
- rung_a/frozen: rows 81, max active ratio 1, MBON06 ratio [n/a, n/a], breaches 0, aborted False
- rung_a/learn: rows 51, max active ratio 1.008, MBON06 ratio [n/a, n/a], breaches 0, aborted False
- rung_a/lesion: rows 51, max active ratio 1, MBON06 ratio [n/a, n/a], breaches 0, aborted False
- rung_a/reversed: rows 51, max active ratio 1.001, MBON06 ratio [n/a, n/a], breaches 0, aborted False
- rung_a/shuffle: rows 51, max active ratio 1.002, MBON06 ratio [n/a, n/a], breaches 0, aborted False

