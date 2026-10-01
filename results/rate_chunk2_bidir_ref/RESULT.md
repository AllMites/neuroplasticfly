# Chunk-2 bidir vs LIF reference

## Rule

RULE (PREREGISTER_rate_chunk2_bidir_ref.md; thresholds fixed in code before any LIF run):
Quantities (identical to the chunk-2 bidir rule): A = DC2 approach-MBON mean at a T probe; level = -A; rise_k = level(F_k) - level(B_(k-1)) (T0 for k = 1); drop_j = level(F_j) - level(B_j); rise_1_norm = rise_1 / A(T0) of the same run. Cycle k = 2..4 relearns iff rise_1 > 0 and rise_k >= 0.5 x rise_1 and timed drop_(k-1) > 0 and timed drop_(k-1) > depress drop_(k-1) (depress paired with timed by seed for the LIF, by base for the rate model; the rise_1 > 0 clause is chunk 2's, unchanged).
Reference: a LIF seed is SCORED iff neither its timed nor its depress run has a guard abort and both runs have a complete T/F/B curve; any abort or incomplete curve in either rule's run voids that seed (FAIL-UNSTABLE / incomplete: not positive, not relearning). R = mean over the SCORED seeds of the timed rise_1_norm (n scored reported; SD is the sample SD). LIF sanity rule, thresholds absolute out of the 5 seeds: the reference is USABLE iff rise_1_norm > 0 in at least 4 seeds and R > 0 and the LIF timed rule relearns (>= 2 of 3 scorable cycles) in at least 4 seeds. Otherwise the label is NO-REFERENCE: the rate bidir is untestable at the MB level.
Re-scored rate label (bidir vs LIF reference), per base c0, c1s0..c1s4: PASS iff (a) rate timed rise_1_norm >= 0.5 x R and (b) >= 2 of the 3 scorable cycles relearn. Otherwise FAIL naming the failing part ((a), (b) or both). Arm (i) = base c0. Arm (ii) = bases c1s0..c1s4: PASS iff at least ceil(0.8 x n) = 4 of the n = 5 seeds PASS, else FAIL. The chunk-2 bidir label (FAIL) is not changed by this result.
Secondary (reported, never gated): (1) dose-matched rung A: the rung-A learn-arm normalised DC2 approach drop, interpolated linearly between the logged test trials 5..30 as a function of |w_mean_frac| (curve in trial order; if |w| is not non-decreasing over the trials it is flagged dose_curve_nonmonotone and gives no value), at the |w_mean_frac| the timed rule reached at F1, and rise_1_norm / that drop (rate: each base's rung_a log; LIF: paper-0 rung A rows condition_o1s0-4 paired by seed). A dose outside the logged range gives no value (an origin-anchored value, 0 weight change = 0 drop, is shown separately and is not part of the prereg). (2) avoid-MBON component and PAM rate per block (DC2 probe, both rules). (3) |w_mean_frac| per block, both rules. (4) The headroom and baseline / end latch rows of every LIF run and rate log.

## Provenance

- prereg sha256 (LIF runs): c6d9414f396943c62299e81928e1738450264abd8b2b39bffbc43ba90e5a067c
- reference.json sha256: 51f00736c27fdef3d9a08b83b98e55bd954ff39104051253e7466992088e669f
- LIF runs git 57af1c1a2fb6c76282e37b419a8f108734cf048a; analyzer git 57af1c1a2fb6c76282e37b419a8f108734cf048a dirty rate=False learn=False
- rate bidir logs git (by design different): 2fe2f2ddafedd1372988b4d33ffb8a7d12b512b2
- rate log sha256 verified against EVALUATION.md: 18 logs; paper-0 rung A rows verified against reference.json: 5
- LIF data sha256 (reported): ['{"data/brain_gpu.npz": "b6308a8f51c0b68d90c78d337c88418a7ccaf10778a37a6de06b7e3e8653f281", "data/neuron_meta.npz": "7d02505d61ce7a5dd673f3f94db29edd35843a0f6cc7cd9f6d5b9b067b276228"}']

## LIF reference: USABLE

- R = 0.3962, SD = 0.0158, scored 5 of 5 seeds; rise_1_norm > 0 in 5; same sign as R in 5; timed rule relearns in 5 seeds
- unstable seeds: none

| seed | rise_1_norm | relearn cycles (of 3) | relearns | abs w F1 | dose-matched rung A drop | ratio |
|---|---|---|---|---|---|---|
| 0 | 0.375 | 3 | True | 0.004046 | n/a (below-range; origin-anchored 0.3564) | n/a |
| 1 | 0.3846 | 3 | True | 0.00419 | n/a (below-range; origin-anchored 0.3648) | n/a |
| 2 | 0.4038 | 3 | True | 0.004546 | 0.3955 (in-range) | 1.021 |
| 3 | 0.4038 | 3 | True | 0.004315 | n/a (below-range; origin-anchored 0.3895) | n/a |
| 4 | 0.4135 | 3 | True | 0.004523 | 0.4095 (in-range) | 1.01 |

## Re-scored rate label (bidir vs LIF reference)

- arm (i) c0: FAIL (failing (a))
- arm (ii): FAIL (0 of 5 seeds PASS, need 4)

| base | rise_1_norm | threshold 0.5xR | (a) | relearn cycles (of 3) | (b) | label | failing | dose-matched rung A drop | ratio |
|---|---|---|---|---|---|---|---|---|---|
| c0 | 0.1307 | 0.1981 | False | 3 | True | FAIL | (a) | 0.1592 (in-range) | 0.8209 |
| c1s0 | 0.1373 | 0.1981 | False | 3 | True | FAIL | (a) | 0.2557 (in-range) | 0.5372 |
| c1s1 | 0.1377 | 0.1981 | False | 3 | True | FAIL | (a) | 0.259 (in-range) | 0.5317 |
| c1s2 | 0.1374 | 0.1981 | False | 3 | True | FAIL | (a) | 0.2566 (in-range) | 0.5356 |
| c1s3 | 0.1367 | 0.1981 | False | 3 | True | FAIL | (a) | 0.2472 (in-range) | 0.5531 |
| c1s4 | 0.1375 | 0.1981 | False | 3 | True | FAIL | (a) | 0.2568 (in-range) | 0.5354 |

## Headroom and latch rows (reported, never gated)

- LIF seed 0 timed headroom dc2: gate_pass=True approach_ratio=1.06
- LIF seed 0 timed latch baseline dc2: latched=False kc_paired=0 appr_frac=0
- LIF seed 0 timed latch end dc2: latched=False kc_paired=0 appr_frac=0
- LIF seed 0 depress headroom dc2: gate_pass=True approach_ratio=1.06
- LIF seed 0 depress latch baseline dc2: latched=False kc_paired=0 appr_frac=0
- LIF seed 0 depress latch end dc2: latched=False kc_paired=0 appr_frac=0
- LIF seed 1 timed headroom dc2: gate_pass=True approach_ratio=1.06
- LIF seed 1 timed latch baseline dc2: latched=False kc_paired=0 appr_frac=0
- LIF seed 1 timed latch end dc2: latched=False kc_paired=0 appr_frac=0
- LIF seed 1 depress headroom dc2: gate_pass=True approach_ratio=1.06
- LIF seed 1 depress latch baseline dc2: latched=False kc_paired=0 appr_frac=0
- LIF seed 1 depress latch end dc2: latched=False kc_paired=0 appr_frac=0
- LIF seed 2 timed headroom dc2: gate_pass=True approach_ratio=1.06
- LIF seed 2 timed latch baseline dc2: latched=False kc_paired=0 appr_frac=0
- LIF seed 2 timed latch end dc2: latched=False kc_paired=0 appr_frac=0
- LIF seed 2 depress headroom dc2: gate_pass=True approach_ratio=1.06
- LIF seed 2 depress latch baseline dc2: latched=False kc_paired=0 appr_frac=0
- LIF seed 2 depress latch end dc2: latched=False kc_paired=0 appr_frac=0
- LIF seed 3 timed headroom dc2: gate_pass=True approach_ratio=1.06
- LIF seed 3 timed latch baseline dc2: latched=False kc_paired=0 appr_frac=0
- LIF seed 3 timed latch end dc2: latched=False kc_paired=0 appr_frac=0
- LIF seed 3 depress headroom dc2: gate_pass=True approach_ratio=1.06
- LIF seed 3 depress latch baseline dc2: latched=False kc_paired=0 appr_frac=0
- LIF seed 3 depress latch end dc2: latched=False kc_paired=0 appr_frac=0
- LIF seed 4 timed headroom dc2: gate_pass=True approach_ratio=1.06
- LIF seed 4 timed latch baseline dc2: latched=False kc_paired=0 appr_frac=0
- LIF seed 4 timed latch end dc2: latched=False kc_paired=0 appr_frac=0
- LIF seed 4 depress headroom dc2: gate_pass=True approach_ratio=1.06
- LIF seed 4 depress latch baseline dc2: latched=False kc_paired=0 appr_frac=0
- LIF seed 4 depress latch end dc2: latched=False kc_paired=0 appr_frac=0
- rate base c0 timed headroom dc2: gate_pass=True approach_ratio=2.349
- rate base c0 timed latch baseline dc2: latched=False kc_paired=0 appr_frac=0.3227
- rate base c0 timed latch end dc2: latched=False kc_paired=0 appr_frac=0.2885
- rate base c0 depress headroom dc2: gate_pass=True approach_ratio=2.349
- rate base c0 depress latch baseline dc2: latched=False kc_paired=0 appr_frac=0.3227
- rate base c0 depress latch end dc2: latched=False kc_paired=0 appr_frac=0.3122
- rate base c1s0 timed headroom dc2: gate_pass=True approach_ratio=2.35
- rate base c1s0 timed latch baseline dc2: latched=False kc_paired=0 appr_frac=0.323
- rate base c1s0 timed latch end dc2: latched=False kc_paired=0 appr_frac=0.1977
- rate base c1s0 depress headroom dc2: gate_pass=True approach_ratio=2.35
- rate base c1s0 depress latch baseline dc2: latched=False kc_paired=0 appr_frac=0.323
- rate base c1s0 depress latch end dc2: latched=False kc_paired=0 appr_frac=0.3119
- rate base c1s1 timed headroom dc2: gate_pass=True approach_ratio=2.35
- rate base c1s1 timed latch baseline dc2: latched=False kc_paired=0 appr_frac=0.323
- rate base c1s1 timed latch end dc2: latched=False kc_paired=0 appr_frac=0.1982
- rate base c1s1 depress headroom dc2: gate_pass=True approach_ratio=2.35
- rate base c1s1 depress latch baseline dc2: latched=False kc_paired=0 appr_frac=0.323
- rate base c1s1 depress latch end dc2: latched=False kc_paired=0 appr_frac=0.3119
- rate base c1s2 timed headroom dc2: gate_pass=True approach_ratio=2.35
- rate base c1s2 timed latch baseline dc2: latched=False kc_paired=0 appr_frac=0.323
- rate base c1s2 timed latch end dc2: latched=False kc_paired=0 appr_frac=0.198
- rate base c1s2 depress headroom dc2: gate_pass=True approach_ratio=2.35
- rate base c1s2 depress latch baseline dc2: latched=False kc_paired=0 appr_frac=0.323
- rate base c1s2 depress latch end dc2: latched=False kc_paired=0 appr_frac=0.3119
- rate base c1s3 timed headroom dc2: gate_pass=True approach_ratio=2.35
- rate base c1s3 timed latch baseline dc2: latched=False kc_paired=0 appr_frac=0.3229
- rate base c1s3 timed latch end dc2: latched=False kc_paired=0 appr_frac=0.1957
- rate base c1s3 depress headroom dc2: gate_pass=True approach_ratio=2.35
- rate base c1s3 depress latch baseline dc2: latched=False kc_paired=0 appr_frac=0.3229
- rate base c1s3 depress latch end dc2: latched=False kc_paired=0 appr_frac=0.3119
- rate base c1s4 timed headroom dc2: gate_pass=True approach_ratio=2.35
- rate base c1s4 timed latch baseline dc2: latched=False kc_paired=0 appr_frac=0.323
- rate base c1s4 timed latch end dc2: latched=False kc_paired=0 appr_frac=0.1971
- rate base c1s4 depress headroom dc2: gate_pass=True approach_ratio=2.35
- rate base c1s4 depress latch baseline dc2: latched=False kc_paired=0 appr_frac=0.323
- rate base c1s4 depress latch end dc2: latched=False kc_paired=0 appr_frac=0.312

## Secondary (reported, never gated): per block, DC2 probe

- LIF seed 0 timed avoid_mbon_hz: T0=0.3704, F1=0.3704, B1=0.3704, F2=0.3704, B2=0.3704, F3=0.3704, B3=0.3704, F4=0.3704, B4=0.3704
- LIF seed 0 timed pam_hz: T0=0, F1=0, B1=0, F2=0, B2=0, F3=0, B3=0, F4=0, B4=0
- LIF seed 0 timed abs_w_mean_frac: T0=0, F1=0.004046, B1=0.001862, F2=0.006379, B2=0.003989, F3=0.008523, B3=0.006181, F4=0.01039, B4=0.00754
- LIF seed 0 depress avoid_mbon_hz: T0=0.3704, F1=0.3704, B1=0.3704, F2=0.3704, B2=0.3704, F3=0.3704, B3=0.3704, F4=0.3704, B4=0.3704
- LIF seed 0 depress pam_hz: T0=0, F1=0, B1=0, F2=0, B2=0, F3=0, B3=0, F4=0, B4=0
- LIF seed 0 depress abs_w_mean_frac: T0=0, F1=0.003466, B1=0.003231, F2=0.006353, B2=0.005921, F3=0.009081, B3=0.008464, F4=0.01166, B4=0.01087
- LIF seed 1 timed avoid_mbon_hz: T0=0.3704, F1=0.3704, B1=0.3704, F2=0.3704, B2=0.3704, F3=0.3704, B3=0.3704, F4=0.3704, B4=0.3704
- LIF seed 1 timed pam_hz: T0=0, F1=0, B1=0, F2=0, B2=0, F3=0, B3=0, F4=0, B4=0
- LIF seed 1 timed abs_w_mean_frac: T0=0, F1=0.00419, B1=0.001327, F2=0.005763, B2=0.003187, F3=0.007714, B3=0.00485, F4=0.009208, B4=0.006535
- LIF seed 1 depress avoid_mbon_hz: T0=0.3704, F1=0.3704, B1=0.3704, F2=0.3704, B2=0.3704, F3=0.3704, B3=0.3704, F4=0.3704, B4=0.3704
- LIF seed 1 depress pam_hz: T0=0, F1=0, B1=0, F2=0, B2=0, F3=0, B3=0, F4=0, B4=0
- LIF seed 1 depress abs_w_mean_frac: T0=0, F1=0.003321, B1=0.003095, F2=0.006355, B2=0.005923, F3=0.008878, B3=0.008275, F4=0.0114, B4=0.01062
- LIF seed 2 timed avoid_mbon_hz: T0=0.3704, F1=0.3704, B1=0.3704, F2=0.3704, B2=0.3704, F3=0.3704, B3=0.3704, F4=0.3704, B4=0.3704
- LIF seed 2 timed pam_hz: T0=0, F1=0, B1=0, F2=0, B2=0, F3=0, B3=0, F4=0, B4=0
- LIF seed 2 timed abs_w_mean_frac: T0=0, F1=0.004546, B1=0.001584, F2=0.005585, B2=0.002861, F3=0.007189, B3=0.004859, F4=0.009508, B4=0.00808
- LIF seed 2 depress avoid_mbon_hz: T0=0.3704, F1=0.3704, B1=0.3704, F2=0.3704, B2=0.3704, F3=0.3704, B3=0.3704, F4=0.3704, B4=0.3704
- LIF seed 2 depress pam_hz: T0=0, F1=0, B1=0, F2=0, B2=0, F3=0, B3=0, F4=0, B4=0
- LIF seed 2 depress abs_w_mean_frac: T0=0, F1=0.003885, B1=0.003621, F2=0.006959, B2=0.006486, F3=0.009647, B3=0.008991, F4=0.0118, B4=0.011
- LIF seed 3 timed avoid_mbon_hz: T0=0.3704, F1=0.3704, B1=0.3704, F2=0.3704, B2=0.3704, F3=0.3704, B3=0.3704, F4=0.3704, B4=0.3704
- LIF seed 3 timed pam_hz: T0=0, F1=0, B1=0, F2=0, B2=0, F3=0, B3=0, F4=0, B4=0
- LIF seed 3 timed abs_w_mean_frac: T0=0, F1=0.004315, B1=0.00218, F2=0.006844, B2=0.003483, F3=0.008085, B3=0.004958, F4=0.01035, B4=0.008215
- LIF seed 3 depress avoid_mbon_hz: T0=0.3704, F1=0.3704, B1=0.3704, F2=0.3704, B2=0.3704, F3=0.3704, B3=0.3704, F4=0.3704, B4=0.3704
- LIF seed 3 depress pam_hz: T0=0, F1=0, B1=0, F2=0, B2=0, F3=0, B3=0, F4=0, B4=0
- LIF seed 3 depress abs_w_mean_frac: T0=0, F1=0.003622, B1=0.003376, F2=0.006834, B2=0.006369, F3=0.009648, B3=0.008992, F4=0.01204, B4=0.01123
- LIF seed 4 timed avoid_mbon_hz: T0=0.3704, F1=0.3704, B1=0.3704, F2=0.3704, B2=0.3704, F3=0.3704, B3=0.3704, F4=0.3704, B4=0.3704
- LIF seed 4 timed pam_hz: T0=0, F1=0, B1=0, F2=0, B2=0, F3=0, B3=0, F4=0, B4=0
- LIF seed 4 timed abs_w_mean_frac: T0=0, F1=0.004523, B1=0.001956, F2=0.005928, B2=0.002996, F3=0.007078, B3=0.004839, F4=0.009091, B4=0.006995
- LIF seed 4 depress avoid_mbon_hz: T0=0.3704, F1=0.3704, B1=0.3704, F2=0.3704, B2=0.3704, F3=0.3704, B3=0.3704, F4=0.3704, B4=0.3704
- LIF seed 4 depress pam_hz: T0=0, F1=0, B1=0, F2=0, B2=0, F3=0, B3=0, F4=0, B4=0
- LIF seed 4 depress abs_w_mean_frac: T0=0, F1=0.003721, B1=0.003468, F2=0.006824, B2=0.006361, F3=0.009553, B3=0.008904, F4=0.01176, B4=0.01096
- rate base c0 timed avoid_mbon_hz: T0=5.274, F1=5.236, B1=5.163, F2=5.14, B2=5.04, F3=5.035, B3=4.856, F4=4.606, B4=4.237
- rate base c0 timed pam_hz: T0=0.3129, F1=0.3094, B1=0.3054, F2=0.3018, B2=0.2973, F3=0.2923, B3=0.2856, F4=0.2715, B4=0.2574
- rate base c0 timed abs_w_mean_frac: T0=0, F1=0.006004, B1=0.005663, F2=0.01121, B2=0.0104, F3=0.01557, B3=0.01441, F4=0.01912, B4=0.01767
- rate base c0 depress avoid_mbon_hz: T0=5.274, F1=5.296, B1=5.294, F2=5.334, B2=5.323, F3=5.434, B3=5.402, F4=5.577, B4=5.499
- rate base c0 depress pam_hz: T0=0.3129, F1=0.3133, B1=0.3133, F2=0.3135, B2=0.3135, F3=0.3128, B3=0.313, F4=0.3131, B4=0.3126
- rate base c0 depress abs_w_mean_frac: T0=0, F1=0.003964, B1=0.003696, F2=0.007409, B2=0.006906, F3=0.0104, B3=0.009695, F4=0.013, B4=0.01212
- rate base c1s0 timed avoid_mbon_hz: T0=5.274, F1=4.888, B1=3.327, F2=1.541, B2=0.2951, F3=0.2339, B3=0.1755, F4=0.2942, B4=0.2327
- rate base c1s0 timed pam_hz: T0=1.399, F1=1.309, B1=1.15, F2=0.9891, B2=0.9407, F3=0.9357, B3=0.9373, F4=0.9353, B4=0.9371
- rate base c1s0 timed abs_w_mean_frac: T0=0, F1=0.009045, B1=0.0112, F2=0.0187, B2=0.019, F3=0.02499, B3=0.02444, F4=0.02987, B4=0.02896
- rate base c1s0 depress avoid_mbon_hz: T0=5.274, F1=5.288, B1=5.287, F2=5.316, B2=5.307, F3=5.391, B3=5.368, F4=5.522, B4=5.449
- rate base c1s0 depress pam_hz: T0=1.399, F1=1.398, B1=1.398, F2=1.395, B2=1.396, F3=1.391, B3=1.392, F4=1.391, B4=1.39
- rate base c1s0 depress abs_w_mean_frac: T0=0, F1=0.004069, B1=0.003794, F2=0.007605, B2=0.00709, F3=0.01068, B3=0.009952, F4=0.01334, B4=0.01244
- rate base c1s1 timed avoid_mbon_hz: T0=5.274, F1=4.871, B1=3.22, F2=1.413, B2=0.2366, F3=0.2382, B3=0.1798, F4=0.2976, B4=0.2361
- rate base c1s1 timed pam_hz: T0=1.435, F1=1.339, B1=1.17, F2=1.006, B2=0.9638, F3=0.9601, B3=0.962, F4=0.9595, B4=0.9615
- rate base c1s1 timed abs_w_mean_frac: T0=0, F1=0.00915, B1=0.01137, F2=0.01893, B2=0.01922, F3=0.02521, B3=0.02466, F4=0.03008, B4=0.02915
- rate base c1s1 depress avoid_mbon_hz: T0=5.274, F1=5.287, B1=5.286, F2=5.315, B2=5.307, F3=5.389, B3=5.367, F4=5.52, B4=5.447
- rate base c1s1 depress pam_hz: T0=1.435, F1=1.433, B1=1.434, F2=1.43, B2=1.431, F3=1.425, B3=1.427, F4=1.425, B4=1.423
- rate base c1s1 depress abs_w_mean_frac: T0=0, F1=0.004073, B1=0.003798, F2=0.007612, B2=0.007096, F3=0.01069, B3=0.009961, F4=0.01335, B4=0.01245
- rate base c1s2 timed avoid_mbon_hz: T0=5.274, F1=4.883, B1=3.296, F2=1.513, B2=0.2855, F3=0.2409, B3=0.1823, F4=0.301, B4=0.2394
- rate base c1s2 timed pam_hz: T0=1.413, F1=1.321, B1=1.154, F2=0.9888, B2=0.9403, F3=0.9353, B3=0.937, F4=0.9348, B4=0.9367
- rate base c1s2 timed abs_w_mean_frac: T0=0, F1=0.009076, B1=0.01125, F2=0.01875, B2=0.01903, F3=0.025, B3=0.02444, F4=0.02987, B4=0.02894
- rate base c1s2 depress avoid_mbon_hz: T0=5.274, F1=5.287, B1=5.287, F2=5.315, B2=5.307, F3=5.39, B3=5.367, F4=5.522, B4=5.448
- rate base c1s2 depress pam_hz: T0=1.413, F1=1.412, B1=1.412, F2=1.409, B2=1.41, F3=1.404, B3=1.405, F4=1.403, B4=1.402
- rate base c1s2 depress abs_w_mean_frac: T0=0, F1=0.004071, B1=0.003795, F2=0.007608, B2=0.007092, F3=0.01068, B3=0.009955, F4=0.01335, B4=0.01244
- rate base c1s3 timed avoid_mbon_hz: T0=5.274, F1=4.93, B1=3.622, F2=1.851, B2=0.5103, F3=0.2037, B3=0.1466, F4=0.2669, B4=0.2056
- rate base c1s3 timed pam_hz: T0=1.304, F1=1.228, B1=1.096, F2=0.947, B2=0.8827, F3=0.8715, B3=0.8727, F4=0.8715, B4=0.8727
- rate base c1s3 timed abs_w_mean_frac: T0=0, F1=0.008777, B1=0.01075, F2=0.0181, B2=0.01845, F3=0.02441, B3=0.02388, F4=0.02934, B4=0.02847
- rate base c1s3 depress avoid_mbon_hz: T0=5.274, F1=5.288, B1=5.287, F2=5.317, B2=5.309, F3=5.395, B3=5.37, F4=5.527, B4=5.453
- rate base c1s3 depress pam_hz: T0=1.304, F1=1.303, B1=1.304, F2=1.302, B2=1.302, F3=1.299, B3=1.299, F4=1.3, B4=1.298
- rate base c1s3 depress abs_w_mean_frac: T0=0, F1=0.00406, B1=0.003785, F2=0.007587, B2=0.007073, F3=0.01065, B3=0.009929, F4=0.01331, B4=0.01241
- rate base c1s4 timed avoid_mbon_hz: T0=5.274, F1=4.882, B1=3.302, F2=1.489, B2=0.2673, F3=0.2097, B3=0.1517, F4=0.2713, B4=0.2097
- rate base c1s4 timed pam_hz: T0=1.412, F1=1.318, B1=1.148, F2=0.9783, B2=0.9282, F3=0.9222, B3=0.9241, F4=0.9214, B4=0.9235
- rate base c1s4 timed abs_w_mean_frac: T0=0, F1=0.009081, B1=0.01126, F2=0.01876, B2=0.01903, F3=0.025, B3=0.02444, F4=0.02987, B4=0.02895
- rate base c1s4 depress avoid_mbon_hz: T0=5.274, F1=5.287, B1=5.287, F2=5.315, B2=5.307, F3=5.39, B3=5.367, F4=5.522, B4=5.448
- rate base c1s4 depress pam_hz: T0=1.412, F1=1.41, B1=1.41, F2=1.406, B2=1.407, F3=1.4, B3=1.402, F4=1.399, B4=1.398
- rate base c1s4 depress abs_w_mean_frac: T0=0, F1=0.004071, B1=0.003796, F2=0.007608, B2=0.007093, F3=0.01068, B3=0.009956, F4=0.01335, B4=0.01244

## Amendment A2 (provenance only)
- run git_heads: H1 57af1c1 (9 runs), H2 66b4f44 (bidir_depress_lifs4.jsonl)
- diff H1..H2: data/CHECKSUMS.json
- analyzer HEAD (real): c89244f; diff H1..HEAD: PREREGISTER_rate_chunk2_bidir_ref_A2.md, data/CHECKSUMS.json, tools/analyze_bidir_ref_a2.py
- analyzer state pinned to H1 per A2 rule 3; amendment sha256 5e08590e62aa
