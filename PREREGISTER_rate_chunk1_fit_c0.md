# Prereg note: chunk-1 fit, rerun on the chunk-0 base

Written 2026-09-25 before any fit on this base. Procedure = PREREGISTER_rate_chunk1_fit.md @ `2838bfe` +
amendments 1-2, unchanged: owned types FDA-I, FDA-II, PAM01-15; per-edge-type gains in [0, cap];
caps {2, 4, 8, 16, 18.287, 32, 64, 128, 256} x seeds 0-4; 400 Adam iterations, lr 0.1, lambda 0.01;
fit rows F1-F5 only; g* = smallest cap with >= 4/5 seeds passing F1-F5; held-out H1-H6 scored at g* only;
bound 18.287 (flyvis 99th pct, D47); labels UNFITTABLE / IMPLAUSIBLE / FIT-NO-TRANSFER / PASS, each
tagged CONSISTENT / INCONSISTENT by the same rule.

Only the base changes: rate/regress/c0_base.json (chunk 0 BASE-PASS `7839719`), via suite.build(base="c0").
The amendment-1 surrogate scale is defined as the model's rest-to-threshold distance; on this base that is
|bias| = 9.519 Hz (arm A: 35.028). Same definition, new constant.

Gate before launch (as amendment 2): G-grad on cap 18.287 seed 0, 10 iterations, nothing saved: fit-term
gradient finite and nonzero, fit term falls. If it fails, stop and report; no batch.

Output: results/rate_chunk1_fit_c0/. The arm-A result (001a54a) and pass 2 are untouched. This is the
primary answer for path A; the arm-A result becomes history.

Seen before writing: on this base, the unfitted control (9d03f3b): reward PAMs 0 Hz in every condition;
Fox drive gives FDA-I 6.6 Hz; sugar gives Fox 10.6 Hz, FDA-I 1.0 Hz. The test_suite stress check (no
threshold + gain 1e4 on reward pools) gains F1/F2/F4/H1 and breaks H3. No fit has been run on this base.
