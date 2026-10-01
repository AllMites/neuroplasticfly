# Prereg note: chunk-1 unfitted control, rerun on the chunk-0 base

Written 2026-09-25 before the rerun. Procedure = PREREGISTER_rate_chunk1_control.md @ `194be67`, unchanged:
same five conditions (BASE, FOX, SUGAR, BITTER, SUGAR+Fox-silenced), one deterministic run each, same
11-row battery (rate/chunk1_targets.md), same labels (UNSTABLE / PASSES-UNFITTED / FAILS-UNFITTED).
No parameter is fitted or tuned.

Only the base changes: one arm, "C0" = rate/regress/c0_base.json (eln8 W, family lin, w_scale 0.00494,
bias -9.52, tau 17.0 ms, r_max 124.5 Hz; chunk 0 BASE-PASS `7839719`). Stability rule = arm A's (finite,
< 1 % of neurons at >= 0.99 r_max in every condition), since C0 has a finite r_max.
Arms A and B are not re-run; their committed result.json is untouched. Output: results/rate_chunk1_control_c0/.

The C0 rows become the new rate/regress/c1_record.json (the old record was measured on arm A).

Descriptive, no label: GRN(sugar) -> Fox synapse count and share of Fox's input (floor-5 W), next to
FDA-I -> PAM's 0.81 %; mean Fox / FDA-I / FDA-II rates per condition, to say where sugar stops.

Seen before writing: on this base only the latch check touched sugar (1 s at 150 Hz, then off: 0 neurons
above 1 Hz after offset). No during-stimulus sugar, Fox or PAM rate on this base has been looked at.
