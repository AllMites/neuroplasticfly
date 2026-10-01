# Provenance of paper 1 (release v0.2)

The preregistrations, results and code of paper 1 were committed in a private research repository, so the
commit hashes in Table S1 of the paper do not resolve on GitHub. This file records each cited commit, its time
(UTC+8, self-attested) and the files it touched that ship here. For every file, *identical* means the shipped
copy has the same content as the committed one (line endings normalised). Full hashes are in PROVENANCE.json;
`python reproduce_paper1.py --verify` checks the shipped files against them, against the hashes recorded in each
run's meta row and against the raw-log hashes in the EVALUATION.md reports.

A file touched by several commits matches only its last version: earlier commits show *changed later*.

## Calibration of the rate model

| Commit | Time | Subject | Files (shipped) |
|---|---|---|---|
| 1a29779 | 2026-09-25 08:35 | prereg: chunk 0 - calibrate the rate base to the stock LIF (path A) | PREREGISTER_rate_chunk0_base.md (changed later) |
| b554245 | 2026-09-25 08:54 | prereg amendment 1: chunk 0 reference = eln8 LIF; latch invariant relative to the LIF (D53 | PREREGISTER_rate_chunk0_base.md (identical) |
| 7839719 | 2026-09-25 12:27 | rate chunk 0: BASE-PASS (lin, 4 global params) against eln8 LIF | rate/regress/c0_base.json (identical); rate/regress/c0_record.json (identical); results/rate_chunk0/RESULT.md (identical); results/rate_chunk0/result.json (identical) |

## Reward circuit without fitting

| Commit | Time | Subject | Files (shipped) |
|---|---|---|---|
| 194be67 | 2026-09-24 20:37 | prereg: rate model chunk-1 unfitted control (path 2 phase 3) | PREREGISTER_rate_chunk1_control.md (identical) |
| 2c0b86d | 2026-09-25 12:28 | prereg note: chunk-1 unfitted control rerun on the chunk-0 base | PREREGISTER_rate_chunk1_control_c0.md (identical) |
| 9d03f3b | 2026-09-25 12:30 | rate: suite switches to the chunk-0 base; chunk-1 control rerun on it | rate/chunk1_control.py (identical); rate/chunk1_fit.py (changed later); rate/chunk1_loops.py (identical); rate/regress/c1_record.json (identical); rate/suite.py (identical); rate/test_suite.py (identical); results/rate_chunk1_control_c0/RESULT.md (identical); results/rate_chunk1_control_c0/result.json (identical) |

## Reward fit

| Commit | Time | Subject | Files (shipped) |
|---|---|---|---|
| 2838bfe | 2026-09-24 22:15 | prereg: rate model chunk-1 fit (per-edge-type gains, g* on cap grid, bound 18.287) | PREREGISTER_rate_chunk1_fit.md (changed later) |
| aca91ff | 2026-09-24 22:27 | prereg amendment 1: dead ReLU gradient -> surrogate gradient (backward only) | PREREGISTER_rate_chunk1_fit.md (changed later) |
| e8ddc3c | 2026-09-24 22:29 | prereg amendment 2: surrogate confined to owned target neurons (gate caught explosion) | PREREGISTER_rate_chunk1_fit.md (identical) |
| ee4ecbf | 2026-09-25 12:55 | prereg note: chunk-1 fit rerun on the chunk-0 base | PREREGISTER_rate_chunk1_fit_c0.md (identical) |
| db4fcfb | 2026-09-25 15:22 | rate: chunk-1 fit on chunk-0 base -> FIT-NO-TRANSFER, g*=16, CONSISTENT | results/rate_chunk1_fit_c0/RESULT.md (identical); results/rate_chunk1_fit_c0/cells/cap002.000_s0.json (identical); results/rate_chunk1_fit_c0/cells/cap002.000_s1.json (identical); results/rate_chunk1_fit_c0/cells/cap002.000_s2.json (identical); results/rate_chunk1_fit_c0/cells/cap002.000_s3.json (identical); results/rate_chunk1_fit_c0/cells/cap002.000_s4.json (identical); results/rate_chunk1_fit_c0/cells/cap004.000_s0.json (identical); results/rate_chunk1_fit_c0/cells/cap004.000_s1.json (identical); results/rate_chunk1_fit_c0/cells/cap004.000_s2.json (identical); results/rate_chunk1_fit_c0/cells/cap004.000_s3.json (identical); results/rate_chunk1_fit_c0/cells/cap004.000_s4.json (identical); results/rate_chunk1_fit_c0/cells/cap008.000_s0.json (identical); results/rate_chunk1_fit_c0/cells/cap008.000_s1.json (identical); results/rate_chunk1_fit_c0/cells/cap008.000_s2.json (identical); results/rate_chunk1_fit_c0/cells/cap008.000_s3.json (identical); results/rate_chunk1_fit_c0/cells/cap008.000_s4.json (identical); results/rate_chunk1_fit_c0/cells/cap016.000_s0.json (identical); results/rate_chunk1_fit_c0/cells/cap016.000_s1.json (identical); results/rate_chunk1_fit_c0/cells/cap016.000_s2.json (identical); results/rate_chunk1_fit_c0/cells/cap016.000_s3.json (identical); results/rate_chunk1_fit_c0/cells/cap016.000_s4.json (identical); results/rate_chunk1_fit_c0/cells/cap018.287_s0.json (identical); results/rate_chunk1_fit_c0/cells/cap018.287_s1.json (identical); results/rate_chunk1_fit_c0/cells/cap018.287_s2.json (identical); results/rate_chunk1_fit_c0/cells/cap018.287_s3.json (identical); results/rate_chunk1_fit_c0/cells/cap018.287_s4.json (identical); results/rate_chunk1_fit_c0/cells/cap032.000_s0.json (identical); results/rate_chunk1_fit_c0/cells/cap032.000_s1.json (identical); results/rate_chunk1_fit_c0/cells/cap032.000_s2.json (identical); results/rate_chunk1_fit_c0/cells/cap032.000_s3.json (identical); results/rate_chunk1_fit_c0/cells/cap032.000_s4.json (identical); results/rate_chunk1_fit_c0/cells/cap064.000_s0.json (identical); results/rate_chunk1_fit_c0/cells/cap064.000_s1.json (identical); results/rate_chunk1_fit_c0/cells/cap064.000_s2.json (identical); results/rate_chunk1_fit_c0/cells/cap064.000_s3.json (identical); results/rate_chunk1_fit_c0/cells/cap064.000_s4.json (identical); results/rate_chunk1_fit_c0/cells/cap128.000_s0.json (identical); results/rate_chunk1_fit_c0/cells/cap128.000_s1.json (identical); results/rate_chunk1_fit_c0/cells/cap128.000_s2.json (identical); results/rate_chunk1_fit_c0/cells/cap128.000_s3.json (identical); results/rate_chunk1_fit_c0/cells/cap128.000_s4.json (identical); results/rate_chunk1_fit_c0/cells/cap256.000_s0.json (identical); results/rate_chunk1_fit_c0/cells/cap256.000_s1.json (identical); results/rate_chunk1_fit_c0/cells/cap256.000_s2.json (identical); results/rate_chunk1_fit_c0/cells/cap256.000_s3.json (identical); results/rate_chunk1_fit_c0/cells/cap256.000_s4.json (identical); results/rate_chunk1_fit_c0/groups.json (identical); results/rate_chunk1_fit_c0/result.json (identical); results/rate_chunk1_fit_c0/run.log (identical) |

## Learning tests on the rate model

| Commit | Time | Subject | Files (shipped) |
|---|---|---|---|
| 9dd7298 | 2026-10-01 07:34 | prereg: rate chunk 2 bridge, two arms (c0 + c1s0..4), approach-component normalised gates; | PREREGISTER_rate_chunk2_bridge.md (changed later); rate/chunk2.py (changed later) |
| 2fe2f2d | 2026-10-01 07:39 | prereg A1: pre-launch audit fixes (stop rule, wider code hashes LF-normalised, data hashes | PREREGISTER_rate_chunk2_bridge.md (identical); rate/chunk2.py (changed later) |
| d0b0c92 | 2026-10-01 08:00 | rate: chunk 2 bridge RESULT (prereg 2fe2f2d): rung A PASS, rung 3 PASS, bidir FAIL in both | results/rate_chunk2/RESULT.md (identical); results/rate_chunk2/batch.log (identical); results/rate_chunk2/result.json (identical) |

## Spiking reference for relearning

| Commit | Time | Subject | Files (shipped) |
|---|---|---|---|
| 40c920b | 2026-10-01 08:06 | prereg: LIF open-loop MB bidir reference + re-score of chunk-2 rate bidir (before any code | PREREGISTER_rate_chunk2_bidir_ref.md (changed later) |
| 57af1c1 | 2026-10-01 09:06 | prereg bidir_ref A1: analyzer-review clarifications, scored-seed R, wall time (before any  | PREREGISTER_rate_chunk2_bidir_ref.md (identical) |
| c89244f | 2026-10-01 13:36 | Amendment A2: accept run-10 git_head drift (CHECKSUMS.json text only) for the bidir LIF re | PREREGISTER_rate_chunk2_bidir_ref_A2.md (identical); tools/analyze_bidir_ref_a2.py (identical) |
| d451801 | 2026-10-01 13:43 | bidir LIF reference: RESULT (USABLE, R 0.396; rate FAIL (a)) + independent evaluation | results/rate_chunk2_bidir_ref/EVALUATION.md (identical); results/rate_chunk2_bidir_ref/RESULT.md (identical); results/rate_chunk2_bidir_ref/batch.log (identical) |
| db5a126 | 2026-10-01 13:43 | bidir LIF reference: result.json (chunk-2 precedent tracks it) | results/rate_chunk2_bidir_ref/result.json (identical) |
