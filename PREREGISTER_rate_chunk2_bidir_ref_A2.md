# Amendment A2 to PREREGISTER_rate_chunk2_bidir_ref.md

Date: 2026-10-01, after all 10 LIF runs finished and before any analysis result existed (the first
`--analyze-bidir-ref` refused on provenance and wrote nothing). Decided by the user (Daniel Asis).

Why a separate file: the prereg sha256 and `rate/chunk2.py` sha256 are pinned in every LIF meta row, so neither
can be edited without voiding the batch. A1 lives in the prereg because it predates the runs; A2 cannot.

## What happened
- Runs 1-9 (timed/depress lifs0..3, timed lifs4) recorded git_head 57af1c1a2fb6c76282e37b419a8f108734cf048a.
- Run 10 (depress lifs4, started 12:48:51) recorded 66b4f44d5f3ad190ffeb7f9ce17c42150722b6a6, committed 12:39:28 by a
  parallel session working in the same repo.
- `git diff 57af1c1 66b4f44` = one line in `data/CHECKSUMS.json` ("sources.annotations": provenance text for the
  flywire_annotations v3.1.0 pin). Not in CODE_FILES, not in DATA_FILES, not read by any simulation code.
- All 10 meta rows carry identical code_sha256 (11 files), data_sha256 (brain_gpu.npz, neuron_meta.npz) and
  prereg_sha256, and git_dirty_rate = git_dirty_learn = False.

## Rule (overrides "one known git_head equal to the analyzer's" in the provenance check, for this batch only)
1. The runs may carry exactly two git_heads, H1 = 57af1c1 and H2 = 66b4f44, H2 only on bidir_depress_lifs4.
2. Accepted only if: H1 is an ancestor of H2; `git diff --name-only H1 H2` touches no CODE_FILES or DATA_FILES
   entry and no file under rate/ or learn/; code_sha256, data_sha256 and prereg_sha256 are identical across the
   10 runs. Any failure refuses, as before.
3. The analyzer HEAD may be later than H1 (this amendment and its wrapper are committed after the runs) provided
   `git diff --name-only H1 HEAD` touches no CODE_FILES entry and nothing under rate/ or learn/; the code_sha256
   check against disk still runs unchanged.
4. Every other check in `check_lif_provenance`, `verify_rate_logs`, `verify_sources` and the scoring code runs
   unchanged. The analysis is run once through `tools/analyze_bidir_ref_a2.py`, which verifies 1-3, then calls the
   unmodified `analyze_bidir_ref`. result.json gets an `amendment_A2` block (both heads, the diffs, the real
   analyzer HEAD) and RESULT.md an A2 section.
5. No run is re-run. The label rules are untouched.

## Disclosure (H, 2026-10-01 13:05)
While checking that run 10's log was being written, Claude printed some learning rows of bidir_timed_lifs4
(T0/F1/B1 avoid_index, approach Hz) before the analysis, breaching "Nobody reads LIF learning rows before the
analysis". Nothing was computed from them and no rule, threshold or code changed after; A2 changes provenance
handling only.

## Prevention
Parallel sessions must not commit to flychess while a preregistered batch runs (or the batch should run from its
own worktree).
