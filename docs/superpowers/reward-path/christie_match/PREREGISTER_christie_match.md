# PREREGISTER christie_match: protocol-matched rerun of Christie et al. 2026 Fig 6A (2026-09-29)

Written 2026-09-29, before any code for this run exists and before any GPU run that drives Christie's GINs or uses 30 trials.
Trigger: private correspondence (2026-09-29) about the difference between the preprint's W_SYN 0.275 mV and the weights
used by Christie et al. (2026), Fig 6A-B. Batch run, one analysis pass; every rule below is fixed before launch.
[Public copy: correspondence details redacted 2026-09-29; all rules and thresholds verbatim.]

**Why this run exists.** `results/wsyn_grid/RESULT.md` labelled Christie C1 "NOT REPRODUCED" (PAM 0/307 at W 0.39, floors 5
and 3; <= 7.8/307 at floor 1). That was NOT a like-for-like test. Christie's Fig 6A legend and STAR Methods (PMC12869359, read
2026-09-29) differ from the grid H1 in two ways, a third is unknown:
| | Christie Fig 6A | wsyn_grid H1 |
|---|---|---|
| Driven cells | unilateral sugar GRNs **and GINs** (Data S4: 7 x 2N, 4 x 3N, 3 x 4N) | left sugar GRNs only (67) |
| Trials | **30 x 1000 ms** | 5 x 1000 ms |
| GRN rates | 10-200 Hz, 10 Hz steps (Data S3) | 150, 200 Hz |
| Weights | 0.37, 0.38, 0.382, 0.386, 0.39 mV | 0.275-0.42 grid |
| Synapse floor | **not stated** (NOT SURE) | 5 / 3 / 1 |
| "Responding" | mean rate over the 30 trials > 0 | same criterion, 5 trials |
Until this run, C1 is UNTESTED, not refuted.

## Facts measured before writing (CPU only, 2026-09-29)
- Data S4 GIN root IDs: all 14 left-side IDs resolve in `data/root_ids_sorted.npy` (v783 build), all `side == left`.
  Types: 2N = AN_GNG_30, CB0366, CB0616, CB0062, CB0499, CB0008, CB0192 (Clavicle, FMIN, G2N-1, Phantom, Rattle, Usnea,
  Zorro); 3N = DNge174, DNge173, CB0038, CB0051 (Bract1, Bract2, Fdg, Sternum); 4N = DNge080, CB0493, CB0553 (Rounddown,
  Roundtree, Roundup). Edges from the 67 left sugar GRNs (floor 5): 2N receive 13-53 each; 3N and 4N receive 0 (higher order),
  consistent with their names.
- Data S3 (Christie's own result) is ALL-OR-NONE and non-monotonic in rate: at each W, Fox-intact counts are either 0-11 or
  186-203 of ~307, and flip between neighbouring rates (W 0.37: 191 at 160 Hz, 0 at 170, 5 at 180, 8 at 190, 189 at 200).
  Fox-silenced rows are 0 everywhere except one 6 (W 0.386, 20 Hz).
- Wall cost from `results/wsyn_grid/ledger.jsonl` (stock regime, 5 seeds x 1000 ms batched): ~1.1 s wall per simulated second
  at floor 5, ~2.7 s at floor 1.

## Model (fixed)
Stock regime exactly as `learn/wsyn_grid_cell.py --regime stock`: Shiu 2024 LIF constants and Shiu signs (no ELN_NEGATE,
no PN_KC_GAIN, no plasticity), dt 0.1 ms, no background current. W_SYN set as `G.W_SYN = w` before `GpuSim()` and asserted.
Brains: `data/brain_gpu.npz` (floor 5, canonical) and `data/brain_gpu_min1.npz` (floor 1 = every proofread synapse, which is
what Shiu's released model keeps; `project_validation_batch`). `sim.net.min_syn` and `n_edges` asserted per process.
Silencing via `run_batch(silence=)` (gates output only; `regime/test_silence.py`).

## Design (fixed; nothing added or moved after any output)
- **Drive (primary, "GRN+GIN")**: the 67 left sugar GRNs (`C.SUGAR` with side == left, as the grid) AND the 14 left GINs
  above, all as independent Poisson at the SAME rate r. ASSUMPTION, NOT SURE: Christie does not state the GIN rate; "same rate
  as the GRNs" is the plain reading of "stimulation of unilateral GRNs and GINs ... at GRN firing rates". Asked of the
  authors 2026-09-29; if they answer BEFORE launch, the GIN rate is changed by a dated amendment below and nothing else moves.
  If she answers AFTER launch, the run stands as registered and her protocol is a separate, new prereg.
- **r** in {10, 20, ..., 200} Hz (20 rates, Christie's Data S3 columns).
- **W** in {0.37, 0.38, 0.382, 0.386, 0.39} (Christie) + 0.275 (canonical control). 6 values.
- **Floor** in {1, 5}. Floor 1 is the Christie-faithful arm (primary); floor 5 is our preprint brain.
- **Fox** intact / silenced (CB0525, both cells, as the reconciliation).
- **Trials**: seeds 0-29, 1000 ms each, one condition = 30 trials (batched; chunked into 2 x 15 if memory requires; chunking
  changes nothing since seeds are independent). Plus **base** (undriven, 30 x 1000 ms) per (W, floor).
- **Ablation "GRN-only"** (isolates the GIN difference): W 0.39, both floors, Fox intact, all 20 rates, GRNs only.
- Main: 6 x 2 x 20 x 2 = 480 conditions; ablation 40; base 12. Simulated time 532 x 30 s = 15,960 s.
  Projected ~1.1 s/s (floor 5) and ~2.7 s/s (floor 1) -> ~8.5 h. **Cap 600 GPU-min.** If the smoke timing projects above the
  cap, drop in this fixed order until under it: (1) Fox-silenced at W 0.38 and 0.382, floor 5; (2) same at floor 1;
  (3) Fox-silenced at r <= 60 Hz everywhere. Dropped conditions are listed as NOT RUN, never imputed.

## Golden check (before the main batch; abort on failure)
Re-run grid cell (W 0.39, floor 1) condition sugarL200, GRN-only, seeds 0-4, 1000 ms with the new script: PAM n_responsive per
seed must equal `results/wsyn_grid/cell_w0.390_m1.json` exactly ([7 11 7 14 0], mean 7.8). This proves the new script is
the grid's H1 pipeline plus the registered changes and nothing else.

## Readouts (per condition; written to JSON before any label is computed)
- **PAM_resp**: count of the 307 PAM-DANs (`cell_type` starts with PAM) whose spike total over all 30 trials is > 0
  (Christie's criterion). Also PAM mean Hz and the count with mean >= 1 Hz (flychess cut).
- **PAM_resp_5**: same criterion over trials 0-4 only (the grid's trial count), from the same data. Isolates the trial-count
  difference at zero extra cost.
- **Per-trial**: for each of the 30 trials, PAM count with >= 1 spike, whole-brain count of neurons with >= 1 spike, and
  super_class=central mean rate.
- Fox, FDA-I, FDA-II mean Hz (cell sets from the reconciliation).

## Labels (computed by script from the JSON; no hand edits)
**GLOBAL trial**: a trial whose central mean rate is > 2x the central mean rate of the same (floor, r, Fox state) condition at
W 0.275, averaged over its 30 trials (the grid's F4 runaway rule, applied per trial). A PAM count that comes only from GLOBAL
trials is recruitment by runaway, not by a selective path.

Per floor, over Christie's 5 weights and 20 rates, Fox intact, GRN+GIN:
- **REPRODUCED (selective)**: some (W, r) has PAM_resp >= 150 AND the same (W, r) Fox-silenced PAM_resp <= 15 AND fewer than
  half of the trials contributing PAM spikes are GLOBAL.
- **REPRODUCED (runaway)**: the same two count conditions hold, but half or more of the PAM-contributing trials are GLOBAL.
- **PARTIAL**: max PAM_resp in [16, 149] with Fox-silenced <= 15 at that (W, r).
- **NOT REPRODUCED**: max PAM_resp <= 15 at every (W, r).
- **NOT FOX-DEPENDENT**: PAM_resp >= 16 somewhere but Fox-silenced > 15 at that (W, r).
Tie-break: report the (W, r) with the highest PAM_resp; list every (W, r) meeting REPRODUCED.

Secondary questions, each answered with one number, no label changes:
- Q-trials: does PAM_resp_5 < PAM_resp at the REPRODUCED cells? (Trial count alone explains the grid's null if yes and
  GRN-only also reproduces.)
- Q-GINs: GRN-only ablation max PAM_resp at W 0.39 vs GRN+GIN at W 0.39, per floor.
- Q-floor: label at floor 1 vs floor 5.
- Q-flicker: at Christie-matching cells, how many of the 30 trials carry the PAM spikes (distribution), and is the pattern
  across r non-monotonic as in Data S3?
- Q-control: W 0.275 max PAM_resp per floor (expected <= 15; if not, the canonical preprint claim itself is wrong at 30 trials).

## What each outcome means for the preprint (fixed now so the result cannot steer the wording)
- Any label: v2 must state that the reward result is for W 0.275 and a 5-synapse floor, and that Christie et al. report PAM
  recruitment at W 0.37-0.39. This is owed regardless.
- REPRODUCED at floor 1 only -> the floor (or floor x W) is the locus; v2 says so and cites this run.
- REPRODUCED (runaway) -> v2 reports that PAM recruitment in this setting coincides with network-wide activity; no claim that
  the reward path is or is not "real" in the fly.
- NOT REPRODUCED at both floors -> v2 keeps the claim with the protocol stated; ask the authors for code/parameters before any
  stronger statement.
- Q-control fails -> stop; the canonical "0/307" claim is rechecked before anything else.
No reward-LEARNING experiment is part of this run. Whether to test reward learning at a Christie setting is a separate
decision after this result.

## Not in scope / NOT SURE
- Christie's synapse floor, GIN rate, GRN set (they may have used both labellar sides' mapping differently; we use the grid's
  67 left sugar GRNs), and whether they used v783 (Data S4 IDs resolve in v783; Fox->CB0233 217 syn matches).
- Seeds are noise replicates of ONE brain, not flies.

## Amendments
(none yet; any change before launch is dated here with its reason; nothing after launch)
