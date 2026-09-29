# EVALUATION christie_match (independent evaluator, 2026-09-29)

Inputs: PREREGISTER_christie_match.md (dd5a6bb, 00:17:34), learn/christie_match*.py + analyze (8e7670c, 00:19:09; first job 00:19:53; `git diff 8e7670c` on all four files = empty). 532/532 conditions present, not_run.json = {}, drop 0, 132 of 600 GPU-min used. Recompute used my own script (scratchpad ev.py/ev2.py), not the analyzer.

## 1. Fidelity
Matches prereg: 67 left sugar GRNs + 14 GINs (IDs = prereg list, asserted type+left), same rate r; r 10..200 step 10; 6 W; floors 1/5 with n_edges asserted; Fox = CB0525 x2 via run_batch(silence=); seeds 0-29 x 1000 ms; base per (W, floor); GRN-only ablation at W 0.39 both floors, Fox intact; PAM = 307 cells with cell_type prefix PAM; PAM_resp (>0 spikes over 30 trials), PAM_resp_5 (trials 0-4), per-trial PAM/brain/central, Fox/FDA-I/FDA-II Hz; GLOBAL = trial central > 2x mean central of the W 0.275 same-r same-Fox-state condition; label thresholds 150/15/16/149.
Deviations / unregistered choices (none changes a label here):
- UNDETERMINED branches (Fox-sil not run at max cell; max >= 150 but not in repro list): unregistered; not triggered (drop 0, max 17).
- Tie-break for equal max PAM_resp: unregistered (first in W-asc, r-asc order). Max is unique at both floors (17; 14). No effect.
- PARTIAL / NOT FOX-DEPENDENT evaluated at the single max cell only. Prereg "somewhere ... at that (W, r)" is ambiguous. Floor 1 cells >= 16 intact: (0.39,190) 16/sil 15, (0.39,200) 17/sil 12, both sil <= 15, so either reading gives PARTIAL. The prereg does not consider a Fox-silenced cell > 15 where intact < 16 (occurs: W 0.39 r180 intact 12, silenced 16).
- Q-flicker implemented only as flips of (>=150) at best W; the trial distribution asked for is not computed by the script (given in 6 below).
- --max-call-s 900 runaway stop, cell order (floor 1 first, W ascending), 2x15 chunking: unregistered or registered-optional; none triggered anything. Chunk independence evidenced: in-run grn200 seeds 0-4 (15-seed batch) = [7,11,7,14,0] = 5-seed golden.
- Golden/timing ran twice in ledger (dry pass then full); second timing 2.7 s = resume skip. Harmless.
- Stock regime relies on gpu_sim defaults for dt/background (only ELN_NEGATE, PN_KC_GAIN, TYPE_W_SCALE asserted). NOT SURE by inspection; golden match to wsyn_grid supports equivalence.

## 2. Independent recompute
npy vs jsonl: 0 mismatches over 532 conditions (PAM_resp, PAM_resp_5, trial_pam_n). Labels: floor 1 PARTIAL (W 0.39 r200, 17, sil 12, global_frac 0.0, 16/30 trials); floor 5 NOT REPRODUCED (max 14, W 0.386 r10, sil 14, global_frac 1.0, 3/30 trials). Q-control 0/0. MATCH.

## 3. Floor 1 PARTIAL: threshold artefact
PAM_resp, cols r = 10..200:
```
floor 1 intact
0.275 [0 x20]
0.370 [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,9,8]
0.380 [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,11,13,12,13]
0.382 [12,0,0,0,0,0,0,0,0,0,0,0,0,0,9,0,0,14,12,12]
0.386 [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,8,13,15,15]
0.390 [13,11,0,0,11,0,0,0,0,0,0,0,10,0,0,13,12,12,16,17]
floor 1 Fox-silenced
0.275 [0 x20]
0.370 [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,7,12,9,0]
0.380 [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,13,8,11,12,12]
0.382 [12,0,0,0,0,0,0,0,0,0,0,0,0,0,0,12,12,10,13,15]
0.386 [0,0,0,0,0,0,0,0,0,0,0,0,0,0,8,11,7,12,15,15]
0.390 [13,0,0,0,9,0,0,0,0,0,0,13,14,12,0,10,13,16,15,12]
floor 1 GRN-only W 0.39: [0,10,11,0,0,0,0,0,0,10,0,9,0,9,12,13,13,12,14,17]
floor 5 intact:  0.38 r10 11; 0.386 r10 14, r20 12; 0.39 r10 11, r20 11; all else 0 (0.275, 0.37, 0.382 all 0)
floor 5 Fox-sil: 0.38 r10 11; 0.386 r10 14; 0.39 r10 11, r20 13; all else 0
floor 5 GRN-only W 0.39: [13,12,11,11,11,12,0,11,0,0,0,0,0,0,0,12,0,0,0,0]
base (all W, both floors): PAM 0, whole brain 0 neurons spiking.
```
Evidence Fox silencing does nothing measurable:
- Pooled over the 100 Christie cells at floor 1: sum PAM_resp intact 291 vs silenced 353; intact > sil in 13 cells, sil > intact in 14, equal in 73. PAM-positive trials 99 vs 83 of 3000.
- Fox-silenced reaches 16 (W 0.39 r180) > the 15 cut; intact max 17 is 1 above the 16 cut.
- Responsive-set overlap intact vs silenced: W 0.39 r200 17/12, 12 shared (all silenced cells are in the intact set); r190 16/15, 14 shared; W 0.386 r200 15/15, 14 shared.
- Only 18 distinct PAM cells ever respond across every floor-1 condition (PAM05 x4, PAM06 x9, PAM08 x5); 7 of them in 65 conditions each. The 17 is a fixed small cluster that co-fires 4-15 cells per trial, not graded recruitment. Max PAM_n_ge1hz anywhere = 11; max PAM mean 0.076 Hz.
- Fox_hz silenced = 0.000 in every silenced row, so silencing was applied; it just does not matter.
- PAM-positive trials at W 0.39 r200 have higher central rate (0.85-1.06 Hz, brain 1186-1947 cells) than PAM-negative trials (mean 0.839): PAM spikes ride trial-level network excursions, below the 2x GLOBAL cut (1.26 Hz).
Verdict on 3: PARTIAL is a threshold artefact. Substantively floor 1 is "not reproduced, and not Fox-dependent": ~17/307 ceiling, identical with Fox silenced.

## 4. Floor 5 best at r 10, global_frac 1.0
Base: brain fully silent at all W (no spontaneous activity). The r10-20 PAM counts come from rare seed-specific network bursts: W 0.386 r10 seeds 11, 26, 27 have central 0.14, 0.64, 0.73 Hz vs 0.03 median (up to 24x; the 2x threshold is 0.019). Intact and silenced are bit-identical (same seeds, 450 spikes each) because Fox is ~silent at r10 (0.10 Hz), so silencing is a no-op there; "Fox-sil 14" is not a Fox test. Same bursts appear GRN-only (W 0.39 r10 seed 4, central 0.61). They vanish at r >= 30 (all 0). Mechanism of low-rate-only bursts: NOT SURE (plausibly high drive keeps inhibitory tone up). Runaway-type, correctly flagged GLOBAL.

## 5. vs Christie Data S3
Christie: 186-203 or 0-11 of ~307, flipping across r. Here max anywhere is 17 (floor 1), 14 (floor 5), 17 GRN-only. No cell anywhere near 150; the whole reachable pool is 18 cells. Pattern is not all-or-none: floor 1 rises roughly monotonically at r >= 170 (W 0.39 PAM-positive trials per r: ...3,4,3,11,16) plus sporadic 1-2-trial events at r10-50.

## 6. Secondary questions
- Q-trials: 0 REPRODUCED cells (N/A). At the best cell PAM_resp_5 = 14 vs 30-trial 17: +3 cells from trials, nowhere near 150.
- Q-GINs: W 0.39 max GRN+GIN vs GRN-only: floor 1 17 vs 17; floor 5 11 vs 13. GINs add nothing.
- Q-floor: floor 1 PARTIAL (substantively null), floor 5 NOT REPRODUCED.
- Q-flicker: 0 flips at >=150 (no such cell); best cell PAM spikes in 16/30 trials (4-14 PAM each); not Data S3-like.
- Q-control: W 0.275 max PAM_resp 0 at both floors (intact and silenced). PASS.

## 7. Preprint clause (verbatim)
The prereg has no clause for PARTIAL. Applies strictly: "Any label: v2 must state that the reward result is for W 0.275 and a 5-synapse floor, and that Christie et al. report PAM recruitment at W 0.37-0.39. This is owed regardless."
Given 3, the substantive match is: "NOT REPRODUCED at both floors -> v2 keeps the claim with the protocol stated; ask the authors for code/parameters before any stronger statement." Applying it would be an evaluator judgment, not the registered label; say so if used.

## 8. Sanity
- Drive applied: central Hz and Fox Hz scale with r (floor 1 W 0.39: central 0.040 / 0.495 / 0.881 at r 10/100/200; Fox 0.18 / 24.8 / 49.3 Hz). Base = 0 everywhere.
- W effect: at r200, central 0.881 (W 0.39) vs 0.630 (W 0.275) floor 1, 0.732 vs 0.532 floor 5; spiking neurons 1242 vs 591 (floor 1). Network is driven harder, no runaway (max per-trial central anywhere 1.41 Hz).
- Fox silenced = 0.000 Hz in all 240 silenced rows; silenced central rate ~= intact (Fox not a network driver).
- FDA-I at r200: 1.16 Hz (W 0.39) vs 0.42 (W 0.275) floor 1; 0.54 vs 0.00 floor 5. FDA-II ~1.3-1.6 Hz, slightly lower at higher W (NOT SURE why; small). The Fox->FDA->PAM relay carries Fox at ~49 Hz into ~1 Hz FDA and ~0 PAM, consistent with the known reward-path gap.
- No silent failures found: 532 rows, shapes [30,307], jsonl/npy consistent, golden PASS twice, exit 0 all jobs.

VERDICT: labels QUALIFIED. Both labels are computed correctly per the prereg and reproduce independently, but floor 1 PARTIAL is a threshold artefact: Fox silencing leaves PAM unchanged (pooled 353 silenced vs 291 intact; silenced reaches 16), and the ~17-cell ceiling is a fixed PAM05/06/08 cluster, about a tenth of Christie's 186-203. Substantively neither floor reproduces Christie Fig 6A.
