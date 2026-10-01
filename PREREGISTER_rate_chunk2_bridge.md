# PREREGISTER rate_chunk2_bridge: paper 0's odour learning on the rate model, two brains (2026-10-01)

Written 2026-10-01, before any chunk-2 learning run on the real brain. Paper 1, child PRD
`.claude/PRPs/prds/paper1-chunk2-bridge.prd.md` (decisions 1-9, all confirmed by the user on 2026-10-01).
Batch run, one analysis pass (feedback_batch_run_then_interpret). Every rule below is fixed before launch and is
enforced in code: `rate/chunk2.py --analyze` refuses to label unless the prereg, reference.json, the code files and
the git HEAD all match what the 24 runs recorded.

## Question
Paper 0 (spiking LIF, v783, eln8) learned odour-punishment associations. Does that learning survive:
- **arm (i), base `c0`:** the swap to the chunk-0 rate model (neuron-model question), and
- **arm (ii), bases `c1s0..c1s4`:** the chunk-0 rate model plus the chunk-1 reward-path edge gains at cap 16
  (one base per fitted seed; regression question: does the reward-path change break paper 0)?
Both outcomes are reported. Reward learning is NOT tested here (paper 1b).

## Model (fixed)
- C0 base from `rate/regress/c0_base.json` (lin, w 0.00494, bias -9.52 Hz, tau 17 ms, r_max 124.5 Hz, eln8),
  no registered chunk params (asserted). Synapse floor 5.
- Arm (ii): `set_edge_gains(chunk-1 owned target pools, cap=16.0)` then `edge_x` from
  `results/rate_chunk1_fit_c0/cells/cap016.000_s{seed}.json`; all 5 seeds F-passed at cap 16 (asserted); gains
  reload reproduces the chunk-1 Fox rates of each seed exactly (golden 31).
- Plasticity: paper 0's code, unchanged (`learn/condition.py`, `learn/persist.py`, `learn/plastic.py`); constants
  identical to paper 0 (decision 1): eta 5e-6, lam `PL.LAM`, W_MIN, W_MAX. Only KC->MBON edges are plastic; none are
  gained edges (asserted in `set_plastic`).
- Fresh state v = 0 per trial (decision 2). Deterministic engine: one run per arm; compared with paper 0's 5-seed
  mean. No statistics are possible; lesion shifts are exactly 0 by construction; shuffle is one permutation.

## Protocols (paper 0's, unchanged)
- rung A (`--mode rung_a`): arms learn, lesion, reversed, frozen, shuffle; 30 trials; DC2 paired with punishment
  (PPL1 driven at the DANs), D unpaired, DA1 never paired.
- rung 3 (`--mode rung3`): persist arms retain, interfere, lesion; train A 30, relax 30, train C 30.
- bidir (`--mode bidir --rule timed|depress`): open loop, MB level (PRD D1): 4 cycles of F (200 ticks of 100 ms,
  cue DC2, US every 25 ticks) and B (US every 25 ticks, cue withheld on the US tick and the one before); T probes
  between; eta 5e-6 fixed in code (ETA_BIDIR), US 60 Hz.
- Run parameters pinned: `PREREG_PARAMS = {train 30, relax 30, probe 1, eta 5e-6, lam PL.LAM, seed0 0}`; full default
  arm lists only (analyze refuses subsets or overrides).

## Reference (paper 0, recomputed from its result files)
`results/rate_chunk2/reference.json`, sha256 `51f00736c27fdef3d9a08b83b98e55bd954ff39104051253e7466992088e669f`,
built by `rate/chunk2_reference.py` from the paper-0 jsonl it hashes (condition_o1s0-4, persist_p1s0-4). All preprint
numbers recompute (rung A DC2 +11.05; rung 3 9.66 / 11.29 = 85.5%). Gate quantities (approach component, decision 8):
rung A normalised DC2 approach drop 0.97146; naive D normalised approach drop 0.93469; rung-3 approach-drop ratio
0.85814. Paper 0's DC2 shift was 99.1% approach-side.

## Pass rules (verbatim from `rate/chunk2.py rule_text()`, code is authoritative)
RULE (fixed in code before the batch; thresholds are fractions of reference.json, paper 0):
Approach component: every gate uses the approach-MBON mean rate, because plasticity acts only on approach synapses in both models (the rate model's avoid baseline is 10-25x paper 0's, so a total avoid_index shift can be moved by avoid MBONs). approach drop = baseline (trial-0, baseline weights) approach-MBON mean minus the approach-MBON mean at the test; positive = learning. Normalised drop = approach drop / the baseline approach-MBON mean of the same arm and odour. The paper-0 normalised drop is its mean approach drop divided by its mean baseline approach (reference.json rung_a.learn_dc2_approach_drop_norm, naive_d_approach_drop_norm, rung3.approach_ratio_of_means). The total avoid_index shift and the avoid-MBON component are reported, never gated.
Headroom gate (baseline weights, each probe odour): PASS iff the baseline approach-MBON mean is > 0 and neither MBON group mean is >= 0.95 x r_max. No absolute rate threshold.
rung A: approach drop = learn-arm DC2 approach at trial 0 minus at the test trial (reference test_trial). PASS iff drop > 0 and normalised drop >= 0.5 x the paper-0 normalised DC2 approach drop and the learn normalised drop exceeds both the lesion and the shuffle arm normalised DC2 drops. D and DA1 are reported, not gated.
rung 3: ratio = D approach drop of the interfere arm (relax_test end minus trainC_test end) / D approach drop of the rung A reversed arm (D paired with punishment on a naive brain; trial 0 minus test trial). PASS iff both drops are > 0 (same direction as paper 0; two negative drops fail), the rate naive D normalised approach drop >= 0.5 x the paper-0 naive D normalised approach drop (a near-zero denominator cannot inflate the ratio) and ratio >= 0.5 x the paper-0 approach-drop ratio.
bidir (timed rule scored against the depress rule; both must be in the batch): learning level = minus the DC2 approach-MBON mean; rise_k = level at F_k minus at B_(k-1) (T0 for k = 1) (an approach drop); drop_j = level at F_j minus at B_j. Cycle k = 2..4 relearns iff rise_k >= 0.5 x rise_1 and timed drop_(k-1) > 0 and timed drop_(k-1) > depress drop_(k-1). PASS iff rise_1 / baseline approach DC2 (timed T0) >= 0.5 x the rung A paper-0 normalised DC2 approach drop (borrowed, no MB-level bidir reference exists) and at least 2 cycles relearn.
Arms: arm (i) = base c0 (the frozen C0 brain); its protocol labels are its verdicts. Arm (ii) = bases c1s0..c1s4 (C0 plus the chunk-1 reward-path gains at cap 16, one base per fitted seed). Each seed is labelled per protocol by the rules above on its own baseline. An arm (ii) protocol label is PASS iff at least ceil(0.8 x n) = 4 of the n = 5 seeds PASS that protocol, else FAIL with the per-seed labels and diagnostics listed. The batch is 4 runs x 6 bases = 24 runs; every run needs its done row.
Diagnostic for a failing protocol (or failing seed), first match wins: READOUT-CEILING > FAIL-UNSTABLE > NO-WEIGHT-CHANGE > LATCH-CONFOUNDED > NEEDS-SPIKING. READOUT-CEILING = any headroom row of the protocol failed its gate (a saturated or a zero baseline); FAIL-UNSTABLE = a guard abort row in an arm the verdict needs, even when its scoring rows exist (an abort voids a PASS); NO-WEIGHT-CHANGE = |w_mean_frac| < 0.1 x |paper-0 value| (rung A learn arm at the test trial; rung 3 interfere arm end of trainC; bidir timed rule, largest |w_mean_frac| over its test rows, rung A value borrowed); LATCH-CONFOUNDED = a BASELINE latch row of the protocol has latched True (end rows are reported, never gated); NEEDS-SPIKING = none of the above. A failing protocol with missing rows and neither an abort row nor a failed headroom row to explain them is INCOMPLETE-DATA (a pipeline fault, not a finding).

## Stability guard (decision 4)
Every probe (300 ms, fresh state) is compared with the same stimulus on the same base before any weight change:
abort the arm if the whole-brain active fraction (> 1 Hz) exceeds 2x, or MBON06 leaves [0.5x, 2x] of reference; if the
reference MBON06 is < 0.5 Hz (it is 0 Hz on C0 odour probes) the band is absolute [0, 1 Hz], so this half of the guard
can only catch a rise. Aborted arm = FAIL-UNSTABLE, reported; the batch continues.

## Latch rule (decision 6)
LATCHED iff any paired-odour KC is > 1 Hz in the last 100 ms of a 200 ms offset window after a 300 ms odour trial, or
latched approach MBONs carry >= 50% of the approach-MBON rate during that odour trial. Checked at baseline (gates) and
at the end of learn / interfere / bidir arms (reported only).

## Facts measured before writing (baseline weights only; `--gate`, 2026-10-01; 31 s for 6 bases)
| base | headroom rows PASS | latched (baseline) | approach DC2 Hz (x paper 0) | avoid DC2 Hz (x paper 0) | latched n | latched PAM | PAM Hz after offset |
|---|---|---|---|---|---|---|---|
| c0 | 4/4 | 0/4 | 26.50 (2.35x) | 5.27 (24.6x) | 740 | 13 | 0.35 |
| c1s0-s4 | 4/4 each | 0/4 each | 26.50 (2.35x) | 5.27 (24.6x) | 757-762 | 43-44 | 2.04-2.24 |
The odour-independent attractor (740 neurons, 0 KCs, 6/29 approach MBONs = 32% of approach rate) is the one the
suite accepts (chunk 0). In arm (ii) PAM keeps firing ~2 Hz after the odour: the PAM -> avoid-MBON branch of the
rule (reward depresses KC -> avoid synapses) will be slightly active there. Avoid-side changes are reported per base.

## What is reported (not gated)
Per base and protocol: total avoid_index shift, avoid-MBON component, D and DA1 (specificity), PAM and PPL1 rates,
|w_mean_frac| and signed w_mean_frac, guard rows, end-of-arm latch rows. Literature (Hige 2015, Handler 2019,
Davidson 2023; `.claude/writing/paper1-bidir-literature.md`) is Discussion context only, never a threshold.

## Disclosures
- Arm (ii) tests punishment-driven odour learning with the reward-path gains on (the US is a clamped PPL1 drive),
  not reward learning.
- NO-WEIGHT-CHANGE compares the absolute value of the SIGNED mean weight change; under the timed rule potentiation
  and depression can cancel in that mean.
- bidir uses fixed ETA_BIDIR/lam constants in code, not the CLI eta/lam (recorded values are still checked).
- Bidir threshold and NO-WEIGHT-CHANGE value are borrowed from rung A (paper 0 has no MB-level bidir number).
- Bidir schedule as coded: each 200-tick block delivers 7 US (ticks 25..175, not 8); B also withholds the cue at
  tick 199, where no US follows.
- Provenance recorded per run: git HEAD (full hash), dirty flags (rate/ + gpu_sim.py, learn/), sha256 of LF-normalised
  CODE_FILES (chunk2, engine, suite, as_gpusim, chunk1, chunk1_fit, regress/c0_base.json, gpu_sim, learn/plastic,
  condition, persist; analyze refuses any change), sha256 of data/brain_gpu.npz and data/neuron_meta.npz (gitignored;
  reported, not refused), chunk-1 gains file sha256 (refused if it differs from the cell on disk).
- Normalisation (decision 5) and approach-only gating (decision 8) are user decisions made 2026-10-01 before any
  learning number; the absolute shifts are reported beside them.

## Runbook (fixed)
Precondition: this file and the code committed; `git status --porcelain -- rate learn` empty (analyze refuses a dirty
or mismatched tree). Nobody opens `results/rate_chunk2/*.jsonl` learning rows before analyze runs.
```bash
cd "F:/Workspace/Project Files/fly/flychess"
for b in "--arm c0" "--arm c1 --fit-seed 0" "--arm c1 --fit-seed 1" "--arm c1 --fit-seed 2" "--arm c1 --fit-seed 3" "--arm c1 --fit-seed 4"; do
  for m in "--mode rung_a" "--mode rung3" "--mode bidir --rule timed" "--mode bidir --rule depress"; do
    .venv/Scripts/python.exe tools/gpu_slot.py -- .venv/Scripts/python.exe rate/chunk2.py $m $b || echo "RUN FAILED: $m $b"
  done
done
.venv/Scripts/python.exe rate/chunk2.py --analyze
```
Stop rules: a crashed run is re-run once with `--overwrite` (recorded as `overwrite: true` in its meta row; "once"
is procedural, not enforced in code). If it crashes again, analyze refuses the whole batch (it needs all 24 done rows):
the batch is reported as INCOMPLETE (no labels) and resolved only by a dated amendment below. No run is re-run for
any other reason. Analyze runs once. After analyze: `rate/suite.py` must still be GREEN (PRD phase 4). RESULT commit:
results/rate_chunk2/RESULT.md + result.json (git add -f), with the label. `--reanalyze` (pipeline fault only,
recorded) must run with the batch commit checked out, because analyze refuses a HEAD other than the runs'.
Freeze: from the batch commit until analyze finishes, no git checkout / reset / stash / switch touching rate/,
learn/ or gpu_sim.py (autocrlf is on; code_sha256 is hashed on LF-normalised bytes, so line-ending flips no longer
break it, but content edits still refuse). Run the loop in Git Bash.

## Amendments
- A1 (2026-10-01, before any learning run; pre-launch Opus audit GO-WITH-FIXES): stop rule for a second crash
  corrected to match code (batch INCOMPLETE, not per-base); CODE_FILES extended + LF-normalised hashing + data hashes
  recorded; bidir 7-US schedule, suite rerun, reanalyze-at-batch-commit and git freeze stated.
