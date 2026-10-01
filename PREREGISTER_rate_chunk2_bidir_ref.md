# PREREGISTER rate_chunk2_bidir_ref: a dose-matched MB-level bidir reference from paper 0's LIF (2026-10-01)

Written 2026-10-01, after the chunk-2 bridge RESULT (flychess d0b0c92, evaluation 22dad53) and before any code for
this run exists and before any LIF bidir run at the mushroom-body level. Batch run, one analysis pass. The chunk-2
preregistered bidir label (FAIL, all 6 bases) STANDS and is reported as is; this is a second, separately
preregistered test, reported beside it, never instead of it.

## Why this run exists
Chunk-2 bidir relearned in 3 of 3 scorable cycles on every base, and failed only its F_1 floor: the first F block's
normalised approach drop was 0.131 (c0) against 0.486, a floor borrowed from rung A because paper 0 has no MB-level
bidir number (its relearning test was closed loop and read walking speed). The independent evaluation measured that
one F block delivers 0.20-0.45x rung A's 30-trial weight change, so the borrowed floor asked for ~2.6x the dose the
schedule gives. The fair reference is paper 0's own model run through the identical open-loop schedule.

## Question
Under the identical open-loop MB bidir schedule, how large is the LIF's first-block approach drop, and does the LIF
relearn by the same size-based rule? Measured against that reference, does the rate model's bidir pass?

## Model (fixed)
- **LIF reference:** paper 0's model exactly: `gpu_sim.GpuSim`, regime eln8 (`ELN_NEGATE` + `PN_KC_GAIN` 8.0 set
  before construction, as `learn/condition.py` main), synapse floor 5 (`data/brain_gpu.npz`, asserted), default
  W_SYN, Shiu constants, paper-0 plasticity code unchanged (`learn/plastic.py`).
- **Seeds:** 5 (0-4). Every 100 ms tick of seed s gets its own counter-RNG seed derived from s, block and tick
  (formula fixed in code before launch, printed in the meta row); T probes use the paper-0 probe seed convention of
  `C.trial` (seed 0 for every probe, so probes are noise-matched and the guard reference is valid). Determinism
  check before launch: seed 0, one F block, run twice, counts identical.
- **Schedule:** `rate/chunk2.bidir_mb` unchanged: 4 cycles of F (200 x 100 ms ticks, cue DC2, US = PPL1 at 60 Hz every
  25 ticks = 7 per block) and B (US every 25 ticks, cue withheld on the US tick and the one before); T probes after
  T0, every F and every B; eta 5e-6 (ETA_BIDIR), lam `PL.LAM`. Rules: timed and depress.
- **Runs:** 5 seeds x 2 rules = 10 LIF runs. The rate logs are NOT re-run: the 12 existing rate bidir logs are
  re-scored, verified against the sha256 list in `results/rate_chunk2/EVALUATION.md` (refuse on mismatch).
- Guard, headroom and baseline latch rows as in chunk 2 (reported; guard abort = FAIL-UNSTABLE for that seed).

## Quantities (identical definitions to the chunk-2 rule text)
A = DC2 approach-MBON mean at a T probe. level = -A. rise_k = level(F_k) - level(B_(k-1)) (T0 for k = 1);
drop_j = level(F_j) - level(B_j). rise_1_norm = rise_1 / A(T0) of the same run.
Relearn in cycle k = 2..4 iff rise_k >= 0.5 x rise_1, and timed drop_(k-1) > 0, and timed drop_(k-1) > depress
drop_(k-1) (depress paired with timed by seed for the LIF; by base for the rate model).

## Reference (LIF)
- R = mean over the 5 seeds of the LIF timed rise_1_norm. Also reported: SD, per-seed values, n same sign, and the
  per-seed relearn count.
- **LIF sanity rule (fixed):** the reference is USABLE iff R > 0 in at least 4 of 5 seeds and the LIF timed rule
  relearns (>= 2 of 3 cycles) in at least 4 of 5 seeds. If not usable, the label is NO-REFERENCE: the MB-level
  open-loop schedule does not show relearning even in paper 0's model, and the rate bidir is reported as untestable
  at the MB level (the paper then cites only paper 0's closed-loop result).

## Re-scored rate label (per base c0, c1s0..c1s4; arm ii PASS iff >= 4 of 5 seeds, as chunk 2)
PASS iff (a) rate timed rise_1_norm >= 0.5 x R, and (b) >= 2 of the 3 scorable cycles relearn (rule above,
unchanged from chunk 2). Otherwise FAIL, with the failing part named ((a), (b) or both). The label is called
"bidir vs LIF reference" in every table, never just "bidir".

## Secondary, reported, never gated (fixed now)
- Dose-matched rung A: for each rate base, the rung-A learn-arm normalised DC2 approach drop interpolated (linear,
  between the logged test trials 5..30) at the |w_mean_frac| that the timed rule reached at F1; reported beside
  rise_1_norm as the ratio rise_1_norm / dose-matched drop. The same for the LIF (paper-0 rung A rows,
  condition_o1s0-4, at the LIF F1 |w_mean_frac|).
- Avoid-MBON component and PAM rate per block for LIF and rate (the c1 avoid-side collapse, tonic PAM).
- |w_mean_frac| per block, both rules.

## Wall-time estimate (NOT SURE)
One LIF bidir run = 1,600 ticks of 100 ms + 27 probes of 300 ms ~ 168 s simulated; at ~1.1 s wall per simulated
second (wsyn_grid ledger, floor 5) ~3-4 min per run, ~35-45 min for 10 runs. Time the first run.

## Runbook (fixed; code written and reviewed after this prereg is committed)
Precondition: code committed, `git status --porcelain -- rate learn gpu_sim.py` empty, determinism check passed.
Nobody reads LIF learning rows before the analysis. Each run is its own process.
```bash
cd "F:/Workspace/Project Files/fly/flychess"
for s in 0 1 2 3 4; do for r in timed depress; do
  .venv/Scripts/python.exe tools/gpu_slot.py -- .venv/Scripts/python.exe rate/chunk2.py --mode bidir --rule $r --arm lif --lif-seed $s || echo "RUN FAILED: $r $s"
done; done
.venv/Scripts/python.exe rate/chunk2.py --analyze-bidir-ref
.venv/Scripts/python.exe tools/gpu_slot.py -- .venv/Scripts/python.exe rate/suite.py
```
Provenance as chunk 2 (prereg, code, git HEAD, data hashes in every meta row; analysis refuses mismatches, refuses
a missing done row, refuses to overwrite a result without `--reanalyze`). Output: `results/rate_chunk2_bidir_ref/`.
Stop rules: a crashed run is re-run once with `--overwrite`; a second crash leaves the batch INCOMPLETE (no label)
until a dated amendment. No run is re-run for any other reason. Independent Opus evaluation of the RESULT before it
is cited.

## What this can and cannot say
- PASS = at the MB level, under the same schedule and dose, the rate model's first-block learning is at least half
  the LIF's and it relearns; bidir then joins rung A and rung 3 as surviving the neuron-model swap.
- FAIL (a) = the rate model learns less per block than the LIF at the same dose (a real rate-vs-spiking difference
  in learning speed, not in relearning). FAIL (b) = relearning itself differs.
- Neither outcome changes the chunk-2 label, and neither speaks to closed-loop behaviour.

## Amendments
- A1 (2026-10-01, before any LIF MB-level run; code review of the analyzer, flychess 1d0b4f6..adb1df2):
  0. "R > 0 in at least 4 of 5 seeds" (above) means each scored seed's own rise_1_norm > 0.
  1. USABLE also requires the mean R > 0 (otherwise 0.5 x R <= 0 would pass every rate base trivially).
  2. R = mean over the SCORED LIF seeds (n reported). A guard abort or an incomplete curve in EITHER rule's run voids
     that seed: it counts as neither positive nor relearning. The USABLE thresholds stay absolute (4 of 5 seeds).
  3. The relearn rule keeps chunk 2's `rise_1 > 0` clause (the chunk-2 labeller as run includes it; its rule text
     omitted it). One shared function computes relearning for the chunk-2 labeller and this re-score.
  4. Dose-matched rung A: no value outside the logged |w_mean_frac| range (trials 5..30, no extrapolation); an
     origin-anchored interpolation is shown separately and labelled outside this prereg. Non-monotone |w| -> no value.
  5. LIF seed s is paired with paper-0 condition_o1s{s} by index only (not noise-matched).
  6. Seeding wording (overrides "the paper-0 probe seed convention" in Model above): T probes use seed 0 for every probe (this prereg's choice; paper 0's rung A probes used
     seed0 + 1000k + j). Per-tick seeds `1000000*(S+1) + 1000*(block+1) + tick` are this run's decorrelation, not
     paper 0's closed-loop seeding.
  7. Measured wall time (Task 1): ~175 s per 200-tick block, ~25 min per run, ~4.3 h for the 10 runs (sequential);
     the 35-45 min estimate above was wrong by ~6x. Overnight batch.
  8. Baseline latch and headroom rows are reported for every LIF seed and rate base (LIF headroom ratios are vs the
     paper-0 LIF seed-mean baseline). The resolved brain path is asserted to be data/brain_gpu.npz and the other
     GpuSim module knobs are asserted at their defaults.
  9. A rate base whose curve is incomplete or aborted labels FAIL (incomplete)/(unstable); none of the 12 verified rate
     logs can hit this.
