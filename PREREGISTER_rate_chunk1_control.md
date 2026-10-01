# PREREGISTER: rate model, chunk-1 unfitted control (path 2, PRD phase 3)

Written and committed 2026-09-24, BEFORE any rate-model run has driven or read a sugar, bitter, Fox,
FDA or PAM neuron. Engine: flychess `d372b1b` (rate/engine.py). Targets: `rate/chunk1_targets.md`
(`fe8ce0d`), used unchanged. Any change after the run is a dated amendment committed on its own.

## Question

Does the rate model of the v783 floor-5 brain, with NO fitted parameter, already pass the chunk-1
battery (sugar reaches the reward PAMs, bitter does not, Fox is needed)? If it does, chunk 1 needs no
fit and the finding is different (PRD open question). If it does not, the chunk-1 fit goes ahead.

## Arms (the parameters are derived, not chosen by looking at outcomes)

**Arm A (PRIMARY): LIF-matched.** A noise-free rate reduction of the paper-0 LIF, from Shiu constants only
(gpu_sim.py: W_SYN 0.275 mV, TAU_SYN 5 ms, TAU_M 20 ms, threshold 7 mV above rest, T_REFR 2.2 ms).
- Mean depolarisation from input: dV = 0.275 x 0.005 x sum_j n_ji r_j (mV).
- LIF f-I curve f(dV) = 1000 / (T_REFR + TAU_M ln(dV / (dV - 7))) Hz, dV > 7.
- Linearised by least squares through the threshold, r = k (dV - 7), over the range where f is 0-200 Hz
  (dV 7 to 53.58 mV): k = 5.0040 Hz/mV (rms error 20.6 Hz, mostly the steep onset near threshold;
  disclosed).
- Engine constants: w_scale = 0.275 x 0.005 x k = 6.880548e-3; bias = -7k = -35.0282 Hz in every pool;
  tau 20 ms and gain 1 everywhere; r_max = 1000 / T_REFR = 454.55 Hz.
- Known differences from the LIF, not corrected: no Poisson fluctuations (so there is no firing below
  threshold), no 1.8 ms delay, no 5 ms synaptic filter, and linear instead of log-shaped above threshold.

**Arm B (SECONDARY): Lappalainen default.** Pure ReLU, bias 0, tau 20 ms, gain 1, no ceiling,
w_scale = 0.5 / rho(|W|) = 1.789e-4 (the contracting scale from the phase-1 goldens). This is the most
permissive version, with no threshold at all. It is reported, but the decision rule uses arm A only.

## Stimuli and runs

- Sets: sugar = `cell_sub_class` "sugar/water", bitter = "bitter" (learn/condition.py);
  Fox = cell_type CB0525 (must be exactly 2 neurons).
- All drives are 150 Hz (condition.GRN_HZ, and the middle of Christie's 100-200 Hz sweep).
- t_run = 1000 ms (Christie's trial length); readout = mean output rate over the whole window.
- The model is deterministic, so there are no seeds; one run per condition per arm.
- Conditions: BASE (no drive), FOX, SUGAR, BITTER, SUGAR_FOXSIL (sugar drive, Fox silenced).
- Response of compartment c: D_c = mean over c's neurons of (rate_condition - rate_BASE).
  Compartment sets are exactly as in chunk1_targets.md; R = union of g4, g5, b2, b'2.

## Gates (checked before any readout; a failure stops the arm and is reported)

- G0: Fox = 2 neurons; print the Fox -> CB0233 synapse count at floor 5 (Christie Data S2A: 217 at
  floor 0) and require > 0 edges. This is consistency evidence for Fox = CB0525, not proof.
- G1: engine goldens pass at `d372b1b` (checked 2026-09-24 before this prereg: `ok rate engine`).
- G2: every set is non-empty; the silence set is disjoint from the drive set.
- S (stability, per arm): all rates finite. Arm A: fewer than 1 % of neurons at r_max in any
  condition. Arm B: max rate < 1e4 Hz. If S fails, the arm's label is UNSTABLE.

## Thresholds and tests

- up: D_c >= 1.0 Hz. none: D_c < 0.5 Hz. Between the two is AMBIGUOUS, which fails both "up" and "none".
- F1-F4: g4, g5, b2, b'2 up under FOX.
- F5: b1 none under FOX.
- H1: g4 and g5 up under SUGAR.
- H2: b2 and b'2 up under SUGAR.
- H3: no compartment of R up under BITTER.
- H4: D_R(SUGAR_FOXSIL) <= 0.5 x D_R(SUGAR), where D_R is the mean over R's neurons. If D_R(SUGAR)
  < 1.0 Hz, H4 is NOT TESTABLE, and that counts as not passed.
- H5: g3 none under FOX.
- H6: D_{g1 u g2}(BITTER) - D_{g1 u g2}(SUGAR) >= 1.0 Hz, and D_{g1 u g2}(BITTER) >= 1.0 Hz.

## Labels (per arm)

- UNSTABLE: gate S fails.
- PASSES-UNFITTED: all 11 rows (F1-F5, H1-H6) pass.
- FAILS-UNFITTED: at least one row fails; the failing rows are listed.

## Decision rule (arm A)

- FAILS-UNFITTED: the chunk-1 fit goes ahead as planned (PRD phases 4-5), with these default
  parameters as its starting point.
- PASSES-UNFITTED: no fit for chunk 1. The finding becomes "the rate reduction alone routes sugar to
  reward PAMs; the LIF block is a spiking and floor effect". Chunk 1 is re-scoped with the user.
- UNSTABLE: no chunk-1 conclusion. The instability is itself the result, and the next step is the
  user's call (a fitted global gain is one option; it needs its own prereg).
- Arm B changes no decision. If A and B disagree, both are reported, and the disagreement says the
  answer depends on the threshold.

## Descriptive (reported, never pass/fail)

- X1: g2 and g3 under SUGAR (Cohn: down).
- Every PAM01-15 and PPL101-106 type: D per condition.
- Fox, FDA-I and FDA-II (christie_sweep sets) mean rate per condition.
- Fraction of the whole brain with rate > 1 Hz, per condition.

## Not in scope

Floor 1 (PRD open question: done after the floor decision); drive-rate sweeps; any parameter change
after seeing outputs; any fitting.
