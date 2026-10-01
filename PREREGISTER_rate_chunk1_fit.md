# PREREGISTER: rate model, chunk-1 fit (sugar -> PAM), path 2 PRD phase 4

Written and committed 2026-09-24, BEFORE any fit ran. Engine `98eee40`; targets `rate/chunk1_targets.md`
(`fe8ce0d`); unfitted control FAILS-UNFITTED (`0c48205`); gain bound `rate/gain_bound.json` (`033d150`).
Decisions: D47 (cap measured, never tuned), D48 (user: per-edge-type gains). Changes after launch are
dated amendments committed on their own.

## Question

Can per-edge-type gains inside a data-derived plausible range make the real v783 wiring pass the
chunk-1 FIT rows, and does that fit then transfer to the held-out rows it never saw?

## Model (arm A base, unchanged)

- w_scale 6.880548e-3, bias -35.0282 Hz in every pool, tau 20 ms, per-pool gain 1, r_max 454.55 Hz.
- Only edge-type gains are trained. Nothing else changes.

## Parameters

- **Owned target types (chunk 1):** FDA-I (CB0546, CB0272, CB3199), FDA-II (CB0233, DNp62, CB0337,
  CB1514, CB1025, CB3470, CB3573), PAM01-PAM15. That is 25 pools (christie_sweep sets, rule 14).
- **One gain per (source pool, target pool)** for every edge into an owned target: 933 gains, 3127
  edges, 29,501 synapses.
- **Not owned:** Fox and everything upstream of it. So sugar -> Fox stays unfitted, and H1/H2 test
  transfer. PPL1 is not owned either, so H6 stays unfitted.
- **Parameterisation:** gain = cap x sigmoid(x), so every gain lies in [0, cap].
  - Init: seed s draws gain = exp(N(0, 0.1^2)) per group, clipped into (0, cap).
  - The lower bound 0 is the flyvis data: 1 % of trained edge gains there are exactly 0.

## Loss (FIT rows only; uses the Fox condition only)

- The BASE condition stays exactly 0 under edge-gain changes (no drive, negative bias). This is
  asserted, so D = the rate under FOX.
- L = sum over c in {g4, g5, b2, b'2} of relu(1.5 - D_c)^2 + relu(D_b1 - 0.25)^2 + 0.01 x sum over
  all groups of log(gain)^2.
- The margins (1.5 above the "up" threshold of 1.0; 0.25 below the "none" threshold of 0.5) and the
  weight 0.01 are AUTHORED. They are not tuned; their job is to prefer the smallest change that
  passes (a pull toward count-proportional weights).
- Optimiser: Adam, lr 0.1 on x, t_run 1000 ms, up to 400 iterations. It stops early once the
  forward pass's D satisfies F1-F5 by the battery thresholds.
- After stopping, the full battery is re-run with `chunk1.measure` (no grad). Those rows are the
  record.

## Cap grid and g*

- Caps: 2, 4, 8, 16, 18.287, 32, 64, 128, 256. Seeds 0-4 at every cap. All 45 fits run.
- A seed F-passes at cap c if battery rows F1-F5 all pass after its fit.
- **g* = the smallest cap at which >= 4 of 5 seeds F-pass.** Held-out rows play NO part in choosing g*.
- **Plausibility bound B = 18.287:** the 99th percentile of the flyvis edge-level trained gain relative
  to the per-model median, fixed in `033d150`.

## Held-out evaluation (only at g*)

- For each seed that F-passes at g*: rows H1-H6 from the battery.
- A seed H-passes if all of H1-H6 pass.
- The regression suite is also run: F5, H3 and H5 were PASS in the unfitted record, so they must stay
  PASS. They are part of H/F already, so no separate test.

## Labels (primary)

- **UNFITTABLE:** no cap in the grid reaches 4/5 F-passing seeds.
- **IMPLAUSIBLE:** g* > B. Reported with g*.
- **FIT-NO-TRANSFER:** g* <= B, and fewer than 4 of 5 seeds H-pass at g*. The failing rows are listed.
- **PASS:** g* <= B, and at least 4 of 5 seeds H-pass at g*.

## Ensemble qualifier (appended to the label)

- Active gains: groups whose |log gain| > 0.05 in at least one seed at g*.
- Spearman rho of log gain over the active gains, for all 10 seed pairs.
- median rho >= 0.5 -> CONSISTENT; otherwise INCONSISTENT.

## Descriptive (never labels)

- Per cap: F-pass count, final loss, iterations.
- g*-fit top-20 active gains with their source and target cell types.
- Largest gain product along any Fox -> X -> R-PAM two-hop path (gains compound across hops;
  per-parameter caps do not bound the product).
- Held-out rows at every cap where >= 4/5 seeds F-pass. Reported as a curve, never used for the label.
- X1 (g2/g3 under sugar).

## Not in scope

Per-pool bias, tau or gain; floor 1; drive-rate sweeps; any rerun with changed margins, lr or weight
after seeing results (that would be an amendment, labelled post hoc).

---

## Amendment 1 (2026-09-24 ~22:10 Manila): dead gradient; surrogate gradient in the backward pass only

**Defect.** At init (gains around 1) every reward-set PAM is below threshold (arm A bias -35 Hz). relu
has slope exactly 0 there, so the fit term had ZERO gradient and the optimiser could not move it.
- Evidence 1: the first batch cell (cap 2, seed 0) ran 400 iterations and ended at loss 9.0000 =
  4 x 1.5^2, i.e. every R compartment still at 0 Hz. The fall in the smoke-run loss (9.094 -> 9.019)
  was the regulariser shrinking the init noise, not the fit term.
- Evidence 2 (a diagnostic, seen before this amendment): with every gain pushed to its cap, the fit
  term is 4.5 at cap 2 (some compartments up, max 210 Hz) and 0.0 at cap 256. So solutions exist in
  the parameter space, and "fails at cap 2" was the optimiser, not the wiring.
- Left unfixed, every cap would read UNFITTABLE for the wrong reason.

**Action.**
- The batch was stopped after 1 cell. That cell and its log were moved to
  `results/rate_chunk1_fit/artefact_amendment1/` and EXCLUDED from the result.
- The fit restarts from scratch.

**Change (the only one).**
- The fit sets `surrogate_beta = 35.0282 Hz`.
- The backward derivative of relu becomes 1 for v > 0 (exact) and (1 + |v|/beta)^-2 for v <= 0
  (SuperSpike fast sigmoid; Zenke & Ganguli 2018 Neural Comput 30:1514).
- beta = the arm-A distance from rest to threshold (|bias| = 7 mV x k). It is derived, not tuned.
- The forward model, and so every battery row, g* and label, is unchanged (engine golden 14: forward
  bit-identical).

**New gate G-grad** (before launch): at init, at cap 18.287, seed 0, the fit term's gradient with
respect to edge_x must be nonzero, and the fit term must fall within 10 iterations.

**Disclosure.** Evidence 2 was seen before this amendment. It changes nothing preregistered (caps,
seeds, loss, labels, bound). beta comes from the model constants, not from any fit outcome.

## Amendment 2 (2026-09-24 ~22:25 Manila): surrogate confined to the owned target neurons

**Defect (caught by gate G-grad; no fit cell was run).** With the amendment-1 surrogate on every
neuron, the init gradient of the fit term was 1.665e36, and the fit term stayed at 9.0 for 10
iterations. The surrogate passes gradient through every silent neuron, so the backward runs the
linearised whole-brain recurrence (w_scale x rho(|W|) ~ 19), which is unstable over 1000 steps. The
forward is stable only because relu cuts off silent neurons.

**Change (the only one).** The surrogate tail applies only to neurons in the chunk-1 owned target
pools (FDA-I, FDA-II, PAM01-15: the same 25 pools whose incoming gains are fitted). Every other neuron
keeps relu's exact 0 slope below threshold. Gradient still reaches every owned gain along
Fox -> FDA -> PAM. Forward unchanged, so every battery row and label rule is unchanged.

**Gate G-grad unchanged:** the init fit-term gradient must be nonzero AND finite, and the fit term
must fall within 10 iterations. If it fails again, the batch does not launch, and the next step goes
to the user.
