# RESULT: chunk-1 fit (primary, preregistered)

Prereg `2838bfe` + amendments 1-2 (`aca91ff`, `e8ddc3c`); script `b09e728`. 45/45 cells, analysed once
(`chunk1_fit.py`, automatic at batch end, 2026-09-25 01:33 Manila). `result.json`, `cells/`, `run.log` here.

## Label: FIT-NO-TRANSFER (CONSISTENT)

- **g* = 8** (<= plausibility bound 18.287). Caps 2 and 4: 0/5 seeds F-pass. Caps 8-256: 5/5.
- **Held-out at g*: 0/5 seeds pass.** Failing rows H1, H2, H4, H6. Passing rows H3 (bitter excluded)
  and H5 (g3 null).
- Regression vs the unfitted record: F5, H3 and H5 were PASS and stay PASS (no regression).
- Ensemble: 134 active gains, median Spearman rho 0.689 over 10 seed pairs -> CONSISTENT.

## What the fit does (g* = 8, per seed)

| seed | FOX: D_R | g4 | g5 | b2 | b'2 | SUGAR D_R | SUGAR g1ug2 | BITTER g1ug2 |
|---|---|---|---|---|---|---|---|---|
| 0 | 27.5 | 74.9 | 11.3 | 1.1 | 9.1 | 0.000 | 0.000 | 0.000 |
| 1 | 21.2 | 62.2 | 5.4 | 1.1 | 5.3 | 0.000 | 0.000 | 0.000 |
| 2 | 24.4 | 70.2 | 9.3 | 1.2 | 5.6 | 0.000 | 0.000 | 0.000 |
| 3 | 27.0 | 73.4 | 10.7 | 1.0 | 9.1 | 0.000 | 0.000 | 0.000 |
| 4 | 27.3 | 74.1 | 8.0 | 1.1 | 11.1 | 0.000 | 0.000 | 0.000 |

- With fitted gains, driving Fox recruits all four reward compartments (g4 strongest; b2 only just
  over the 1.0 Hz threshold in every seed).
- Real sugar input still gives **exactly 0** in every reward compartment. The fit owns FDA and PAM
  inputs but NOT sugar -> Fox (by design). Unfitted, sugar drives Fox to only ~9 Hz (control), and
  that stays below what the fitted FDA -> PAM path needs.
- H6 fails because neither bitter nor sugar reaches PPL1 g1/g2 at all (both 0). H4 is NOT TESTABLE
  (sugar D_R = 0).

## Gains (descriptive)

- The largest moves are about 5-6x up, well inside the per-parameter bound: Fox(CB0525) -> CB0272 (FDA-I)
  x5.90; CB0546 (FDA-I) -> PAM01 x5.81; CB0272 -> PAM01 x5.14; CRE011 -> PAM07/PAM08 x5.3-5.5.
  Down about 0.15-0.2x: MBON11 -> PAM03, CRE048 -> PAM03, CB0124 -> CB0272, SMP116 -> CB0272.
- **Compounding:** the largest Fox -> X -> R-PAM two-hop product is **30.3x** (CB0525 -> CB0272 -> PAM01),
  above the 18.287 per-parameter bound. The prereg caps each parameter, not the path; this is disclosed.

## Reading

1. With plausible per-edge gains (g* = 8), the connectome CAN carry Fox activation to the reward
   compartments Christie imaged, and the fitted solution is consistent across seeds.
2. It does NOT transfer to real sugar. The block moves upstream, to sugar GRN -> Fox, which chunk 1
   deliberately left unfitted. So the negative from the three neuron models is sharpened, not
   overturned: taste input never reaches the Fox/FDA hop strongly enough.
3. It is not a valence failure: bitter stays excluded (H3).

## Caveats

- g* may be over-estimated: at caps 2-4 the optimiser oscillated across the all-or-none ignition
  (H37). The POST-HOC pass 2 (no pull, keep-best) tests this; it is running.
- b2 passes by a thin margin (1.0-1.2 Hz against a 1.0 threshold).
- Signs-only targets; ex vivo Christie data; floor 5 only; arm-A rate reduction (no noise).
- Scope decisions (for example owning sugar -> Fox in a later chunk, or a combined chunk) are the
  user's, not made here.
