# RESULT: chunk-1 fit PASS 2 (POST-HOC)

Prereg `0f0b2f7` Part 1 (designed after seeing one primary cell, H37). Script `4a69198`. 45/45 cells,
analysed once at batch end (2026-09-25 ~01:50 Manila).

## Label: PASS2 (POST-HOC) FIT-NO-TRANSFER (INCONSISTENT)

- **g* = 8, the same as the primary.** Caps 2 and 4: 0/5 seeds F-pass (all 400 iterations, even with
  no pull and keep-best). Caps 8-256: 5/5.
- Held-out at g*: 0/5. Failing rows H1, H2, H4, H6, identical to the primary.
- Ensemble: 928 active gains, median rho 0.419 -> INCONSISTENT. Expected: with no pull toward 1,
  gains that do not matter drift freely. The primary (with the pull) is CONSISTENT and remains the
  preregistered answer.
- Largest Fox -> X -> R two-hop product: 25.7x (CB0525 -> CB0272 -> PAM01).

## Reading

g* = 8 is NOT an artefact of the primary's oscillation (H37). Removing the pull and keeping the best
iterate does not lower it. The primary's g* and label stand.
