# Chunk-1 fit on the chunk-0 base: RESULT

Prereg: `PREREGISTER_rate_chunk1_fit.md` @ 2838bfe + amendments 1-2, plus
`PREREGISTER_rate_chunk1_fit_c0.md` @ ee4ecbf (base = chunk-0 calibrated `lin`).
Script `rate/chunk1_fit.py --base c0` @ 2b3ff30. Batch 2026-09-25, 45 cells, analysed once
(auto at batch end). This is path A's chunk-1 answer; `results/rate_chunk1_fit/` (arm-A base,
g* = 8) is superseded history.

## Label

**FIT-NO-TRANSFER, CONSISTENT** (g* = 16, rho_median 0.93 over 84 active edge gains).

| cap | F-pass (of 5) | H-pass (of F-passing) |
|---|---|---|
| 2, 4, 8 | 0 | - |
| **16** | **5** | 0 |
| 18.287, 32, 64, 128, 256 | 5 | 0 |

- g* = 16 is inside the flyvis 99th-percentile bound (18.287), with little margin.
- Every seed at every passing cap fails the same held-out rows: H1, H2, H4, H6.

## Rows at g* = 16 (5 seeds, range)

| Row | Result | Values |
|---|---|---|
| F1 g4 up (Fox) | pass | 5.0-5.5 Hz |
| F2 g5 up (Fox) | pass | 5.6-5.9 Hz |
| F3 b2 up (Fox) | pass, marginal | 1.01-1.03 Hz vs 1.0 |
| F4 b'2 up (Fox) | pass | 3.8-4.1 Hz |
| F5 b1 flat (Fox) | pass | 0.000 |
| H1 g4/g5 up (sugar) | FAIL | g4 0.000, g5 <= 0.0011 Hz |
| H2 b2/b'2 up (sugar) | FAIL | 0.000 |
| H3 bitter excludes reward PAMs | pass | 0.000 |
| H4 Fox silencing drops sugar response | NOT TESTABLE | sugar response < 1.0 Hz |
| H5 g3 flat (Fox) | pass | 0.000 |
| H6 PPL1 g1/g2 bitter > sugar | FAIL | both 0.000 |

## Descriptive (post-hoc, no label effect)

- **Max gain product along a Fox path: 80x** (CB0525 -> CB0546 -> PAM03). No single gain exceeds
  the cap, but the chain product is 4.4x the per-edge bound. Same caveat as the arm-A fit (30x),
  larger here.
- **Fox also drives PPL1 g1 (5.7-6.1 Hz) and g2 (2.0-2.2 Hz)** in every seed, i.e. punishment
  compartments. No fitted or held-out row covers PPL1 under Fox drive (H6 is bitter vs sugar),
  so this is unscored; it should be reported, since a reward interneuron recruiting
  punishment DANs would be a problem if it held up.
- Sugar still stops before the fitted neurons: consistent with the chunk-0 control rerun
  (sugar GRN -> Fox has 0 direct synapses at floor 5; Fox reaches only 10.6 Hz under sugar).
- Compared with the arm-A fit: g* rises 8 -> 16 and Fox-driven PAM rates fall from 60-75 Hz to
  ~4-6 Hz (the C0 base has a 124.5 Hz ceiling and a lower weight scale). Label unchanged.
