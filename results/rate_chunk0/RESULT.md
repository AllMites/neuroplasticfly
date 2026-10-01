# Chunk 0 (rate base calibrated to the eln8 LIF) — RESULT

Prereg `1a29779` + Amendment 1 `b554245`. Analysed once (`rate/chunk0.py --analyze`).

**LABEL: BASE-PASS (family `lin`, ladder step 1: 4 global params, no super-class gains).**

| Family | Cal loss (best of 5 starts) | L1 latch | L2 held-out median r (bar 0.759 = 0.8 x C 0.949) | L3 KC frac | L4 offset tail |
|---|---|---|---|---|---|
| lin | 0.0223 | pass | 0.856 pass | pass | pass |
| lif | 0.0344 | pass | 0.786 pass | FAIL | pass |

Chosen `lin` params (start S1; all 5 starts within 0.0223-0.0264, all latch-OK):
w_scale = 0.00494, bias = -9.52 Hz, tau = 17.0 ms, r_max = 124.5 Hz.

Latch (last 500 ms of 1 s off, > 1 Hz; bound = worst eln8 LIF seed + 139):
FOX 0/833, SUGAR 0/139, BITTER 0/834, ORN_DM4 741/838 (13 in reward set; all at r_max).

Held-out KC active fraction: model median 0.056 vs LIF 0.049 (25 odours; one odour 0.022 vs 0.062,
inside the L3 slack). Offset tail median 0 in both.

## Descriptive (no label change)
- The fitted ceiling (124.5 Hz) is what bounds the DM4 latch: all 741 latched neurons sit at r_max.
  The latch survives calibration at the LIF's own size (LIF worst seed ~700), as Amendment 1 allows.
- `lif` family fails only L3 (KC coding too sparse on some held-out odours).

## Next (plan tasks 5-6)
Suite switches to this base; chunk-1 record is stale (measured on arm A) and is re-recorded by
re-running the chunk-1 unfitted control (procedure `194be67`) on this base, with no training.
