# STOP before calibration: the reference LIF latches too (prereg 1a29779 pass rule is contradictory)

2026-09-25 ~09:30 Manila. LIF reference generated per prereg (`rate/lif_ref.py` @ 24118ce; lif_ref.npz md5 5801785ec06d…,
C = 0.9783). NO calibration was run; no rate-model metric exists.

## What the reference shows
- 39 of 53 single-glomerulus drives (40 Hz, 300 ms) leave > 100 neurons firing in the last 100 ms of the
  200 ms off period; each latched set is ~5,300 neurons (e.g. ORN_DM4: 5,348). The other 14 barely ignite
  (ORN_DM5: 47 active during the drive, 0 after). All-or-none.
- Descriptive probe (scratchpad lif_latch_probe.py, latch-check timing: DM4 40 Hz 1 s on, then off,
  seeds 0-2):
  | regime | on-window > 1 Hz | off 500-1000 ms > 1 Hz | KC in the latched set | max Hz |
  |---|---|---|---|---|
  | stock | 5,382 (KC frac 0.332) | 5,316 | 1,713 | 419 |
  | eln8 (paper-0 op point) | 1,493 (KC frac 0.100) | 700 | 0 | 193 |
- Rung 4's "whole brain silent 40-60 ms after offset" (offset_persistence_dc2.json) was eln8 + DC2
  odour drive: whole-brain ignition mean 0.17 Hz. It is true for that weak drive, not in general.

## Why this stops chunk 0
The pass rule needs L1 (suite latch check: every neuron < 1 Hz, 500-1000 ms after offset) AND L4
(off-period activity no larger than the LIF's). The LIF itself fails L1 for 39/53 odours, so a faithful
base must fail L1 and an L1-passing base must be unfaithful. The loss penalty would push the fit away
from the reference. Running it would produce NO-BASE by construction.

## Corrections this forces
1. results/rate_latch_check/RESULT.md "what it means" #1 was wrong: the latch is not something the
   rate reduction added. The stock LIF has the same all-or-none latched state (5,316 vs the rate
   base's 7,411 neurons).
2. The latch check's premise ("the paper-0 LIF is silent after offset", prereg c9f0357) holds only
   for weak drives. The invariant must be restated relative to the LIF, or scoped to drives.
3. Paper 0: any rung-4 claim ("no activity memory") must be scoped to the measured drive (eln8, DC2).

## Decision needed (user) before an amendment
See the session report.
