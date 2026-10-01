# PREREGISTER: chunk 0 — calibrate the rate base to the stock LIF (path A, step 1)

Written and committed 2026-09-25 08:40 Manila, before any LIF reference is generated and before any
calibration run. Plan: `fly/.claude/PRPs/plans/rate-chunk0-base.plan.md`. User chose path A (fix the base
before any further chunk). Motivation: the arm-A rate base latches (latch check `2c7d0c8`, D52/H38):
7,411 neurons stay on after ORN_DM4, 411 after sugar; the paper-0 LIF is silent 40-60 ms after offset.

## Reference model
Stock LIF (`gpu_sim`, `REGIMES["stock"]`: ELN_NEGATE False, PN_KC_GAIN 1.0), W_SYN 0.275, floor 5,
`data/brain_gpu.npz`. Not eln8: the authored paper-0 edits belong to the chunk-2 bridge.

## Stimuli and split
Single-glomerulus drives: every neuron of one `ORN_*` cell type at 40 Hz. All 53 ORN types in
`data/neuron_meta.npz` (each has >= 12 neurons), sorted by name. Even positions (0, 2, ...) =
CALIBRATION, odd = HELD-OUT, except ORN_DM4 (odd position 19), which goes to CALIBRATION because it is
the latch condition. Result: 28 calibration, 25 held-out. No other drive enters any loss or metric.
Timing: 300 ms on from rest, then 200 ms with no drive (state carried), in 10 bins of 20 ms.

## LIF reference
Seeds 0-4, `rng="counter"`. Per condition: per-neuron mean rate over the 300 ms on-window (per seed),
and per-neuron rate in each 20 ms off-bin (seed mean). Counts / seconds -> Hz.

**Noise ceiling C.** For each held-out condition: Pearson r between the seed-0-1 mean and the seed-2-4
mean of log1p(on-window rate), over neurons with nonzero rate in either half. C = median over held-out
conditions. (Two- vs three-seed halves understate the reliability of the 5-seed mean, so 0.8 x C is
lenient; disclosed, not corrected.)

## Rate-model families (both calibrated; no others)
- **lin** — current engine: r = min(relu(v), r_max). Params (4, global): log w_scale, bias, log tau,
  log r_max. Arm A = (6.880548e-3, -35.0282, 20 ms, 454.5 Hz).
- **lif** — v = mean depolarisation above rest (mV), r = LIF f-I:
  f(v) = 1000 / (T_REFR + TAU_M ln(v / (v - 7))) for v > 7 mV, else 0 (TAU_M 20, T_REFR 2.2, from
  gpu_sim). Params (3, global): log w_scale, bias (mV), log tau. Derived start: w_scale = W_SYN x
  TAU_SYN / 1000 = 1.375e-3 mV per synapse-Hz, bias 0, tau 20 ms.
- Bounds (both): tau clamped to [2, 200] ms (Euler dt 1 ms); r_max to [10, 1000] Hz.

## Optimiser
scipy Nelder-Mead, deterministic, maxfev 300, xatol 1e-3, fatol 1e-4, from 5 fixed starts per family:

| start | lin (w factor, bias, tau, r_max) | lif (w factor, bias, tau) |
|---|---|---|
| S0 | 1, -35.0282, 20, 454.5 | 1, 0, 20 |
| S1 | 0.5, -35.0282, 20, 454.5 | 0.5, 0, 20 |
| S2 | 0.25, -35.0282, 20, 454.5 | 0.25, 0, 20 |
| S3 | 1, -70.0564, 20, 454.5 | 1, -7, 20 |
| S4 | 0.5, -35.0282, 5, 454.5 | 0.5, 0, 5 |

(w factor multiplies the family's start w_scale.)

## Loss (calibration conditions only; held-out data never loaded by the loss)
mean over calibration conditions of
  MSE_neurons(log1p r_rate_on, log1p r_LIF_on)  +  MSE_neurons,bins(log1p r_rate_off, log1p r_LIF_off)
plus 10 if the suite latch check (prereg `c9f0357`, unchanged: FOX, SUGAR, BITTER 150 Hz, ORN_DM4 40 Hz,
1 s on / 1 s off, last 500 ms, 1 Hz) fails. Sugar, bitter and Fox enter ONLY through that pass/fail
penalty, never as rate targets.

**Chosen solution per family** = lowest calibration loss among its 5 starts whose end point passes the
latch check. If both families pass the pass rule, the base is the one with the lower CALIBRATION loss
(never chosen on held-out metrics).

## Pass rule (held-out conditions only, scored once on each family's chosen solution)
- **L1** — the suite latch check passes.
- **L2** — median over held-out conditions of Pearson r(log1p rate_on, log1p LIF_on), over neurons
  nonzero in the LIF 5-seed mean or the model, >= 0.8 x C.
- **L3** — every held-out condition: |KC active frac (model) - KC active frac (LIF)| <= 0.5 x LIF +
  0.01, active = on-window rate > 1 Hz (LIF: 5-seed mean). Kenyon cells = cell_class "Kenyon_Cell".
- **L4** — every held-out condition: fraction of all neurons > 1 Hz over the last 100 ms of the off
  period (bins 6-10) <= the LIF's fraction + 0.001.

## Labels
- **BASE-PASS** — some family passes L1-L4. Record family, params, ladder step. The suite switches to it.
- **LATCH-ONLY** — a chosen solution passes L1 but no family passes L2-L4.
- **NO-BASE** — no chosen solution passes L1.
Every label reports L1-L4 values for both families and the spread of end points across starts.

## Ladder
Step 1 = the global params above. Step 2 runs ONLY if step 1 is not BASE-PASS: add one gain per
super_class (neuron_meta `super_class`, all values), same optimiser (starts = step-1 chosen solution
with all gains 1, plus the four S1-S4 perturbations applied to w_scale/tau only), same loss, same pass
rule. Stop at the first step with BASE-PASS. No step 3 without a new prereg.

## Afterwards (only on BASE-PASS)
The suite switches to the chunk-0 base, and the chunk-1 unfitted control (procedure of prereg `194be67`,
battery unchanged) is re-run on it to re-locate the sugar block, with no training. Descriptive: GRN(sugar)
-> Fox synapse count and share of Fox input.

## Disclosure
An older probe (`results/kc_sparsity_probe.json`, 2026-09-20) shows the stock LIF activating ~33 % of
KCs for ORN_DM4. Chunk 0 matches the LIF, not biology's 5-10 %; sparsity is a chunk-2 (eln8) question.
Not claimable from any label: anything about persistent activity or sparsity in the animal.

---

## AMENDMENT 1 — written and committed 2026-09-25 08:54 Manila, before any eln8 reference or any calibration run

**Why.** The stock reference (generated per the original text, lif_ref.py @ 24118ce) latches: 39/53 odours
leave ~5,300 neurons firing after offset, so L1 (absolute return to rest) contradicts L4 (match the LIF);
see results/rate_chunk0/STOP.md (4fc1cae), H39. User decision D53 = option B: eln8 reference, relative
latch invariant.

**Disclosure of what was seen before this amendment.** The full stock reference (C = 0.9783, per-condition
active counts). One eln8 LIF probe for ORN_DM4 only (1 s on / off, seeds 0-2): 1,493 active on, 700 still
active 500-1000 ms after offset, KC active frac 0.100. No rate-model metric of any kind was computed.

**Changes (everything not listed stays as written above):**
1. **Reference model** = eln8 LIF: ELN_NEGATE True, PN_KC_GAIN 8.0 (`learn/condition.py` REGIMES["eln8"]),
   paper 0's odour operating point. `rate/lif_ref.py` is rerun with this regime; the stock files are kept
   under results/rate_chunk0/stock/ as the record of the stop.
2. **Rate W** gets the same two authored edits before calibration: negate every out-edge of the 44 eLNs
   (`gpu_sim._eln_idx`), and multiply every ALPN -> Kenyon-cell edge by 8. Implemented once in the rate
   engine and checked edge-by-edge equal to gpu_sim's eln8 device weights / W_SYN before any run.
3. **L1 becomes relative (L1').** LIF latch reference: the eln8 LIF under the suite latch-check protocol
   (FOX, SUGAR, BITTER 150 Hz; ORN_DM4 40 Hz; 1000 ms on from rest, 1000 ms off, carried; seeds 0-4).
   Per seed and condition: n_LIF = neurons with rate > 1 Hz over the last 500 ms. L1' passes if, for every
   condition, the rate model's n_above <= max over seeds of n_LIF + 139 (0.001 x 139,248 neurons, the same
   slack as L4). The loss penalty (+10) uses L1'.
4. **The suite latch invariant** (prereg c9f0357) is amended the same way: return to rest is replaced by
   "no more latched neurons than the eln8 LIF + 139", reference numbers stored in
   rate/regress/latch_ref.json. A condition where the LIF returns to rest still requires the rate model to.
5. Labels: NO-BASE now means no chosen solution passes L1'; LATCH-ONLY means L1' passes but L2-L4 do not.
