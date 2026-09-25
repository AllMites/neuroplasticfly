# KC excitability under ELN_NEGATE: recruitment is reachable, dead channels are the wall

Phase 1 of the project plan. The gate asked for KC active fraction 5-10%, odour Jaccard
< 0.3, and zero silent channels, at a physiological ORN rate, under `ELN_NEGATE`.

**Verdict: the gate as written FAILS, on one criterion only — silent channels. Recruitment
and separation both pass comfortably, at the stock spike threshold.** Whether the failing
criterion matters depends on how many odour channels the experiment actually needs; see the
verdict section.

Reproduce: `.venv/Scripts/python.exe kc_excitability_sweep.py`.
Raw: `results/kc_excitability_sweep.json`. Test: `regime/test_kc_th.py`.
Eight single ORN channels (D, DA1, DA2, DA3, DA4l, DA4m, DC1, DC2) at 80 Hz, `ELN_NEGATE`
on, 300 ms, Jaccard scored over ignited pairs only.

## Two new AUTHORED levers, both default-off

`KC_V_TH_DELTA = 0.0` — mV added to the spike threshold of Kenyon cells only. Negative is
more excitable. `PN_KC_GAIN = 1.0` — multiplier on the 21,509 ALPN -> Kenyon-cell edges.
Both follow the existing convention: default is a no-op, so prior results reproduce bit for
bit (verified against the 2026-09-20 `kc_sparsity_fine.py` output). `V_TH` itself is a Shiu
constant and is untouched; `regime/test_kc_th.py` asserts all nine of them.

## 1. The threshold lever works, and is bounded by V_REST

| `KC_V_TH_DELTA` | 0.0 | -2.0 | -4.0 | -5.0 | -6.0 | -6.5 | **-7.0** | -7.5 |
|---|---|---|---|---|---|---|---|---|
| KC active frac | 0.0074 | 0.0117 | 0.0219 | 0.0233 | 0.0281 | 0.0394 | **0.0787** | 1.0000 |
| KC Jaccard | 0.000 | 0.004 | 0.008 | 0.012 | 0.016 | 0.020 | 0.052 | 1.000 |
| silent channels | 4 | 2 | 2 | 2 | 1 | 1 | 1 | 0 |
| APL Hz | 10.0 | 17.3 | 37.7 | 59.6 | 81.5 | 90.6 | 105.4 | 453.3 |
| KC frac, zero drive | 0 | 0 | 0 | 0 | 0 | 0 | 0 | **1.0000** |

`V_TH` is -45 and `V_REST` is -52, so delta -7.0 puts a Kenyon cell's threshold **exactly on
its resting potential** and -7.5 puts it below: the whole brain free-runs, every KC active
with no input at all, Jaccard 1.0. The zero-drive control row exists to catch precisely this
and it did.

So the band is reachable at -7.0 — 0.0787 active, Jaccard 0.052, nothing firing
spontaneously — but only at the degenerate limit, where a KC has no threshold margin left
and fires on any positive input whatsoever. That is the opposite of the high-threshold
sparse-firing cell the mechanism is supposed to model. **Reaching the gate this way would be
a number that passes and a cell that does not exist.** Hence the second lever.

## 2. PN->KC gain reaches the band at the stock threshold

| `PN_KC_GAIN` (delta 0.0) | 1.0 | 1.5 | 2.0 | 3.0 | 4.0 | 6.0 | **8.0** |
|---|---|---|---|---|---|---|---|
| KC active frac | 0.0074 | 0.0141 | 0.0218 | 0.0301 | 0.0378 | 0.0454 | **0.0538** |
| KC Jaccard | 0.000 | 0.004 | 0.007 | 0.016 | 0.022 | 0.022 | 0.025 |
| silent channels | 4 | 2 | 2 | 2 | 2 | 1 | 1 |
| APL Hz | 10.0 | 22.5 | 39.0 | 78.5 | 107.1 | 138.5 | 160.8 |
| MBON active | 4.6 | 5.5 | 5.8 | 5.0 | 4.1 | 8.1 | 10.4 |
| KC frac, zero drive | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

At gain 8.0: **5.4% of Kenyon cells active, Jaccard 0.025 against a 0.3 target, no
spontaneous firing, MBON population healthy, and `V_TH` never touched.** This is the
operating point to prefer.

Combining the levers (delta -3.0 with gain 1.5-4.0) is strictly worse than gain alone — it
reaches only 0.0433 at gain 4.0 while spending threshold margin. Do not combine them.

## 3. APL is the clamp, and it re-engages on its own

APL mean rate climbs 10 -> 161 Hz across the gain sweep while KC recruitment climbs only
0.7% -> 5.4%. Under `ELN_NEGATE` APL starts nearly silent (the PN layer barely fires); as
Kenyon cells are recruited, APL wakes up and inhibits them back. That negative feedback is
why recruitment rises sublinearly in both sweeps, and it is the correct behaviour — APL
doing its actual job of enforcing sparse coding. It was the predicted top risk in the plan
and it is a feature, not an obstacle: it is what keeps the code sparse instead of letting it
run away as the drive goes up.

## 4. The real wall is dead channels, not recruitment

Per-channel KC recruitment at `PN_KC_GAIN = 8.0`, 80 Hz:

| channel | D | DA1 | DA2 | **DA3** | DA4l | DA4m | DC1 | DC2 |
|---|---|---|---|---|---|---|---|---|
| ORNs | 31 | 126 | 39 | 30 | 40 | 40 | 39 | 20 |
| KC active frac | 0.0552 | **0.0778** | 0.0010 | **0.0000** | 0.0363 | 0.0137 | 0.1331 | 0.0591 |

DA3 reaches zero Kenyon cells at any setting tested. DA2 reaches five. Everything else is
live, and four channels — DA1, DC2, D, DA4l — sit in or near the 5-10% band with DC1 a
little above it.

The gate's zero-silent-channels criterion therefore fails on a **pre-existing defect in two
specific channels**, not on the excitability work. That defect is already a separate open
question in the project plan and may be another connectome gap of the same family as the APL
in-degree bug.

## Verdict

- **Recruitment: solved.** `PN_KC_GAIN = 8.0` gives 5.4% mean KC activity with the stock
  spike threshold and no spontaneous firing.
- **Separation: solved, with enormous margin.** Jaccard 0.025 against a 0.3 target.
- **Zero silent channels: FAILED.** DA3 is dead, DA2 nearly so.

The honest question is whether the third criterion is the right one. It was written as
"all CS channels live", and the reel needs **one** odour as the CS — two if a discrimination
arm is added. On that reading the gate is satisfied: pick the CS from DA1 (0.0778) and, if a
second odour is wanted, DC2 (0.0591) or D (0.0552). Both are squarely in band and mutually
separable.

**This doc does not declare the gate passed.** Redefining a criterion after seeing the
result is exactly how the 40 Hz false-pass happened last time. The recommendation is to
narrow the criterion deliberately and in advance — "the chosen CS channels are live and in
band", with DA1 and DC2 named — and to record that as a decision in the plan rather than as
a silent reinterpretation here.

## Measured out — do not re-run

- **ORN drive rate.** Logarithmic; 2.3% at 800 Hz (2026-09-20).
- **APL gain as a sparsity fix.** Cannot reach the merge; PN Jaccard invariant at 0.955-0.961.
- **KC threshold beyond -6.5.** Hard-bounded by `V_REST`; -7.0 is degenerate and -7.5 free-runs.
- **Threshold and gain combined.** Worse than gain alone at every point tested.

## Open, in order of value

1. Why DA3 reaches zero Kenyon cells and DA2 reaches five. Check their ALPN in/out degrees
   against the raw Zenodo export, the way the APL in-degree bug was caught.
2. Whether `PN_KC_GAIN = 8.0` is defensible as a disclosed authored parameter. It is a
   large multiplier; the honest framing is that the connectome gives synapse *counts*, not
   conductances, and `W_SYN` is a single global mV-per-synapse constant for the whole brain.
   Worth stating exactly that way in the pinned comment.
3. Whether the gain changes the KC code's *content*, not just its size — the Jaccard says
   channels stay distinct, but nothing here checks that the identity of the recruited cells
   is stable as gain rises.
