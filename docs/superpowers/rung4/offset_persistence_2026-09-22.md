# Rung 4a: nothing outlasts the passive decay. There is no activity memory in this brain.

**Date:** 2026-09-22 (Manila morning / 2026-09-21 ~22:30Z)
**Tool:** `regime/offset_persistence.py`
**Raw:** `results/offset_persistence_dc2.json`, `results/offset_persistence_cx.json` (gitignored)
**Regime:** `eln8` (ELN_NEGATE, PN_KC_GAIN 8.0) -- the same operating point as rungs A and 3.

## The question, and why it is not the LAM trap again

Rung 3 measured an accumulating brain, but every bit of that accumulation lives in synaptic
weights and every update needs a US trial. Between trials the brain holds nothing:
`results/persist_p1s0.jsonl`, all 90 relax rows, `central_active 0.0`. So the capability the
mushroom body does not have is **carrying state in activity**.

The trap is that this is trivially true for a reason we chose. `V_REST == V_RESET == -52.0`,
`V_TH == -45.0`, and `gpu_sim.py` has no background or noise current anywhere. A leaky
integrate-and-fire neuron with no input decays to rest and never spikes. **"The simulated
brain is silent at rest" is not a finding; it is `V_REST`** -- the exact shape of rung 3's
`0.99^30 = 0.7397` forgetting curve.

The question that is not a restatement of a constant: **after the stimulus is switched off,
does activity outlast the passive time constants?** `TAU_M = 20 ms`, so five membrane time
constants is **100 ms**. A spike at t_off + 100 ms cannot be leak. That threshold was fixed
before the run.

## Protocol

Both arms share one 300 ms ignition phase at the same seed, then diverge:

| arm | after t_off |
|---|---|
| **ON** (positive control) | drive continues for 500 ms |
| **OFF** | drive removed, same brain state carried forward |

The carry uses `run_batch(..., state=, return_state=)`, whose exactness is independently
checked by `regime/test_carry.py` (3 x 100 ms carried == 1 x 300 ms, bit for bit; re-run and
passing on this machine 2026-09-22). 500 ms is measured in 25 x 20 ms bins. 3 seeds.

## Result: FAIL, twice, with the control passing both times

### Arm 1 -- odour CS (DC2, 80 Hz, the rung-A stimulus)

| population | n | ignition Hz | ON last spike | OFF last spike (3 seeds) |
|---|---|---|---|---|
| mb:Kenyon_Cell | 5177 | 3.53 | 500 ms | **20, 20, 20 ms** |
| mb:MBON | 96 | 3.51 | 500 ms | **20, 20, 20 ms** |
| mb:ALPN | 685 | 0.81 | 500 ms | **20, 20, 20 ms** |
| all:central | 32383 | 0.67 | 500 ms | **40, 20, 20 ms** |
| all:brain | 139248 | 0.17 | 500 ms | **40, 20, 20 ms** |

KC active fraction at ignition 0.0603 -- identical to the gate. The ON arm fires for the full
500 ms; the OFF arm's last spike anywhere in 139,248 neurons is at 40 ms, and 0.000 Hz at the
100 ms line in every population and every seed.

**Every central-complex population was NOT_ENGAGED in this arm.** Ignition rate 0.000 for
EPG, PEG, PEN, ER, ExR, EL, FB, FC, FS and PFL. The odour CS does not reach the central
complex at all. That is worth recording on its own, and it is why arm 2 exists: "silent after
the offset" and "never switched on" are the same row of zeros and completely different claims,
so the script separates them and excludes NOT_ENGAGED populations from the verdict.

### Arm 2 -- the central complex driven directly (`--drive EPG,PEN,PEG`, 113 neurons, 80 Hz)

| population | n | ignition Hz | ON last spike | OFF last spike (3 seeds) |
|---|---|---|---|---|
| cx:EPG | 51 | 79.74 | 500 ms | **0, 0, 0** (no spike in any post-offset bin) |
| cx:PEG | 20 | 81.56 | 500 ms | **0, 0, 0** |
| cx:PEN | 42 | 78.62 | 500 ms | **20, 20, 20 ms** |
| cx:ER | 278 | 3.26 | 500 ms | 20, 20, 20 ms |
| cx:ExR | 26 | 69.44 | 500 ms | 20, 20, 20 ms |
| cx:FB | 593 | 0.34 | 500 ms | 0, 20, 0 ms |
| cx:PFL | 50 | 0.33 | 480 ms | 0, 0, 0 |
| all:brain | 139248 | 0.12 | 500 ms | **40, 40, 60 ms** |

12 of 21 populations engaged, 8 of them central-complex. The ring-attractor substrate is all
present in this matrix (EPG 47, PEN 42, PEG 20, ER 278, ExR 26, FB 593, PFL 50 by cell type)
and it was firing at ~80 Hz. **The driven EPG population stops completely inside one 20 ms
bin.** Brain-wide the last spike is at 60 ms, still inside the 100 ms passive ceiling.

## The claim

> In the connectome as simulated, no population outlasts its own passive decay. With the
> mushroom-body pathway lit, the whole brain is silent 40 ms after stimulus offset; with the
> central complex lit directly at 80 Hz, 60 ms. The 100 ms passive ceiling is never
> approached, let alone crossed. There is no activity-held state anywhere in this model:
> no working memory, no attractor, no bump to hold.

That answers rung 4's framing question. **What the MB alone cannot do is carry state in
activity -- and neither can anything else here.** Memory in this brain is synaptic or it does
not exist.

## What this is NOT

- **Not a claim about the animal.** `gpu_sim` has no background current, no neuromodulatory
  tone and no intrinsic excitability dynamics. A real ring attractor is sustained by
  recurrent excitation on top of a depolarised baseline that this model does not have. The
  honest statement is bounded to the simulation.
- **Not a claim that the CX wiring is missing.** It is present and it conducts: driving EPG
  lit PEN, PEG, ER, ExR, EL, FB and PFL, and one descending neuron (DNa02, 10.6 Hz). The
  loop carries signal; it just does not sustain it.
- **Not `V_REST` restated.** The measurement is the decay time after an offset, against a
  ceiling computed from `TAU_M` in advance. The result beats the ceiling by 2x, not by a
  hair, in both arms.
- **Not seed-dependent.** Three seeds, identical to the bin in almost every row.

## Consequence: `cx_bump.py` is moot

The plan made a spatial bump-stability test conditional on this one passing. It did not pass,
and arm 2 closes the specific loophole that would have justified running it anyway: the CX was
engaged at 80 Hz and died with everything else. **Do not build `regime/cx_bump.py`.** A bump
cannot survive in a brain where a population driven at 80 Hz for 300 ms stops within 20 ms.

What WOULD make rung 4b meaningful is a change to the neuron model -- a background current, or
intrinsic excitability, or graded/NMDA-like recurrent synapses. That is a real modelling
decision with a real cost, and it is the user's call, not a detail. It is **not** something to
add quietly in order to get a bump: adding a depolarising current until activity persists
would manufacture exactly the result being tested for, which is the `PL.LAM` failure again one
level up.

## Reproduce

```
.venv/Scripts/python.exe regime/test_carry.py
.venv/Scripts/python.exe regime/offset_persistence.py --seeds 3 --off-ms 500 --tag dc2
.venv/Scripts/python.exe regime/offset_persistence.py --seeds 3 --off-ms 500 --drive EPG,PEN,PEG --tag cx
```
~4 min each on the local GPU.
