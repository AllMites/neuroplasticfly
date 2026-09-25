<!-- AUTHORITATIVE record of the P1 mechanism work and the layer audit that ended it.
`results/` is gitignored (`.gitignore` line 7), so every raw measurement quoted below
exists only on the host that produced it. This committed file is the record. Nothing
here can be regenerated from a clean clone without re-running the probes. -->

# Regime stability -- handoff

Date: 2026-09-19. Branch `master`. HEAD at time of writing `33cbf57`.
Continues: `docs/superpowers/regime-p0/findings.md`.
Spec: `docs/superpowers/specs/2026-09-19-regime-stability-design.md` (now partly stale, see
"Known bugs and gaps").
Plan: `docs/superpowers/plans/2026-09-19-regime-stability-p0-p1.md`.

---

## Headline

**Odour identity is destroyed at ORN -> antennal lobe, in one synaptic step.**

Jaccard between the two odour responses goes 0.000 -> 0.976 (ALLN) in a single hop, then
stays 0.93-0.98 at every layer downstream. Identical with all mechanisms off and in the
sparse regime. Everything built in P1 operates three layers below the actual failure.

### Evidence line 1: the connectome is perfectly separated

Monosynaptic excitatory targets:

| source | targets | Jaccard |
|---|---|---|
| ORN_DA1-left -> ALPN | 17 ALPNs | - |
| ORN_DM4-left -> ALPN | 4 ALPNs | - |
| intersection | 0 | **0.000** |

Two hops uPN -> KC gives 0.050. ALLN monosynaptic targets overlap 6/27, J=0.222.
The wiring carries the odour. The dynamics discard it.

### Evidence line 2: a fixed global attractor with a per-channel trigger threshold

21 DM4 neurons at 2 Hz ignite 4,034 central neurons. DA1 produces 0 central at 2 Hz,
1 at 20 Hz, 8 at 40 Hz, then 5,114 at 60 Hz. Below threshold nothing; above it, the same
everything.

### Evidence line 3: the ignited set is drive-nested and odour-independent

DM4@2Hz against DA1@60Hz -- different odours, different rates, far apart on the drive axis:

| layer | Jaccard |
|---|---|
| ALPN | 0.861 (311 of DM4's 312 active ALPNs lie inside DA1's set) |
| uPN | 0.924 |
| KC | 0.843 |
| central | 0.782 |

### Rate does not rescue it

Mean-subtracted cosine removes the shared gain term, so it tests whether the rate PATTERN
across cells differs even when the level does not:

| layer | mechanisms off | sparse regime |
|---|---|---|
| ALPN | 0.997 | 0.997 |
| KC | 0.995 | 0.988 |
| central | 0.998 | 0.998 |

The pattern across cells is the same for both odours, not merely the level.

### Consequence

**The connectome data is sufficient.** The failure is in the simulation's dynamics:
uniform `W_SYN` plus heavy-tailed convergence means any suprathreshold input triggers a
brain-wide cascade. This is a modelling failure, not a missing-physiology failure.

### And it names the right component

**APL cannot be the fix, in any functional form, at any strength.** APL is 2 neurons, so
its effect on any Kenyon cell is one scalar per hemisphere times that cell's APL weight --
an odour-INDEPENDENT per-cell factor varying only about 1.4x across KCs. APL can set HOW
MANY KCs fire. It can never set WHICH.

**ALLN is the untouched candidate.** 429 neurons at exactly the layer where identity dies.
Lateral inhibition among antennal lobe local neurons is the standard mechanism for odour
decorrelation. Nothing in P1 touched it.

---

## Layer table

60 Hz, 3 seeds, mean. Condition A = all mechanisms off. Condition B = graded APL, delayed,
`APL_MAX_HZ=900` (KC 4.4% active).

| layer | n | A: act DA1/DM4 | A: Jaccard | B: act DA1/DM4 | B: Jaccard |
|---|---|---|---|---|---|
| ORN driven union | 81 | 60 / 21 | 0.000 | 60 / 21 | 0.000 |
| ORN all | 2282 | 60 / 21 | 0.000 | 60 / 21 | 0.000 |
| ALLN | 429 | 231 / 230 | 0.976 | 230 / 228 | 0.983 |
| ALPN all | 685 | 360 / 345 | 0.940 | 342 / 327 | 0.928 |
| ALPN uni | 277 | 196 / 184 | 0.935 | 189 / 174 | 0.921 |
| LHLN | 514 | 394 / 396 | 0.988 | 385 / 385 | 0.982 |
| KC | 5177 | 1701 / 1712 | 0.964 | 229 / 241 | 0.891 |
| MBON | 96 | 35 / 35 | 0.981 | 15 / 15 | 1.000 |
| central | 32383 | 5113 / 5097 | 0.946 | 3368 / 3369 | 0.932 |

Selectors, all from `data/neuron_meta.npz` `cell_class` except uPN/mPN which need
`cell_sub_class`: olfactory, ALLN, ALIN+ALON (38, soft), ALPN, ALPN+uniglomerular,
ALPN+multiglomerular, LHLN, Kenyon_Cell, MBON, DAN (331, soft),
`super_class == central`.

**This is NOT saturation.** In condition A only about 50% of ALPN and 33% of KC are
active. A defined subset fires. The subsets simply coincide.

---

## What exists in code

All AUTHORED mechanisms live in `gpu_sim.py`, ALL DEFAULT OFF. The OFF path is verified
bit-identical: `central>1Hz` exactly 0.172, and `learn/condition.py --state v1 --check`
printed ACCEPT after every commit.

| constant | default | what it is |
|---|---|---|
| `TAU_A` | 100.0 | spike-frequency adaptation decay; state `a` threaded through `_body` |
| `SFA_B_INC` | 0.0 | adaptation increment per spike (0 = SFA off) |
| `APL_GRADED` | False | graded non-spiking APL |
| `APL_SCALE` | 7.0 | mV above rest at which graded output saturates |
| `APL_MAX_HZ` | 450.0 | rate-equivalence factor; act=1 matches the spiking ceiling |
| `APL_DELAYED` | True | 18-step (1.8 ms) delay line on graded APL output |
| `APL_DIVISIVE` | False | divisive form, `g/(1+inh/S)` |
| `APL_DIV_SCALE` | 11.75 | divisor half-point |
| `NORM_TOTAL_TARGET` | 0.0 | E/I-preserving homeostatic normalisation (0 = off) |

Supporting code: the `no_spike` mask in `_body`; `GpuSim._apl_activation`,
`_deliver_apl`, `_input_scale`. In `learn/plastic.py`, `signed_values(w_syn, scale=None)`
and `push()` both pass `sim.norm_scale`, so normalisation is not silently reverted on the
18,674 plastic edges.

`regime/` package: `reach.py` (graph reachability; bool-mask guard, saturation flag,
`min_w` floor), `weights.py` (per-neuron input totals), `probe.py` (ORN drive probe;
interface fixed by contract), `measure.py` (six-arm), plus `test_reach.py`,
`test_weights.py`, `test_sfa.py`, `test_apl.py`, `test_norm.py`.

---

## Negative results -- record these so nobody repeats them

All at 60 Hz, both channels, replicated where stated. Reference with everything off:
KC 0.33 active, central 0.157, Jaccard 0.96.

**Spike-frequency adaptation is a RATE control, not a sparsity control.**
`SFA_B_INC` 0 -> 4.0 cuts KC mean rate 10.45 -> 3.99 Hz while occupancy barely moves,
0.328 -> 0.315, and Jaccard does not move at all, 0.961 -> 0.960. `TAU_A` was not swept.

**`APL_SCALE` is saturated before you start.** Measured mean activation is already 0.851
at the 7.0 default. A 20x cut to 0.35 buys 0.851 -> 0.896. The whole axis holds about
1.2x more inhibition. Instrumenting actual activation was load-bearing: without it the
sweep reads as "APL does nothing".

**`APL_MAX_HZ` scales occupancy smoothly but never identity.**

| `APL_MAX_HZ` | 450 | 600 | 750 | 900 | 1100 | 1350 |
|---|---|---|---|---|---|---|
| KC active | 0.279 | 0.166 | 0.084 | 0.044 | 0.019 | 0.005 |
| Jaccard | 0.962 | 0.952 | 0.947 | 0.938 | 0.921 | 0.758 |

The 1350 row is meaningless -- KC 0.6%, union about 170. **This killed the earlier "true
bistability" finding** in `findings.md` P0.3: bistability was an artifact of varying only
the drive rate.

**The 1.8 ms loop delay does not matter.** Delayed against zero-delay matches to 2-6%
relative at every rate; at matched occupancy, 0.891 +/- 0.006 against 0.918 +/- 0.021 --
a 0.027 gap against about 0.02 pooled spread. Reason: 1.8 ms is short against `TAU_M` 20 ms
and `TAU_SYN` 5 ms.

**Divisive inhibition is no better than subtractive, and slightly worse.** At matched 4.7%
occupancy: divisive 0.920 +/- 0.004 against subtractive 0.891 +/- 0.007. Reason: `inh_j` is
one scalar activation per hemisphere times a per-cell weight, so the divisor is
odour-independent and applies identically to both patterns. This is the same argument as
the APL headline, arriving by a different route.

**Homeostatic normalisation is structurally self-defeating in a fixed-threshold model.**
Threshold is `V_TH - V_REST` = 7 mV. Normalising total input to budget T makes every
neuron need the same fraction 7/T of its entire input coincident within `TAU_SYN`,
independent of convergence. At the derived T = 29.15 mV that is 24%. Reaching a plausible
2-3% needs T of about 230-350 mV, 8-12x the median -- a brain-wide gain increase, not
normalisation. Excitation-only normalisation was worse still: it inverted E/I (41% of
neurons carried more inhibition than their entire excitatory budget) and nothing
propagated one hop.

**Ruled out earlier, kept here for completeness:** global weight scaling (Jaccard
0.96-0.99 at every scale, collapse at 0.3x) and APL connectivity (the corrected matrix
restores APL in-degree 2,843 / 2,818 and it still ignites).

---

## Known bugs and gaps

**FIXED 2026-09-19 pm (see `alln-findings.md` s1; every number below re-measured within 0.01).** `regime/probe.py:measure()` assigned DIFFERENT seeds to the two channels.
Line 33: `sim.run_batch(drives, list(range(seed0, seed0 + len(drives))), ...)` gives pairs
(0,1), (2,3), (4,5). Odour differences are therefore confounded with noise differences in
EVERY measurement in this document. The confound inflates dissimilarity, so it cannot have
manufactured the high Jaccards -- true overlap may be higher still -- but fix it before
publishing any of these numbers.

**The six-arm `all` result in commit `c2dae63` is VOID.** Graded APL's weights bypassed
normalisation: `apl_w` was frozen before the normalisation block. Fixed in `8777c62`. Not
re-measured.

**Unswept / untested.** `TAU_A` was never swept. Divisive inhibition on split E/I channels
is untested -- `g` is one pooled conductance. The time course of ignition is unmeasured;
the attractor claim rests on the drive sweep and the cross-rate set nesting, not on
watching it propagate.

**The spec's P1 is stale.** It describes three mechanisms as the plan. Four in-flight
mechanism changes have happened since (`9e256b6`, `ae11976`, `8777c62`, `33cbf57`).
Acceptance criterion 2 (survives plasticity) was never run, because criterion 1 never
passed.

---

## Provenance

`data/brain_gpu.npz` was modified during P0.3: the selector table grew 26 -> 132, because
no `cell_type=ORN_*` selectors had ever been exported.

- W matrix and pools are byte-identical to `data/brain_gpu.npz.bak`. `nnz` still 2,700,429.
- All 106 added selectors independently verified index-for-index against
  `data/neuron_meta.npz`.
- `brain_state/v1` still loads with its edge-fingerprint assert passing.
- Quote per-array shas as provenance, never the whole-file one (which is stale):
  `W_indptr` `cd16205c5b82e581`, `W_indices` `5c77ceec505cdd2c`,
  `W_data` `cbc207ce1f1196f9`.
- 29 ORN neurons have `side == "na"` and are unreachable by any left/right selector.
- `export_brain.py` gained `orn_selectors(n)`. That code has NEVER been executed.

---

## Next steps

Steps 1, 2 and 4 done in `alln-findings.md` (2026-09-19 pm). Step 3 dropped: no KC-layer
mechanism can matter while the AL merges the odours.

1. **Fix the probe seed confound first.** It touches every number in this document.
2. **The ALLN lateral-inhibition question.** 429 neurons at the layer where identity dies,
   with enough degrees of freedom to separate patterns, against APL's 2 which provably
   cannot. This deserves a fresh spec, not a fifth in-flight mechanism change.
3. **Re-measure the void `all` arm** on current HEAD.
4. **Measure at the ALPN layer, not the KC layer.** Whatever comes next, measuring at KC
   has been looking three synapses past the problem all day.
