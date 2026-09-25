<!-- AUTHORITATIVE record of the P0 diagnostic runs. The raw measurements live in
results/regime_p0_reach.json, regime_p0_weights.json, regime_p0_sweep.json and
regime_p0_sweep_fine.json, all of which are gitignored and local-only. This file is the
committed record; the JSONs carry more fields but nothing that contradicts what is below. -->

# Regime stability P0 -- findings

Date: 2026-09-19
Answers: `docs/superpowers/specs/2026-09-19-regime-stability-design.md`, section "P0 - Diagnostic gate".
Plan: `docs/superpowers/plans/2026-09-19-regime-stability-p0-p1.md`, Task 3.
Read-only run. No change to `gpu_sim.py` or `learn/`.

Headline, stated before the detail:

1. The explode-or-die hypothesis is **half supported**. The explode half holds. The die
   half does not: both dead pathways' first-hop targets sit near the TOP of the input
   distribution. Ignition and the dead pathways need separate explanations.
2. The first reachability conclusion -- "both dead pathways are dynamics failures, not
   wiring failures" -- is **withdrawn**. Unweighted reachability does not beat a
   size-matched random null on this graph.
3. Both gates open. P3 (audio -> KC) PROCEED. P2 (gustatory US) PROCEED, QUALIFIED.

---

## P0.1 Reachability -- and a retraction

### What was withdrawn, and why

The first run reported JO -> KC at 2 hops and sugar_GRN -> PAM at 2 hops, each with at
least 8 (later at least 64) vertex-disjoint routes, and that was read as both dead
pathways being dynamics failures rather than wiring failures.

**That conclusion is withdrawn.** A size-matched random null was then run and showed that
arbitrary neuron sets are 1 hop apart with >= 64 disjoint routes. On a 139,248-node /
2,700,429-edge small-world graph, unweighted reachability connects almost anything to
anything and carries no information. Supporting detail: the shortest JO -> KC path found
was two synapses both at exactly weight 5.0 -- the `min_syn` floor, where 20.7% of all
edges sit. The paths existed, but they were made of the weakest edges in the connectome.

The module was fixed in commit `e6905d1`: bool-mask inputs now raise instead of silently
returning `hops=0`; `n_disjoint` returns a saturation flag so a capped count is not read
as a measurement; and a `min_w` floor lets the question be asked over non-trivial
synapses.

### Re-run

`n_disjoint` cap 64. Null = mean over 5 seeds, size-matched random source/destination sets
drawn disjointly. `*` marks a saturated lower bound, not a measurement.

| pair | min_w | real hops | real n_disj | real weight | null hops | null n_disj |
|---|---|---|---|---|---|---|
| JO->KC | 0 | 2 | 64* | 10.0 | 1.0 | 64.0* |
| JO->KC | 10 | 3 | 64* | 58.0 | 1.0 | 64.0* |
| JO->KC | 25 | 4 | 64* | 165.0 | 1.0 | 64.0* |
| sugar_GRN->PAM | 0 | 2 | 64* | 13.0 | 1.0 | 64.0* |
| sugar_GRN->PAM | 10 | 3 | 64* | 60.0 | 1.2 | 64.0* |
| sugar_GRN->PAM | 25 | 6 | 5 | 266.0 | 2.0 | 64.0* |
| bitter_GRN->PPL1 | 0 | 3 | 64* | 43.0 | 2.0 | 64.0* |
| bitter_GRN->PPL1 | 10 | 3 | 45 | 46.0 | 2.4 | 64.0* |
| bitter_GRN->PPL1 | 25 | 4 | 12 | 216.0 | 3.2 | 21.6 |

Set sizes: JO 1104, sugar_GRN 129, bitter_GRN 65, KC 5177, PAM 307, PPL1 16.
Wall time 61 s.

### Two things that have to be said plainly

**1. Delta-hops is not a usable statistic.** Random pairs join in 1 hop THROUGH A HUB. Any
specific sensory relay chain -- receptor, projection neuron, target -- is necessarily
longer than that, by construction. The real pathways coming out "longer than random" is an
artifact of the null's hub route, not evidence that they are poorly connected. Do not quote
the `delta_real_minus_null.hops` field in the JSON as a finding.

**2. The informative statistic is disjoint routes once the weight floor bites.** There the
three pathways separate from EACH OTHER, which needs no null at all:

- **JO -> KC**: >= 64 routes at every floor, never thins. Richly wired even over strong
  synapses.
- **bitter_GRN -> PPL1**: 45 routes at min_w=10, 12 at min_w=25. Thins moderately.
- **sugar_GRN -> PAM**: 5 routes at min_w=25, and 6 hops to get there. Collapses.

That ordering matches the measured physiology exactly: audio produces 0 spiking KCs;
bitter -> PPL1 gives 0.83-1.9 Hz, seed-dependent; sugar -> PAM gives exactly 0.00 Hz on
every seed (see `docs/superpowers/condition-v1/condition_v1_notes.md`, disclosure section).

### Caveat to record

The null is size-matched but NOT degree-matched or region-matched. A random pair that
happens to sit next to a hub is an unfair comparator for a sensory chain. The relative
ordering of the three pathways does not depend on the null in any way, and that ordering is
the finding to rely on.

---

## P0.2 Weight distribution

Brain-wide summed excitatory input per neuron. Because `W_SYN` is uniform, these are
summed synapse counts.

| quantile | p1 | p5 | p25 | p50 | p75 | p95 | p99 |
|---|---|---|---|---|---|---|---|
| exc input | 0.0 | 0.0 | 15.0 | 50.0 | 107.0 | 565.0 | 1761.0 |

E/I median ratio 1.362. 14,064 neurons have zero input at all.

| group | n | exc median | exc pct rank | in-degree median |
|---|---|---|---|---|
| KC | 5177 | 71.0 | 61.2 | 6.0 |
| PAM | 307 | 16.0 | 25.7 | 4.0 |
| PPL1 | 16 | 987.0 | 97.5 | 124.0 |
| APL | 2 | 62443.5 | 100.0 | 2830.5 |
| MBON | 96 | 1233.0 | 98.2 | 165.0 |

Dead-pathway sources and their first-hop targets. Brain median exc input 50.0, brain median
in-degree 9.0:

| set | n | exc median | exc pct rank | in-degree median |
|---|---|---|---|---|
| sugar GRN source | 129 | 5.0 | 15.9 | 4.0 |
| sugar GRN targets | 149 | 608.0 | 95.4 | 61.0 |
| JO source | 1104 | 0.0 | 7.8 | 1.0 |
| JO targets | 991 | 152.0 | 83.2 | 31.0 |

### Verdict, which the spec asked for explicitly

**The explode half is SUPPORTED. The die half is NOT.**

**Explode: supported.** A 35x spread from median (50) to p99 (1761) under a single uniform
`W_SYN = 0.275 mV`, with APL at 62,443 summed excitatory input and in-degree 2,831. That is
a real explode-or-die geometry: the same per-synapse weight that leaves a median neuron
near threshold puts the convergent tail far past it.

**Die: not supported.** Both dead pathways' first-hop targets sit near the TOP of the
distribution -- percentile rank 95.4 for sugar GRN targets and 83.2 for JO targets. Those
are exactly the neurons that should be easiest to drive. They are not dying for want of
convergence at the first hop, and the "long thin paths die" clause of the spec's hypothesis
does not describe them.

**What fits instead: PAM is intrinsically thin.** In-degree 4 and percentile rank 25.7,
against PPL1's in-degree 124 and rank 97.5 -- a 30x asymmetry in in-degree between the two
DAN populations. The sugar pathway dies at the LAST hop, not the first.

### min_syn = 5 was tested as a cause and RULED OUT

The corrected build was compared against the superseded
`brain_gpu_badbuild_2026-09-16.npz`, which kept 854k sub-threshold pairs. In-degree roughly
halves uniformly across all groups: KC 8 -> 6, MBON 224 -> 165, PPL1 257 -> 124, PAM
8 -> 4. PAM is not disproportionately affected, and the PAM/PPL1 asymmetry persists in both
builds. The threshold is not the reason sugar -> PAM is dead.

### Consequence

The earlier working hypothesis -- that ONE uncalibrated `W_SYN` explains both ignition and
both dead pathways -- is **wrong**. It explains ignition. It does not explain the dead
pathways. The two need separate explanations and should not be fixed with one knob.

---

## P0.3 Regime characterisation

Channels: `cell_type=ORN_DA1,side=left` (60 neurons) and `cell_type=ORN_DM4,side=left`
(21 neurons). `t_run = 300` ms.

### True bistability, not a steep sigmoid

Across roughly 90 seed-runs, every single one landed either at exactly `kc_active` 0.0000
or inside 0.265-0.333. ZERO intermediate values were observed anywhere in the sweep
(`intermediate: 0` on every seed-spread entry).

Drive rate controls the PROBABILITY of ignition, not its amplitude:

| channel | drive Hz | seeds ignited (of 8) |
|---|---|---|
| ORN_DA1-left | 40 | 3 |
| ORN_DA1-left | 44 | 5 |
| ORN_DA1-left | 46 | 6 |
| ORN_DA1-left | 46.5 | 7 |
| ORN_DA1-left | 48 | 8 |
| ORN_DM4-left | 1.10 | 0 |
| ORN_DM4-left | 1.15 | 1 |
| ORN_DM4-left | 2 | 3 |
| ORN_DM4-left | 3 | 4 |
| ORN_DM4-left | 5 | 8 |

### The step is sharp

At fixed seed, the width between `kc_active < 0.05` and `kc_active > 0.25` is <= 0.05 Hz
for DM4 (1.10 -> 1.15) and <= 0.5 Hz for DA1 (46.0 -> 46.5). That is about 1% of threshold
for both. The stochastic band -- the range over which the ignited fraction of seeds goes
from some to all -- is wider: 40-48 Hz for DA1, 1.15-5 Hz for DM4.

### The two channels differ 40x in threshold

DM4 ignites at 1.15 Hz. DA1 ignites at 46.5 Hz. That is a 40x difference in threshold
against only a 3x difference in neuron count (21 vs 60), and **the SMALLER channel is the
more excitable one**.

Consequence, flagged for P1: any P1 fix must be measured on BOTH channels. A mechanism
tuned on DA1 alone could leave DM4 still igniting at 1 Hz and look like a success.

### A second step exists within the ignited branch

DA1 sits at KC 3.4 Hz / APL 103 Hz from 46.5 Hz up to 50 Hz, then jumps to KC 10.1 Hz /
APL 298 Hz at 52 Hz and above. `kc_active` barely moves across that jump (0.318 -> 0.328),
so the ignited state is not one state; it is at least two.

### No sparse odour-specific window exists anywhere on the drive axis

Matched-drive Jaccard between the two channels is 0.941-0.970 at every rate where both
ignited. Where Jaccard is low it is only because one channel is completely silent, which is
not selectivity.

The most favourable test available was run explicitly: each channel at its own
just-suprathreshold state, DM4 at 1.15 Hz against DA1 at 46.5 Hz. That gives Jaccard 0.834.
But DM4's own low-vs-high branch -- same channel, same odour, 1.15 Hz vs 10 Hz -- gives
0.804. So the 0.834 is branch difference, not odour identity. Identity is erased at the
faintest possible ignition.

This independently reproduces the spec's "global weight scaling is not enough" result on a
different axis: there is no drive rate at which this brain is sparse and selective.

---

## The gate

The spec's gate was conditional on reachability: no path means the piece is cancelled and
the authored workaround becomes permanent. Both pathways have paths, so neither is
cancelled. The disjoint-route counts then say how much confidence each deserves.

- **P3 (audio -> KC): PROCEED.** Wiring is robust even over strong synapses: >= 64 disjoint routes at min_w=25, never thinning at any floor. The failure is dynamics, not wiring.
- **P2 (gustatory US): PROCEED, QUALIFIED.** Two independent measurements agree the pathway is genuinely thin -- 5 disjoint routes at min_w=25, and PAM intrinsically low in-degree (4, pct rank 25.7). This is not purely a regime problem, and P2 is harder than the spec assumed.

Said plainly for P2: fixing the regime may well not be sufficient to make sugar -> PAM
carry signal. The connectome route from sugar GRNs to PAM is real but narrow, and PAM is
the least-driven population measured in this audit. P2 should be planned on the assumption
that it needs its own answer, not that it falls out of P1.

---

## Provenance

`data/brain_gpu.npz` was modified during P0.3. The selector table grew from 26 to 132
entries, because no `cell_type=ORN_*` selectors had ever been exported and the probe could
not run without them. What that patch did and did not touch:

- The W matrix and the pools are byte-identical to the pre-patch backup. `nnz` is still
  2,700,429. `brain_state/v1` still loads with its edge-fingerprint assert passing.
- All 106 added selectors were independently verified index-for-index against
  `data/neuron_meta.npz`. The 26 original selectors are unchanged.
- A whole-file sha256 recorded in an earlier commit message is therefore now **stale** and
  should not be quoted. The load-bearing per-array shas, which are invariant to selector
  additions, are: `W_indptr` `cd16205c5b82e581`, `W_indices` `5c77ceec505cdd2c`, `W_data`
  `cbc207ce1f1196f9` (sha256, first 16 hex chars). Quote these as provenance instead.
- `data/brain_gpu.npz.bak` is the pre-patch snapshot: the same matrix minus those 106
  selectors. `data/` is gitignored, so it is local only.
- 29 ORN neurons have `side == "na"` and are therefore not reachable by any exported
  left/right selector. Worst case is ORN_VA1v, where 10 of 94 cells are unreachable this
  way. Quote per-channel neuron counts (DA1-left 60, DM4-left 21) with that in mind.
- `export_brain.py` gained an `orn_selectors(n)` helper so that a real container export
  reproduces the patch. **That code has never been executed.** Run it before anyone relies
  on it.

---

## Consequence for P1

The spec's re-scope clause is now live. It reads:

> **Consequence.** If ignition is mostly a convergence artifact of uniform `W_SYN`, then
> weight calibration stops being a diagnostic and becomes the mechanism, and P1 is
> re-scoped before implementation rather than after.

Half of the antecedent is met. The convergence geometry is real -- 35x median-to-p99 spread
under one uniform weight, APL at in-degree 2,831 -- so ignition does look like a convergence
artifact. The other half of the original hypothesis, that the same uniform weight kills the
long thin pathways, is refuted, so the dead pathways are not an argument for calibration.

What that means for the two planned mechanisms, laid out without deciding between them:

- **Graded APL (P1.1) and spike-frequency adaptation (P1.2)** are gain control. They act on
  the symptom of convergence -- too much summed drive arriving at once -- without changing
  the underlying weight geometry. They remain defensible without external data, which was
  their main argument, and P0.3's bistability is exactly the kind of runaway that
  adaptation is supposed to damp. Against them: the spec already records that a perfect APL
  reached only KC 53% active, and P0.3 shows the ignited branch has internal structure (the
  second step at 52 Hz) that a single gain term may not resolve. P0.3 also adds a new
  requirement: both mechanisms must be measured on DA1 and DM4 separately, because their
  thresholds differ 40x.
- **Weight calibration** attacks the geometry directly and is now supported by evidence for
  ignition specifically. Against it: the spec rejected per-connection fitting to calcium
  data as fitting rather than plasticity, and that objection is unchanged by P0. It also
  does nothing for sugar -> PAM, which is thin in route count, not miscalibrated.

There are three live shapes -- mechanisms as designed, calibration instead, or calibration
plus mechanisms -- and P0 does not pick between them. **That call is the human's to make
before P1 implementation starts.** This report does not make it.
