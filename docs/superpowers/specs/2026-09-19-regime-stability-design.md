# Regime stability: a brain that can hold a representation and still change

Date: 2026-09-19
Status: design approved; amended 2026-09-19 after the P0 gate fired (see "Re-scope" under P1)
Supersedes nothing. Depends on the song-conditioning design (shipped as condition v1; that design doc is not part of this release).

## The problem, stated properly

The whole-brain FlyWire v783 LIF has two failure modes that look different and are
probably one thing:

- **Runaway.** Drive any ORN channel past threshold and the brain goes to a single
  global attractor. On the corrected v783 matrix: KC 33% active, central 16%,
  inter-odour Jaccard 0.944. Odour identity is erased. Below threshold, nothing
  fires at all. There is no middle.
- **Silence.** Long, weak, multi-hop pathways deliver nothing. 150 Hz on 129 sugar
  GRNs leaves mean PAM at exactly 0.00 Hz. Audio never reaches Kenyon cells at any
  gain.

Both are consistent with one root cause: a uniform `W_SYN = 0.275 mV` for every
connection in the brain. Convergent paths explode; long thin paths die. That is a
hypothesis, not a finding, and P0 exists to test it.

## What this is NOT trying to do

The literature targets - KC 5-10% active, inter-odour Jaccard < 0.3 - are **symptoms,
not goals**. The underlying property is:

> a regime in which input-specific activity persists without collapsing into a global
> attractor, and which stays in that regime while its own synapses change.

Sparsity and Jaccard are how we read that property off. They are evidence, not the
pass line. A mechanism that hits 5% for the wrong reason has not succeeded.

The second clause is the one that matters for this project's actual goal. A brain that
is beautifully sparse and then ignites the moment plasticity switches on has solved
nothing. "Molds itself" requires that the regime survive its own learning.

Two clarifications, recorded because they were argued and settled:

1. **This is repair to fly parity, not aiming low.** Jaccard 0.944 means the model
   cannot discriminate two odours. Real flies discriminate odours easily. The model is
   currently *below* fly capability, not at it. Sparse high-dimensional coding is a
   capacity precondition - the same mechanism as dentate gyrus pattern separation and
   cerebellar granule cells - not a fly limitation.
2. **Intelligence is a different axis and is out of scope here.** The fly's real
   architectural poverty is the absence of recurrence, working memory and temporal
   credit assignment. That work (central complex, ring attractors) is blocked by this
   one, because none of it is testable on a brain that ignites. The fly connectome is
   also this project's credibility: every claim in `WHAT_IS_REAL.md` rests on it being
   a real fly. Fly parity is the floor; anything beyond it is a separate, separately
   labelled axis.

## Decomposition

Four pieces. This spec covers P0 and P1 only.

| piece | scope | gated on |
|---|---|---|
| **P0** | Diagnostic gate: reachability, weight distribution, regime characterisation | - |
| **P1** | The regime mechanism, validated on olfaction | P0 |
| P2 | Gustatory US: bitter -> PPL1, sugar -> PAM carry real signal | P1, P0 gate |
| P3 | Audio -> KC, retiring the authored fingerprint | P1, P0 gate |
| P4 | Plastic coverage beyond the current 0.7% of synapses | P1 |

P4 is on the track because it, not KC sparsity, is the real ceiling on self-molding:
18,674 of 2,700,429 edges are plastic, depression-only, one-directional, and saturating
on training trial 1. It comes after P1 because on an unstable regime you cannot tell
instability from learning.

## Evidence base

Measured, in `.claude/PRPs/reports/plasticity-exploration-2026-09-19.md`. These rule out
the two obvious mechanisms, which is why neither is proposed below.

**APL restoration alone is not enough**, and this is now doubly established.

The graft probe below was run on the superseded `brain_gpu_badbuild_2026-09-16.npz`,
where APL in-degree was 0 because the bad build had dropped every input onto it. It
grafted KC->APL synapses back:

| KC->APL synapses per KC | KC active | APL Hz | central | Jaccard |
|---|---|---|---|---|
| 0 | 0.856 | 0 | 0.274 | 0.996 |
| 1 | 0.665 | 200-220 | 0.244 | 0.975 |
| 5 | 0.564 | 303-327 | 0.229 | 0.954 |
| 10 | 0.530 | 337-360 | 0.223 | 0.932 |

APL saturates at its refractory ceiling (~450 Hz) and cannot deliver graded inhibition.
**Real APL is non-spiking.** Forcing it to spike is a model artifact, and it is the
clearest single biological mismatch in the sim.

The corrected v783 matrix then settled the question independently. It restores APL's
inputs for real - in-degree 2,843 / 2,818, not 0 - so no graft is needed. Ignition
persists anyway: KC 33% active, central 16%, Jaccard 0.944 at ORN 60 Hz. A fully and
genuinely connected APL does not sparsify this network. That is why P1.1 changes APL's
*neuron model* rather than its connectivity, and why it is not proposed alone.

**Global weight scaling is not enough.** KC active 0.86 at 1.0x, 0.60 at 0.5x, 0.40 at
0.4x, collapse to 0 at 0.3x. Jaccard stays 0.96-0.99 at *every* scale. No sparse,
odour-specific regime exists anywhere on that axis.

**Ignition is brain-wide**, not MB-local: central stays 22-27% regardless of what is
done to APL. Any mechanism confined to the mushroom body is therefore insufficient.

## P0 - Diagnostic gate

Read-only. No changes to `gpu_sim.py` or `learn/`. Output is a report plus a JSON of
the measurements.

### P0.1 Reachability audit

For each of JO -> KC, sugar GRN -> PAM, bitter GRN -> PPL1: hop count of the shortest
path, number of edge-disjoint paths, cumulative signed weight along the best paths, and
the bottleneck neurons (those whose removal disconnects the pair). Computed on the CSR
matrix directly.

**Gate.** If JO -> KC has no path, P3 is cancelled: the authored fly-hash fingerprint is
permanent and is disclosed as permanent. If sugar GRN -> PAM has no path, the same
applies to P2 and the direct-DAN US injection becomes permanent.

### P0.2 Weight-distribution diagnostic

Distribution of summed excitatory input weight per neuron, and of in-degree. Locate the
igniting population within that distribution. The specific question: does a uniform
`W_SYN` produce a bimodal explode-or-die structure, with the igniting neurons sitting in
a high-convergence tail and the dead pathways in a low-convergence one?

This is approach C's *measurement*, deliberately not its fit. Fitting per-connection
weights to calcium data was rejected for learning (it is fitting, not plasticity); the
measurement is still the fastest way to learn whether P1's mechanisms can possibly be
enough.

**Consequence.** If ignition is mostly a convergence artifact of uniform `W_SYN`, then
weight calibration stops being a diagnostic and becomes the mechanism, and P1 is
re-scoped before implementation rather than after.

### P0.3 Regime characterisation

Where the ORN drive transition sits, how sharp it is, and whether it is true bistability
or a steep sigmoid. Sweep drive rate finely around the transition on at least two ORN
channels.

## P1 - The regime mechanism

Three mechanisms, all defensible without external data. P1.1 and P1.2 are gain control
and were specced before P0 ran. P1.3 was added by the re-scope below.

### Re-scope, 2026-09-19: the P0.2 gate fired

It fired with a split verdict (`docs/superpowers/regime-p0/findings.md`).

**Explode: supported.** Under the single uniform `W_SYN = 0.275 mV`, summed excitatory
input per neuron runs median 50, p95 565, p99 1761 - a 35x median-to-p99 spread - with
APL at 62,443 on in-degree 2,831. Ignition is a convergence artifact.

**Die: refuted.** Both dead pathways' first-hop targets sit near the TOP of that same
distribution: sugar GRN targets at percentile rank 95.4, JO targets at 83.2. They are
not dying for want of convergence, so they are not an argument for anything in P1 and
get their own explanation in P2/P3.

The clause said calibration would become the mechanism if this happened. It does not.
Calibration against published physiology stays rejected: it is fitting rather than
plasticity, it needs external reference data, and the half of the hypothesis it would
have answered is the half that failed. What is added instead is **P1.3**, homeostatic
normalisation of incoming excitatory weight, which flattens the heavy tail directly and
needs no reference data at all. P1.1 and P1.2 are unchanged and are NOT replaced: P1.3
is a third measurement arm, and each mechanism is still measured alone before any
combination.

**Normalisation is not the global scaling already ruled out.** The evidence base above
records KC active 0.86 at 1.0x, 0.40 at 0.4x, collapse at 0.3x, Jaccard 0.96-0.99 at
every scale. That axis multiplies every weight by one constant, so it cannot change any
neuron's input RELATIVE to any other's: a heterogeneity problem is invariant under it by
construction. The supported half of the hypothesis is specifically about heterogeneity -
35x between median and tail under one uniform weight - so normalisation is untested, not
refuted. Scaling moves the whole distribution; normalisation changes its shape.

### P1.1 Graded APL

Replace the spiking APL units with a continuous rate unit: inhibitory output is a
function of weighted input, injected each step, with no spike threshold and no
refractory ceiling. This matches the biology (APL is non-spiking) and removes the exact
failure the graft probe exposed.

### P1.2 Spike-frequency adaptation

One adaptation state variable per neuron (139,248 float32), decaying with `tau_a`,
incremented on each spike, subtracted from membrane drive. Brain-wide, because ignition
is brain-wide.

### P1.3 Homeostatic normalisation of incoming excitatory weight

Scale each neuron's incoming EXCITATORY edges so that its summed incoming excitatory
weight meets one authored target; inhibitory edges untouched. This is synaptic scaling -
real, documented homeostasis - not a fit to anyone's recordings, and it is the only one
of the three mechanisms that acts on the weight geometry P0.2 measured rather than on
its symptom.

The target is AUTHORED and disclosed as such, in the same register as `TAU_A`,
`SFA_B_INC` and `APL_SCALE`. The natural first value is the brain-wide median,
50 synapses * 0.275 mV = 13.75 mV, which leaves the median neuron untouched and moves
only the tail.

One interaction, named here because it fails silently: `learn/plastic.py`'s
`Plastic.push(sim)` rewrites its 18,674 KC->MBON offsets in `sim.data` from `w0 * W_SYN`,
which would revert exactly those edges to unnormalised values after every training
trial, corrupting criterion 2. The plan's Task 5b resolves it in `Plastic.signed_values`,
the single point where `w()` becomes a `sim.data` value, and tests for the regression.

### Data flow

CSR delivery is unchanged. Two new per-step vector operations over existing state
arrays, plus P1.3's one-off rescale of the weight tensor's values at construction. No
change to the weight tensor layout, so `learn/plastic.py` offsets and `set_plastic`
continue to work untouched - but the values written through them are affected, which is
the interaction P1.3 names.

### Acceptance

1. **Stable input-specific regime.** Two ORN channels driven separately must produce KC
   populations whose Jaccard is materially below the current 0.944, and a KC active
   fraction materially below the current 0.33, with neither channel silent. Sparsity
   and Jaccard are reported against the fly reference (5-10%, < 0.3) as the benchmark
   for how far the mechanism got. Evidence, not pass line: a mechanism that reaches 5%
   by silencing the network has failed criterion 1, not passed it.
2. **Survives its own learning.** Run the existing conditioning protocol in the new
   regime. Concretely: `central_active` stays below **0.10** on every trial of the full
   training run - the same ignition bound already asserted throughout
   `learn/condition.py` - and the lesion arm's drift stays within baseline SD. This
   reuses the acceptance machinery that already exists.
3. **Determinism and batch-invariance preserved.** The checks that survive the
   connectome correction (`verify_gpu.py` checks 2-3): identical output for identical
   seeds, and per-item output independent of batch composition.

### Testing

House style - plain assert scripts, no framework:

- graded-unit maths on a toy network (monotone, bounded, no threshold)
- adaptation decay and accumulation on a toy spike train
- normalisation: the scale hits the target, leaves inhibition and zero-input neurons
  alone, and survives a `Plastic.push`
- a measurement script producing the criterion-1 and criterion-2 evidence

### Archival

New default, old brain archived. Specifically: a second correction section in
`WHAT_IS_REAL.md` in the same form as the 2026-09-19 connectome correction; prior
results, including condition v1 and reels 1-3, marked superseded and kept, not re-run.

## Risks

- **A+B may not close the gap. THIS RISK FIRED, and is resolved.** The graft probe
  reached KC 53% with a perfect APL, against a 5-10% benchmark. P0.2 was the early
  warning and it rang: convergence is the story for ignition (35x median-to-p99 spread),
  though not for the dead pathways. Resolved before building, as the clause required, by
  the re-scope above: A and B are kept, P1.3 normalisation is added as a third
  mechanism and a third measurement arm, and calibration against published physiology
  stays rejected. The gap may still not close; the plan's Task 6 names "no arm moves
  Jaccard below ~0.8" as the stop-and-report outcome rather than a tuning target.
- **`tau_a` has no fly-specific grounding.** It will be tuned against criterion 1 with
  no external reference. Disclose it as authored, in the same register as `eta`,
  `W_MIN` and the compartment table. The same applies to P1.3's normalisation target:
  13.75 mV is the measured brain-wide median, not a published number.
- **Criterion 1 has no hard pass line, by design.** That is the right call - the fly
  numbers are a benchmark, not the property - but it means the plan must state, before
  any run, what result would count as failure. Otherwise "materially below 0.944" gets
  decided after seeing the number.
- **Three mechanisms at once confounds attribution.** The plan must measure each alone
  before measuring them together, or a passing result will not say which one mattered.
  That is five arms - off, sfa, apl, norm, both - and an `all` arm only if something
  passes.
- **Normalisation could flatten the code as well as the tail.** Equalising summed
  excitatory input removes a real property of the connectome: some neurons genuinely
  integrate more than others. If the `norm` arm reaches sparsity by making every neuron
  identical, criterion 1's Jaccard is the check that catches it - the same check that
  catches silencing.
