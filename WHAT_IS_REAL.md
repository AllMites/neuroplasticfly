# What is real here

A fly brain plays chess in this repo. That sentence is doing a lot of work, so
here is exactly how much of it is true, in four buckets: **Real**, **Authored**,
**Trained**, **Absent**.

If you only read one line: *the wiring is real and it is genuinely in the loop of
every move, but the wiring learned nothing about chess and the decision is made
by a small trained layer bolted on top of it.*

---

## Real

Measured, not written by me.

- **The connectome.** FlyWire v783, 139,248 annotated neurons that carry edges in
  v783, signed synapse
  counts aggregated per neuron pair with a 5-synapse floor. Annotations from
  Schlegel et al. 2024 (flywire_annotations **v3.1.0**, commit 8587524c);
  connections from Dorkenwald et al. 2024 (Zenodo
  10676866, md5 f48f972d262323a102aed49af1396b8a). Nothing in the matrix is edited, pruned, or tuned. No gradient ever
  reaches it.
  Recertified 2026-10-01: all three Zenodo files hash-match the certified release
  byte-for-byte, and 2,700,429 reproduces from them alone (floor-5 pairs are
  2,700,513 raw; the index rule drops 84 pairs touching the 14 neurons below).
  **139,248 is our index, not FlyWire's proofread count — the certified
  `proofread_root_ids_783.npy` holds 139,255.** 14 of those ids appear in the
  connectivity file and are outside our index because the annotation release has
  no row for them (one, 720575940633242449, has 51 partners at floor 5); 7
  annotation rows have no edges in v783 at all. Both gaps are a property of the
  published files, not of the build. See
  `Projects/Fly-Connectome/connectome-recertification-2026-10-01.md`.

- **The dynamics.** Leaky integrate-and-fire with the constants from Shiu et al.
  2024 (*Nature* 634:210-219): `w_syn` 0.275 mV, `v_th` -45 mV, `tau_m` 20 ms,
  `tau_syn` 5 ms, `t_refr` 2.2 ms, 1.8 ms synaptic delay, 0.1 ms steps. GABA and
  glutamate inhibitory, everything else excitatory.
- **One 300 ms whole-brain simulation per move.** Not a cached lookup, not an
  approximation of a simulation. Every move the bot plays costs ~0.9 s of CPU
  because the whole brain is actually integrated.
- **The firing rates.** `results/timing.json` records the measurement: at the
  start position the drive lights **0.172** of the central brain above 1 Hz. The
  **28.0%** here before it was measured on the superseded 2026-09-16 build - see
  *Correction, 2026-09-19* below.
- **The descending-neuron readouts** in every recorded move (`dn_table`): DNg29,
  DNg84, DNp01, DNp04, DNa01, DNa02, DNa03, DNp09 per side, plus the ingestion
  motor neurons. These are measured rates. Spec 2 (the body) consumes them
  directly.
- **The sugar and bitter neurons** used for reward visualisation are the real
  gustatory populations (`cell_sub_class=sugar/water`, 129 neurons;
  `cell_sub_class=bitter`, 65 neurons), and the "fed" reading is the measured
  ingestion-motor-neuron rate that follows.

## Correction, 2026-09-19: the connectome file was a bad build

Every number in this file dated before 2026-09-19, and reels 1-3, ran on
`data/brain_gpu_badbuild_2026-09-16.npz`. Checked against the raw FlyWire v783
release (Zenodo 10676866) and Shiu et al.'s own connectivity table, that file had
3,532,411 neuron pairs where the 5-synapse floor gives 2,700,429; its weights were
25% too high on median; it kept 854,098 pairs below the floor; and it dropped every
input onto the two APL neurons and several other giant cells (22,638 edges). The
cause sits in the Docker volume that built it and is not resolved. Shiu et al. 2024
applied no synapse threshold at all; "as in Shiu" above was wrong.

`data/brain_gpu.npz` is now built on the host from the Zenodo file (loader logic
identical to flypoke's). The central-brain activation figure and every result in
`results/` marked `_v1` or later use it. Older results are kept, not re-run.

## Authored

Chosen by me. Tuning a piano is not playing it.

Every module-level constant in `encode.py`:

| Constant | Value | What it decides |
|---|---|---|
| `MAX_HZ` | 150.0 | ceiling on any Poisson drive |
| `PIECE_HZ` | P 60, N 80, B 80, R 100, Q 130, K 150 | how loudly each piece shouts at the retina |
| `PIECE_VALUE` | 1/3/3/5/9/0 | material, for the taste and smell channels |
| `COUNT_HZ` | 40.0 | Hz per piece in the smell piece-count channels |
| `MATERIAL_HZ` | 30.0 | Hz per pawn of material difference |
| `HANGING_HZ` | 50.0 | Hz per hanging own piece, on the grooming mechanosensors |
| `BIT_HZ` | 120.0 | Hz for a set boolean smell bit |
| `CHECK_HZ` | 150.0 | Hz on Johnston's organ when in check |
| `DRIVE_GAIN` | 1.0 | global multiplier on every rate above; 1.0 is what the 200k precompute ran at. Lower values probe whether a weaker drive leaves the chaotic regime — see *Why: the pooled vector destroys board-space locality* |
| `GRID` | 8 | retinal patches per axis per eye |
| `T45` | T4a-d, T5a-d | which neurons are the "retina" |
| `SUGAR` / `BITTER` | `cell_sub_class=sugar/water` / `bitter` | the taste channel |
| `GROOMING` / `JOHNSTON` | `cell_sub_class=grooming` / `cell_type=JO-B1_a\|JO-FV` | the touch channel |
| `ORN_CHANNELS` | 40 named channels | which olfactory class means which chess fact |
| `CACHE` | `data/visual_map.npz` | where the retinal map is frozen once built; delete it and the 8x8 patches are rebuilt, which invalidates every existing checkpoint |

Also authored:

- **Own pieces go to the left eye, the opponent's to the right.** A fly has no
  such convention. It is a wiring decision that makes the two sides separable.
- **T4/T5 rather than photoreceptors.** Measured, not aesthetic: R1-6 drive never
  leaves the lamina in this model, so a photoreceptor "retina" would have been a
  dead channel. T4/T5 at 150 Hz reaches 4.3% of the central brain and 14% of
  descending neurons.
- **The 8x8 retinal patches** come from a PCA of T4/T5 soma positions split by
  nested quantiles. The fly's real retinotopy is a hexagonal lattice; this is a
  square grid imposed on it because a chessboard is square.
- **300 ms and one trial.** Long enough to propagate, short enough to play at.
  One trial means single-move reservoir vectors carry Poisson noise; see
  "measured honestly" below.
- **The reward thresholds** in `play.py`: a move losing under 50 centipawns
  (Stockfish skill 20, 50 ms) earns sugar, otherwise bitter.
- **The Elo anchors** in `elo.py`: random mover = 400, Stockfish skill 0 = 1350.
- **Everything in `search.py`.** The depth (2 plies), the evaluation (material
  from the same `PIECE_VALUE` table `encode.py` uses, scaled to centipawns),
  `MATE` = 100000 cp for a mate score, and `MOBILITY` = 1 cp per own legal
  reply. The mobility term is a tie-break, not chess knowledge: with material
  alone every quiet move ties, the search shuffles when ahead, and won games hit
  the 300-ply cap and score as draws. `OUT` there is the `$FLYCHESS_OUT` write
  guard, not a tuning knob. Phase-1 numbers are the search alone, and they
  are the baseline the fly is measured against, not a result about the fly.
  Anything `elo.py` writes is labelled `est_elo` and is an estimate against those
  anchors, not a rating.
- **The pruning arms in `search.py`.** `ARMS` names the four priors - `none`,
  `fly`, `random`, `planes` - and `CKPT` is an authored mapping from the two
  head arms to the checkpoints they load: the fly arm is the `reservoir_only`
  head behind the 0.410-vs-random row, the planes arm is the `ablation` head.
  `k`, the number of candidates a prior keeps, is authored too. What this buys
  the fly is exact and narrow: **the fly chooses which moves are considered; the
  search chooses among them.** At `k` >= the legal move count the prior is
  skipped entirely and the fly has chosen nothing, which is why `prune=none`
  records `k` as null rather than a number. The search itself - `evaluate`,
  `negamax`, `search` - is unchanged by every arm, so any difference between
  arms is a difference in the prior and nothing else. The `random` arm is a
  seeded sample, not a weaker model: it is the coin flip the fly has to beat.

## Trained

- **18,674 KC->MBON synapses, and only those**, via dopamine-gated depression
  (`learn/plastic.py`). No gradient. The song reaches the Kenyon cells through an
  authored fly-hash fingerprint (`learn/fingerprint.py`), not the fly's ear.
  Brain state (the synapse deltas) lives in `brain_state/<name>/` and carries
  over between reels.
- **What the conditioning did (condition v1).** Acceptance check: `learn: misery +6.077
  skyhigh -5.354 (baseline sd 0.939); lesion: misery +0.437` -> ACCEPT. The caveat on that
  result: the two songs move `avoid_index` by *different* mechanisms, not one. Misery is
  near-total silencing of the approach-MBON population - 6.64 -> 0.34 Hz, about 95%, with
  its avoid-MBONs starting at 0.22 Hz and ending at 0.00. Sky High is majority avoid-side -
  avoid-MBONs 3.04 -> 0.07 Hz, the depression the rule was designed to produce - while its
  approach-MBONs *rose*, 6.87 -> 9.26 Hz, which depression-only does not predict. An earlier
  version of this caveat quoted 0.185 Hz against 26.4 Hz as the baseline; those came from a
  smoke trial with direct DAN drive running, not from the protocol baseline. Avoidance drive
  went up for neither song, so **"the fly learned to avoid Misery" is not supportable.**
- **It generalised broadly, not narrowly.** Mozart was probed and never trained. Its KC
  fingerprint overlaps Misery's at Jaccard 0.39, and it shifted **+5.568** - larger in
  magnitude than trained Sky High's -5.354. An untrained song moved as much as a trained
  one, so no claim of song-specific learning is available either.
- **The weights saturate on the first training trial.** `w_min_frac` sits at its
  floor-plus-recovery value of -0.891 from training trial 1 onward, and `w_mean_frac` is at
  asymptote by trial 2 of 30. The 30-trial curve is a step, not a learning curve.
- **Punishment and reward were injected straight into the dopaminergic
  neurons** (PAM and PPL1), not tasted. The fly's real gustatory pathway does
  not reach them in this model. A later 10-seed sweep of that check says the
  reward arm is the solid half: 150 Hz on the 129 sugar GRNs leaves mean PAM
  indistinguishable from baseline, 0/10 seeds over threshold. The punish arm is
  seed-dependent: 150 Hz on the 65 bitter GRNs does weakly reach PPL1, about
  1 Hz on average but with a range of 0.28 to 1.94 Hz across seeds, 4/10 over
  threshold. The rule needs both arms, so the fallback fires 10/10 regardless.
  The choice was made by that measurement at run time and is recorded as
  `us_path` in `brain_state/<name>/provenance.json`.
- **One MLP head**, and nothing else: `Linear(d, 2048) - GELU - Dropout(0.2) -
  Linear(2048, 1024) - GELU - Linear(1024, 1858)`. Its input is the 1152 board
  planes concatenated with the standardised `log1p` of the ~8,865-dimensional
  pooled rate vector.
- **Data**: ~200k positions sampled from Lichess rated games between 1500 and
  2000 Elo, base time 300 s or longer, normal termination, sampled after ply 8.
  Cross-entropy against the human move.
- **Phase B**: REINFORCE against Stockfish skill 0-3, win/loss reward, reservoir
  frozen throughout.
- **The ablation is mandatory.** An identical head is trained on the board planes
  alone. The call was to be made from the numbers once they existed, not before.
  They exist now — see *Measured: does the fly help?* below. The answer is no.

## Absent

Things people will assume and that are not true.

- **The fly did not learn chess.** No synapse changed. It cannot improve.
- **The fly does not know it is playing chess.** It is being poked with patterns
  derived from a board, the way flypet pokes it with sugar.
- **The reward in `play.py --reward-viz` teaches nothing.** It is shown after the
  move is already chosen. It is a real measurement of a real circuit reacting to
  real sugar neurons, played as a visualisation. `rl.py` is where reward actually
  moves weights, and it moves only the head's.
- **No search.** No MCTS, no tree, no lookahead. One forward pass per move.
- **This is not the first chess-playing fly.** flychess-hq (Ernesto Lopez) and
  lichess FlyBrainChess got there first, on MaleCNS. What is new here is per-move
  whole-brain activity at neuron resolution, a verified rating, and this file.

## Measured: does the fly help?

Measured 2026-09-16 on the full 200,019-position set, 20 epochs, identical head
and hyperparameters for every row. Metric is validation top-1 agreement with the
human move over a held-out 10%.

| Input to the head | dims | val top-1 |
|---|---|---|
| always predict the single most common move | — | 0.035 |
| 12 piece counts + material + move number | 14 | 0.047 |
| **pooled fly vector alone** (`reservoir_only`), no board | 8,865 | **0.091** |
| **board planes alone** (ablation) | 1,152 | **0.269** |
| board planes + pooled fly vector (full) | 10,017 | **0.169** |
| board planes + fly through a 64-dim bottleneck | 1,216 | 0.266 |

Two things follow, and they point in opposite directions.

**The fly does carry board information.** 0.091 against a 0.035 baseline is
about 2.6x, from a vector that never sees the board except through Poisson drive
on sensory neurons. That is a real measurement, not an artefact: the shard loader
was verified byte-identical to a naive reference implementation, and the row
index is the identity permutation over all 200,019 rows.

**The fly does not help a head that already has the board.** Adding it *lowers*
top-1 by 10 points, at every epoch, not merely at convergence. Narrowing the fly
channel monotonically improves the result — 3,360 dims gives 0.169, 256 gives
0.255, 64 gives 0.266 — and extrapolates to the ablation number at width zero.
The head does best when it can ignore the fly almost entirely.

So the honest claim is the weaker one: *the fly brain reacts, is measured, and
carries some board information on its own; the trained layer is what decides.*
Not *"fly wiring helps chess"*. Anyone quoting the 0.091 must also quote the
0.269, because the interesting-sounding number and the damaging one come from
the same experiment.

Of the 8,865 pooling groups, **5,502 are exactly constant** across all 200k
positions — 62% of the fly's own pooled vector never moves in response to a
chessboard at all. Those dims are inert after standardisation (they become zero
columns), so they are not the cause of the damage; the 3,363 live ones are.

### Does the fly-only head actually play chess?

No. Measured 2026-09-16, 60 games per block at temperature 0.5 against a uniform
random mover, `results/elo.json`:

| head | checkpoint | W-D-L | score |
|---|---|---|---|
| `reservoir_only` | epoch 0 (untrained) | 2-56-2 | 0.500 |
| `reservoir_only` | epoch 6 (best val) | 4-47-9 | **0.458** |
| `reservoir_only` | epoch 20 (overfit) | 2-30-28 | 0.283 |
| `ablation` | best | 15-44-1 | **0.617** |

The board-planes head beats a random mover. The fly-only head does not: at its
best checkpoint it is within noise of 0.5 over 60 games, and training it further
makes it clearly worse. Plan 1c fixed the reading of this before the numbers
existed — *`reservoir_only` ~= random* means the honest claim is only that **the
fly brain reacts and is measured; the trained layer decides.** "Fly wiring alone
plays chess" is not available.

This is not for want of competence on individual positions. On held-out
*dataset* positions the fly head hangs material on 1.8% of moves against a random
mover's 9.7%, and answers a threat 82.5% of the time against random's 28.1%. It
looks capable right up until it has to play a game, because a game against a
random mover leaves the human-game distribution within a few moves.

### The ceiling is the representation, not the head or the data

| n_train | `ablation` | `reservoir_only` |
|---|---|---|
| 11,251 | 0.0695 | 0.0555 |
| 45,004 | 0.1723 | 0.0680 |
| 180,018 | 0.2600 | 0.0938 |

`ablation` gains ~0.045 top-1 per doubling of data and is still climbing — it is
underfit and data-starved, and more Lichess games would buy it real strength.
`reservoir_only` gains ~0.010 per doubling from the identical head on the
identical rows. Capacity and regularisation are ruled out separately: over
hidden sizes 32 to 2048, dropout 0.2 to 0.6 and weight decay 0.01 to 1.0, val
top-1 moves between 0.063 and 0.094, and *shrinking* the head makes it worse.

So the overfitting is a symptom, not the disease. The ceiling is ~0.094 and
early stopping is the only thing regularisation buys. Which is the locality
result restated in loss terms: with no smooth structure to generalise along,
memorising is the only fit available, and memorising does not transfer.

### Why: the board reaches the head with a quarter of it wrong

Three theories were tested and the first two were wrong. Recording all three,
because the wrong ones are the ones a reader would otherwise repeat.

**Wrong theory 1: seed noise swamps the signal.** Refuted by direct measurement
(`snr_trials.py`, 120 positions x 8 seeds). Variance SNR at a *single* trial is
**4.31** — signal already beats seed noise fourfold — and trial averaging tracks
the predicted k x SNR line exactly (8.70 / 17.21 / 34.48 at k = 2 / 4 / 8). More
trials buy nothing. The earlier "the reservoir is noisy at 1 trial" framing came
from comparing mean-absolute differences as if they were RMS, which they are not.

**Wrong theory 2: it only resolves coarse position categories.** Refuted:
one-ply-apart boards separate at SNR **3.94** against **4.22** for random
positions — a ratio of 1.1x. The reservoir distinguishes two boards one move
apart nearly as strongly as two unrelated ones.

**What the evidence actually supports:** that last number is the problem, not a
success. A useful feature map puts similar inputs near each other. Over 4,000
random position pairs, the correlation between board-space distance and
reservoir-space distance is **r = 0.08**, and the most similar 2% of board pairs
sit at 0.76x the reservoir distance of median pairs — barely closer at all. The
map is very nearly distance-destroying: a chaotic hash of the board rather than a
representation of it.

**That story is descriptive, not causal, and the difference matters.** Two
controls cut it down. First, standardising the *board planes* — a representation
known to work, 0.269 top-1 — drops their own locality from r=1.00 to r=0.42 and
their near/median ratio from 0.300 to 0.550, so some of the measured collapse is
the preprocessing rather than the representation. Second, and decisively: raw Hz
has ~45% better locality than the shipped `log1p`+standardise recipe (r 0.214 vs
0.147, near/median 0.621 vs 0.813) and trains to *identical* accuracy — 0.0915
against 0.0919. A large locality gain bought nothing. Locality is real and
measurable here, but it does not predict what the head can learn, and no claim
should rest on it.

**What does explain the ceiling: the board arrives damaged.** A decoder trained
on the pooled vector reconstructs which piece stands on which square at
**occupied-F1 0.745**, against 0.0 for an always-empty baseline. So the board
largely survives the fly — but roughly a quarter of the piece-squares are wrong.

> **Circularity warning — read before quoting any reconstruction number.**
> flypoke turns a stimulated neuron into a pure Poisson source and drops its
> network input (`encode.stimuli` says so in its own docstring). A driven
> neuron's firing rate therefore *is* the drive we imposed, with Poisson noise
> on top. 13,686 of the 27,857 responsive neurons are driven this way, so a
> reconstruction that includes them is partly measuring our own encoder being
> read back, not the fly carrying anything.
>
> Excluding every drivable neuron, so that only the fly's own response counts:
>
> | readout, driven neurons excluded | occupied-F1 |
> |---|---|
> | mean rate per `cell_type` (ships) | 0.606 |
> | named, live-weighted bins | 0.746 |
> | sparse random projection | 0.804 |
>
> These are the numbers any claim about *the fly* must use. The higher figures
> elsewhere in this file (0.745, 0.852, 0.866) include driven neurons and are
> the right numbers only for the engineering question "how well can the head
> read this vector", never for "what does the fly compute".
A position with eight pieces misplaced cannot be played, however good the head
is, and that reconciles every other measurement in this file: sound-looking
moves on dataset positions, no playing strength over the board.

**Wrong theory 3: the drive is too strong, pushing the network past the edge of
chaos.** Refuted by `drive_sweep.py`. Lowering the global drive makes locality
*worse*, monotonically — near/median distance ratio 0.758 / 0.797 / 0.853 / 0.892
at `DRIVE_GAIN` 1.0 / 0.5 / 0.25 / 0.1. The shipped drive is the most local of
the four. Weaker drive simply means fewer spikes and noisier rates.

**Theory 4: pooling by `cell_type` averages the retinotopy away, leaving only
material. Half right, and the half that is right is the important one.** The
"only material" part is refuted — a head on 12 piece counts plus material and
move number reaches **0.047** against the fly vector's **0.091**, so what
survives is about twice material. But the pooling *is* heavily lossy, which a
later measurement caught after this one had been written off. See *The readout,
not the fly* below.

### The readout, not the fly

`reservoir.pool` reduces 139,248 neurons to 8,865 numbers by averaging every
neuron of a cell type together. That is a readout choice made in this repo, not
anything the fly does. `readout_probe.py` compares three readouts of the *same*
5,000 simulations at the same width and the same decoder budget:

| readout of the same sims, 8,865 dims | occupied-F1 |
|---|---|
| mean rate per `cell_type` (**what ships**) | **0.610** |
| fixed sparse random projection | **0.852** |
| a random 8,865-neuron subset, no pooling at all | 0.850 |

A random projection recovers the board far better than the considered,
biologically-named grouping — and a random *subset* of neurons does just as well
on 1,780 live dimensions.

**The retinotopy explanation given here earlier was wrong.** It said a type like
T4a spans the whole retina, so averaging it destroys which patch fired. Tested
directly (`readout_analysis.py`): splitting every `cell_type` into *k* **spatial**
sub-groups performs the same as splitting it into *k* **random** sub-groups, at
every *k* and every width — and where the two differ at all, the random control
wins. Nothing specific to retinotopy is being lost. Nor is it a shortage of
dimensions: spatial splitting at 37,143 dims reaches 0.702, still losing to a
random projection at **256** dims (0.822).

**The measured mechanism is that dimensions are allocated by taxonomy while
signal is allocated by population size.** 64% of the 27,857 responsive neurons
sit in 20 cell types, which receive 20 of the 8,865 dimensions between them,
while 5,698 permanently silent types receive one dimension each. Pooling itself
is not the problem: averaging 1,024 neurons drawn from across the brain scores
0.844, while averaging 1,024 neurons of one cell type scores 0.610. Pooling a
large population is harmless; pooling a large *type* is not.

A readout that keeps the biological names but allocates bins in proportion to a
type's live count scores **0.866** — matching the random projection while
remaining fully nameable. So interpretability is not what costs the accuracy.

This is the cheapest possible place for the fault to be: no authored constant
changes, no encoder redesign, nothing about the connectome. It does require new
shards, because only the pooled vector was ever written to disk.

An earlier version of this file blamed the encoder instead, on the strength of
the stage table in `where_locality_dies.py` (drive r=0.019, rates r=0.227, pooled
r=0.274). That attribution used the locality metric, which the control above
shows does not predict usable information. The reconstruction metric says the
opposite. The stage numbers are left in the repo because they are real, but they
should not be used to locate the loss.

The honest status, after ten measurements: **the board survives the fly, and
then most of it is thrown away by our own readout.** Six explanations were
tested and refuted along the way — seed noise, coarse resolution, drive
amplitude, material-only collapse, head capacity, and preprocessing — and one,
the locality story, was promoted too early and then withdrawn. What is settled,
and what governs every claim made about this project, is that the fly as
currently read out does not improve a head that already sees the board.

## Measured honestly: the reservoir's own noise

A single 300 ms trial is noisy, and the pooled vector has a large common mode
that is nearly identical for every board. On the raw rate vector, two *different*
positions have cosine similarity ~0.984-0.9997 — but the same position simulated
with two different seeds sits at ~0.9997 too. A raw-cosine threshold therefore
cannot distinguish signal from noise, and the plan's original "cosine < 0.99"
acceptance criterion is unreachable by construction.

The check that replaced it measures both numbers in the space the head actually
consumes (standardised `log1p`) and requires different positions to be *less*
similar than the seed-noise floor. Measured: different positions **-0.166**,
seed-noise floor **+0.122**. Positions four plies apart are close to the noise
floor; structurally different positions separate strongly. Both numbers are in
`results/timing.json`.

## Measured: does a better readout help?

The shipped readout took the mean rate per `cell_type`. That allocates
dimensions by taxonomy while signal is allocated by population, and the cost is
visible directly in the feature matrix: **5,513 of its 8,865 columns have zero
variance across all 200,019 positions**, so its effective width was 3,352. The
replacement (`readout.py`: live-weighted `cell_type` x spatial bins) wastes none
of the 8,865.

Move prediction improves a lot. Playing strength does not improve at all.

| head | readout | val top-1 |
|---|---|---|
| `ablation` (board planes, no fly) | - | **0.265** |
| `reservoir_only` | `named`, driven included | 0.230 |
| `full` (planes + fly) | `named_nodriven` | 0.174 |
| `reservoir_only` | `named_nodriven` (honest) | **0.154** |
| `reservoir_only` | `pooled` (what shipped) | 0.089 |
| majority-class baseline | - | 0.035 |

Ladder, `named_nodriven`, 100 games per checkpoint against a uniform-random
mover (`runs/fix_nodriven/results/elo.json`):

| checkpoint | record | score |
|---|---|---|
| epoch 0, untrained | 7-85-8 | 0.495 |
| epoch 3, best val top-1 | 3-76-21 | **0.410** |
| epoch 20, overfit | 3-93-4 | 0.495 |

**The trained head plays worse than the untrained one**, winning 3 and losing 21
to random play. v1 saw the same shape with the old readout (0.500 / 0.458 /
0.283), so this is the second independent observation and the better readout did
not change it.

Why: an untrained head spreads probability over legal moves, shuffles, and draws
by repetition - 85 and 93 draws in the two flat rows above. A trained head
confidently plays what looks like a human move with no tactical evaluation
behind it, hangs a piece, and a random opponent eventually takes it. Predicting
the move a human played and not losing material are different tasks; the fly
vector supports the first weakly and the second not at all.

So a 69% relative gain in move prediction bought no playing strength. Top-1 is a
proxy for strength and this is a clean demonstration that the proxy does not
carry. `PREREGISTER_readout.md` records that the >= 0.15 threshold was committed
before any of these numbers existed, was cleared, and was placed on the wrong
metric.

The claim is therefore unchanged: **the fly brain reacts and is measured; the
trained layer decides.** And `full` at 0.174 against `ablation` at 0.265 still
says the fly channel costs a head that can already see the board.

### The bug this nearly hid

The first ladder on this readout returned 0-100-0 and 0-50-50: zero wins in 300
games, two checkpoints drawing every game. That was not a measurement. The named
readout bins 2-5 neurons per dimension, so some bins never move in training and
land at sd = 1e-6; `(x - mu) / 1e-6` then fed a head trained on z-scores a
feature value of **4,912,655**, 478 dims above |100|. It was isolated by
elimination - the GPU-trained `pooled` head was clean, which ruled out the
train-on-GPU/play-on-CPU mismatch, and v1's `pooled` checkpoints were clean,
which is why the v1 Elo numbers in this file stand. A column with no variance
over 200k positions is now zeroed in training and inference alike, and the rest
clipped to +-8; the norm block carries both so a checkpoint cannot be played
with different preprocessing than it was trained with.

A degenerate all-draws result scores 0.500 and passes a "did the untrained head
play like random" check. It looked like a finding. It was a division.

## Two engines

The model is unchanged; the integrator was reimplemented. `flypoke` (CPU, in the
container) remains the reference implementation and the oracle. `gpu_sim.py`
(host, CUDA) integrates the same equations with the same Shiu et al. 2024
constants over a batch of positions, at 29.4 sims/s against the CPU pool's 7.66.

They are not bit-identical: float32 reduction order in the synaptic scatter
differs, and production uses a counter-based Poisson stream instead of numpy's
PCG64. Measured on 64 positions (`results/gpu_verify.json`):

- given flypoke's *own* random stream - reproduced exactly, 2,567,198
  stimulated-neuron spikes identical on both engines - per-neuron rate agreement
  is **r = 0.9998**, mean 0.062 Hz.
- on the production stream, **r = 0.9916**, against **r = 0.9915** for two
  flypoke runs of the same position with different seeds. The engines differ by
  no more than the reference differs from itself.
- `pooled` r = 0.99963 (CPU vs itself: 0.99964); `live` r = 0.98842 (0.98859).
- central-brain live fraction 28.1% on both (superseded 2026-09-16 build;
  the corrected v783 build gives 0.172 - see *Correction, 2026-09-19*).
- flypet's five behaviour labels reproduce on the GPU engine.

`verify_gpu.py` enforces all of it and is the reason the claim can be made.
Every shard directory carries an `ENGINE.json` saying which engine wrote it:
`data/reservoir_v2` is CPU, `data/reservoir_gpu` is GPU. They are never mixed in
one directory, because mixed provenance is unrecoverable after the fact.

Re-verified on the shards actually being shipped, not only on the agent's test
run: `data/reservoir_gpu/shard_0000.npz` against flypoke's own counts for the
same rows gives `pooled` r = 0.99955 and `live` r = 0.98863, the latter above
the CPU-vs-CPU seed-noise floor of 0.98859.

The GPU engine is deliberately not wired into `play.py` or the lichess loop:
at batch size 1 it runs 0.35 sims/s, three times slower than a single CPU
worker.

## Destroyed on 2026-09-16

`reservoir_only`'s checkpoints `epoch_00`-`epoch_02`, its `best.pt`, and its
entire 20-epoch history in `results/train.json` were overwritten by a 1,000-row
smoke run of the named readout, because `train.py` wrote checkpoints and results
to hardcoded paths regardless of how small the run was. `epoch_03`-`epoch_20`
survived, so the Elo-measured checkpoints (6 and 20) are intact and `best.pt`
was restored from `epoch_06.pt`. The headline numbers quoted in this file -
0.091 best val top-1 at epoch 6 - were already written down here and are
unaffected; the per-epoch curve is gone and is recomputable from the surviving
checkpoints if it is ever needed. `results/train.json` carries a `RECOVERED`
note saying so rather than silently showing the smoke's 0.040. Output paths now
honour `$FLYCHESS_OUT`.

## Citations

- Shiu, P. K. et al. (2024). A *Drosophila* computational brain model reveals
  sensorimotor processing. *Nature* 634, 210-219.
- Schlegel, P. et al. (2024). Whole-brain annotation and multi-connectome cell
  typing of *Drosophila*. *Nature* 634, 139-152.
- Dorkenwald, S. et al. (2024). Neuronal wiring diagram of an adult brain.
  *Nature* 634, 124-138. Connections: Zenodo 10676866.
- ESA ACT (2025). The *Drosophila* connectome as a reservoir computer.
  PMC12109256 — peer-reviewed precedent for the method used here.
