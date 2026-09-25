# Pre-registration: does board fidelity buy playing strength?

Written 2026-09-16 09:40, before any shard of the v2/GPU sets was trained on.
The point of writing it now is that every number below is falsifiable by a run
that finishes in about two hours, and I will not be able to claim afterwards
that I expected whatever comes out.

## What is already measured

Board reconstruction (occupied-F1, piece-square occupancy, 5,000 identical sims,
equal 8,865 width). `readout.py` reproduces the analysis agent's numbers
independently to within 0.002:

| readout | all live | driven excluded |
|---|---|---|
| mean rate per `cell_type` (what the 200k run used) | 0.608 | 0.606 |
| live-weighted `cell_type` x spatial bins (`named`) | 0.866 | 0.746 |
| sparse random projection | 0.856 | 0.804 |

Playing strength with the 0.608 readout: `reservoir_only` 0.091 val top-1
(majority 0.035, material-only 0.047), Elo-ladder score 0.458 at its best
checkpoint against 0.500 for the untrained head. Fly-only plays no better than
random.

## The prediction

**Nothing measured so far predicts that the named readout moves top-1.** Board
reconstruction and next-move prediction are different tasks; a readout can carry
the position and still fail to expose it in a form a 3-layer MLP can use. The
honest prior is that this could go any of three ways.

Interpretation fixed in advance:

| `reservoir_only` top-1 on `named_nodriven` | reading |
|---|---|
| >= 0.15 | the readout was the bottleneck. Worth a ladder run and a real claim. |
| 0.10 - 0.15 | real but small. Report the delta; do not call the fly useful. |
| <= 0.10 (i.e. no better than 0.091) | **fidelity does not buy strength.** The 0.746 F1 is then a fact about decodability, not about chess, and the project's claim stays exactly where WHAT_IS_REAL.md already puts it: the fly reacts and is measured, the trained layer decides. |

The third outcome is the one I consider most likely, and it is not a failure of
the experiment - it is the experiment working.

## Controls that run alongside, and why

1. `reservoir_only --view pooled` on the **GPU** shards. Same readout as the
   0.091 run, different engine. If this does not land near 0.091, the engine is
   a confound and every cross-set comparison below is void. This is the control
   that makes the rest interpretable.
2. `named` vs `named_nodriven`. The gap between them is the size of the
   circularity: flypoke turns a stimulated neuron into a pure Poisson source and
   drops its network input, so 13,686 driven neurons read back our own drive.
   Only `named_nodriven` supports a claim about the fly.
3. `full --view named_nodriven`. The 0.608 readout made `full` (0.169) *worse*
   than board-only `ablation` (0.269). If the named readout does not close that
   gap, the fly channel is still actively harmful to a head that has the board,
   and that is the headline regardless of what `reservoir_only` does.

`ablation` is not re-run: it never touches the reservoir, so it is
engine-independent and readout-independent at 0.269.

## What would make me discard the whole comparison

- Control 1 lands outside 0.08-0.10.
- The GPU and CPU sets disagree on `pooled` `reservoir_only` top-1 by more than
  0.01 once both exist (~16:20).
- `named` beats `named_nodriven` by more than the ~0.12 F1 gap would suggest,
  which would mean the driven neurons are carrying more than the drive.

---

# Outcome, written 2026-09-16 the same day

## What the pre-registered threshold said, and what happened

`reservoir_only` on `named_nodriven` reached **0.154** top-1 (0.152 before a
standardisation fix, 0.154 after). The table above says >= 0.15 means "the
readout was the bottleneck. Worth a ladder run and a real claim."

Half of that is right. The readout *was* the bottleneck for move prediction:
0.089 -> 0.154 with nothing changed but which neurons share a dimension, and the
mechanism is now measured rather than argued - the shipped `pooled` readout has
5,513 of 8,865 feature columns with zero variance across 200,019 positions, so
its effective width was 3,352, while the named readout wastes none.

The other half is wrong, and the ladder is what says so:

| checkpoint | record vs random | score |
|---|---|---|
| epoch 0, untrained | 7-85-8 | 0.495 |
| epoch 3, best top-1 (0.154) | 3-76-21 | **0.410** |

**The trained head plays worse than the untrained one.** It wins 3 games and
loses 21 to an opponent choosing uniformly among legal moves. v1 showed the same
shape with the old readout (0.500 untrained, 0.458 best, 0.283 overfit), so this
is the second independent observation of it, and the better readout did not
improve it.

## The methodological error, stated plainly

I pre-registered a threshold on the wrong quantity. Top-1 agreement with the
human move is a proxy for playing strength, and this experiment is a clean
demonstration that the proxy does not carry: a 69% relative gain in move
prediction produced no gain in strength, and possibly a loss.

The mechanism is ordinary once seen. An untrained head spreads probability over
legal moves, shuffles, and draws by repetition. A trained head confidently plays
what *looks* like a human move with no tactical evaluation behind it, hangs a
piece, and a random opponent eventually takes it. Predicting the move a human
played and not losing material are different tasks. The fly vector supports the
first weakly and the second not at all.

If this experiment is ever repeated, the pre-registered threshold belongs on the
ladder score, not on top-1. Top-1 is the cheap screen, not the claim.

## What the project's claim is, unchanged

Exactly what WHAT_IS_REAL.md already said before today: **the fly brain reacts
and is measured; the trained layer decides.** Nothing measured today licenses
"fly wiring helps chess", and `full` at 0.174 against board-planes-only at 0.265
continues to say the fly channel actively costs a head that can already see the
board.

## Discard conditions: none triggered

- Control 1 required `pooled` on GPU shards within 0.08-0.10: **0.089**. Passed.
  Two independently written integrators agree on downstream top-1 to 0.0015.
- `named` (0.230) minus `named_nodriven` (0.154) = 0.076, close to the 0.12 F1
  gap and in the expected direction, so the driven neurons carry about what
  reading back our own injected drive should carry. No anomaly.
- The CPU v2 set was stopped at 102/201 to free cores for the corrected ladder.
  Its purpose - a third engine cross-check - was already served by the control
  above, and the shards on disk are resumable.
