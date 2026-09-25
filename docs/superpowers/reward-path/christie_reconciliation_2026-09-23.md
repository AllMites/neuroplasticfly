# Christie reconciliation: the sugar -> Fox -> FDA path conducts, and dies at the last hop onto PAM (2026-09-23)

Christie et al. 2026 (Curr Biol, PMC12869359) report that sugar GRN activation
recruits PAM-DANs through Fox (CB0525) and 11 ascending "FDA" pairs, in a
Shiu-2024 Brian2 LIF on the FlyWire connectome. flychess had measured PAM silent
under the same 150 Hz sugar drive (0.024 Hz vs 0.021 Hz baseline,
`dan_floor_2026-09-21.md`) and had recorded the reward path as CLOSED on a
synapse-fraction argument (`floor1_convergence_2026-09-21.md`).

Hypothesised: either flychess's matrix is missing Christie's path, or it has it
and the disagreement is at some identifiable hop. Happened: **the anatomy is
present and matches the paper to the synapse, hops 1 and 2 conduct, and the
chain dies at hop 3 - FDA-I -> PAM, 129 synapses, 0.81% of PAM's total input -
where not one of 307 PAM-DANs emitted a single spike in any of 80 runs.** The
CLOSED verdict survives its own falsifier, but the reason it survives is now a
measured hop, not a fraction.

Scripts: `learn/christie_sweep.py` (sweep), `regime/test_silence.py` (kwarg test).
Matrices: `data/brain_gpu.npz` (stock; NOT the danfloor patch).
Raw: `results/christie_sweep_v783stock.json`.
Supplement: `docs/superpowers/reward-path/christie_supp/` (paper text + Data S2-S6).

Command lines:

```
.venv/Scripts/python.exe regime/test_silence.py
.venv/Scripts/python.exe learn/christie_sweep.py --seeds 1 --tag smoke --hz 150
.venv/Scripts/python.exe learn/christie_sweep.py --seeds 5 --tag v783stock
```

`run_batch` gained a `silence=` kwarg for this run (`gpu_sim.py`): an int64
index array whose neurons never cross threshold. It gates OUTPUT only - a
silenced cell still receives synaptic input, which is what Christie's silencing
means too. `regime/test_silence.py` asserts the empty set is bit-identical to
the untouched path, that a silenced Fox emits zero counts, that a device tensor
behaves like an ndarray, and that overlapping the Poisson-driven set is refused
(`no_spike` masks `fired`, not `spk = fired | poisson`, so a
stimulated-and-silenced neuron would still spike).

## 1. Cell-set resolution: the anatomy matches Christie to the synapse

The CB* FlyWire labels are in `neuron_meta.npz`'s **`cell_type`** column.
`hemibrain_type` reads `"none"` for every one of them, so the obvious resolver
returns the empty set silently. All 11 Christie types resolve:

| set | types | n cells |
|---|---|---|
| Fox | CB0525 | 2 |
| FDA-I (direct onto PAM) | CB0546 2, CB0272 2, CB3199 4 | 8 |
| FDA-II | CB0233 2, DNp62 2, CB0337 2, CB1514 3, CB1025 3, CB3470 4, CB3573 2 | 18 |
| PAM | `cell_type` starts with `PAM` | 307 |

Two independent synapse counts, computed from `brain_gpu.npz` CSR and compared
against Christie Data S2A **before** the run:

| edge | this matrix | Christie Data S2A |
|---|---|---|
| Fox -> CB0233 | **217** synapses (4 edges) | 217 |
| Fox -> FDA-I (sum) | **171** synapses (10 edges) | 106 + 55 + 10 = 171 |

Both exact. The paper does not state a FlyWire version; these matches confirm
v783, and confirm that the reward path Christie describes is physically present
in the matrix flychess simulates. Whatever the disagreement is, it is not a
missing wire.

Two further counts, not in the paper, that turn out to carry the result:

| edge | synapses | edges |
|---|---|---|
| FDA-I -> PAM | **129** | 19 |
| FDA-II -> PAM | 0 | 0 |
| Fox -> PAM | 0 | 0 |
| total input onto the 307 PAMs | 15,854 | - |

So FDA-I -> PAM is **0.81%** of everything PAM receives, and it is the *only*
direct route in this set. FDA-II reaches PAM only through FDA-I or not at all.

We silence every cell of each type, where Christie silenced specific pairs
(CB3199 has 4 cells here, CB1514 and CB1025 have 3). Silencing extra cells of
the same type can only strengthen a null result, which is the direction this
went.

## 2. Hop rates: 1 and 2 conduct, 3 does not

5 seeds per cell, 300 ms, stock matrix, sugar GRNs alone (no CS song). Mean Hz
over the whole set.

| cond | hz | Fox (n=2) | FDA-I (n=8) | FDA-II (n=18) | PAM (n=307) |
|---|---|---|---|---|---|
| base | 0 | 0.00 | 0.00 | 0.00 | 0.000 |
| sugar | 100 | 9.00 | 0.08 | 0.67 | **0.000** |
| sugar-fox | 100 | 0.00 | 0.08 | 0.37 | 0.000 |
| sugar-fdaI | 100 | 9.00 | 0.00 | 0.67 | 0.000 |
| sugar-fdaII | 100 | 9.00 | 0.08 | 0.00 | 0.000 |
| sugar-fdaAll | 100 | 9.00 | 0.00 | 0.00 | 0.000 |
| sugar | 150 | 14.67 | 0.58 | 0.93 | **0.000** |
| sugar-fox | 150 | 0.00 | 0.50 | 0.41 | 0.000 |
| sugar-fdaI | 150 | 14.67 | 0.00 | 0.93 | 0.000 |
| sugar-fdaII | 150 | 13.67 | 0.58 | 0.00 | 0.000 |
| sugar-fdaAll | 150 | 13.67 | 0.00 | 0.00 | 0.000 |
| sugar | 200 | 19.67 | 1.25 | 1.33 | **0.000** |
| sugar-fox | 200 | 0.00 | 1.42 | 0.56 | 0.000 |
| sugar-fdaI | 200 | 19.67 | 0.00 | 1.33 | 0.000 |
| sugar-fdaII | 200 | 20.33 | 1.25 | 0.00 | 0.000 |
| sugar-fdaAll | 200 | 20.33 | 0.00 | 0.00 | 0.000 |

Reading it hop by hop:

- **Drive landed.** Sugar GRNs measure 150.25 Hz under the 150 Hz command.
- **Hop 1 conducts and is sugar-driven.** Fox goes 0.00 -> 9.00 -> 14.67 ->
  19.67 Hz, monotone in the GRN rate, and to exactly 0.00 when silenced. It is
  one of the two Fox cells doing it (`n_responsive` 1.0/2 at 100 and 150 Hz,
  1.6/2 at 200 Hz); the other is silent.
- **Hop 2 conducts, weakly, and is half Fox-dependent.** FDA-II mean rises
  0.67 -> 0.93 -> 1.33 Hz with drive, carried by 1-2 responsive cells out of 18
  whose individual rates reach 10.67 / 16.00 / 24.00 Hz. Silencing Fox roughly
  halves it (0.37 / 0.41 / 0.56 Hz), so about half of FDA-II's activity here is
  Fox-driven and half arrives from the GRNs by some other route.
- **FDA-I is not Fox-dependent in this sim.** Silencing Fox leaves FDA-I
  unchanged or marginally higher (0.58 -> 0.50 at 150 Hz, 1.25 -> 1.42 at 200
  Hz) - noise, on 1 responsive cell of 8. The 171 Fox -> FDA-I synapses are real
  but never dominate FDA-I's input here.
- **Hop 3 does not conduct at all.** PAM `max` is 0.00 Hz in **every one of the
  80 runs**. Not "low" - zero spikes, across 307 cells x 16 conditions x 5
  seeds. There is nothing for a silencing condition to remove, which is why
  every silenced row reads the same as the intact one.

## 3. Against Christie's own numbers

Christie's readout is a COUNT of PAM-DANs with nonzero mean rate over 30 x 1000
ms trials, which is far more lenient than flychess's usual `>= 1 Hz` cut. Both
are reported; theirs leads, for comparability.

| condition | Christie (Data S3/S5) | this run (150 and 200 Hz) |
|---|---|---|
| Fox intact, high GRN rate | ~200 responsive PAMs at Wsyn 0.39 | **0 / 307** |
| Fox intact, one Wsyn step lower | 0-11 responsive at Wsyn 0.37-0.386 | - |
| Fox silenced | 0 everywhere | 0 / 307 |
| CB0233 alone silenced | 0 | (not run separately; FDA-II incl. CB0233: 0) |
| FDA-II silenced | 0 | 0 / 307 |
| FDA-all silenced | 0 | 0 / 307 |
| FDA-I silenced | still ~200 at 120/190 Hz | 0 / 307 |

Our result is Christie's *silenced* column everywhere, including where they
report an intact response. The one structural thing we reproduce is the
asymmetry's direction: in their model FDA-II matters and FDA-I does not, and in
ours FDA-II is the arm Fox actually drives while FDA-I is Fox-independent.

## 4. Model differences, stated rather than closed over

These are real and none of them was tuned after the result:

- **Run length.** 300 ms, 5 seeds (`C.T_RUN`) vs Christie's 1000 ms x 30 trials.
  A PAM with a long onset latency would be missed. Mitigated but not excluded by
  PAM max being exactly 0 - there is no sub-threshold trickle to extrapolate.
  `--t-run 1000` is one flag if this is ever challenged.
- **No scalar Wsyn.** Christie's result lives in a narrow band: ~200 responsive
  PAMs at Wsyn 0.39 mV, 0-11 one step lower at 0.386. flychess has no single
  global synaptic weight to sweep; its matrix is normalised per target
  (`NORM_TOTAL_TARGET`), so "run it at their weight" is not a flag, it is a
  different experiment. **This is the most likely locus of the disagreement.**
- **LIF constants.** Shiu 2024 defaults (Vth -45, Vrest -52, tau 5 ms, delay
  1.8 ms) were not ported. Do not claim the two models are the same model.
- **Responsiveness criterion.** Theirs: nonzero mean rate over 30 s of
  simulated time. Ours: nonzero rate in one 300 ms window. Ours is strictly
  harder to satisfy for a rare spiker - but a cell firing once per second would
  still land in ~30% of our 300 ms windows across 5 seeds, and we saw zero.
- **Set membership.** Type-level, so 26 FDA cells vs Christie's 11 pairs. Only
  strengthens a null.

## Verdict

**The CLOSED verdict survives, on the falsifier written before the output was
read.** That falsifier was: CLOSED survives iff, with Fox intact at 150 and 200
Hz, PAM `n_responsive` (Christie's criterion, rate > 0) is < 15/307 **and** PAM
mean < 0.5 Hz. Measured: **0/307 and 0.000 Hz** at both rates. Retraction
required `>= 150/307` with Fox intact; nothing came close.

**What changes is the reason.** The floor-1 argument was that sugar's
contribution is 0.09% of PAM's input, therefore no reward path. That inference
was not a firing test and should not have been stated as a closed one. The
firing test now exists, and it says something more specific and more defensible:

> The Christie path is present in the v783 matrix to the synapse (Fox -> CB0233
> = 217, Fox -> FDA-I = 171). Sugar drives Fox to 15-20 Hz and Fox drives about
> half of FDA-II's response. The chain dies at FDA-I -> PAM, a 129-synapse,
> 0.81%-of-input connection that never brings a single one of 307 PAM-DANs to
> threshold at 100, 150 or 200 Hz GRN drive. In Christie et al. the same chain
> recruits ~200 PAMs, but only at a global synaptic weight one step below
> runaway (Wsyn 0.39; 0-11 PAMs at 0.386), which our per-target-normalised
> matrix does not reach.

That is a disagreement about the operating point of two models, not about the
connectome. It is publishable as such, and it is not a claim that Christie is
wrong.

## How to apply

- The preprint's reward section may say: the anatomical path exists and
  conducts to the ascending FDA cells; PAM recruitment fails at the last hop at
  our operating point; Christie et al. obtain it at a near-runaway global
  weight. Cite both. Do **not** write "there is no reward path in the
  connectome" - that is now measurably too strong.
- Do **not** cite the 0.09% synapse fraction as the reason the reward path is
  closed. It remains a true statement about synapse fractions; it was never a
  firing test. Cite this document instead.
- The reel may say the fly's sim brain does not learn reward the way it learns
  punishment, and that the wiring for it is there. It may not say the wiring is
  missing.
- Before any Rung A attempt with a real sugar US: this is still a no. The US
  would be silent.
- The open follow-up, if reward matters enough to spend on: a global
  weight-scale sweep, to find whether flychess has a regime where FDA-I -> PAM
  crosses threshold and, if so, what else in the brain has gone unstable by
  then. Christie's own model is bimodal across one Wsyn step, so "it fires at
  the right weight" and "it fires just below runaway" may be the same sentence.
