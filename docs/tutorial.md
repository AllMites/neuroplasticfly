# Tutorial: from the connectome to a learned odour

This page walks one path from start to end. You build the brain, train it to stop approaching one odour by pairing
that odour with dopamine, read out what changed in the mushroom-body output neurons and the descending neurons,
and then ask where along the olfactory pathway an odour signal is carried or lost. Every code block below was run
for this page, in order and in one Python session, and the output under it is what that run printed, trimmed only
where marked. Everything you will see is a property of the model; the last section, [Limits](#limits), says how far
that carries.

## A few terms

An odour is sensed by **olfactory receptor neurons (ORNs)**, which project to the antennal lobe, where **projection
neurons (PNs)** carry it on to the mushroom body. There it is recoded by about 5,000 **Kenyon cells (KCs)**, each
of which responds to only a few odours. The KCs connect to **mushroom-body output neurons (MBONs)**; in this model
each MBON is labelled either *approach* or *avoid*, and the difference between the two groups' rates, the **avoid
index** (avoid minus approach, in Hz), is the readout of what the brain makes of an odour. **Dopaminergic neurons
(DANs)** project onto the same compartments: the **PPL1** cluster carries punishment and the **PAM** cluster
reward. **Descending neurons (DNs)** send signals from the brain to the body.

In conditioning, the odour is the **conditioned stimulus (CS)** and the dopamine is the **unconditioned stimulus
(US)**. When the two arrive together, the KC-to-MBON synapses that the odour activated are weakened. These 18,674
synapses are the only ones in the model that change.

## Before you start

You need the package installed and `data/` built, as described in [install.md](install.md), and a CUDA GPU. Run
the Python below from the root of your clone: the signal locator in `regime/` and two older modules that the engine
imports are not part of the installed package, so they are found only with the clone root on the path.

## 1. Build the brain

The connectome is downloaded and built once, on your machine:

```
python scripts/fetch_data.py
python scripts/build_data.py
```

These two commands were not re-run for this page (the download is 852 MB; [install.md](install.md) describes
what they fetch and check). To check an existing build without rebuilding it, verify its content hashes:

```
python scripts/checksum_data.py --verify
```

```
  brain_gpu.npz          OK
  neuron_meta.npz        OK
  nt_conf.npz            OK
  root_ids_sorted.npy    OK
OK 4/4
```

Now load it onto the GPU. The odour experiments run in the `eln8` regime, an operating point that we chose and
that is explained under [Limits](#limits). It has to be set before the simulator is constructed, because it is
written into the weights at that moment.

```python
from neuroplasticfly import engine as G
from neuroplasticfly.learn import condition as C
from neuroplasticfly.learn import plastic as PL

G.ELN_NEGATE, G.PN_KC_GAIN = C.REGIMES["eln8"]   # the odour operating point; set before GpuSim()
sim = G.GpuSim()                                  # loads data/brain_gpu.npz onto the GPU
P = PL.Plastic.real()                             # the 18,674 plastic KC->MBON synapses
print(sim.net.n, "neurons,", sim.net.n_edges, "synapses,", P.n_edges, "plastic")
```

```
139248 neurons, 2700429 synapses, 18674 plastic
```

`sim.net.n_edges` counts connected neuron pairs, each carrying its synapse count as a weight.

## 2. One trial, before training

A trial is 300 ms of simulation with one odour channel driven. `C.trial` drives the 20 receptor neurons of the
DC2 glomerulus at 80 Hz and returns the firing rate of every neuron; `C.readout` reduces that to the numbers used
throughout.

```python
rates = C.trial(sim, "dc2", seed=0)       # 300 ms with ORN_DC2 at 80 Hz; rates in Hz, one per neuron
before = C.readout(rates, P)
for k in ("mbon_approach_hz", "mbon_avoid_hz", "avoid_index", "ppl1_hz", "kc_active"):
    print("%-18s %8.3f" % (k, before[k]))
```

```
mbon_approach_hz     11.954
mbon_avoid_hz         0.370
avoid_index         -11.584
ppl1_hz               0.000
kc_active             0.060
```

Before training, the odour drives the approach MBONs at about 12 Hz and the avoid MBONs hardly at all, so the
avoid index is strongly negative. About 6% of the Kenyon cells respond, which is the sparse code the `eln8`
regime was chosen to give. The first trial also compiles the simulation step, which takes a few seconds, and torch
may print warnings from its compile cache; in our runs these did not stop the run.

## 3. A conditioning experiment

`C.run_arm` runs the published protocol for one arm. It first probes each odour (the *baseline*), then trains
for `n_train` trials, and probes again every five trials. In the `learn` arm, DC2 is paired with punishment on
every training trial, while the odour D is presented on the same trials without it; a third odour, DA1, is only
ever probed. The punishment is delivered by driving the PPL1 dopamine neurons directly at 100 Hz
(`us_path="dan"`), and `eta` sets the learning rate.

```python
import io
log = io.StringIO()                       # one JSON line per trial; a file works too
P, rows = C.run_arm(sim, "learn", C.ODOUR_ARMS["learn"], n_train=30, n_probe=3,
                    us_path="dan", eta=5e-6, lam=PL.LAM, log=log, seed0=0,
                    probes=C.STIM_SETS["odour"])
cv = C.curves(rows)["learn"]              # {odour: {trial: [mean, sd]}} of the avoid index
for odour in ("dc2", "d", "da1"):
    t = sorted(cv[odour])
    print("%-4s avoid index %+6.2f -> %+6.2f Hz  (shift %+6.2f)"
          % (odour, cv[odour][t[0]][0], cv[odour][t[-1]][0], cv[odour][t[-1]][0] - cv[odour][t[0]][0]))
print(P.summary())
```

```
dc2  avoid index -11.01 ->  +0.10 Hz  (shift +11.11)
d    avoid index -11.94 -> -10.58 Hz  (shift  +1.36)
da1  avoid index  -1.99 ->  -1.69 Hz  (shift  +0.31)
{'mean_frac': -0.019755352288484573, 'min_frac': -0.8910000324249268, 'n_at_floor': 189, 'n_at_ceiling': 0, 'max_frac': 0.0, 'sha': 'fcf68616dc5f488c'}
```

This took about five minutes on an RTX 5070 Ti that was shared with another job. In the model, the avoid index of
the paired odour rises by about 11 Hz, while the unpaired odour moves by about 1.4 Hz and the never-paired odour
hardly at all. The published five-seed result for the same protocol, with five probe trials instead of three, is
+11.05 Hz for the paired odour, +1.43 Hz for the unpaired one and +0.56 Hz for the never-paired one. The summary
shows where the change sits: 189 of the 18,674 plastic synapses have been driven to their floor (the weakest they
can become, about 11% of their starting weight once recovery is counted), and the mean change over all of them is
only 2%, because only the synapses from KCs that DC2 activated were weakened.

The same experiment, with the lesion control (the same pairings with the plasticity rule switched off) and the
results written to `results/` and `brain_state/`, runs from the command line:

```
python learn/condition.py --state my_run --cs odour --regime eln8 --arms learn,lesion --us-path dan --eta 5e-6 --seed0 0
```

## 4. Read out MBON and DN rates

Run the same DC2 trial on the trained brain and compare it with the trial from step 2:

```python
rates = C.trial(sim, "dc2", seed=0)
after = C.readout(rates, P)
for k in ("mbon_approach_hz", "mbon_avoid_hz", "avoid_index", "dn_DNp01", "dn_DNa02", "dn_DNg29"):
    print("%-18s %8.3f -> %8.3f" % (k, before[k], after[k]))
```

```
mbon_approach_hz     11.954 ->    0.230
mbon_avoid_hz         0.370 ->    0.370
avoid_index         -11.584 ->    0.140
dn_DNp01              0.000 ->    0.000
dn_DNa02              0.000 ->    0.000
dn_DNg29              0.000 ->    0.000
```

The whole change is in the approach MBONs, which fall from about 12 Hz to almost nothing, while the avoid MBONs do
not move at all. In the model, then, training removes the attraction to the odour rather than adding an aversion
to it. The descending neurons that `readout` reports (DNp01, DNg29, DNg84, DNa02 and DNp04) are silent during this
odour trial both before and after training, so in this protocol the learned change does not reach them.

`rates` holds one rate per neuron, indexed in the order of `data/neuron_meta.npz`, and `C.ct` and `C.cc` hold every
neuron's cell type and class, so any cell type can be read by name:

```python
import numpy as np
mbon01 = np.flatnonzero(C.ct == "MBON01")
dna02 = C.DN["DNa02"]                     # DNs used by readout(): DNp01, DNg29, DNg84, DNa02, DNp04
print("MBON01", mbon01, rates[mbon01])
print("DNa02 ", dna02, rates[dna02])
```

```
MBON01 [ 64817 132824] [0. 0.]
DNa02  [  904 93410] [0. 0.]
```

## 5. Run a probe: where is the odour signal lost?

The silent DNs raise a question that the signal locator answers directly: along the pathway from receptor neurons
to descending neurons, at which stage does an odour stop making a difference? `regime/locator.py` drives several
stimulus groups at several seeds, reads one number per stage, fits the slope of that number against the stimulus,
and labels the first stage where the slope is no longer clearly above its noise ([probes.md](probes.md) describes
it in full).

The label rules belong in a preregistration, written before any data exist. For this page they go in a file
`PREREGISTER_tutorial.md` next to where you run the code:

````markdown
# Tutorial probe: does a DC2 dose signal reach the descending neurons?

```locator
stages: ORN > PN > KC > MBON > DN
t_min: 2
silent_hz: 0.1
```
````

A stage carries the signal when its slope is at least twice its standard error across seeds and its mean rate is
at least 0.1 Hz. For a real experiment, commit the file and freeze it with `prereg.py` before the run, as described
in [prereg.md](prereg.md); `locate` records the file's hash in its output either way.

Then drive DC2 at three strengths, four seeds each, and read five stages: the DC2 receptor neurons, the DC2
projection neurons, all Kenyon cells, the approach MBONs, and the five DN types above.

```python
from regime import locator as L
x = np.array([0.0, 40.0, 80.0])           # ORN_DC2 drive per stimulus group, Hz
orn = np.flatnonzero(C.ct == "ORN_DC2")
drives = [C.to_prob(orn, np.full(len(orn), hz, np.float32)) if hz > 0
          else (np.empty(0, np.int64), np.empty(0))          # no drive: an empty pair, not a 0 Hz rate
          for hz in x]
seeds = [0, 1, 2, 3]
rates = L.simulate(sim, drives, seeds, t_run=1000.0, warm=200.0)   # [3 groups, 4 seeds, N]
appr = np.unique(P.mbon_of_edge[P.mbon_valence_of_edge == "approach"])
cells = {"ORN": orn, "PN": np.flatnonzero(C.ct == "DC2_adPN"),
         "KC": np.flatnonzero(C.cc == "Kenyon_Cell"), "MBON": appr,
         "DN": np.concatenate(list(C.DN.values()))}
stages, hz = {}, {}
for name, idx in cells.items():
    stages[name], hz[name] = L.readout(rates, idx)
out = L.locate(stages, x, "PREREGISTER_tutorial.md", mean_hz=hz)
print(L.table(out))
```

```
stage              slope       sem       t      snr   meanHz  carries
ORN              +1.0252    0.0090  +113.9 1792.970    41.05  yes
PN               +1.7906    0.0092  +194.6 2318.848    77.40  yes
KC               +0.0461    0.0003  +160.6 1523.826     1.94  yes
MBON             +0.0000    0.0000    +0.0      inf     0.00  no SILENT
DN               +0.0000    0.0000    +0.0      inf     0.00  no SILENT
LABEL: LOST AT MBON (SILENT)
```

Each row is one stage: `slope` is the change of the stage's mean rate per Hz of receptor drive, `t` is that slope
over its standard error across the four seeds, and `snr` compares the variance between stimulus groups with the
variance between seeds. A silent stage has no variance at all, which is why its `snr` reads `inf`; read it as no
signal. The odour signal passes the receptor neurons, the projection neurons and the Kenyon cells, and stops at
the approach MBONs, which are silent at every strength.

`sim` still holds the trained weights from step 3, so this is the trained brain. Running the same probe on the
naive brain separates what training did from what the wiring does:

```python
PL.Plastic.real().push(sim)               # back to the untrained weights
rates = L.simulate(sim, drives, seeds, t_run=1000.0, warm=200.0)
stages, hz = {}, {}
for name, idx in cells.items():
    stages[name], hz[name] = L.readout(rates, idx)
print(L.table(L.locate(stages, x, "PREREGISTER_tutorial.md", mean_hz=hz)))
```

```
stage              slope       sem       t      snr   meanHz  carries
ORN              +1.0252    0.0090  +113.9 1792.970    41.05  yes
PN               +1.7898    0.0075  +239.3 2519.043    77.40  yes
KC               +0.0472    0.0003  +186.6 1718.200     1.97  yes
MBON             +0.1564    0.0017   +92.4  867.826     5.51  yes
DN               +0.0000    0.0000    +0.0      inf     0.00  no SILENT
LABEL: LOST AT DN (SILENT)
```

On the naive brain the approach MBONs do follow the dose, so in the model it is the training that silenced them.
The five DN types are silent with or without training, so the loss at that stage comes from the wiring and the
regime rather than from learning; a probe that asks whether an odour reaches behaviour would need other
descending neurons, or a longer pathway, in its stage list. The two probes took about 10 s each. Running the whole
page, from loading the brain to the second probe, took a little over five minutes on the shared RTX 5070 Ti.

These labels are results of a tutorial run, with three stimulus strengths, four seeds and no frozen
preregistration; report anything like them as "in the model, in our testing", and only after the steps in
[prereg.md](prereg.md).

## Limits

Everything above is a statement about a simulation. The model is built from a measured wiring diagram, but much of
what happens on top of that wiring was chosen by us. [WHAT_IS_REAL.md](../WHAT_IS_REAL.md) sorts every part of the
repository into what is measured, authored, trained and absent, and the README's *What this is not* lists the limits
of the odour result. The main ones, in plain words:

- **The wiring is measured; the neurons are not.** Every neuron is the same simple integrate-and-fire unit with the
  constants of Shiu et al. 2024, and connections of fewer than five synapses are dropped. A neuron in the model has
  no cell-type-specific channels, no gap junctions and no neuromodulation other than the dopamine rule above.
- **The operating point is authored.** The `eln8` regime is a bridge we added, not physiology. It flips the sign
  of the outputs of 44 excitatory antennal-lobe local neurons, because in the model their chemical output merges
  every odour into one pattern (in the fly that lateral excitation is thought to be electrical, which this model
  cannot express), and it multiplies the input from projection neurons to Kenyon cells by 8, so that a sparse
  fraction of Kenyon cells responds to one odour. Some results move with that gain: the small transfer of learning
  to the unpaired odour, 12.9% of the paired shift in the published runs, is 13.9% at a gain of 8 and -0.2% at 6.
- **The punishment is injected, not tasted.** The conditioning drives the PPL1 dopamine neurons directly. In the
  model the sugar pathway reaches the PAM reward neurons anatomically but does not make them fire, so there is no
  reward arm.
- **Only the KC-to-MBON synapses learn**, by a rule we wrote: dopamine-gated depression with a constant recovery
  toward baseline. Forgetting is that constant, `LAM`; the 74% of the weight change that remains 30 trials later
  is `0.99^30`, a line in `learn/plastic.py` rather than a measured property of memory.
- **The learning is a loss of approach, not a gain of avoidance.** The approach MBONs fall almost to silence and
  the avoid MBONs do not move, as step 4 showed, so the supported reading is that the model stops finding the odour
  attractive, not that it learns to avoid it. Because the effect runs into that floor, "it learned" holds and "it
  learned this much" does not.
- **No learned DN code has been shown.** In the published runs the MBON readout separates learning from the
  shuffle control (the same training with the eligibility of each synapse read from a random Kenyon cell), but the
  descending-neuron population readout does not.
- **Seeds are noise replicates of one brain, not different flies.** There is one connectome, from one fly.
- **Some paths are not runnable here.** Song conditioning needs audio tooling and commercial music that are not
  distributed, and the closed-loop test of paper 0, which drives a simulated body, is not included yet.
