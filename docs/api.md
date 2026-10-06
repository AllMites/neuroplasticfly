# API reference

The public entry points, with their signatures as they appear in the code and one line on each. The
[tutorial](tutorial.md) shows them working together; the docstrings in the source files have the details.

Four modules are reachable through the `neuroplasticfly` package after an editable install
([install](install.md)); each name is the same module object as the file it aliases, so a setting made through
one is seen by the other.

| import | file |
|---|---|
| `from neuroplasticfly import engine` | `gpu_sim.py`, the spiking engine |
| `from neuroplasticfly.learn import plastic, condition` | `learn/plastic.py`, `learn/condition.py` |
| `from neuroplasticfly.rate import engine as rate_engine` | `rate/engine.py`, the rate model |
| `from neuroplasticfly import prereg` | `prereg.py` |

The signal locator is not part of the installed package; import it as `from regime import locator` with the root
of your clone on the path (run from the clone root, or import `neuroplasticfly.learn.condition` first, which puts
the root on `sys.path`). The same holds for `GpuSim()` with no brain argument, which imports `reservoir.py` from the
clone root.

Units used throughout: rates in Hz, times in ms, and a *drive* is a pair `(idx, prob)` of neuron indices and the
probability of a Poisson spike per 0.1 ms step (`hz * DT / 1000`). A driven neuron fires that Poisson train and
ignores its synaptic input.

## Spiking engine: `gpu_sim` (`neuroplasticfly.engine`)

Leaky integrate-and-fire neurons with the constants of Shiu et al. 2024 (`DT` 0.1 ms, `V_TH` -45 mV, `TAU_M`
20 ms, `TAU_SYN` 5 ms, `T_REFR` 2.2 ms, `DELAY` 1.8 ms, `W_SYN` 0.275 mV per synapse).

**Module settings.** Set these before constructing `GpuSim`, because some are written into the weights at
construction. All of them are authored choices, and all default to off.

| name | default | what it does |
|---|---|---|
| `ELN_NEGATE` | `False` | negates the outputs of the 44 excitatory antennal-lobe local neurons (the `eln8` regime sets it) |
| `PN_KC_GAIN` | `1.0` | multiplies projection-neuron to Kenyon-cell weights (8.0 in `eln8`) |
| `KC_V_TH_DELTA` | `0.0` | mV added to the Kenyon-cell spike threshold, read on every run |
| `W_SYN` | `0.275` | mV per synapse |
| `BRAIN` | `data/brain_gpu.npz` | the brain file; the environment variable `FLYCHESS_BRAIN` overrides it |

The module also carries the switches of the regime experiments (`SFA_B_INC`, `APL_GRADED`, `NORM_TOTAL_TARGET`,
`GAP_COUPLE`, `SLOW_FRAC`, `TYPE_W_SCALE`); their comments in `gpu_sim.py` say what each one adds.

**`GpuSim(n=None, device="cuda", compile_=True, deterministic=True)`**
Loads the brain onto the device and holds it there for every later run. `n` defaults to `brain()`;
`compile_=False` runs the step without `torch.compile`.

**`GpuSim.run_batch(drives, seeds, t_run=300.0, rng="counter", progress=None, state=None, return_state=False, silence=None, bias=None)`**
Runs B simulations at once, one per drive and seed, and returns spike counts as an int32 tensor `[B, N]`; divide
by `t_run / 1000` for rates. `state`/`return_state` carry the integrator across calls exactly, `silence` stops the
listed neurons from firing while they still receive input, and `bias` adds a constant depolarisation in mV.

**`GpuSim.set_plastic(offsets, values)`**
Overwrites the weights at flat CSR offsets with signed values in mV; `Plastic.push` calls it.

**`brain()`**
The process-wide `Brain`, loaded from `BRAIN` on first use.

**`Brain(path=BRAIN)`, `Brain.select(selector)`**
The arrays of the brain file; `select` returns the neuron indices of an exported selector string such as
`"cell_type=DNa02,side=left"`, and raises on a selector that was not exported.

**`drive_of_selector(selector, rate, n=None)`**
The drive for one selector at one rate in Hz.

**`bias_mv_for_hz(hz)`**
The `bias` in mV that makes an isolated neuron fire at `hz`.

## Plasticity: `learn/plastic.py` (`neuroplasticfly.learn.plastic`)

The only synapses that change are the 18,674 from Kenyon cells onto mushroom-body output neurons. Constants:
`ETA = 1e-3`, `LAM = 0.01` (recovery toward baseline per trial), `W_MIN = 0.1` and `W_MAX = 1.8` (floor and
ceiling as fractions of the baseline weight).

**`Plastic.real(brain=BRAIN)`**
The plastic synapse set of the real brain, with each synapse's Kenyon cell, output neuron, valence and compartment,
and the dopamine neurons of each valence.

**`Plastic.toy()`**
A 7-neuron set for tests.

**`Plastic.update(rates, eta=ETA, lam=LAM, lesion="none", compartment=False)`**
One trial of the depression-only rule: each synapse is weakened in proportion to its Kenyon cell's rate times the
rate of the dopamine neurons that oppose its output neuron's valence, then recovers by `lam`. `lesion` is
`"none"`, `"punish"`, `"reward"` or `"all"`. Returns the weight change applied.

**`Plastic.update_timed(rates, eta, lam=LAM, lesion="none", compartment=False)`**
One tick of the timing rule: odour before or with dopamine depresses, dopamine before odour potentiates. Call it
on every tick; `reset_lag()` clears the one-tick memory.

**`Plastic.push(sim)`**
Writes the current weights into a `GpuSim` (or `RateAsGpuSim`).

**`Plastic.w()`, `Plastic.summary()`**
The current weights, and a summary of the change (`mean_frac`, `min_frac`, `max_frac`, `n_at_floor`,
`n_at_ceiling`, `sha`).

**`Plastic.save(d)`, `Plastic.load(d)`**
Store or restore the weight change in directory `d` (`dw.npy` plus a hash in `state.json`).

## Conditioning: `learn/condition.py` (`neuroplasticfly.learn.condition`)

Importing the module reads `data/neuron_meta.npz`, so `data/` must be built.

Constants: `ODOURS` (`dc2`, `d` and `da1`, mapped to the receptor-neuron types `ORN_DC2`, `ORN_D` and `ORN_DA1`),
`ORN_HZ = 80.0`, `DAN_HZ = 100.0`, `T_RUN = 300.0` ms, `STIM_SETS["odour"]` (paired, unpaired, never paired),
`REGIMES` (`stock`, `eln8`, `eln6`, each a pair `(ELN_NEGATE, PN_KC_GAIN)`), and `ODOUR_ARMS` (`learn`, `lesion`,
`reversed`, `frozen`, `shuffle`). The arrays `ct` and `cc` hold every neuron's cell type and class, and `DN` maps
five descending-neuron types to their indices.

**`odour_drive(name)`**
`(idx, hz)` for one odour channel at `ORN_HZ`.

**`us_drive(us, path)`**
`(idx, hz)` for the unconditioned stimulus: `us` is `None`, `"punish"` or `"reward"`, and `path` is `"dan"` (drive
the dopamine neurons directly) or `"grn"` (drive the taste neurons).

**`to_prob(idx, hz)`**
Converts `(idx, hz)` into a `run_batch` drive.

**`trial(sim, song, us=None, us_path="grn", seed=0)`**
One 300 ms trial of a stimulus (an odour name, for this release), optionally paired with the unconditioned
stimulus; returns the rate of every neuron in Hz. For odour work pass `us_path="dan"`.

**`readout(rates, P)`**
Summarises one trial: `mbon_approach_hz`, `mbon_avoid_hz`, `avoid_index` (avoid minus approach), `pam_hz`,
`ppl1_hz`, `kc_active`, `central_active`, and `dn_<type>` for each type in `DN`.

**`run_arm(sim, name, arm, n_train, n_probe, us_path, eta, lam, log, seed0=0, dump=None, probes=(...))`**
One arm of the protocol: probe every stimulus, train for `n_train` trials (probing every 5), and return the
trained `Plastic` and one row per trial; each row is also written to `log` as a JSON line. For odours pass
`probes=STIM_SETS["odour"]`.

**`curves(rows)`**
The avoid index of the probe rows, as `{arm: {stimulus: {trial: [mean, sd]}}}`.

**Command line.** `python learn/condition.py --state NAME --cs odour --regime eln8 --arms learn,lesion --us-path
dan --eta 5e-6` runs the published protocol and writes `results/condition_NAME.json` and `brain_state/NAME/`.

## Rate model: `rate/engine.py` (`neuroplasticfly.rate.engine`)

The same connectome with each neuron replaced by a rate unit, `tau dv/dt = -v + gain * w_scale * W r + bias`,
with parameters per cell-type pool (paper 1).

**`RateSim(w_scale, path=G.BRAIN, device="cuda", dtype=torch.float32, csr=None, pool=None, regime="stock")`**
Builds the model on the brain file, or on a given CSR matrix and pool ids; `regime="eln8"` applies the same two
edits as the spiking `eln8`. `w_scale` is required.

**`RateSim.run(drives, t_run, silence=None, state=None, return_state=False, record=None)`**
Drives here are `(idx, hz)` pairs; returns mean rates `[B, N]` over `t_run`, plus traces of the `record` neurons
and the state when asked.

**`RateSim.set_plastic(offsets, values)`**
The plastic edges, as signed synapse counts at the same offsets `Plastic` uses.

**`RateSim.set_trainable(*names, pools=None)`, `get_params(pools)`, `load_params(pools, vals)`,
`set_edge_gains(target_pools, cap)`, `spectral_bound(iters=100)`**
Fitting helpers: per-pool `log_tau`, `bias` and `gain`, and per-edge-type gains.

**`rate.suite.build(chunks=CHUNKS, base="c0", **kw)`**
The calibrated rate model of paper 1 (the chunk-0 base and every fitted chunk).

**`rate.as_gpusim.RateAsGpuSim(rsim)`**
Wraps a `RateSim` with the `run_batch` and `set_plastic` of `GpuSim`, so `learn/condition.py` runs on the rate
model unchanged.

## Preregistration: `prereg.py` (`neuroplasticfly.prereg`)

[docs/prereg.md](prereg.md) has the workflow and a worked example.

**`holds(flags, k=4)`**
True when at least `k` of the per-seed checks are true.

**`verdict(scores, pairs, totals=None, mech=None, positive=(), k=4)`**
Labels one preregistered ordering `PASS`, `PASS BY RULE, NOT SUBSTANTIVE`, `PASS, MECHANISM NOT SHOWN` or `FAIL`.

**`freeze(prereg, code, out)`, `verify(meta_path, results=())`**
Record a committed preregistration and the hashes of its code before the run; re-hash everything afterwards.

**Command line.** `python prereg.py freeze PREREGISTER_x.md --code a.py b.py -o results/x_meta.json`, then
`python prereg.py verify results/x_meta.json [--results ...]`. Installed as `neuroplasticfly-prereg`.

## Signal locator: `regime/locator.py`

[docs/probes.md](probes.md) has the inputs, outputs and a worked example.

**`simulate(sim, drives, seeds, t_run=2000.0, warm=200.0)`**
Runs every drive at every seed on a `GpuSim` and returns rates `[G, S, N]`.

**`readout(rates, plus, minus=None)`**
One stage's readout `y [G, S]` (mean over `plus` cells, minus the mean over `minus` cells) and its mean rate.

**`locate(stages, x, rules, mean_hz=None)`**
The per-stage slope of the readout against the stimulus contrast `x`, and the label (`CARRIED TO`, `LOST AT` or
`INSTRUMENT FAIL`) under the rules of a preregistration's `locator` block.

**`read_rules(path)`, `slope(y, x)`, `snr(y)`, `table(out)`**
The rule parser, the per-seed slope, the variance signal-to-noise ratio, and a plain-text table of a `locate`
result.

## Command-line entry points

| command | script | what it does |
|---|---|---|
| `neuroplasticfly-reproduce` | `reproduce.py` | the paper-0 specificity test, end to end (accepts only `--skip-download`) |
| `neuroplasticfly-reproduce-paper1` | `reproduce_paper1.py` | `--verify` the shipped hashes, `--rescore` the labels from the shipped logs |
| `neuroplasticfly-prereg` | `prereg.py` | `freeze` and `verify` |
