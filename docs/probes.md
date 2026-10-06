# Signal locator: where along a pathway is the signal lost?

`regime/locator.py` is one API for the probes this repository first wrote per experiment
(`regime/layers.py`, `regime/alln_probe*.py`, `snr_trials.py`, `where_locality_dies.py`). They all have the same
shape: drive groups of stimuli, read rates at successive stages of a pathway, compute one statistic per stage, and
report the stage where it collapses. The old scripts are kept as run; new experiments use this module.

It works on plain numpy arrays, so it runs on CPU without the brain. `simulate()` is an optional GPU path.

## Inputs

| name | shape | meaning |
|---|---|---|
| `x` | `[G]` | signed stimulus contrast per stimulus group, e.g. left-minus-right ORN drive in Hz `(-10, -5, 0, 5, 10)`; for two odours with no order, `(0, 1)` |
| `stages` | `{name: y [G, S]}` | one readout per stage, per group and seed (e.g. left-minus-right mean rate). Build it with `readout(rates, plus, minus)` from `rates [G, S, N]` |
| `rules` | prereg path or dict | the label rules, read from the preregistration's `locator` block |
| `mean_hz` | `{name: Hz}`, optional | the stage's mean rate, for the silence check (`readout` returns it) |

`simulate(sim, drives, seeds, t_run, warm)` gives `rates [G, S, N]` from a `gpu_sim.GpuSim` (needs `data/` and a GPU):
one `(stim_idx, stim_prob)` drive per group, every group run at every seed (common random numbers, as
`regime/probe.py`).

A group with no drive cannot be given as a 0 Hz rate: the engine's `run_batch` refuses any stimulus with a rate
of 0 or less. Pass an empty pair instead, `(np.empty(0, np.int64), np.empty(0))`, as the tutorial does.

## Outputs

`locate(stages, x, rules, mean_hz)` returns a dict:

- `label`: `CARRIED TO <last stage>`, `LOST AT <stage>` (with ` (SILENT)` if every cell group there is silent),
  or `INSTRUMENT FAIL (<first stage>)` when the stimulated stage itself does not carry the signal
- `lost_at`: the stage name, or `None`
- `stages[name]`: `slope` (mean of the per-seed OLS slopes of y on x), `sem` (across seeds), `t` = slope / SEM,
  `per_seed`, `snr` (variance SNR as in `snr_trials.decompose`: variance across groups over variance across seeds),
  `mean_hz`, `silent`, `carries`, `ratio_to_first` (slope relative to the first stage)
- `rules`: the rules used, with the preregistration's path and sha256

`table(out)` prints it as plain ASCII.

## Label rules in the preregistration

Put one fenced block with the info string `locator` in `PREREGISTER_<name>.md` (section in
`docs/PREREGISTER_TEMPLATE.md`), commit it, and `python prereg.py freeze` it before any data exist:

````markdown
```locator
stages: ORN > uPN > LHN | KC > MBON > DNa02   # pathway order; "a | b" = parallel, either may carry it
t_min: 2          # a stage carries the signal if |slope| >= t_min x SEM ...
silent_hz: 0.1    # ... and its mean rate is not below this ...
k: 4              # ... and (optional) >= k seeds have the slope's sign (prereg.holds)
```
````

One `key: value` per line, `#` starts a comment. `stages` is required; the others default to `t_min 2`,
`silent_hz 0.1`, no `k`. An unknown key is refused (a typo would otherwise run with the default), and so is a
preregistered stage with no data. The walk goes along `stages` in order and stops at the first step where no member
carries the signal.

## Worked example

A left/right odour signal driven at the ORNs, read at four stages (synthetic rates; `regime/test_locator.py` runs
this case). Each stage has 10 left and 10 right cells; the left-right difference has gain 1 at ORN, 0.6 at uPN and 0
from KC on, plus noise.

```python
import numpy as np
from regime import locator as L

x = np.array([-10.0, -5.0, 0.0, 5.0, 10.0])        # left-minus-right ORN drive per stimulus group, Hz
# rates: [5 groups, 8 seeds, N cells], from simulate() or any other source
stages, hz = {}, {}
for st in ("ORN", "uPN", "KC", "MBON"):
    stages[st], hz[st] = L.readout(rates, left[st], right[st])   # left-minus-right mean rate

out = L.locate(stages, x, "PREREGISTER_toy.md", mean_hz=hz)
print(L.table(out))
```

with `stages: ORN > uPN > KC > MBON`, `t_min: 2`, `silent_hz: 0.1`, `k: 6` in the preregistration:

```
stage              slope       sem       t      snr   meanHz  carries
ORN              +0.9880    0.0107   +91.9  308.498    49.99  yes
uPN              +0.6103    0.0066   +92.8  138.026    19.93  yes
KC               -0.0004    0.0134    -0.0   -0.089     5.02  no
MBON             -0.0072    0.0113    -0.6   -0.094     9.95  no
LABEL: LOST AT KC
```

Writing `uPN | KC` instead (parallel branches) moves the label to `LOST AT MBON`; raising `t_min` to `1e9` gives
`INSTRUMENT FAIL (ORN)`. The label comes from the file, so changing it after the data exist shows up in
`prereg.py verify`.

A negative SNR means the variance across groups is smaller than the noise predicts: no signal, read it as zero.
Report a label as "in the sim, in our testing".
