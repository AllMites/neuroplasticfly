# Preregister, label and verify your own experiment

Every result in this repository went through the same steps: preregistration committed before data, smoke gate,
full run, one analysis pass, label. `prereg.py` packages those steps for any experiment. It needs only Python and git.

## 1. Write the preregistration and commit it

Copy `docs/PREREGISTER_TEMPLATE.md` to `PREREGISTER_<name>.md`, fill every `<...>`, and commit it.
The label rules go in now, before you have seen any data.

```bash
cp docs/PREREGISTER_TEMPLATE.md PREREGISTER_loom.md
git add PREREGISTER_loom.md
git commit -m "prereg: loom selectivity"
```

## 2. Freeze

```bash
python prereg.py freeze PREREGISTER_loom.md --code my_run.py my_analysis.py -o results/loom_meta.json
```

`freeze` refuses a preregistration that is uncommitted or modified. It records the preregistration's commit and
commit time, and the sha256 of the preregistration and every code file. Run the smoke gate, then the full run.

## 3. Label, once

Score each seed, then call `verdict` with the rule exactly as written in the preregistration.
In this example the rule is LOOM > RECEDE and LOOM > DIM, with LOOM > 0, in at least 4 of 5 seeds:

```python
import json
from prereg import verdict

# one dict per seed. scores: baseline-subtracted rate; totals: raw spike counts over the stimulus frames;
# mech: the cell's own summed synaptic input
r = verdict(scores, pairs=[("LOOM", "RECEDE"), ("LOOM", "DIM")], positive=("LOOM",),
            totals=totals, mech=mech, k=4)
json.dump(r, open("results/loom.json", "w"), indent=1)
print(r["label"])
```

| label | meaning |
|---|---|
| `PASS` | ordering holds on scores, raw totals and (if given) input drive |
| `PASS BY RULE, NOT SUBSTANTIVE` | ordering holds only after baseline subtraction; raw totals do not show it |
| `PASS, MECHANISM NOT SHOWN` | output ordered, but the cell's own input is not |
| `FAIL` | ordering fails on the scores |

Why SUBST exists: when a cell is nearly silent, subtracting a baseline can produce an ordering out of a few spikes.
A toy case with the same shape as the false passes it caught in our own development:

| seed | LOOM score | RECEDE score | DIM score | LOOM total | RECEDE total | DIM total |
|---|---|---|---|---|---|---|
| 1-5 | 10 | 2 | 4 | 2 | 2 | 3 |

The scores give `PASS` in 5/5 seeds; the raw totals are flat, so the label is `PASS BY RULE, NOT SUBSTANTIVE`.
Report that label as an artefact, not as a result. `test_prereg.py` runs this case.

## 4. Record the results and verify

```bash
python prereg.py verify results/loom_meta.json --results results/loom.json   # once, right after the run
python prereg.py verify results/loom_meta.json                               # any time later, by anyone
```

`verify` re-hashes the preregistration, the code and the results, and checks that the preregistration's commit is
in the history. It exits 1 on any mismatch. Recording a result that is already recorded with a different hash is refused.

## Changing the plan

Add a dated **Amendment** to the preregistration, commit it before the data it applies to, and freeze again to a new
meta file. Keep the old label next to the new one; never edit a rule after its data exist.

Paper 1's own check (`reproduce_paper1.py --verify`) predates this module and is kept as run.
