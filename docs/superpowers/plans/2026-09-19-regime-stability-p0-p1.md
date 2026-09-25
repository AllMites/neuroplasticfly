# Regime Stability P0+P1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure why the whole-brain LIF ignites (P0), then give it two gain-control mechanisms - a graded non-spiking APL and spike-frequency adaptation - and measure whether the resulting regime holds an input-specific representation and survives its own plasticity (P1).

**Architecture:** A new read-only `regime/` package does the P0 measurements against `data/brain_gpu.npz` and `data/neuron_meta.npz`. P1 adds two elementwise/scatter operations to the existing `gpu_sim.py` step loop plus a one-off rescale of the weight tensor's values at construction, all three OFF by default until the final task, so every intermediate commit reproduces current behaviour bit-for-bit. Nothing about the CSR layout changes, so `learn/plastic.py` offsets keep working; the rescale does change the values written through them, which Task 5b handles in `Plastic.signed_values`.

**Tech Stack:** Python 3.13 in `flychess/.venv` (torch 2.11 cu128, numpy, scipy), RTX 5070 Ti. Run every command from the repository root with `P=.venv/Scripts/python.exe`. No test framework: each test file is a plain script with asserts, run with `$P regime/test_x.py`. Venv stdout is cp1252: never print non-ASCII.

Spec: `docs/superpowers/specs/2026-09-19-regime-stability-design.md`.

---

## Conventions used below

- `N = 139248` neurons. `meta = np.load("data/neuron_meta.npz")`: `cell_type`, `cell_class`, `side`, `super_class`, `cell_sub_class` (bytes arrays; use `.astype(str)`).
- `b = np.load("data/brain_gpu.npz")`: `W_indptr`, `W_indices`, `W_data`, `n_edges` (2,700,429). **CSR rows are PRESYNAPTIC, `W_indices` are POSTSYNAPTIC.** A neuron's outgoing edges are the contiguous slice `W_indices[indptr[i]:indptr[i+1]]`.
- APL is `cell_type == "APL"`, exactly 2 neurons at indices **17917** and **67229**. Out-degree 2763 / 2761, summed weight -50012 / -59464 (all inhibitory), 2586 / 2580 of those onto Kenyon cells. In-degree 2843 / 2818.
- Kenyon cells are `cell_class == "Kenyon_Cell"`, 5,177 of them.
- `sim.run_batch(drives, seeds, t_run=300.0)` takes a list of `(idx int64[], prob float64[])` where `prob = hz * G.DT / 1000`; returns int32 counts `[B, N]` on CUDA. Rates = counts / (t_run/1000).
- `.gitignore` excludes `data/` and `results/`. Commit code and `docs/`; never `git add -f` a data file. Where a step lists a `results/` path in `git add`, skip that path and say so.
- Commit after each task with the trailer `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.

## File map

| file | responsibility |
|---|---|
| `regime/__init__.py` | empty |
| `regime/reach.py` | P0.1 reachability between neuron sets over the CSR graph |
| `regime/weights.py` | P0.2 per-neuron input-weight distribution |
| `regime/probe.py` | P0.3 + P1 measurement: drive ORN channels, report KC sparsity / Jaccard / central |
| `regime/test_reach.py`, `regime/test_weights.py`, `regime/test_sfa.py`, `regime/test_apl.py`, `regime/test_norm.py` | assert scripts |
| `gpu_sim.py` | SFA state + graded APL in the step loop, + normalisation of the weight tensor at construction; all three default-off until Task 7 |
| `learn/plastic.py` | `signed_values`/`push` reapply the normalisation scale (Task 5b) |
| `results/regime_p0.json`, `results/regime_p1.json` | measurements (gitignored) |
| `docs/superpowers/regime-p0/` | committed copies of the P0 findings |
| `WHAT_IS_REAL.md` | second correction section (Task 7) |

---

### Task 0: Scaffold and the reachability audit (P0.1)

**Files:**
- Create: `regime/__init__.py` (empty), `regime/reach.py`, `regime/test_reach.py`

- [ ] **Step 1: Write the failing test**

`regime/test_reach.py`:
```python
"""Reachability over the CSR graph: hops, disjoint paths, bottlenecks."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from regime import reach as R

# toy: 0 -> 1 -> 3, 0 -> 2 -> 3, so 0 reaches 3 in 2 hops by two disjoint paths
indptr = np.array([0, 2, 3, 4, 4], np.int64)
indices = np.array([1, 2, 3, 3], np.int64)
data = np.array([5.0, 5.0, 5.0, 5.0], np.float32)
g = R.Graph(indptr, indices, data)
assert g.hops(np.array([0]), np.array([3])) == 2, g.hops(np.array([0]), np.array([3]))
assert g.hops(np.array([3]), np.array([0])) is None, "3 must not reach 0"
assert g.n_disjoint(np.array([0]), np.array([3])) == 2
# removing both middles disconnects; removing one does not
assert set(g.bottlenecks(np.array([0]), np.array([3]))) == set(), "no single cut vertex here"
indptr2 = np.array([0, 1, 2, 2], np.int64)          # 0 -> 1 -> 2, 1 is a cut vertex
indices2 = np.array([1, 2], np.int64)
g2 = R.Graph(indptr2, indices2, np.array([5.0, 5.0], np.float32))
assert g2.bottlenecks(np.array([0]), np.array([2])) == [1], g2.bottlenecks(np.array([0]), np.array([2]))
assert g2.hops(np.array([0]), np.array([2])) == 2
# weight along the best path is reported
w = g2.best_path_weight(np.array([0]), np.array([2]))
assert abs(w - 10.0) < 1e-6, w
print("ok reach: hops, disjoint paths, bottleneck, weight")
```

- [ ] **Step 2: Run it, expect failure**

Run: `$P regime/test_reach.py`
Expected: `ModuleNotFoundError: No module named 'regime'`. Create the empty `regime/__init__.py`, rerun, expect `ImportError` for `reach`.

- [ ] **Step 3: Implement `regime/reach.py`**

```python
"""Reachability between neuron sets over the connectome, as a plain graph.

READ ONLY. Answers one question for P0.1: is there a path at all from the
sensory neurons to the target, how long, how redundant, and what would cut it.
Signs are ignored here - an inhibitory path is still a path. Weight is the
sum of |synapse count| along the path, reported for scale, not for dynamics.
"""
from collections import deque

import numpy as np


class Graph:
    def __init__(self, indptr, indices, data):
        self.indptr = np.asarray(indptr, np.int64)
        self.indices = np.asarray(indices, np.int64)
        self.data = np.asarray(data, np.float32)
        self.n = len(self.indptr) - 1

    def _succ(self, i, blocked=None):
        s = self.indices[self.indptr[i]:self.indptr[i + 1]]
        return s if blocked is None else s[~blocked[s]]

    def bfs(self, src, dst, blocked=None):
        """Shortest path src-set -> dst-set as a list of node indices, or None."""
        dst = set(int(x) for x in dst)
        prev = {}
        seen = np.zeros(self.n, bool)
        q = deque()
        for s in src:
            s = int(s)
            if blocked is not None and blocked[s]:
                continue
            seen[s] = True
            q.append(s)
            if s in dst:
                return [s]
        while q:
            i = q.popleft()
            for j in self._succ(i, blocked):
                j = int(j)
                if seen[j]:
                    continue
                seen[j] = True
                prev[j] = i
                if j in dst:
                    path = [j]
                    while path[-1] in prev:
                        path.append(prev[path[-1]])
                    return path[::-1]
                q.append(j)
        return None

    def hops(self, src, dst):
        p = self.bfs(src, dst)
        return None if p is None else len(p) - 1

    def best_path_weight(self, src, dst):
        """Summed |weight| along the shortest path, or None."""
        p = self.bfs(src, dst)
        if p is None:
            return None
        tot = 0.0
        for a, b in zip(p, p[1:]):
            s = slice(self.indptr[a], self.indptr[a + 1])
            hit = self.indices[s] == b
            tot += float(np.abs(self.data[s][hit]).sum())
        return tot

    def n_disjoint(self, src, dst, cap=8):
        """Greedy vertex-disjoint path count: find a path, block its interior, repeat."""
        blocked = np.zeros(self.n, bool)
        k = 0
        while k < cap:
            p = self.bfs(src, dst, blocked)
            if p is None:
                return k
            for node in p[1:-1]:
                blocked[node] = True
            k += 1
        return k

    def bottlenecks(self, src, dst, cap=32):
        """Interior nodes whose individual removal disconnects src from dst."""
        p = self.bfs(src, dst)
        if p is None or len(p) <= 2:
            return []
        out = []
        for node in p[1:-1][:cap]:
            blocked = np.zeros(self.n, bool)
            blocked[node] = True
            if self.bfs(src, dst, blocked) is None:
                out.append(int(node))
        return out


def load(brain="data/brain_gpu.npz"):
    b = np.load(brain, allow_pickle=False)
    return Graph(b["W_indptr"], b["W_indices"], b["W_data"])
```

- [ ] **Step 4: Run test, expect pass**

Run: `$P regime/test_reach.py`
Expected: `ok reach: hops, disjoint paths, bottleneck, weight`

- [ ] **Step 5: Run the real audit and record it**

```bash
$P - <<'EOF'
import json, numpy as np
from regime import reach as R
m = np.load("data/neuron_meta.npz", allow_pickle=False)
ct = m["cell_type"].astype(str); cc = m["cell_class"].astype(str)
csc = m["cell_sub_class"].astype(str); sc = m["super_class"].astype(str)
g = R.load()
sets = {
    "JO": np.flatnonzero(np.char.startswith(ct, "JO")),
    "sugar_GRN": np.flatnonzero(csc == "sugar/water"),
    "bitter_GRN": np.flatnonzero(csc == "bitter"),
    "KC": np.flatnonzero(cc == "Kenyon_Cell"),
    "PAM": np.flatnonzero(np.char.startswith(ct, "PAM")),
    "PPL1": np.flatnonzero(np.char.startswith(ct, "PPL1")),
}
out = {k: int(len(v)) for k, v in sets.items()}
pairs = [("JO", "KC"), ("sugar_GRN", "PAM"), ("bitter_GRN", "PPL1")]
res = {}
for a, b in pairs:
    if len(sets[a]) == 0 or len(sets[b]) == 0:
        res["%s->%s" % (a, b)] = {"error": "empty set"}
        continue
    res["%s->%s" % (a, b)] = {
        "hops": g.hops(sets[a], sets[b]),
        "weight": g.best_path_weight(sets[a], sets[b]),
        "n_disjoint": g.n_disjoint(sets[a], sets[b]),
        "bottlenecks": g.bottlenecks(sets[a], sets[b]),
    }
    print(a, "->", b, res["%s->%s" % (a, b)])
json.dump({"set_sizes": out, "paths": res}, open("results/regime_p0_reach.json", "w"), indent=1)
EOF
```
Expected: three lines, each with an integer `hops` or `null`. RECORD them. If `JO` resolves to 0 neurons, find the actual auditory cell-type prefix by printing `sorted(set(ct[sc == "sensory"]))` and use that instead - do NOT leave the set empty and call the path dead.

**GATE:** `hops: null` for `JO->KC` means P3 is cancelled and the authored fingerprint is permanent. `hops: null` for `sugar_GRN->PAM` means P2 is cancelled and direct-DAN injection is permanent. Write whichever applies into the report in Task 3.

- [ ] **Step 6: Commit**

```bash
git add regime/__init__.py regime/reach.py regime/test_reach.py
git commit -m "feat(regime): reachability audit over the connectome graph" -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```
`results/` is gitignored, so the JSON does not commit.

---

### Task 1: Weight-distribution diagnostic (P0.2)

**Files:**
- Create: `regime/weights.py`, `regime/test_weights.py`

- [ ] **Step 1: Write the failing test**

`regime/test_weights.py`:
```python
"""Per-neuron input weight totals, split by sign."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from regime import weights as W

# 3 neurons. 0 -> 2 (+5), 1 -> 2 (+3), 1 -> 0 (-4)
indptr = np.array([0, 1, 3, 3], np.int64)
indices = np.array([2, 2, 0], np.int64)
data = np.array([5.0, 3.0, -4.0], np.float32)
exc, inh, deg = W.input_totals(indptr, indices, data, 3)
assert np.allclose(exc, [0.0, 0.0, 8.0]), exc
assert np.allclose(inh, [-4.0, 0.0, 0.0]), inh
assert np.array_equal(deg, [1, 0, 2]), deg
# bimodality: a set concentrated at the top of the excitatory distribution
q = W.percentile_of(exc, np.array([2]))
assert q[0] == 100.0, q
print("ok weights: totals by sign, in-degree, percentile lookup")
```

- [ ] **Step 2: Run it, expect failure**

Run: `$P regime/test_weights.py`
Expected: `ImportError: cannot import name 'weights'`.

- [ ] **Step 3: Implement `regime/weights.py`**

```python
"""Per-neuron summed input weight, split by sign.

READ ONLY. P0.2 asks whether a uniform W_SYN produces an explode-or-die
structure: do the neurons that ignite sit in a high-convergence tail of the
excitatory-input distribution, and do the dead pathways sit in a low one?
"""
import numpy as np


def input_totals(indptr, indices, data, n):
    """Returns (exc_sum[n], inh_sum[n], in_degree[n]). Rows are presynaptic."""
    indptr = np.asarray(indptr, np.int64)
    data = np.asarray(data, np.float32)
    indices = np.asarray(indices, np.int64)
    exc = np.zeros(n, np.float64)
    inh = np.zeros(n, np.float64)
    deg = np.zeros(n, np.int64)
    pos = data > 0
    np.add.at(exc, indices[pos], data[pos])
    np.add.at(inh, indices[~pos], data[~pos])
    np.add.at(deg, indices, 1)
    return exc, inh, deg


def percentile_of(values, idx):
    """Percentile rank of each idx within `values`, 0-100."""
    values = np.asarray(values, np.float64)
    order = np.argsort(values, kind="stable")
    rank = np.empty(len(values), np.float64)
    rank[order] = np.arange(len(values), dtype=np.float64)
    return 100.0 * rank[np.asarray(idx, np.int64)] / max(len(values) - 1, 1)


def load(brain="data/brain_gpu.npz"):
    b = np.load(brain, allow_pickle=False)
    n = int(b["n_neurons"])
    return input_totals(b["W_indptr"], b["W_indices"], b["W_data"], n)
```

- [ ] **Step 4: Run test, expect pass**

Run: `$P regime/test_weights.py`
Expected: `ok weights: totals by sign, in-degree, percentile lookup`

- [ ] **Step 5: Run the real diagnostic**

```bash
$P - <<'EOF'
import json, numpy as np
from regime import weights as W
exc, inh, deg = W.load()
m = np.load("data/neuron_meta.npz", allow_pickle=False)
cc = m["cell_class"].astype(str); csc = m["cell_sub_class"].astype(str)
ct = m["cell_type"].astype(str)
groups = {"KC": cc == "Kenyon_Cell", "PAM": np.char.startswith(ct, "PAM"),
          "PPL1": np.char.startswith(ct, "PPL1"), "APL": ct == "APL",
          "MBON": np.char.startswith(ct, "MBON")}
out = {"quantiles": {str(q): float(np.percentile(exc, q)) for q in (1, 5, 25, 50, 75, 95, 99)},
       "ei_ratio_median": float(np.median(exc[exc > 0]) / abs(np.median(inh[inh < 0]))),
       "n_zero_input": int((deg == 0).sum())}
for k, mask in groups.items():
    i = np.flatnonzero(mask)
    out[k] = {"n": len(i), "exc_median": float(np.median(exc[i])),
              "exc_pct_rank_median": float(np.median(W.percentile_of(exc, i))),
              "in_degree_median": float(np.median(deg[i]))}
    print("%-5s n=%5d exc_median=%10.1f pct_rank=%5.1f in_deg=%6.1f" % (
        k, out[k]["n"], out[k]["exc_median"], out[k]["exc_pct_rank_median"], out[k]["in_degree_median"]))
print("exc quantiles:", {k: round(v, 1) for k, v in out["quantiles"].items()})
print("E/I median ratio: %.3f  neurons with zero input: %d" % (out["ei_ratio_median"], out["n_zero_input"]))
json.dump(out, open("results/regime_p0_weights.json", "w"), indent=1)
EOF
```
Expected: one line per group plus the quantiles. RECORD all of it.

**Interpretation to write down in Task 3, not to act on here:** if KC's excitatory percentile rank is far above the median while PAM's is far below, the explode-or-die hypothesis is supported and Task 3 records that weight calibration may need to become the mechanism. If both sit near the median, the hypothesis is weakened and P1's two mechanisms stand as designed.

- [ ] **Step 6: Commit**

```bash
git add regime/weights.py regime/test_weights.py
git commit -m "feat(regime): per-neuron input weight distribution diagnostic" -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 2: Regime characterisation probe (P0.3)

**Files:**
- Create: `regime/probe.py`

This module is reused unchanged by every P1 measurement, so its interface is the one thing in this plan that must not drift.

- [ ] **Step 1: Implement `regime/probe.py`**

```python
"""Drive ORN channels and report the regime. Used by P0.3 and by every P1 run.

One call = one 300 ms batch. `channels` are selector strings already exported
in brain_gpu.npz (e.g. "cell_type=ORN_DA1,side=left"); `measure` drives each
channel separately in one batch and reports sparsity, overlap and ignition.
"""
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _HERE)
import gpu_sim as G

META = os.path.join(_HERE, "data", "neuron_meta.npz")
_meta = np.load(META, allow_pickle=False)
CC = _meta["cell_class"].astype(str)
CT = _meta["cell_type"].astype(str)
CEN = _meta["super_class"].astype(str) == "central"
KC = CC == "Kenyon_Cell"
ACTIVE_HZ = 1.0


def drive(sim, selector, hz):
    idx = sim.net.select(selector).astype(np.int64)
    return idx, np.full(len(idx), hz * G.DT / 1000.0, np.float64)


def measure(sim, selectors, hz, t_run=300.0, seed0=0):
    """Returns (rates [C, N] float32, report dict). One drive per selector."""
    drives = [drive(sim, s, hz) for s in selectors]
    counts = sim.run_batch(drives, list(range(seed0, seed0 + len(drives))), t_run=t_run)
    rates = counts.cpu().numpy().astype(np.float32) / (t_run / 1000.0)
    act = rates > ACTIVE_HZ
    rep = {"hz": hz, "selectors": list(selectors),
           "kc_active": [float(a[KC].mean()) for a in act],
           "kc_hz": [float(r[KC].mean()) for r in rates],
           "central_active": [float(a[CEN].mean()) for a in act],
           "apl_hz": [float(r[CT == "APL"].mean()) for r in rates]}
    rep["jaccard"] = jaccard_matrix(act[:, KC])
    return rates, rep


def jaccard_matrix(act_kc):
    """act_kc: bool [C, n_kc] -> C x C list of lists."""
    C = act_kc.shape[0]
    out = [[0.0] * C for _ in range(C)]
    for i in range(C):
        for j in range(C):
            inter = int((act_kc[i] & act_kc[j]).sum())
            union = int((act_kc[i] | act_kc[j]).sum())
            out[i][j] = float(inter / union) if union else 0.0
    return out


def pair_jaccard(rep):
    """The off-diagonal Jaccard for a two-selector measure()."""
    return rep["jaccard"][0][1]
```

- [ ] **Step 2: Smoke it and run the P0.3 sweep**

```bash
$P - <<'EOF'
import json, numpy as np, gpu_sim as G
from regime import probe as PR
sim = G.GpuSim()
sels = ["cell_type=ORN_DA1,side=left", "cell_type=ORN_DM4,side=left"]
rows = []
for hz in (10, 20, 30, 40, 50, 60, 80, 120):
    _, rep = PR.measure(sim, sels, hz)
    rows.append(rep)
    print("hz %3d  KC %.3f / %.3f  central %.3f / %.3f  jac %.3f" % (
        hz, rep["kc_active"][0], rep["kc_active"][1],
        rep["central_active"][0], rep["central_active"][1], PR.pair_jaccard(rep)))
json.dump(rows, open("results/regime_p0_sweep.json", "w"), indent=1)
EOF
```
Expected: KC active near 0 at low rates, jumping to roughly 0.33 and Jaccard near 0.94 once it ignites. RECORD the transition rate and how sharp it is (how many Hz between "nothing" and "ignited"). If either selector raises `KeyError`, list the exported ORN selectors with `[s for s in sim.net._sel if "ORN" in s][:20]` and use two that exist.

- [ ] **Step 3: Commit**

```bash
git add regime/probe.py
git commit -m "feat(regime): ORN drive probe reporting sparsity, overlap and ignition" -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 3: P0 findings report and the gate decision

**Files:**
- Create: `docs/superpowers/regime-p0/findings.md`

- [ ] **Step 1: Write the report**

Write `docs/superpowers/regime-p0/findings.md` containing, with the real numbers from Tasks 0-2:
- the three reachability results (hops, disjoint paths, summed weight, bottlenecks) and, for each of P2 and P3, an explicit **PROCEED** or **CANCELLED** line
- the weight-distribution table, and an explicit verdict sentence on the explode-or-die hypothesis: supported, weakened, or ambiguous, with the KC and PAM percentile ranks as the evidence
- the transition rate and sharpness from the P0.3 sweep
- if the hypothesis is SUPPORTED, a line stating that weight calibration is now a candidate mechanism for P1 and that the spec's risk section anticipated this

ASCII only. Terse and factual, matching the register of `docs/superpowers/condition-v1/condition_v1_notes.md`.

- [ ] **Step 2: Commit**

```bash
git add docs/superpowers/regime-p0/findings.md
git commit -m "docs(regime): P0 findings and the P2/P3 gate decision" -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

**STOP HERE if the P0.2 verdict is SUPPORTED.** Report to the human before starting Task 4: the spec says P1 is re-scoped before implementation, not after, if convergence turns out to be the story.

---

### Task 4: Spike-frequency adaptation

**Files:**
- Modify: `gpu_sim.py` - constants near `W_SYN`, `_body`, `_step_counter`, `_step_mask`, `run_batch`
- Create: `regime/test_sfa.py`

Default OFF (`SFA_B_INC = 0.0`), so this task must leave existing behaviour bit-identical.

- [ ] **Step 1: Write the failing test**

`regime/test_sfa.py`:
```python
"""Adaptation current: decays, accumulates on spikes, and is off by default."""
import os, sys
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gpu_sim as G

assert G.SFA_B_INC == 0.0, "SFA must ship default-off until Task 7"
decay_a = float(np.float32(np.exp(-G.DT / G.TAU_A)))
z8 = torch.tensor(0, dtype=torch.int8)
one = torch.tensor(1.0)

# one neuron, one batch column, no input, forced to spike by a Poisson mask
v = torch.full((1, 1), G.V_REST); g = torch.zeros((1, 1)); a = torch.zeros((1, 1))
r = torch.zeros((1, 1), dtype=torch.int8); prob = torch.zeros((1, 1), dtype=torch.float64)
no_spike = torch.zeros((1, 1), dtype=torch.bool)
spike_now = torch.ones((1, 1), dtype=torch.bool)
v, g, r, a, spk = G._body(v, g, r, a, prob, spike_now, no_spike, G.V_REST,
                          float(np.float32(np.exp(-G.DT / G.TAU_M))),
                          float(np.float32(1.0 - np.exp(-G.DT / G.TAU_M))),
                          float(np.float32(np.exp(-G.DT / G.TAU_SYN))), decay_a, 0.5,
                          G.V_TH, torch.tensor(G.V_RESET), torch.tensor(21, dtype=torch.int8), z8)
assert bool(spk[0, 0]), "forced spike did not fire"
assert abs(float(a[0, 0]) - 0.5) < 1e-6, float(a[0, 0])   # incremented, not yet decayed

# next step, no spike: decays by exactly decay_a
no_poisson = torch.zeros((1, 1), dtype=torch.bool)
v, g, r, a2, spk = G._body(v, g, r, a, prob, no_poisson, no_spike, G.V_REST,
                           float(np.float32(np.exp(-G.DT / G.TAU_M))),
                           float(np.float32(1.0 - np.exp(-G.DT / G.TAU_M))),
                           float(np.float32(np.exp(-G.DT / G.TAU_SYN))), decay_a, 0.5,
                           G.V_TH, torch.tensor(G.V_RESET), torch.tensor(21, dtype=torch.int8), z8)
assert abs(float(a2[0, 0]) - 0.5 * decay_a) < 1e-6, float(a2[0, 0])

# no_spike mask suppresses threshold crossing
v3 = torch.full((1, 1), G.V_TH + 5.0); a3 = torch.zeros((1, 1))
g3 = torch.zeros((1, 1)); r3 = torch.zeros((1, 1), dtype=torch.int8)
_, _, _, _, spk3 = G._body(v3, g3, r3, a3, prob, no_poisson, torch.ones((1, 1), dtype=torch.bool),
                           G.V_REST, float(np.float32(np.exp(-G.DT / G.TAU_M))),
                           float(np.float32(1.0 - np.exp(-G.DT / G.TAU_M))),
                           float(np.float32(np.exp(-G.DT / G.TAU_SYN))), decay_a, 0.0,
                           G.V_TH, torch.tensor(G.V_RESET), torch.tensor(21, dtype=torch.int8), z8)
assert not bool(spk3[0, 0]), "no_spike mask did not suppress the spike"
print("ok sfa: increment, decay, no_spike mask, default off")
```

- [ ] **Step 2: Run it, expect failure**

Run: `$P regime/test_sfa.py`
Expected: `AttributeError: module 'gpu_sim' has no attribute 'SFA_B_INC'`.

- [ ] **Step 3: Add the constants**

In `gpu_sim.py`, immediately after the `W_SYN = 0.275` line, add:
```python

# AUTHORED, not Shiu. Spike-frequency adaptation and a graded APL, added
# 2026-09-19 to stop brain-wide runaway. Both default OFF so that every result
# before that date reproduces exactly; Task 7 of the regime plan flips them.
# See docs/superpowers/specs/2026-09-19-regime-stability-design.md.
TAU_A = 100.0       # ms, adaptation decay
SFA_B_INC = 0.0     # mV-equivalent added to the adaptation current per spike
APL_GRADED = False  # replace APL's spiking output with a graded rate
APL_SCALE = 7.0     # mV above rest at which graded APL output saturates
```

- [ ] **Step 4: Rewrite `_body` to carry the adaptation state**

Replace the whole `_body` function with:
```python
def _body(v, g, r, a, prob, poisson, no_spike, v_rest, decay_m, gain, decay_s,
          decay_a, b_inc, v_th, v_reset, refr_m1, zero_i8):
    """Steps 2-7 of run_trial over the whole [N, B] state, in one pass.

    `prob` is the per-step Poisson probability (rate * dt / 1000) and doubles as
    flypoke's `is_stim`: every stimulated neuron has a strictly positive rate, so
    `prob > 0` is exactly the `is_stim` mask and no second array is needed.

    `a` is the AUTHORED adaptation current, subtracted from the synaptic drive
    and incremented on each spike. With b_inc = 0 it stays zero for every step
    and `(g - a)` is `g`, so the arithmetic is unchanged from the Shiu port.
    `no_spike` is the AUTHORED mask for neurons whose output is delivered by
    some other mechanism (graded APL); it only gates threshold crossing.
    """
    active = r == zero_i8
    vn = torch.where(active, v_rest + (v - v_rest) * decay_m + (g - a) * gain, v)
    g = g * decay_s
    a = a * decay_a
    fired = (vn > v_th) & active & (prob <= 0.0) & ~no_spike
    spk = fired | poisson
    v = torch.where(spk, v_reset, vn)
    g = torch.where(spk, torch.zeros((), dtype=g.dtype, device=g.device), g)
    a = torch.where(spk, a + b_inc, a)
    r = torch.where(spk, refr_m1, torch.clamp(r - 1, min=0))
    return v, g, r, a, spk
```

- [ ] **Step 5: Thread the new arguments through both step wrappers**

Replace `_step_counter` and `_step_mask` with:
```python
def _step_counter(v, g, r, a, prob, seed32, step_const, neuron, no_spike, v_rest,
                  decay_m, gain, decay_s, decay_a, b_inc, v_th, v_reset, refr_m1,
                  zero_i8):
    """Throughput path: the Poisson draw is computed in-register, not stored."""
    poisson = _uniform(seed32, step_const, neuron) < prob
    return _body(v, g, r, a, prob, poisson, no_spike, v_rest, decay_m, gain,
                 decay_s, decay_a, b_inc, v_th, v_reset, refr_m1, zero_i8)


def _step_mask(v, g, r, a, prob, poisson, no_spike, v_rest, decay_m, gain,
               decay_s, decay_a, b_inc, v_th, v_reset, refr_m1, zero_i8):
    """Verification path: the Poisson mask came from numpy's PCG64 stream."""
    return _body(v, g, r, a, prob, poisson, no_spike, v_rest, decay_m, gain,
                 decay_s, decay_a, b_inc, v_th, v_reset, refr_m1, zero_i8)
```

- [ ] **Step 6: Allocate the state in `run_batch` and pass it**

In `run_batch`, after the line `decay_s = float(np.float32(np.exp(-DT / TAU_SYN)))`, add:
```python
        decay_a = float(np.float32(np.exp(-DT / TAU_A)))
```
After the line `r = torch.zeros((N, B), dtype=torch.int8, device=dev)`, add:
```python
        a = torch.zeros((N, B), dtype=torch.float32, device=dev)
        no_spike = torch.zeros((N, 1), dtype=torch.bool, device=dev)
```
Replace the counter-path call with:
```python
                v, g, r, a, spk = step_fn(v, g, r, a, prob, seed32,
                                          step_consts[step], neuron, no_spike,
                                          V_REST, decay_m, gain, decay_s,
                                          decay_a, SFA_B_INC, V_TH, v_reset_t,
                                          refr_m1, zero_i8)
```
and the mask-path call with:
```python
                v, g, r, a, spk = step_fn(v, g, r, a, prob, poisson_buf,
                                          no_spike, V_REST, decay_m, gain,
                                          decay_s, decay_a, SFA_B_INC, V_TH,
                                          v_reset_t, refr_m1, zero_i8)
```

- [ ] **Step 7: Run the test, expect pass**

Run: `$P regime/test_sfa.py`
Expected: `ok sfa: increment, decay, no_spike mask, default off`

- [ ] **Step 8: Prove the refactor changed nothing**

```bash
$P - <<'EOF'
import numpy as np, gpu_sim as G, chess
sim = G.GpuSim()
d = G.drive_of(chess.Board(), sim.net)
c1 = sim.run_batch([d]*4, [0,1,2,3]).cpu().numpy()
c2 = sim.run_batch([d]*4, [0,1,2,3]).cpu().numpy()
assert (c1 == c2).all(), "not deterministic"
assert (c1[2] == sim.run_batch([d], [2]).cpu().numpy()[0]).all(), "not batch invariant"
cen = np.load("data/neuron_meta.npz")["super_class"].astype(str) == "central"
print("ok central>1Hz %.3f" % ((c1[0][cen]/0.3 > 1).mean()))
EOF
$P learn/condition.py --state v1 --check
```
Expected: `ok central>1Hz 0.172` - **the same 0.172 measured before this task** - and the v1 acceptance still printing `ACCEPT`. If either changed, the SFA-off path is not a no-op; fix it before continuing. Do not proceed with a different number.

- [ ] **Step 9: Commit**

```bash
git add gpu_sim.py regime/test_sfa.py
git commit -m "feat(sim): spike-frequency adaptation state, default off" -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 5: Graded non-spiking APL

**Files:**
- Modify: `gpu_sim.py` - `GpuSim.__init__`, `run_batch`
- Create: `regime/test_apl.py`

- [ ] **Step 1: Write the failing test**

`regime/test_apl.py`:
```python
"""Graded APL: its outgoing weights are delivered in proportion to depolarisation."""
import os, sys
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gpu_sim as G

assert G.APL_GRADED is False, "graded APL must ship default-off until Task 7"
sim = G.GpuSim()
assert sim.apl_idx.tolist() == [17917, 67229], sim.apl_idx.tolist()
assert sim.apl_tgt.numel() == sim.apl_w.numel()
assert sim.apl_tgt.numel() == 2763 + 2761, sim.apl_tgt.numel()
assert float(sim.apl_w.sum()) < 0.0, "APL output must be inhibitory"

# activation: 0 at rest, 1 at APL_SCALE above rest, clamped above that
v = torch.tensor([[G.V_REST], [G.V_REST + G.APL_SCALE]], device=sim.device)
act = sim._apl_activation(v)
assert abs(float(act[0, 0]) - 0.0) < 1e-6, float(act[0, 0])
assert abs(float(act[1, 0]) - 1.0) < 1e-6, float(act[1, 0])
v2 = torch.tensor([[G.V_REST - 5.0], [G.V_REST + 100.0]], device=sim.device)
act2 = sim._apl_activation(v2)
assert float(act2[0, 0]) == 0.0 and float(act2[1, 0]) == 1.0, "activation not clamped to [0,1]"

# delivery scales linearly and lands on APL's targets only
g = torch.zeros((sim.net.n, 1), dtype=torch.float32, device=sim.device)
sim._deliver_apl(g, torch.full((2, 1), 1.0, device=sim.device))
full = g.clone()
assert float(full.sum()) < 0.0
g.zero_()
sim._deliver_apl(g, torch.full((2, 1), 0.5, device=sim.device))
assert torch.allclose(g, full * 0.5, atol=1e-5), "delivery is not linear in activation"
touched = (full != 0).squeeze(1).nonzero().squeeze(1)
assert set(touched.tolist()) <= set(sim.apl_tgt.tolist()), "delivered outside APL targets"

# left and right APL share targets, so the scatter has duplicate indices and
# must be deterministic: same input, same output, every time.
g.zero_(); sim._deliver_apl(g, torch.full((2, 1), 1.0, device=sim.device))
once = g.clone()
for _ in range(5):
    g.zero_(); sim._deliver_apl(g, torch.full((2, 1), 1.0, device=sim.device))
    assert torch.equal(g, once), "APL scatter is not bit-reproducible"
assert sim.apl_tgt.unique().numel() < sim.apl_tgt.numel(), \
    "expected shared targets between left and right APL; if this ever fails the " \
    "determinism guard in _deliver_apl is no longer load-bearing, not that it is wrong"
print("ok apl: indices, activation curve, linear inhibitory delivery, determinism")
```

- [ ] **Step 2: Run it, expect failure**

Run: `$P regime/test_apl.py`
Expected: `AttributeError: 'GpuSim' object has no attribute 'apl_idx'`.

- [ ] **Step 3: Precompute APL's outgoing slice in `GpuSim.__init__`**

At the end of `GpuSim.__init__`, after the existing body, add:
```python
        # AUTHORED: graded APL. CSR rows are presynaptic, so APL's outgoing
        # edges are two contiguous slices. Cached once; the weights already
        # carry W_SYN so _deliver_apl matches _deliver's units.
        meta = np.load(os.path.join(_HERE, "data", "neuron_meta.npz"),
                       allow_pickle=False)
        ct = meta["cell_type"].astype(str)
        apl = np.flatnonzero(ct == "APL").astype(np.int64)
        ip, ix, w = self.net.W_indptr, self.net.W_indices, self.net.W_data
        tgt = np.concatenate([ix[ip[i]:ip[i + 1]] for i in apl])
        wts = np.concatenate([w[ip[i]:ip[i + 1]] for i in apl]) * np.float32(W_SYN)
        row = np.concatenate([np.full(ip[i + 1] - ip[i], k, np.int64)
                              for k, i in enumerate(apl)])
        self.apl_idx = torch.as_tensor(apl, device=self.device)
        self.apl_tgt = torch.as_tensor(tgt.astype(np.int64), device=self.device)
        self.apl_w = torch.as_tensor(wts.astype(np.float32), device=self.device)
        self.apl_row = torch.as_tensor(row, device=self.device)
```

- [ ] **Step 4: Add the two methods, right after `set_plastic`**

```python
    def _apl_activation(self, v_apl):
        """v_apl: [n_apl, B] -> graded output in [0, 1]. Real APL is non-spiking."""
        return torch.clamp((v_apl - V_REST) / APL_SCALE, 0.0, 1.0)

    def _deliver_apl(self, g, act):
        """Add APL's outgoing inhibition, scaled by `act` [n_apl, B], into g.

        Left and right APL share postsynaptic targets, so this scatter has
        duplicate indices within a single call and CUDA's float `index_add_`
        would resolve them with atomics in a run-varying order - the exact
        nondeterminism `_deliver` documents. Same guard, same reason.
        """
        contrib = self.apl_w.unsqueeze(1) * act[self.apl_row]
        if self.deterministic:
            torch.use_deterministic_algorithms(True)
            try:
                g.index_add_(0, self.apl_tgt, contrib)
            finally:
                torch.use_deterministic_algorithms(False)
        else:
            g.index_add_(0, self.apl_tgt, contrib)
```

- [ ] **Step 5: Wire it into the step loop**

In `run_batch`, replace the `no_spike` allocation from Task 4 with:
```python
        a = torch.zeros((N, B), dtype=torch.float32, device=dev)
        no_spike = torch.zeros((N, 1), dtype=torch.bool, device=dev)
        if APL_GRADED:
            no_spike[self.apl_idx] = True
```
and immediately after the `self._deliver(g, pre, bcol, B)` call inside the step loop, add:
```python
            if APL_GRADED:
                self._deliver_apl(g, self._apl_activation(v[self.apl_idx]))
```
Note this sits INSIDE the `for step` loop but OUTSIDE the `if pre.numel():` guard, so APL delivers on every step including those with no arriving spikes. Getting that nesting wrong is the likeliest bug in this task.

- [ ] **Step 6: Run the test, expect pass**

Run: `$P regime/test_apl.py`
Expected: `ok apl: indices, activation curve, linear inhibitory delivery`

- [ ] **Step 7: Prove the refactor still changed nothing**

Run the same script as Task 4 Step 8, plus `$P learn/condition.py --state v1 --check`.
Expected: `ok central>1Hz 0.172` and `ACCEPT`, unchanged. `APL_GRADED` is still False, so this must hold exactly.

- [ ] **Step 8: Commit**

```bash
git add gpu_sim.py regime/test_apl.py
git commit -m "feat(sim): graded non-spiking APL, default off" -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 5b: Homeostatic normalisation of incoming excitatory weight

**Files:**
- Modify: `gpu_sim.py` - constants near `W_SYN`, `GpuSim.__init__`; `learn/plastic.py` - `signed_values`, `push`
- Create: `regime/test_norm.py`

The spec's P1.3, added by the 2026-09-19 re-scope. P0.2 measured a 35x spread from
median (50) to p99 (1761) summed excitatory input under one uniform `W_SYN`; this scales
each neuron's incoming EXCITATORY edges so the sum meets one target, leaving inhibition
alone. It is a transform over the flat weight tensor `sim.data` applied after
construction - the CSR layout, the offsets and `set_plastic` are untouched.

`NORM_EXC_TARGET` is **AUTHORED**, in the same register as `TAU_A`, `SFA_B_INC` and
`APL_SCALE`. Its on-value 13.75 mV is 50 synapses (the measured brain-wide median) times
`W_SYN = 0.275`, chosen so the median neuron is unchanged and only the tail moves. It is
not a published figure.

Default OFF (`NORM_EXC_TARGET = 0.0`), like the other two mechanisms, so this commit
still reproduces current behaviour bit-identically.

**The interaction that must be handled here, not discovered later.** `Plastic.push(sim)`
writes `signed_values(W_SYN)` - recomputed from `w0`, the raw npz weights - into 18,674
KC->MBON offsets of `sim.data`. Normalise, then push, and those 18,674 edges silently
revert to unnormalised values. `learn/condition.py` calls `push()` after every training
trial, so this lands directly on the spec's criterion 2. Three fixes were available:

1. apply the scale inside `Plastic.signed_values`;
2. hand `Plastic` an already-calibrated `w0`;
3. re-apply normalisation after every `push()`.

**Take (1).** `signed_values` is already the single place where `w()` becomes a
`sim.data` value - it is where `sign` and `W_SYN` are applied for exactly this reason -
so the per-neuron scale belongs beside them. (2) would corrupt the learning rule's
reference: `w0` is the real synapse count the depression rule and the `W_MIN` floor are
defined against, and the `real()` assert that reads it straight from the npz. (3) is
correct only as long as every future caller remembers, which is the failure mode being
fixed.

- [ ] **Step 1: Write the failing test**

`regime/test_norm.py`:
```python
"""Homeostatic normalisation of incoming excitatory weight, and the one
interaction that would silently undo it on 18,674 edges."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gpu_sim as G
from regime import weights as W
from learn.plastic import Plastic

assert G.NORM_EXC_TARGET == 0.0, "normalisation must ship default-off until Task 7"

sim = G.GpuSim()
assert sim.norm_scale.shape == (sim.net.n,), sim.norm_scale.shape
assert sim.norm_scale.dtype == np.float32, sim.norm_scale.dtype
assert (sim.norm_scale == 1.0).all(), "scale must be all ones while the target is 0.0"

# the scale maths, checked against regime/weights.py on the real matrix
ip, ix, wd, N = sim.net.W_indptr, sim.net.W_indices, sim.net.W_data, sim.net.n
exc, _, _ = W.input_totals(ip, ix, wd, N)
exc = exc * float(G.W_SYN)
target = 13.75
hit = exc > 0
want = np.ones(N, np.float32)
want[hit] = (target / exc[hit]).astype(np.float32)
got = G.GpuSim._exc_scale(sim.net, target)
assert np.allclose(got, want, rtol=1e-5), "scale is not target / summed excitatory input"
assert (got[~hit] == 1.0).all(), "neurons with no excitatory input must be left alone"
assert hit.sum() > 100000, int(hit.sum())

# applying it hits the target, and does not touch inhibition
pos = wd > 0
vals = (wd * np.float32(G.W_SYN)).astype(np.float32)
vals[pos] = vals[pos] * got[ix[pos]]
exc2, inh2, _ = W.input_totals(ip, ix, vals, N)
assert np.allclose(exc2[hit], target, rtol=1e-4), (exc2[hit].min(), exc2[hit].max())
_, inh0, _ = W.input_totals(ip, ix, (wd * np.float32(G.W_SYN)).astype(np.float32), N)
assert np.allclose(inh2, inh0, rtol=1e-5), "inhibitory input was rescaled"

# THE REGRESSION. With normalisation on, an untrained Plastic (dw = 0) must push
# values identical to what normalisation already wrote. Before the signed_values
# fix this reverts 18,674 edges to w0 * W_SYN.
p = Plastic.real()
assert p.n_edges == 18674, p.n_edges
G.NORM_EXC_TARGET = target
try:
    sim2 = G.GpuSim()
    off = np.asarray(p.offsets, np.int64)
    before = sim2.data.cpu().numpy()[off].copy()
    p.push(sim2)
    after = sim2.data.cpu().numpy()[off]
    n_bad = int((np.abs(before - after) > 1e-6).sum())
    assert n_bad == 0, "push() reverted %d normalised edges" % n_bad
    assert not np.allclose(before, p.signed_values(G.W_SYN)), \
        "normalisation did not change the plastic edges at all - the test is vacuous"
finally:
    G.NORM_EXC_TARGET = 0.0
print("ok norm: scale maths, inhibition untouched, default off, survives Plastic.push")
```

- [ ] **Step 2: Run it, expect failure**

Run: `$P regime/test_norm.py`
Expected: `AttributeError: module 'gpu_sim' has no attribute 'NORM_EXC_TARGET'`.

- [ ] **Step 3: Add the constant**

In `gpu_sim.py`, immediately after the `APL_SCALE = 7.0` line added in Task 4, add:
```python
NORM_EXC_TARGET = 0.0   # mV of summed incoming EXCITATORY weight per neuron; 0.0 = off.
                        # AUTHORED. 13.75 = 50 synapses (the P0.2 brain-wide median)
                        # times W_SYN, so the median neuron is unchanged and only the
                        # heavy tail moves. Not a published figure.
```

- [ ] **Step 4: Compute and apply the scale in `GpuSim`**

Add this static method to `GpuSim`, immediately after `set_plastic`:
```python
    @staticmethod
    def _exc_scale(net, target):
        """Per-neuron factor bringing summed incoming EXCITATORY weight to `target` mV.

        CSR rows are PRESYNAPTIC and W_indices are POSTSYNAPTIC, so incoming
        excitatory weight is a scatter-add over W_indices restricted to positive
        entries - the same computation as regime/weights.py:input_totals.
        Neurons with no excitatory input get 1.0: there is nothing to normalise
        and target/0 would put a NaN in the weight tensor.
        """
        scale = np.ones(net.n, np.float32)
        if target <= 0.0:
            return scale
        pos = net.W_data > 0
        exc = np.zeros(net.n, np.float64)
        np.add.at(exc, net.W_indices[pos].astype(np.int64), net.W_data[pos])
        exc *= float(W_SYN)
        hit = exc > 0
        scale[hit] = (float(target) / exc[hit]).astype(np.float32)
        return scale
```

At the very end of `GpuSim.__init__`, after the APL block added in Task 5, add:
```python
        # AUTHORED: homeostatic normalisation of incoming excitatory weight.
        # The scale is KEPT, not just applied, because learn/plastic.py rewrites
        # 18,674 of these edges from w0 and has to reapply it; see
        # Plastic.signed_values. It is float32[N] on the host, not a device
        # tensor: nothing in the step loop reads it.
        self.norm_scale = self._exc_scale(self.net, NORM_EXC_TARGET)
        if NORM_EXC_TARGET > 0.0:
            pos = self.net.W_data > 0
            vals = (self.net.W_data * np.float32(W_SYN)).astype(np.float32)
            vals[pos] = vals[pos] * self.norm_scale[self.net.W_indices[pos]]
            self.data = torch.from_numpy(vals).to(self.device)
```

- [ ] **Step 5: Make `Plastic` reapply the scale**

In `learn/plastic.py`, replace `signed_values` and `push` with:
```python
    def signed_values(self, w_syn, scale=None):
        """Values to write into sim.data. w0 holds magnitudes; sign carries the raw
        matrix's polarity - 5 of the 18,674 edges are negative in v783 (KCg-m onto
        MBON01/05/09/11), so it must be reapplied or push() would flip them.

        `scale` is gpu_sim's per-neuron homeostatic normalisation factor
        (float32[n_neurons], all ones when NORM_EXC_TARGET is 0.0), indexed by the
        POSTsynaptic MBON. Without it push() would rewrite these 18,674 edges from
        w0 and silently revert them to unnormalised values after every training
        trial. It applies to the excitatory edges only, because normalisation
        leaves inhibition alone.
        """
        v = self.w() * self.sign * np.float32(w_syn)
        if scale is not None:
            s = np.asarray(scale, np.float32)[self.mbon_of_edge]
            v = np.where(self.sign > 0, v * s, v)
        return v.astype(np.float32)

    def push(self, sim):
        import gpu_sim as G
        sim.set_plastic(self.offsets,
                        self.signed_values(G.W_SYN, getattr(sim, "norm_scale", None)))
```
`getattr` with a default keeps `Plastic` usable against a sim built before this task
and against `Plastic.toy()`, which has no sim at all.

- [ ] **Step 6: Run the test, expect pass**

Run: `$P regime/test_norm.py`
Expected: `ok norm: scale maths, inhibition untouched, default off, survives Plastic.push`

Then the existing plastic test, which must be unaffected:
Run: `$P learn/test_plastic.py`
Expected: its usual `ok ...` line, still reporting `real edges 18674`.

- [ ] **Step 7: Prove the change is still a no-op**

Run the same script as Task 4 Step 8, plus `$P learn/condition.py --state v1 --check`.
Expected: `ok central>1Hz 0.172` - the same 0.172 as before Task 4 - and `ACCEPT`.
`NORM_EXC_TARGET` is 0.0, so `norm_scale` is all ones, `sim.data` is rebuilt from
nothing, and `signed_values` multiplies by 1.0. If 0.172 moved, the off-path is not a
no-op; fix it before continuing.

- [ ] **Step 8: Commit**

```bash
git add gpu_sim.py learn/plastic.py regime/test_norm.py
git commit -m "feat(sim): homeostatic normalisation of incoming excitatory weight, default off" -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 6: Measure each mechanism alone, then together

**Files:**
- Create: `regime/measure.py`

The spec's risk section requires measuring each mechanism alone before measuring them together, or a passing result will not say which one mattered. With P1.3 that is three isolating arms (`sfa`, `apl`, `norm`) against the `off` control.

**Judgement on the `all` arm.** It is worth running, and it is in the list. The cheap
argument against it - that `both` plus the `norm` arm tells you what `all` would do - is
wrong here: normalisation changes the operating point the other two act on. Flattening
the tail changes how much there is for adaptation to damp and how hard APL is driven, so
the three do not superpose. The concrete reason is Task 7: it ships exactly one
configuration, and shipping a combination that was never measured is not an option.
Six arms at four rates is one extra sweep, which is cheap next to that.

- [ ] **Step 1: Write `regime/measure.py`**

```python
"""Criterion-1 evidence: sparsity and odour separation under each mechanism.

Five arms isolate the three mechanisms - off (the shipped regime, the control),
sfa, apl, norm - plus `both` (the two gain-control mechanisms) and `all`. Each
arm sets the gpu_sim constants in-process, builds a fresh GpuSim, and runs the
same ORN sweep. An arm is (SFA_B_INC, APL_GRADED, NORM_EXC_TARGET).

Run: .venv/Scripts/python.exe regime/measure.py
"""
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _HERE)
import gpu_sim as G
from regime import probe as PR

SELECTORS = ["cell_type=ORN_DA1,side=left", "cell_type=ORN_DM4,side=left"]
NORM = 13.75          # AUTHORED: 50 synapses (P0.2 median) * W_SYN. See Task 5b.
ARMS = {"off":  (0.0, False, 0.0),
        "sfa":  (0.5, False, 0.0),
        "apl":  (0.0, True,  0.0),
        "norm": (0.0, False, NORM),
        "both": (0.5, True,  0.0),
        "all":  (0.5, True,  NORM)}
RATES = (20, 40, 60, 120)


def run_arm(b_inc, graded, norm_target, rates=RATES):
    G.SFA_B_INC = b_inc
    G.APL_GRADED = graded
    G.NORM_EXC_TARGET = norm_target     # read by GpuSim.__init__, so set it first
    sim = G.GpuSim()
    out = []
    for hz in rates:
        _, rep = PR.measure(sim, SELECTORS, hz)
        rep["jac"] = PR.pair_jaccard(rep)
        out.append(rep)
        print("  hz %3d  KC %.3f  central %.3f  APL %.1f Hz  jac %.3f" % (
            hz, rep["kc_active"][0], rep["central_active"][0],
            rep["apl_hz"][0], rep["jac"]), flush=True)
    return out


def main():
    res = {}
    for name, (b_inc, graded, norm_target) in ARMS.items():
        print("arm", name, "SFA_B_INC", b_inc, "APL_GRADED", graded,
              "NORM_EXC_TARGET", norm_target, flush=True)
        res[name] = run_arm(b_inc, graded, norm_target)
    os.makedirs(os.path.join(_HERE, "results"), exist_ok=True)
    json.dump(res, open(os.path.join(_HERE, "results", "regime_p1.json"), "w"), indent=1)
    print("wrote results/regime_p1.json")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it**

Run: `$P regime/measure.py`
Expected: six arms x four rates. The `off` arm must reproduce KC ~0.33, central ~0.16, jac ~0.94 at 60 Hz - that is the known shipped regime and is the control. RECORD the full table.

- [ ] **Step 3: Tune the three constants against criterion 1**

If the best arm at 60 Hz has not moved KC active materially below 0.33 and Jaccard materially below 0.944, re-run `run_arm` with `SFA_B_INC` in (1.0, 2.0, 4.0), `APL_SCALE` in (3.0, 7.0, 14.0) and `NORM_EXC_TARGET` in (6.875, 13.75, 27.5) - half, median and double the P0.2 median. Log EVERY value tried, with its KC/central/Jaccard, into `results/regime_p1_notes.md`.

Three failure modes to name explicitly rather than accept:
- **Silencing.** KC active near 0 with both channels dead is a FAILURE of criterion 1, not a pass. Report it as such.
- **Flattening.** The `norm` arm reaching low KC active while Jaccard stays high is normalisation erasing the code along with the tail, not a pass. Jaccard is the discriminator.
- **No movement.** If no arm and no setting moves Jaccard below ~0.8, stop and report. That is the spec's stated risk, now with normalisation also excluded - and at that point the remaining candidate is the calibration route the spec rejected, which is a decision for the human, not a tuning step.

- [ ] **Step 4: Criterion 2 - does the regime survive its own learning?**

With the best setting from Step 3 held in `gpu_sim.py`'s constants, run the existing conditioning protocol and check the ignition bound on every trial:
```bash
$P learn/condition.py --state regime1 --train 30 --probe 5 --arms learn,lesion
$P - <<'EOF'
import json
rows = [json.loads(l) for l in open("results/condition_regime1.jsonl")]
worst = max(r["central_active"] for r in rows)
print("max central_active over %d trials: %.4f (bound 0.10)" % (len(rows), worst))
assert worst < 0.10, "IGNITED during training - criterion 2 failed"
print("criterion 2 PASS")
EOF
$P learn/condition.py --state regime1 --check
```
Expected: `criterion 2 PASS`, then the learn/lesion line and `ACCEPT`. If `--check` fails in the new regime, that is a real finding - report the numbers, do NOT weaken the assert and do NOT tune eta to rescue it.

This protocol calls `Plastic.push()` after every trial, so it is where an unhandled normalisation/`push` interaction would surface - as a slow drift no assert names. Task 5b's `regime/test_norm.py` is what catches it before this step, and it must still pass here.

- [ ] **Step 5: Criterion 3 - determinism and batch invariance**

Run the script from Task 4 Step 8 again. Both asserts must pass. The `central>1Hz` figure WILL now differ from 0.172 - that is the point of the change. Record the new value; Task 7 writes it into the docs.

- [ ] **Step 6: Commit**

```bash
git add regime/measure.py
git commit -m "feat(regime): six-arm measurement of SFA, graded APL and normalisation" -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```
`results/` is gitignored; copy `results/regime_p1_notes.md` to `docs/superpowers/regime-p0/` in Task 7 so the evidence is versioned.

---

### Task 7: Make it the default, archive the old brain

**Files:**
- Modify: `gpu_sim.py` (the three constants), `WHAT_IS_REAL.md`
- Create: `docs/superpowers/regime-p0/p1-results.md`

Do this task ONLY if Task 6 passed all three criteria. If it did not, stop and report; the spec's archival step is conditional on the mechanism working.

- [ ] **Step 1: Flip the defaults**

In `gpu_sim.py`, set `SFA_B_INC`, `APL_GRADED` and `NORM_EXC_TARGET` to the values that passed Task 6, and update the comment above them to say which are now ON by default and to name the commit that measured them.

- [ ] **Step 2: Copy the evidence into docs**

Copy `results/regime_p1_notes.md` to `docs/superpowers/regime-p0/p1-results.md`, adding a first line saying it is the authoritative copy and that `results/` is gitignored. Include the six-arm table, every tuned value tried, and the three criteria verdicts.

- [ ] **Step 3: Second correction section in `WHAT_IS_REAL.md`**

Insert after the existing "Correction, 2026-09-19: the connectome file was a bad build" section, in the same register:
- that the LIF now carries three AUTHORED mechanisms, spike-frequency adaptation, a graded non-spiking APL and homeostatic normalisation of incoming excitatory weight, with their constants and with whichever of them shipped ON
- that this is a change to the neuron model, so every number measured before it is superseded, including reels 1-3, the chess results and condition v1
- the old central-activation figure 0.172 and the new one from Task 6 Step 5
- that prior results are kept, marked superseded, and NOT re-run
- that real APL is non-spiking and the spiking model was the artifact being corrected

Also add a Trained/Authored bullet naming `TAU_A`, `SFA_B_INC`, `APL_SCALE` and `NORM_EXC_TARGET` as authored, in the same register as `eta`, `W_MIN` and the compartment table. Say plainly that normalisation changes 2,700,429 real synapse weights away from the measured counts: the counts remain the source, the per-neuron sum is authored.

- [ ] **Step 4: Verify the whole suite**

```bash
$P regime/test_reach.py && $P regime/test_weights.py && $P regime/test_sfa.py && $P regime/test_apl.py && $P regime/test_norm.py
$P learn/test_fingerprint.py && $P learn/test_plastic.py
```
Expected: all seven print their `ok ...` line. `learn/test_plastic.py` must still report `real edges 18674` - the weight tensor layout is untouched by this plan, and if that number moved, something in Task 5 wrote outside APL's slice or Task 5b's rescale hit the offsets.

- [ ] **Step 5: Commit**

```bash
git add gpu_sim.py learn/plastic.py WHAT_IS_REAL.md docs/superpowers/regime-p0/p1-results.md
git commit -m "feat(sim): graded APL and spike-frequency adaptation are now the default regime" -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

## Self-review

- **Spec coverage.** P0.1 Task 0; P0.2 Task 1; P0.3 Task 2; the P0 gate Task 3 (with an explicit STOP if the explode-or-die hypothesis is supported, per the spec's re-scope clause - it fired, and the spec's "Re-scope, 2026-09-19" section records the outcome); P1.1 graded APL Task 5; P1.2 SFA Task 4; P1.3 normalisation Task 5b; measure-each-alone Task 6 Steps 1-2, now three isolating arms (`sfa`, `apl`, `norm`) against the `off` control, with `both` and `all` as the combinations; criterion 1 Task 6 Steps 2-3; criterion 2 Task 6 Step 4; criterion 3 Task 6 Step 5; archival Task 7. The spec's risk "criterion 1 has no hard pass line" is handled by Task 6 Step 3 naming all three failure modes - silencing, flattening, no movement - before any run.
- **The P1.3 interaction is specified, not deferred.** `Plastic.push()` would revert 18,674 normalised edges from `w0`, after every training trial, with no assert firing. Task 5b fixes it in `Plastic.signed_values` (one place: the only function that turns `w()` into a `sim.data` value) and `regime/test_norm.py` fails loudly if it regresses.
- **Names consistent across tasks.** `Graph.hops/bfs/n_disjoint/bottlenecks/best_path_weight`, `input_totals/percentile_of`, `probe.measure/drive/pair_jaccard/jaccard_matrix`, `_body(v, g, r, a, prob, poisson, no_spike, ...)`, `_apl_activation/_deliver_apl`, `apl_idx/apl_tgt/apl_w/apl_row`, `_exc_scale/norm_scale`, `signed_values(w_syn, scale=None)`, constants `TAU_A/SFA_B_INC/APL_GRADED/APL_SCALE/NORM_EXC_TARGET`. Arm tuples are `(SFA_B_INC, APL_GRADED, NORM_EXC_TARGET)` everywhere.
- **Default-off discipline.** Tasks 4, 5 and 5b each end with a bit-identity check against the pre-change `central>1Hz 0.172` and the v1 `ACCEPT`. Only Task 7 changes behaviour, and only after all three criteria pass.
- **Known simplification (ponytail).** `n_disjoint` is greedy vertex-disjoint, not a max-flow min-cut, and `bottlenecks` only tests interior nodes of one shortest path. Both are enough to answer "is this pathway thin or redundant" and neither is load-bearing for a decision beyond the P2/P3 gate. Upgrade path: scipy max-flow on a unit-capacity split-vertex graph if the gate answer is contested.
