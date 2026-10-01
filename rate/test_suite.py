"""Regression harness checks (path 2, phase 4 harness v1).

The failures guarded here are all silent: a gradient leaking into an earlier chunk's frozen params,
a stale hook intersecting two masks, two chunks owning one pool, and a suite that stays green when a
row it recorded as passing breaks. Test 5 deliberately breaks one and requires the suite to go RED.

Run: .venv/Scripts/python.exe tools/gpu_slot.py -- .venv/Scripts/python.exe rate/test_suite.py
"""
import os
import sys

import numpy as np
import scipy.sparse as sp
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rate import suite as S  # noqa: E402
from rate.engine import RateSim  # noqa: E402

# Toy: 0 -(+2)-> 1 -(+1)-> 2, one pool per neuron, 0 driven; loss = rate of neuron 2.
toy = sp.csr_matrix((np.array([2.0, 1.0]), np.array([1, 2]), np.array([0, 1, 2, 2])), shape=(3, 3))
t = RateSim(0.1, device="cpu", dtype=torch.float64, csr=toy, pool=np.arange(3))
t.bias[1:] = 1.0


def grad_of(pools):
    t.bias.grad = None
    t.set_trainable("bias", pools=pools)
    t.run([(np.array([0]), 10.0)], t_run=100.0)[0, 2].backward()
    g = t.bias.grad.clone()
    t.bias.requires_grad_(False)
    return g


# 1. Unmasked: pools 1 and 2 both carry gradient (else the mask tests prove nothing).
g = grad_of(None)
assert g[1] != 0 and g[2] != 0, g
# 2. Mask: only the owned pool gets gradient.
g = grad_of([1])
assert g[1] != 0 and g[0] == 0 and g[2] == 0, g
# 3. Re-masking replaces the old hook instead of intersecting with it.
g = grad_of([2])
assert g[2] != 0 and g[1] == 0, g

# 4. get/load round trip touches only the given pools.
before = t.get_params([0, 2])
t.load_params([1], {"log_tau": [3.5], "bias": [7.0], "gain": [0.5]})
after = t.get_params([1])
assert after["bias"][0] == 7.0 and after["gain"][0] == 0.5 and after["log_tau"][0] == 3.5
assert all(np.array_equal(before[k], t.get_params([0, 2])[k]) for k in RateSim.PARAMS)

# 5. Ownership: a pool shared with an unowned type is refused; two owners of one pool refused.
pool, ct = np.array([0, 0, 1]), np.array(["A", "B", "C"])
for chunks in ([{"name": "x", "owns": ["A"]}],
               [{"name": "x", "owns": ["C"]}, {"name": "y", "owns": ["C"]}]):
    try:
        S.check_ownership(chunks, pool, ct)
        raise SystemExit("FAIL: ownership error accepted: %s" % chunks)
    except AssertionError:
        pass
assert S.check_ownership([{"name": "x", "owns": ["C"]}], pool, ct) == {1: "x"}

# 5b. Latch check on toys: a feed-forward chain returns to rest (PASS); a self-exciting neuron
#     (1 -> 1, weight 12 at w_scale 0.1: loop gain 1.2 > 1) keeps firing after the drive (LATCH).
ff = RateSim(0.1, device="cpu", dtype=torch.float64, csr=toy, pool=np.arange(3))
ok_ff, per = S.latch_check(ff, {"d": (np.array([0]), 10.0)})
assert ok_ff, per
loop = sp.csr_matrix((np.array([2.0, 12.0]), np.array([1, 1]), np.array([0, 1, 2, 2])), shape=(3, 3))
lt = RateSim(0.1, device="cpu", dtype=torch.float64, csr=loop, pool=np.arange(3))
lt.r_max = 100.0
ok_l, per = S.latch_check(lt, {"d": (np.array([0]), 10.0)})
assert not ok_l and per["d"]["n_above"] == 1, per
print("latch toys: feed-forward PASS, self-loop LATCH", flush=True)

# 6. Real brain: the suite is green on the model its record came from, with nothing gained.
sim = S.build()
ok, rep = S.run_suite(sim)
assert ok and not rep["c1_sugar_pam"]["gained"], rep
print("real brain: suite green, rows identical to record", flush=True)

# 7. Params reach the battery: remove threshold + raise gain on the reward-set pools. 28/216 R
#    neurons get net positive Fox input in arm A (RESULT.md post-hoc), so some F1-F4 must be GAINED.
#    (b1 cannot be sabotaged this way: all 293 of its presynaptic partners are silent under Fox.)
ct_real = S.C1.sets()["ct"]
R = S.owned_pools(sim.pool.cpu().numpy(), ct_real, S.C1.COMP["R"])
v = sim.get_params(R)
sim.load_params(R, dict(v, bias=np.zeros(len(R)), gain=np.full(len(R), 1e4)))
ok, rep = S.run_suite(sim)
gained = rep["c1_sugar_pam"]["gained"]
assert any(r in gained for r in ("F1", "F2", "F3", "F4")), rep
print("param effect: gained %s, regressions %s" % (gained, rep["c1_sugar_pam"]["regressions"]),
      flush=True)
sim.load_params(R, v)

# 8. Regression logic: a record claiming F1 passed, run on the base model, must go RED on F1.
import json, tempfile  # noqa: E401,E402
rec = json.load(open(os.path.join(S._HERE, S.CHUNKS[1]["record"])))
rec["rows"]["F1"] = True
tmp = os.path.join(tempfile.mkdtemp(), "rec.json")
json.dump(rec, open(tmp, "w"))
ok, rep = S.run_suite(sim, [dict(S.CHUNKS[1], record=tmp)])
assert not ok and rep["c1_sugar_pam"]["regressions"] == ["F1"], rep
print("regression logic: RED on F1 as required", flush=True)

print("ok regression harness")
