"""Fly-hash fingerprint: sparse, deterministic, similar inputs overlap."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from learn import fingerprint as F

rng = np.random.default_rng(0)
x = rng.random(12).astype(np.float32)
idx1, hz1 = F.kc_drive(x)
idx2, hz2 = F.kc_drive(x)
assert np.array_equal(idx1, idx2), "not deterministic"
n_kc = len(F.kc_indices())
assert 0.04 <= len(idx1) / n_kc <= 0.06, "sparsity %.3f" % (len(idx1) / n_kc)
assert np.all(hz1 == F.KC_HZ)
y = x + 0.05 * rng.random(12).astype(np.float32)          # similar song
z = rng.random(12).astype(np.float32)                      # different song
def jac(a, b): a, b = set(a), set(b); return len(a & b) / len(a | b)
assert jac(idx1, F.kc_drive(y)[0]) > jac(idx1, F.kc_drive(z)[0]), "similar inputs must overlap more"
assert set(idx1) <= set(F.kc_indices()), "drives non-KC"
print("ok fingerprint: %d KCs driven, jac(sim)=%.2f jac(diff)=%.2f" % (
    len(idx1), jac(idx1, F.kc_drive(y)[0]), jac(idx1, F.kc_drive(z)[0])))
