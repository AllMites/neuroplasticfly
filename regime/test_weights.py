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
