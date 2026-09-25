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
assert g.n_disjoint(np.array([0]), np.array([3])) == (2, False)
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

# FIX 1: a boolean mask must raise, not silently become nodes 0/1.
# On this toy graph the mask form would otherwise answer hops=0.
mask_src = np.zeros(4, bool); mask_src[0] = True
mask_dst = np.zeros(4, bool); mask_dst[3] = True
for bad_src, bad_dst, who in [(mask_src, np.array([3]), "src"),
                              (np.array([0]), mask_dst, "dst")]:
    try:
        g.hops(bad_src, bad_dst)
    except AssertionError as e:
        assert who in str(e), str(e)
    else:
        raise AssertionError("bool mask for %s must raise" % who)
# and the correct form still works
assert g.hops(np.flatnonzero(mask_src), np.flatnonzero(mask_dst)) == 2

# FIX 2: saturation is visible. Exhausted -> False, capped -> True.
k, sat = g.n_disjoint(np.array([0]), np.array([3]), cap=8)
assert (k, sat) == (2, False), (k, sat)
k, sat = g.n_disjoint(np.array([0]), np.array([3]), cap=1)
assert (k, sat) == (1, True), (k, sat)

# FIX 3: min_w drops weak edges. 0 -> 1 is weak (2.0), 0 -> 2 -> 3 is strong.
indptr3 = np.array([0, 2, 3, 4, 4], np.int64)
indices3 = np.array([1, 2, 3, 3], np.int64)
data3 = np.array([2.0, 5.0, 2.0, 5.0], np.float32)   # weak: 0->1, 1->3
g3 = R.Graph(indptr3, indices3, data3)
assert g3.n_disjoint(np.array([0]), np.array([3]), cap=8) == (2, False)
assert g3.hops(np.array([0]), np.array([3]), min_w=0.0) == 2
assert g3.hops(np.array([0]), np.array([3]), min_w=3.0) == 2, "strong route survives"
assert g3.best_path_weight(np.array([0]), np.array([3]), min_w=3.0) == 10.0
# with the floor, only the strong route remains, so redundancy drops to 1
assert g3.n_disjoint(np.array([0]), np.array([3]), cap=8, min_w=3.0) == (1, False)
# raise the floor above every edge and the path is gone
assert g3.hops(np.array([0]), np.array([3]), min_w=6.0) is None
assert g3.n_disjoint(np.array([0]), np.array([3]), cap=8, min_w=6.0) == (0, False)
assert g3.bottlenecks(np.array([0]), np.array([3]), min_w=6.0) == []
# and node 2 is a cut vertex once the weak route is filtered out
assert g3.bottlenecks(np.array([0]), np.array([3]), min_w=3.0) == [2]

print("ok reach: hops, disjoint paths, bottleneck, weight, mask guard, saturation, min_w")
