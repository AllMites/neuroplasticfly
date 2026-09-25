"""Reachability between neuron sets over the connectome, as a plain graph.

READ ONLY. Answers one question for P0.1: is there a path at all from the
sensory neurons to the target, how long, how redundant, and what would cut it.
Signs are ignored here - an inhibitory path is still a path. Weight is the
sum of |synapse count| along the path, reported for scale, not for dynamics.

Three guards, each from a measured failure:

- Node sets must be integer indices, never boolean masks. A 139248-element
  bool mask iterates to int(False)/int(True), silently turning the query into
  "does node 0 or 1 reach node 0 or 1" and answering hops=0. Asserted, not
  coerced: the caller must say which form it meant.
- ``n_disjoint`` returns ``(k, saturated)``. Hitting the cap and genuinely
  running out of paths are different facts and must not share a number.
- ``min_w`` drops edges below a weight floor. Unweighted reachability on a
  2.7M-edge small-world graph does not beat a size-matched random null, so
  the useful question is reachability over non-trivial synapses.
"""
from collections import deque

import numpy as np


def _check_nodes(a, name):
    a = np.asarray(a)
    assert a.dtype.kind in "iu", (
        "%s must be integer node indices, not %s - a boolean mask silently "
        "becomes nodes 0/1; use np.flatnonzero(mask)" % (name, a.dtype)
    )
    return a


class Graph:
    def __init__(self, indptr, indices, data):
        self.indptr = np.asarray(indptr, np.int64)
        self.indices = np.asarray(indices, np.int64)
        self.data = np.asarray(data, np.float32)
        self.n = len(self.indptr) - 1

    def _succ(self, i, blocked=None, min_w=0.0):
        s = slice(self.indptr[i], self.indptr[i + 1])
        succ = self.indices[s]
        if min_w > 0.0:
            succ = succ[np.abs(self.data[s]) >= min_w]
        return succ if blocked is None else succ[~blocked[succ]]

    def bfs(self, src, dst, blocked=None, min_w=0.0):
        """Shortest path src-set -> dst-set as a list of node indices, or None."""
        src = _check_nodes(src, "src")
        dst = set(int(x) for x in _check_nodes(dst, "dst"))
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
            for j in self._succ(i, blocked, min_w):
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

    def hops(self, src, dst, min_w=0.0):
        p = self.bfs(src, dst, min_w=min_w)
        return None if p is None else len(p) - 1

    def best_path_weight(self, src, dst, min_w=0.0):
        """Summed |weight| along the shortest path, or None."""
        p = self.bfs(src, dst, min_w=min_w)
        if p is None:
            return None
        tot = 0.0
        for a, b in zip(p, p[1:]):
            s = slice(self.indptr[a], self.indptr[a + 1])
            hit = self.indices[s] == b
            tot += float(np.abs(self.data[s][hit]).sum())
        return tot

    def n_disjoint(self, src, dst, cap=8, min_w=0.0):
        """Greedy vertex-disjoint path count as ``(k, saturated)``.

        ``saturated`` is True when the search stopped at ``cap`` rather than
        running out of paths, i.e. k is a lower bound, not a measurement.
        """
        blocked = np.zeros(self.n, bool)
        k = 0
        while k < cap:
            p = self.bfs(src, dst, blocked, min_w=min_w)
            if p is None:
                return k, False
            for node in p[1:-1]:
                blocked[node] = True
            k += 1
        return k, True

    def bottlenecks(self, src, dst, cap=32, min_w=0.0):
        """Interior nodes whose individual removal disconnects src from dst."""
        p = self.bfs(src, dst, min_w=min_w)
        if p is None or len(p) <= 2:
            return []
        out = []
        for node in p[1:-1][:cap]:
            blocked = np.zeros(self.n, bool)
            blocked[node] = True
            if self.bfs(src, dst, blocked, min_w=min_w) is None:
                out.append(int(node))
        return out


def load(brain="data/brain_gpu.npz"):
    b = np.load(brain, allow_pickle=False)
    return Graph(b["W_indptr"], b["W_indices"], b["W_data"])
