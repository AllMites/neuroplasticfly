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
