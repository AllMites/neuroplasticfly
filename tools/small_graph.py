"""Synthetic small-graph fixture for the CPU tests (issue #4).

Writes <root>/data/brain_gpu.npz and <root>/data/neuron_meta.npz in the same format and
dtypes as the built brain (export_brain.py), so gpu_sim.Brain, gpu_sim.GpuSim and
rate.engine.RateSim load it unmodified. Nothing here is connectome data.

The engine asserts the v783 neuron count (139,248), so the graph has that many rows, but
only the first N_WIRED neurons carry edges; every other neuron is unconnected and must stay
silent. The circuit (synapse counts, sign = transmitter):

    IN[i]    -(+60)-> RELAY[i]    i = 0..7, IN is the driven layer
    RELAY[i] -(+120)-> OUT[i]
    RELAY[i] -(+10)-> INH
    INH      -(-40)-> OUT[i]      i = 0..3 only (OUT 4..7 get no inhibition)

gpu_sim reads data/neuron_meta.npz from its module-level _HERE; point that at <root> in the
test process (G._HERE = root). gpu_sim.py itself is hashed (PROVENANCE.md) and is not edited.
"""
import os

import numpy as np

N_V783 = 139248
IN, RELAY, OUT = np.arange(0, 8), np.arange(8, 16), np.arange(16, 24)
INH = 24
N_WIRED = 25
W_IN_RELAY, W_RELAY_OUT, W_RELAY_INH, W_INH_OUT = 60.0, 120.0, 10.0, -40.0


def edges():
    """(pre, post, synapses) of the wired circuit."""
    e = [(i, r, W_IN_RELAY) for i, r in zip(IN, RELAY)]
    e += [(r, o, W_RELAY_OUT) for r, o in zip(RELAY, OUT)]
    e += [(r, INH, W_RELAY_INH) for r in RELAY]
    e += [(INH, o, W_INH_OUT) for o in OUT[:4]]
    return e


def build(root, n=N_V783):
    """Write the fixture under root/data/ and return the brain npz path."""
    d = os.path.join(root, "data")
    os.makedirs(d, exist_ok=True)
    pre, post, w = (np.array(c) for c in zip(*sorted(edges())))
    indptr = np.zeros(n + 1, np.int64)
    np.add.at(indptr, pre + 1, 1)
    indptr = np.cumsum(indptr)                       # pre-major CSR, as export_brain.py
    names = np.array(["none", "IN", "RELAY", "OUT", "INH"])
    gid = np.zeros(n, np.int32)                      # one pool per layer
    gid[IN], gid[RELAY], gid[OUT], gid[INH] = 1, 2, 3, 4
    ct = names[gid]
    sizes = np.bincount(gid, minlength=len(names))
    path = os.path.join(d, "brain_gpu.npz")
    np.savez(path, n_neurons=np.int64(n), min_syn=np.int64(5), W_indptr=indptr,
             W_indices=post.astype(np.int32), W_data=w.astype(np.float32),
             n_edges=np.int64(len(w)), pool_group_id=gid, pool_sizes=sizes.astype(np.float32),
             pool_names=names, selector_names=np.array(["cell_type=IN"]),
             selector_0000=IN.astype(np.int64))
    np.savez(os.path.join(d, "neuron_meta.npz"), group_id=gid, group_sizes=sizes.astype(np.int32),
             group_names=names, cell_type=ct, cell_class=ct.copy())
    return path
