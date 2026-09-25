"""ELN_NEGATE bridge: selects the 44 known eLNs, negates only their rows, off by default."""
import os, sys
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gpu_sim as G

assert G.ELN_NEGATE is False, "ELN_NEGATE must ship default-off"
sim = G.GpuSim()
# brain() is a module singleton, so `on.net is sim.net`; keep an independent copy
# or the host-W_data check below would compare an array with itself.
w_before = sim.net.W_data.copy()
eln = G._eln_idx(sim.net)
assert len(eln) == 44
meta = np.load(os.path.join(G._HERE, "data", "neuron_meta.npz"), allow_pickle=False)
assert (meta["cell_class"].astype(str)[eln] == "ALLN").all()
ip = sim.net.W_indptr
rows = np.concatenate([np.arange(ip[i], ip[i + 1]) for i in eln])
assert len(rows) == 7318, len(rows)
off = sim.data.clone()
assert sim.eln_idx.numel() == 0
assert float(off[torch.as_tensor(rows, device=sim.device)].min()) > 0.0, "off path must leave eLN rows positive"

# plastic edges (KC->MBON) are disjoint from eLN rows
from learn import plastic as PL
P = PL.Plastic.real()
assert not np.intersect1d(P.offsets, rows).size, "plastic edges overlap eLN rows"

G.ELN_NEGATE = True
try:
    on = G.GpuSim()
    assert on.eln_idx.tolist() == eln.tolist()
    rt = torch.as_tensor(rows, device=on.device)
    assert torch.equal(on.data[rt], -off[rt]), "eLN rows are not exactly negated"
    mask = torch.ones(off.numel(), dtype=torch.bool, device=off.device); mask[rt] = False
    assert torch.equal(on.data[mask], off[mask]), "a non-eLN weight changed"
    assert np.array_equal(on.net.W_data, w_before), "host W_data was modified"
finally:
    G.ELN_NEGATE = False
print("ok eln: 44 selected, 7318 rows negated, off path untouched, plastic rows disjoint")
