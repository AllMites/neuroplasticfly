"""Three-factor KC->MBON rule on a toy and on the real offsets."""
import os, sys, tempfile
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from learn import plastic as PL

# --- toy: 3 KCs, 2 MBONs (mbon0 approach, mbon1 avoid), 2 DANs (dan0 punish, dan1 reward)
P = PL.Plastic.toy()
w0 = P.w0.copy()
rates = np.zeros(P.n_neurons, np.float32)
rates[P.kc_of_edge] = 10.0                     # every KC at 10 Hz
rates[P.dan_idx["punish"]] = 20.0              # punishment DAN at 20 Hz, reward DAN silent
dw = P.update(rates, eta=1e-3, lam=0.0)
# punish depresses edges onto approach MBONs only: dw = -eta * 10 * 20 * w0 = -0.2 w0
onto_appr = P.mbon_valence_of_edge == "approach"
assert np.allclose(dw[onto_appr], -0.2 * w0[onto_appr]), dw
assert np.all(dw[~onto_appr] == 0), "reward edges moved with no reward"
# floor and recovery
for _ in range(50): P.update(rates, eta=1e-3, lam=0.0)
assert np.all(P.w()[onto_appr] >= PL.W_MIN * w0[onto_appr] - 1e-6), "below floor"
rates[:] = 0.0
before = P.w()[onto_appr].copy()
P.update(rates, eta=1e-3, lam=0.5)
assert np.all(P.w()[onto_appr] > before), "no recovery"
# lesion arm: no change
P2 = PL.Plastic.toy(); rates[:] = 10.0
assert np.all(P2.update(rates, eta=1e-3, lam=0.0, lesion="all") == 0)
# single-valence lesion: reward arm off, punish arm still depresses
P4 = PL.Plastic.toy(); rates[:] = 0.0
rates[P4.kc_of_edge] = 10.0; rates[P4.dan_idx["punish"]] = 20.0
d4 = P4.update(rates, eta=1e-3, lam=0.0, lesion="reward")
assert np.all(d4[P4.mbon_valence_of_edge == "approach"] < 0), "punish arm lesioned too"
assert np.all(d4[P4.mbon_valence_of_edge == "avoid"] == 0), "lesioned reward arm moved"
# --- state round trip
d = tempfile.mkdtemp(); P.save(d); P3 = PL.Plastic.toy(); P3.load(d)
assert np.allclose(P3.dw, P.dw)
# --- real offsets exist and are all KC->MBON
R = PL.Plastic.real()
assert R.n_edges == 18674, R.n_edges
assert np.all(R.w0 > 0), "w0 is a magnitude: no zero-weight edge"
import json as _json
_meta = np.load(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "data", "neuron_meta.npz"), allow_pickle=False)
_ct = _meta["cell_type"].astype(str); _cc = _meta["cell_class"].astype(str)
_table = _json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "compartments.json")))
assert np.all(_cc[R.kc_of_edge] == "Kenyon_Cell"), "a plastic edge is not from a Kenyon cell"
assert np.all(np.isin(_ct[R.mbon_of_edge], list(_table["mbon"]))), "a plastic edge is not onto a table MBON"
# per-edge sign: 5 of the 18,674 are negative in v783 and must stay negative downstream
assert (R.sign < 0).sum() == 5, int((R.sign < 0).sum())
import gpu_sim as _G                             # here, not at module level: import may init CUDA
_v = R.signed_values(_G.W_SYN)                    # the scale push() uses, not a literal
assert np.all(_v[R.sign < 0] < 0) and np.all(_v[R.sign > 0] > 0), "signed_values lost polarity"
print("ok plastic: toy rule, floor, recovery, lesion, state; real edges", R.n_edges)
