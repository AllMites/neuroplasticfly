"""Song -> Kenyon-cell drive by fly hash (Dasgupta, Stevens, Navlakha 2017).

AUTHORED, disclosed in the pinned comment: the fly's ear never reaches its
Kenyon cells in this LIF (0 of 5,177 KCs spike under any song drive, measured
2026-09-19), so the song's 12-band spectrum is projected onto the KCs by a fixed
sparse random matrix and the top 5% are driven. Same algorithm the real MB uses
for odours (random projection + winner-take-all), different input.

REAL: which neurons are Kenyon cells, everything downstream of them.
"""
import os
import numpy as np

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
META = os.path.join(_HERE, "data", "neuron_meta.npz")
M_PATH = os.path.join(_HERE, "data", "fingerprint_M.npz")

N_BANDS = 12
BANDS_PER_KC = 4     # each KC samples 4 of 12 bands (fly: ~6 of 50 glomeruli)
SPARSITY = 0.05      # fraction of KCs driven per input (fly: 5-10%)
KC_HZ = 50.0         # authored drive rate for a winning KC
SEED = 20260919

_kc = None
_M = None


def kc_indices():
    global _kc
    if _kc is None:
        meta = np.load(META, allow_pickle=False)
        _kc = np.flatnonzero(meta["cell_class"].astype(str) == "Kenyon_Cell").astype(np.int64)
    return _kc


def projection():
    """Fixed binary [N_BANDS, n_kc] matrix. Built once, saved, never regenerated."""
    global _M
    if _M is None:
        if os.path.exists(M_PATH):
            z = np.load(M_PATH)
            assert int(z["seed"]) == SEED and int(z["bands_per_kc"]) == BANDS_PER_KC and \
                z["M"].shape == (N_BANDS, len(kc_indices())), \
                "stale data/fingerprint_M.npz (SEED/BANDS_PER_KC/KC count changed) - delete it to rebuild"
            _M = z["M"]
        else:
            n_kc = len(kc_indices())
            rng = np.random.default_rng(SEED)
            M = np.zeros((N_BANDS, n_kc), np.float32)
            for j in range(n_kc):
                M[rng.choice(N_BANDS, BANDS_PER_KC, replace=False), j] = 1.0
            np.savez(M_PATH, M=M, seed=SEED, bands_per_kc=BANDS_PER_KC)
            _M = M
    return _M


def kc_drive(bands, hz=KC_HZ):
    """bands: float[12] in [0,1] -> (kc_idx int64[k], rate float32[k]), k = 5% of KCs."""
    x = np.asarray(bands, np.float32)
    assert x.shape == (N_BANDS,), x.shape
    y = x @ projection()
    k = int(round(SPARSITY * y.shape[0]))
    win = np.argpartition(-y, k)[:k]
    win = win[np.argsort(-y[win], kind="stable")]     # deterministic order
    return kc_indices()[win], np.full(k, hz, np.float32)


def song_bands(wav, seconds=3.0, offset=0.0):
    """Mean 12-band vector of a clip, same band definition as flywatch/sim_song.py."""
    import sys
    sys.path.insert(0, os.path.join(_HERE, "flywatch"))
    from sim_song import band_energy
    return band_energy(wav, seconds, offset).mean(0)
