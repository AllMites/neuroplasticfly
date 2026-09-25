"""One 300 ms whole-brain sim per position -> a pooled rate vector.

REAL: the 139,255-neuron connectivity matrix and the LIF dynamics (Shiu et al.
2024 constants, via flypoke). Nothing in the wiring is trained or edited.
AUTHORED: the 300 ms trial length, the single trial, and the pooling by
cell_type. TRAINED: nothing here - the reservoir is frozen, train.py owns the
only learned weights.

The pooled vector is what the head sees. Full 139k rates and spike rasters are
kept only for recorded games (they are ~1 MB a move), never for the 200k
training positions.
"""
import json
import os
import zlib

import chess
import numpy as np

import encode

T_RUN = 300.0          # ms. Measured: ~1.1 s wall clock. Raise drive, not this.
N_TRIALS = 1

# Descending readouts frozen for the body interface (spec 2). Every one of these
# was measurable in flypet's stimulus matrix, so spec 2 never has to re-simulate.
DN_TYPES = ["DNg29", "DNg84", "DNp01", "DNp04", "DNa01", "DNa02", "DNa03", "DNp09"]
INGESTION = "cell_sub_class=ingestion_motor_neuron"


def _net():
    """`behaviors.net()`, imported at call time rather than at module scope.

    behaviors lives in flypet and flypoke lives in the sim image, so neither is
    importable on the Windows host. Both used to be module-level imports here,
    which made `import reservoir` fail on the host and defeated the stub machinery
    in gpu_sim._install_stubs() that names this module as one it exists to support.
    encode.stimuli() and drive_sweep already import their container-only
    dependencies inside the function that needs them; this matches.
    """
    import behaviors as B
    return B.net()

_pool = None
_dn_idx = None


def pooling(n=None):
    """(group_id per neuron, group sizes, group names). ~8-9k groups."""
    global _pool
    if _pool is not None:
        return _pool
    n = n or _net()
    df = n.neurons
    key = (df.cell_type.astype("string")
             .fillna(df.cell_class.astype("string"))
             .fillna(df.super_class.astype("string"))
             .fillna("unannotated"))
    names, group_id = np.unique(key.to_numpy(), return_inverse=True)
    sizes = np.bincount(group_id, minlength=len(names)).astype(np.float32)
    _pool = (group_id.astype(np.int32), sizes, [str(x) for x in names])
    return _pool


def n_groups(n=None):
    return len(pooling(n)[2])


def dn_index(n=None):
    """{cell_type: {side: indices}} for the frozen readout list, plus ingestion."""
    global _dn_idx
    if _dn_idx is not None:
        return _dn_idx
    n = n or _net()
    table = {}
    for ct in DN_TYPES:
        table[ct] = {side: n.select("cell_type=%s,side=%s" % (ct, side))
                     for side in ("left", "right")}
    table["ingestion_motor_neuron"] = {"both": n.select(INGESTION)}
    _dn_idx = table
    return _dn_idx


def pool(rates, n=None):
    """139k Hz -> per-group mean Hz, float16 (the head standardises it anyway)."""
    group_id, sizes, _ = pooling(n)
    summed = np.bincount(group_id, weights=rates, minlength=len(sizes))
    return (summed / sizes).astype(np.float16)


# ---------------------------------------------------------------- readouts
# Measured 2026-09-16 (`readout_probe.py`): reconstructing which piece stands on
# which square from 5,000 identical simulations, at identical width and decoder
# budget, gives occupied-F1 0.610 for `pool` above, 0.852 for a sparse random
# projection and 0.850 for a random subset of neurons. Averaging every neuron of
# a cell type together destroys which part of the retina fired, which is which
# square held the piece. So the shipped readout throws away most of the board.
#
# These tables are derived from a fixed seed and the neuron count, so any process
# that loads the same network reconstructs them identically. They are never
# trained - picking neurons is not learning.
READOUT_SEED = 12345
PROJ_WIDTH = 8865          # same width as `pool`, so comparisons stay fair
PROJ_FAN = 64              # neurons summed per projection dimension

_tables = None


def readout_tables(n=None):
    """(projection index table [PROJ_WIDTH, PROJ_FAN], subset [PROJ_WIDTH])."""
    global _tables
    if _tables is not None:
        return _tables
    n = n or _net()
    rng = np.random.default_rng(READOUT_SEED)
    proj = rng.integers(0, n.n, size=(PROJ_WIDTH, PROJ_FAN), dtype=np.int32)
    subset = rng.choice(n.n, PROJ_WIDTH, replace=False).astype(np.int32)
    _tables = (proj, subset)
    return _tables


LIVE_MASK = os.environ.get("FLYCHESS_LIVE_MASK", "/app/data/live_mask.npz")
_live = None


def live_mask():
    """Bool over neurons that ever respond to a chessboard, or None.

    Measured over the 5,000-position rate dump: 27,857 of 139,248 neurons have
    any across-position variance at all. The rest are silent for every board, so
    storing them is storing zeros.
    """
    global _live
    if _live is None and os.path.exists(LIVE_MASK):
        blob = np.load(LIVE_MASK)
        _live = blob["sd"] > 1e-3 if "sd" in blob.files else blob["live"]
    return _live


def views(rates, n=None):
    """What a shard keeps per position, float16.

    `live` is the raw rate of every neuron that ever responds, NOT a derived
    readout, because this project has now twice paid for storing a summary and
    later needing the thing it summarised. From `live` any readout can be
    recomputed on the host in seconds: cell_type means, random projections,
    neuron subsets, region masks, and -- importantly -- readouts that exclude
    the 13,686 live neurons the encoder drives directly.

    That exclusion matters for honesty, not taste. flypoke turns a stimulated
    neuron into a pure Poisson source and drops its network input (see
    encode.stimuli), so a driven neuron's rate IS the drive. Reconstructing the
    board from driven neurons measures our own encoder, not the fly.

    `pooled` is kept alongside only so results stay comparable with the first
    200k run; it is the lossy readout this run exists to replace.
    """
    out = {"pooled": pool(rates, n)}
    live = live_mask()
    if live is not None:
        out["live"] = rates[live].astype(np.float16)
    return out


def dn(rates, n=None):
    """{cell_type: {side: Hz}} for the body interface."""
    out = {}
    for ct, sides in dn_index(n).items():
        out[ct] = {side: (float(rates[idx].mean()) if idx.size else 0.0)
                   for side, idx in sides.items()}
    return out


def seed_for(board):
    """Deterministic per position: the same FEN always gives the same sim."""
    return zlib.crc32(board.fen().encode()) & 0x7FFFFFFF


def simulate(board, seed=None, full=False, n=None, all_views=False):
    """Run one position through the brain.

    Returns pooled/dn_table/sensory_summary always; `all_views` adds the two
    better readouts (cheap, no spike raster); the 139k rate vector and the spike
    raster only when `full`, because those are ~1 MB per move.
    """
    # Imported here, not at module scope, so this file can be imported on a host
    # with no flypoke - the same reason encode.stimuli() does it. gpu_sim installs
    # a stub whose run() raises rather than falling back, so a host that actually
    # reaches this line still fails loudly.
    from flypoke import sim as S
    n = n or _net()
    stims = encode.stimuli(board, n)
    rec = S.run(n, stims, S.Params(t_run=T_RUN), n_trials=N_TRIALS,
                seed=seed_for(board) if seed is None else seed)
    rates = rec.rates(n.n)
    out = {
        "pooled": pool(rates, n),
        "dn_table": dn(rates, n),
        "sensory_summary": encode.summary(board, n),
        "n_stim_neurons": int(sum(s.indices.size for s in stims)),
    }
    if all_views:
        out.update(views(rates, n))
    if full:
        out["rates"] = rates.astype(np.float16)
        out["spike_t"] = rec.times.astype(np.float32)
        out["spike_i"] = rec.ids.astype(np.int32)
    return out


# Diverse positions used only to measure that the reservoir separates positions
# at all: opening, endgame, a closed middlegame, a king-and-pawn study.
PROBE_FENS = [
    "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 4 4",
    "8/8/8/4k3/8/8/4K3/8 w - - 0 1",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "r2q1rk1/1b1nbppp/p2ppn2/1p6/3NPP2/1BN1B3/PPPQ2PP/2KR3R w - - 0 12",
]


def discriminability(n=None, start_vec=None):
    """(mean cross-position cosine, mean same-position/different-seed cosine, raw cos).

    Both numbers are measured in the space the head actually consumes -
    standardised log1p - because the raw rate vector is dominated by a common
    mode that is identical for every board.
    """
    n = n or _net()
    start = chess.Board()
    base = np.asarray(start_vec if start_vec is not None
                      else simulate(start, n=n)["pooled"], dtype=np.float64)
    others = [np.asarray(simulate(chess.Board(f), n=n)["pooled"], dtype=np.float64)
              for f in PROBE_FENS]
    reseeds = [np.asarray(simulate(start, seed=s, n=n)["pooled"], dtype=np.float64)
               for s in (1, 2)]

    def cos(u, v):
        return float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v)))

    batch = [base] + others
    lg = [np.log1p(v) for v in batch]
    mu = np.mean(lg, axis=0)
    sd = np.std(lg, axis=0) + 1e-6
    z = [(v - mu) / sd for v in lg]
    zr = [(np.log1p(v) - mu) / sd for v in reseeds]
    signal = float(np.mean([cos(z[0], zi) for zi in z[1:]]))
    noise = float(np.mean([cos(zr[0], zr[1])] + [cos(z[0], zi) for zi in zr]))
    return signal, noise, cos(base, others[0])


def selfcheck(verbose=True):
    import time
    n = _net()
    ok = True
    _, _, names = pooling(n)
    if verbose:
        print("OK  pooling: %d groups over %d neurons" % (len(names), n.n))

    t0 = time.time()
    start = chess.Board()
    a = simulate(start, full=True, n=n)
    secs = time.time() - t0

    central = n.select("super_class=central")
    live = float((a["rates"][central].astype(np.float32) > 1.0).mean())
    alive = live >= 0.10
    ok &= alive
    if verbose:
        print("%s central brain live fraction %.1f%% (need >= 10%%)"
              % ("OK " if alive else "BAD", 100 * live))
        print("    %d neurons driven, %.1f s wall clock for %.0f ms of sim"
              % (a["n_stim_neurons"], secs, T_RUN))

    fast = secs <= 3.0
    ok &= fast
    if verbose:
        print("%s one sim takes %.1f s (budget 3 s)" % ("OK " if fast else "BAD", secs))

    b = simulate(start, n=n)
    same = np.array_equal(np.asarray(a["pooled"]), np.asarray(b["pooled"]))
    ok &= same
    signal, noise, cos_raw = discriminability(n=n, start_vec=a["pooled"])
    distinct = signal < noise
    ok &= distinct
    if verbose:
        print("%s same FEN twice identical=%s" % ("OK " if same else "BAD", same))
        print("%s discriminability: different FENs cos=%+.3f vs seed-noise floor "
              "cos=%+.3f (need lower)" % ("OK " if distinct else "BAD", signal, noise))
        print("    raw cosine on un-standardised rates is %.5f, but the seed-noise"
              % cos_raw)
        print("    floor there is ~0.9997: common mode dominates, so the plan's")
        print("    fixed 0.99 threshold is unreachable. Measured against noise instead.")

    dn_table = a["dn_table"]
    complete = all(ct in dn_table for ct in DN_TYPES) and \
        "ingestion_motor_neuron" in dn_table
    ok &= complete
    if verbose:
        print("%s dn_table has all %d frozen readouts" % ("OK " if complete else "BAD",
                                                          len(DN_TYPES) + 1))
        print("    " + json.dumps({k: {s: round(h, 1) for s, h in v.items()}
                                   for k, v in dn_table.items()}))

    os.makedirs("/app/results", exist_ok=True)
    json.dump({"t_run_ms": T_RUN, "n_trials": N_TRIALS, "seconds_per_sim": round(secs, 2),
               "n_groups": len(names), "stim_neurons_start_position": a["n_stim_neurons"],
               "central_live_fraction": round(live, 4),
               "cos_different_positions": round(signal, 4),
               "cos_seed_noise_floor": round(noise, 4),
               "cos_raw_unstandardised": round(cos_raw, 5),
               # seconds_per_sim is only meaningful on an idle box: 16 precompute
               # workers push it from ~0.9 s to ~4 s. Record the load with it.
               "loadavg_1min": round(os.getloadavg()[0], 1),
               "cpu_count": os.cpu_count()},
              open("/app/results/timing.json", "w"), indent=1)
    if verbose:
        print("    wrote /app/results/timing.json")
    return ok


if __name__ == "__main__":
    import sys
    good = selfcheck()
    print("SELFCHECK", "PASS" if good else "FAIL")
    sys.exit(0 if good else 1)
