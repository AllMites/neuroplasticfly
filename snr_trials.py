"""Does averaging trials actually recover signal? Measure, don't assume.

The 200k precompute ran N_TRIALS=1, and the trained heads say the pooled vector
carries almost nothing the board planes do not already have. The theory is that
Poisson seed noise swamps it. This tests that theory before anyone pays for a
GPU port to afford more trials.

Runs P positions x K seeds and decomposes the per-dimension variance of
log1p(pooled) into signal (across positions) and noise (across seeds of the same
position). If the theory holds, SNR should climb in proportion to the number of
averaged trials, and the measured SNR at k=2,4,8 should land on that line.

Run: docker compose run --rm sim python -u snr_trials.py
"""
import argparse
import multiprocessing as mp
import os

import chess
import numpy as np
import pyarrow.parquet as pq

import behaviors as B
import reservoir as R

DATA = "/app/data"
_net = None


def _init():
    global _net
    _net = B.net()


def _one(args):
    p, k, fen = args
    # seed must depend on BOTH position and trial, or every trial is identical
    out = R.simulate(chess.Board(fen), seed=R.seed_for(chess.Board(fen)) + 1009 * k,
                     n=_net)
    return p, k, np.asarray(out["pooled"], dtype=np.float32)


def decompose(x):
    """x[P, K, G] of log1p rates -> (signal_var, noise_var) per live dim.

    noise_var  = mean over positions of the across-seed variance
    total_var  = variance across positions of the per-position seed mean,
                 which still contains noise_var/K, so subtract it off.
    """
    P, K, _ = x.shape
    noise = x.var(axis=1, ddof=1).mean(axis=0)
    means = x.mean(axis=1)
    signal = means.var(axis=0, ddof=1) - noise / K
    return signal, noise


def sibling_fens(n_pairs, rng):
    """(fen, fen-after-one-ply) pairs: the resolution move prediction needs.

    Random positions span openings to endgames, so separating them is easy and
    says little. Two boards one ply apart is the case that actually decides a
    move.
    """
    t = pq.read_table(os.path.join(DATA, "positions.parquet"),
                      columns=["fen", "move_uci"])
    fens = t.column("fen").to_pylist()
    ucis = t.column("move_uci").to_pylist()
    out = []
    for i in rng.permutation(len(fens)):
        b = chess.Board(fens[i])
        mv = chess.Move.from_uci(ucis[i])
        if mv not in b.legal_moves:
            continue
        b.push(mv)
        out.append((fens[i], b.fen()))
        if len(out) == n_pairs:
            return out
    raise RuntimeError("not enough usable positions")


def run_siblings(n_pairs, n_seeds, workers):
    rng = np.random.default_rng(0)
    pairs = sibling_fens(n_pairs, rng)
    flat = [f for pair in pairs for f in pair]

    B.net()
    R.pooling()
    jobs = [(p, k, f) for p, f in enumerate(flat) for k in range(n_seeds)]
    print("%d one-ply pairs x 2 x %d seeds = %d sims" % (n_pairs, n_seeds, len(jobs)))

    ctx = mp.get_context("fork")
    x = None
    with ctx.Pool(workers, initializer=_init) as pool:
        for done, (p, k, vec) in enumerate(pool.imap_unordered(_one, jobs, chunksize=1), 1):
            if x is None:
                x = np.zeros((len(flat), n_seeds, len(vec)), dtype=np.float32)
            x[p, k] = vec
            if done % 200 == 0:
                print("  %d/%d" % (done, len(jobs)), flush=True)

    np.log1p(np.clip(x, 0, None, out=x), out=x)
    live = x.reshape(-1, x.shape[2]).std(axis=0) > 1e-4
    x = x[:, :, live]
    noise = x.var(axis=1, ddof=1).mean(axis=0)
    means = x.mean(axis=1)
    a, b = means[0::2], means[1::2]

    # E[(a-b)^2] = 2*sibling_signal + 2*noise/K
    sib = ((a - b) ** 2).mean(axis=0) / 2 - noise / n_seeds
    glob = means.var(axis=0, ddof=1) - noise / n_seeds
    ok = noise > 0
    print("\nlive dims: %d" % int(live.sum()))
    print("SNR across random positions : %.3f" % float(np.sum(glob[ok]) / np.sum(noise[ok])))
    print("SNR across one-ply siblings : %.3f" % float(np.sum(sib[ok]) / np.sum(noise[ok])))
    print("ratio (how much coarser)    : %.1fx" % float(np.sum(glob[ok]) / max(np.sum(sib[ok]), 1e-9)))


def run(n_pos, n_seeds, workers):
    fens = pq.read_table(os.path.join(DATA, "positions.parquet"),
                         columns=["fen"]).column("fen").to_pylist()
    rng = np.random.default_rng(0)
    picks = [fens[i] for i in rng.choice(len(fens), n_pos, replace=False)]

    B.net()
    R.pooling()
    jobs = [(p, k, f) for p, f in enumerate(picks) for k in range(n_seeds)]
    print("%d positions x %d seeds = %d sims on %d workers"
          % (n_pos, n_seeds, len(jobs), workers))

    ctx = mp.get_context("fork")
    x = None
    with ctx.Pool(workers, initializer=_init) as pool:
        for done, (p, k, vec) in enumerate(pool.imap_unordered(_one, jobs, chunksize=1), 1):
            if x is None:
                x = np.zeros((n_pos, n_seeds, len(vec)), dtype=np.float32)
            x[p, k] = vec
            if done % 100 == 0:
                print("  %d/%d" % (done, len(jobs)), flush=True)

    np.log1p(np.clip(x, 0, None, out=x), out=x)
    live = x.reshape(-1, x.shape[2]).std(axis=0) > 1e-4
    x = x[:, :, live]
    signal, noise = decompose(x)
    print("\nlive dims: %d of %d" % (int(live.sum()), len(live)))

    ok = noise > 0
    snr1 = float(np.sum(signal[ok]) / np.sum(noise[ok]))
    print("variance SNR at 1 trial: %.4f  (signal %.4g / noise %.4g)"
          % (snr1, float(np.sum(signal[ok])), float(np.sum(noise[ok]))))

    # measured: average k seeds, recompute how well positions separate
    print("\n k   measured SNR   predicted (k x SNR1)")
    for k in [j for j in (1, 2, 4, 8, 16) if j <= n_seeds]:
        sub = x[:, :k, :]
        s, nz = decompose(sub) if k > 1 else (signal, noise)
        if k > 1:
            s = sub.mean(axis=1).var(axis=0, ddof=1) - nz / k
        eff = float(np.sum(s[ok]) / np.sum(nz[ok] / k))
        print(" %-3d %12.3f %14.3f" % (k, eff, k * snr1))

    for target in (1.0, 4.0):
        need = target / snr1 if snr1 > 0 else float("inf")
        print("trials for SNR %.0f: %.0f" % (target, np.ceil(need)))
    return snr1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--positions", type=int, default=120)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--siblings", type=int, default=0,
                    help="instead: N one-ply pairs, the resolution moves need")
    args = ap.parse_args()
    if args.siblings:
        run_siblings(args.siblings, args.seeds, args.workers)
    else:
        run(args.positions, args.seeds, args.workers)
