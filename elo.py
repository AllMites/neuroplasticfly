"""Checkpoint ladder: play each checkpoint against fixed anchors and estimate Elo.

The numbers this writes are ESTIMATES against two weak anchors, not ratings. The
only rating in this project that means anything is the lichess one, which is
earned against humans. Everything here is labelled `est_elo` for that reason.

Cost: a fly move is one 300 ms sim (~0.9 s), so a 400-game round for one
checkpoint is ~40 min on 16 workers. Evaluate 3-4 checkpoints per run.

Run: docker compose run --rm sim python -u elo.py --ckpts 0,10,20 --parallel 16
"""
import argparse
import json
import math
import multiprocessing as mp
import os
import time

import behaviors as B
import play

# Anchor strengths, used only to place the estimate on a scale.
ANCHOR_ELO = {"random": 400, "stockfish_skill0": 1350,
              "stockfish_skill2": 1550, "stockfish_skill5": 1750}
# Same guard train.py carries: a ladder run against a different checkpoint set
# must not overwrite the ladder that produced the numbers in WHAT_IS_REAL.md.
OUT = os.environ.get("FLYCHESS_OUT", ".")

OPPONENTS = [("random", None), ("stockfish_skill0", 0),
             ("stockfish_skill2", 2), ("stockfish_skill5", 5)]

_player_cache = {}


def _init():
    B.net()


def _one_game(job):
    ckpt, opp_name, skill, fly_is_white, seed, temperature = job
    key = (ckpt, temperature, seed % 4)
    if key not in _player_cache:
        _player_cache[key] = play.FlyPlayer(ckpt, temperature=temperature, seed=seed)
    player = _player_cache[key]
    kind = "random" if opp_name == "random" else "stockfish"
    opp, limit = play.opponent(kind, skill or 0, movetime=0.02)
    try:
        board, _ = play.play_game(player, opp, limit, fly_is_white,
                                  reward_engine=None, record_dir=None, verbose=False)
    finally:
        opp.quit()
    return opp_name, play.score_for_fly(board, fly_is_white)


def est_elo(score, anchor):
    """Logistic inverse. Clamped, because 0 or 1 is infinite Elo."""
    s = min(max(score, 0.01), 0.99)
    return anchor - 400.0 * math.log10(1.0 / s - 1.0)


def run(ckpts, runs, games_per_side, parallel, temperature, opponents=None):
    """One (checkpoint, opponent) block at a time, so results stay attributable."""
    chosen = ([o for o in OPPONENTS if o[0] in opponents] if opponents
              else list(OPPONENTS))
    if not chosen:
        raise SystemExit("no opponent matched; pick from %s"
                         % ",".join(o[0] for o in OPPONENTS))
    B.net()
    rows = []
    ctx = mp.get_context("fork")
    t0 = time.time()
    with ctx.Pool(parallel, initializer=_init) as pool:
        for run_name in runs:
            for epoch in ckpts:
                path = os.path.join(OUT, "checkpoints", run_name,
                                    "epoch_%02d.pt" % epoch)
                if not os.path.exists(path):
                    print("skip missing", path)
                    continue
                for opp_name, skill in chosen:
                    jobs = [(path, opp_name, skill, white, 1000 + g, temperature)
                            for white in (True, False)
                            for g in range(games_per_side)]
                    scores = [s for _, s in pool.imap_unordered(_one_game, jobs, 1)]
                    w = sum(1 for s in scores if s == 1.0)
                    d = sum(1 for s in scores if s == 0.5)
                    lo = len(scores) - w - d
                    mean = sum(scores) / len(scores)
                    row = {"ckpt": epoch, "run": run_name, "opponent": opp_name,
                           "w": w, "d": d, "l": lo, "score": round(mean, 3),
                           "est_elo": round(est_elo(mean, ANCHOR_ELO[opp_name]))}
                    rows.append(row)
                    print("%-9s ckpt %2d vs %-17s  %d-%d-%d  score %.3f  est %d Elo "
                          "(%.1f min)" % (run_name, epoch, opp_name, w, d, lo, mean,
                                          row["est_elo"], (time.time() - t0) / 60))
    os.makedirs(os.path.join(OUT, "results"), exist_ok=True)
    json.dump({"anchors": ANCHOR_ELO, "games_per_side": games_per_side,
               "temperature": temperature, "rows": rows},
              open(os.path.join(OUT, "results", "elo.json"), "w"), indent=1)
    print("wrote", os.path.join(OUT, "results", "elo.json"))
    return rows


def check_baseline(rows):
    """Checkpoint 0 is an untrained head: vs a random mover it should be a coin flip."""
    base = [r for r in rows if r["ckpt"] == 0 and r["opponent"] == "random"]
    if not base:
        print("no checkpoint-0 vs random row to check")
        return True
    ok = all(0.40 <= r["score"] <= 0.60 for r in base)
    for r in base:
        print("%s untrained ckpt 0 vs random scored %.3f (expect 0.40-0.60)"
              % ("OK " if 0.40 <= r["score"] <= 0.60 else "BAD", r["score"]))
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", default="0,10,20")
    ap.add_argument("--runs", default="full,ablation")
    ap.add_argument("--games", type=int, default=50, help="games per colour")
    ap.add_argument("--parallel", type=int, default=16)
    ap.add_argument("--temperature", type=float, default=0.5)
    ap.add_argument("--opponents", default=None,
                    help="comma list; default all of %s"
                         % ",".join(o[0] for o in OPPONENTS))
    args = ap.parse_args()

    rows = run([int(x) for x in args.ckpts.split(",")],
               args.runs.split(","), args.games, args.parallel, args.temperature,
               args.opponents.split(",") if args.opponents else None)
    good = check_baseline(rows)
    print("SELFCHECK", "PASS" if good else "FAIL")
    raise SystemExit(0 if good else 1)
