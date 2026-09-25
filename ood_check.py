"""Does the fly head fall apart off the human-game distribution?

Standing puzzle as of 2026-09-16: on dataset positions the reservoir_only head
looks competent -- it hangs material 1.8% of the time against a random mover's
9.7%, and it answers a threat 82.5% of the time against random's 28.1% -- yet it
loses to that same random mover over the board (score 0.283).

Those measurements were all taken on positions from human games between 1500 and
2000 Elo. A game against a random mover leaves that distribution within a few
moves. The pooled vector correlates with board-space distance at r=0.08, so a
head reading it has no smooth structure to generalise along and nothing sensible
to say about a position unlike the ones it memorised.

This measures the same threat-response rate at increasing ply depth INSIDE real
games against a random mover, which is exactly the walk off-distribution. If the
theory holds, the rate starts near the dataset number and decays. If it stays
flat, the theory is wrong and the discrepancy is somewhere else.

Run: docker compose run --rm sim python -u ood_check.py --games 12
"""
import argparse
import multiprocessing as mp

import chess
import numpy as np

import behaviors as B
import play

PV = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
      chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}
BANDS = [(0, 20), (20, 40), (40, 80), (80, 300)]


def threatened(b):
    return [sq for sq, pc in b.piece_map().items()
            if pc.color == b.turn and PV[pc.piece_type] >= 3
            and b.is_attacked_by(not b.turn, sq)
            and not b.is_attacked_by(b.turn, sq)]


def saved(b, mv, sq):
    nb = b.copy()
    nb.push(mv)
    if mv.from_square == sq:
        return True
    if nb.piece_at(sq) is None:
        return True
    return not (nb.is_attacked_by(nb.turn, sq)
                and not nb.is_attacked_by(not nb.turn, sq))


_net = None


def _init():
    global _net
    _net = B.net()


def _one(job):
    ckpt, fly_white, seed = job
    player = play.FlyPlayer(ckpt, temperature=0.5, seed=seed, device="cpu")
    player.net = _net
    rng = np.random.default_rng(seed)
    board = chess.Board()
    hits = [[0, 0] for _ in BANDS]
    while not board.is_game_over(claim_draw=True) and board.ply() < 300:
        if board.turn == (chess.WHITE if fly_white else chess.BLACK):
            th = threatened(board)
            mv, _, _ = player.choose(board)
            if mv is None:
                break
            if th:
                for k, (lo, hi) in enumerate(BANDS):
                    if lo <= board.ply() < hi:
                        hits[k][0] += saved(board, mv, th[0])
                        hits[k][1] += 1
        else:
            lm = list(board.legal_moves)
            mv = lm[rng.integers(len(lm))]
        board.push(mv)
    return hits, play.score_for_fly(board, fly_white)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/reservoir_only/best.pt")
    ap.add_argument("--games", type=int, default=12)
    ap.add_argument("--parallel", type=int, default=8)
    args = ap.parse_args()

    B.net()
    jobs = [(args.ckpt, g % 2 == 0, 3000 + g) for g in range(args.games)]
    ctx = mp.get_context("fork")
    with ctx.Pool(args.parallel, initializer=_init) as pool:
        out = pool.map(_one, jobs)

    total = [[0, 0] for _ in BANDS]
    for hits, _ in out:
        for k in range(len(BANDS)):
            total[k][0] += hits[k][0]
            total[k][1] += hits[k][1]
    print("\n%s over %d games (score %.3f)"
          % (args.ckpt, args.games, sum(s for _, s in out) / len(out)))
    print("threat-response rate by ply band (dataset positions: 0.825):")
    for (lo, hi), (g, n) in zip(BANDS, total):
        print("  ply %3d-%-3d  %s  (%d/%d)"
              % (lo, hi, ("%.3f" % (g / n)) if n else "  -  ", g, n))
