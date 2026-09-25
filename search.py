"""Depth-2 alpha-beta over material: the baseline the fly is measured against.

AUTHORED: everything in this file - the depth, the eval, MATE, MOBILITY, the
tie-break. REAL: nothing. No sim, no checkpoint, no fly. The fly enters in
phase 2 through candidates(), which is the only seam that anticipates it.

Phase-1 numbers from this file are the search alone. Any later fly arm is a
delta on top of this row, not on top of zero.

Run: docker compose run --rm sim python -u search.py --games 50 --parallel 16
     uv run python search.py --selfcheck          # host, no flypoke needed
"""
import argparse
import json
import os
import random
import time

import chess

from encode import PIECE_VALUE

MATE = 100_000       # centipawns; beats any material swing
MOBILITY = 1         # cp per own legal reply. Authored. Breaks the
                     # all-quiet-moves tie that otherwise shuffles won games
                     # into the 300-ply cap.

OUT = os.environ.get("FLYCHESS_OUT", ".")

ARMS = ("none", "fly", "random", "planes")
# Authored: which checkpoint each arm's prior comes from. The fly arm is the
# 0.410-vs-random head from runs/fix_nodriven; any other checkpoint makes the
# phase-4 table incomparable to the plan's Evidence section. checkpoints/full is
# planes AND fly, so it is neither control nor treatment and is not an arm.
CKPT = {"fly": "runs/fix_nodriven/checkpoints/reservoir_only/best.pt",
        "planes": "checkpoints/ablation/best.pt"}


def evaluate(board, moves=None):
    """Side-to-move POV, centipawns. Terminal nodes are the caller's job.

    `moves` is the already-generated legal move list, because generating it
    twice per leaf is the single biggest cost in this file.
    """
    mat = 0
    for piece in board.piece_map().values():
        v = PIECE_VALUE[piece.piece_type] * 100
        mat += v if piece.color == board.turn else -v
    # ponytail: own mobility only (it is the side to move); the opponent's comes
    # free through the negation one ply up
    n = len(moves) if moves is not None else board.legal_moves.count()
    return mat + MOBILITY * n


def negamax(board, depth, alpha, beta, counter):
    counter[0] += 1
    moves = list(board.legal_moves)
    if not moves:
        # mate (faster is better, hence -depth) or stalemate
        return -MATE - depth if board.is_check() else 0
    # Draws, cheapest test first. `halfmove_clock` gates the two expensive ones:
    # a threefold needs >= 8 reversible plies, a fifty-move claim needs 100.
    # ponytail: this is `is_repetition(3)`, not play_game's claim_draw=True,
    # which also pushes every legal move looking for a claimable repetition -
    # too expensive per node, and worth < 1 cp of search quality here.
    if board.is_insufficient_material() or board.halfmove_clock >= 100:
        return 0
    if board.halfmove_clock >= 8 and board.is_repetition(3):
        return 0
    if depth == 0:
        return evaluate(board, moves)
    best = -MATE * 2
    for move in moves:
        board.push(move)
        score = -negamax(board, depth - 1, -beta, -alpha, counter)
        board.pop()
        if score > best:
            best = score
        if best > alpha:
            alpha = best
        if alpha >= beta:
            break
    return best


def search(board, candidates, depth, rng):
    """(move, {uci: score}, nodes). Root over `candidates` only, ties by rng.

    Full window per root candidate on purpose: phase 3 needs an exact score for
    every candidate, and the root is bounded at ~40 moves.
    """
    counter = [0]
    scores = {}
    for move in candidates:
        board.push(move)
        scores[move.uci()] = -negamax(board, depth - 1, -MATE * 2, MATE * 2, counter)
        board.pop()
    top = max(scores.values())
    best = [m for m in candidates if scores[m.uci()] == top]
    return rng.choice(best), scores, counter[0]


class SearchPlayer:
    """Drop-in for play.FlyPlayer: choose() -> (move, top5, res).

    The prior picks WHICH moves the search looks at; the search picks among
    them. At k >= the legal move count the prior has chosen nothing.
    """

    def __init__(self, depth=2, prune="none", k=None, seed=0, record_dir=None,
                 ckpt=None):
        if prune not in ARMS:
            raise SystemExit("prune=%r is not one of %s" % (prune, ARMS))
        # Before any import of play: this must fail on the host too, where
        # play/behaviors do not exist.
        if prune in ("fly", "planes") and k is None:
            raise SystemExit("--k is required for prune=%s" % prune)
        self.depth, self.prune, self.k = depth, prune, k
        self.ckpt = ckpt or CKPT.get(prune)
        self.rng = random.Random(seed)
        self.record_dir = record_dir
        self.nodes, self.moves, self.seconds = 0, 0, 0.0
        self.prior_seconds = 0.0
        self._fly = None

    def fly(self):
        """Lazy, so prune=none/random never import play or load a checkpoint."""
        if self._fly is None:
            import play
            # temperature=0: the prior is a ranking, so sampling is irrelevant
            # and would only make topk_moves non-deterministic.
            self._fly = play.FlyPlayer(self.ckpt, temperature=0, seed=0)
        return self._fly

    def candidates(self, board):
        """The prior's top-k legal moves, or all of them for prune=none."""
        legal = list(board.legal_moves)
        # ponytail: k >= len(legal) skips the prior entirely, so the fly arm
        # runs no sim on a forced move. If phase 4 wants strict per-move sim
        # parity between arms, move this return below the prior call - one line.
        if self.prune == "none" or self.k is None or self.k >= len(legal):
            return legal
        t0 = time.time()
        if self.prune == "random":
            # shares self.rng with the search tie-break on purpose: one seeded
            # stream per player. Reproducible; do not add a second Random.
            out = self.rng.sample(legal, self.k)
        else:
            out, _res = self.fly().topk_moves(board, self.k)
        self.prior_seconds += time.time() - t0
        return out or legal          # never hand search() an empty list

    def choose(self, board):
        cands = self.candidates(board)
        if not cands:
            return None, [], {}
        t0 = time.time()
        move, scores, nodes = search(board, cands, self.depth, self.rng)
        self.seconds += time.time() - t0
        self.nodes += nodes
        self.moves += 1
        order = sorted(scores.items(), key=lambda kv: -kv[1])[:5]
        top5 = [{"uci": u, "score": s} for u, s in order]
        return move, top5, {"nodes": nodes, "depth": self.depth}

    def record(self, board, move, top5, res):
        # ponytail: no per-move npz without a sim; phase 2 records via the fly.
        # Exists so play.py --player search cannot crash on a missing method.
        return None


# ---------------------------------------------------------------- selfcheck

def _line(good, msg):
    print("%s %s" % ("OK " if good else "BAD", msg))
    return good


def _raises(fn):
    try:
        fn()
    except SystemExit:
        return True
    return False


def selfcheck(verbose=True):
    """Assert-based checks. Host-runnable: imports nothing from play/reservoir."""
    ok = True

    # mate in 1 (back rank)
    b = chess.Board("6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1")
    assert b.is_valid()
    move, top5, _ = SearchPlayer(seed=0).choose(b)
    ok &= _line(move.uci() == "a1a8",
                "mate-in-1 played %s (want a1a8, score %d)"
                % (move.uci(), top5[0]["score"]))

    # free queen
    b = chess.Board("4k3/8/8/3q4/8/8/8/3RK3 w - - 0 1")
    assert b.is_valid()
    move, _, _ = SearchPlayer(seed=0).choose(b)
    ok &= _line(move.uci() == "d1d5", "free queen taken by %s (want d1d5)" % move.uci())

    # defended pawn: Qxd5 loses the queen to cxd5, so it must not be chosen
    b = chess.Board("2k5/8/2p5/3p4/8/8/8/3QK3 w - - 0 1")
    assert b.is_valid()
    p = SearchPlayer(seed=0)
    move, _, _ = p.choose(b)
    _, scores, _ = search(b, list(b.legal_moves), 2, random.Random(0))
    ok &= _line(move.uci() != "d1d5" and scores["d1d5"] < max(scores.values()),
                "defended pawn: played %s, Qxd5 scores %d vs best %d"
                % (move.uci(), scores["d1d5"], max(scores.values())))

    # exactly one legal move
    b = chess.Board("k7/7R/8/8/8/8/8/K7 b - - 0 1")
    assert b.is_valid()
    move, _, _ = SearchPlayer(seed=0).choose(b)
    ok &= _line(b.legal_moves.count() == 1 and move in b.legal_moves,
                "single legal move returned %s" % move.uci())

    # game over -> (None, [], {})
    b = chess.Board("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3")
    assert b.is_valid() and b.is_checkmate()
    move, top5, res = SearchPlayer(seed=0).choose(b)
    ok &= _line(move is None and top5 == [] and res == {},
                "checkmated board -> (None, [], {})")

    # legality + node counter, d1 < d2
    b = chess.Board()
    p1, p2 = SearchPlayer(depth=1, seed=0), SearchPlayer(depth=2, seed=0)
    legal = True
    for p in (p1, p2):
        bb = chess.Board()
        for _ in range(2):
            m, _, _ = p.choose(bb)
            legal &= m in bb.legal_moves
            bb.push(m)
            bb.push(next(iter(bb.legal_moves)))
    ok &= _line(legal and 0 < p1.nodes < p2.nodes,
                "2 plies legal at both depths, nodes d1=%d < d2=%d"
                % (p1.nodes, p2.nodes))

    # determinism
    b = chess.Board()
    m_a, _, _ = SearchPlayer(seed=0).choose(b)
    m_b, _, _ = SearchPlayer(seed=0).choose(b)
    ok &= _line(m_a == m_b, "seed 0 twice -> %s twice" % m_a.uci())

    # mobility term live: with material alone every quiet move here ties
    b = chess.Board("3k4/8/8/8/8/8/8/R3K2R w - - 0 1")
    assert b.is_valid()
    _, scores, _ = search(b, list(b.legal_moves), 2, random.Random(0))
    spread = max(scores.values()) - min(scores.values())
    ok &= _line(spread > 0, "mobility separates %d quiet moves, spread %d cp"
                % (len(scores), spread))

    # ---- arms. Nothing below may touch self.fly(): this runs on the host,
    # where play and behaviors do not exist.
    start = chess.Board()

    p = SearchPlayer(prune="random", k=4, seed=0)
    cands = p.candidates(start)
    ok &= _line(len(cands) == 4 and all(m in start.legal_moves for m in cands)
                and SearchPlayer(prune="random", k=4, seed=0).choose(start)[0]
                in start.legal_moves,
                "random arm kept %d of %d legal moves, all legal"
                % (len(cands), start.legal_moves.count()))

    wide = SearchPlayer(prune="random", k=99, seed=0).candidates(start)
    ok &= _line(len(wide) == start.legal_moves.count(),
                "k=99 >= 20 legal -> all %d moves, prior skipped" % len(wide))

    b = chess.Board("k7/7R/8/8/8/8/8/K7 b - - 0 1")
    assert b.is_valid()
    one = SearchPlayer(prune="random", k=4, seed=0).candidates(b)
    mv, _, _ = SearchPlayer(prune="random", k=4, seed=0).choose(b)
    ok &= _line(len(one) == 1 and mv == one[0],
                "single legal move survives pruning: %s" % mv.uci())

    only = SearchPlayer(prune="random", k=1, seed=7).candidates(start)
    mv, _, _ = SearchPlayer(prune="random", k=1, seed=7).choose(start)
    ok &= _line(len(only) == 1 and mv == only[0],
                "k=1 -> prior alone decides, search played %s" % mv.uci())

    a = SearchPlayer(prune="random", k=4, seed=0).candidates(start)
    c = SearchPlayer(prune="random", k=4, seed=0).candidates(start)
    ok &= _line(a == c, "seed 0 twice -> identical candidates %s"
                % [m.uci() for m in a])

    ok &= _line(_raises(lambda: SearchPlayer(prune="stockfish")),
                "unknown arm raises SystemExit")
    # must fail on the missing k before any import of play
    ok &= _line(_raises(lambda: SearchPlayer(prune="fly")),
                "prune=fly without --k raises SystemExit, loads no checkpoint")

    b = chess.Board("6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1")
    mv, _, _ = SearchPlayer(prune="random", k=99, seed=0).choose(b)
    ok &= _line(mv.uci() == "a1a8",
                "mate-in-1 still found through the pruning path (%s)" % mv.uci())

    if verbose:
        print("    start position evaluates to %d cp (material 0 + mobility)"
              % evaluate(chess.Board()))
    return bool(ok)


def selfcheck_fly(verbose=True):
    """The checks that load a checkpoint. Container only: needs flypoke."""
    import play
    ok = True
    start = chess.Board()

    p = SearchPlayer(prune="fly", k=8, seed=0)
    t0 = time.time()
    cands = p.candidates(start)
    fly_s = (time.time() - t0)
    ok &= _line(len(cands) == 8 and all(m in start.legal_moves for m in cands)
                and len(set(m.uci() for m in cands)) == 8,
                "fly kept 8 distinct legal moves: %s"
                % [m.uci() for m in cands])

    again = SearchPlayer(prune="fly", k=8, seed=0).candidates(start)
    ok &= _line(cands == again,
                "fly prior deterministic (seed_for(board)) across players")

    ok &= _line(p.fly().needs_sim() and fly_s > 0.3,
                "fly arm runs the sim: needs_sim=True, %.2f s for the prior"
                % fly_s)

    q = SearchPlayer(prune="planes", k=8, seed=0)
    t0 = time.time()
    pc = q.candidates(start)
    planes_s = time.time() - t0
    ok &= _line(not q.fly().needs_sim() and planes_s < 0.1 and len(pc) == 8,
                "planes arm skips the sim: needs_sim=False, %.3f s for the prior"
                % planes_s)

    r = SearchPlayer(prune="fly", k=8, seed=0)
    opp, limit = play.opponent("random", 0, movetime=0.02)
    try:
        board, _ = play.play_game(r, opp, limit, True, reward_engine=None,
                                  record_dir=None, verbose=False)
    finally:
        opp.quit()
    per_move = (r.seconds + r.prior_seconds) / max(r.moves, 1)
    ok &= _line(board.is_valid() and r.moves > 0 and per_move <= 2.0,
                "legal game vs random: %s in %d plies, %.2f s/move (<= 2.0)"
                % (board.result(claim_draw=True), board.ply(), per_move))
    return bool(ok)


# ------------------------------------------------------------------- ladder

_player_cache = {}


def _init():
    """Build the connectome once per worker, not once per checkpoint load."""
    import behaviors as B      # container-only import
    B.net()


def _one_game(job):
    depth, prune, k, ckpt, opp_name, skill, fly_is_white, seed = job
    import play
    # One player per worker for the head arms: each holds a FlyPlayer, and four
    # of those per worker x 16 workers thrashes memory. none/random keep the
    # phase-1 seed % 4 so their rows stay reproducible.
    slot = 0 if prune in ("fly", "planes") else seed % 4
    key = (depth, prune, k, ckpt, slot)
    if key not in _player_cache:
        _player_cache[key] = SearchPlayer(depth=depth, prune=prune, k=k,
                                          seed=seed, ckpt=ckpt)
    player = _player_cache[key]
    kind = "random" if opp_name == "random" else "stockfish"
    opp, limit = play.opponent(kind, skill or 0, movetime=0.02)
    n0, m0, s0 = player.nodes, player.moves, player.seconds
    p0 = player.prior_seconds
    try:
        board, _ = play.play_game(player, opp, limit, fly_is_white,
                                  reward_engine=None, record_dir=None, verbose=False)
    finally:
        opp.quit()
    return (play.score_for_fly(board, fly_is_white), player.nodes - n0,
            player.moves - m0, player.seconds - s0, board.result(claim_draw=True),
            player.prior_seconds - p0)


def run(depth, prune, k, ckpt, opp_name, games, parallel):
    """One (arm, opponent) block, appended to results/search.json."""
    import multiprocessing as mp

    import elo
    skill = {"random": None, "stockfish_skill0": 0}[opp_name]
    if prune == "none":
        k, ckpt = None, None      # the k = all legal moves point of the sweep
    ckpt = ckpt or CKPT.get(prune)
    jobs = [(depth, prune, k, ckpt, opp_name, skill, white, 1000 + g)
            for white in (True, False) for g in range(games)]
    t0 = time.time()
    ctx = mp.get_context("fork") if hasattr(os, "fork") else mp.get_context()
    # B.net() is the slowest thing in the file; a none/random run must not pay it.
    init = _init if prune in ("fly", "planes") else None
    with ctx.Pool(parallel, initializer=init) as pool:
        out = list(pool.imap_unordered(_one_game, jobs, 1))
    scores = [o[0] for o in out]
    w = sum(1 for s in scores if s == 1.0)
    d = sum(1 for s in scores if s == 0.5)
    lo = len(scores) - w - d
    unfinished = sum(1 for o in out if o[4] == "*")
    nodes, moves, secs, prior = (sum(o[i] for o in out) for i in (1, 2, 3, 5))
    mean = sum(scores) / len(scores)
    row = {"prune": prune, "k": k, "depth": depth, "ckpt": ckpt,
           "opponent": opp_name,
           "games": len(scores), "w": w, "d": d, "l": lo, "unfinished": unfinished,
           "score": round(mean, 3),
           "est_elo": round(elo.est_elo(mean, elo.ANCHOR_ELO[opp_name])),
           "nodes_per_move": round(nodes / max(moves, 1), 1),
           "s_per_move": round(secs / max(moves, 1), 3),
           "prior_s_per_move": round(prior / max(moves, 1), 3),
           "date": time.strftime("%Y-%m-%d")}
    print("prune=%-6s k=%-4s d=%d vs %-17s  %d-%d-%d (%d unfinished)  score %.3f  "
          "est %d Elo  %.0f nodes/move  %.2f s/move  (prior %.2f)  (%.1f min)"
          % (prune, k, depth, opp_name, w, d, lo, unfinished, mean, row["est_elo"],
             row["nodes_per_move"], row["s_per_move"], row["prior_s_per_move"],
             (time.time() - t0) / 60))
    os.makedirs(os.path.join(OUT, "results"), exist_ok=True)
    path = os.path.join(OUT, "results", "search.json")
    rows = json.load(open(path))["rows"] if os.path.exists(path) else []
    rows.append(row)
    json.dump({"rows": rows}, open(path, "w"), indent=1)
    print("wrote", path)
    return row


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=50, help="games per colour")
    ap.add_argument("--parallel", type=int, default=16)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--prune", default="none", choices=list(ARMS))
    ap.add_argument("--k", type=int, default=None,
                    help="candidates the prior keeps; ignored by prune=none")
    ap.add_argument("--ckpt", default=None, help="override the arm default in CKPT")
    ap.add_argument("--vs", default="random",
                    choices=["random", "stockfish_skill0"])
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--selfcheck-fly", action="store_true",
                    help="container only: the checks that load a checkpoint")
    args = ap.parse_args()

    if args.selfcheck or args.selfcheck_fly:
        good = selfcheck_fly() if args.selfcheck_fly else selfcheck()
        print("SELFCHECK", "PASS" if good else "FAIL")
        raise SystemExit(0 if good else 1)

    # Same guard as SearchPlayer.__init__, hoisted here so a bad invocation
    # exits 2 with a usage message instead of raising SystemExit - a
    # BaseException - inside a pool worker, which hangs the ladder forever.
    if args.prune in ("fly", "planes") and args.k is None:
        ap.error("--k is required for --prune %s" % args.prune)
    run(args.depth, args.prune, args.k, args.ckpt, args.vs, args.games,
        args.parallel)
