"""Play a game: board -> brain -> head -> move, and record what the brain did.

REAL: the 300 ms sim behind every move. TRAINED: the head. AUTHORED: the
temperature, and the reward-visualisation thresholds.

`--reward-viz` is the literal basis of the launch story and it is honest in a
specific way: the sugar and bitter drives are real gustatory neurons, the second
sim really happens, and the ingestion motor neuron rate it reports is measured -
but the reward is shown to the fly AFTER the move is already chosen. It teaches
the fly nothing. rl.py is where reward actually changes weights.

Run: docker compose run --rm sim python -u play.py \
        --ckpt checkpoints/full/best.pt --vs stockfish --skill 0 \
        --games 1 --record games/demo1 --reward-viz
"""
import argparse
import json
import os
import time

import chess
import chess.engine
import chess.pgn
import numpy as np
import torch

import behaviors as B
import encode
import reservoir as R
import vocab

STOCKFISH = os.environ.get("STOCKFISH", "/usr/games/stockfish")

# Readouts that are derived from the raw `live` rates rather than read straight
# out of the sim. A head trained on one of these needs all_views=True sims.
NAMED_VIEWS = ("named", "named_nodriven")

# --reward-viz: a move losing less than this many centipawns earns sugar.
GOOD_MOVE_CP = 50
REWARD_HZ = 150.0
REWARD_MS = R.T_RUN
EVAL_MS = 50


def load_head(path, device="cpu"):
    blob = torch.load(path, map_location=device, weights_only=False)
    meta = blob["meta"]
    import train
    model = train.head(meta["d_in"]).to(device)
    model.load_state_dict(blob["model"])
    model.eval()
    return model, meta


class FlyPlayer:
    def __init__(self, ckpt, use_reservoir=None, temperature=1.0, record_dir=None,
                 device="cpu", seed=0):
        self.model, self.meta = load_head(ckpt, device)
        self.device = device
        self.use_reservoir = (self.meta["use_reservoir"] if use_reservoir is None
                              else use_reservoir)
        self.temperature = temperature
        self.record_dir = record_dir
        self.rng = np.random.default_rng(seed)
        self.net = B.net()
        self.ply = 0
        self._member, self._nbins = None, None
        if record_dir:
            os.makedirs(os.path.join(record_dir, "moves"), exist_ok=True)

    def mode(self):
        """Checkpoints written before reservoir_only existed carry no `mode`."""
        return self.meta.get("mode",
                             "full" if self.use_reservoir else "ablation")

    def readout_view(self):
        """Which readout the head was trained on. Pre-`view` checkpoints are pooled."""
        return self.meta.get("view", "pooled")

    def _binned(self, res):
        """Live rates -> the named readout this checkpoint was trained on.

        The partition is a fixed function of the neuron tables, so it is built
        once per game, not once per move.
        """
        import readout
        view = self.readout_view()
        if self._member is None:
            self._member, self._nbins, _ = readout.partition(
                exclude_driven=view == "named_nodriven")
        if "live" not in res:
            raise RuntimeError("checkpoint wants the %r readout but this sim "
                               "returned no `live` rates; call "
                               "reservoir.simulate(..., all_views=True)" % view)
        one = np.asarray(res["live"], dtype=np.float32)[None, :]
        return readout.apply(one, self._member, self._nbins)[0]

    def features(self, res, board):
        mode = self.mode()
        planes = encode.board_planes(board).reshape(-1)
        if mode == "ablation" or not self.use_reservoir:
            return planes

        norm = self.meta.get("norm")
        if norm is None:
            raise RuntimeError("checkpoint has no norm block; feeding raw Hz to "
                               "a head trained on z-scores plays nonsense while "
                               "looking fine")
        raw = (self._binned(res) if self.readout_view() in NAMED_VIEWS
               else np.asarray(res["pooled"], dtype=np.float32))
        lg = np.log1p(np.clip(raw, 0, None))
        lg = ((lg - np.asarray(norm["mu"], dtype=np.float32))
              / np.asarray(norm["sd"], dtype=np.float32)).astype(np.float32)
        # Mirror train.py exactly. Without this a bin that was flat across the
        # training set divides a normal fluctuation by ~0; measured 4.9e6 on a
        # head trained to expect z-scores, which plays shuffling draws and looks
        # like a finding rather than a bug.
        dead = norm.get("dead")
        if dead is not None:
            lg[np.asarray(dead, dtype=bool)] = 0.0
        clip = norm.get("clip")
        if clip is not None:
            np.clip(lg, -clip, clip, out=lg)
        if mode == "reservoir_only":
            return lg
        return np.concatenate([planes, lg])

    def needs_sim(self):
        """An ablation head reads board planes only; simulating for it burns
        0.9 s of wall clock per move on a value that features() discards."""
        return self.mode() != "ablation" and self.use_reservoir

    def policy(self, board):
        """(logits over the 1858 vocab, mask, pairs, res). One sim per call
        unless the head is planes-only. `logits[~mask]` is already -inf."""
        if self.needs_sim():
            res = R.simulate(board, full=self.record_dir is not None, n=self.net,
                             all_views=self.readout_view() in NAMED_VIEWS)
        elif self.record_dir:
            raise RuntimeError("a planes-only checkpoint runs no sim, so "
                               "--record would write an npz with no rates, "
                               "spike_t or dn_table; record with a fly head")
        else:
            res = {}
        x = torch.from_numpy(self.features(res, board)[None, :]).to(self.device)
        with torch.no_grad():
            logits = self.model(x)[0].cpu().numpy().astype(np.float64)
        mask, pairs = vocab.legal_mask(board)
        if pairs:
            logits[~mask] = -np.inf
        return logits, mask, pairs, res

    def topk_moves(self, board, k):
        """The k highest-logit legal moves, best first. Always legal, never
        empty when the board has a legal move.

        Ranks over `pairs` - the legal (idx, move) list - not over an argsort of
        the 1858 logits, because idx_to_move returns None for indices that are
        not legal here and the list would silently shrink below k.
        """
        logits, _mask, pairs, res = self.policy(board)
        ranked = sorted(pairs, key=lambda pr: -logits[pr[0]])
        return [m for _i, m in ranked[:k]], res

    def choose(self, board):
        """(move, top5, payload). One brain sim per move, always."""
        logits, _mask, pairs, res = self.policy(board)
        if not pairs:
            return None, [], res
        if self.temperature <= 0:
            idx = int(np.argmax(logits))
        else:
            z = logits / self.temperature
            z -= z.max()
            p = np.exp(z)
            p /= p.sum()
            idx = int(self.rng.choice(len(p), p=p))
        move = vocab.idx_to_move(board, idx)
        if move is None or move not in board.legal_moves:
            move = max(pairs, key=lambda pr: logits[pr[0]])[1]

        order = np.argsort(logits)[::-1][:5]
        probs = np.exp(logits[order] - logits[order].max())
        probs /= probs.sum()
        top5 = [{"uci": (vocab.idx_to_move(board, int(i)) or chess.Move.null()).uci(),
                 "idx": int(i), "p": float(pp)} for i, pp in zip(order, probs)
                if np.isfinite(logits[int(i)])]
        return move, top5, res

    def record(self, board, move, top5, res):
        """games/<id>/moves/NNN.npz with the frozen spec-3 keys."""
        if not self.record_dir:
            return None
        path = os.path.join(self.record_dir, "moves", "%03d.npz" % self.ply)
        np.savez_compressed(
            path,
            rates=res["rates"], spike_t=res["spike_t"], spike_i=res["spike_i"],
            fen=np.array(board.fen()), move=np.array(move.uci()),
            top5=np.array(json.dumps(top5)),
            dn_table=np.array(json.dumps(res["dn_table"])),
            sensory_summary=np.array(json.dumps(res["sensory_summary"])),
        )
        self.ply += 1
        return path


def show_reward(player, board_after, cp_loss, path_stem):
    """Second sim: sugar if the move was good, bitter if it was not.

    Shown to the fly, not learned from. The label and the ingestion motor rate
    are both measured from this extra run.
    """
    fed = cp_loss is not None and cp_loss < GOOD_MOVE_CP
    sel = encode.SUGAR if fed else encode.BITTER
    n = player.net
    from flypoke import sim as S
    idx = n.select(sel)
    rec = S.run(n, [S.Stimulus(idx, REWARD_HZ)], S.Params(t_run=REWARD_MS),
                n_trials=1, seed=R.seed_for(board_after))
    rates = rec.rates(n.n)
    ingest = float(rates[n.select(R.INGESTION)].mean())
    np.savez_compressed(
        path_stem + "reward.npz",
        rates=rates.astype(np.float16),
        spike_t=rec.times.astype(np.float32), spike_i=rec.ids.astype(np.int32),
        label=np.array("fed" if fed else "bitter"),
        centipawn_loss=np.array(-1.0 if cp_loss is None else float(cp_loss)),
        ingestion_motor_hz=np.array(ingest),
        dn_table=np.array(json.dumps(R.dn(rates, n))),
    )
    return ("fed" if fed else "bitter"), ingest


def cp_loss_of(engine, board_before, move):
    """Centipawn loss of `move`, from the mover's point of view."""
    def score(board):
        info = engine.analyse(board, chess.engine.Limit(time=EVAL_MS / 1000.0))
        return info["score"].pov(board_before.turn).score(mate_score=10000)
    try:
        before = score(board_before)
        after_board = board_before.copy()
        after_board.push(move)
        after = score(after_board)
        return max(0, before - after)
    except (chess.engine.EngineError, KeyError, TypeError):
        return None


class RandomMover:
    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def play(self, board, *_a, **_k):
        moves = list(board.legal_moves)
        return chess.engine.PlayResult(moves[self.rng.integers(len(moves))], None)

    def quit(self):
        pass

    def configure(self, *_a, **_k):
        pass


def opponent(kind, skill=0, movetime=0.02):
    if kind == "random":
        return RandomMover(), chess.engine.Limit(time=movetime)
    eng = chess.engine.SimpleEngine.popen_uci(STOCKFISH)
    eng.configure({"Skill Level": int(skill)})
    return eng, chess.engine.Limit(time=movetime)


def play_game(player, opp, limit, fly_is_white=True, reward_engine=None,
              record_dir=None, max_plies=300, verbose=True):
    board = chess.Board()
    game = chess.pgn.Game()
    node = game
    t0 = time.time()
    while not board.is_game_over(claim_draw=True) and board.ply() < max_plies:
        if board.turn == (chess.WHITE if fly_is_white else chess.BLACK):
            move, top5, res = player.choose(board)
            if move is None:
                break
            move_path = player.record(board, move, top5, res) if record_dir else None
            before = board.copy()
            board.push(move)
            if reward_engine is not None and move_path:
                loss = cp_loss_of(reward_engine, before, move)
                label, ingest = show_reward(player, board, loss,
                                            move_path[:-len(".npz")] + "_")
                if verbose:
                    print("   fly %-6s cp_loss=%s -> %s (ingestion %.1f Hz)"
                          % (move.uci(), loss, label, ingest))
            elif verbose:
                print("   fly %-6s top5 %s" % (move.uci(),
                                               [t["uci"] for t in top5[:3]]))
        else:
            move = opp.play(board, limit).move
            board.push(move)
        node = node.add_variation(move)
    game.headers["Result"] = board.result(claim_draw=True)
    game.headers["White"] = "fly" if fly_is_white else "opponent"
    game.headers["Black"] = "opponent" if fly_is_white else "fly"
    if verbose:
        print("   %s in %d plies (%.0f s)" % (game.headers["Result"], board.ply(),
                                              time.time() - t0))
    return board, game


def score_for_fly(board, fly_is_white):
    res = board.result(claim_draw=True)
    if res == "1/2-1/2" or res == "*":
        return 0.5
    win_white = res == "1-0"
    return 1.0 if win_white == fly_is_white else 0.0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--vs", default="stockfish", choices=["stockfish", "random"])
    ap.add_argument("--skill", type=int, default=0)
    ap.add_argument("--movetime", type=float, default=0.02)
    ap.add_argument("--games", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--record", default=None, help="directory for per-move npz")
    ap.add_argument("--reward-viz", action="store_true")
    ap.add_argument("--no-reservoir", action="store_true")
    args = ap.parse_args()

    player = FlyPlayer(args.ckpt, use_reservoir=not args.no_reservoir or None,
                       temperature=args.temperature, record_dir=args.record)
    opp, limit = opponent(args.vs, args.skill, args.movetime)
    reward_engine = None
    if args.reward_viz:
        reward_engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH)
        reward_engine.configure({"Skill Level": 20})

    total = 0.0
    for g in range(args.games):
        white = g % 2 == 0
        print("game %d, fly plays %s" % (g + 1, "white" if white else "black"))
        board, game = play_game(player, opp, limit, white, reward_engine, args.record)
        total += score_for_fly(board, white)
        if args.record:
            with open(os.path.join(args.record, "game_%02d.pgn" % g), "w") as fh:
                print(game, file=fh)
    print("fly scored %.1f / %d" % (total, args.games))
    opp.quit()
    if reward_engine:
        reward_engine.quit()
