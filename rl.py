"""REINFORCE fine-tuning of the head. The reservoir stays frozen.

This is the only place where reward changes anything. play.py --reward-viz shows
the fly sugar after the fact; here the win/loss signal actually moves weights -
but only the head's weights. No gradient touches the connectome, ever.

Runs entirely in the container on CPU: the sims are the cost and they are CPU
work anyway, and the head is small enough that a CPU step is free next to them.

Run: docker compose run --rm sim python -u rl.py --hours 6
"""
import argparse
import json
import multiprocessing as mp
import os
import time

import chess
import numpy as np
import torch
import torch.nn.functional as F

import behaviors as B
import encode
import play
import reservoir as R
import vocab

LR = 1e-5
ENTROPY_BETA = 0.01
CURRICULUM = [0, 1, 2, 3]          # Stockfish skill levels, stepped as we improve
CHECKPOINT_EVERY = 100             # games
MAX_PLIES = 200

_ctx = {}


def _init(ckpt, temperature):
    B.net()
    _ctx["player"] = play.FlyPlayer(ckpt, temperature=temperature)


def _rollout(job):
    """Play one game, returning the features/actions the fly chose along the way."""
    skill, fly_is_white, seed = job
    player = _ctx["player"]
    player.rng = np.random.default_rng(seed)
    opp, limit = play.opponent("stockfish", skill, movetime=0.02)
    board = chess.Board()
    feats, actions, masks = [], [], []
    try:
        while not board.is_game_over(claim_draw=True) and board.ply() < MAX_PLIES:
            if board.turn == (chess.WHITE if fly_is_white else chess.BLACK):
                res = R.simulate(board, n=player.net)
                x = player.features(res, board)
                mask, pairs = vocab.legal_mask(board)
                if not pairs:
                    break
                logits = player.model(
                    torch.from_numpy(x[None, :])).detach().numpy()[0].astype(np.float64)
                logits[~mask] = -np.inf
                z = logits / max(player.temperature, 1e-3)
                z -= z.max()
                p = np.exp(z)
                p /= p.sum()
                idx = int(player.rng.choice(len(p), p=p))
                move = vocab.idx_to_move(board, idx)
                if move is None or move not in board.legal_moves:
                    idx, move = max(pairs, key=lambda pr: logits[pr[0]])
                feats.append(x.astype(np.float16))
                actions.append(idx)
                masks.append(np.flatnonzero(mask).astype(np.int32))
                board.push(move)
            else:
                board.push(opp.play(board, limit).move)
    finally:
        opp.quit()
    score = play.score_for_fly(board, fly_is_white)
    reward = {1.0: 1.0, 0.5: 0.0, 0.0: -1.0}[score]
    return np.array(feats, dtype=np.float16), np.array(actions), masks, reward, score


def step(model, opt, batch, baseline):
    """One REINFORCE update over the moves of a batch of finished games."""
    losses, n_moves = [], 0
    opt.zero_grad(set_to_none=True)
    for feats, actions, masks, reward, _score in batch:
        if len(actions) == 0:
            continue
        advantage = reward - baseline
        x = torch.from_numpy(feats.astype(np.float32))
        logits = model(x)
        neg_inf = torch.full_like(logits, float("-inf"))
        keep = torch.zeros_like(logits, dtype=torch.bool)
        for row, legal in enumerate(masks):
            keep[row, torch.from_numpy(legal.astype(np.int64))] = True
        logits = torch.where(keep, logits, neg_inf)
        logp = F.log_softmax(logits, dim=1)
        chosen = logp[torch.arange(len(actions)), torch.from_numpy(actions)]
        entropy = -(logp.exp() * torch.nan_to_num(logp, neginf=0.0)).sum(dim=1).mean()
        losses.append(-(advantage * chosen.mean()) - ENTROPY_BETA * entropy)
        n_moves += len(actions)
    if not losses:
        return 0.0, 0
    loss = torch.stack(losses).mean()
    loss.backward()
    opt.step()
    return float(loss), n_moves


def train(ckpt, hours, parallel, temperature, out_dir):
    """Rollout batches until the wall-clock cap; checkpoint every ~100 games.

    The workers hold their own copy of the head, so after every checkpoint the
    pool is torn down and rebuilt from the file just written. That is the whole
    reason for the restart: a stale worker would sample from an old policy and
    quietly bias the gradient.
    """
    os.makedirs(out_dir, exist_ok=True)
    model, meta = play.load_head(ckpt)
    for p in model.parameters():
        p.requires_grad_(True)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    B.net()
    ctx = mp.get_context("fork")
    deadline = time.time() + hours * 3600
    baseline, games, history, skill_i = 0.0, 0, [], 0
    worker_ckpt = ckpt

    while time.time() < deadline:
        pool = ctx.Pool(parallel, initializer=_init, initargs=(worker_ckpt, temperature))
        try:
            since_checkpoint = 0
            while time.time() < deadline and since_checkpoint < CHECKPOINT_EVERY:
                jobs = [(CURRICULUM[skill_i], g % 2 == 0, games + g)
                        for g in range(parallel)]
                batch = pool.map(_rollout, jobs)
                mean_score = float(np.mean([b[4] for b in batch]))
                baseline = 0.9 * baseline + 0.1 * float(np.mean([b[3] for b in batch]))
                loss, n_moves = step(model, opt, batch, baseline)
                games += len(batch)
                since_checkpoint += len(batch)
                history.append({"games": games, "skill": CURRICULUM[skill_i],
                                "score": round(mean_score, 3), "loss": round(loss, 4),
                                "baseline": round(baseline, 3), "moves": n_moves})
                print("games %4d  skill %d  score %.3f  baseline %+.3f  loss %+.4f  "
                      "%d moves  %.0f min left"
                      % (games, CURRICULUM[skill_i], mean_score, baseline, loss,
                         n_moves, (deadline - time.time()) / 60))
        finally:
            pool.close()
            pool.join()

        path = os.path.join(out_dir, "rl_%05d.pt" % games)
        torch.save({"model": model.state_dict(), "meta": meta}, path)
        worker_ckpt = os.path.join(out_dir, "best.pt")
        torch.save({"model": model.state_dict(), "meta": meta}, worker_ckpt)
        print("  wrote", path)
        if history and history[-1]["score"] > 0.6 and skill_i + 1 < len(CURRICULUM):
            skill_i += 1
            print("  curriculum -> Stockfish skill", CURRICULUM[skill_i])
    return history


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/full/best.pt")
    ap.add_argument("--hours", type=float, default=6.0)
    ap.add_argument("--parallel", type=int, default=16)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--out", default="checkpoints/rl")
    args = ap.parse_args()

    hist = train(args.ckpt, args.hours, args.parallel, args.temperature, args.out)
    os.makedirs("results", exist_ok=True)
    json.dump({"lr": LR, "entropy_beta": ENTROPY_BETA, "curriculum": CURRICULUM,
               "start_ckpt": args.ckpt, "history": hist},
              open("results/rl.json", "w"), indent=1)
    print("wrote results/rl.json after %d batches" % len(hist))
    print("Now re-run elo.py including checkpoints/rl/best.pt. If it does not beat "
          "the imitation checkpoint by more than 50 Elo, ship the imitation one "
          "and say so.")
