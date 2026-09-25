"""Train the only learned weights in the project: a small MLP head.

TRAINED: this file, and nothing else. The reservoir is frozen - no gradient ever
reaches the connectome, which is the whole point of calling it a reservoir.

The ablation run is mandatory and not optional: the identical head trained on the
board planes alone. If the two Elos come out the same, the honest claim is "the
fly brain reacts, the trained layer decides", not "fly wiring helps chess".

Runs on the host (Windows, uv, CUDA) because it is the one GPU-shaped step:
    uv run train.py --run full
    uv run train.py --run ablation
"""
import argparse
import json
import os
import time

import chess
import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn as nn

import encode
import vocab

DATA = os.environ.get("FLYCHESS_DATA", "data")
SHARD_DIR = os.environ.get("FLYCHESS_SHARDS", os.path.join(DATA, "reservoir"))
# Where checkpoints and results land. A smoke run on a handful of shards writes
# real-looking files; on 2026-09-16 one overwrote reservoir_only's checkpoints
# and its epoch history. Point this somewhere disposable for any run whose
# numbers you would not quote.
OUT = os.environ.get("FLYCHESS_OUT", ".")

# Which readout of the simulation the head is fed. Measured 2026-09-16 by
# reconstructing piece-square occupancy from 5,000 identical sims at equal width:
# `pooled` (mean per cell_type, what the first 200k run stored) reaches F1 0.610,
# `proj` 0.852, `subset` 0.850. Shards written before that finding carry only
# `pooled`, so this defaults to it and falls back with a warning.
VIEW = os.environ.get("FLYCHESS_VIEW", "pooled")

# Readouts derived from the stored raw `live` rates rather than read straight
# out of the npz. `named_nodriven` drops the 13,686 neurons the encoder drives:
# flypoke makes a stimulated neuron a pure Poisson source and discards its
# network input, so reading one back is reading our own drive. Any claim about
# what the FLY carries has to come from that one.
NAMED_VIEWS = ("named", "named_nodriven")

# A feature column whose sd over 200,019 positions is below this never varied,
# so it carries nothing and its inverse is a catastrophe waiting at inference.
DEAD_SD = 1e-3
# Standardised features are bounded before the head sees them, in training and
# in play.py alike, so an out-of-distribution board cannot hand the head a
# number a thousand times larger than anything it was trained on.
CLIP = 8.0
PLANE_DIM = 18 * 8 * 8          # 1152

BATCH = 1024
EPOCHS = 20
LR = 3e-4
VAL_FRACTION = 0.1


def head(d_in, hidden=2048, mid=1024, dropout=0.2):
    return nn.Sequential(
        nn.Linear(d_in, hidden), nn.GELU(), nn.Dropout(dropout),
        nn.Linear(hidden, mid), nn.GELU(),
        nn.Linear(mid, vocab.N_MOVES),
    )


def load_shards(with_pooled=True):
    """(pooled float32 [N, G] or None, row_ids int32) in row order.

    Scatters each shard straight into its sorted slot in one preallocated
    array. The obvious concat-then-`pooled[order]` needs three copies of a
    7 GB array live at once, which does not fit in host RAM.
    """
    files = sorted(f for f in os.listdir(SHARD_DIR) if f.endswith(".npz"))
    if not files:
        raise SystemExit("no reservoir shards in %s; run precompute.py first" % SHARD_DIR)
    paths = [os.path.join(SHARD_DIR, f) for f in files]
    ids = np.concatenate([np.load(p)["row_ids"] for p in paths])
    order = np.argsort(ids)
    if not with_pooled:
        return None, ids[order]

    available = np.load(paths[0]).files
    view, xform = VIEW, None
    if view in NAMED_VIEWS:
        if "live" not in available:
            raise SystemExit("--view %s needs the raw 'live' rates; these shards "
                             "carry %s. Re-run precompute, or use --view pooled."
                             % (view, sorted(f for f in available if f != "row_ids")))
        import readout
        excl = view == "named_nodriven"
        member, n_bins, n_types = readout.partition(exclude_driven=excl)
        print("named readout: %d bins over %d live cell types%s"
              % (n_bins, n_types, ", driven neurons excluded" if excl else ""))
        xform = lambda blk: readout.apply(blk, member, n_bins)
        view = "live"
    elif view not in available:
        print("shards carry %s but not %r; falling back to 'pooled'"
              % (sorted(f for f in available if f != "row_ids"), view))
        view = "pooled"

    rank = np.empty(len(ids), dtype=np.int64)
    rank[order] = np.arange(len(ids))          # where each shard row lands
    pooled = None
    k = 0
    for p in paths:
        block = np.load(p)[view]
        if xform is not None:
            block = xform(block)          # per shard: the full live matrix is 11 GB
        if pooled is None:
            pooled = np.empty((len(ids), block.shape[1]), dtype=np.float32)
        pooled[rank[k:k + len(block)]] = block
        k += len(block)
    return pooled, ids[order]


MODES = ("full", "ablation", "reservoir_only")


def build_features(mode):
    """(X float32 [N, d], y int64 [N], stats) for the positions we have sims for.

    full           board planes + standardised log1p(pooled)
    ablation       board planes alone
    reservoir_only board alone reaches the head through fly neurons or not at all
    """
    if mode not in MODES:
        raise ValueError("mode must be one of %s, got %r" % (MODES, mode))
    use_reservoir = mode != "ablation"
    table = pq.read_table(os.path.join(DATA, "positions.parquet"),
                          columns=["fen", "move_uci"])
    fens = table.column("fen").to_pylist()
    ucis = table.column("move_uci").to_pylist()

    pooled, row_ids = load_shards(with_pooled=use_reservoir)
    print("%d positions sampled, %d with reservoir vectors" % (len(fens), len(row_ids)))

    planes = np.zeros((len(row_ids), PLANE_DIM), dtype=np.float32)
    y = np.zeros(len(row_ids), dtype=np.int64)
    keep = np.ones(len(row_ids), dtype=bool)
    for k, row in enumerate(row_ids):
        board = chess.Board(fens[row])
        planes[k] = encode.board_planes(board).reshape(-1)
        try:
            y[k] = vocab.move_to_idx(board, chess.Move.from_uci(ucis[row]))
        except (KeyError, ValueError):
            keep[k] = False
    dropped = int((~keep).sum())
    if dropped:
        print("dropped %d positions whose move is outside the 1858 vocab" % dropped)
        planes, y = planes[keep], y[keep]
        if pooled is not None:
            pooled = pooled[keep]

    stats = {"n": int(len(y)), "dropped": dropped}
    if not use_reservoir:
        return planes, y, stats

    # standardise log1p(pooled) in place; the copies are what blow out RAM
    np.clip(pooled, 0, None, out=pooled)
    np.log1p(pooled, out=pooled)
    mu, sd = pooled.mean(axis=0), pooled.std(axis=0)
    # A column that does not move across 200k positions is not a feature, and
    # dividing by its ~0 sd is how a z-score becomes 4.9e6 at inference: the
    # named readout bins only 2-5 neurons each, so plenty of bins are flat in
    # training and still fluctuate on an unseen board. Zero them in both paths
    # instead, and bound whatever is left.
    dead = sd <= DEAD_SD
    sd = np.where(dead, 1.0, sd)
    pooled -= mu
    pooled /= sd
    pooled[:, dead] = 0.0
    np.clip(pooled, -CLIP, CLIP, out=pooled)
    print("%d/%d dims flat in training (zeroed), features clipped to +-%g"
          % (int(dead.sum()), len(sd), CLIP))
    stats["reservoir_dim"] = int(pooled.shape[1])
    stats["norm"] = {"mu": mu.tolist(), "sd": sd.tolist(),
                     "dead": dead.tolist(), "clip": CLIP}
    if mode == "reservoir_only":
        return pooled, y, stats          # no board planes: the fly is the only path

    X = np.empty((len(y), PLANE_DIM + pooled.shape[1]), dtype=np.float32)
    X[:, :PLANE_DIM] = planes
    X[:, PLANE_DIM:] = pooled
    return X, y, stats


def train(run, epochs=EPOCHS, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    X, y, stats = build_features(run)
    n = len(y)
    rng = np.random.default_rng(0)
    perm = rng.permutation(n)
    n_val = int(n * VAL_FRACTION)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    Xt = torch.from_numpy(X)
    yt = torch.from_numpy(y)
    model = head(X.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    ckpt_dir = os.path.join(OUT, "checkpoints", run)
    os.makedirs(ckpt_dir, exist_ok=True)
    # `mode` is authoritative; `use_reservoir` stays for checkpoints written
    # before reservoir_only existed, which play.py must still be able to load.
    norm = stats.pop("norm", None)
    if run != "ablation" and norm is None:
        raise RuntimeError("%s needs the norm block; without it play.py would "
                           "feed raw Hz to a head trained on z-scores" % run)
    meta = {"run": run, "mode": run, "use_reservoir": run != "ablation",
            "view": VIEW,
            "d_in": int(X.shape[1]),
            "plane_dim": 0 if run == "reservoir_only" else PLANE_DIM,
            "norm": norm}
    torch.save({"model": model.state_dict(), "meta": meta},
               os.path.join(ckpt_dir, "epoch_00.pt"))   # genuinely untrained

    history = []
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        order = tr_idx[torch.randperm(len(tr_idx)).numpy()]
        total = 0.0
        for i in range(0, len(order), BATCH):
            b = order[i:i + BATCH]
            xb, yb = Xt[b].to(device), yt[b].to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(b)

        model.eval()
        hits1 = hits5 = 0
        with torch.no_grad():
            for i in range(0, len(val_idx), BATCH):
                b = val_idx[i:i + BATCH]
                xb, yb = Xt[b].to(device), yt[b].to(device)
                logits = model(xb)
                top5 = logits.topk(5, dim=1).indices
                hits1 += int((top5[:, 0] == yb).sum())
                hits5 += int((top5 == yb[:, None]).any(dim=1).sum())
        row = {"epoch": epoch, "train_loss": total / len(order),
               "val_top1": hits1 / len(val_idx), "val_top5": hits5 / len(val_idx),
               "seconds": round(time.time() - t0, 1)}
        history.append(row)
        print("%-9s epoch %2d  loss %.4f  top1 %.3f  top5 %.3f  (%.0fs)"
              % (run, epoch, row["train_loss"], row["val_top1"], row["val_top5"],
                 row["seconds"]))
        torch.save({"model": model.state_dict(), "meta": meta},
                   os.path.join(ckpt_dir, "epoch_%02d.pt" % epoch))

    best = max(history, key=lambda r: r["val_top1"])
    torch.save({"model": model.state_dict(), "meta": meta},
               os.path.join(ckpt_dir, "best.pt"))
    return {"run": run, "device": device, "n_train": len(tr_idx), "n_val": len(val_idx),
            "d_in": int(X.shape[1]), "best_val_top1": best["val_top1"],
            "best_epoch": best["epoch"], "history": history, **stats}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="all",
                    choices=list(MODES) + ["both", "all"])
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--view", default=None,
                    choices=["pooled", "live"] + list(NAMED_VIEWS),
                    help="which readout of the sim to feed the head")
    args = ap.parse_args()
    if args.view:
        VIEW = args.view

    runs = {"both": ["full", "ablation"], "all": list(MODES)}.get(
        args.run, [args.run])
    print("shards: %s   view: %s" % (SHARD_DIR, VIEW))
    os.makedirs(os.path.join(OUT, "results"), exist_ok=True)
    path = os.path.join(OUT, "results", "train.json")
    out = json.load(open(path)) if os.path.exists(path) else {}
    for r in runs:
        out[r] = train(r, args.epochs)
        json.dump(out, open(path, "w"), indent=1)
    for r, v in out.items():
        print("%-9s best val top-1 %.3f at epoch %d (d_in=%d)"
              % (r, v["best_val_top1"], v["best_epoch"], v["d_in"]))
    print("wrote", path)
