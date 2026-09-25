"""e-prop smoke test: can a learning rule move the fly brain at 139k scale?

The question is mechanics, not chess: with per-presynaptic-neuron output gains
theta (W_eff[i, :] = W[i, :] * exp(theta_i), sign and topology untouched) trained
by e-prop (Bellec et al. 2020: eligibility traces + broadcast learning signal,
O(1) memory per step, no BPTT), does a scalar readout of the MBONs fit a board
target better than training the readout alone? And do the dynamics stay sane?

Arms (`--arm`):
  readout   theta frozen at 0; only the 96 MBON readout weights learn. Baseline.
  eprop     readout + theta, per neuron (`--param neuron`, 139k gains) or per
            cell type (`--param type`, 8,865 gains = the A0 rung).

Target (`--target`): `material` = side-to-move material in pawns / 5, which the
taste channel injects directly, so it MUST be learnable if the machinery works;
`e4` = is e4 occupied, which only the visual pathway carries.

AUTHORED: GAMMA, DELTA (surrogate derivative), the feedback matrix, every lr.
REAL: the forward dynamics, unchanged from gpu_sim.run_batch except the gains.

Run: uv run eprop_smoke.py --arm eprop --param neuron --epochs 6
"""
import argparse
import json
import os
import time
from collections import deque

import chess
import numpy as np
import torch

import gpu_sim as G

GAMMA = 0.3         # surrogate derivative peak
DELTA = 3.5         # mV half-width of the surrogate; V_TH - V_REST is 7 mV
THETA_CLAMP = 1.5   # exp(1.5) = 4.5x gain either way; a runaway stops here
OUT = os.path.join(G._HERE, "results")


def load_positions(n, seed=0):
    import pyarrow.parquet as pq
    fens = pq.read_table(os.path.join(G._HERE, "data", "positions.parquet"),
                         columns=["fen"]).column("fen").to_pylist()
    idx = np.random.default_rng(seed).choice(len(fens), n, replace=False)
    return [fens[i] for i in idx]


def target_of(board, kind):
    import encode
    if kind == "material":
        tot = 0
        for p in board.piece_map().values():
            v = encode.PIECE_VALUE[p.piece_type]
            tot += v if p.color == board.turn else -v
        return float(np.clip(tot, -5, 5)) / 5.0
    if kind == "e4":
        return 1.0 if board.piece_at(chess.E4) else -1.0
    raise SystemExit("unknown target %r" % kind)


class EpropSim(G.GpuSim):
    def __init__(self, readout_idx, param="neuron", seed=0, **kw):
        super().__init__(**kw)
        N, dev = self.net.n, self.device
        g = torch.Generator(device="cpu").manual_seed(seed)
        self.theta = torch.zeros(N, device=dev)
        self.param = param
        self.ro = torch.as_tensor(np.asarray(readout_idx, dtype=np.int64), device=dev)
        self.c = (torch.randn(len(readout_idx), generator=g) * 0.1).to(dev)
        # Broadcast alignment: hidden neurons get a fixed random learning signal.
        self.fb_hidden = (torch.randn(N, generator=g) * 0.1).to(dev)
        self.W = torch.sparse_csr_tensor(self.indptr, self.indices, self.data,
                                         size=(N, N))          # pre-major

    # -------------------------------------------------- forward with gains
    def _deliver(self, g, pre, bcol, B):
        start = self.indptr[pre]
        cnt = self.indptr[pre + 1] - start
        total = int(cnt.sum().item())
        if total == 0:
            return
        csum = torch.cumsum(cnt, 0) - cnt
        ar = torch.arange(total, device=self.device)
        offs = ar - torch.repeat_interleave(csum, cnt, output_size=total) \
            + torch.repeat_interleave(start, cnt, output_size=total)
        post = self.indices[offs]
        gain = torch.repeat_interleave(torch.exp(self.theta[pre]), cnt, output_size=total)
        w = self.data[offs] * gain
        bb = torch.repeat_interleave(bcol, cnt, output_size=total)
        torch.use_deterministic_algorithms(True)
        try:
            g.view(-1).index_add_(0, post * B + bb, w)
        finally:
            torch.use_deterministic_algorithms(False)

    # -------------------------------------------------- forward + eligibility
    def run_eprop(self, drives, seeds, t_run=300.0, learn=True):
        """counts [B, N] and, if learn, Gacc [N, B] = sum_t eps_i(t) * u_i(t)."""
        dev = self.device
        N, B = self.net.n, len(drives)
        n_steps = int(round(t_run / G.DT))
        delay_steps = max(1, int(round(G.DELAY / G.DT)))
        refr_steps = int(round(G.T_REFR / G.DT))
        decay_m = float(np.float32(np.exp(-G.DT / G.TAU_M)))
        decay_s = float(np.float32(np.exp(-G.DT / G.TAU_SYN)))
        decay_a = float(np.float32(np.exp(-G.DT / G.TAU_A)))
        gain = float(np.float32(1.0 - np.float32(np.exp(-G.DT / G.TAU_M))))

        prob_np = np.zeros((N, B), dtype=np.float32)
        for b, (ii, pp) in enumerate(drives):
            if len(ii):
                prob_np[np.asarray(ii, dtype=np.int64), b] = pp
        prob = torch.from_numpy(prob_np).to(dev)
        seeds = np.asarray(seeds, dtype=np.int64)
        seed32 = torch.from_numpy((seeds & 0xFFFFFFFF).astype(np.uint32)
                                  .astype(np.int32)).to(dev).view(1, B)
        neuron = torch.arange(N, dtype=torch.int32, device=dev).view(N, 1)
        step_consts = torch.from_numpy(np.array([G._w32(t * G._S1) for t in range(n_steps)],
                                                dtype=np.int32)).to(dev)
        v = torch.full((N, B), G.V_REST, dtype=torch.float32, device=dev)
        g = torch.zeros((N, B), dtype=torch.float32, device=dev)
        r = torch.zeros((N, B), dtype=torch.int8, device=dev)
        a = torch.zeros((N, B), dtype=torch.float32, device=dev)
        no_spike = torch.zeros((N, 1), dtype=torch.bool, device=dev)
        counts = torch.zeros(N * B, dtype=torch.int32, device=dev)
        refr_m1 = torch.tensor(refr_steps - 1, dtype=torch.int8, device=dev)
        zero_i8 = torch.tensor(0, dtype=torch.int8, device=dev)
        v_reset_t = torch.tensor(G.V_RESET, dtype=torch.float32, device=dev)
        ones = torch.ones(1 << 14, dtype=torch.int32, device=dev)
        empty = torch.empty(0, dtype=torch.int64, device=dev)
        pending = deque([(empty, empty) for _ in range(delay_steps)])
        step_fn = G._kernel("counter", G._step_counter, self.compile_)

        if learn:
            eps_s = torch.zeros((N, B), device=dev)
            eps_m = torch.zeros((N, B), device=dev)
            Gacc = torch.zeros((N, B), device=dev)
            z = torch.zeros((N, B), device=dev)
            fb = self.fb_hidden.clone()
            fb[self.ro] = self.c / self.sd          # dy/drate_j = c_j / sd_j
            fb = fb.view(N, 1)

        for step in range(n_steps):
            pre, bcol = pending.popleft()
            if pre.numel():
                self._deliver(g, pre, bcol, B)
            if learn:
                z.zero_()
                if pre.numel():
                    z[pre, bcol] = 1.0
                eps_s += z                       # mirrors g += delivered
                eps_m = eps_m * decay_m + gain * eps_s   # mirrors v <- ... + g*gain
                eps_s *= decay_s                 # mirrors g *= decay_s
            v, g, r, a, spk = step_fn(v, g, r, a, prob, seed32, step_consts[step], neuron,
                                      no_spike, G.V_REST, decay_m, gain, decay_s,
                                      decay_a, G.SFA_B_INC, G.V_TH,
                                      v_reset_t, refr_m1, zero_i8)
            if learn:
                psi = GAMMA * torch.clamp(1.0 - (v - G.V_TH).abs() / DELTA, min=0.0)
                psi = torch.where(spk, torch.full_like(psi, GAMMA), psi)
                psi = psi * (prob <= 0.0)        # driven neurons are Poisson sources: no dv/dg
                u = torch.sparse.mm(self.W, fb * psi)      # u_i = sum_post w_i,post fb_post psi_post
                Gacc += eps_m * u
            s_pre, s_b = spk.nonzero(as_tuple=True)
            if s_pre.numel():
                flat = s_pre * B + s_b
                if flat.numel() > ones.numel():
                    ones = torch.ones(flat.numel(), dtype=torch.int32, device=dev)
                counts.index_add_(0, flat, ones[:flat.numel()])
            pending.append((s_pre, s_b))
        counts = counts.view(N, B).T.contiguous()
        return counts, (Gacc if learn else None)

    # -------------------------------------------------- readout + update
    def set_norm(self, counts_list, t_run):
        """Per-MBON mean/sd over the training set, so the readout starts at y=0."""
        ro = torch.cat([c.to(torch.float32)[:, self.ro] for c in counts_list]) / (t_run / 1000.0)
        self.mu = ro.mean(0)
        # 29/96 MBONs have sd 0 over the training set; a 1e-3 floor turned a 3 Hz
        # drift into z = 3333 (measured). Floor at 1 Hz and clamp z.
        self.sd = torch.clamp(ro.std(0), min=1.0)
        self.c.zero_()
        self.b = torch.zeros((), device=self.device)

    def predict(self, counts, t_run):
        rates = counts.to(torch.float32) / (t_run / 1000.0)      # [B, N] Hz
        z = torch.clamp((rates[:, self.ro] - self.mu) / self.sd, -5.0, 5.0)
        return z @ self.c + self.b, z

    def grads(self, y, ystar, z, Gacc, t_run):
        B = len(y)
        e = 2.0 * (y - ystar) / B                                # dL/dy
        dc = (e[:, None] * z).sum(0)
        db = e.sum()
        dtheta = None
        if Gacc is not None:
            dtheta = torch.exp(self.theta) * (Gacc @ e) / (t_run / 1000.0)
            if self.param == "type":
                # A0 rung: one gain per cell type = mean of its neurons' gradients
                gsum = torch.zeros(len(self.net.pool_sizes), device=self.device)
                gsum.index_add_(0, self.group_id, dtheta)
                dtheta = (gsum / self.group_sizes.to(torch.float32))[self.group_id]
        return dc, db, dtheta


class Adam:
    def __init__(self, shape, lr, device, b1=0.9, b2=0.999):
        self.m = torch.zeros(shape, device=device)
        self.v = torch.zeros(shape, device=device)
        self.t, self.lr, self.b1, self.b2 = 0, lr, b1, b2

    def step(self, grad):
        self.t += 1
        self.m = self.b1 * self.m + (1 - self.b1) * grad
        self.v = self.b2 * self.v + (1 - self.b2) * grad * grad
        mh = self.m / (1 - self.b1 ** self.t)
        vh = self.v / (1 - self.b2 ** self.t)
        return self.lr * mh / (vh.sqrt() + 1e-8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["readout", "eprop"], default="eprop")
    ap.add_argument("--param", choices=["neuron", "type"], default="neuron")
    ap.add_argument("--target", choices=["material", "e4"], default="material")
    ap.add_argument("--n-train", type=int, default=64)
    ap.add_argument("--n-val", type=int, default=16)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--lr-theta", type=float, default=0.02)
    ap.add_argument("--lr-c", type=float, default=0.01)
    ap.add_argument("--t-run", type=float, default=300.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    tag = "%s_%s_%s" % (args.arm, args.param if args.arm == "eprop" else "none", args.target)

    import reservoir as R
    meta = np.load(os.path.join(G._HERE, "data", "neuron_meta.npz"), allow_pickle=False)
    live = np.load(os.path.join(G._HERE, "data", "live_mask.npz"))["sd"] > 0
    ro_idx = np.flatnonzero(meta["cell_class"].astype(str) == "MBON")
    sim = EpropSim(ro_idx, param=args.param, seed=args.seed)
    n, dev = sim.net.n, sim.device
    live_t = torch.from_numpy(live).to(dev)

    fens = load_positions(args.n_train + args.n_val, seed=args.seed)
    boards = [chess.Board(f) for f in fens]
    drives = [G.drive_of(b, sim.net) for b in boards]
    seeds = [R.seed_for(b) for b in boards]
    ystar = torch.tensor([target_of(b, args.target) for b in boards], device=dev)
    tr = np.arange(args.n_train)
    va = np.arange(args.n_train, args.n_train + args.n_val)
    print("target %s: train mean %.3f sd %.3f | val mean %.3f sd %.3f"
          % (args.target, ystar[tr].mean(), ystar[tr].std(), ystar[va].mean(), ystar[va].std()))

    # Readout normalisation from one forward pass over the training set.
    norm_counts = []
    for lo in range(0, len(tr), args.batch):
        b = tr[lo:lo + args.batch]
        norm_counts.append(sim.run_eprop([drives[i] for i in b], [seeds[i] for i in b],
                                         args.t_run, learn=False)[0])
    sim.set_norm(norm_counts, args.t_run)
    del norm_counts
    opt_c = Adam(sim.c.shape, args.lr_c, dev)
    opt_b = Adam((), args.lr_c, dev)
    opt_t = Adam(sim.theta.shape, args.lr_theta, dev)
    rng = np.random.default_rng(args.seed)
    log = []

    def evaluate(idx):
        ys, ls = [], []
        for lo in range(0, len(idx), args.batch):
            b = idx[lo:lo + args.batch]
            counts, _ = sim.run_eprop([drives[i] for i in b], [seeds[i] for i in b],
                                      args.t_run, learn=False)
            y, _ = sim.predict(counts, args.t_run)
            ys.append(y)
            ls.append(((y - ystar[b]) ** 2).sum())
        y = torch.cat(ys)
        # baseline: predicting the train mean
        base = ((ystar[idx] - ystar[tr].mean()) ** 2).mean()
        return float(sum(ls) / len(idx)), float(base), float(y.std())

    t0 = time.time()
    val0 = evaluate(va)
    print("epoch 0  val mse %.4f (predict-mean baseline %.4f, y sd %.3f)" % val0)
    log.append({"epoch": 0, "val_mse": val0[0], "val_base": val0[1], "y_sd": val0[2]})
    for ep in range(1, args.epochs + 1):
        order = rng.permutation(tr)
        for lo in range(0, len(order), args.batch):
            b = order[lo:lo + args.batch]
            tb = time.time()
            counts, Gacc = sim.run_eprop([drives[i] for i in b], [seeds[i] for i in b],
                                         args.t_run, learn=args.arm == "eprop")
            y, z = sim.predict(counts, args.t_run)
            loss = float(((y - ystar[b]) ** 2).mean())
            dc, db, dth = sim.grads(y, ystar[b], z, Gacc, args.t_run)
            sim.c -= opt_c.step(dc)
            sim.b -= opt_b.step(db)
            if dth is not None:
                sim.theta -= opt_t.step(dth)
                sim.theta.clamp_(-THETA_CLAMP, THETA_CLAMP)
            rates = counts.to(torch.float32) / (args.t_run / 1000.0)
            live_hz = float(rates[:, live_t].mean())
            hot = float((rates > 200.0).float().sum(1).mean())
            row = {"epoch": ep, "batch": lo // args.batch, "train_mse": round(loss, 4),
                   "y_sd": round(float(y.std()), 4), "live_hz": round(live_hz, 2),
                   "neurons_over_200hz": round(hot, 1),
                   "theta_absmax": round(float(sim.theta.abs().max()), 3),
                   "theta_sd": round(float(sim.theta.std()), 4),
                   "dtheta_norm": (round(float(dth.norm()), 4) if dth is not None else None),
                   "sec": round(time.time() - tb, 1)}
            log.append(row)
            print("ep %d b%d  mse %.4f  y sd %.3f  live %.2f Hz  >200Hz %.0f  "
                  "|theta| max %.3f sd %.4f  %s  %.1fs"
                  % (ep, row["batch"], loss, row["y_sd"], live_hz, hot,
                     row["theta_absmax"], row["theta_sd"],
                     ("grad %.2e" % row["dtheta_norm"]) if dth is not None else "",
                     row["sec"]), flush=True)
        vm, vb, vsd = evaluate(va)
        print("epoch %d  val mse %.4f (baseline %.4f, y sd %.3f)  %.1f min"
              % (ep, vm, vb, vsd, (time.time() - t0) / 60), flush=True)
        log.append({"epoch": ep, "val_mse": vm, "val_base": vb, "y_sd": vsd})

    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "eprop_smoke_%s.json" % tag)
    json.dump({"args": vars(args), "gamma": GAMMA, "delta": DELTA, "log": log,
               "minutes": round((time.time() - t0) / 60, 1)}, open(path, "w"), indent=1)
    print("wrote", path)


if __name__ == "__main__":
    main()
