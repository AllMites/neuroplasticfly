"""Where the board is lost between 139,248 neurons and the 8,865 numbers we read.

Measured 2026-09-16. `readout_probe.py` established that reconstructing which
piece stands on which square from the SAME 5,000 simulations gives occupied-F1
0.610 for the shipped `cell_type` mean, 0.852 for a sparse random projection and
0.850 for a random 8,865-neuron subset, all at equal width and equal decoder
budget. So the loss is in our readout, not in the fly. That measurement said
*that* the pooling is lossy; it did not say *why*, *how many dimensions are
actually needed*, *which neurons carry the board*, or whether any readout can be
both biologically interpretable and high-fidelity.

This script answers those four from `data/rates_dump/` -- full 139,248-dim rate
vectors for 5,000 positions, written once by dump_rates.py -- so every question
below costs seconds on the host GPU and zero simulation.

Sections, each at matched width against the shipped readout and against a random
projection of the same width. Run one with --only e1c.

  E0  reproduce readout_probe.py from the dump. Got 0.608 / 0.856 / 0.860.
  E1  split every `cell_type` group into k sub-groups, spatial or random. The
      hypothesis was that spatial splits recover the board and random ones do
      not. REFUTED both ways: k=16 at 37,143 dims reaches only 0.702, and the
      random control is never worse than the spatial one.
  E1b the same total width as the shipped readout, bins allocated by group size:
      0.843 at 8,865 dims.
  E1c the knob E1 failed to vary -- neurons pooled PER DIMENSION, held equal
      against a random projection of the same width and the same fan. Within a
      cell type, F1 falls 0.867 -> 0.610 as the pool grows 2 -> 1,024 neurons;
      a brain-wide pool of the same size holds 0.844. That is the mechanism:
      within-type pooling averages near-copies and collapses rank, and it has
      nothing to do with retinotopy specifically.
  E2  width sweep. The random projection saturates by ~1,024 dims, so the
      shipped 8,865 is ~9x wider than the board needs.
  E3  which neurons carry it, per super_class, at one matched width -- with and
      without the 14,521 neurons the encoder can drive directly. flypoke turns a
      stimulated neuron into a pure Poisson source (see encode.stimuli), so
      reading T4/T5 back out is reading our own encoder, not the fly. Every
      claim about the fly has to use the driven-excluded rows.
  E3b the same question over each population's LIVE neurons only, so a mostly
      silent population is not scored on its silence.
  E4  named readouts: cell_type x side, T4/T5 retinotopic bins, hybrids.
  E5  how wide a named `cell_type` x spatial-bin readout has to be to match its
      random reference.
  E6  the same, but with bins allocated by MEASURED live neurons per type rather
      than by type size. 0.866 at the shipped 8,865 dims -- above the random
      projection. This is the readout the report recommends.

Two-step run. The metadata export needs the connectome and therefore the
container, but it runs no simulation and takes about a minute:

  docker compose run --rm sim python -u readout_analysis.py export-meta
  uv run python readout_analysis.py                # host, CUDA, ~10 min

Decoder is byte-for-byte the one in readout_probe.py (log1p, drop dead dims,
standardise, Linear(d,1024)-GELU-Linear(1024,768), AdamW 1e-3/0.01, BCE, 16
epochs, batch 256, best val occupied-F1 over epochs) on the identical
train/val split, so every number here is comparable to the three already in
WHAT_IS_REAL.md. Standardisation uses all rows, as it does there; it is the
same leak for every row of every table, so it cannot flip a comparison.
"""
import argparse
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.abspath(__file__))
DUMP = os.path.join(REPO, "data", "rates_dump")
META = os.path.join(REPO, "data", "neuron_meta.npz")
VISUAL_MAP = os.path.join(REPO, "data", "visual_map.npz")
OUT_JSON = os.path.join(REPO, "results", "readout_analysis.json")

SEED = 12345               # same table seed reservoir.readout_tables uses
FAN = 64                   # neurons summed per random-projection dimension
SHIPPED_WIDTH = 8865       # width of the pooled vector that ships
GRID = 8                   # encoder's retinal patches per axis per eye


# ---------------------------------------------------------------- metadata export
def export_meta():
    """Dump the shipped pooling and per-neuron annotations to the host.

    Runs inside the sim container because `behaviors.net()` needs the connectome
    and `encode._soma_xyz` reads the annotation TSV out of the flypoke data
    volume, which is not mounted on the host. No simulation happens here.
    """
    import behaviors as B
    import encode
    import reservoir as R

    n = B.net()
    group_id, sizes, names = R.pooling(n)
    xyz = encode._soma_xyz(n)
    df = n.neurons

    def col(name):
        if name not in df.columns:
            return np.array(["missing"] * n.n)
        return df[name].astype("string").fillna("none").to_numpy().astype(str)

    out = os.path.join("/app", "data", "neuron_meta.npz")
    np.savez_compressed(
        out,
        group_id=group_id.astype(np.int32),
        group_sizes=sizes.astype(np.int32),
        group_names=np.array([str(x) for x in names]),
        xyz=xyz.astype(np.float32),
        side=col("side"),
        super_class=col("super_class"),
        cell_class=col("cell_class"),
        cell_sub_class=col("cell_sub_class"),
        cell_type=col("cell_type"),
        columns=np.array([str(c) for c in df.columns]),
    )
    print("wrote %s: %d neurons, %d groups" % (out, n.n, len(names)))
    print("columns available:", list(df.columns))


# ---------------------------------------------------------------- data loading
def load_rates():
    """(rates float16 [N, 139248], occ float32 [N, 768])."""
    parts = sorted(f for f in os.listdir(DUMP) if f.startswith("rates_"))
    if not parts:
        raise SystemExit("no rates parts in %s; dump_rates.py has not run" % DUMP)
    chunks, ids = [], []
    for p in parts:
        b = np.load(os.path.join(DUMP, p))
        chunks.append(b["rates"])
        ids.append(b["row_ids"])
    rates = np.concatenate(chunks)
    row_ids = np.concatenate(ids)
    tgt = np.load(os.path.join(DUMP, "targets.npz"))
    # dump_rates writes targets in sorted row_id order and the parts come back in
    # that same order, but check rather than trust: a silent misalignment would
    # look exactly like "the readout lost the board".
    if not np.array_equal(row_ids, tgt["row_ids"][:len(row_ids)]):
        raise SystemExit("rate parts and targets.npz disagree on row order")
    occ = tgt["planes"][:len(rates), :768].astype(np.float32)
    return rates, occ


def load_meta():
    if not os.path.exists(META):
        raise SystemExit("run `docker compose run --rm sim python -u "
                         "readout_analysis.py export-meta` first")
    return np.load(META, allow_pickle=False)


# ---------------------------------------------------------------- readout builders
class Reducer:
    """Turns the [N, 139248] rate matrix into a [N, W] readout on the GPU.

    Two primitives cover every readout in this file: a partition (each neuron
    belongs to at most one output dimension -- pooling, subsets, spatial bins)
    and a sparse projection (each output dimension sums FAN random neurons).
    """

    def __init__(self, rates, device):
        import torch
        self.torch = torch
        self.device = device
        self.N, self.M = rates.shape
        # 5,000 x 139,248 float16 is 1.4 GB, which fits; everything downstream
        # reads slices of it, so keeping it resident avoids re-uploading per row.
        self.R = torch.from_numpy(rates).to(device)

    def partition(self, member, n_out, mean=True, chunk=512):
        """member: int32 [139248], -1 = neuron not used. Sum or mean per bin."""
        torch = self.torch
        m = torch.from_numpy(member.astype(np.int64)).to(self.device)
        keep = m >= 0
        idx = torch.nonzero(keep, as_tuple=True)[0]
        tgt = m[idx]
        counts = torch.bincount(tgt, minlength=n_out).clamp(min=1).float()
        out = torch.empty((self.N, n_out), dtype=torch.float32, device=self.device)
        for a in range(0, self.N, chunk):
            b = min(a + chunk, self.N)
            acc = torch.zeros((b - a, n_out), dtype=torch.float32, device=self.device)
            acc.index_add_(1, tgt, self.R[a:b, idx].float())
            out[a:b] = acc / counts if mean else acc
        return out.cpu().numpy()

    def project(self, table, budget=2 ** 25):
        """table: int32 [W, FAN] of neuron indices; returns their mean per dim."""
        torch = self.torch
        t = torch.from_numpy(table.astype(np.int64).reshape(-1)).to(self.device)
        W, fan = table.shape
        chunk = max(1, budget // (W * fan))     # keep the gather under ~64 MB fp16
        out = np.empty((self.N, W), dtype=np.float32)
        for a in range(0, self.N, chunk):
            b = min(a + chunk, self.N)
            g = self.R[a:b][:, t].float().view(b - a, W, fan)
            out[a:b] = (g.sum(-1) / fan).cpu().numpy()
        return out

    def live_neurons(self, thresh=1e-3, chunk=512):
        """Per-neuron: does this rate move at all across the 5,000 boards?

        Measured from the dump, with no reference to the decoder or the board
        labels, so using it to allocate readout dimensions is not fitting the
        target -- it is the same kind of fixed table reservoir.readout_tables
        already builds.
        """
        torch = self.torch
        s1 = torch.zeros(self.M, dtype=torch.float64, device=self.device)
        s2 = torch.zeros(self.M, dtype=torch.float64, device=self.device)
        for a in range(0, self.N, chunk):
            x = self.R[a:min(a + chunk, self.N)].double()
            s1 += x.sum(0)
            s2 += (x * x).sum(0)
        var = (s2 / self.N - (s1 / self.N) ** 2).clamp(min=0)
        return (var.sqrt() > thresh).cpu().numpy()

    def subset(self, idx, chunk=512):
        torch = self.torch
        t = torch.from_numpy(np.asarray(idx, dtype=np.int64)).to(self.device)
        out = np.empty((self.N, len(idx)), dtype=np.float32)
        for a in range(0, self.N, chunk):
            b = min(a + chunk, self.N)
            out[a:b] = self.R[a:b][:, t].float().cpu().numpy()
        return out


def pca2(xyz):
    """Two principal axes of a soma cloud -- the same trick encode._pca2 uses."""
    c = xyz - xyz.mean(axis=0)
    if len(c) < 3:
        return np.zeros((len(c), 2))
    _, _, vt = np.linalg.svd(c, full_matrices=False)
    return c @ vt[:2].T


def _factor(k):
    """k -> (a, b) with a*b == k and a >= b, as square as possible."""
    b = int(np.floor(np.sqrt(k)))
    while k % b:
        b -= 1
    return k // b, b


def spatial_bins(idx, xyz, k, rng=None):
    """Split one group's neurons into <= k bins by soma position.

    Nested quantile split on the group's own two principal axes, which is how
    encode._grid_split builds the 8x8 retina: independent quantiles leave empty
    corner cells whenever the axes correlate. With `rng` given the split is
    random instead of spatial, at identical bin sizes -- that is the control
    that separates "retinotopy destroyed" from "too many neurons averaged".
    """
    k = min(k, len(idx))
    if k <= 1:
        return [idx]
    if rng is not None:
        return np.array_split(rng.permutation(idx), k)
    a, b = _factor(k)
    uv = pca2(xyz[idx])
    out = []
    order_u = np.argsort(uv[:, 0], kind="stable")
    for chunk in np.array_split(order_u, a):
        order_v = chunk[np.argsort(uv[chunk, 1], kind="stable")]
        out.extend(idx[c] for c in np.array_split(order_v, b))
    return [o for o in out if len(o)]


def split_member(group_id, xyz, k, rng=None):
    """Member vector for "every cell_type group split into <= k sub-groups"."""
    member = np.full(len(group_id), -1, dtype=np.int32)
    order = np.argsort(group_id, kind="stable")
    bounds = np.searchsorted(group_id[order], np.arange(group_id.max() + 2))
    nxt = 0
    for g in range(group_id.max() + 1):
        idx = order[bounds[g]:bounds[g + 1]]
        if not len(idx):
            continue
        for part in spatial_bins(idx, xyz, k, rng):
            member[part] = nxt
            nxt += 1
    return member, nxt


def budget_member(group_id, sizes, xyz, width, rng=None):
    """Spend a FIXED total of `width` dimensions across the cell_type groups.

    Dimensions go where the neurons are: a group of s neurons gets round(width *
    s / 139248) spatial bins, so T4a (thousands of neurons spread over the whole
    retina) gets hundreds of retinotopic bins and a 4-neuron type gets none. This
    is the only way to compare "one mean per cell_type" against "cell_type x
    spatial bin" at exactly the shipped 8,865 dimensions.
    """
    total = sizes.sum()
    quota = np.floor(width * sizes / total).astype(np.int64)
    quota = np.minimum(quota, sizes)
    short = width - quota.sum()
    if short > 0:                      # hand the remainder to the largest groups
        frac = width * sizes / total - np.floor(width * sizes / total)
        for g in np.argsort(-frac):
            if short <= 0:
                break
            if quota[g] < sizes[g]:
                quota[g] += 1
                short -= 1
    member = np.full(len(group_id), -1, dtype=np.int32)
    order = np.argsort(group_id, kind="stable")
    bounds = np.searchsorted(group_id[order], np.arange(len(sizes) + 1))
    nxt = 0
    for g in range(len(sizes)):
        if quota[g] <= 0:
            continue
        idx = order[bounds[g]:bounds[g + 1]]
        for part in spatial_bins(idx, xyz, int(quota[g]), rng):
            member[part] = nxt
            nxt += 1
    return member, nxt, int((quota > 0).sum())


def proj_table(width, n_neurons, seed=SEED):
    return np.random.default_rng(seed).integers(0, n_neurons, size=(width, FAN),
                                                dtype=np.int32)


def rand_subset(width, n_neurons, seed=SEED):
    width = min(width, n_neurons)
    return np.random.default_rng(seed).choice(n_neurons, width,
                                              replace=False).astype(np.int32)


def driven_mask(meta, n_neurons):
    """Neurons the encoder can turn into a pure Poisson source, over any board.

    This matters more than it looks. flypoke drops a stimulated neuron's network
    input entirely (see encode.stimuli), so a stimulated neuron's rate IS the
    drive -- reading T4/T5 back out is reading our own encoder, not the fly. Any
    "the brain represents the board at F1 X" claim has to exclude these, or it is
    measuring the chessboard we just wrote down.

    The union over all boards: the 8x8xside retinal patches and the 40 ORN
    classes from the encoder's own frozen map, plus the taste, grooming and
    Johnston's-organ selections.
    """
    blob = np.load(VISUAL_MAP, allow_pickle=False)
    m = np.zeros(n_neurons, dtype=bool)
    for side in ("left", "right"):
        for f in range(GRID):
            for r in range(GRID):
                m[blob["%s_%d_%d" % (side, f, r)]] = True
    for k in range(40):
        m[blob["orn_%02d" % k]] = True
    csc, ct = meta["cell_sub_class"], meta["cell_type"]
    m |= np.isin(csc, ["sugar/water", "bitter", "grooming"])
    m |= np.isin(ct, ["JO-B1_a", "JO-FV"])
    return m


def fixed_pool_member(group_id, xyz, per_bin, rng=None):
    """Every cell_type group cut into bins of about `per_bin` neurons.

    The knob E1 got wrong. Splitting a group into a FIXED NUMBER of sub-groups
    leaves the huge groups still averaging hundreds of neurons per dimension
    (R1-6 has 8,452 neurons, so k=16 still means 528 per number). Fixing the
    number of neurons PER BIN instead is what actually varies the thing under
    test, and it is the quantity a random projection holds at FAN.
    """
    member = np.full(len(group_id), -1, dtype=np.int32)
    order = np.argsort(group_id, kind="stable")
    bounds = np.searchsorted(group_id[order], np.arange(group_id.max() + 2))
    nxt = 0
    for g in range(group_id.max() + 1):
        idx = order[bounds[g]:bounds[g + 1]]
        if not len(idx):
            continue
        k = max(1, int(round(len(idx) / float(per_bin))))
        for part in spatial_bins(idx, xyz, k, rng):
            member[part] = nxt
            nxt += 1
    return member, nxt


def t45_retino_member(meta, n_neurons, by_type=True):
    """T4/T5 neurons binned by the encoder's own 8x8 retinal patches.

    The map in data/visual_map.npz is the same object the encoder drives, so
    these bins are not a new authored choice -- they are the squares of the
    chessboard as the encoder defines them. `by_type` splits each patch by
    T4a..T5d, giving 2 sides x 64 patches x 8 subtypes.
    """
    blob = np.load(VISUAL_MAP, allow_pickle=False)
    ct = meta["cell_type"]
    member = np.full(n_neurons, -1, dtype=np.int32)
    subtypes = ["T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d"]
    nxt = 0
    for side in ("left", "right"):
        for f in range(GRID):
            for r in range(GRID):
                idx = blob["%s_%d_%d" % (side, f, r)]
                if by_type:
                    for st in subtypes:
                        sel = idx[ct[idx] == st]
                        if len(sel):
                            member[sel] = nxt
                        nxt += 1          # keep the bin layout fixed and readable
                else:
                    member[idx] = nxt
                    nxt += 1
    return member, nxt


# ---------------------------------------------------------------- decoder
def decode(X, occ, tr, val, epochs=16, hidden=1024, seed=0):
    """occupied-F1 of an MLP reconstructing the 12x8x8 piece planes.

    Identical to readout_probe.decode so the numbers join the existing table.
    """
    import torch
    import torch.nn as nn
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    # In place throughout: at width 32,768 a single copy of X is 650 MB and the
    # naive version of this held four of them at once.
    X = np.ascontiguousarray(X, dtype=np.float32)
    np.clip(X, 0, None, out=X)
    np.log1p(X, out=X)
    # Column statistics in slabs. The widest readouts here are 5,000 x 69,838,
    # which is 1.3 GB, and `X.std(axis=0)` allocates a second whole copy as a
    # temporary -- that is what ran the host out of memory on the first full run.
    keep = np.empty(X.shape[1], dtype=bool)
    mu = np.empty(X.shape[1], dtype=np.float32)
    sd = np.empty(X.shape[1], dtype=np.float32)
    for a in range(0, X.shape[1], 4096):
        s = X[:, a:a + 4096]
        mu[a:a + 4096] = s.mean(0)
        sd[a:a + 4096] = s.std(0)
    keep = sd > 1e-4
    if not keep.any():
        return 0.0, 0
    if not keep.all():
        X = np.ascontiguousarray(X[:, keep])
        mu, sd = mu[keep], sd[keep]
    X -= mu
    X /= sd + 1e-6
    Xt, T = torch.from_numpy(X), torch.from_numpy(occ)
    torch.manual_seed(seed)
    m = nn.Sequential(nn.Linear(X.shape[1], hidden), nn.GELU(),
                      nn.Linear(hidden, 768)).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.01)
    lf = nn.BCEWithLogitsLoss()
    best = 0.0
    for _ in range(epochs):
        m.train()
        o = tr[torch.randperm(len(tr)).numpy()]
        for i in range(0, len(o), 256):
            b = o[i:i + 256]
            opt.zero_grad(set_to_none=True)
            lf(m(Xt[b].to(dev)), T[b].to(dev)).backward()
            opt.step()
        m.eval()
        tp = fp = fn = 0.0
        with torch.no_grad():
            for i in range(0, len(val), 256):
                b = val[i:i + 256]
                p = (torch.sigmoid(m(Xt[b].to(dev))) > 0.5).float()
                t = T[b].to(dev)
                tp += float((p * t).sum())
                fp += float((p * (1 - t)).sum())
                fn += float(((1 - p) * t).sum())
        best = max(best, 2 * tp / max(2 * tp + fp + fn, 1.0))
    return best, int(keep.sum())


class Bench:
    def __init__(self, occ, n):
        perm = np.random.default_rng(1).permutation(n)   # readout_probe's split
        nv = max(200, int(n * 0.15))
        self.val, self.tr = perm[:nv], perm[nv:]
        self.occ = occ
        self.rows = []

    def run(self, tag, X, note=""):
        f1, live = decode(X, self.occ, self.tr, self.val)
        row = {"readout": tag, "width": int(X.shape[1]), "live_dims": live,
               "f1": round(f1, 4), "note": note}
        self.rows.append(row)
        print("  %-52s W=%-6d live=%-6d F1 %.4f %s"
              % (tag, X.shape[1], live, f1, note), flush=True)
        del X
        return row


# ---------------------------------------------------------------- experiments
ALL = ("e0", "e1", "e1c", "e2", "e3", "e3b", "e4", "e5", "e6")


def experiments(bench, red, meta, n_neurons, want=ALL):
    group_id = meta["group_id"]
    sizes = meta["group_sizes"].astype(np.int64)
    xyz = meta["xyz"]
    sc = meta["super_class"]
    ct = meta["cell_type"]
    side = meta["side"]
    G = len(sizes)
    res = {}
    # Hoisted out of E3/E4b so any single section can be run on its own with
    # --only: these masks are shared by three of them.
    driven = driven_mask(meta, n_neurons)
    und = np.nonzero(~driven)[0]
    und_mask = np.zeros(n_neurons, dtype=bool)
    und_mask[und] = True
    sizes_u = np.bincount(group_id[und], minlength=G).astype(np.int64)
    res["n_driven"] = int(driven.sum())
    if "e0" in want:
        # ---- reference rows -------------------------------------------------
        # Every table below is read against these three. They reproduce
        # readout_probe.py's 0.610 / 0.852 / 0.850 from the dumped rates, which is
        # also the check that the dump and the probe agree.
        print("\nE0 references (reproduce readout_probe.py from the dump)")
        bench.run("cell_type mean, 8865 (SHIPPED)", red.partition(group_id, G))
        bench.run("sparse random projection, 8865",
                  red.project(proj_table(SHIPPED_WIDTH, n_neurons)))
        bench.run("random neuron subset, 8865",
                  red.subset(rand_subset(SHIPPED_WIDTH, n_neurons)))
        # How much of a difference between two rows is just the decoder? The
        # same readout is scored under three decoder seeds. `best F1 over 16
        # epochs` is a max over noisy values, so this spread is the resolution of
        # every table in this file -- anything smaller than it is not a result.
        # (Fixing the seed makes a row exactly repeatable, but it does not make a
        # 0.005 difference between two rows mean anything.)
        rep_X = red.subset(rand_subset(SHIPPED_WIDTH, n_neurons))
        rep = [decode(rep_X, bench.occ, bench.tr, bench.val, seed=s)[0]
               for s in (0, 1, 2)]
        del rep_X
        res["decoder_spread"] = {"readout": "random neuron subset, 8865, seeds 0/1/2",
                                 "f1": [round(x, 4) for x in rep],
                                 "range": round(max(rep) - min(rep), 4)}
        print("  same readout under 3 decoder seeds: %s  spread %.4f"
              % (["%.4f" % x for x in rep], max(rep) - min(rep)))
    if "e1" in want:
        # ---- E1: averaging within a group, or too few dimensions? -----------
        print("\nE1 split each cell_type group into sub-groups (spatial vs random)")
        for k in (2, 4, 8, 16):
            mem, w = split_member(group_id, xyz, k)
            bench.run("cell_type x <=%d spatial sub-groups" % k, red.partition(mem, w))
            memr, wr = split_member(group_id, xyz, k, np.random.default_rng(7))
            bench.run("cell_type x <=%d RANDOM sub-groups (control)" % k,
                      red.partition(memr, wr))
            bench.run("random projection, same width", red.project(proj_table(w, n_neurons)),
                      "matched-width reference for k=%d" % k)
    if "e1" in want:
        print("\nE1b same TOTAL width as the shipped readout (8865 dims)")
        mem, w, used = budget_member(group_id, sizes, xyz, SHIPPED_WIDTH)
        bench.run("cell_type x spatial bins, budget 8865", red.partition(mem, w),
                  "%d of %d groups get >=1 bin" % (used, G))
        memr, wr, usedr = budget_member(group_id, sizes, xyz, SHIPPED_WIDTH,
                                        np.random.default_rng(7))
        bench.run("cell_type x RANDOM bins, budget 8865 (control)", red.partition(memr, wr),
                  "%d of %d groups" % (usedr, G))
    if "e1c" in want:
        # ---- E1c: pool size, at matched width AND matched neurons-per-dim ----
        # The decisive one. Each row pools about `p` neurons into one number. The
        # partition rows pool WITHIN a cell_type (so a bin is a nameable piece of a
        # named type); the projection row pools p neurons drawn from anywhere in the
        # brain, at the same width. Same width, same neurons averaged per dimension,
        # different composition -- so any gap is composition, and any common trend
        # with p is pool size.
        print("\nE1c fixed neurons-per-dimension (the knob E1 failed to vary)")
        res["pool_sweep"] = []
        for p in (2, 4, 16, 64, 256, 1024):
            mem, w = fixed_pool_member(group_id, xyz, p)
            a = bench.run("cell_type x spatial bins of ~%d neurons" % p,
                          red.partition(mem, w), "~%.1f neurons/dim" % (n_neurons / w))
            memr, wr = fixed_pool_member(group_id, xyz, p, np.random.default_rng(7))
            b = bench.run("cell_type x RANDOM bins of ~%d neurons (control)" % p,
                          red.partition(memr, wr))
            c = bench.run("random projection, width %d, fan %d" % (w, p),
                          red.project(np.random.default_rng(SEED).integers(
                              0, n_neurons, size=(w, max(1, p)), dtype=np.int32)),
                          "matched width and matched neurons/dim")
            res["pool_sweep"].append({"per_bin": p, "width": w, "spatial": a["f1"],
                                      "random_bins": b["f1"], "projection": c["f1"]})
    if "e2" in want:
        # ---- E2: how many dimensions does the board need? -------------------
        print("\nE2 width sweep")
        for w in (256, 1024, 2048, 4096, 8865, 16384, 32768):
            bench.run("random projection, width %d" % w, red.project(proj_table(w, n_neurons)))
        for w in (256, 1024, 4096, 8865, 32768):
            bench.run("random neuron subset, width %d" % w,
                      red.subset(rand_subset(w, n_neurons)))
    if "e3" in want:
        # ---- E3: which neurons carry the board? -----------------------------
        # Matched width: a random subset of MATCH neurons from each population, so a
        # population is not credited simply for being large. Populations smaller
        # than MATCH contribute everything they have, which is noted in the row.
        print("\nE3 which neurons carry it (matched width, random subset of each)")
        MATCH = 2048
        print("  (%d of %d neurons can be driven directly; their rate is the drive,"
              " not the fly)" % (driven.sum(), n_neurons))
        t45 = np.nonzero(np.isin(ct, ["T4a", "T4b", "T4c", "T4d",
                                      "T5a", "T5b", "T5c", "T5d"]))[0]
        pops = [("all neurons", np.arange(n_neurons)),
                ("all neurons, DRIVEN EXCLUDED", np.nonzero(~driven)[0]),
                ("T4/T5 (driven: this reads the stimulus)", t45),
                ("every driven neuron", np.nonzero(driven)[0])]
        for name in sorted(set(sc.tolist())):
            idx = np.nonzero(sc == name)[0]
            pops.append(("super_class=%s" % name, idx))
            rest_idx = idx[~driven[idx]]      # not `und`: that name is hoisted
            if len(rest_idx) < len(idx):
                pops.append(("super_class=%s, driven excluded" % name, rest_idx))
        res["pop_sizes"] = {}
        for name, idx in pops:
            if len(idx) < 64:
                continue
            res["pop_sizes"][name] = int(len(idx))
            pick = idx if len(idx) <= MATCH else np.random.default_rng(3).choice(
                idx, MATCH, replace=False)
            bench.run("subset of %s" % name, red.subset(np.sort(pick)),
                      "population %d neurons%s" % (len(idx),
                                                   "" if len(idx) > MATCH else " (all used)"))
    if "e3b" in want:
        # ---- E3b: which named populations, once silence is not held against --
        # E3 draws its matched-width subset from a whole population, so a
        # population that is 90% permanently silent is scored mostly on silence.
        # Here the subset is drawn from that population's LIVE neurons only, at
        # the same width, which asks the different and more interesting question:
        # given the neurons that move, how much board does this population hold?
        print("\nE3b named populations, matched width, live neurons only")
        live_n = red.live_neurons()
        MATCH = 1024
        kc = np.char.startswith(ct, "KC")
        cand = [("all live", live_n),
                ("all live, driven excluded", live_n & und_mask),
                ("T4/T5 (driven)", live_n & np.isin(ct, ["T4a", "T4b", "T4c", "T4d",
                                                         "T5a", "T5b", "T5c", "T5d"])),
                ("Kenyon cells (mushroom body)", live_n & kc),
                ("central, non-Kenyon", live_n & (sc == "central") & ~kc),
                ("optic, undriven", live_n & (sc == "optic") & und_mask),
                ("visual_projection", live_n & (sc == "visual_projection")),
                ("descending", live_n & (sc == "descending"))]
        for name, mask in cand:
            idx = np.nonzero(mask)[0]
            if len(idx) < 64:
                continue
            pick = idx if len(idx) <= MATCH else np.random.default_rng(3).choice(
                idx, MATCH, replace=False)
            bench.run("live subset of %s" % name, red.subset(np.sort(pick)),
                      "%d live neurons%s" % (len(idx),
                                             "" if len(idx) > MATCH else " (all used)"))
    if "e4" in want:
        # ---- E4: interpretable and high-fidelity ----------------------------
        print("\nE4 readouts that are both interpretable and high-fidelity")
        key = np.char.add(np.char.add(group_id.astype(str), "|"), side)
        names2, gid2 = np.unique(key, return_inverse=True)
        bench.run("cell_type x side", red.partition(gid2.astype(np.int32), len(names2)))
        bench.run("random projection, matched to cell_type x side",
                  red.project(proj_table(len(names2), n_neurons)))

        mem, w = t45_retino_member(meta, n_neurons, by_type=False)
        bench.run("T4/T5 retinotopic bins (2 sides x 64 patches)", red.partition(mem, w),
                  "DRIVEN: reads the encoder, %d neurons" % int((mem >= 0).sum()))
        mem_t, w_t = t45_retino_member(meta, n_neurons, by_type=True)
        bench.run("T4/T5 retinotopic bins x subtype", red.partition(mem_t, w_t),
                  "DRIVEN: reads the encoder")

        # hybrid: keep the retina retinotopic, keep everything else named by type
        hyb = np.where(mem_t >= 0, mem_t, group_id + w_t)
        bench.run("T4/T5 retinotopic x subtype + cell_type mean elsewhere",
                  red.partition(hyb.astype(np.int32), w_t + G))
        bench.run("random projection, matched to hybrid",
                  red.project(proj_table(w_t + G, n_neurons)))
    if "e4" in want:
        # Everything above still contains the driven neurons. The rows below are the
        # honest version of the question: can a NAMED readout of the neurons the fly
        # itself computed reach the fidelity of a random projection over the same
        # neurons? Same width on both sides of every pair.
        print("\nE4b the same question with every directly driven neuron removed")
        und = np.nonzero(~driven)[0]
        gid_u = np.where(und_mask, group_id, -1).astype(np.int32)
        live_groups = np.unique(group_id[und])
        bench.run("cell_type mean, driven excluded", red.partition(gid_u, G),
                  "%d groups survive" % len(live_groups))
        memb_u, wu, used_u = budget_member(np.where(und_mask, group_id, G),
                                           np.append(sizes_u, 0), xyz, SHIPPED_WIDTH)
        bench.run("cell_type x spatial bins, budget 8865, driven excluded",
                  red.partition(memb_u, wu), "%d groups get >=1 bin" % used_u)
        bench.run("random projection over undriven neurons, 8865",
                  red.project(np.random.default_rng(SEED).choice(
                      und, size=(SHIPPED_WIDTH, FAN)).astype(np.int32)))

        # optic lobe binned spatially by cell_type, central brain left as means:
        # the cheapest interpretable thing that spends dimensions where E3 says the
        # board is, without inventing a new spatial map for the central brain.
        optic = np.nonzero((sc == "optic") & ~driven)[0]
        om = np.full(n_neurons, -1, dtype=np.int32)
        nxt = 0
        for g in np.unique(group_id[optic]):
            idx = optic[group_id[optic] == g]
            bins = max(1, int(round(len(idx) / 8.0)))    # ~8 neurons per bin
            for part in spatial_bins(idx, xyz, bins):
                om[part] = nxt
                nxt += 1
        rest = np.nonzero((sc != "optic") & ~driven)[0]
        om[rest] = nxt + group_id[rest]
        bench.run("undriven optic spatial bins (~8/bin) + cell_type mean elsewhere",
                  red.partition(om, nxt + G))
        bench.run("random projection over undriven neurons, matched width",
                  red.project(np.random.default_rng(SEED).choice(
                      und, size=(nxt + G, FAN)).astype(np.int32)))
    if "e5" in want:
        # ---- E5: how wide does the NAMED readout have to be? ----------------
        # The recommendation rides on this table: at what width does "cell_type x
        # spatial bin" stop losing to a random projection, with and without the
        # neurons the encoder drives directly.
        print("\nE5 width sweep of the interpretable readout vs its random reference")
        res["budget_sweep"] = []
        for w_t_ in (2048, 8865, 32768, 65536):
            mem, w, used = budget_member(group_id, sizes, xyz, w_t_)
            a = bench.run("cell_type x spatial bins, budget %d" % w_t_,
                          red.partition(mem, w), "%d groups used" % used)
            b = bench.run("random projection, width %d" % w_t_,
                          red.project(proj_table(w_t_, n_neurons)))
            memu, wu2, usedu = budget_member(np.where(und_mask, group_id, G),
                                             np.append(sizes_u, 0), xyz, w_t_)
            c = bench.run("cell_type x spatial bins, budget %d, driven excluded" % w_t_,
                          red.partition(memu, wu2), "%d groups used" % usedu)
            d = bench.run("random projection over undriven, width %d" % w_t_,
                          red.project(np.random.default_rng(SEED).choice(
                              und, size=(w_t_, FAN)).astype(np.int32)))
            res["budget_sweep"].append({"width": w_t_, "named": a["f1"], "proj": b["f1"],
                                        "named_undriven": c["f1"], "proj_undriven": d["f1"]})
    if "e6" in want:
        # ---- E6: spend the dimensions where the variance is -----------------
        # E5 allocates bins to a cell_type in proportion to how many neurons it
        # has. That hands 538 bins to R1-6, whose 8,452 photoreceptors are the
        # one population flypet measured as a dead channel. Allocating in
        # proportion to how many of a type's neurons actually MOVE across the
        # 5,000 boards is still a named, reproducible readout -- the counts come
        # from the rate dump, not from the decoder or the labels -- and it is the
        # readout this analysis ends up recommending.
        print("\nE6 allocate bins by measured live neurons, not by group size")
        live_n = red.live_neurons()
        res["n_live_neurons"] = int(live_n.sum())
        print("  %d of %d neurons move at all across the 5,000 boards"
              % (live_n.sum(), n_neurons))
        live_sizes = np.bincount(group_id[live_n], minlength=G).astype(np.int64)
        live_sizes_u = np.bincount(group_id[live_n & und_mask], minlength=G).astype(np.int64)
        res["live_sweep"] = []
        for w_t_ in (2048, 8865, 32768):
            # bins go to live neurons only, so a bin never averages a live neuron
            # together with a permanently silent one
            gid_live = np.where(live_n, group_id, G).astype(np.int32)
            mem, w, used = budget_member(gid_live, np.append(live_sizes, 0), xyz, w_t_)
            a = bench.run("live-weighted cell_type x spatial bins, budget %d" % w_t_,
                          red.partition(mem, w), "%d groups used" % used)
            gid_lu = np.where(live_n & und_mask, group_id, G).astype(np.int32)
            memu, wu2, usedu = budget_member(gid_lu, np.append(live_sizes_u, 0),
                                             xyz, w_t_)
            c = bench.run("live-weighted, budget %d, driven excluded" % w_t_,
                          red.partition(memu, wu2), "%d groups used" % usedu)
            res["live_sweep"].append({"width": w_t_, "named_live": a["f1"],
                                      "named_live_undriven": c["f1"]})
    return res


def main(want=ALL):
    import torch
    rates, occ = load_rates()
    meta = load_meta()
    n_neurons = rates.shape[1]
    if len(meta["group_id"]) != n_neurons:
        raise SystemExit("neuron_meta.npz has %d neurons, rates have %d"
                         % (len(meta["group_id"]), n_neurons))
    print("%d positions x %d neurons, %d cell_type groups"
          % (rates.shape[0], n_neurons, len(meta["group_sizes"])))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("device %s" % dev)
    red = Reducer(rates, dev)
    n_pos = rates.shape[0]
    del rates                  # the GPU copy is the only one anyone reads again
    bench = Bench(occ, n_pos)
    extra = experiments(bench, red, meta, n_neurons, want)
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    # Merge rather than overwrite, so a --only rerun of one section does not
    # discard the tables the other sections already produced.
    blob = {}
    if os.path.exists(OUT_JSON):
        blob = json.load(open(OUT_JSON))
    rows = [r for r in blob.get("rows", [])
            if r["readout"] not in {x["readout"] for x in bench.rows}]
    blob.update(extra)
    blob["n_positions"] = int(n_pos)
    blob["rows"] = rows + bench.rows
    json.dump(blob, open(OUT_JSON, "w"), indent=1)
    print("\nwrote %s" % OUT_JSON)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", nargs="?", default="run",
                    choices=["run", "export-meta"])
    ap.add_argument("--only", default="", help="comma-separated: %s" % ",".join(ALL))
    args = ap.parse_args()
    if args.mode == "export-meta":
        export_meta()
        sys.exit(0)
    main(tuple(args.only.split(",")) if args.only else ALL)
