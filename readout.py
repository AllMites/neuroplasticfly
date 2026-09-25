"""Turn stored live-neuron rates into the readout the head is trained on.

Shards store `live`: the raw float16 rate of the 27,857 neurons that ever
respond to a chessboard. That is deliberately NOT a readout -- it is the raw
material, so that choosing a readout never costs another 7-hour precompute
again. This module is where a choice actually gets made.

Why the shipped readout is being replaced. Reconstructing which piece stands on
which square, from 5,000 identical simulations at 8,865 dimensions
(`readout_analysis.py`, and see the report at
.claude/PRPs/reports/readout-signal-loss.md):

    mean rate per cell_type (what shipped)        occupied-F1 0.608
    sparse random projection                                  0.856
    live-weighted cell_type x spatial bins                    0.866

The failure is not that pooling is bad, and not retinotopy -- splitting a type
spatially and splitting it at random score the same. It is that dimensions are
allocated by taxonomy while signal is allocated by population: 64% of responsive
neurons sit in 20 cell types holding 20 of the 8,865 dimensions, while 5,698
permanently silent types hold one each. Give every type a share of the budget
proportional to how many of ITS neurons actually move, and a named readout
matches an unnamed random projection.

Allocating by raw group size instead would hand 538 bins to R1-6, the 8,452
photoreceptors flypet measured as a dead channel. Live counts come from the rate
dump, never from the decoder or the labels, so this stays a fixed partition of
neurons and nothing here is trained.

CIRCULARITY. flypoke turns a stimulated neuron into a pure Poisson source and
drops its network input (see encode.stimuli), so a driven neuron's rate IS the
drive we imposed. `exclude_driven=True` drops all 13,686 of them, leaving only
the fly's own response. Use it for any claim about what the FLY carries; the
numbers are lower and they are the honest ones:

    mean rate per cell_type, driven excluded      occupied-F1 0.606
    live-weighted bins, driven excluded                       0.746
    sparse random projection, driven excluded                 0.804
"""
import os

import numpy as np

DATA = os.environ.get("FLYCHESS_DATA", "data")
META = os.path.join(DATA, "neuron_meta.npz")
LIVE = os.path.join(DATA, "live_mask.npz")
DRIVABLE = os.path.join(DATA, "drivable.npz")
WIDTH = 8865               # same as the shipped readout, so comparisons are fair


def _factor(k):
    """k -> (a, b) with a*b == k and a >= b, as square as possible."""
    b = int(np.floor(np.sqrt(k)))
    while k % b:
        b -= 1
    return k // b, b


def _pca2(xyz):
    centered = xyz - xyz.mean(axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return centered @ vt[:2].T


def _spatial_bins(idx, xyz, k):
    """Split one type's neurons into <= k bins by soma position.

    Nested quantile split on the group's own two principal axes, matching how
    encode._grid_split builds the retina: independent quantiles leave empty
    corner bins whenever the axes correlate.
    """
    k = min(int(k), len(idx))
    if k <= 1:
        return [idx]
    a, b = _factor(k)
    uv = _pca2(xyz[idx])
    out = []
    order_u = np.argsort(uv[:, 0], kind="stable")
    for chunk in np.array_split(order_u, a):
        order_v = chunk[np.argsort(uv[chunk, 1], kind="stable")]
        out.extend(idx[c] for c in np.array_split(order_v, b))
    return [o for o in out if len(o)]


def live_index():
    """Indices, into the full 139,248 neuron space, of the stored `live` columns."""
    blob = np.load(LIVE)
    mask = blob["sd"] > 1e-3 if "sd" in blob.files else blob["live"]
    return np.flatnonzero(mask), mask


def partition(width=WIDTH, exclude_driven=False):
    """(member int32 over the stored live columns, n_bins, n_types_used).

    `member[i]` is which output dimension stored live-column i feeds, or -1 if
    it feeds none. Deterministic: same inputs, same partition, every process.
    """
    meta = np.load(META, allow_pickle=True)
    group_id, xyz = meta["group_id"], meta["xyz"]
    live_idx, _ = live_index()

    keep = np.ones(len(live_idx), dtype=bool)
    if exclude_driven:
        drivable = np.load(DRIVABLE)["drivable"]
        keep = ~drivable[live_idx]

    gid = group_id[live_idx]
    n_groups = int(group_id.max()) + 1
    sizes = np.bincount(gid[keep], minlength=n_groups).astype(np.int64)
    total = sizes.sum()
    if total == 0:
        raise ValueError("no neurons left after filtering")

    # dimensions follow neurons, not taxonomy -- the whole point
    quota = np.floor(width * sizes / total).astype(np.int64)
    quota = np.minimum(quota, sizes)
    short = width - int(quota.sum())
    if short > 0:
        frac = width * sizes / total - np.floor(width * sizes / total)
        for g in np.argsort(-frac):
            if short <= 0:
                break
            if quota[g] < sizes[g]:
                quota[g] += 1
                short -= 1

    member = np.full(len(live_idx), -1, dtype=np.int32)
    local = np.flatnonzero(keep)                 # positions within the live block
    order = local[np.argsort(gid[local], kind="stable")]
    bounds = np.searchsorted(gid[order], np.arange(n_groups + 1))
    nxt = 0
    for g in range(n_groups):
        if quota[g] <= 0:
            continue
        idx = order[bounds[g]:bounds[g + 1]]
        if not len(idx):
            continue
        for part in _spatial_bins(idx, xyz[live_idx], int(quota[g])):
            member[part] = nxt
            nxt += 1
    return member, nxt, int((quota > 0).sum())


def apply(live_rates, member, n_bins):
    """[N, n_live] float -> [N, n_bins] float32 mean rate per bin."""
    use = member >= 0
    m = member[use].astype(np.int64)
    counts = np.bincount(m, minlength=n_bins).astype(np.float32)
    counts[counts == 0] = 1.0
    x = np.asarray(live_rates, dtype=np.float32)[:, use]
    out = np.zeros((len(x), n_bins), dtype=np.float32)
    for k in range(0, len(x), 4096):           # chunked: the full matrix is ~11 GB
        blk = x[k:k + 4096]
        acc = np.zeros((len(blk), n_bins), dtype=np.float32)
        np.add.at(acc.T, m, blk.T)
        out[k:k + 4096] = acc / counts
    return out


def selfcheck():
    for excl in (False, True):
        member, n_bins, used = partition(exclude_driven=excl)
        live_idx, _ = live_index()
        assigned = int((member >= 0).sum())
        sizes = np.bincount(member[member >= 0], minlength=n_bins)
        print("%-16s bins=%-6d types=%-5d neurons assigned=%d/%d  "
              "bin size min/median/max = %d/%d/%d"
              % ("driven excluded" if excl else "all live", n_bins, used,
                 assigned, len(live_idx), sizes.min(), int(np.median(sizes)),
                 sizes.max()))
        assert n_bins <= WIDTH, "budget overspent"
        assert sizes.min() >= 1, "empty bin"
    rng = np.random.default_rng(0)
    member, n_bins, _ = partition()
    fake = rng.random((8, len(member))).astype(np.float32)
    got = apply(fake, member, n_bins)
    want0 = fake[0][member == 0].mean()
    assert abs(got[0, 0] - want0) < 1e-4, (got[0, 0], want0)
    print("apply() matches a direct per-bin mean")


if __name__ == "__main__":
    selfcheck()
