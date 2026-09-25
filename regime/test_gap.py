"""GAP_COUPLE electrical eLN->ALPN coupling: cache, stability bound, symmetry,
sub-threshold guarantee, refractory mask, determinism, off-path identity."""
import os, sys
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gpu_sim as G

assert G.GAP_COUPLE == 0.0, "GAP_COUPLE must ship default-off"
assert G.GAP_NORM is False, "GAP_NORM must ship default-off"
sim = G.GpuSim()
assert sim.gap_k.numel() == 0, "off path must cache no coupling"
assert sim.gap_offsets.size == 0

offs, pre, post, count = G._gap_edges(sim.net)
assert len(offs) == 3802, len(offs)
eln = set(G._eln_idx(sim.net).tolist())
assert set(pre.tolist()) <= eln, "a gap pre is not a known eLN"
meta = np.load(os.path.join(G._HERE, "data", "neuron_meta.npz"), allow_pickle=False)
cc = meta["cell_class"].astype(str)
assert (cc[post] == "ALPN").all(), "a gap post is not an ALPN"
# measured on the v783 export, 2026-09-19; a re-export that moves it must be loud
assert float(count.sum()) == 55471.0, float(count.sum())

# brain() is a module singleton, so `on.net is sim.net`; keep an independent copy
# or the host-W_data check below would compare an array with itself.
w_before = sim.net.W_data.copy()
off_data = sim.data.clone()

# stability bound: coefficient at GAP_COUPLE = 1 is the summed synapse count
N = sim.net.n
s = np.zeros(N, np.float64)
np.add.at(s, post, count.astype(np.float64))
np.add.at(s, pre, count.astype(np.float64))
kmax = 1.0 / s.max()
print("kmax %.6g (max summed count %g)" % (kmax, s.max()))
# Heterogeneity of the count-proportional coefficient, for phase 3: whether it
# should be normalised per target is decided off this spread, so measure it over
# TARGETS (what the coupling does to a PN), not over both ends of every pair.
s_t = np.zeros(N, np.float64)
np.add.at(s_t, post, count.astype(np.float64))
nz = s_t[s_t > 0]
print("per-target summed counts over %d ALPNs: median %g  p90 %g  max %g  (%.1fx median)"
      % (len(nz), np.median(nz), np.percentile(nz, 90), nz.max(),
         nz.max() / np.median(nz)))

prev = G.GAP_COUPLE
try:
    G.GAP_COUPLE = kmax * 1.01
    try:
        G.GpuSim()
        raise SystemExit("GAP_COUPLE above the stability bound did not assert")
    except AssertionError as e:
        assert "summed per-neuron coefficient" in str(e), str(e)

    G.GAP_COUPLE = kmax * 0.5
    on = G.GpuSim()
    ot = torch.as_tensor(offs, device=on.device)
    assert float(on.data[ot].abs().max()) == 0.0, "chemical entries not zeroed"
    mask = torch.ones(off_data.numel(), dtype=torch.bool, device=off_data.device)
    mask[ot] = False
    assert torch.equal(on.data[mask], off_data[mask]), "a non-gap weight changed"
    assert np.array_equal(on.net.W_data, w_before), "host W_data was modified"
    assert on.gap_k.numel() == 3802, on.gap_k.numel()
    k_np = (count * np.float32(kmax * 0.5)).astype(np.float32)
    assert torch.allclose(on.gap_k, torch.as_tensor(k_np, device=on.device))

    dev = on.device
    in_play = np.zeros(N, bool); in_play[pre] = True; in_play[post] = True
    outside = torch.as_tensor(np.flatnonzero(~in_play).astype(np.int64), device=dev)

    # symmetry / linearity: one eLN lifted 10 mV, everything else at rest
    v = torch.full((N, 1), float(G.V_REST), dtype=torch.float32, device=dev)
    r = torch.zeros((N, 1), dtype=torch.int8, device=dev)
    v[pre[0]] = G.V_REST + 10.0
    on._deliver_gap(v, r)
    up = 10.0 * float(k_np[(pre == pre[0]) & (post == post[0])].sum())
    down = 10.0 * float(k_np[pre == pre[0]].sum())
    assert abs(float(v[post[0]]) - (G.V_REST + up)) < 1e-3, float(v[post[0]])
    assert abs(float(v[pre[0]]) - (G.V_REST + 10.0 - down)) < 1e-3, float(v[pre[0]])
    assert float((v[outside] - G.V_REST).abs().max()) == 0.0, "an uncoupled neuron moved"

    # sub-threshold by construction: 200 steps with every eLN pinned at V_TH
    v = torch.full((N, 1), float(G.V_REST), dtype=torch.float32, device=dev)
    r = torch.zeros((N, 1), dtype=torch.int8, device=dev)
    v[torch.as_tensor(sorted(eln), device=dev)] = float(G.V_TH)
    for _ in range(200):
        on._deliver_gap(v, r)
    assert float(v.max()) <= G.V_TH + 1e-3, float(v.max())
    assert float(v.min()) >= G.V_REST - 1e-3, float(v.min())

    # refractory end takes nothing; the non-refractory partner still gives
    v = torch.full((N, 1), float(G.V_REST), dtype=torch.float32, device=dev)
    r = torch.zeros((N, 1), dtype=torch.int8, device=dev)
    v[pre[0]] = G.V_REST + 10.0
    r[post[0]] = 3
    on._deliver_gap(v, r)
    assert float(v[post[0]]) == float(G.V_REST), "refractory post was driven"
    assert float(v[pre[0]]) < G.V_REST + 10.0, "pre did not give charge"

    # determinism: same snapshot, five calls, identical results
    snap = torch.full((N, 1), float(G.V_REST), dtype=torch.float32, device=dev)
    snap[pre[0]] = G.V_REST + 10.0
    snap[pre[-1]] = G.V_REST + 7.0
    r = torch.zeros((N, 1), dtype=torch.int8, device=dev)
    first = None
    for _ in range(5):
        vv = snap.clone()
        on._deliver_gap(vv, r)
        if first is None:
            first = vv
        else:
            assert torch.equal(vv, first), "_deliver_gap is not deterministic"
    del on
    torch.cuda.empty_cache()

    # off path is unchanged: same counts twice, and the known central fraction
    G.GAP_COUPLE = 0.0
    import chess
    sim2 = G.GpuSim()
    d = G.drive_of(chess.Board(), sim2.net)
    c1 = sim2.run_batch([d] * 2, [0, 1])
    c2 = sim2.run_batch([d] * 2, [0, 1])
    assert torch.equal(c1, c2), "off path is not deterministic"
    cen = meta["super_class"].astype(str) == "central"
    frac = (c1[0].cpu().numpy()[cen] / 0.3 > 1).mean()
    assert round(float(frac), 3) == 0.172, float(frac)
    from regime.alln_multiglom import multi_drive, A, B as ODB
    odour = [multi_drive(sim2, A, 60.0), multi_drive(sim2, ODB, 60.0)]
    counts_off = sim2.run_batch(odour, [0, 0]).clone()
    del sim2
    torch.cuda.empty_cache()

    # Reference arm for the hook-live check below. kmax*1e-6 zeroes the SAME
    # 3,802 chemical edges but couples ~nothing, so it isolates the delivery
    # from the zeroing. Comparing the live arm against GAP_COUPLE = 0 alone
    # cannot detect a deleted hook: the zeroing changes the counts by itself.
    G.GAP_COUPLE = kmax * 1e-6
    sim_t = G.GpuSim()
    counts_tiny = sim_t.run_batch(
        [multi_drive(sim_t, A, 60.0), multi_drive(sim_t, ODB, 60.0)], [0, 0]).clone()
    del sim_t
    torch.cuda.empty_cache()

    # smoke ON: two 8-glomerulus odours, finite counts, report the separation
    # kmax*0.5 is the NUMERICAL bound halved, not a physiological point: measured
    # 2026-09-19 it clamps every PN to the eLN pool within ~2 steps and kills the
    # brain (ALPN 11/11 active, KC 0.0000). kmax*0.01 is the live smoke point.
    G.GAP_COUPLE = kmax * 0.01
    sim3 = G.GpuSim()
    counts = sim3.run_batch([multi_drive(sim3, A, 60.0),
                             multi_drive(sim3, ODB, 60.0)], [0, 0])
    assert not torch.equal(counts, counts_off),         "GAP_COUPLE changes the brain"
    # the step-loop hook is LIVE: same drive, same seed, same edges zeroed,
    # only the coupling strength differs, so a deleted hook shows up here.
    assert not torch.equal(counts, counts_tiny),         "coupling strength changed nothing; the run_batch hook is not firing"
    arr = counts.cpu().numpy().astype(np.float64)
    assert np.isfinite(arr).all(), "non-finite spike counts with coupling on"
    act = (arr / 0.3) > 1.0
    is_pn = cc == "ALPN"
    is_kc = cc == "Kenyon_Cell"
    a0, a1 = act[0, is_pn], act[1, is_pn]
    u = int((a0 | a1).sum())
    jac = float((a0 & a1).sum() / u) if u else 0.0
    print("info gap on: ALPN jaccard %.3f  ALPN %d/%d  KC active %.4f/%.4f"
          % (jac, int(a0.sum()), int(a1.sum()),
             float(act[0, is_kc].mean()), float(act[1, is_kc].mean())))
    del sim3
    torch.cuda.empty_cache()

    # normalisation EXCLUDES the gap edges: a target PN's remaining incoming
    # chemical weight must still sum to the full target, not to target minus the
    # budget the zeroed edges would have eaten.
    G.NORM_TOTAL_TARGET = 29.15
    G.GAP_COUPLE = kmax * 0.01
    simn = G.GpuSim()
    ix = simn.net.W_indices.astype(np.int64)
    dn = simn.data.abs().cpu().numpy().astype(np.float64)
    is_gap = np.zeros(len(ix), bool); is_gap[offs] = True
    tot = np.zeros(N, np.float64)
    np.add.at(tot, ix[~is_gap], dn[~is_gap])
    tgt = np.unique(post)
    assert len(tgt) == 337, len(tgt)
    rel = np.abs(tot[tgt] - 29.15) / 29.15
    assert rel.max() < 1e-2, "gap targets off budget by up to %.4f" % rel.max()
    del simn
    torch.cuda.empty_cache()
    G.NORM_TOTAL_TARGET = 0.0

    # ELN_NEGATE and GAP_COUPLE are mutually exclusive, loudly
    G.ELN_NEGATE = True
    G.GAP_COUPLE = kmax * 0.01
    try:
        G.GpuSim()
        raise SystemExit("ELN_NEGATE + GAP_COUPLE did not assert")
    except AssertionError as e:
        assert "ELN_NEGATE and GAP_COUPLE" in str(e), str(e)
    G.ELN_NEGATE = False

    # GAP_NORM: every coupled PN carries a summed coefficient of exactly
    # GAP_COUPLE, whatever its synapse count. The eLN (source) side is not
    # normalised, so the <= 1 stability assert still has something to bite on.
    G.GAP_NORM = True
    G.GAP_COUPLE = 0.001
    simg = G.GpuSim()
    kg = simg.gap_k.cpu().numpy().astype(np.float64)
    sp = np.zeros(N, np.float64); np.add.at(sp, post, kg)
    tgtg = np.unique(post)
    # RELATIVE tolerance: the coefficients are float32 and the summed count they
    # are divided by runs 1 to 1958, so the absolute error of the sum scales with
    # S. A fixed 1e-6 absolute bound would mean something different at S = 0.001
    # than at the S = 0.003 the sweep also runs; relative says the same thing at
    # every S.
    rel = np.abs(sp[tgtg] - 0.001) / 0.001
    assert rel.max() <= 1e-3, float(rel.max())
    del simg
    torch.cuda.empty_cache()

    G.GAP_COUPLE = 0.0
    try:
        G.GpuSim()
        raise SystemExit("GAP_NORM with GAP_COUPLE = 0 did not assert")
    except AssertionError as e:
        assert "GAP_NORM needs" in str(e), str(e)
    G.GAP_NORM = False
finally:
    G.GAP_COUPLE = prev
    G.NORM_TOTAL_TARGET = 0.0
    G.ELN_NEGATE = False
    G.GAP_NORM = False
print("ok gap: 3802 edges, zeroed chemical, symmetric, sub-threshold, "
      "refractory mask, determinism, off identity, hook live, norm exclusion, "
      "negate guard, gap_norm")
