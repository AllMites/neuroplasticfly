"""The oracle harness: does gpu_sim reproduce flypoke, or only something plausible?

flypoke is the reference. This script never trusts the GPU path; it compares it
against `data/ref_dump.npz`, which `ref_dump.py` produced by running the real
`flypoke.sim.run_trial` in the container on the same FENs with the same seeds.

Seven checks, in increasing order of how much they would hurt if they failed:

 1. DRIVE      the host's (stim_idx, stim_prob) is byte-identical to the
               container's. If the two engines are not shown the same thing,
               nothing below means anything.
 2. DETERMINISM the same batch run twice gives identical spike counts.
 3. BATCH-INVARIANCE a position gives the same counts at B=1 and in slot k of a
               large batch. This is the check that catches state leaking across
               the batch dimension.
 4. VIEWS      the GPU's own pooling reduction equals `reservoir.pool`, and
               every readout `reservoir.views` currently keeps agrees between
               the engines. The shard writer calls `reservoir.views` itself,
               so the readout cannot silently fork.
 5. RATES      per-neuron rate agreement against flypoke on the SAME Poisson
               stream (rng="numpy"), plus the same comparison on the production
               stream (rng="counter"). Reported as Pearson r, mean/max absolute
               Hz difference, and total spike count ratio.
 6. POOLED     the GPU must be closer to the CPU, in the standardised log1p
               space the head consumes, than two CPU runs with different seeds
               are to each other. Measured against the CPU's own seed-noise
               floor, not against a fixed threshold, because the raw cosine floor
               is ~0.9997 and a fixed 0.99 bar is unreachable by construction
               (WHAT_IS_REAL.md, "the reservoir's own noise").
 7. BEHAVIOURS flypet's five measured stimulus -> label pairs must still come out
               eating / disgusted / jumping / grooming / chilling.

Writes results/gpu_verify.json. Prints SELFCHECK PASS only if all seven hold.

Run: uv run verify_gpu.py            (host, needs data/ref_dump.npz)
"""
import json
import os
import sys

import numpy as np
import torch

import gpu_sim as G

REF = os.environ.get("FLYCHESS_REF", "data/ref_dump.npz")
OUT = "results/gpu_verify.json"

# Copied from flypet's behaviors.py so the host can replay them without flypoke.
# The labels are the ones flypet MEASURED, which is what makes this the strongest
# available check: nobody chose them to make a port look good.
BEHAVIOR_STIMULI = {
    "feed": ("cell_sub_class=sugar/water", 150.0),
    "yuck": ("cell_sub_class=bitter", 150.0),
    "scare": ("cell_type=LC4|LPLC2", 150.0),
    "touch": ("cell_sub_class=grooming", 150.0),
    "idle": (None, 0.0),
}
BEHAVIOR_READOUTS = {"eating": "cell_sub_class=ingestion_motor_neuron",
                     "jumping": "cell_type=DNp04|DNp01",
                     "grooming": "cell_type=DNg29|DNg84"}
EXPECT = {"feed": "eating", "yuck": "disgusted", "scare": "jumping",
          "touch": "grooming", "idle": "chilling"}
THRESH = 5.0


def label(name, rates, baseline_eating=None, thresh=THRESH):
    live = [k for k, v in rates.items() if v >= thresh]
    if name == "feed":
        return "eating" if rates["eating"] >= thresh else "chilling"
    if name == "yuck":
        collapsed = baseline_eating is not None and baseline_eating >= thresh \
            and rates["eating"] < thresh
        return "disgusted" if collapsed else "chilling"
    if name == "scare":
        return "jumping" if rates["jumping"] >= thresh else "chilling"
    if name == "touch":
        return "grooming" if rates["grooming"] >= thresh else "chilling"
    return max(live, key=lambda k: rates[k]) if live else "chilling"


def _pearson(a, b):
    a = a - a.mean()
    b = b - b.mean()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / d) if d > 0 else float("nan")


def _zspace(vectors, mu=None, sd=None):
    """Standardised log1p - the space train.py actually feeds the head."""
    lg = np.log1p(np.asarray(vectors, dtype=np.float64))
    if mu is None:
        mu, sd = lg.mean(axis=0), lg.std(axis=0) + 1e-6
    return (lg - mu) / sd, mu, sd


def _cos(u, v):
    d = np.linalg.norm(u) * np.linalg.norm(v)
    return float(u @ v / d) if d > 0 else float("nan")


def _bar(samples):
    """mean - 2 sd of the CPU engine's own agreement with itself.

    Every acceptance test here is "the GPU must not differ from the CPU by more
    than two CPU runs differ from each other". A bare `>= mean` would be a coin
    flip whenever the two really are equivalent - measured, they agree to the
    third decimal - so the bar sits two standard deviations of the CPU-vs-CPU
    distribution below its own mean. A fixed threshold is not available here:
    the seed-noise floor is already r ~ 0.99, so a 0.99 bar tests nothing
    (WHAT_IS_REAL.md, "the reservoir's own noise").
    """
    a = np.asarray(samples, dtype=np.float64)
    return float(a.mean() - 2.0 * a.std())


def main(n_fens=None, batch=32):
    import chess

    net = G.brain()                          # installs the flypoke/behaviors stubs
    import reservoir as R
    ref = np.load(REF, allow_pickle=False)
    fens = [str(f) for f in ref["fens"]]
    if n_fens:
        fens = fens[:n_fens]
    seeds = ref["seeds"][:len(fens)]
    t_run = float(ref["t_run"])
    cpu_counts = ref["counts"][:len(fens)].astype(np.int64)
    # Same positions, same engine, different seeds: the scale everything below
    # is measured against.
    rr = ref["rerun_counts"].astype(np.int64)
    rrr = rr / (t_run / 1000.0)
    rr_pairs = [(2 * i, 2 * i + 1) for i in range(len(rr) // 2)]
    report, ok = {}, True
    sim = G.GpuSim()

    # ---------------------------------------------------------------- 1. drive
    offs, idx_ref, prob_ref = ref["stim_offsets"], ref["stim_idx"], ref["stim_prob"]
    drives, drive_ok = [], True
    for k, fen in enumerate(fens):
        ii, pp = G.drive_of(chess.Board(fen), net)
        drives.append((ii, pp))
        lo, hi = offs[k], offs[k + 1]
        drive_ok &= np.array_equal(ii, idx_ref[lo:hi]) and \
            np.array_equal(pp, prob_ref[lo:hi])
    ok &= drive_ok
    report["drive_identical"] = bool(drive_ok)
    report["n_stim_mean"] = float(np.mean([len(d[0]) for d in drives]))
    print("%s drive: host == container for all %d positions (%.0f stim neurons)"
          % ("OK " if drive_ok else "BAD", len(fens), report["n_stim_mean"]))

    # ---------------------------------------------------------------- 2/3. GPU runs
    gpu_counter = []
    for lo in range(0, len(fens), batch):
        gpu_counter.append(sim.run_batch(drives[lo:lo + batch],
                                         seeds[lo:lo + batch], t_run=t_run,
                                         rng="counter").cpu().numpy())
    gpu_counter = np.concatenate(gpu_counter).astype(np.int64)

    again = sim.run_batch(drives[:batch], seeds[:batch], t_run=t_run,
                          rng="counter").cpu().numpy().astype(np.int64)
    deterministic = np.array_equal(again, gpu_counter[:batch])
    ok &= deterministic
    report["deterministic_same_seed_twice"] = bool(deterministic)
    print("%s determinism: same batch twice identical=%s"
          % ("OK " if deterministic else "BAD", deterministic))

    slot = min(7, len(fens) - 1)
    solo = sim.run_batch([drives[slot]], [seeds[slot]], t_run=t_run,
                         rng="counter").cpu().numpy().astype(np.int64)[0]
    batch_inv = np.array_equal(solo, gpu_counter[slot])
    ok &= batch_inv
    report["batch_invariant"] = bool(batch_inv)
    report["batch_invariance_slot"] = int(slot)
    print("%s batch-invariance: position %d identical at B=1 and slot %d of B=%d"
          % ("OK " if batch_inv else "BAD", slot, slot, batch))

    # ---------------------------------------------------------------- 4. views
    # Two separate things. (a) The GPU's own float64 pooling lands on the same
    # float16 as numpy's bincount, so the rate vector itself is right and not
    # merely the CPU code being reused. (b) Every readout reservoir.views
    # currently keeps agrees between the two engines - whatever those keys are
    # this week, since the readout is being redesigned around this port.
    counts_t = torch.from_numpy(gpu_counter[:8].astype(np.int32)).to(sim.device)
    pool_ok = all(np.array_equal(np.asarray(R.pool(gpu_counter[k] /
                                                   (t_run / 1000.0), net)),
                                 sim.pool_gpu(counts_t, t_run)[k])
                  for k in range(8))
    ok &= pool_ok
    report["gpu_pool_equals_reservoir_pool"] = bool(pool_ok)
    print("%s views: GPU float64 pooling == reservoir.pool bit-for-bit (float16)"
          % ("OK " if pool_ok else "BAD"))

    gv = sim.views_of(counts_t, t_run)
    cv = {k: np.stack([np.asarray(R.views(cpu_counts[i] / (t_run / 1000.0),
                                          net)[k]) for i in range(8)])
          for k in gv}
    view_stats, view_ok = {}, True
    for k in gv:
        a = cv[k].astype(np.float64).ravel()
        b = gv[k].astype(np.float64).ravel()
        got = _pearson(a, b)
        # The CPU's own seed-to-seed agreement on the SAME readout. For `live`,
        # which is raw per-neuron Hz, this is the only meaningful bar: a fixed
        # 0.99 would fail two identical CPU runs as well.
        pair_r = [_pearson(np.asarray(R.views(rrr[x], net)[k], dtype=np.float64),
                           np.asarray(R.views(rrr[y], net)[k], dtype=np.float64))
                  for x, y in rr_pairs]
        bar = _bar(pair_r)
        view_stats[k] = {"dim": int(gv[k].shape[1]),
                         "pearson_vs_cpu": round(got, 6),
                         "cpu_seed_noise_pearson_mean": round(
                             float(np.mean(pair_r)), 6),
                         "bar_mean_minus_2sd": round(bar, 6),
                         "mean_abs_diff": round(float(np.abs(a - b).mean()), 4)}
        view_ok &= got >= bar
        print("    readout %-8s dim %-6d r=%.5f vs flypoke (CPU vs CPU on the "
              "same readout %.5f, bar %.5f)  mean|d|=%.3f"
              % (k, view_stats[k]["dim"], got, float(np.mean(pair_r)), bar,
                 view_stats[k]["mean_abs_diff"]))
    ok &= view_ok
    report["views"] = view_stats
    report["views_keys"] = sorted(gv)

    # ---------------------------------------------------------------- 5. rates
    cpu_rates = cpu_counts / (t_run / 1000.0)
    n_exact = min(len(fens), 32)          # numpy-RNG mode materialises the stream
    gpu_numpy = []
    for lo in range(0, n_exact, 16):
        gpu_numpy.append(sim.run_batch(drives[lo:lo + 16], seeds[lo:lo + 16],
                                       t_run=t_run, rng="numpy").cpu().numpy())
    gpu_numpy = np.concatenate(gpu_numpy).astype(np.int64)

    for tag, gc, m in (("numpy_rng", gpu_numpy, n_exact),
                       ("counter_rng", gpu_counter, len(fens))):
        gr = gc[:m] / (t_run / 1000.0)
        rs = [_pearson(cpu_rates[i], gr[i]) for i in range(m)]
        dev = np.abs(cpu_rates[:m] - gr)
        stat = {"positions": int(m),
                "pearson_mean": round(float(np.mean(rs)), 6),
                "pearson_min": round(float(np.min(rs)), 6),
                "mean_abs_hz": round(float(dev.mean()), 4),
                "max_abs_hz": round(float(dev.max()), 2),
                "spike_total_ratio": round(float(gc[:m].sum() / cpu_counts[:m].sum()), 5),
                "live_fraction_cpu": round(float(
                    (cpu_rates[:m][:, net.select("super_class=central")] > 1.0).mean()), 4),
                "live_fraction_gpu": round(float(
                    (gr[:, net.select("super_class=central")] > 1.0).mean()), 4)}
        report[tag] = stat
        print("    %-12s r=%.4f (min %.4f)  mean|dHz|=%.3f  max|dHz|=%.1f  "
              "spikes x%.4f  live %.3f vs %.3f"
              % (tag, stat["pearson_mean"], stat["pearson_min"],
                 stat["mean_abs_hz"], stat["max_abs_hz"],
                 stat["spike_total_ratio"], stat["live_fraction_cpu"],
                 stat["live_fraction_gpu"]))

    # Does the counter RNG actually behave like a Poisson source? A stimulated
    # neuron ignores the network entirely and fires at exactly its drive rate, so
    # its measured rate is a direct read on the random stream. flypoke's PCG64 is
    # the yardstick: if the two engines miss the nominal rate by the same margin,
    # the substitute stream is doing the same job.
    stim_stat = {}
    for tag, gc, m in (("numpy_rng", gpu_numpy, n_exact),
                       ("counter_rng", gpu_counter, len(fens))):
        errs_g, errs_c = [], []
        for i in range(m):
            ii, pp = drives[i]
            if not len(ii):
                continue
            want = pp * 1000.0 / G.DT                    # nominal Hz
            errs_g.append(gc[i][ii] / (t_run / 1000.0) - want)
            errs_c.append(cpu_counts[i][ii] / (t_run / 1000.0) - want)
        eg, ec = np.concatenate(errs_g), np.concatenate(errs_c)
        stim_stat[tag] = {"gpu_bias_hz": round(float(eg.mean()), 4),
                          "gpu_rms_hz": round(float(np.sqrt((eg ** 2).mean())), 3),
                          "cpu_bias_hz": round(float(ec.mean()), 4),
                          "cpu_rms_hz": round(float(np.sqrt((ec ** 2).mean())), 3)}
        print("    stim rate  %-12s gpu bias %+.3f rms %.2f Hz | cpu(PCG64) "
              "bias %+.3f rms %.2f Hz"
              % (tag, stim_stat[tag]["gpu_bias_hz"], stim_stat[tag]["gpu_rms_hz"],
                 stim_stat[tag]["cpu_bias_hz"], stim_stat[tag]["cpu_rms_hz"]))
    report["stimulated_neuron_rate_error"] = stim_stat

    # The CPU engine's own seed-to-seed spread on the same position, so the
    # numbers above can be read against something instead of against zero.
    noise, rate_bar = {"pearson_mean": float("nan")}, 0.99
    if len(rr):
        rs = [_pearson(rrr[a], rrr[b]) for a, b in rr_pairs]
        dev = np.abs(rrr[[a for a, _ in rr_pairs]] - rrr[[b for _, b in rr_pairs]])
        rate_bar = _bar(rs)
        noise = {"pairs": len(rr_pairs),
                 "pearson_mean": round(float(np.mean(rs)), 6),
                 "pearson_sd": round(float(np.std(rs)), 6),
                 "bar_mean_minus_2sd": round(rate_bar, 6),
                 "mean_abs_hz": round(float(dev.mean()), 4),
                 "max_abs_hz": round(float(dev.max()), 2)}
        print("    %-12s r=%.4f +- %.4f  mean|dHz|=%.3f  max|dHz|=%.1f   <- CPU "
              "vs CPU, two seeds, same position, %d pairs"
              % ("cpu_seednoise", noise["pearson_mean"], noise["pearson_sd"],
                 noise["mean_abs_hz"], noise["max_abs_hz"], len(rr_pairs)))
    report["cpu_seed_noise"] = noise

    rates_ok = (report["numpy_rng"]["pearson_mean"] >= rate_bar
                and report["counter_rng"]["pearson_mean"] >= rate_bar)
    ok &= bool(rates_ok)
    report["rates_beat_cpu_seed_noise"] = bool(rates_ok)
    print("%s rates: both GPU streams agree with flypoke at least as well as two "
          "flypoke runs agree with each other (bar r >= %.4f)"
          % ("OK " if rates_ok else "BAD", rate_bar))

    # ---------------------------------------------------------------- 6. pooled
    cpu_pool = np.stack([np.asarray(R.pool(cpu_rates[i], net), dtype=np.float64)
                         for i in range(len(fens))])
    gpu_pool = np.stack([np.asarray(R.pool(gpu_counter[i] / (t_run / 1000.0), net),
                                    dtype=np.float64) for i in range(len(fens))])
    gpu_pool_exact = np.stack([np.asarray(R.pool(gpu_numpy[i] / (t_run / 1000.0),
                                                  net), dtype=np.float64)
                               for i in range(n_exact)])
    zc, mu, sd = _zspace(cpu_pool)
    zg, _, _ = _zspace(gpu_pool, mu, sd)
    zx, _, _ = _zspace(gpu_pool_exact, mu, sd)
    cross = float(np.mean([_cos(zc[i], zg[i]) for i in range(len(fens))]))
    cross_exact = float(np.mean([_cos(zc[i], zx[i]) for i in range(n_exact)]))
    floor, floor_bar = float("nan"), float("-inf")
    if len(rr):
        rr_pool = np.stack([np.asarray(R.pool(rrr[i], net), dtype=np.float64)
                            for i in range(len(rrr))])
        zr, _, _ = _zspace(rr_pool, mu, sd)
        fl = [_cos(zr[x], zr[y]) for x, y in rr_pairs]
        floor, floor_bar = float(np.mean(fl)), _bar(fl)
    # Two scales for "different position". Adjacent rows of positions.parquet
    # are consecutive plies of the same game, so they are genuinely similar and
    # make a misleadingly high baseline; all distinct pairs is the honest one.
    pairs = [(i, j) for i in range(len(fens)) for j in range(i + 1, len(fens))]
    unrelated = float(np.mean([_cos(zc[i], zc[j]) for i, j in pairs]))
    adjacent = float(np.mean([_cos(zc[i], zc[i + 1]) for i in range(len(fens) - 1)]))
    pooled_ok = cross >= floor_bar
    ok &= bool(pooled_ok)
    report["pooled"] = {"bar_mean_minus_2sd": round(floor_bar, 4),
                        "cos_gpu_vs_cpu_same_position": round(cross, 4),
                        "cos_gpu_numpy_rng_vs_cpu": round(cross_exact, 4),
                        "cos_cpu_vs_cpu_two_seeds": round(floor, 4),
                        "cos_cpu_all_distinct_position_pairs": round(unrelated, 4),
                        "cos_cpu_adjacent_rows_same_game": round(adjacent, 4)}
    print("%s pooled (standardised log1p): GPU-vs-CPU cos=%+.4f "
          "(%+.4f on flypoke's own RNG stream), CPU seed-noise floor cos=%+.4f, "
          "unrelated positions cos=%+.4f, adjacent plies cos=%+.4f"
          % ("OK " if pooled_ok else "BAD", cross, cross_exact, floor,
             unrelated, adjacent))

    # DN readouts, the frozen body interface.
    dn_cpu = [R.dn(cpu_rates[i], net) for i in range(len(fens))]
    dn_gpu = [R.dn(gpu_counter[i] / (t_run / 1000.0), net) for i in range(len(fens))]
    dn_rows = {}
    for ct in dn_cpu[0]:
        for side in dn_cpu[0][ct]:
            a = np.array([d[ct][side] for d in dn_cpu])
            b = np.array([d[ct][side] for d in dn_gpu])
            dn_rows["%s/%s" % (ct, side)] = {
                "cpu_mean_hz": round(float(a.mean()), 2),
                "gpu_mean_hz": round(float(b.mean()), 2),
                "max_abs_hz": round(float(np.abs(a - b).max()), 2),
                "pearson": round(_pearson(a, b), 4) if a.std() > 0 else None}
    # Same comparison between two CPU runs of the same position under different
    # seeds: without it, "27 Hz apart on one readout" has no scale.
    if len(rr):
        dn_rr = [R.dn(rrr[i], net) for i in range(len(rrr))]
        for key in dn_rows:
            ct, side = key.rsplit("/", 1)
            a = np.array([dn_rr[2 * i][ct][side] for i in range(len(dn_rr) // 2)])
            b = np.array([dn_rr[2 * i + 1][ct][side] for i in range(len(dn_rr) // 2)])
            dn_rows[key]["cpu_seed_noise_max_abs_hz"] = round(
                float(np.abs(a - b).max()), 2)
    report["dn_table"] = dn_rows
    worst = max(dn_rows.items(), key=lambda kv: kv[1]["max_abs_hz"])

    # The worst dn gap is on a readout with 28 neurons, so before calling it a
    # port defect, measure what that readout does on the SAME position under
    # other seeds on the SAME engine. If the GPU-CPU gap sits inside the GPU's
    # own seed spread, the readout is seed-dominated on that board, not wrong.
    wct, wside = worst[0].rsplit("/", 1)
    widx = net.select(R.INGESTION if wct == "ingestion_motor_neuron"
                      else "cell_type=%s,side=%s" % (wct, wside))
    gaps = np.abs(np.array([d[wct][wside] for d in dn_cpu])
                  - np.array([d[wct][wside] for d in dn_gpu]))
    wpos = int(np.argmax(gaps))
    alt = sim.run_batch([drives[wpos]] * 4, [11, 22, 33, 44], t_run=t_run,
                        rng="counter").cpu().numpy().astype(np.int64)
    spread = (alt[:, widx].mean(axis=1) / (t_run / 1000.0)).round(1).tolist()
    report["dn_worst_case"] = {
        "readout": worst[0], "n_neurons": int(widx.size), "position": wpos,
        "cpu_hz": round(float([d[wct][wside] for d in dn_cpu][wpos]), 1),
        "gpu_hz": round(float([d[wct][wside] for d in dn_gpu][wpos]), 1),
        "gpu_four_other_seeds_hz": spread}
    print("    worst dn case: row %d, cpu %.1f Hz vs gpu %.1f Hz on %d neurons; "
          "the GPU's own four other seeds give %s Hz"
          % (wpos, report["dn_worst_case"]["cpu_hz"],
             report["dn_worst_case"]["gpu_hz"], widx.size, spread))
    print("    dn_table: %d readouts, worst single-position gap %s = %.1f Hz "
          "(cpu mean %.1f Hz; the same readout moves %.1f Hz between two CPU "
          "seeds)" % (len(dn_rows), worst[0], worst[1]["max_abs_hz"],
                      worst[1]["cpu_mean_hz"],
                      worst[1].get("cpu_seed_noise_max_abs_hz", float("nan"))))

    # ---------------------------------------------------------------- 7. behaviours
    names = [str(x) for x in ref["behavior_names"]]
    bh_drives = [G.drive_of_selector(*BEHAVIOR_STIMULI[nm], n=net) for nm in names]
    bt = float(ref["behavior_t_run"])
    bc = sim.run_batch(bh_drives, [0] * len(names), t_run=bt,
                       rng="counter").cpu().numpy().astype(np.int64)
    cpu_bc = ref["behavior_counts"].astype(np.int64)

    def _labels(counts):
        rates = counts / (bt / 1000.0)
        base, out, tab = None, {}, {}
        for k, nm in enumerate(names):
            rr_ = {lbl: float(rates[k][net.select(sel)].mean())
                   for lbl, sel in BEHAVIOR_READOUTS.items()}
            if nm == "feed":
                base = rr_["eating"]
            out[nm] = label(nm, rr_, base)
            tab[nm] = {k2: round(v, 1) for k2, v in rr_.items()}
        return out, tab

    gpu_lbl, gpu_tab = _labels(bc)
    cpu_lbl, cpu_tab = _labels(cpu_bc)
    beh_ok = all(gpu_lbl[nm] == EXPECT[nm] for nm in names)
    ok &= beh_ok
    report["behaviours"] = {"gpu_labels": gpu_lbl, "cpu_labels": cpu_lbl,
                            "expected": EXPECT, "gpu_rates_hz": gpu_tab,
                            "cpu_rates_hz": cpu_tab, "t_run_ms": bt}
    for nm in names:
        print("    %-6s gpu -> %-10s cpu -> %-10s want %-10s  gpu %s"
              % (nm, gpu_lbl[nm], cpu_lbl[nm], EXPECT[nm], gpu_tab[nm]))
    print("%s behaviours: all five flypet labels reproduce on the GPU engine"
          % ("OK " if beh_ok else "BAD"))

    report["all_ok"] = bool(ok)
    os.makedirs("results", exist_ok=True)
    json.dump(report, open(OUT, "w"), indent=1)
    print("wrote", OUT)
    return ok


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--fens", type=int, default=None)
    ap.add_argument("--batch", type=int, default=32)
    a = ap.parse_args()
    good = main(a.fens, a.batch)
    print("SELFCHECK", "PASS" if good else "FAIL")
    sys.exit(0 if good else 1)
