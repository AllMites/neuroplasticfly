"""Chunk-1 fit, exactly as PREREGISTER_rate_chunk1_fit.md (2838bfe + amendments 1-2).

Per-edge-type gains into the owned targets (FDA-I, FDA-II, PAM01-15), gain = cap*sigmoid(x) in [0, cap],
fitted on the Fox condition to the FIT rows only; 9 caps x 5 seeds; g* = smallest cap with >= 4/5
seeds passing F1-F5; held-out H1-H6 scored at g* only.

Run:      .venv/Scripts/python.exe tools/gpu_slot.py -- .venv/Scripts/python.exe rate/chunk1_fit.py
Gate:     ... rate/chunk1_fit.py --gate         (G-grad: 10 iterations, cap 18.287 seed 0, nothing saved)
Analyse:  .venv/Scripts/python.exe rate/chunk1_fit.py --analyze   (CPU, reads the saved cells)
Cells are saved one file each, so a killed batch resumes where it stopped.
"""
import argparse
import itertools
import json
import os
import sys
import time

import numpy as np
import torch
from scipy.stats import spearmanr

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _HERE)
import gpu_sim as G  # noqa: E402
from rate import chunk1 as C1  # noqa: E402
from rate import suite as S  # noqa: E402

OUT = os.path.join(_HERE, "results", "rate_chunk1_fit")
CAPS = [2.0, 4.0, 8.0, 16.0, 18.287, 32.0, 64.0, 128.0, 256.0]
SEEDS = [0, 1, 2, 3, 4]
BOUND = 18.287
MAX_IT, LR, LAM, UP_T, NONE_T = 400, 0.1, 0.01, 1.5, 0.25
BETA = 35.0282  # amendment 1: surrogate slope scale = arm-A rest-to-threshold distance (Hz)
KEEP_BEST = False  # pass 2 (PREREGISTER_rate_chunk1_pass2_loops.md): True, with LAM = 0
PREREG = "PREREGISTER_rate_chunk1_fit.md @ 2838bfe + amendments 1-2"
OWN = C1.FDA_I + C1.FDA_II + ["PAM%02d" % i for i in range(1, 16)]
F_ROWS = ["F1", "F2", "F3", "F4", "F5"]
H_ROWS = ["H1", "H2", "H3", "H4", "H5", "H6"]


def fit_cell(sim, St, tpools, cap, seed, max_it=MAX_IT):
    tab = sim.set_edge_gains(tpools, cap)
    sim.surrogate_beta = BETA  # backward only; forward (and every battery row) unchanged
    sim.surrogate_mask = np.flatnonzero(np.isin(sim.pool.cpu().numpy(), tpools))  # amendment 2
    g = torch.Generator().manual_seed(seed)
    g0 = torch.exp(0.1 * torch.randn(len(tab["n_edges"]), generator=g, dtype=torch.float64))
    g0 = g0.clamp(1e-3, cap * 0.999)
    sim.edge_x = torch.log(g0 / (cap - g0)).to(sim.device, sim.dtype).requires_grad_(True)
    opt = torch.optim.Adam([sim.edge_x], lr=LR)
    R = {c: torch.as_tensor(St["comp"][c], device=sim.device) for c in C1.R_COMP + ["b1"]}
    trace, fit_trace, grad0, it = [], [], None, 0
    best_fit, best_x = float("inf"), None
    for it in range(max_it):
        rates = sim.run([(St["fox"], C1.HZ)], C1.T_RUN)[0]
        D = {c: rates[i].mean() for c, i in R.items()}
        done = all(float(D[c]) >= C1.UP for c in C1.R_COMP) and float(D["b1"]) < C1.NONE
        gain = sim.edge_gain()
        fit = sum(torch.relu(UP_T - D[c]) ** 2 for c in C1.R_COMP) + torch.relu(D["b1"] - NONE_T) ** 2
        loss = fit + LAM * (torch.log(gain.clamp_min(1e-12)) ** 2).sum()
        trace.append(float(loss))
        fit_trace.append(float(fit))
        if float(fit) < best_fit:  # the x that produced this forward pass
            best_fit, best_x = float(fit), sim.edge_x.detach().clone()
        if done:
            break
        opt.zero_grad()
        if grad0 is None:  # G-grad: the fit term alone must carry gradient at init
            grad0 = float(torch.autograd.grad(fit, sim.edge_x, retain_graph=True)[0].abs().max())
        loss.backward()
        opt.step()
    sim.edge_x.requires_grad_(False)
    if KEEP_BEST and not done:
        sim.edge_x = best_x
    rates, D = C1.measure(sim, St)
    assert not rates["BASE"].any(), "BASE not silent: prereg assumption broken"
    rows = C1.battery(D)
    return {"cap": cap, "seed": seed, "iters": it + 1, "stopped_early": done, "loss_trace": trace,
            "fit_trace": fit_trace, "fit_grad0_max": grad0, "keep_best": KEEP_BEST, "lam": LAM,
            "edge_x": sim.edge_x.detach().cpu().double().numpy().tolist(),
            "gains": sim.edge_gain().detach().cpu().double().numpy().tolist(),
            "rows": {k: {"pass": p, "detail": d} for k, (p, d) in rows.items()}, "D_comp": D,
            "fpass": all(rows[r][0] for r in F_ROWS), "hpass": all(rows[r][0] for r in H_ROWS)}, tab


def label(cells):
    """cells: list of {cap, seed, fpass, hpass, rows}. Prereg labels, qualifier excluded."""
    by = {}
    for c in cells:
        by.setdefault(c["cap"], []).append(c)
    passing = [cap for cap in sorted(by) if sum(c["fpass"] for c in by[cap]) >= 4]
    if not passing:
        return {"label": "UNFITTABLE", "g_star": None}
    gs = passing[0]
    out = {"g_star": gs, "held_out_seeds": sum(c["hpass"] for c in by[gs] if c["fpass"])}
    if gs > BOUND:
        out["label"] = "IMPLAUSIBLE"
    elif out["held_out_seeds"] >= 4:
        out["label"] = "PASS"
    else:
        fails = sorted({r for c in by[gs] if c["fpass"] for r in H_ROWS if not c["rows"][r]["pass"]})
        out.update(label="FIT-NO-TRANSFER", failing_rows=fails)
    return out


def _selftest():
    mk = lambda cap, s, f, h, bad=(): {"cap": cap, "seed": s, "fpass": f, "hpass": h,  # noqa: E731
                                       "rows": {r: {"pass": r not in bad} for r in H_ROWS}}
    assert label([mk(2, s, False, False) for s in range(5)])["label"] == "UNFITTABLE"
    cells = [mk(4, s, s < 4, s < 4) for s in range(5)] + [mk(8, s, True, True) for s in range(5)]
    assert label(cells) == {"g_star": 4, "held_out_seeds": 4, "label": "PASS"}
    cells = [mk(4, s, True, False, ("H3",)) for s in range(5)]
    assert label(cells)["label"] == "FIT-NO-TRANSFER" and label(cells)["failing_rows"] == ["H3"]
    assert label([mk(32, s, True, True) for s in range(5)])["label"] == "IMPLAUSIBLE"
    assert label([mk(4, s, s < 3, True) for s in range(5)])["label"] == "UNFITTABLE"


def analyze():
    cells = [json.load(open(os.path.join(OUT, "cells", f)))
             for f in sorted(os.listdir(os.path.join(OUT, "cells")))]
    tab = json.load(open(os.path.join(OUT, "groups.json")))
    assert len(cells) == len(CAPS) * len(SEEDS), "incomplete batch: %d cells" % len(cells)
    lab = label(cells)
    res = {"prereg": PREREG, **lab, "per_cap": {}}
    for cap in CAPS:
        cs = [c for c in cells if c["cap"] == cap]
        res["per_cap"][str(cap)] = {
            "f_pass": sum(c["fpass"] for c in cs), "h_pass_of_f": sum(c["hpass"] for c in cs if c["fpass"]),
            "final_loss": [round(c["loss_trace"][-1], 4) for c in cs], "iters": [c["iters"] for c in cs],
            "rows_pass_count": {r: sum(c["rows"][r]["pass"] for c in cs) for r in F_ROWS + H_ROWS}}
    if lab["g_star"] is not None:
        cs = [c for c in cells if c["cap"] == lab["g_star"]]
        L = np.log(np.maximum(np.array([c["gains"] for c in cs]), 1e-12))
        act = np.flatnonzero((np.abs(L) > 0.05).any(0))
        rho = [spearmanr(L[i, act], L[j, act])[0] for i, j in itertools.combinations(range(len(cs)), 2)]
        med = float(np.median(rho)) if len(act) > 1 else float("nan")
        res.update(n_active=int(len(act)), rho_median=med,
                   qualifier="CONSISTENT" if med >= 0.5 else "INCONSISTENT")
        names = tab["pool_names"]
        mean_g = np.exp(L.mean(0))
        top = act[np.argsort(-np.abs(np.log(mean_g[act])))][:20]
        res["top_gains"] = [{"src": names[tab["src_pool"][k]], "tgt": names[tab["tgt_pool"][k]],
                             "geo_mean_gain": float(mean_g[k]), "n_syn": tab["n_syn"][k]} for k in top]
        # two-hop Fox -> X -> R compounding
        key = {(s, t): k for k, (s, t) in enumerate(zip(tab["src_pool"], tab["tgt_pool"]))}
        fox_p, r_p = set(tab["fox_pools"]), set(tab["r_pools"])
        best = (0.0, None)
        for (s, x), k1 in key.items():
            if s in fox_p:
                for (x2, t), k2 in key.items():
                    if x2 == x and t in r_p and mean_g[k1] * mean_g[k2] > best[0]:
                        best = (float(mean_g[k1] * mean_g[k2]), [names[s], names[x], names[t]])
        res["max_fox_x_r_product"] = best
    json.dump(res, open(os.path.join(OUT, "result.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k not in ("top_gains", "per_cap")}, indent=1))
    for cap, v in res["per_cap"].items():
        print("cap %-7s F-pass %d/5  H-pass(of F) %d  iters %s" % (cap, v["f_pass"], v["h_pass_of_f"], v["iters"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--pass2", action="store_true", help="POST-HOC pass 2: LAM 0 + keep-best")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--base", default="A", choices=["A", "c0"], help="c0: PREREGISTER_rate_chunk1_fit_c0.md")
    a = ap.parse_args()
    global LAM, KEEP_BEST, OUT, PREREG, BETA
    if a.base == "c0":
        OUT = OUT + "_c0"
        PREREG += " + PREREGISTER_rate_chunk1_fit_c0.md @ ee4ecbf (chunk-0 base)"
    if a.pass2:
        LAM, KEEP_BEST = 0.0, True
        OUT = OUT + "_pass2"
        PREREG = "PREREGISTER_rate_chunk1_pass2_loops.md @ 0f0b2f7 (PASS2, POST-HOC)"
    _selftest()
    if a.analyze:
        return analyze()
    St = C1.sets()
    sim = S.build(chunks=S.CHUNKS[1:], base=a.base)  # "A" = history (001a54a); "c0" = path-A rerun
    if a.base == "c0":
        BETA = -float(sim.bias[0])  # same definition (rest-to-threshold distance), c0 constant 9.519
        assert abs(BETA - 9.519) < 1e-3, BETA
    pool = sim.pool.cpu().numpy()
    tpools = S.owned_pools(pool, St["ct"], OWN)
    if a.gate:
        t0 = time.time()
        cell, _ = fit_cell(sim, St, tpools, BOUND, 0, max_it=10)
        ft = cell["fit_trace"]
        print("G-grad: fit-term grad at init %.3e; fit term %s; %.0f s"
              % (cell["fit_grad0_max"] or 0.0, [round(x, 4) for x in ft], time.time() - t0))
        g0 = cell["fit_grad0_max"]
        assert g0 and 0 < g0 < float("inf"), "G-grad FAIL: gradient zero or not finite (%s)" % g0
        assert min(ft) < ft[0], "G-grad FAIL: fit term did not fall in 10 iterations"
        print("G-grad PASS")
        return
    os.makedirs(os.path.join(OUT, "cells"), exist_ok=True)
    tab = sim.set_edge_gains(tpools, BOUND)  # group table is cap-independent; fit_cell resets x
    for cap in CAPS:
        for seed in SEEDS:
            f = os.path.join(OUT, "cells", "cap%07.3f_s%d.json" % (cap, seed))
            if os.path.exists(f):
                continue
            t0 = time.time()
            cell, _ = fit_cell(sim, St, tpools, cap, seed)
            json.dump(cell, open(f, "w"))
            print("cap %7.3f seed %d: %3d iters %5.0f s  F %s  H %s  loss %.4f"
                  % (cap, seed, cell["iters"], time.time() - t0, cell["fpass"], cell["hpass"],
                     cell["loss_trace"][-1]), flush=True)
    names = [str(x) for x in np.load(G.BRAIN, allow_pickle=False)["pool_names"]]
    json.dump({"src_pool": tab["src_pool"].tolist(), "tgt_pool": tab["tgt_pool"].tolist(),
               "n_edges": tab["n_edges"].tolist(), "n_syn": tab["n_syn"].tolist(), "pool_names": names,
               "fox_pools": np.unique(pool[St["fox"]]).tolist(),
               "r_pools": np.unique(pool[St["comp"]["R"]]).tolist()},
              open(os.path.join(OUT, "groups.json"), "w"))
    analyze()


if __name__ == "__main__":
    main()
