"""Protocol-matched rerun of Christie et al. 2026 Fig 6A.

Preregistration: docs/superpowers/reward-path/christie_match/PREREGISTER_christie_match.md (written and
committed in the private research tree as dd5a6bb before this script existed; the run used 8e7670c).
One process = one (W_SYN, floor) cell, stock regime: Shiu 2024 LIF constants and signs, no ELN_NEGATE,
PN_KC_GAIN 1, no plasticity. Needs data/brain_gpu.npz (scripts/build_data.py) and, for floor 1,
data/brain_gpu_min1.npz (scripts/build_floor_brains.py).

  --mode cell   : base + GRN+GIN x 20 rates x Fox intact/silenced (+ GRN-only ablation when W == 0.39),
                  30 trials (seeds 0-29) x 1000 ms, batched in chunks of --chunk seeds.
  --mode golden : W 0.39, floor 1, sugarL200 GRN-only, seeds 0-4, 1000 ms; PAM n_responsive per seed must equal
                  [7 11 7 14 0], the private-tree W_SYN grid result for that cell. Abort (exit 2) on mismatch.
  --mode timing : one GRN+GIN 200 Hz condition, 30 trials, for the cap projection (smoke; results/christie_match/smoke).

Resumable: every finished condition is appended to results/christie_match/parts/cell_w{W}_m{k}.jsonl; a rerun
skips conditions already present. Per-condition PAM spike counts [30, 307] go to .../pam/{cell}_{cond}.npy.
Readouts only; labels are computed by learn/analyze_christie_match.py.
--drop N applies the prereg's fixed drop order (1: Fox-sil at W 0.38/0.382 floor 5; 2: +floor 1; 3: +Fox-sil r<=60).
Exit 3 = a GPU call exceeded --max-call-s (runaway stop; what exists is written).

Run:
  python learn/christie_match.py --mode golden             (W 0.39, floor 1; about a minute)
  python learn/christie_match_run.py                        (the whole preregistered batch)
"""
import argparse
import json
import os
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument("--mode", required=True, choices=["cell", "golden", "timing"])
ap.add_argument("--w-syn", type=float, default=0.39)
ap.add_argument("--min-syn", type=int, default=1, choices=[1, 5])
ap.add_argument("--chunk", type=int, default=15, help="seeds per batched call (seeds are independent)")
ap.add_argument("--drop", type=int, default=0, choices=[0, 1, 2, 3])
ap.add_argument("--max-call-s", type=float, default=900.0)
A = ap.parse_args()

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BRAIN = os.path.join(_HERE, "data", "brain_gpu.npz" if A.min_syn == 5 else "brain_gpu_min1.npz")
os.environ["FLYCHESS_BRAIN"] = BRAIN          # before gpu_sim import (read at import)
NNZ = {1: 15090883, 5: 2700429}
sys.path.insert(0, _HERE)
import numpy as np  # noqa: E402

OUT = os.path.join(_HERE, "results", "christie_match")
RATES = [float(r) for r in range(10, 201, 10)]
SEEDS = list(range(30))
T_RUN = 1000.0
# Christie Data S4, left side (v783 root ids); resolved 2026-09-29, one left cell per type.
GINS = {"2N": {"AN_GNG_30": 720575940655014049, "CB0366": 720575940614763666, "CB0616": 720575940620874757,
               "CB0062": 720575940616103218, "CB0499": 720575940638103349, "CB0008": 720575940632648612,
               "CB0192": 720575940629888530},
        "3N": {"DNge174": 720575940645045527, "DNge173": 720575940610001220, "CB0038": 720575940631997032,
               "CB0051": 720575940632365905},
        "4N": {"DNge080": 720575940627847752, "CB0493": 720575940632252743, "CB0553": 720575940623211725}}
GOLDEN_PAM = [7, 11, 7, 14, 0]

meta = np.load(os.path.join(_HERE, "data", "neuron_meta.npz"), allow_pickle=False)
CT = meta["cell_type"].astype(str)
SIDE = meta["side"].astype(str)
CENTRAL = meta["super_class"].astype(str) == "central"
RID = np.load(os.path.join(_HERE, "data", "root_ids_sorted.npy"))
CELL = "cell_w%.3f_m%d" % (A.w_syn, A.min_syn)


class Runaway(Exception):
    pass


def gin_index():
    idx = []
    for order in GINS.values():
        for t, r in order.items():
            i = np.flatnonzero(RID == r)
            assert len(i) == 1 and CT[i[0]] == t and SIDE[i[0]] == "left", (t, r)
            idx.append(int(i[0]))
    return np.array(idx, np.int64)


def dropped(w, k, r, fox_sil):
    if not fox_sil or A.drop == 0:
        return False
    if w in (0.38, 0.382) and (k == 5 or A.drop >= 2):
        return True
    return A.drop >= 3 and r <= 60


def main():
    import gpu_sim as G
    from learn import condition as C
    from learn.christie_sweep import FOX, FDA_I, FDA_II, type_index
    G.ELN_NEGATE, G.PN_KC_GAIN = False, 1.0
    assert not getattr(G, "TYPE_W_SCALE", {}), "regime drift: TYPE_W_SCALE set"   # knob absent in v0.1 gpu_sim
    fox = type_index(CT, FOX); fda_i = type_index(CT, FDA_I); fda_ii = type_index(CT, FDA_II)
    assert len(fox) == 2
    pam = np.flatnonzero(np.char.startswith(CT, "PAM")).astype(np.int64)
    assert len(pam) == 307, len(pam)
    sugL = C.SUGAR[SIDE[C.SUGAR] == "left"]
    assert len(sugL) == 67
    gins = gin_index()
    assert len(gins) == 14 and not np.intersect1d(gins, sugL).size and not np.intersect1d(gins, fox).size
    grn_gin = np.concatenate([sugL, gins])

    G.W_SYN = A.w_syn
    sim = G.GpuSim()
    assert G.W_SYN == A.w_syn, "W_SYN not applied"
    assert os.path.abspath(G.BRAIN) == os.path.abspath(BRAIN), "FLYCHESS_BRAIN not honoured: %s" % G.BRAIN
    assert sim.net.min_syn == A.min_syn and sim.net.n_edges == NNZ[A.min_syn], (sim.net.min_syn, sim.net.n_edges)
    print("%s mode %s brain %s n_edges %d" % (CELL, A.mode, BRAIN, sim.net.n_edges), flush=True)

    def run(idx, hz, sil, seeds, label):
        """-> spike counts [len(seeds), N] int32, in --chunk batches."""
        out = []
        for c0 in range(0, len(seeds), A.chunk):
            ss = seeds[c0:c0 + A.chunk]
            drv = C.to_prob(idx, np.full(len(idx), hz, np.float32)) if hz > 0 else \
                C.to_prob(np.zeros(0, np.int64), np.zeros(0, np.float32))
            t = time.time()
            out.append(sim.run_batch([drv] * len(ss), list(ss), t_run=T_RUN, silence=sil).cpu().numpy())
            dt = time.time() - t
            print("  call %-26s seeds %2d-%2d %7.1f s" % (label, ss[0], ss[-1], dt), flush=True)
            if dt > A.max_call_s:
                raise Runaway("%s took %.0f s > max-call-s %.0f" % (label, dt, A.max_call_s))
        return np.concatenate(out, 0)

    if A.mode == "golden":
        assert A.w_syn == 0.39 and A.min_syn == 1
        c = run(sugL, 200.0, None, [0, 1, 2, 3, 4], "golden sugarL200")
        got = [int(x) for x in (c[:, pam] > 0).sum(1)]
        ok = got == GOLDEN_PAM
        os.makedirs(OUT, exist_ok=True)
        json.dump({"got": got, "want": GOLDEN_PAM, "pass": ok}, open(os.path.join(OUT, "golden.json"), "w"))
        print("golden PAM n_responsive %s vs %s -> %s" % (got, GOLDEN_PAM, "PASS" if ok else "FAIL"), flush=True)
        return 0 if ok else 2

    conds = []                                            # (name, idx, hz, silence)
    if A.mode == "timing":
        conds = [("gg200", grn_gin, 200.0, None)]
    else:
        conds.append(("base", None, 0.0, None))
        for r in RATES:
            conds.append(("gg%d" % r, grn_gin, r, None))
            if not dropped(A.w_syn, A.min_syn, r, True):
                conds.append(("gg%d-fox" % r, grn_gin, r, fox))
        if A.w_syn == 0.39:
            conds += [("grn%d" % r, sugL, r, None) for r in RATES]

    odir = os.path.join(OUT, "smoke" if A.mode == "timing" else "")
    os.makedirs(os.path.join(odir, "parts"), exist_ok=True); os.makedirs(os.path.join(odir, "pam"), exist_ok=True)
    part = os.path.join(odir, "parts", CELL + ".jsonl")
    done = {json.loads(l)["cond"] for l in open(part) if l.strip()} if os.path.exists(part) else set()
    for name, idx, hz, sil in conds:
        if name in done:
            continue
        t = time.time()
        try:
            c = run(idx, hz, sil, SEEDS, name)
        except Runaway as e:
            print("RUNAWAY %s" % e, flush=True)
            return 3
        pc = c[:, pam]                                   # [30, 307] spike counts
        np.save(os.path.join(odir, "pam", "%s_%s.npy" % (CELL, name)), pc.astype(np.int32))
        rates = c.astype(np.float64) / (T_RUN / 1000.0)
        tot = pc.sum(0)
        row = {"cond": name, "w_syn": A.w_syn, "min_syn": A.min_syn, "hz": hz,
               "drive": "base" if idx is None else ("grn" if name.startswith("grn") else "grn_gin"),
               "fox_silenced": sil is not None, "n_trials": len(SEEDS),
               "PAM_resp": int((tot > 0).sum()), "PAM_resp_5": int((pc[:5].sum(0) > 0).sum()),
               "PAM_mean_hz": float(rates[:, pam].mean()),
               "PAM_n_ge1hz": int((rates[:, pam].mean(0) >= 1.0).sum()),
               "trial_pam_n": [int(x) for x in (pc > 0).sum(1)],
               "trial_brain_n": [int(x) for x in (c > 0).sum(1)],
               "trial_central_hz": rates[:, CENTRAL].mean(1).tolist(),
               "fox_hz": float(rates[:, fox].mean()), "fdaI_hz": float(rates[:, fda_i].mean()),
               "fdaII_hz": float(rates[:, fda_ii].mean()), "wall_s": round(time.time() - t, 1)}
        with open(part, "a") as f:
            f.write(json.dumps(row) + "\n")
        print("%-10s PAM_resp %3d (5-trial %3d)  %.0f s" % (name, row["PAM_resp"], row["PAM_resp_5"], row["wall_s"]),
              flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
