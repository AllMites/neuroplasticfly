"""Dynamic check of the sugar -> PAM intermediates (levers L2 / L3 / L4).

Runs the sugar-US condition against a baseline trial, N seeds, and dumps mean
rates for the neuron groups the structural analysis (regime/signed_paths.py)
named: the hop-2 intermediates, the hop-3 last-hop types, the restored non-KC
unlabelled presynaptic partners of PAM, and PAM split by hemibrain_type
(PAM01..PAM15).

Matrix comes from FLYCHESS_BRAIN, exactly like learn/us_path_sweep.py.

Run:
  FLYCHESS_BRAIN=data/brain_gpu_danfloor1.npz .venv/Scripts/python.exe \
      learn/pam_intermediates_sweep.py --seeds 3 --tag danfloor1
  FLYCHESS_BRAIN=data/brain_gpu_danfloor1_nokc.npz .venv/Scripts/python.exe \
      learn/pam_intermediates_sweep.py --seeds 3 --tag nokc
  FLYCHESS_BRAIN=data/brain_gpu_danfloor1.npz .venv/Scripts/python.exe \
      learn/pam_intermediates_sweep.py --seeds 3 --tag danfloor1_norm --norm 29.15

--norm sets gpu_sim.NORM_TOTAL_TARGET BEFORE GpuSim() is constructed (the value
is read once in __init__). Numbers from a --norm run are not comparable with
numbers from a run without it: the per-postsynaptic factor divides out the
in-degree the DAN-floor patch restored.

Measurement rule for every group: mean is reported next to n, the number of
neurons below 1 Hz, and the max. A mean over a mostly-silent population reads
far better than the population behaves.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "flywatch"))
import gpu_sim as G  # noqa: E402

SIGNED = os.path.join(_HERE, "results", "signed_paths.json")
ANN = os.path.join(_HERE, "data", "ext", "Supplemental_file1_neuron_annotations.tsv")


def hemibrain_type(root_ids):
    import csv as _csv
    hb = {}
    with open(ANN, newline="", encoding="utf-8") as fh:
        for r in _csv.DictReader(fh, delimiter="\t"):
            v = (r.get("hemibrain_type") or "").strip()
            if v:
                hb[int(r["root_id"])] = v
    return np.array([hb.get(int(x), "") for x in root_ids], dtype=object)


def build_groups(n_top_types):
    """(label -> int64 index array), from neuron_meta plus signed_paths.json."""
    meta = np.load(os.path.join(_HERE, "data", "neuron_meta.npz"), allow_pickle=False)
    ct = meta["cell_type"].astype(str)
    csc = meta["cell_sub_class"].astype(str)
    root_ids = np.load(os.path.join(_HERE, "data", "root_ids_sorted.npy"))
    assert os.path.exists(SIGNED), "run regime/signed_paths.py first (%s missing)" % SIGNED
    sp_res = json.load(open(SIGNED))

    groups = {}
    order = []

    def add(label, idx):
        idx = np.asarray(idx, np.int64)
        if len(idx):
            groups[label] = idx
            order.append(label)

    is_pam = np.char.startswith(ct, "PAM")
    add("PAM (all)", np.flatnonzero(is_pam))
    add("proboscis MN (control)",
        np.flatnonzero(meta["cell_sub_class"].astype(str) == "proboscis_motor_neuron"))
    add("sugar GRN (drive check)", np.flatnonzero(csc == "sugar/water"))

    # (a) hop-2 intermediates
    add("hop2 intermediates (all)", sp_res["hop2_top_neurons"])
    for t in sp_res["hop2_top_types"]:
        add("hop2 type %s" % t, np.flatnonzero(ct == t))

    # hop-3 last-hop types, top N by |contribution|
    for rec in sp_res["hop3_intermediates"][:n_top_types]:
        t = rec["cell_type"]
        add("hop3 type %s" % t, np.flatnonzero(ct == t))

    # (b) restored non-KC unlabelled ("none"-class SMP/CRE) presyn to PAM
    add("PAM presyn unlabelled (restored, non-KC)", sp_res["pam_presyn_unlabelled_idx"])

    # (c) PAM by hemibrain subtype
    hb = hemibrain_type(root_ids)
    pam_idx = np.flatnonzero(is_pam)
    subs = {}
    for i in pam_idx:
        subs.setdefault(str(hb[i]) or "(no hemibrain_type)", []).append(int(i))
    for k in sorted(subs):
        add("PAM sub %s" % k, subs[k])
    return groups, order


def summarise(rates, idx):
    r = rates[idx]
    return {"n": int(len(idx)), "mean": float(r.mean()), "max": float(r.max()),
            "n_silent": int((r < 1.0).sum()),
            "frac_silent": float((r < 1.0).mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--norm", type=float, default=0.0,
                    help="NORM_TOTAL_TARGET; 29.15 is the regime/measure.py value")
    ap.add_argument("--top-types", type=int, default=20)
    a = ap.parse_args()

    t0 = time.time()
    groups, order = build_groups(a.top_types)
    if a.norm > 0.0:
        G.NORM_TOTAL_TARGET = a.norm  # must precede GpuSim()
    from learn import condition as C
    from learn import plastic as PL

    sim = G.GpuSim()
    assert G.NORM_TOTAL_TARGET == a.norm, "NORM_TOTAL_TARGET not applied"
    P = PL.Plastic.real()

    acc = {"base": {}, "sugar": {}}
    regime = {"base": [], "sugar": []}
    for seed in range(a.seeds):
        rb = C.trial(sim, "misery", seed=seed)
        rs = C.trial(sim, "misery", us="reward", us_path="grn", seed=seed)
        for cond, rates in (("base", rb), ("sugar", rs)):
            for lab in order:
                acc[cond].setdefault(lab, []).append(summarise(rates, groups[lab]))
            r = C.readout(rates, P)
            regime[cond].append((r["kc_active"], r["central_active"]))
        print("seed %d done (%.1f s)" % (seed, time.time() - t0))

    def agg(cond, lab):
        rows = acc[cond][lab]
        return {"n": rows[0]["n"],
                "mean": float(np.mean([r["mean"] for r in rows])),
                "max": float(np.max([r["max"] for r in rows])),
                "n_silent": float(np.mean([r["n_silent"] for r in rows])),
                "frac_silent": float(np.mean([r["frac_silent"] for r in rows]))}

    b = np.load(G.BRAIN, allow_pickle=False)
    out = {"brain": os.path.abspath(G.BRAIN), "n_edges": int(sim.net.n_edges),
           "dan_floor": int(b["dan_floor"]) if "dan_floor" in b.files else None,
           "dan_floor_targets": (str(b["dan_floor_targets"][0])
                                 if "dan_floor_targets" in b.files else None),
           "norm_total_target": G.NORM_TOTAL_TARGET, "seeds": a.seeds,
           "groups": {lab: {"base": agg("base", lab), "sugar": agg("sugar", lab)}
                      for lab in order},
           "regime": {k: {"kc_active": float(np.mean([x[0] for x in v])),
                          "central_active": float(np.mean([x[1] for x in v]))}
                      for k, v in regime.items()}}
    os.makedirs(os.path.join(_HERE, "results"), exist_ok=True)
    path = os.path.join(_HERE, "results", "pam_intermediates_%s.json" % a.tag)
    json.dump(out, open(path, "w"), indent=1)

    print("")
    print("brain=%s  n_edges=%d  dan_floor=%s  targets=%s  NORM_TOTAL_TARGET=%s  seeds=%d"
          % (G.BRAIN, out["n_edges"], out["dan_floor"], out["dan_floor_targets"],
             out["norm_total_target"], a.seeds))
    print("| group | n | base mean | sugar mean | delta | sugar max | base silent | sugar silent |")
    print("|-------|---|-----------|------------|-------|-----------|-------------|--------------|")
    for lab in order:
        gb, gs = out["groups"][lab]["base"], out["groups"][lab]["sugar"]
        print("| %-42s | %5d | %9.3f | %10.3f | %+8.3f | %9.2f | %5.1f/%d | %5.1f/%d |"
              % (lab, gb["n"], gb["mean"], gs["mean"], gs["mean"] - gb["mean"],
                 gs["max"], gb["n_silent"], gb["n"], gs["n_silent"], gs["n"]))
    print("")
    print("regime: base kc_active %.4f central_active %.4f | sugar kc_active %.4f central_active %.4f"
          % (out["regime"]["base"]["kc_active"], out["regime"]["base"]["central_active"],
             out["regime"]["sugar"]["kc_active"], out["regime"]["sugar"]["central_active"]))
    print("(kc_active is authored-invariant under the fingerprint graft; read central_active)")
    print("wrote %s  (%.1f s)" % (path, time.time() - t0))


if __name__ == "__main__":
    main()
