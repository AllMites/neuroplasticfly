"""Driver for PREREGISTER_christie_match.md. Order: golden -> timing (floors 1, 5) -> cap projection + drop level
-> 12 cells (floor 1 first, W ascending) -> analyzer. Each cell runs in its own process (one GpuSim per brain).

  python learn/christie_match_run.py [--dry]      (--dry: golden + timing + projection only)

Ledger: results/christie_match/ledger.jsonl. Cap 600 GPU-min (the private run used 132): no cell starts once the ledger reaches it; the
unstarted cells go to not_run.json (never imputed). Cells are resumable, so a crash reruns the cell once.
"""
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
CELLPY = os.path.join(HERE, "learn", "christie_match.py")
DIR = os.path.join(HERE, "results", "christie_match")
LEDGER = os.path.join(DIR, "ledger.jsonl")
CAP_MIN = 600.0
WS = (0.275, 0.37, 0.38, 0.382, 0.386, 0.39)
FLOORS = (1, 5)
STARTUP_S = 60.0          # ponytail: sim build + compile per process, rough; measured by timing jobs below


def used_min():
    if not os.path.exists(LEDGER):
        return 0.0
    return sum(json.loads(l)["wall_s"] for l in open(LEDGER) if l.strip()) / 60.0


def job(name, argv):
    os.makedirs(os.path.join(DIR, "logs"), exist_ok=True)
    t = time.time()
    with open(os.path.join(DIR, "logs", name + ".log"), "a") as f:
        rc = subprocess.call([PY, CELLPY] + argv, cwd=HERE, stdout=f, stderr=subprocess.STDOUT)
    dt = time.time() - t
    with open(LEDGER, "a") as f:
        f.write(json.dumps({"job": name, "exit": rc, "wall_s": round(dt, 1),
                            "t_end": time.strftime("%Y-%m-%d %H:%M:%S")}) + "\n")
    print("%-24s exit %s  %.1f min  (ledger %.1f GPU-min)" % (name, rc, dt / 60, used_min()), flush=True)
    return rc, dt


def n_conds(w, k, drop):
    n = 1
    for r in range(10, 201, 10):
        n += 1
        fox_drop = drop and ((w in (0.38, 0.382) and (k == 5 or drop >= 2)) or (drop >= 3 and r <= 60))
        n += 0 if fox_drop else 1
    return n + (20 if w == 0.39 else 0)


def main():
    os.makedirs(DIR, exist_ok=True)
    rc, _ = job("golden", ["--mode", "golden", "--w-syn", "0.39", "--min-syn", "1"])
    if rc != 0:
        print("GOLDEN FAIL (exit %s): abort, see logs/golden.log" % rc); return 2
    per_cond = {}
    for k in FLOORS:
        rc, dt = job("timing_m%d" % k, ["--mode", "timing", "--w-syn", "0.39", "--min-syn", str(k)])
        if rc != 0:
            print("TIMING FAIL m%d: abort" % k); return 2
        row = [json.loads(l) for l in open(os.path.join(DIR, "smoke", "parts", "cell_w0.390_m%d.jsonl" % k))][-1]
        per_cond[k] = row["wall_s"]
    left = CAP_MIN - used_min()
    proj, drop = None, 0
    for drop in (0, 1, 2, 3):
        proj = sum(n_conds(w, k, drop) * per_cond[k] + STARTUP_S for w in WS for k in FLOORS) / 60.0
        if proj <= left:
            break
    plan = {"s_per_cond": per_cond, "projected_min": round(proj, 1), "budget_left_min": round(left, 1),
            "drop": drop, "fits": proj <= left}
    json.dump(plan, open(os.path.join(DIR, "plan.json"), "w"), indent=1)
    print("projection %s" % plan, flush=True)
    if "--dry" in sys.argv:
        return 0
    not_run = {}
    for k in FLOORS:
        for w in WS:
            cell = "cell_w%.3f_m%d" % (w, k)
            if used_min() >= CAP_MIN:
                not_run[cell] = "cap %.0f GPU-min reached" % CAP_MIN; continue
            argv = ["--mode", "cell", "--w-syn", "%g" % w, "--min-syn", str(k), "--drop", str(drop)]
            rc, _ = job(cell, argv)
            if rc not in (0, 3):
                rc, _ = job(cell + "_retry", argv)          # resumable: finished conditions are kept
                if rc not in (0, 3):
                    not_run[cell] = "crash twice (partial conditions kept)"
            if rc == 3:
                not_run[cell] = "runaway stop (max-call-s); partial conditions kept"
    json.dump(not_run, open(os.path.join(DIR, "not_run.json"), "w"), indent=1)
    return subprocess.call([PY, os.path.join(HERE, "learn", "analyze_christie_match.py")], cwd=HERE)


if __name__ == "__main__":
    sys.exit(main())
