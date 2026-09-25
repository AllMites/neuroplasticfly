"""Clone -> rung A. One command: fetch the connectome, build data/, gate the CS channels,
run one conditioning seed, and check the paired odour's avoid-index shift against the
published +11.05 Hz.

    python reproduce.py                 # ~852 MB download, ~1 min build, ~22 min rung A
    python reproduce.py --skip-download # data/ext already populated

Each step is skipped when its output is already there, so a rerun after a crash resumes.
Exits non-zero the moment a step fails, and prints wall clock per step.

Published (docs/superpowers/option-a/odour_conditioning_2026-09-21.md, 5 seeds, one brain):
paired ORN_DC2 +11.05 +- 0.20 Hz, unpaired ORN_D +1.43, never-paired ORN_DA1 +0.56,
lesion -0.73. This script runs the learn and lesion arms of seed 0 only; shuffle and
reversed are documented there and cost ~20 min each.
"""
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
STATE = "repro_s0"
RESULT = os.path.join(HERE, "results", "condition_%s.json" % STATE)

TARGET = 11.05          # Hz, paired DC2, mean of 5 seeds
BAND = 0.6              # 3 x the 0.195 Hz across-seed sd


def step(title, argv, skip=False, why=""):
    if skip:
        print("\n== %s: SKIP (%s)" % (title, why))
        return 0.0
    print("\n== %s\n   %s" % (title, " ".join(argv[1:])))
    t0 = time.time()
    rc = subprocess.call(argv, cwd=HERE)
    dt = time.time() - t0
    print("   [%s] %.1f s" % ("ok" if rc == 0 else "EXIT %d" % rc, dt))
    if rc:
        raise SystemExit("reproduce.py: '%s' failed with exit %d" % (title, rc))
    return dt


def data_is_built():
    # stderr too: on a first run this legitimately reports "missing data/brain_gpu.npz",
    # which is the question being asked, not a failure worth printing.
    return subprocess.call([PY, os.path.join("scripts", "checksum_data.py"), "--verify"],
                           cwd=HERE, stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL) == 0


def require_cuda():
    import torch
    if not torch.cuda.is_available():
        raise SystemExit(
            "reproduce.py: CUDA is required. The sim is one flat CUDA tensor and the CPU\n"
            "  path is not supported. Install the cu128 torch wheel: uv sync")
    print("device: %s, torch %s" % (torch.cuda.get_device_name(0), torch.__version__))


def shift_from(path):
    """Avoid-index shift, last probe minus baseline, per arm and stimulus.

    Same reduction as learn/analyze_seeds.py:26-45.
    """
    cv = json.load(open(path))["curves"]
    out = {}
    for arm, per_stim in cv.items():
        for stim, d in per_stim.items():
            t = {int(k): v for k, v in d.items()}
            out[(arm, stim)] = t[max(t)][0] - t[0][0]
    return out


def main(argv):
    t_all = time.time()
    require_cuda()
    times = {}

    times["fetch"] = step("1/4 fetch FlyWire v783", [PY, os.path.join("scripts", "fetch_data.py")],
                          skip="--skip-download" in argv, why="--skip-download")

    built = data_is_built()
    times["build"] = step("2/4 build data/", [PY, os.path.join("scripts", "build_data.py")],
                          skip=built, why="checksums already match data/CHECKSUMS.json")

    times["gate"] = step("3/4 gate CS channels (PN_KC_GAIN 8)",
                         [PY, os.path.join("learn", "gate_cs_channels.py"), "--gain", "8"])

    times["condition"] = step(
        "4/4 rung A, seed 0, arms learn+lesion (~22 min)",
        [PY, os.path.join("learn", "condition.py"), "--state", STATE, "--cs", "odour",
         "--regime", "eln8", "--arms", "learn,lesion", "--us-path", "dan",
         "--eta", "5e-6", "--train", "30", "--probe", "5", "--seed0", "0"],
        skip=os.path.exists(RESULT), why="results/condition_%s.json exists" % STATE)

    sh = shift_from(RESULT)
    dc2 = sh.get(("learn", "dc2"))
    if dc2 is None:
        raise SystemExit("no learn/dc2 curve in %s; arms were %s"
                         % (RESULT, sorted({a for a, _ in sh})))
    ok = abs(dc2 - TARGET) <= BAND

    print("\n" + "-" * 68)
    for (arm, stim), v in sorted(sh.items()):
        print("  %-8s %-6s %+7.2f Hz" % (arm, stim, v))
    for k, v in times.items():
        print("  %-10s %6.1f s" % (k, v))
    print("  %-10s %6.1f min total" % ("wall clock", (time.time() - t_all) / 60.0))
    print("rung A seed 0: DC2 %+.2f Hz (target %+.2f +- %.1f)  %s  wall clock %.0f min"
          % (dc2, TARGET, BAND, "PASS" if ok else "FAIL", (time.time() - t_all) / 60.0))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
