"""Preregister an experiment, then label and verify its result.

Workflow (docs/prereg.md has a worked example):
  1. write PREREGISTER_<name>.md from docs/PREREGISTER_TEMPLATE.md and commit it BEFORE any data exists
  2. python prereg.py freeze PREREGISTER_<name>.md --code my_run.py my_analysis.py -o results/<name>_meta.json
     (refuses an uncommitted or modified preregistration; records its commit, time and every sha256)
  3. run, then label with verdict() below: one analysis pass, rules exactly as written
  4. python prereg.py verify results/<name>_meta.json [--results results/<name>.json ...]
     (re-hashes the preregistration, the code and the results; checks the preregistration commit exists)

Label rules (verdict):
  PASS   an ordering (a > b for every pair) holds on baseline-subtracted scores in >= k of n seeds
  SUBST  the same ordering holds on raw totals in >= k of n seeds. A PASS without SUBST is labelled
         "PASS BY RULE, NOT SUBSTANTIVE": subtracting a baseline can manufacture an ordering out of near-silence
  MECH   optional: the ordering holds on the cell's own synaptic input in >= k of n seeds (the mechanism check)
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys


def holds(flags, k=4):
    """Seed-count rule: True when at least k per-seed checks are True (default 4 of 5)."""
    return sum(bool(f) for f in flags) >= k


def ordered(x, pairs, positive=()):
    """One seed: every (a, b) in pairs has x[a] > x[b], and every name in positive has x[name] > 0."""
    return all(x[a] > x[b] for a, b in pairs) and all(x[p] > 0 for p in positive)


def verdict(scores, pairs, totals=None, mech=None, positive=(), k=4):
    """Label one preregistered ordering.

    scores, totals, mech: lists (one per seed) of {condition: value}. scores are baseline-subtracted,
    totals are raw (e.g. spike counts over the stimulus window), mech is the cell's own input drive.
    pairs: [(a, b), ...] meaning a > b. positive: conditions whose score must also be > 0.
    """
    r = {"n_seeds": len(scores), "k": k, "n_ok": sum(ordered(x, pairs, positive) for x in scores)}
    r["PASS"] = r["n_ok"] >= k
    for key, v in (("SUBST", totals), ("MECH", mech)):
        if v is not None:
            r["n_" + key.lower()] = sum(ordered(x, pairs) for x in v)
            r[key] = r["n_" + key.lower()] >= k
    if not r["PASS"]:
        r["label"] = "FAIL"
    elif r.get("SUBST") is False:
        r["label"] = "PASS BY RULE, NOT SUBSTANTIVE"
    elif r.get("MECH") is False:
        r["label"] = "PASS, MECHANISM NOT SHOWN"
    else:
        r["label"] = "PASS"
    return r


# ---------- freeze / verify ----------
def sha_lf(path):
    # ponytail: every file hashed with CRLF -> LF so a checkout on any OS verifies; byte-exact mode if ever needed
    return hashlib.sha256(open(path, "rb").read().replace(b"\r\n", b"\n")).hexdigest()


def git(*a):
    return subprocess.run(["git", *a], capture_output=True, text=True, check=True).stdout.strip()


def freeze(prereg, code, out):
    if git("status", "--porcelain", "--", prereg):
        sys.exit("refused: %s is uncommitted or modified; commit it before any data exists" % prereg)
    commit = git("log", "-1", "--format=%H", "--", prereg)
    if not commit:
        sys.exit("refused: %s has no commit" % prereg)
    meta = {"prereg": prereg, "prereg_sha256": sha_lf(prereg), "prereg_commit": commit,
            "prereg_commit_time": git("show", "-s", "--format=%ci", commit),
            "code_sha256": {f: sha_lf(f) for f in code}}
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    json.dump(meta, open(out, "w"), indent=1)
    print("frozen %s at %s (%s)" % (prereg, commit[:12], meta["prereg_commit_time"]))


def verify(meta_path, results=()):
    meta, bad = json.load(open(meta_path)), []
    if results:
        rec = meta.setdefault("results_sha256", {})
        clash = [f for f in results if f in rec and rec[f] != sha_lf(f)]
        if clash:
            sys.exit("refused: %s already recorded with a different hash" % ", ".join(clash))
        rec.update({f: sha_lf(f) for f in results})
        json.dump(meta, open(meta_path, "w"), indent=1)
        print("recorded %d result file(s) in %s" % (len(results), meta_path))
    files = {meta["prereg"]: meta["prereg_sha256"], **meta["code_sha256"], **meta.get("results_sha256", {})}
    for f, h in files.items():
        if not os.path.exists(f) or sha_lf(f) != h:
            bad.append(f)
    try:
        git("cat-file", "-e", meta["prereg_commit"] + ":" + meta["prereg"].replace(os.sep, "/"))
    except subprocess.CalledProcessError:
        bad.append("prereg commit %s missing or without %s" % (meta["prereg_commit"][:12], meta["prereg"]))
    for b in bad:
        print("FAIL", b)
    print("%s: %d files checked, %d failed" % (meta_path, len(files), len(bad)))
    return not bad


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("freeze", help="record a committed preregistration and the code hashes before running")
    f.add_argument("prereg")
    f.add_argument("--code", nargs="*", default=[])
    f.add_argument("-o", "--out", required=True)
    v = sub.add_parser("verify", help="re-hash the preregistration, code and results named in a meta file")
    v.add_argument("meta")
    v.add_argument("--results", nargs="*", default=[], help="record these result files first (once, after the run)")
    a = ap.parse_args()
    if a.cmd == "freeze":
        freeze(a.prereg, a.code, a.out)
    else:
        sys.exit(0 if verify(a.meta, a.results) else 1)


if __name__ == "__main__":
    main()
