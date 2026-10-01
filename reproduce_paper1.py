"""Paper 1 (rate model) from the shipped files, without a GPU.

  python reproduce_paper1.py --verify    hashes: shipped files vs PROVENANCE.json, run meta rows, EVALUATION.md lists,
                                         reference.json sources
  python reproduce_paper1.py --rescore   re-run the preregistered labelling of the reward fit, the learning tests and the
                                         spiking relearning reference on the shipped logs (in a temporary copy) and
                                         compare every label and number with the shipped result.json files

The original analyzers refuse to label unless the repository HEAD is the commit the runs were made at, which exists
only in the private development repository (PROVENANCE.md). --rescore passes that check by supplying the recorded
commit and code hashes, after --verify has checked the shipped bytes. Regenerating the runs themselves needs a CUDA
GPU (README: "Run it, paper 1").
"""
import argparse
import copy
import hashlib
import json
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
RES = os.path.join(HERE, "results")
CLEAN = {"rate": False, "learn": False}
FAILS = []


def sha_raw(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def sha_lf(p):
    return hashlib.sha256(open(p, "rb").read().replace(b"\r\n", b"\n")).hexdigest()


def check(ok, what):
    print("%s  %s" % ("PASS" if ok else "FAIL", what))
    if not ok:
        FAILS.append(what)


def metas(d):
    out = {}
    for f in sorted(os.listdir(d)):
        if f.endswith(".jsonl"):
            with open(os.path.join(d, f), encoding="utf-8") as fh:
                row = json.loads(fh.readline())
            if row.get("phase") == "meta":
                out[f] = row
    return out


def verify():
    prov = json.load(open(os.path.join(HERE, "PROVENANCE.json"), encoding="utf-8"))
    files = prov["files_sha256"]
    bad = [p for p, e in files.items() if not os.path.isfile(os.path.join(HERE, p))
           or (sha_raw if e["mode"] == "raw" else sha_lf)(os.path.join(HERE, p)) != e["sha256"]]
    check(not bad, "%d shipped files match PROVENANCE.json (results and preregistrations byte for byte, code with LF line endings)%s"
          % (len(files), (" (bad: %s)" % bad[:5]) if bad else ""))
    for t in prov["table_S1"]:
        last = {}
        for r in t["commits"]:
            for f in r["files"]:
                last[f["path"]] = f
        bad = [p for p, f in last.items() if f["sha256_shipped_lf"] != sha_lf(os.path.join(HERE, p))]
        check(not bad, "%s: shipped files equal their last cited commit (%d files)%s" % (t["test"], len(last), (" bad %s" % bad) if bad else ""))
    import re
    for d, prereg in (("rate_chunk2", "PREREGISTER_rate_chunk2_bridge.md"), ("rate_chunk2_bidir_ref", "PREREGISTER_rate_chunk2_bidir_ref.md")):
        dd = os.path.join(RES, d)
        listed = dict(re.findall(r"^- (\S+\.jsonl) ([0-9a-f]{64})\s*$", open(os.path.join(dd, "EVALUATION.md"), encoding="utf-8").read(), re.M))
        here = {f: sha_raw(os.path.join(dd, f)) for f in listed if os.path.isfile(os.path.join(dd, f))}
        check(len(here) == len(listed) and all(here[f] == h for f, h in listed.items()),
              "%s: %d raw logs match the EVALUATION.md hash list" % (d, len(listed)))
        ms = metas(dd)
        ph = sha_raw(os.path.join(HERE, prereg))
        check(all(m["prereg_sha256"] == ph for m in ms.values()), "%s: %d run meta rows name this %s (sha256 %s)" % (d, len(ms), prereg, ph[:12]))
        rh = sha_raw(os.path.join(RES, "rate_chunk2", "reference.json"))
        check(all(m["reference_sha256"] == rh for m in ms.values()), "%s: meta rows name the shipped reference.json" % d)
        diff = sorted({f for m in ms.values() for f, h in m["code_sha256"].items() if sha_lf(os.path.join(HERE, f)) != h})
        print("INFO  %s: code files that changed after these runs: %s" % (d, diff or "none"))
        if d == "rate_chunk2":
            check(diff == ["rate/chunk2.py"], "rate_chunk2: only rate/chunk2.py changed after the 24 runs (extended for the spiking reference; PROVENANCE.md)")
        else:
            check(not diff, "rate_chunk2_bidir_ref: every code file equals what the 10 runs recorded")
    ref = json.load(open(os.path.join(RES, "rate_chunk2", "reference.json"), encoding="utf-8"))
    srcs = ref["sources"]
    bad = [k for k, h in srcs.items() if sha_raw(os.path.join(HERE, *k.split("/"))) != h]
    check(not bad, "reference.json: %d paper-0 source files match%s" % (len(srcs), (" bad %s" % bad) if bad else ""))


def strip(d, drop=("analyzer", "reanalyzed", "previous_result_sha256", "provenance", "amendment_A2")):
    d = json.loads(json.dumps(d))  # fresh results carry int dict keys; the shipped JSON has strings
    return {k: v for k, v in d.items() if k not in drop}


def same(a, b, path="", out=None, tol=1e-9):
    out = [] if out is None else out
    if isinstance(a, dict) and isinstance(b, dict):
        for k in set(a) | set(b):
            if k not in a or k not in b:
                out.append(path + "/" + str(k) + " (missing)")
            else:
                same(a[k], b[k], path + "/" + str(k), out, tol)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append(path + " (length)")
        for i, (x, y) in enumerate(zip(a, b)):
            same(x, y, "%s[%d]" % (path, i), out, tol)
    elif isinstance(a, float) or isinstance(b, float):
        if a is None or b is None or abs(a - b) > tol * max(1.0, abs(a), abs(b)):
            out.append("%s (%r != %r)" % (path, a, b))
    elif a != b:
        out.append("%s (%r != %r)" % (path, a, b))
    return out


def rescore():
    from rate import chunk2 as C2
    from rate import chunk1_fit as F1
    tmp = tempfile.mkdtemp(prefix="np_rescore_")
    try:
        # reward fit (chunk 1, chunk-0 base): label + per-cap table from the 45 shipped cells
        d = os.path.join(tmp, "rate_chunk1_fit_c0")
        shutil.copytree(os.path.join(RES, "rate_chunk1_fit_c0"), d)
        os.remove(os.path.join(d, "result.json"))
        F1.OUT = d
        F1.PREREG += " + PREREGISTER_rate_chunk1_fit_c0.md @ ee4ecbf (chunk-0 base)"
        F1.analyze()
        diff = same(strip(json.load(open(os.path.join(d, "result.json")))),
                    strip(json.load(open(os.path.join(RES, "rate_chunk1_fit_c0", "result.json")))))
        check(not diff, "reward fit: label, g* and per-cap rows reproduce%s" % ("" if not diff else " (%s)" % diff[:4]))

        # learning tests (chunk 2, 24 runs)
        d = os.path.join(tmp, "rate_chunk2")
        shutil.copytree(os.path.join(RES, "rate_chunk2"), d)
        for f in ("result.json", "RESULT.md"):
            os.remove(os.path.join(d, f))
        ms = metas(d)
        heads = {m["git_head"] for m in ms.values()}
        code = next(iter(ms.values()))["code_sha256"]
        res = C2.analyze(d, C2.load_reference(), state=(heads.pop(), CLEAN), code_sha=code)
        diff = same(strip(res), strip(json.load(open(os.path.join(RES, "rate_chunk2", "result.json")))))
        check(not diff, "learning tests: every label and number of the 24 runs reproduces%s" % ("" if not diff else " (%s)" % diff[:4]))

        # spiking relearning reference (10 LIF runs; amendment A2: run 10 recorded a later commit, same code/data/prereg)
        d = os.path.join(tmp, "rate_chunk2_bidir_ref")
        shutil.copytree(os.path.join(RES, "rate_chunk2_bidir_ref"), d)
        for f in ("result.json", "RESULT.md"):
            os.remove(os.path.join(d, f))
        ms = metas(d)
        h1 = max(({m["git_head"] for m in ms.values()}), key=lambda h: sum(m["git_head"] == h for m in ms.values()))
        read = C2.read_rows

        def read_a2(path):
            rows = read(path)
            for r in rows:
                if r.get("phase") == "meta":
                    r["git_head"] = h1
            return rows
        C2.read_rows = read_a2
        code = next(iter(ms.values()))["code_sha256"]
        res = C2.analyze_bidir_ref(d, C2.load_reference(), state=(h1, CLEAN), code_sha=code)
        C2.read_rows = read
        diff = same(strip(res), strip(json.load(open(os.path.join(RES, "rate_chunk2_bidir_ref", "result.json")))))
        check(not diff, "spiking relearning reference + re-scored rate model reproduce%s" % ("" if not diff else " (%s)" % diff[:4]))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--rescore", action="store_true")
    a = ap.parse_args()
    if not (a.verify or a.rescore):
        a.verify = a.rescore = True
    if a.verify:
        verify()
    if a.rescore:
        rescore()
    print("\n%s (%d failure%s)" % ("ALL PASS" if not FAILS else "FAILED", len(FAILS), "" if len(FAILS) == 1 else "s"))
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
