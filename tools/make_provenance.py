"""Author-side: write PROVENANCE.json + PROVENANCE.md for release v0.2 from the private development repository.

The preregistrations, results and code of paper 1 were committed in a private research tree whose history is not
public. This script records, for every commit the paper cites (Table S1), its time and the files it touched, and the
sha256 of each such file as committed (LF content) next to the sha256 of the shipped copy (LF-normalised), so a reader
can check that the shipped bytes are the committed ones. Commit times are self-attested; the file hashes are not.

Usage (author only): python tools/make_provenance.py --dev ../flychess
Readers verify the shipped side with: python reproduce_paper1.py --verify
"""
import argparse
import hashlib
import json
import os
import subprocess

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Table S1 of paper 1: (label, commits in order). Each commit's touched files that ship in this release are listed.
TABLE_S1 = [
    ("Calibration of the rate model", ["1a29779", "b554245", "7839719"]),
    ("Reward circuit without fitting", ["194be67", "2c0b86d", "9d03f3b"]),
    ("Reward fit", ["2838bfe", "aca91ff", "e8ddc3c", "ee4ecbf", "db4fcfb"]),
    ("Learning tests on the rate model", ["9dd7298", "2fe2f2d", "d0b0c92"]),
    ("Spiking reference for relearning", ["40c920b", "57af1c1", "c89244f", "d451801", "db5a126"]),
]


def lf(b):
    return b.replace(b"\r\n", b"\n")


def sha(b):
    return hashlib.sha256(b).hexdigest()


def git(dev, *a, raw=False):
    out = subprocess.run(("git", "-C", dev) + a, capture_output=True, check=True).stdout
    return out if raw else out.decode("utf-8", "replace")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", required=True, help="path to the private development repository")
    a = ap.parse_args()
    shipped = set(subprocess.run(("git", "-C", HERE, "ls-files"), capture_output=True, check=True, text=True).stdout.split())
    shipped |= {p for p in subprocess.run(("git", "-C", HERE, "diff", "--cached", "--name-only"), capture_output=True,
                                           check=True, text=True).stdout.split()}
    tests = []
    for label, commits in TABLE_S1:
        rows = []
        for c in commits:
            full, when, subj = git(a.dev, "show", "-s", "--format=%H%n%ci%n%s", c).splitlines()[:3]
            files = []
            for path in git(a.dev, "show", "--name-only", "--format=", c).split():
                if path not in shipped:
                    continue
                at_commit = lf(git(a.dev, "show", "%s:%s" % (c, path), raw=True))
                here = lf(open(os.path.join(HERE, path), "rb").read()) if os.path.exists(os.path.join(HERE, path)) else None
                files.append({"path": path, "sha256_at_commit_lf": sha(at_commit),
                              "sha256_shipped_lf": sha(here) if here is not None else None,
                              "identical_to_shipped": here == at_commit})
            rows.append({"commit": full, "short": c, "time": when, "subject": subj, "files": files})
        tests.append({"test": label, "commits": rows})
    raw = {}
    for p in sorted(shipped):
        if p.startswith(("results/", "PREREGISTER_rate_", "rate/", "tools/analyze_bidir_ref_a2.py")) and os.path.isfile(os.path.join(HERE, p)):
            b = open(os.path.join(HERE, p), "rb").read()
            exact = p.startswith(("results/", "PREREGISTER_"))  # stored -text (.gitattributes): exact bytes on every OS
            raw[p] = {"mode": "raw" if exact else "lf", "sha256": sha(b if exact else lf(b))}
    out = {"note": "Commit times are self-attested (private repository). sha256_at_commit_lf = committed content, "
                   "sha256_shipped_lf = this release's copy with CRLF normalised to LF; files_sha256: mode raw = exact bytes (results and preregistrations, stored byte-for-byte, see .gitattributes), "
                   "mode lf = code, CRLF normalised to LF so a checkout on any OS verifies.",
           "dev_repository_head": git(a.dev, "rev-parse", "HEAD").strip(), "table_S1": tests, "files_sha256": raw}
    json.dump(out, open(os.path.join(HERE, "PROVENANCE.json"), "w", encoding="utf-8"), indent=1)

    md = ["# Provenance of paper 1 (release v0.2)", "",
          "The preregistrations, results and code of paper 1 were committed in a private research repository, so the",
          "commit hashes in Table S1 of the paper do not resolve on GitHub. This file records each cited commit, its time",
          "(UTC+8, self-attested) and the files it touched that ship here. For every file, *identical* means the shipped",
          "copy has the same content as the committed one (line endings normalised). Full hashes are in PROVENANCE.json;",
          "`python reproduce_paper1.py --verify` checks the shipped files against them, against the hashes recorded in each",
          "run's meta row and against the raw-log hashes in the EVALUATION.md reports.", "",
          "A file touched by several commits matches only its last version: earlier commits show *changed later*.", ""]
    for t in tests:
        md += ["## " + t["test"], "", "| Commit | Time | Subject | Files (shipped) |", "|---|---|---|---|"]
        for r in t["commits"]:
            fs = "; ".join("%s (%s)" % (f["path"], "identical" if f["identical_to_shipped"] else "changed later")
                           for f in r["files"]) or "(no shipped file)"
            md.append("| %s | %s | %s | %s |" % (r["short"], r["time"][:16], r["subject"].replace("|", "/")[:90], fs))
        md.append("")
    open(os.path.join(HERE, "PROVENANCE.md"), "w", encoding="utf-8").write("\n".join(md))
    n = sum(len(r["files"]) for t in tests for r in t["commits"])
    print("PROVENANCE: %d commits, %d file entries, %d raw file hashes" % (sum(len(t["commits"]) for t in tests), n, len(raw)))


if __name__ == "__main__":
    main()
