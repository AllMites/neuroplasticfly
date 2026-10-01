"""Run --analyze-bidir-ref under amendment A2 (PREREGISTER_rate_chunk2_bidir_ref_A2.md).

Verifies the A2 conditions, presents run 10's git_head as H1 to the unmodified analyzer, pins the analyzer
state to H1 (real dirty flags kept), then records the A2 provenance in result.json and RESULT.md.
Lives outside CODE_FILES on purpose: editing rate/chunk2.py would void the runs' code_sha256.
"""
import json
import os
import subprocess
import sys

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_HERE, "rate"))
sys.path.insert(0, _HERE)
import chunk2 as C2  # noqa: E402

H1 = "57af1c1a2fb6c76282e37b419a8f108734cf048a"
H2 = "66b4f44d5f3ad190ffeb7f9ce17c42150722b6a6"
H2_LOG = "bidir_depress_lifs4.jsonl"
A2_DOC = os.path.join(_HERE, "PREREGISTER_rate_chunk2_bidir_ref_A2.md")


def git(*args):
    return subprocess.run(("git",) + args, cwd=_HERE, capture_output=True, text=True, timeout=30, check=True).stdout


def forbidden(paths):
    return [p for p in paths if p in C2.CODE_FILES or p in C2.DATA_FILES or p.startswith(("rate/", "learn/"))]


def check_a2():
    if not os.path.exists(A2_DOC):
        raise SystemExit("A2 refused: amendment file missing")
    metas = {}
    for s in C2.LIF_SEEDS:
        for r in C2.RULES:
            n = C2.lif_log_name(r, s)
            metas[n] = [x for x in C2.read_rows(os.path.join(C2.OUT_REF, n)) if x.get("phase") == "meta"][0]
    heads = {n: m["git_head"] for n, m in metas.items()}
    if {n for n, h in heads.items() if h == H2} != {H2_LOG} or set(heads.values()) != {H1, H2}:
        raise SystemExit("A2 refused: heads are not H1 everywhere + H2 on %s only: %s" % (H2_LOG, heads))
    if subprocess.run(("git", "merge-base", "--is-ancestor", H1, H2), cwd=_HERE).returncode != 0:
        raise SystemExit("A2 refused: H1 is not an ancestor of H2")
    d12 = git("diff", "--name-only", H1, H2).split()
    head = git("rev-parse", "HEAD").strip()
    d1h = git("diff", "--name-only", H1, head).split()
    if forbidden(d12) or forbidden(d1h):
        raise SystemExit("A2 refused: diff touches code/data: H1..H2 %s, H1..HEAD %s" % (forbidden(d12), forbidden(d1h)))
    for k in ("code_sha256", "data_sha256", "prereg_sha256"):
        if len({json.dumps(m.get(k), sort_keys=True) for m in metas.values()}) != 1:
            raise SystemExit("A2 refused: runs carry different %s" % k)
    return {"H1": H1, "H2": H2, "H2_log": H2_LOG, "diff_H1_H2": d12, "analyzer_head_real": head, "diff_H1_HEAD": d1h,
            "amendment_sha256": C2.sha256_file(A2_DOC)}


def main():
    a2 = check_a2()
    read = C2.read_rows

    def read_a2(path):  # ponytail: in-memory head swap for run 10 only; the log on disk is untouched
        rows = read(path)
        if os.path.basename(path) == H2_LOG:
            for x in rows:
                if x.get("phase") == "meta" and x.get("git_head") == H2:
                    x["git_head"] = H1
        return rows

    C2.read_rows = read_a2
    _, dirty = C2.git_state()
    res = C2.analyze_bidir_ref(C2.OUT_REF, C2.load_reference(), reanalyze="--reanalyze" in sys.argv, state=(H1, dirty))
    rj, rm = os.path.join(C2.OUT_REF, C2.RESULT_JSON), os.path.join(C2.OUT_REF, C2.RESULT_MD)
    res["amendment_A2"] = a2
    with open(rj + ".tmp", "w") as f:
        json.dump(res, f, indent=1)
    os.replace(rj + ".tmp", rj)
    with open(rm, "a") as f:
        f.write("\n## Amendment A2 (provenance only)\n- run git_heads: H1 %s (9 runs), H2 %s (%s)\n- diff H1..H2: %s\n"
                "- analyzer HEAD (real): %s; diff H1..HEAD: %s\n- analyzer state pinned to H1 per A2 rule 3; amendment sha256 %s\n" % (
                    H1[:7], H2[:7], H2_LOG, ", ".join(a2["diff_H1_H2"]), a2["analyzer_head_real"][:7],
                    ", ".join(a2["diff_H1_HEAD"]) or "(none)", a2["amendment_sha256"][:12]))
    L, arms = res["lif_reference"], res["rate"]["arms"]
    print("%s: LIF reference %s (R %s, %d of %d seeds positive, %d relearn)" % (
        C2.REF_LABEL, L["label"], C2._f(L["R"]), L["n_positive"], L["n_seeds"], L["n_relearn_seeds"]))
    print("rate arm (i) c0 %s; arm (ii) %s (%d of %d)" % (
        arms["arm_i"]["label"], arms["arm_ii"]["label"], arms["arm_ii"]["n_pass"], arms["arm_ii"]["n_seeds"]))


if __name__ == "__main__":
    main()
