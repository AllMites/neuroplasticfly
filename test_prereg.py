"""prereg.py: label rules, and freeze -> verify on a throwaway git repository."""
import json
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prereg as P

pairs = [("LOOM", "RECEDE"), ("LOOM", "DIM")]

# clean pass: ordering holds on scores, raw totals and input drive in 5/5 seeds
s = [{"LOOM": 10, "RECEDE": 2, "DIM": 4}] * 5
r = P.verdict(s, pairs, totals=[{"LOOM": 30, "RECEDE": 12, "DIM": 18}] * 5,
              mech=[{"LOOM": 15e3, "RECEDE": -48e3, "DIM": 0}] * 5, positive=("LOOM",))
assert r["label"] == "PASS" and r["n_ok"] == 5 and r["n_subst"] == 5, r

# the baseline-subtraction trap: scores ordered, raw totals flat (near-silent cell) -> not substantive
r = P.verdict(s, pairs, totals=[{"LOOM": 2, "RECEDE": 2, "DIM": 3}] * 5)
assert r["PASS"] and not r["SUBST"] and r["label"] == "PASS BY RULE, NOT SUBSTANTIVE", r

# 3 of 5 seeds is not enough under the default k=4
r = P.verdict(s[:3] + [{"LOOM": 0, "RECEDE": 2, "DIM": 4}] * 2, pairs)
assert r["n_ok"] == 3 and r["label"] == "FAIL", r

# positive: an ordering among negative scores fails
r = P.verdict([{"LOOM": -1, "RECEDE": -5, "DIM": -4}] * 5, pairs, positive=("LOOM",))
assert r["label"] == "FAIL", r

# mechanism missing
r = P.verdict(s, pairs, mech=[{"LOOM": 0, "RECEDE": 1, "DIM": 0}] * 5)
assert r["label"] == "PASS, MECHANISM NOT SHOWN", r

# freeze refuses an uncommitted prereg, then freeze -> verify -> tamper -> verify fails
here = os.getcwd()
with tempfile.TemporaryDirectory() as d:
    os.chdir(d)
    g = lambda *a: subprocess.run(["git", *a], check=True, capture_output=True)
    g("init", "-q"); g("config", "user.email", "t@t"); g("config", "user.name", "t")
    open("PREREGISTER_x.md", "w").write("rule: LOOM > RECEDE in >= 4/5 seeds\n")
    open("run.py", "w").write("print(1)\n")
    try:
        P.freeze("PREREGISTER_x.md", ["run.py"], "meta.json")
        raise AssertionError("freeze accepted an uncommitted prereg")
    except SystemExit:
        pass
    g("add", "PREREGISTER_x.md"); g("commit", "-qm", "prereg")
    P.freeze("PREREGISTER_x.md", ["run.py"], "meta.json")
    open("res.json", "w").write(json.dumps({"label": "PASS"}))
    assert P.verify("meta.json", ["res.json"])
    assert P.verify("meta.json")
    open("res.json", "w").write(json.dumps({"label": "PASS!"}))
    assert not P.verify("meta.json"), "tampered result must fail"
    try:
        P.verify("meta.json", ["res.json"])
        raise AssertionError("re-recording a changed result was accepted")
    except SystemExit:
        pass
    os.chdir(here)
print("test_prereg OK")
