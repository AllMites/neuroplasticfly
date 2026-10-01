"""Supplementary Tables S2 and S3 for paper 1, read from committed result files (Table S1 is hand-written from git log).
Writes table_S2.md and table_S3.md next to this file."""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("NEUROPLASTICFLY_ROOT") or os.path.normpath(os.path.join(HERE, "..", ".."))
RES = os.path.join(ROOT, "results")

ROWS = [("F1", "Fox drive raises γ4", "Christie et al. (2026)"), ("F2", "Fox drive raises γ5", "Christie et al. (2026)"),
        ("F3", "Fox drive raises β2", "Christie et al. (2026)"), ("F4", "Fox drive raises β′2", "Christie et al. (2026)"),
        ("F5", "Fox drive leaves β1 flat", "Christie et al. (2026); Siju et al. (2020)"),
        ("H1", "sugar raises γ4 and γ5", "Cohn et al. (2015)"), ("H2", "sugar raises β2 and β′2", "Siju et al. (2020)"),
        ("H3", "bitter leaves the reward PAMs silent", "Siju et al. (2020)"),
        ("H4", "silencing Fox lowers the sugar response", "Christie et al. (2026)"),
        ("H5", "Fox drive leaves γ3 flat", "Christie et al. (2025, preprint)"),
        ("H6", "PPL1 γ1 and γ2 respond more to bitter than to sugar", "Siju et al. (2020)")]


def s2():
    cells = [json.load(open(os.path.join(RES, "rate_chunk1_fit_c0", "cells", "cap016.000_s%d.json" % s))) for s in range(5)]
    out = ["# Table S2. Rows of the reward fit at a cap of 16, for each of the 5 starting points",
           "", "F rows were in the fit; H rows were held out. A cell gives pass or fail and the measured value.", "",
           "| Row | Test | Source | " + " | ".join("start %d" % s for s in range(5)) + " |",
           "|---|---|---|" + "---|" * 5]
    for rid, test, src in ROWS:
        vals = []
        for c in cells:
            r = c["rows"][rid]
            vals.append("not testable (sugar response 0 Hz)" if str(r["detail"]).startswith("NOT TESTABLE") else "%s (%s)" % ("pass" if r["pass"] else "fail", r["detail"]))
        out.append("| %s | %s | %s | %s |" % (rid, test, src, " | ".join(vals)))
    open(os.path.join(HERE, "table_S2.md"), "w", encoding="utf-8").write("\n".join(out) + "\n")


def s3():
    c2 = json.load(open(os.path.join(RES, "rate_chunk2", "result.json")))
    br = json.load(open(os.path.join(RES, "rate_chunk2_bidir_ref", "result.json")))
    out = ["# Table S3. Learning tests on the bare rate model and on the 5 versions with the reward gains", "",
           "| Version | Specificity: normalised drop (bar 0.486) | Accumulation: trained / naive (bar 0.429) | "
           "Relearning: first-block drop (bar 0.198) | Relearned cycles (of 3) | Avoid MBONs T0 → B4 (Hz) | PAM at T0 (Hz) |",
           "|---|---|---|---|---|---|---|"]
    vers = [("bare", c2["c0"]["protocols"], "c0")] + [("gains, start %d" % s, c2["c1"]["seeds"][str(s)]["protocols"], "c1s%d" % s)
                                                      for s in range(5)]
    for name, p, base in vers:
        b = br["rate"]["bases"][base]
        av, pam = b["series"]["timed"]["avoid_mbon_hz"], b["series"]["timed"]["pam_hz"]
        out.append("| %s | %.3f | %.3f | %.3f | %d | %.2f → %.2f | %.2f |" % (
            name, p["rung_a"]["numbers"]["norm_approach_drop"], p["rung3"]["numbers"]["ratio"],
            b["numbers"]["rise_1_norm"], b["numbers"]["n_relearn"], av["T0"], av["B4"], pam["T0"]))
    out += ["", "Spiking-model references: specificity 0.971, accumulation 0.858 (ratios of the means of 5 noise "
            "replicates), first-block drop 0.396 (mean of 5 replicates, SD 0.016, all 5 relearned 3 of 3 cycles)."]
    open(os.path.join(HERE, "table_S3.md"), "w", encoding="utf-8").write("\n".join(out) + "\n")


if __name__ == "__main__":
    s2()
    s3()
