"""Add the missing `hemibrain_type` column to data/neuron_meta.npz.

The npz's own `columns` array already LISTS hemibrain_type -- the column was
read from the annotation TSV when the meta was built (inside the sim container,
readout_analysis.py) and then dropped on write. Every probe that needs PAM01..
PAM15 or any other hemibrain subtype therefore re-parses the 139k-row TSV at
runtime: regime/sugar_pam_floor1.py:41-49 does exactly that. Flagged as a
pipeline debt in the 2026-09-22 project notes.

Docker is down, so this does NOT rebuild the meta. It reads the existing npz,
maps root_ids_sorted.npy -> the TSV, and rewrites the same file with one extra
array. Row order is untouched: gpu_sim.py asserts neuron_meta rows == brain
neurons and assumes row alignment, so a reordered meta would silently offset
every cell type in the simulator.

FALSIFIER, checked before anything is written: cell_type rebuilt from the TSV
through root_ids_sorted must equal the stored cell_type on all 139,248 rows. If
the alignment assumption is wrong, that assert fires and no file is touched.

Run: .venv/Scripts/python.exe tools/add_hemibrain_type.py
"""
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
META = os.path.join(HERE, "data", "neuron_meta.npz")
ROOTS = os.path.join(HERE, "data", "root_ids_sorted.npy")
ANN = os.path.join(HERE, "data", "ext", "Supplemental_file1_neuron_annotations.tsv")
COL = "hemibrain_type"

csv.field_size_limit(1 << 24)  # the synonyms column is long in a handful of rows

# The meta writer encodes a missing annotation as the STRING "none", never as an
# empty string: measured 2026-09-22, cell_type has 1528 "none" and 0 "", cell_class
# 31730 "none" and 0 "". The alignment check below compares full cell_type against
# the TSV on all 139,248 rows, so it only passes if this encoding is applied to the
# TSV side too -- and the new column has to use the same encoding as its neighbours.
MISSING = "none"


def read_tsv(fields):
    """root_id -> tuple(fields), for the rows we can key. Streams; no pandas."""
    out = {}
    with open(ANN, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            rid = r.get("root_id")
            if not rid:
                continue
            out[int(rid)] = tuple((r.get(f) or "").strip() or MISSING for f in fields)
    return out


def main():
    meta = dict(np.load(META, allow_pickle=False))
    root_ids = np.load(ROOTS)
    n = len(root_ids)
    assert len(meta["cell_type"]) == n, (
        "neuron_meta has %d rows, root_ids_sorted has %d" % (len(meta["cell_type"]), n))
    if COL in meta:
        print("%s already present (%d annotated); nothing to do"
              % (COL, int((meta[COL].astype(str) != MISSING).sum())))
        return 0

    tsv = read_tsv(["cell_type", COL])
    print("tsv rows keyed by root_id: %d" % len(tsv))
    pairs = [tsv.get(int(x), (MISSING, MISSING)) for x in root_ids]
    ct_tsv = np.array([p[0] for p in pairs])
    hb = np.array([p[1] for p in pairs])

    # --- falsifier: the row order assumption, checked against a column we already have
    ct_npz = meta["cell_type"].astype(str)
    bad = np.flatnonzero(ct_tsv != ct_npz)
    if len(bad):
        for i in bad[:5]:
            print("  row %d root_id %d: npz %r != tsv %r"
                  % (i, root_ids[i], ct_npz[i], ct_tsv[i]))
        raise SystemExit("ALIGNMENT FAILED on %d/%d rows; nothing written" % (len(bad), n))
    print("ALIGNMENT OK %d/%d rows (cell_type rebuilt from the tsv matches)" % (n, n))

    named = hb != MISSING
    print("%s: %d annotated (%.1f%%), %d distinct"
          % (COL, int(named.sum()), 100.0 * named.mean(), len(set(hb[named]))))
    assert named.sum() > 20000, (
        "only %d rows carry a hemibrain_type; the column looks wrong" % named.sum())

    # PAM01..PAM15 must sum to the 307 PAM already reported in
    # docs/superpowers/reward-path/floor1_convergence_2026-09-21.md.
    pam = {k: int((hb == k).sum()) for k in ("PAM%02d" % i for i in range(1, 16))}
    total = sum(pam.values())
    print("PAM01..PAM15: %s -> %d" % (" ".join("%s=%d" % kv for kv in pam.items()), total))
    if total != 307:
        print("WARNING: PAM total %d, floor1_convergence_2026-09-21.md reports 307" % total)

    # Fixed-width unicode, not object: the readers all pass allow_pickle=False.
    meta[COL] = hb.astype("<U%d" % max(1, max(len(x) for x in hb)))
    tmp = META + ".tmp.npz"
    np.savez(tmp, **meta)
    os.replace(tmp, META)
    print("wrote %s (+%s, dtype %s)" % (META, COL, meta[COL].dtype))
    return 0


if __name__ == "__main__":
    sys.exit(main())
