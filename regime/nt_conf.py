"""Build data/nt_conf.npz: per-neuron FlyWire v783 top_nt, top_nt_conf, known_nt,
known_nt_source, in neuron_meta.npz order.

Source: the public annotation TSV (Schlegel et al. 2024), sorted by root_id. That is
flypoke's neuron order; verified here by asserting cell_class / cell_type / side /
super_class match neuron_meta.npz row for row (they did, 139,248 / 139,248).
No docker needed.

Run: .venv/Scripts/python.exe regime/nt_conf.py
"""
import csv
import os
import sys
import urllib.request

import numpy as np

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
URL = ("https://raw.githubusercontent.com/flyconnectome/flywire_annotations/v3.1.0/"
       "supplemental_files/Supplemental_file1_neuron_annotations.tsv")
TSV = os.path.join(_HERE, "data", "ext", "Supplemental_file1_neuron_annotations.tsv")
OUT = os.path.join(_HERE, "data", "nt_conf.npz")


def main():
    if not os.path.exists(TSV):
        os.makedirs(os.path.dirname(TSV), exist_ok=True)
        urllib.request.urlretrieve(URL, TSV)
    m = np.load(os.path.join(_HERE, "data", "neuron_meta.npz"), allow_pickle=False)
    rows = list(csv.DictReader(open(TSV, encoding="utf-8"), delimiter="\t"))
    rows.sort(key=lambda r: int(r["root_id"]))
    assert len(rows) == len(m["cell_class"]), (len(rows), len(m["cell_class"]))
    col = lambda k, d="none": np.array([r[k] if r[k] != "" else d for r in rows])
    for k in ("cell_class", "cell_type", "side", "super_class"):
        assert (col(k) == m[k].astype(str)).all(), "column %s does not match" % k
    conf = np.array([float(r["top_nt_conf"]) if r["top_nt_conf"] else np.nan for r in rows],
                    np.float32)
    np.savez(OUT, root_id=np.array([int(r["root_id"]) for r in rows], np.int64),
             top_nt=col("top_nt"), top_nt_conf=conf, known_nt=col("known_nt", ""),
             known_nt_source=col("known_nt_source", ""))
    print("wrote", OUT, len(rows))


if __name__ == "__main__":
    main()
