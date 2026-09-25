"""Content hashes for the four data/ artefacts, written to / verified against data/CHECKSUMS.json.

Source: the arrays themselves, not the files. An .npz is a zip and its entry headers
carry mtimes, so the same arrays hash differently on every rebuild: data/brain_gpu.npz
is b6308a8f... today and 3b477936... in commit 5b11326 with byte-identical contents.

The hash walks keys in sorted order and feeds, per key: the key name, the dtype kind
and shape, then the values. Text arrays (kind U/S) are fed as NUL-joined utf-8 so a
<U60 vs <U63 width difference does not change the digest while a changed string does.

Run: python scripts/checksum_data.py --verify      (exit 1 and name the key on mismatch)
     python scripts/checksum_data.py --write       (publish current data/ as the reference)
"""
import hashlib
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(HERE, "data")
OUT = os.path.join(DATA, "CHECKSUMS.json")
FILES = ["brain_gpu.npz", "neuron_meta.npz", "nt_conf.npz", "root_ids_sorted.npy"]

SOURCES = {
    "connections": "Zenodo 10676866 proofread_connections_783.feather (md5 f48f972d262323a102aed49af1396b8a)",
    "annotations": "flyconnectome/flywire_annotations supplemental_files/Supplemental_file1_neuron_annotations.tsv",
    "loader": "flypoke.data.build_network(min_syn=5) @ 24814fe83224ca2d25c116c1107440b877b07762",
}


def _feed(h, arr):
    arr = np.ascontiguousarray(arr)
    h.update(("|%s|%s|" % (arr.dtype.kind, arr.shape)).encode())
    if arr.dtype.kind in "US":
        h.update(b"\0".join(str(x).encode("utf-8") for x in arr.ravel()))
    else:
        h.update(arr.tobytes())


def content_hash(path):
    h = hashlib.sha256()
    if path.endswith(".npy"):
        _feed(h, np.load(path, allow_pickle=False))
    else:
        with np.load(path, allow_pickle=False) as z:
            for k in sorted(z.files):
                h.update(k.encode())
                _feed(h, z[k])
    return h.hexdigest()


def current():
    out = {}
    for name in FILES:
        path = os.path.join(DATA, name)
        if not os.path.exists(path):
            raise SystemExit("missing %s; run scripts/build_data.py first" % path)
        out[name] = {"sha256_content": content_hash(path)}
    with np.load(os.path.join(DATA, "brain_gpu.npz"), allow_pickle=False) as z:
        out["brain_gpu.npz"]["n_edges"] = int(z["n_edges"])
        out["brain_gpu.npz"]["n_neurons"] = int(z["n_neurons"])
        out["brain_gpu.npz"]["n_selectors"] = int(len(z["selector_names"]))
        out["brain_gpu.npz"]["n_groups"] = int(len(z["pool_names"]))
    return out


def main(argv):
    if "--write" in argv:
        blob = {"built": time.strftime("%Y-%m-%d"), "sources": SOURCES, "files": current()}
        json.dump(blob, open(OUT, "w"), indent=1, sort_keys=True)
        print("wrote %s" % OUT)
        for k, v in sorted(blob["files"].items()):
            print("  %-22s %s" % (k, v["sha256_content"]))
        return 0

    if not os.path.exists(OUT):
        raise SystemExit("no %s; run --write once on a known-good data/" % OUT)
    want = json.load(open(OUT))["files"]
    have = current()
    bad = [k for k in FILES if have[k]["sha256_content"] != want.get(k, {}).get("sha256_content")]
    for k in FILES:
        print("  %-22s %s" % (k, "OK" if k not in bad else "MISMATCH"))
        if k in bad:
            print("      want %s" % want.get(k, {}).get("sha256_content"))
            print("      have %s" % have[k]["sha256_content"])
    if bad:
        print("VERIFY FAILED %d/%d: %s" % (len(bad), len(FILES), ", ".join(bad)))
        return 1
    print("OK %d/%d" % (len(FILES), len(FILES)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
