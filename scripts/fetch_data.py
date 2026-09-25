"""Fetch the two FlyWire v783 inputs into data/ext/. Never vendored: Zenodo states
CC-BY-4.0 for record 10676866, flywire.ai states CC BY-NC. You fetch them, you accept
whichever terms apply to you.

  * Supplemental_file1_neuron_annotations.tsv  flyconnectome/flywire_annotations, GitHub
    release v3.1.0 (pinned; the commit SHA is recorded in
    data/ext/annotations_provenance.json)
  * proofread_connections_783.feather          Zenodo 10676866, 852,022,274 bytes,
    md5 f48f972d262323a102aed49af1396b8a

Resumable: writes <name>.part, sends Range: bytes=<have>- on retry, renames only after
the md5 matches. A server that ignores Range restarts the file; the md5 still gates.

Run: python scripts/fetch_data.py [--force]
"""
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXT = os.path.join(HERE, "data", "ext")

TSV_URL = ("https://raw.githubusercontent.com/flyconnectome/flywire_annotations/v3.1.0/"
           "supplemental_files/Supplemental_file1_neuron_annotations.tsv")
TSV = os.path.join(EXT, "Supplemental_file1_neuron_annotations.tsv")
COMMITS_URL = "https://api.github.com/repos/flyconnectome/flywire_annotations/commits/v3.1.0"

CON_URL = "https://zenodo.org/api/records/10676866/files/proofread_connections_783.feather/content"
CON = os.path.join(EXT, "proofread_connections_783.feather")
CON_MD5 = "f48f972d262323a102aed49af1396b8a"
CON_SIZE = 852022274

CHUNK = 1 << 20
RETRIES = 5
PROV = os.path.join(EXT, "annotations_provenance.json")


def _md5(path):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(CHUNK)
            if not b:
                return h.hexdigest()
            h.update(b)


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(CHUNK)
            if not b:
                return h.hexdigest()
            h.update(b)


def download(url, dest, want_md5=None, want_size=None):
    """Resumable GET. Returns when dest exists and (if given) its md5 matches."""
    part = dest + ".part"
    for attempt in range(1, RETRIES + 1):
        have = os.path.getsize(part) if os.path.exists(part) else 0
        if want_size and have > want_size:            # a bad partial; start over
            os.remove(part)
            have = 0
        if want_size and have == want_size:
            break
        req = urllib.request.Request(url)
        if have:
            req.add_header("Range", "bytes=%d-" % have)
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                resumed = resp.status == 206
                if have and not resumed:              # server ignored Range
                    print("  server ignored Range (status %s); restarting" % resp.status)
                    have = 0
                print("  %s at %d bytes (%s)"
                      % ("resuming" if resumed else "starting", have,
                         resp.headers.get("Content-Length") or "unknown length"))
                t0, last = time.time(), have
                with open(part, "ab" if have else "wb") as fh:
                    while True:
                        b = resp.read(CHUNK)
                        if not b:
                            break
                        fh.write(b)
                        have += len(b)
                        if have - last > 64 * CHUNK:
                            last = have
                            print("    %d MB  %.1f MB/s"
                                  % (have >> 20, (have >> 20) / max(time.time() - t0, 1e-9)))
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            print("  attempt %d/%d failed: %s" % (attempt, RETRIES, exc))
            time.sleep(min(2 ** attempt, 30))
            continue
        if want_size and os.path.getsize(part) != want_size:
            print("  short read: %d of %d bytes" % (os.path.getsize(part), want_size))
            continue
        break
    else:
        raise SystemExit("could not download %s after %d attempts.\n"
                         "  Fetch it by hand: %s\n  md5 %s -> %s"
                         % (os.path.basename(dest), RETRIES, url, want_md5 or "-", dest))

    if want_md5:
        got = _md5(part)
        if got != want_md5:
            os.remove(part)
            raise SystemExit("md5 mismatch for %s: got %s want %s (partial removed, rerun)"
                             % (os.path.basename(dest), got, want_md5))
    os.replace(part, dest)
    print("  wrote %s (%d bytes)" % (dest, os.path.getsize(dest)))


def annotations_provenance():
    """Record which commit of the annotation repo (pinned tag) we took the TSV from."""
    sha = "unknown"
    try:
        with urllib.request.urlopen(COMMITS_URL, timeout=30) as resp:
            sha = json.load(resp)["sha"]
    except Exception as exc:                          # offline is not fatal; the hash is
        print("  could not read the flywire_annotations v3.1.0 sha: %s" % exc)
    blob = {"url": TSV_URL, "flywire_annotations_commit": sha,
            "fetched": time.strftime("%Y-%m-%d"), "sha256": _sha256(TSV),
            "bytes": os.path.getsize(TSV)}
    json.dump(blob, open(PROV, "w"), indent=1, sort_keys=True)
    print("  provenance -> %s (commit %s)" % (PROV, sha[:12]))


def main(argv):
    force = "--force" in argv
    os.makedirs(EXT, exist_ok=True)

    print("annotations TSV:")
    if force or not os.path.exists(TSV):
        download(TSV_URL, TSV)
        annotations_provenance()
    else:
        print("  present, %d bytes" % os.path.getsize(TSV))
        if not os.path.exists(PROV):
            annotations_provenance()

    print("connections feather (852 MB):")
    if force or not os.path.exists(CON):
        download(CON_URL, CON, want_md5=CON_MD5, want_size=CON_SIZE)
    elif os.path.getsize(CON) != CON_SIZE:
        print("  wrong size (%d != %d); refetching" % (os.path.getsize(CON), CON_SIZE))
        os.remove(CON)
        download(CON_URL, CON, want_md5=CON_MD5, want_size=CON_SIZE)
    else:
        print("  present, %d bytes" % os.path.getsize(CON))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
