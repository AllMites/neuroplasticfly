"""The installed package imports from outside the repo, with no path hacks (issue #1).

Needs `pip install -e .` first. Runs in a fresh interpreter whose cwd is a temp dir, so the
repo root is not on sys.path by accident.
"""
import os
import shutil
import subprocess
import sys
import tempfile

CHECK = r"""
import sys
import neuroplasticfly
from neuroplasticfly import engine, prereg
from neuroplasticfly.learn import plastic
import neuroplasticfly.learn.plastic as plastic2
from neuroplasticfly.rate import engine as rate_engine

# one module object per file: settings on the alias reach the code that reads them
assert engine is sys.modules["gpu_sim"] and engine.__spec__.name == "gpu_sim"
assert plastic is plastic2 is sys.modules["learn.plastic"]
assert rate_engine is sys.modules["rate.engine"]
assert prereg.holds([True] * 4 + [False])
print("ok", engine.__file__)
"""

with tempfile.TemporaryDirectory() as tmp:
    r = subprocess.run([sys.executable, "-c", CHECK], cwd=tmp, capture_output=True, text=True)
    print(r.stdout + r.stderr)
    assert r.returncode == 0, "import from a temp cwd failed (is the package installed: pip install -e .?)"
    for cli in ("neuroplasticfly-prereg", "neuroplasticfly-reproduce-paper1"):
        exe = shutil.which(cli, path=os.path.dirname(sys.executable) + os.pathsep + os.environ.get("PATH", ""))
        assert exe, "entry point %s not installed" % cli
        assert subprocess.run([exe, "--help"], cwd=tmp, capture_output=True).returncode == 0, cli
print("test_import: PASS")
