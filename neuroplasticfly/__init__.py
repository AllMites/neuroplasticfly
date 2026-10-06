"""Import package for neuroplasticfly (`pip install -e .` from the repo root).

The code stays at its original paths (gpu_sim.py, learn/, rate/, prereg.py) because
PROVENANCE.json and the run meta rows hash those files by path and content. This package
gives them one importable home without moving a byte:

    from neuroplasticfly import engine        # gpu_sim.py, the spiking engine
    from neuroplasticfly.learn import plastic  # learn/plastic.py, the plasticity rule
    from neuroplasticfly.rate import engine as rate_engine  # rate/engine.py, the rate model
    from neuroplasticfly import prereg         # prereg.py, preregister/label/verify

Each name is the same module object as its root-level original (neuroplasticfly.engine is
gpu_sim), so settings such as engine.PN_KC_GAIN = 8 reach the code that reads them.
Data paths resolve next to the source files, so install editable from a clone.
"""
import importlib
import importlib.abc
import importlib.util
import sys

__version__ = "0.2.0"

_ALIAS = {"engine": "gpu_sim", "learn": "learn", "rate": "rate", "prereg": "prereg"}


class _AliasFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """neuroplasticfly.<alias>[.sub] -> the original module object, never a second copy."""

    def find_spec(self, name, path=None, target=None):
        head, _, rest = name.partition(".")
        if head == __name__ and rest.split(".")[0] in _ALIAS:
            return importlib.util.spec_from_loader(name, self)
        return None

    def create_module(self, spec):
        first, _, tail = spec.name.split(".", 1)[1].partition(".")
        module = importlib.import_module(_ALIAS[first] + ("." + tail if tail else ""))
        spec.loader_state = module.__spec__
        return module

    def exec_module(self, module):
        module.__spec__ = module.__spec__.loader_state  # import machinery overwrote it; keep the original's


sys.meta_path.insert(0, _AliasFinder())


def __getattr__(name):
    if name in _ALIAS:
        return importlib.import_module(__name__ + "." + name)
    raise AttributeError("module %r has no attribute %r" % (__name__, name))


def _reproduce():
    """Console entry point: reproduce.main takes argv explicitly."""
    import reproduce
    sys.exit(reproduce.main(sys.argv[1:]))
