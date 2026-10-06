# Install

This page covers what you need, how to install the package, how to build the brain in `data/`, and how to check
that everything works. The [tutorial](tutorial.md) picks up from there, and the [API reference](api.md) lists the
functions it uses.

## Requirements

- **Windows or Linux.** The published runs were made on Windows and CI runs on Linux; macOS has no CUDA and is
  not supported.
- **An NVIDIA GPU with CUDA.** The simulation keeps all 2.7 million synapses in one flat tensor on the GPU and
  steps it every 0.1 ms. There is no supported CPU path for the real brain; only the small synthetic test below
  runs on the CPU. The published runs used an RTX 5070 Ti with 16 GB, on which two simulations at once were the
  tested ceiling.
- **Python 3.11 or newer.** The published runs used Python 3.14.4, torch 2.11.0+cu128, numpy 2.5.3 and scipy
  1.18.1.
- **[uv](https://docs.astral.sh/uv/)** (recommended) or pip, and **git**, because one dependency (`flypoke`, used
  only for its connectome loader) installs from a pinned GitHub commit.
- **About 1.3 GB of disk** for `data/`: 884 MB of downloaded inputs and about 430 MB of built arrays.

## Install the package

With uv, from a clone:

```
git clone https://github.com/AllMites/neuroplasticfly
cd neuroplasticfly
uv sync
```

`uv sync` creates `.venv/`, installs the CUDA 12.8 build of torch from the PyTorch index named in
`pyproject.toml`, and installs this repository in editable mode. Run later commands with `uv run python ...`, or
activate the environment first (`.venv\Scripts\activate` on Windows, `source .venv/bin/activate` on Linux).

With pip, install the CUDA build of torch first and then the package in editable mode:

```
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -e .
```

The install has to be editable. The code stays at the paths that `PROVENANCE.json` hashes (`gpu_sim.py`,
`learn/`, `rate/`, `prereg.py`), and the `neuroplasticfly` package only gives those files one importable name, so
`neuroplasticfly.engine` is `gpu_sim.py` itself rather than a copy. Data paths are resolved next to the source
files, which is why the package has to point at a clone.

On Windows, `triton-windows` is installed as the backend for `torch.compile`. Without it the engine prints
`torch.compile unavailable (...); running eager` and runs uncompiled, which gives the same model at a slower speed.

## Build the brain (`data/`)

The connectome is fetched from its publishers and built locally; it is never stored in this repository.

```
python scripts/fetch_data.py      # downloads into data/ext/
python scripts/build_data.py      # builds data/ from data/ext/
```

`fetch_data.py` downloads two files into `data/ext/`:

- `proofread_connections_783.feather`, the FlyWire v783 synapse table (Dorkenwald et al. 2024, Zenodo record
  10676866), 852,022,274 bytes. The download resumes after an interruption, and the file is renamed into place
  only after its md5 matches `f48f972d262323a102aed49af1396b8a`.
- `Supplemental_file1_neuron_annotations.tsv`, the cell annotations (Schlegel et al. 2024), from the
  `flywire_annotations` release pinned at v3.1.0. Its commit and sha256 are written to
  `data/ext/annotations_provenance.json`.

A file that is already present is skipped, so the script is safe to rerun; `--force` downloads both again.

`build_data.py` builds the network with `flypoke`'s loader at a 5-synapse floor and writes four files:
`brain_gpu.npz` (the weight matrix and the named neuron groups), `neuron_meta.npz` (cell type, class and side of
every neuron), `nt_conf.npz` (transmitter confidence) and `root_ids_sorted.npy` (the FlyWire id of every row).
Before anything is written under its final name, the script checks that there are 139,248 neurons and 2,700,429
connections, and at the end it compares the content hashes of the four files with `data/CHECKSUMS.json`; a
mismatch exits with status 1. Its options are `--no-cache`, which ignores the cached aggregation in
`data/ext/cache/` and starts again from the feather file, and `--update-checksums`, which is only for a deliberate
change to the inputs or the loader.

Measured on a cold clone (README, 2026-09-23), the download took 7 minutes and the build 8 seconds.

The data licence is unresolved: the Zenodo record states CC-BY-4.0 and flywire.ai states CC BY-NC. Treat the
stricter one as binding for your use.

## Check the install

Run these from the repository root. The first three need no GPU and no `data/`.

```
python test_small_graph.py
```

This runs both engines, the spiking one and the rate one, on a synthetic 25-neuron circuit on the CPU (about 15 s).
It checks wiring rather than biology: silence at rest, propagation only along edges, the sign of inhibition,
reproducibility, carried state and silencing. Expected output:

```
ok gpu_sim (cpu, 25 wired of 139248 neurons): rest, propagation, inhibition sign, determinism, batch invariance, carry, silence
ok rate engine (cpu): rest, analytic fixed point, determinism, carry
```

```
python test_import.py              # the package imports from outside the clone; the CLIs are installed
python reproduce_paper1.py --verify   # every shipped file still hashes to what the paper's runs recorded
```

The second ends with `ALL PASS (0 failures)`. After building `data/`, check the build without rebuilding it:

```
python scripts/checksum_data.py --verify
```

```
  brain_gpu.npz          OK
  neuron_meta.npz        OK
  nt_conf.npz            OK
  root_ids_sorted.npy    OK
OK 4/4
```

Finally, on a CUDA machine, the end-to-end check:

```
python reproduce.py
```

This fetches and builds `data/` if needed (each step is skipped when its output already exists), gates the odour
channels, and runs one conditioning seed of the paper-0 specificity test. Once `data/` exists it needs about
11 minutes on the GPU above, and it ends with a line such as

```
rung A seed 0: DC2 +10.90 Hz (target +11.05 +- 0.6)  PASS  wall clock 11 min
```

`reproduce.py` accepts only `--skip-download`. It has no `--help`: any other argument is ignored and the pipeline
starts, so do not call it to read its usage; the docstring at the top of the file has it.

The CI workflow (`.github/workflows/tests.yml`) runs the CPU checks on every push. The tests that need `data/` or a
GPU (most of `regime/`, `rate/` and `learn/`) are run by hand on a CUDA machine.

## Sharing one GPU

`tools/gpu_slot.py` runs a command once one of N GPU slots is free (`GPU_SLOTS`, default 2), so that parallel jobs
on one card wait for each other instead of running out of memory:

```
python tools/gpu_slot.py -- python reproduce.py
```

It is optional; every command it wraps also runs without it.
