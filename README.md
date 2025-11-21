# Open Seismic Processing

Seismic processing utilities for reading SEG-Y data, running common preprocessing pipelines, and visualising seismic volumes. The package bundles a CUDA-based migration backend (`libEikonal.so`) so you can install and import `openseismicprocessing` directly from a clone.

## Requirements

- Python 3.8 – 3.11 (Python 3.12+ is not supported)
- Linux, macOS, or Windows (GPU features currently target Linux + CUDA)
- C++/CUDA toolchain **only** if you plan to rebuild the shared library from source

## Quick Start (recommended)

```bash
git clone <repository-url>
cd Open\ Seismic\ Processing
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install --upgrade pip setuptools wheel
pip install .
```

This installs the `openseismicprocessing` package together with its runtime dependencies:

- `numpy`, `pandas`, `scipy`, `matplotlib`
- `opencv-python`, `pylops`, `numba`
- `segyio`, `ipython`

### Verify the install

```bash
python - <<'PY'
import openseismicprocessing
print("openseismicprocessing version:", openseismicprocessing.__version__)
from openseismicprocessing import read_data  # any function exposed via SignalProcessing
print("Import OK")
PY
```

## Editable/developer install

If you intend to make code changes, install in editable mode and use `requirements.txt` (includes CuPy by default):

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -e .
# or, if you want to manage dependencies manually (CPU-only):
pip install -r requirements.txt          # installs GPU deps too (CuPy)
# If CuPy fails to install on your platform, fall back to the CPU-only list:
pip install -r requirements-cpu.txt
```

After installing, you can run scripts inside `examples/` or import the package from the virtual environment.

## Optional GPU extras

Some migration utilities rely on NVIDIA DALI in addition to CuPy. Install the extra only if you have a CUDA-compatible environment:

```bash
pip install '.[gpu]'
# or, from a published wheel:
pip install 'openseismicprocessing[gpu]'
```

Notes:

- `cupy` wheels target specific CUDA versions; check [CuPy installation docs](https://docs.cupy.dev/en/stable/install.html) if your install fails.
- The `nvidia-dali-cuda120` wheel is published for Linux; adjust the extra or install manually if you target another CUDA runtime.

GPU helpers fall back to informative `ImportError`s when the GPU stack is unavailable, so CPU-only functionality remains available.

## Rebuilding the CUDA shared library (optional)

The prebuilt `libEikonal.so` is bundled under `src/openseismicprocessing/lib/`. If you need to rebuild it:

1. Install the CUDA toolkit and ensure `nvcc` is on your `PATH`.
2. From the repository root, run the command recorded in `lib/so_compiling_command.txt`, adjusting paths as needed.
3. Copy the resulting `libEikonal.so` into `src/openseismicprocessing/lib/`.

```bash
# Example (edit CUDA version/flags to match your setup)
cd lib
/usr/local/cuda-12.8/bin/nvcc -use_fast_math -Xcompiler -fPIC -shared \
    -o libEikonal.so eikonal2D.cu -lnppig
cp libEikonal.so ../src/openseismicprocessing/lib/
```

## Building distributable artifacts

```bash
rm -rf dist build
python -m build             # requires 'build' package (pip install build)
python -m twine check dist/*
```

The resulting wheel and sdist will include the shared library because of the `MANIFEST.in` rule and `package_data` configuration.

## Project layout

- `src/openseismicprocessing/` – main package modules
- `src/openseismicprocessing/lib/` – bundled shared library and stub `__init__.py`
- `examples/` – usage examples and scripts
- `lib/` – CUDA source and build notes (not imported by the package)

Open an issue or submit a PR if you encounter installation problems on a supported platform.
