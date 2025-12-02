from pathlib import Path

from setuptools import find_packages, setup


BASE_DIR = Path(__file__).parent.resolve()
README_PATH = BASE_DIR / "README.md"

if README_PATH.exists():
    long_description = README_PATH.read_text(encoding="utf-8")
else:
    long_description = "Seismic processing utilities for data I/O, processing, and visualization workflows."


setup(
    name="openseismicprocessing",
    version="0.1.0",
    description="Seismic Processing Toolkit",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Ammir Ayman Karsou",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    package_data={"openseismicprocessing": ["lib/*.so"]},
    python_requires=">=3.8,<3.12",
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "platformdirs",
        "scipy",
        "opencv-python",
        "pylops",
        "numba",
        "segyio",
        "ipython",
        "zarr<3",
        "numcodecs",
        "tqdm",
        "PyQt6",
        "vtk",
        "pyarrow",
        "fastparquet",
    ],
    extras_require={
        "gpu": [
            "cupy>=12.0",
            "nvidia-dali-cuda120; platform_system == 'Linux'",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering",
    ],
    keywords=["seismic", "processing", "geophysics"],
)
