"""Utilities for working with SEG-Y data stored in Zarr format."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence, Optional, Tuple, Dict, Union, Any

import numpy as np
import segyio
import segyio.tools
import zarr
from numcodecs import Blosc
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd


def _resolve_segy_inputs(paths: Union[str, Path, Sequence[Union[str, Path]]]) -> list[Path]:
    if isinstance(paths, (str, Path)):
        candidates = [Path(paths)]
    else:
        candidates = [Path(p) for p in paths]

    files: list[Path] = []
    for candidate in candidates:
        if candidate.is_dir():
            files.extend(sorted(candidate.glob("*.sgy")))
            files.extend(sorted(candidate.glob("*.segy")))
        elif candidate.is_file():
            files.append(candidate)
        else:
            raise FileNotFoundError(f"SEG-Y path not found: {candidate}")

    files = sorted({f.resolve() for f in files})
    if not files:
        raise FileNotFoundError(f"No SEGY files resolved from {paths}")
    return files


def preview_segy_headers(
    context: dict,
    segy_paths: Union[str, Path, Sequence[Union[str, Path]]],
    headers: Optional[Sequence[str]] = None,
    n_traces: int = 1000,
    max_files: int = 5,
    figsize: Optional[Tuple[float, float]] = None,
    plot: bool = False,
    output: Optional[str] = None,
) -> pd.DataFrame:
    """Inspect header statistics (table) over the first ``n_traces`` across SEG-Y files."""

    files = _resolve_segy_inputs(segy_paths)

    if headers is None:
        try:
            header_keys = sorted(segyio.tracefield.keys.keys())
        except AttributeError:
            header_keys = []
        headers = list(dict.fromkeys(header_keys + ["offset"]))

    lower_to_original = {h.lower(): h for h in headers}
    collected: Dict[str, list[float]] = {h: [] for h in headers}

    remaining = int(n_traces)
    for file_path in files[:max_files]:
        if remaining <= 0:
            break
        try:
            f = segyio.open(file_path, "r", ignore_geometry=False)
            _ = f.ilines
        except Exception:
            f = segyio.open(file_path, "r", ignore_geometry=True)

        with f:
            ntr = len(f.trace)
            count = min(remaining, ntr)

            if "offset" in lower_to_original:
                try:
                    sx = np.fromiter((f.header[i][segyio.TraceField.SourceX] for i in range(count)), dtype=np.float64)
                    gx = np.fromiter((f.header[i][segyio.TraceField.GroupX] for i in range(count)), dtype=np.float64)
                    collected[lower_to_original["offset"]].extend(np.abs(sx - gx))
                except Exception:
                    pass

            for lower, original in lower_to_original.items():
                if lower == "offset":
                    continue
                try:
                    field = getattr(segyio.TraceField, original)
                except AttributeError:
                    continue
                values = np.fromiter((f.header[i][field] for i in range(count)), dtype=np.float64)
                collected[original].extend(values)

        remaining -= count

    summary: Dict[str, Dict[str, float]] = {}
    for header, values in collected.items():
        if not values:
            continue
        arr = np.asarray(values, dtype=np.float64)
        summary[header] = {
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "unique": int(np.unique(arr).size),
        }

    fig = None
    if plot and summary:
        headers_to_plot = list(summary.keys())
        if figsize is None:
            figsize = (10, max(3.0, 2.0 * len(headers_to_plot)))
        fig, axes = plt.subplots(len(headers_to_plot), 1, sharex=True, figsize=figsize)
        if len(headers_to_plot) == 1:
            axes = [axes]
        x = np.arange(min(len(collected[h]) for h in headers_to_plot))
        for ax, header in zip(axes, headers_to_plot):
            data = np.asarray(collected[header][: len(x)], dtype=np.float64)
            ax.plot(x, data)
            stats = summary[header]
            ax.set_ylabel(header)
            ax.grid(True, linewidth=0.3, alpha=0.5)
            ax.set_title(
                f"min={stats['min']:.3f}, max={stats['max']:.3f}, "
                f"mean={stats['mean']:.3f}, std={stats['std']:.3f}, unique={stats['unique']}"
            )
        axes[-1].set_xlabel("Trace index")
        fig.suptitle(f"Header preview (first {n_traces} traces)")
        fig.tight_layout()

    df = pd.DataFrame.from_dict(summary, orient="index").sort_index()
    context.setdefault("segy_header_preview", df)
    if output and fig is not None:
        context[output] = fig
    elif output:
        context[output] = df

    return df


def segy_directory_to_zarr(
    context: dict,
    segy_input: Union[str, Path, Sequence[Union[str, Path]]],
    zarr_out: str | Path,
    headers: Sequence[str] | None = None,
    chunk_trace: int = 512,
    compressor: Blosc | None = None,
) -> str:
    """Convert all SEG-Y files in ``segy_dir`` into a single Zarr store."""

    if headers is None:
        headers = ("SourceX", "GroupX", "offset")

    files = _resolve_segy_inputs(segy_input)

    lower_to_original = {header.lower(): header for header in headers}

    total_traces = 0
    ns = None
    for file_path in tqdm(files, desc="Counting traces"):
        with segyio.open(file_path, "r", ignore_geometry=True) as f:
            total_traces += len(f.trace)
            if ns is None:
                ns = len(f.samples)

    if ns is None:
        raise RuntimeError("Could not determine number of samples per trace.")

    compressor = compressor or Blosc(cname="zstd", clevel=5, shuffle=Blosc.SHUFFLE)
    zarr_out = Path(zarr_out)
    zarr_out.parent.mkdir(parents=True, exist_ok=True)
    root = zarr.open(zarr_out, mode="w")
    file_metadata: list[Dict[str, Any]] = []
    root.attrs.update(
        {
            "description": "SEG-Y to Zarr flat trace layout",
            "headers": list(headers),
            "samples": int(ns),
            "source_inputs": [str(f) for f in files],
        }
    )

    amp = root.create_dataset(
        "amplitude",
        shape=(total_traces, ns),
        chunks=(chunk_trace, ns),
        dtype="float32",
        compressor=compressor,
    )
    amp.attrs["_ARRAY_DIMENSIONS"] = ["trace", "sample"]

    header_arrays = {}
    for header in headers:
        header_arrays[header] = root.create_dataset(
            header,
            shape=(total_traces,),
            chunks=(chunk_trace,),
            dtype="float64",
            compressor=compressor,
        )
        header_arrays[header].attrs["_ARRAY_DIMENSIONS"] = ["trace"]

    offset = 0
    for file_path in tqdm(files, desc="Streaming SEGY files"):
        try:
            f = segyio.open(file_path, "r", ignore_geometry=False)
            _ = f.ilines  # trigger inline parsing; fails for 2D
        except Exception:
            f = segyio.open(file_path, "r", ignore_geometry=True)

        with f:
            entry: Dict[str, Any] = {"file": str(file_path)}
            try:
                entry["binary_header"] = {
                    str(key): int(value) for key, value in f.bin.items()
                }
            except Exception:
                entry["binary_header"] = {}

            try:
                wrapped_text = segyio.tools.wrap(f.text[0])
                entry["text_header"] = wrapped_text.splitlines()
            except Exception:
                entry["text_header"] = []

            ntr = len(f.trace)
            data = np.empty((ntr, ns), dtype=np.float32)
            for i, tr in enumerate(f.trace):
                data[i, :] = tr

            geom: dict[str, Iterable[float]] = {}
            if "offset" in lower_to_original:
                try:
                    sx = np.fromiter((f.header[i][segyio.TraceField.SourceX] for i in range(ntr)), dtype=np.float64)
                    gx = np.fromiter((f.header[i][segyio.TraceField.GroupX] for i in range(ntr)), dtype=np.float64)
                    geom[lower_to_original["offset"]] = np.abs(sx - gx)
                except Exception:
                    pass

            for lower, original in lower_to_original.items():
                if lower == "offset":
                    continue
                try:
                    field = getattr(segyio.TraceField, original)
                except AttributeError:
                    print(f"⚠️ Unknown header '{original}' — skipped")
                    continue
                geom[original] = np.fromiter(
                    (f.header[i][field] for i in range(ntr)), dtype=np.float64
                )

        idx_end = offset + ntr
        amp[offset:idx_end, :] = data
        for header_name, values in geom.items():
            header_arrays[header_name][offset:idx_end] = values
        offset = idx_end
        file_metadata.append(entry)

    root.attrs["file_metadata"] = file_metadata
    zarr_path = str(Path(zarr_out).resolve())
    context["zarr_store"] = zarr_path
    return zarr_path


def load_zarr_amplitude(
    context: dict,
    zarr_store: str | Path,
    dataset: str = "amplitude",
    output: str = "data",
) -> np.ndarray:
    """Load a dataset from a Zarr store into the pipeline context."""

    store = zarr.open(zarr_store, mode="r")
    if dataset not in store:
        raise KeyError(f"Dataset '{dataset}' not found in Zarr store {zarr_store}")
    array = store[dataset][:]
    context[output] = array
    return array


def preview_zarr_headers(
    context: dict,
    zarr_store: str | Path,
    headers: Optional[Sequence[str]] = None,
    n_traces: int = 1000,
    figsize: Optional[Tuple[float, float]] = None,
    output: Optional[str] = None,
) -> plt.Figure:
    """Plot and summarize the first ``n_traces`` for each header in a Zarr store."""

    store = zarr.open(zarr_store, mode="r")
    if headers is None:
        headers = store.attrs.get("headers", [])

    headers = list(headers)
    if not headers:
        raise ValueError("No headers supplied and none recorded in Zarr store.")

    n_total = store[headers[0]].shape[0]
    n = min(int(n_traces), n_total)

    if figsize is None:
        figsize = (10, max(3.0, 2.0 * len(headers)))

    fig, axes = plt.subplots(len(headers), 1, sharex=True, figsize=figsize)
    if len(headers) == 1:
        axes = [axes]

    x = np.arange(n)
    summary = {}
    for ax, header in zip(axes, headers):
        if header not in store:
            ax.text(0.5, 0.5, f"Missing header '{header}'", ha="center", va="center")
            ax.set_axis_off()
            continue
        data = store[header][:n]
        ax.plot(x, data)
        ax.set_ylabel(header)
        ax.grid(True, linewidth=0.3, alpha=0.5)

        summary[header] = {
            "min": float(np.min(data)),
            "max": float(np.max(data)),
            "mean": float(np.mean(data)),
            "std": float(np.std(data)),
        }

    axes[-1].set_xlabel("Trace index")
    fig.suptitle(f"First {n} traces — {Path(zarr_store).name}", y=0.98)
    fig.tight_layout()

    if output:
        context[output] = fig
        context[f"{output}_stats"] = summary

    context.setdefault("header_preview_stats", summary)

    return fig


def load_zarr_datasets(
    context: dict,
    zarr_store: Union[str, Path, Dict[str, Any]],
    headers: Optional[Sequence[str]] = None,
    include_amplitude: bool = True,
    output: str = "zarr_data",
    eager: bool = False,
) -> Dict[str, np.ndarray]:
    """Load selected datasets from a Zarr store into memory.

    Parameters
    ----------
    context : dict
        Pipeline context that will receive the loaded arrays.
    zarr_store : str, Path, or mapping
        Path to the Zarr store on disk, or a dataset mapping returned by
        :func:`golem.catalog.get_dataset` containing a ``file_path`` key.
    headers : sequence of str, optional
        Names of header datasets to load. If ``None``, attempts to read the
        ``headers`` attribute stored in the Zarr metadata; otherwise falls back
        to all array keys except ``amplitude``.
    include_amplitude : bool, default True
        Whether to load the ``amplitude`` dataset in addition to headers.
    output : str, default "zarr_data"
        Context key where the resulting dictionary of arrays will be stored.

    Returns
    -------
    dict
        Mapping of dataset names to NumPy arrays (fully materialized in memory).
    """

    if isinstance(zarr_store, dict) and "file_path" in zarr_store:
        store_path = zarr_store["file_path"]
    else:
        store_path = zarr_store

    store = zarr.open(store_path, mode="r")

    if headers is None:
        headers_attr = store.attrs.get("headers")
        if headers_attr:
            headers = list(headers_attr)
        else:
            headers = [key for key in store.array_keys() if key != "amplitude"]

    selected = list(dict.fromkeys(headers)) if headers else []
    data: Dict[str, Any] = {}

    def _maybe_materialize(arr):
        return arr[:] if eager else arr

    if include_amplitude and "amplitude" in store:
        data["amplitude"] = _maybe_materialize(store["amplitude"])

    for header in selected:
        if header not in store:
            continue
        data[header] = _maybe_materialize(store[header])

    context[output] = data
    return data


__all__ = [
    "segy_directory_to_zarr",
    "load_zarr_amplitude",
    "load_zarr_datasets",
    "preview_zarr_headers",
    "preview_segy_headers",
    "extract_zarr_text_headers",
    "extract_zarr_binary_headers",
]

def extract_zarr_text_headers(context: dict, zarr_store: Union[str, Path, Dict[str, Any]], output: str = "text_headers") -> list:
    """Extract SEG-Y text headers stored in Zarr metadata."""
    if isinstance(zarr_store, dict) and "file_path" in zarr_store:
        store_path = zarr_store["file_path"]
    else:
        store_path = zarr_store

    store = zarr.open(store_path, mode="r")
    metadata = store.attrs.get("file_metadata", [])
    text_headers = [entry.get("text_header", []) for entry in metadata]
    context[output] = text_headers
    return text_headers


def extract_zarr_binary_headers(context: dict, zarr_store: Union[str, Path, Dict[str, Any]], output: str = "binary_headers") -> list:
    """Extract SEG-Y binary headers stored in Zarr metadata."""
    if isinstance(zarr_store, dict) and "file_path" in zarr_store:
        store_path = zarr_store["file_path"]
    else:
        store_path = zarr_store

    store = zarr.open(store_path, mode="r")
    metadata = store.attrs.get("file_metadata", [])
    binary_headers = [entry.get("binary_header", {}) for entry in metadata]
    context[output] = binary_headers
    return binary_headers
