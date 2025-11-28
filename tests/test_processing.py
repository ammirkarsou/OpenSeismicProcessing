import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Stub pylops to avoid numba caching issues during tests
if "pylops" not in sys.modules:
    sys.modules["pylops"] = types.SimpleNamespace()

from openseismicprocessing import processing


def test_create_header_with_factorize():
    df = pd.DataFrame({"fldr": [10, 10, 20, 30, 30, 30]})
    context = {"geometry": df.copy()}

    result = processing.create_header(context, "FFID", "factorize(fldr)[0] + 1")

    assert "FFID" in result.columns
    # factorize assigns 0-based unique ids; we add 1
    expected = np.array([1, 1, 2, 3, 3, 3])
    assert np.array_equal(result["FFID"].to_numpy(), expected)


def test_create_header_missing_column_raises():
    df = pd.DataFrame({"a": [1, 2, 3]})
    context = {"geometry": df}
    with pytest.raises(ValueError):
        processing.create_header(context, "b", "nonexistent + 1")


def test_save_header_writes_back_to_geometry(tmp_path):
    geom_path = tmp_path / "geom.parquet"
    df = pd.DataFrame({"fldr": [1, 2], "x": [10.0, 20.0]})
    df.to_parquet(geom_path)
    context = {"geometry": df.copy(), "_geometry_path": str(geom_path)}

    # add a new header then save
    processing.create_header(context, "FFID", "factorize(fldr)[0] + 1")
    written = processing.save_header(context, "FFID")

    assert Path(written) == geom_path
    reloaded = pd.read_parquet(geom_path)
    assert "FFID" in reloaded.columns
    assert reloaded["FFID"].tolist() == [1, 2]


def test_save_header_requires_header(tmp_path):
    geom_path = tmp_path / "geom.parquet"
    df = pd.DataFrame({"fldr": [1, 2]})
    df.to_parquet(geom_path)
    context = {"geometry": df.copy(), "_geometry_path": str(geom_path)}
    with pytest.raises(ValueError):
        processing.save_header(context, "missing")


def test_sort_reorders_geometry_and_data():
    geom = pd.DataFrame({"fldr": [2, 1], "trace_id": [0, 1]})
    data = np.array([[10, 20], [30, 40]])
    context = {"geometry": geom, "data": data}
    sorted_df = processing.sort(context, "fldr")
    assert sorted_df["fldr"].tolist() == [1, 2]
    # columns should be reordered in data
    assert context["data"][:, 0].tolist() == [20, 40]


def test_trim_samples():
    data = np.arange(12).reshape(6, 2)
    context = {"data": data.copy()}
    trimmed = processing.trim_samples(context, 3)
    assert trimmed.shape == (3, 2)
    assert np.array_equal(context["data"], trimmed)


def test_zero_phase_wavelet_returns_real():
    wavelet = np.array([1.0, 0.5, 0.0, -0.5])
    context = {"data": wavelet}
    out = processing.zero_phase_wavelet(context)
    assert out.shape == wavelet.shape
    assert np.isrealobj(out)


def test_generate_local_coordinates():
    model = pd.DataFrame({"SourceX": [0, 10], "SourceY": [0, 0]})
    geom = pd.DataFrame(
        {
            "SourceX": [0, 10],
            "SourceY": [0, 0],
            "GroupX": [0, 10],
            "GroupY": [0, 0],
        }
    )
    context = {"model geometry": model, "geometry": geom.copy()}
    updated = processing.generate_local_coordinates(context)
    assert isinstance(updated, pd.DataFrame)
    for col in ["SourceX", "SourceY", "GroupX", "GroupY"]:
        assert col in updated.columns


def test_scale_coordinate_units():
    df = pd.DataFrame(
        {
            "SourceX": [100.0],
            "SourceY": [200.0],
            "GroupX": [300.0],
            "GroupY": [400.0],
            "SourceDepth": [500.0],
            "ReceiverDatumElevation": [600.0],
        }
    )
    context = {"geometry": df.copy()}
    out = processing.scale_coordinate_units(context, XY_Scaler=100.0, elevation_scaler=100.0)
    assert np.isclose(out["SourceX"].iloc[0], 1.0)
    assert np.isclose(out["GroupY"].iloc[0], 4.0)
    assert np.isclose(out["SourceDepth"].iloc[0], 5.0)


def test_resample_changes_length():
    data = np.arange(8).reshape(4, 2)
    context = {"data": data}
    out = processing.resample(context, dt_in=2.0, dt_out=1.0)
    assert out.shape[0] == 8  # upsampled


def test_kill_traces_outside_box():
    geom = pd.DataFrame({"SourceX": [1, -1], "SourceY": [1, 2], "GroupX": [1, 2], "GroupY": [1, 2]})
    data = np.arange(8).reshape(4, 2)  # samples x traces
    context = {"geometry": geom, "data": data}
    processing.kill_traces_outside_box(context, columns=["SourceX", "SourceY", "GroupX", "GroupY"])
    assert len(context["geometry"]) == 1
    assert context["data"].shape[1] == 1
