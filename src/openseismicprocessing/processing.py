from scipy.signal import resample_poly
import numpy as np
import pandas as pd
import os
import shutil
import inspect
import re
from typing import Sequence
import pylops
from pathlib import Path
from datetime import datetime
from uuid import uuid4
import json
import zarr
from numcodecs import Blosc
from scipy.signal import butter, sosfiltfilt, fftconvolve, chirp

try:
    import cupy as cp
except ImportError:  # pragma: no cover - optional dependency
    cp = None


class Backend:
    def __init__(self, name: str = "numpy"):
        name = str(name).lower()
        self.name = name
        if name == "numpy":
            import numpy as xp
            from scipy import signal as sig
            self.xp = xp
            self.signal = sig
            self.to_numpy = lambda a: a
        elif name == "cupy":
            try:
                import cupy as xp
                from cupyx.scipy import signal as sig
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise RuntimeError("backend='cupy' requested but CuPy/cupyx.scipy is not installed") from exc
            self.xp = xp
            self.signal = sig
            self.to_numpy = xp.asnumpy
        else:
            raise ValueError(f"Unknown backend '{name}'")

    def sosfiltfilt(self, sos, x, axis=-1):
        sig = self.signal
        xp = self.xp
        if hasattr(sig, "sosfiltfilt"):
            return sig.sosfiltfilt(sos, x, axis=axis)
        if hasattr(sig, "sosfilt"):
            y = sig.sosfilt(sos, x, axis=axis)
            y = sig.sosfilt(sos, xp.flip(y, axis=axis), axis=axis)
            return xp.flip(y, axis=axis)
        raise RuntimeError("Backend does not provide sosfiltfilt or sosfilt")


def get_backend(backend):
    if isinstance(backend, Backend):
        return backend
    return Backend(backend)


def _reshape_kernel_for_axis_xp(w, ndim: int, axis: int, xp):
    shape = [1] * ndim
    shape[axis] = w.size
    return w.reshape(shape)

def load_data(context: dict, key_data: str = "data", key_geometry: str = "geometry") -> tuple:
    """
    Load the currently selected dataset (via context['_manifest_path']) into context.

    Geometry is loaded as a DataFrame; data is kept as a Zarr array view to avoid
    loading everything into memory.
    """
    manifest_path = context.get("_manifest_path")
    if manifest_path is None or not Path(manifest_path).exists():
        raise ValueError("load_data: context['_manifest_path'] must point to a valid manifest.")
    meta = json.loads(Path(manifest_path).read_text())
    geom_path = meta.get("geometry_parquet")
    zarr_path = meta.get("zarr_store")
    if not geom_path or not zarr_path:
        raise ValueError("load_data: manifest missing geometry_parquet or zarr_store.")
    context[key_geometry] = pd.read_parquet(geom_path)
    # Keep a Zarr array view to avoid loading the entire volume into RAM
    context[key_data] = zarr.open(zarr_path, mode="r")["amplitude"]
    context["_geometry_path"] = geom_path
    context["geometry_parquet"] = geom_path
    context["zarr_store"] = zarr_path
    return context[key_data], context[key_geometry]

def kill_traces_outside_box(context, key_geometry='geometry', key_data='data', columns=['SourceX','SourceY','GroupX','GroupY']):
    """
    Eliminates rows in the geometry DataFrame and corresponding columns in the seismogram array 
    where any of the specified columns have negative values.
    
    Parameters:
        context (dict): Dictionary containing 'geometry' (a DataFrame) and 'data' (a 2D NumPy array).
        key_geometry (str): Key in context for the geometry DataFrame.
        key_data (str): Key in context for the seismogram array.
        columns (list): List of column names to check for nonnegative values.
    
    Returns:
        None. The function updates the context in-place.
    """
    import numpy as np
    
    geometry_df = context.get(key_geometry)
    data = context.get(key_data)
    
    if geometry_df is None or data is None:
        print("❌ Error: Missing geometry DataFrame or data array in context.")
        return
    
    # Create a boolean mask for rows (traces) where all specified columns are >= 0
    mask = np.ones(len(geometry_df), dtype=bool)
    for col in columns:
        if col not in geometry_df.columns:
            print(f"❌ Warning: Column '{col}' not found in geometry DataFrame. Skipping this column.")
            continue
        mask &= (geometry_df[col] >= 0)
    
    # Filter the geometry DataFrame using the mask and reset the index
    filtered_geometry_df = geometry_df[mask].reset_index(drop=True)
    
    # Filter the seismogram array along the second axis (columns)
    # Assuming data shape is (num_samples, num_traces)
    try:
        filtered_seismogram = data[:, mask]
    except Exception as e:
        print(f"❌ Error filtering seismogram array: {e}")
        return
    
    # Update the context with the filtered results
    context[key_geometry] = filtered_geometry_df
    context[key_data] = filtered_seismogram

def calculate_azimuth(corner_x,corner_y):
    
    x1=corner_x[0]
    x2=corner_x[2]
    y1=corner_y[0]
    y2=corner_y[2]

    return np.arctan2(y2 - y1, x2 - x1) 

def get_local_coordinates(x,y,x0,y0,theta):
    
    x=x-x0
    y=y-y0
    
    x_local = x * np.cos(theta) + y * np.sin(theta)
    y_local =-x * np.sin(theta) + y * np.cos(theta)
    
    
    return x_local,y_local

def generate_local_coordinates(context, key_model_geometry='model geometry', key_geometry='geometry', isVelocityModel=False):
    """
    Converts acquisition geometry coordinates to local coordinates using the model geometry as reference.
    The reference is determined using the 'SourceX' and 'SourceY' columns in the model DataFrame.
    For velocity models, only source coordinates are converted; for other cases, both source and group
    coordinates are processed.

    Parameters:
        context (dict): Dictionary containing the geometry DataFrames.
        key_model_geometry (str): Key for the model geometry DataFrame in context (default 'model geometry').
        key_geometry (str): Key for the acquisition geometry DataFrame in context (default 'geometry').
        isVelocityModel (bool): If True, only source coordinates are processed (default False).

    Returns:
        pd.DataFrame: The updated acquisition geometry DataFrame with local coordinates,
                      or None if an error occurs.
    """
    # Validate context
    if context is None or not isinstance(context, dict):
        print("❌ Error: Invalid context provided.")
        return None

    # Retrieve DataFrames
    model_df = context.get(key_model_geometry)
    geom_df = context.get(key_geometry)

    if model_df is None or not isinstance(model_df, pd.DataFrame):
        print(f"❌ Error: Model geometry DataFrame not found or invalid under key '{key_model_geometry}'.")
        return None

    if geom_df is None or not isinstance(geom_df, pd.DataFrame):
        print(f"❌ Error: Acquisition geometry DataFrame not found or invalid under key '{key_geometry}'.")
        return None

    # Verify required columns in model geometry
    for col in ['SourceX', 'SourceY']:
        if col not in model_df.columns:
            print(f"❌ Error: Column '{col}' not found in model geometry DataFrame.")
            return None

    # Verify required columns in acquisition geometry
    required_geom = ['SourceX', 'SourceY']
    if not isVelocityModel:
        required_geom += ['GroupX', 'GroupY']
    for col in required_geom:
        if col not in geom_df.columns:
            print(f"❌ Error: Column '{col}' not found in acquisition geometry DataFrame.")
            return None

    try:
        # Compute corner coordinates from the model geometry
        corner_x = [np.min(model_df['SourceX']), np.max(model_df['SourceX']),
                    np.min(model_df['SourceX']), np.max(model_df['SourceX'])]
        corner_y = [np.min(model_df['SourceY']), np.min(model_df['SourceY']),
                    np.max(model_df['SourceY']), np.max(model_df['SourceY'])]
    except Exception as e:
        print(f"❌ Error computing model corner coordinates: {e}")
        return None

    try:
        # Calculate the azimuth angle using the model corners
        angle = calculate_azimuth(corner_x, corner_y)
    except Exception as e:
        print(f"❌ Error calculating azimuth: {e}")
        return None

    try:
        # Extract source coordinates from acquisition geometry
        sx = geom_df['SourceX'].to_numpy()
        sy = geom_df['SourceY'].to_numpy()
    except Exception as e:
        print(f"❌ Error extracting source coordinates: {e}")
        return None

    gx = gy = None
    if not isVelocityModel:
        try:
            gx = geom_df['GroupX'].to_numpy()
            gy = geom_df['GroupY'].to_numpy()
        except Exception as e:
            print(f"❌ Error extracting group coordinates: {e}")
            return None

    try:
        # Convert source coordinates to local coordinates
        sx_local, sy_local = get_local_coordinates(sx, sy, corner_x[0], corner_y[0], angle)
    except Exception as e:
        print(f"❌ Error converting source coordinates to local coordinates: {e}")
        return None

    if not isVelocityModel:
        try:
            # Convert group coordinates to local coordinates
            gx_local, gy_local = get_local_coordinates(gx, gy, corner_x[0], corner_y[0], angle)
        except Exception as e:
            print(f"❌ Error converting group coordinates to local coordinates: {e}")
            return None

    # Update the acquisition geometry DataFrame with local coordinates
    geom_df['SourceX'] = sx_local
    geom_df['SourceY'] = sy_local
    if not isVelocityModel:
        geom_df['GroupX'] = gx_local
        geom_df['GroupY'] = gy_local

    # Update context and return the updated geometry
    context[key_geometry] = geom_df
    return geom_df

def create_header(context, header_name, expression, key="geometry"):
    df = context.get(key)

    if df is None:
        raise ValueError("'geometry' not found in context.")

    if not isinstance(df, pd.DataFrame):
        raise ValueError("'geometry' in context is not a valid DataFrame.")

    # Extract variable names from the expression (basic regex for word-like tokens)
    tokens = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', expression)
    # Allowed helper names (will be injected into eval namespace)
    allowed = {"abs", "log", "sqrt", "sin", "cos", "tan", "exp", "min", "max", "mean", "std", "sum", "np", "unique", "pd", "factorize", "True", "False", "return_inverse"}
    columns_used = [t for t in tokens if t not in allowed]

    # Check if all used columns exist
    missing = [col for col in columns_used if col not in df.columns]
    if missing:
        raise ValueError(f"Column(s) not found in geometry: {missing}")

    env = {c: df[c] for c in df.columns}
    env.update(
        {
            "np": np,
            "unique": np.unique,
            "pd": pd,
            "factorize": pd.factorize,
            "True": True,
            "False": False,
            "None": None,
        }
    )
    try:
        result = eval(expression, {"__builtins__": {}}, env)
        if hasattr(result, "__len__") and len(result) == len(df):
            df[header_name] = result
        else:
            df[header_name] = result
        return df
    except Exception:
        try:
            df[header_name] = df.eval(
                expression,
                local_dict=env,
                engine="python",
            )
            return df
        except Exception as e:
            raise ValueError(f"Failed to create header '{header_name}': {e}") from e


def save_header(context, header_name: str | None = None, key: str = "geometry") -> Path:
    """
    Persist the current geometry DataFrame (including any new headers) back to its source Parquet/CSV.

    Parameters
    ----------
    context : dict
        Pipeline context containing the geometry DataFrame under ``key``.
    header_name : str, optional
        Optional header name to ensure it exists before saving.
    key : str, optional
        Context key for the geometry DataFrame (default "geometry").

    Returns
    -------
    Path
        The path written.
    """
    df = context.get(key)
    if df is None or not isinstance(df, pd.DataFrame):
        raise ValueError("'geometry' not found or invalid in context.")
    if header_name and header_name not in df.columns:
        raise ValueError(f"Header '{header_name}' not found in geometry.")
    src_path = context.get("_geometry_path")
    if not src_path:
        raise ValueError("No geometry path available in context to save.")
    out_path = Path(src_path)
    try:
        df.to_parquet(out_path)
    except Exception:
        # fall back to CSV if parquet fails
        out_path = out_path.with_suffix(".csv")
        df.to_csv(out_path, index=False)
    return out_path
def subset_geometry_by_condition(context, condition, key_input="data", key_output="data", key_geometry_input="geometry", key_geometry_output="geometry"):
    df = context.get(key_geometry_input)
    data = context.get(key_input)

    if df is None or not isinstance(df, pd.DataFrame):
        print("❌ Error: 'geometry' not found or invalid.")
        return None

    if data is None or not hasattr(data, "__getitem__"):
        print("❌ Error: 'data' not found or not indexable.")
        return None

    try:
        # Filter the geometry using the condition string
        filtered_df = df.query(condition)
        if filtered_df.empty:
            print(f"⚠️ Warning: No rows match the condition: {condition}")
            return None

        # Subset the data using the filtered indices
        filtered_data = data[:, filtered_df.index.to_numpy()]

        # Update both geometry and data in context
        context[key_geometry_output] = filtered_df.reset_index(drop=True)
        context[key_output] = filtered_data

        return filtered_df
    except Exception as e:
        print(f"❌ Failed to apply condition '{condition}': {e}")
        return None
def stack_data_along_axis(context, axis, method="sum"):
    data = context.get("data")

    if data is None or not isinstance(data, np.ndarray):
        print("❌ Error: 'data' not found or is not a NumPy array.")
        return None

    if data.ndim != 2:
        print(f"❌ Error: 'data' must be 2D. Got shape: {data.shape}")
        return None

    if axis not in [0, 1]:
        print(f"❌ Error: 'axis' must be 0 (stack over time) or 1 (stack over traces). Got: {axis}")
        return None

    if method not in ["sum", "mean"]:
        print(f"❌ Error: method must be either 'sum' or 'mean'. Got: {method}")
        return None

    try:
        if method == "sum":
            stacked = np.sum(data, axis=axis)
        elif method == "mean":
            stacked = np.mean(data, axis=axis)

        context["data"] = stacked
        return stacked
    except Exception as e:
        print(f"❌ Failed to stack data along axis {axis} using method '{method}': {e}")
        return None
def mute_data(context, start_sample):
    data = context.get("data")

    if data is None or not isinstance(data, np.ndarray):
        print("❌ Error: 'data' not found or not a NumPy array.")
        return None

    if not isinstance(start_sample, int) or start_sample < 0:
        print(f"❌ Error: 'start_sample' must be a non-negative integer. Got: {start_sample}")
        return None

    try:
        muted = data.copy()

        if data.ndim == 2:
            if start_sample >= data.shape[0]:
                print(f"⚠️ Warning: 'start_sample' ({start_sample}) exceeds number of samples ({data.shape[0]}). No muting applied.")
                return data
            muted[start_sample:, :] = 0

        elif data.ndim == 1:
            if start_sample >= data.shape[0]:
                print(f"⚠️ Warning: 'start_sample' ({start_sample}) exceeds trace length ({data.shape[0]}). No muting applied.")
                return data
            muted[start_sample:] = 0

        else:
            print(f"❌ Error: Unsupported data shape: {data.shape}")
            return None

        context["data"] = muted
        return muted

    except Exception as e:
        print(f"❌ Failed to mute data: {e}")
        return None
def resample(context, dt_in, dt_out, key='data', method="polyphase"):
    data = context.get(key)
    # survey_time = (len(data) - 1) * dt_in

    if data is None or not isinstance(data, np.ndarray):
        print("❌ Error: 'data' not found or invalid in context.")
        return None

    if not isinstance(dt_in, (int, float)) or dt_in <= 0:
        print(f"❌ Error: 'dt_in' must be a positive number. Got: {dt_in}")
        return None

    if not isinstance(dt_out, (int, float)) or dt_out <= 0:
        print(f"❌ Error: 'dt_out' must be a positive number. Got: {dt_out}")
        return None

    if method != "polyphase":
        print(f"❌ Error: Unsupported resampling method '{method}'. Only 'polyphase' is implemented.")
        return None

    try:
        ratio = dt_in / dt_out

        if data.ndim == 1:
            up = int(round(len(data) * ratio))
            down = len(data)
            resampled = resample_poly(data, up, down)

        elif data.ndim == 2:
            num_samples = data.shape[0]
            up = int(round(num_samples * ratio))
            down = num_samples
            resampled = resample_poly(data, up, down, axis=0)

        else:
            print(f"❌ Error: Unsupported data shape: {data.shape}")
            return None

        # context["data"] = resampled
        return resampled

    except Exception as e:
        print(f"❌ Failed to resample data: {e}")
        return None
def trim_samples(context, target_samples):
    data = context.get("data")

    if data is None or not isinstance(data, np.ndarray):
        print("❌ Error: 'data' not found or is not a NumPy array.")
        return None

    if not isinstance(target_samples, int) or target_samples <= 0:
        print(f"❌ Error: 'target_samples' must be a positive integer. Got: {target_samples}")
        return None

    try:
        current_samples = data.shape[0]

        if target_samples > current_samples:
            print(f"⚠️ Warning: target_samples ({target_samples}) > current_samples ({current_samples}). No trimming applied.")
            return data

        trimmed = data[:target_samples] if data.ndim == 1 else data[:target_samples, :]
        context["data"] = trimmed

        print(f"✅ Trimmed data from {current_samples} to {target_samples} samples.")
        return trimmed

    except Exception as e:
        print(f"❌ Failed to trim data: {e}")
        return None
def zero_phase_wavelet(context, key="data", shift=True):
    data = context.get(key)

    if data is None or not isinstance(data, np.ndarray):
        print("❌ Error: 'data' not found or is not a NumPy array.")
        return None

    if data.ndim != 1:
        print(f"❌ Error: Zero-phase conversion expects 1D wavelet. Got shape: {data.shape}")
        return None

    try:
        wavelet = np.copy(data)
        W_f = np.fft.fft(wavelet)
        magnitude = np.abs(W_f)
        zero_phase_W_f = magnitude
        zero_phase = np.fft.ifft(zero_phase_W_f)

        if shift:
            zero_phase = np.fft.ifftshift(zero_phase)

        print("✅ Created zero-phase version of wavelet.")
        return np.real(zero_phase)

    except Exception as e:
        print(f"❌ Failed to compute zero-phase wavelet: {e}")
        return None
def find_wavelet_main_lobe_center(wavelet, threshold_ratio=0.002):
    if not isinstance(wavelet, np.ndarray) or wavelet.ndim != 1:
        raise ValueError("Input must be a 1D NumPy array (wavelet)")

    max_amp = np.max(np.abs(wavelet))
    threshold = max_amp * threshold_ratio

    # Boolean mask where amplitude exceeds threshold
    above_thresh = np.abs(wavelet) >= threshold
    indices = np.where(above_thresh)[0]

    if len(indices) == 0:
        print("❌ No part of the wavelet exceeds the threshold.")
        return None, None, None

    start = indices[0]
    end = indices[-1] + 1  # Make the window inclusive

    wavelet = wavelet[start:end]

    center = len(wavelet) // 2
    length = len(wavelet)

    # print(f"✅ Wavelet main lobe: start={start}, end={end}, center={center}, length={length}")
    return wavelet, center


def default_duration(fmin: float | None, n_cycles: float = 4.0, max_duration: float = 1.0) -> float:
    if fmin is None or fmin <= 0:
        return 0.5
    return float(min(n_cycles / fmin, max_duration))


def ricker_wavelet(dt: float, fmax: float, duration: float):
    """
    Zero-phase Ricker wavelet (Mexican hat) derived from a highest frequency fmax
    using f0 = fmax / (3 * sqrt(pi)).
    """
    if fmax <= 0:
        raise ValueError("ricker_wavelet requires fmax > 0.")
    f0 = fmax / (3.0 * np.sqrt(np.pi))
    nsamp = int(round(duration / dt))
    if nsamp % 2 == 0:
        nsamp += 1
    t = (np.arange(nsamp) - nsamp // 2) * dt
    pf2 = (np.pi * f0) ** 2
    w = (1.0 - 2.0 * pf2 * t**2) * np.exp(-pf2 * t**2)
    if np.max(np.abs(w)) > 0:
        w /= np.max(np.abs(w))
    return w, t


def _trapezoid_spectrum(freqs, f1, f2, f3, f4):
    amp = np.zeros_like(freqs, dtype=float)
    m = (freqs >= f1) & (freqs < f2)
    if f2 > f1:
        amp[m] = (freqs[m] - f1) / (f2 - f1)
    m = (freqs >= f2) & (freqs <= f3)
    amp[m] = 1.0
    m = (freqs > f3) & (freqs <= f4)
    if f4 > f3:
        amp[m] = (f4 - freqs[m]) / (f4 - f3)
    return amp


def ormsby_impulse(dt: float, duration: float, f1: float, f2: float, f3: float, f4: float):
    nsamp = int(round(duration / dt))
    if nsamp % 2 == 0:
        nsamp += 1
    freqs = np.fft.rfftfreq(nsamp, dt)
    amp = _trapezoid_spectrum(freqs, f1, f2, f3, f4)
    w = np.fft.irfft(amp, nsamp)
    w = np.fft.fftshift(w)
    if np.max(np.abs(w)) > 0:
        w /= np.max(np.abs(w))
    t = (np.arange(nsamp) - nsamp // 2) * dt
    return w, t


def ramp_impulse(dt: float, duration: float, f1: float, f2: float, f3: float):
    nsamp = int(round(duration / dt))
    if nsamp % 2 == 0:
        nsamp += 1
    freqs = np.fft.rfftfreq(nsamp, dt)
    amp = np.zeros_like(freqs, dtype=float)
    up = (freqs >= f1) & (freqs <= f2)
    if f2 > f1:
        amp[up] = (freqs[up] - f1) / (f2 - f1)
    down = (freqs > f2) & (freqs <= f3)
    if f3 > f2:
        amp[down] = (f3 - freqs[down]) / (f3 - f2)
    w = np.fft.irfft(amp, nsamp)
    w = np.fft.fftshift(w)
    if np.max(np.abs(w)) > 0:
        w /= np.max(np.abs(w))
    t = (np.arange(nsamp) - nsamp // 2) * dt
    return w, t


def klauder_wavelet(dt: float, f1: float, f2: float, duration: float):
    nsamp = int(round(duration / dt))
    if nsamp < 2:
        nsamp = 2
    t_sweep = np.arange(nsamp) * dt
    sweep = chirp(t_sweep, f0=f1, f1=f2, t1=duration, method="linear")
    spec = np.fft.rfft(sweep)
    ac = np.fft.irfft(np.abs(spec) ** 2)
    ac = np.fft.fftshift(ac)
    if np.max(np.abs(ac)) > 0:
        ac /= np.max(np.abs(ac))
    t_ac = (np.arange(ac.size) - ac.size // 2) * dt
    return ac, t_ac


def bandpass(
    trace,
    dt: float,
    method: str = "butterworth",
    fmin: float | None = None,
    fmax: float | None = None,
    duration: float | None = None,
    order: int = 4,
    f1: float | None = None,
    f2: float | None = None,
    f3: float | None = None,
    f4: float | None = None,
    corners: tuple | list | None = None,
    filter_type: str = "bandpass",
    axis: int = 0,
    backend: str | Backend = "numpy",
):
    bk = get_backend(backend)
    xp = bk.xp
    sig = bk.signal

    x = xp.asarray(trace, dtype=xp.float32)
    nyq = 0.5 / dt
    method = method.lower()
    ndim = x.ndim
    if method == "butterworth":
        shape = (filter_type or "bandpass").lower()
        if shape == "lowpass":
            if fmax is None:
                raise ValueError("Butterworth lowpass requires fmax.")
            high = fmax / nyq
            if not (0.0 < high < 1.0):
                raise ValueError("Butterworth band must be within (0, Nyquist).")
            sos = sig.butter(order, high, btype="low", output="sos")
        elif shape == "highpass":
            if fmin is None:
                raise ValueError("Butterworth highpass requires fmin.")
            low = fmin / nyq
            if not (0.0 < low < 1.0):
                raise ValueError("Butterworth band must be within (0, Nyquist).")
            sos = sig.butter(order, low, btype="high", output="sos")
        else:  # bandpass default
            if fmin is None or fmax is None:
                raise ValueError("Butterworth bandpass requires fmin and fmax.")
            low = fmin / nyq
            high = fmax / nyq
            if not (0.0 < low < high < 1.0):
                raise ValueError("Butterworth band must be within (0, Nyquist).")
            sos = sig.butter(order, [low, high], btype="band", output="sos")
        return bk.sosfiltfilt(sos, x, axis=axis)
    if duration is None:
        if method in {"ormsby", "ramp"}:
            if f2 is not None:
                f_rep = f2
            elif f1 is not None:
                f_rep = f1
            elif corners is not None and len(corners) > 1:
                f_rep = corners[1]
            else:
                f_rep = None
        elif method == "ricker" and fmax is not None:
            f_rep = fmax
        elif fmin is not None:
            f_rep = fmin
        else:
            f_rep = None
        duration = default_duration(f_rep)
    if method == "ormsby":
        if corners is None:
            if None in (f1, f2, f3, f4):
                raise ValueError("Ormsby requires f1, f2, f3, f4.")
            corners = (f1, f2, f3, f4)
        f1, f2, f3, f4 = corners
        w_np, _ = ormsby_impulse(dt, duration, f1, f2, f3, f4)
        w = xp.asarray(w_np, dtype=xp.float32)
        if x.ndim == 1:
            return sig.fftconvolve(x, w, mode="same")
        w_nd = _reshape_kernel_for_axis_xp(w, ndim, axis, xp)
        return sig.fftconvolve(x, w_nd, mode="same")
    if method == "ricker":
        if fmax is None:
            if fmin is not None:
                fmax = fmin
            else:
                raise ValueError("Ricker requires fmax.")
        w_np, _ = ricker_wavelet(dt, fmax, duration)
        w = xp.asarray(w_np, dtype=xp.float32)
        if x.ndim == 1:
            return sig.fftconvolve(x, w, mode="same")
        w_nd = _reshape_kernel_for_axis_xp(w, ndim, axis, xp)
        return sig.fftconvolve(x, w_nd, mode="same")
    if method == "klauder":
        if fmin is not None and fmax is not None:
            f1, f2 = fmin, fmax
        elif corners is not None and len(corners) >= 2:
            f1, f2 = corners[0], corners[1]
        else:
            raise ValueError("Klauder requires fmin/fmax or first two corners.")
        w_np, _ = klauder_wavelet(dt, f1, f2, duration)
        w = xp.asarray(w_np, dtype=xp.float32)
        if x.ndim == 1:
            return sig.fftconvolve(x, w, mode="same")
        w_nd = _reshape_kernel_for_axis_xp(w, ndim, axis, xp)
        return sig.fftconvolve(x, w_nd, mode="same")
    if method == "ramp":
        if corners is None:
            if None in (f1, f2, f3):
                raise ValueError("Ramp requires f1, f2, f3.")
            corners = (f1, f2, f3)
        f1, f2, f3 = corners
        w_np, _ = ramp_impulse(dt, duration, f1, f2, f3)
        w = xp.asarray(w_np, dtype=xp.float32)
        if x.ndim == 1:
            return sig.fftconvolve(x, w, mode="same")
        w_nd = _reshape_kernel_for_axis_xp(w, ndim, axis, xp)
        return sig.fftconvolve(x, w_nd, mode="same")
    raise ValueError(f"Unknown method '{method}'")


def filter_bandpass_zarr_array(
    zin,
    dt: float,
    method: str = "butterworth",
    axis: int = 0,
    out=None,
    trace_block: int | None = None,
    backend: str | Backend = "numpy",
    **bp_kwargs,
):
    """
    Apply bandpass() to a large Zarr array in blocks (trace-wise) to avoid loading all data.

    Parameters
    ----------
    zin : zarr.Array or array-like
        Expected shape (nt, ntraces) with time on axis 0.
    dt : float
        Sample interval [s].
    method : str
        Filter method passed to bandpass.
    axis : int
        Time axis (only axis=0 supported in this helper).
    out : zarr.Array or None
        Optional destination; if None a NumPy array is returned (not recommended for huge data).
    trace_block : int or None
        Number of traces per block; defaults to zin.chunks[1] when available or 1024.
    backend : {'numpy', 'cupy', Backend}
        Compute backend for filtering.
    bp_kwargs : dict
        Extra args passed to bandpass (fmin/fmax/etc).
    """
    if axis != 0:
        raise NotImplementedError("filter_bandpass_zarr_array currently supports axis=0 (time) only.")
    nt, ntr = zin.shape
    if trace_block is None:
        try:
            trace_block = zin.chunks[1]
        except Exception:
            trace_block = 1024
    if out is None:
        out = np.empty((nt, ntr), dtype=np.float32)

    start = 0
    while start < ntr:
        stop = min(start + trace_block, ntr)
        block = np.asarray(zin[:, start:stop], dtype=np.float32)
        filt_block = bandpass(
            block,
            dt=dt,
            method=method,
            axis=axis,
            backend=backend,
            **bp_kwargs,
        )
        filt_block = get_backend(backend).to_numpy(filt_block).astype(np.float32, copy=False)
        out[:, start:stop] = filt_block
        start = stop

    return out


def filter(
    context: dict,
    method: str = "butterworth",
    fmin: float | None = None,
    fmax: float | None = None,
    order: int = 4,
    f1: float | None = None,
    f2: float | None = None,
    f3: float | None = None,
    f4: float | None = None,
    corners: tuple | list | None = None,
    filter_type: str = "bandpass",
    axis: int = 0,
    key_data: str = "data",
    output_key: str | None = None,
    trace_block: int | None = None,
    out_zarr: str | None = None,
):
    data = context.get(key_data)
    if data is None:
        raise ValueError("filter: 'data' missing.")
    dt_val = context.get("z_increment", context.get("z_inc", None))
    if dt_val is not None:
        dt_val = float(dt_val) / 1000.0
    else:
        dt_val = context.get("dt", None)
    if dt_val is None:
        raise ValueError("filter: dt missing; ensure context has z_increment/z_inc or dt.")

    def _to_float(val):
        if val is None:
            return None
        if isinstance(val, str) and val.strip() == "":
            return None
        try:
            return float(val)
        except Exception:
            return None

    fmin_val = _to_float(fmin)
    fmax_val = _to_float(fmax)
    f1_val = _to_float(f1)
    f2_val = _to_float(f2)
    f3_val = _to_float(f3)
    f4_val = _to_float(f4)
    if method == "butterworth" and (fmin_val is None or fmin_val <= 0):
        fmin_val = 1e-6

    try:
        import zarr as _z
    except Exception:
        _z = None

    # Zarr path: stream to new zarr on disk
    if _z is not None and isinstance(data, _z.core.Array):
        manifest_path = context.get("_manifest_path")
        if manifest_path is None:
            raise ValueError("filter: Zarr input detected but no manifest path in context.")
        src_path = context.get("zarr_store")
        base_name = Path(src_path).stem if src_path else "filtered"
        bin_dir = Path(manifest_path).parent
        allow_overwrite = bool(context.get("allow_overwrite", False))
        if out_zarr:
            out_path = Path(out_zarr)
            if not out_path.is_absolute():
                out_path = bin_dir / out_path
        else:
            out_path = bin_dir / f"{base_name}_filtered.zarr"
        if out_path.exists() and not allow_overwrite:
            idx = 1
            base = out_path.with_suffix("")
            suffix = out_path.suffix
            while True:
                cand = Path(f"{base}_v{idx}{suffix}")
                if not cand.exists():
                    out_path = cand
                    break
                idx += 1
        out_path.parent.mkdir(parents=True, exist_ok=True)
        compressor = Blosc(cname="zstd", clevel=5, shuffle=Blosc.SHUFFLE)
        chunks = getattr(data, "chunks", None)
        if chunks is None or len(chunks) < 2:
            chunks = (data.shape[0], min(512, data.shape[1]))
        root = zarr.open(out_path, mode="w" if allow_overwrite else "w-")
        out_arr = root.create_dataset(
            "amplitude",
            shape=data.shape,
            chunks=chunks,
            dtype=np.float32,
            compressor=compressor,
        )
        out_arr.attrs["_ARRAY_DIMENSIONS"] = ["sample", "trace"]
        filter_bandpass_zarr_array(
            data,
            dt=dt_val,
            method=method,
            axis=axis,
            out=out_arr,
            trace_block=trace_block,
            fmin=fmin_val,
            fmax=fmax_val,
            f1=f1_val,
            f2=f2_val,
            f3=f3_val,
            f4=f4_val,
            corners=corners,
            filter_type=filter_type,
            order=order,
        )
        geom_path = context.get("geometry_parquet") or context.get("_geometry_path", "")
        base_meta = {}
        try:
            base_meta = json.loads(Path(manifest_path).read_text())
        except Exception:
            base_meta = {}
        manifest = {
            "dataset_id": str(uuid4()),
            "parent_store": str(src_path) if src_path else "",
            "zarr_store": str(out_path.resolve()),
            "geometry_parquet": str(Path(geom_path).resolve()) if geom_path else "",
            "dataset_type": base_meta.get("dataset_type", context.get("dataset_type", "")) or "",
            "trace_count": int(data.shape[1]),
            "samples": int(data.shape[0]),
            "chunk_trace": int(chunks[1] if len(chunks) > 1 else data.shape[1]),
            "created_at": datetime.utcnow().isoformat() + "Z",
            "selected_headers": base_meta.get("selected_headers", {}) or {},
        }
        manifest_path_out = Path(str(out_path) + ".manifest.json")
        manifest_path_out.write_text(json.dumps(manifest, indent=2))
        context["data"] = zarr.open(out_path, mode="r")["amplitude"]
        context["zarr_store"] = str(out_path.resolve())
        context["_manifest_path"] = str(manifest_path_out.resolve())
        context["geometry_parquet"] = str(Path(geom_path).resolve()) if geom_path else context.get("geometry_parquet")
        return context["data"]

    if not isinstance(data, np.ndarray):
        raise ValueError("filter: 'data' missing or not array.")
    try:
        filtered = np.apply_along_axis(
            lambda tr: bandpass(
                tr,
                dt_val,
                method=method,
                fmin=fmin_val,
                fmax=fmax_val,
                order=order,
                f1=f1_val,
                f2=f2_val,
                f3=f3_val,
                f4=f4_val,
                corners=corners,
                filter_type=filter_type,
            ),
            axis,
            data,
        )
    except Exception as exc:
        raise RuntimeError(f"filter failed: {exc}") from exc
    context[output_key or key_data] = filtered
    return filtered


def save_dataset(
    context: dict,
    out_zarr: str | Path,
    key_data: str = "data",
    key_geometry: str = "geometry_parquet",
    allow_overwrite: bool = False,
) -> str:
    """
    Persist a dataset in the context to a new Zarr store, reusing an existing
    geometry parquet path from the context.

    If data is already a Zarr array, this will rename/move the store to the
    requested out_zarr (if different). If out_zarr matches the current store,
    it becomes a no-op.

    key_data points to the numpy array to save, key_geometry points to the
    parquet path (or context entries _geometry_path / geometry_parquet fallback).
    """
    data = context.get(key_data)
    try:
        import zarr as _z
    except Exception:
        _z = None
    if data is None:
        raise ValueError("save_dataset: 'data' missing.")
    if _z is not None and isinstance(data, _z.core.Array):
        # Already on disk; optionally rename/move to requested out_zarr
        current_store = context.get("zarr_store")
        current_manifest = context.get("_manifest_path")
        if current_store is None or current_manifest is None:
            raise ValueError("save_dataset: zarr data present but missing zarr_store/_manifest_path in context.")
        target = Path(out_zarr)
        if not target.is_absolute():
            bin_dir = Path(current_manifest).parent
            target = bin_dir / target
        if Path(current_store).resolve() == target.resolve():
            return str(target.resolve())
        if target.exists():
            if not allow_overwrite:
                raise FileExistsError(f"{target} exists. Set allow_overwrite=True to replace.")
            # remove existing target
            if target.is_dir():
                shutil.rmtree(target, ignore_errors=True)
            else:
                target.unlink(missing_ok=True)
        target.parent.mkdir(parents=True, exist_ok=True)
        # move store
        shutil.move(str(current_store), str(target))
        # update manifest
        manifest_path_old = Path(str(current_store) + ".manifest.json")
        manifest_path_new = Path(str(target) + ".manifest.json")
        try:
            meta = json.loads(manifest_path_old.read_text())
            meta["zarr_store"] = str(target.resolve())
            manifest_path_new.write_text(json.dumps(meta, indent=2))
            # remove old manifest if different
            if manifest_path_new.resolve() != manifest_path_old.resolve():
                manifest_path_old.unlink(missing_ok=True)
        except Exception:
            pass
        context["zarr_store"] = str(target.resolve())
        context["_manifest_path"] = str(manifest_path_new.resolve())
        return str(target.resolve())
    if not isinstance(data, np.ndarray):
        raise ValueError("save_dataset: data must be numpy array or zarr array.")
    if data.ndim != 2:
        raise ValueError("save_dataset: data must be 2D (samples x traces).")

    geom_val = None
    for candidate_key in (key_geometry, "_geometry_path", "geometry_parquet"):
        candidate = context.get(candidate_key)
        if isinstance(candidate, (str, Path)):
            geom_val = candidate
            break
        if candidate is not None and not isinstance(candidate, (str, Path)):
            # skip non-path values (e.g., DataFrame) silently and keep searching
            continue
    if geom_val is None:
        raise ValueError("save_dataset: geometry path not found; provide key_geometry or load manifest first.")
    geom_path = Path(geom_val)
    if not geom_path.exists():
        raise FileNotFoundError(f"Geometry file not found: {geom_path}")

    parent_store = context.get("zarr_store")
    # Inherit metadata from source manifest if available
    base_meta = {}
    manifest_src = context.get("_manifest_path")
    if manifest_src and Path(manifest_src).exists():
        try:
            base_meta = json.loads(Path(manifest_src).read_text())
        except Exception:
            base_meta = {}
    base_selected = base_meta.get("selected_headers", {}) if isinstance(base_meta, dict) else {}
    base_chunk = base_meta.get("chunk_trace", None) if isinstance(base_meta, dict) else None
    base_dtype = base_meta.get("dataset_type", "") if isinstance(base_meta, dict) else ""

    dataset_type = context.get("dataset_type", "") or base_dtype or ""
    if not isinstance(allow_overwrite, bool):
        allow_overwrite = bool(allow_overwrite)
    if manifest_src:
        bin_dir = Path(manifest_src).parent
    else:
        raise ValueError("save_dataset: missing survey manifest path; load a dataset before saving.")
    out_name = Path(out_zarr).name
    out_zarr = bin_dir / out_name
    if out_zarr.exists() and not allow_overwrite:
        raise FileExistsError(f"{out_zarr} exists. Set allow_overwrite=True to replace.")
    out_zarr.parent.mkdir(parents=True, exist_ok=True)

    n_samples, n_traces = data.shape
    trace_ids = context.get("trace_ids")
    if trace_ids is None:
        geom_df = context.get("geometry")
        if isinstance(geom_df, pd.DataFrame):
            if len(geom_df) == n_traces:
                # geometry already aligned with data
                if "trace_id" in geom_df.columns:
                    trace_ids = geom_df["trace_id"].to_numpy()
                else:
                    trace_ids = np.arange(n_traces, dtype=np.int64)
            else:
                # cannot infer mapping for subset without explicit trace_ids
                raise ValueError(
                    "save_dataset: trace_ids missing for subset data; "
                    "provide context['trace_ids'] mapping to geometry rows."
                )
    if trace_ids is not None:
        trace_ids = np.asarray(trace_ids, dtype=np.int64)
        if trace_ids.shape[0] != n_traces:
            raise ValueError("trace_ids length must match number of traces in data.")

    chunk_trace = int(context.get("chunk_trace", base_chunk if base_chunk else min(n_traces, 512)))
    compressor_val = context.get("compressor")
    compressor = compressor_val if isinstance(compressor_val, Blosc) else Blosc(cname="zstd", clevel=5, shuffle=Blosc.SHUFFLE)

    root = zarr.open(out_zarr, mode="w-" if not allow_overwrite else "w")
    root.attrs.update(
        {
            "description": "Filtered seismic amplitude",
            "parent_store": str(parent_store) if parent_store else "",
            "geometry_parquet": str(geom_path.resolve()),
            "dataset_type": dataset_type,
            "created_at": datetime.utcnow().isoformat() + "Z",
        }
    )

    if trace_ids is not None:
        root.create_dataset(
            "trace_ids",
            data=trace_ids,
            shape=trace_ids.shape,
            chunks=(min(len(trace_ids), chunk_trace),),
            dtype="int64",
        )

    amp_ds = root.create_dataset(
        "amplitude",
        shape=(n_samples, n_traces),
        chunks=(n_samples, min(chunk_trace, n_traces)),
        dtype="float32",
        compressor=compressor,
    )
    amp_ds.attrs["_ARRAY_DIMENSIONS"] = ["sample", "trace"]

    # copy in manageable blocks along trace dimension
    block = 4096
    if isinstance(data, np.ndarray):
        for start in range(0, n_traces, block):
            end = min(start + block, n_traces)
            amp_ds[:, start:end] = data[:, start:end]
    else:
        # zarr -> zarr copy
        for start in range(0, n_traces, block):
            end = min(start + block, n_traces)
            amp_ds[:, start:end] = np.asarray(data[:, start:end], dtype=np.float32)

    manifest = {
        "dataset_id": str(uuid4()),
        "parent_store": str(parent_store) if parent_store else "",
        "zarr_store": str(out_zarr.resolve()),
        "geometry_parquet": str(geom_path.resolve()),
        "dataset_type": dataset_type,
        "trace_count": int(n_traces),
        "samples": int(n_samples),
        "chunk_trace": int(chunk_trace),
        "created_at": datetime.utcnow().isoformat() + "Z",
    }
    if trace_ids is not None:
        manifest["trace_ids_dataset"] = True
    if base_selected:
        manifest["selected_headers"] = base_selected

    manifest_path = Path(str(out_zarr) + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2))

    context["zarr_store"] = str(out_zarr.resolve())
    context["geometry_parquet"] = str(geom_path.resolve())
    return str(out_zarr.resolve())


def calculate_convolution_operator(context, key="data", threshold_ratio=0.002):
    wavelet = context.get(key)

    if wavelet is None or not isinstance(wavelet, np.ndarray) or wavelet.ndim != 1:
        print(f"❌ Error: '{key}' in context must be a valid 1D NumPy wavelet.")
        return None

    try:
        wavelet_cut, center = find_wavelet_main_lobe_center(wavelet, threshold_ratio=threshold_ratio)

        if wavelet_cut is None or center is None:
            print("❌ Error: Failed to extract wavelet main lobe.")
            return None

        Cop = pylops.signalprocessing.Convolve1D(len(wavelet), h=wavelet_cut, offset=center, dtype="float32")

        print(f"✅ Convolution operator created from context['{key}']")
        return Cop

    except Exception as e:
        print(f"❌ Failed to create convolution operator: {e}")
        return None
    
def free_gpu_memory(func):
    def wrapper_func(*args, **kwargs):
        retval = func(*args, **kwargs)
        if cp is not None:
            cp._default_memory_pool.free_all_blocks()
        return retval
    return wrapper_func

@free_gpu_memory
def apply_designature(context, key_input="wavelet_input", key_output="wavelet_output", data_key="data", mode="cpu"):
    wavelet_in = context.get(key_input)
    wavelet_out = context.get(key_output)
    data = context.get(data_key)
    # operator = context.get("operator")
    if wavelet_out is None or data is None:
        print("❌ Error: wavelet or data not found in context.")
        return None

    try:
        
        if data.ndim == 1:
            print("Entrei aqui")
            # Cop_out = pylops.signalprocessing.Convolve1D(len(data), h=wavelet_cut, offset=offset)
            # reflectivity = operator / data
            # modeled = Cop @ reflectivity
            # print("✅ 1D designature applied using operator inversion.")
            # return modeled

        elif data.ndim == 2:
            n_samples, n_traces = data.shape
            modeled = np.zeros_like(data)

            wavelet_in_cut, offset_in = find_wavelet_main_lobe_center(wavelet_in)
            wavelet_out_cut, offset_out = find_wavelet_main_lobe_center(wavelet_out)

           
            if mode == "gpu":
                if cp is None:
                    raise ImportError("cupy is required for GPU designature. Install openseismicprocessing with the 'gpu' extra or set mode='cpu'.")

                wavelet_in_cut_gpu = cp.array(wavelet_in_cut)
                wavelet_out_cut_gpu = cp.array(wavelet_out_cut)
                data_gpu = cp.array(data)

                Cop_in = pylops.signalprocessing.Convolve1D(dims=[n_samples,n_traces], h=wavelet_in_cut_gpu, offset=offset_in, axis=0, dtype='float32')
                Cop_out = pylops.signalprocessing.Convolve1D(dims=[n_samples,n_traces], h=wavelet_out_cut_gpu, offset=offset_out, axis=0, dtype='float32')
            
                reflectivity_gpu = Cop_in / data_gpu                
                cp.fft.config.clear_plan_cache()
                
                modeled_gpu = Cop_out * reflectivity_gpu
                cp.fft.config.clear_plan_cache()

                modeled = cp.asnumpy(modeled_gpu)

                print("✅ 2D designature using GPU.")
                return modeled.reshape([n_samples,n_traces])
            else:
                Cop_in = pylops.signalprocessing.Convolve1D(dims=[n_samples,n_traces], h=wavelet_in_cut, offset=offset_in, axis=0, dtype='float32')
                Cop_out = pylops.signalprocessing.Convolve1D(dims=[n_samples,n_traces], h=wavelet_out_cut, offset=offset_out, axis=0, dtype='float32')

                reflectivity = Cop_in / data.flatten()
                modeled = Cop_out @ reflectivity

                print("✅ 2D designature using CPU.")
                return modeled.reshape([n_samples,n_traces])

        else:
            print(f"❌ Unsupported data dimension: {data.ndim}")
            return None

    except Exception as e:
        print(f"❌ Failed to apply designature: {e}")
        return None


def apply_deghost(
    context: dict,
    ffid_header: str = "FFID",
    velocity: float = 1500.0,
    dz: float = 12.5,
    dt: float | None = None,
    pad: int = 30,
    npad: int = 5,
    ntaper: int = 11,
    key_data: str = "data",
    key_geometry: str = "geometry",
    output_key: str | None = None,
):
    """Apply receiver-side deghosting gather-by-gather using pylops.waveeqprocessing.Deghosting."""
    data = context.get(key_data)
    geom = context.get(key_geometry)
    if data is None or not isinstance(data, np.ndarray):
        raise ValueError("apply_deghost: 'data' missing or not numpy array.")
    if geom is None or not isinstance(geom, pd.DataFrame):
        raise ValueError("apply_deghost: 'geometry' missing or not DataFrame.")
    required = [ffid_header, "GroupX", "GroupY", "SourceX", "SourceY"]
    missing = [c for c in required if c not in geom.columns]
    if missing:
        raise ValueError(f"apply_deghost: missing geometry columns {missing}")
    # Derive dt from context z_increment (ms) if not provided
    if dt is None:
        dt_ms = context.get("z_increment", context.get("z_inc", None))
        if dt_ms is not None:
            dt_sec = float(dt_ms) / 1000.0
        else:
            dt_sec = context.get("dt", 0.001)
    else:
        dt_sec = dt
    out = np.zeros_like(data)
    ffids = geom[ffid_header].to_numpy()
    unique_ffids = np.unique(ffids)

    for ffid in unique_ffids:
        mask = ffids == ffid
        if not mask.any():
            continue
        gather_idx = geom.index[mask].to_numpy()
        sis = data[:, gather_idx]
        if pad and pad > 0:
            sis = np.pad(sis, pad_width=pad, mode="edge")
        r = np.asarray([geom.loc[mask, "GroupX"].to_numpy(), geom.loc[mask, "GroupY"].to_numpy()])
        s = np.asarray([geom.loc[mask, "SourceX"].to_numpy(), geom.loc[mask, "SourceY"].to_numpy()])
        nr = r.shape[1]
        if nr < 2:
            out[:, gather_idx] = data[:, gather_idx]
            continue
        dr = float(np.mean(np.diff(r[0])))
        win = np.ones_like(sis)
        taper = min(ntaper, max(1, nr // 2))

        arr_mod = cp if cp is not None else np
        sis_arr = arr_mod.asarray(sis)
        win_arr = arr_mod.asarray(win)
        zeros_arr = arr_mod.zeros_like(sis_arr).ravel()

        try:
            res = pylops.waveeqprocessing.Deghosting(
                sis_arr,
                sis_arr.shape[0],
                sis_arr.shape[1],
                dt_sec,
                dr,
                velocity,
                r[1, 0] + dz,
                win=win_arr,
                npad=npad,
                ntaper=taper,
                solver=pylops.optimization.basic.cgls,
                dottest=False,
                dtype="complex128",
                **dict(x0=zeros_arr, damp=1e-0, niter=60),
            )
            down = res[1]
            if cp is not None and hasattr(down, "get"):
                down = down.get()
        except Exception as exc:
            raise RuntimeError(f"Deghosting failed for FFID {ffid}: {exc}") from exc
        finally:
            if cp is not None:
                try:
                    del sis_arr, win_arr, zeros_arr, res
                except Exception:
                    pass
                try:
                    cp.get_default_memory_pool().free_all_blocks()
                    cp.get_default_pinned_memory_pool().free_all_blocks()
                    cp.fft.config.clear_plan_cache()
                except Exception:
                    pass

        if pad and pad > 0:
            down = down[pad:-pad, pad:-pad]
        down = np.asarray(down, dtype=np.float32)
        out[:, gather_idx] = down

    context[output_key or key_data] = out
    return out
def sort(context, header1, header2=None, key_data="data", key_geometry="geometry"):
    df = context.get(key_geometry)
    data = context.get(key_data)

    if df is None or not isinstance(df, pd.DataFrame):
        print("❌ Error: 'geometry' not found or invalid.")
        return None

    if data is None or not isinstance(data, np.ndarray) or data.ndim != 2:
        print("❌ Error: 'data' not found or not a valid 2D array.")
        return None

    if header1 not in df.columns or (header2 and header2 not in df.columns):
        print(f"❌ Error: One or both headers not found in geometry. Got: '{header1}', '{header2}'")
        return None

    try:
        # Sort geometry
        sort_cols = [header1] if header2 is None else [header1, header2]
        sorted_df = df.sort_values(by=sort_cols, ignore_index=True)
        sorted_indices = sorted_df.index.to_numpy()

        # Use original DataFrame to find old-to-new index mapping
        sorted_positions = df.sort_values(by=sort_cols).index.to_numpy()

        # Reorder seismic data columns to match new geometry
        sorted_data = data[:, sorted_positions]

        # Update context
        context[key_geometry] = sorted_df
        context[key_data] = sorted_data

        print(f"✅ Geometry and data sorted by: {', '.join(sort_cols)}")
        return sorted_df

    except Exception as e:
        print(f"❌ Failed to sort geometry: {e}")
        return None
    
def scale_coordinate_units(context, key="geometry", XY_headers=['SourceX', 'SourceY', 'GroupX', 'GroupY'], elevation_headers=['SourceDepth', 'ReceiverDatumElevation'], 
                           XY_Scaler = 100., elevation_scaler = 100.):
    df = context.get(key)

    if df is None or not isinstance(df, pd.DataFrame):
        print("❌ Error: 'geometry' not found or invalid.")
        return None

    try:

        for header in XY_headers:
            df[header]/= XY_Scaler

        for header in elevation_headers:
            df[header]/= elevation_scaler

        return df

    except Exception as e:
        print(f"❌ Failed to scale geometry: {e}")
        return None
