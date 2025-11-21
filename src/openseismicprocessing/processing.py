from scipy.signal import resample_poly
import numpy as np
import pandas as pd
import os
import inspect
import re
import pylops

try:
    import cupy as cp
except ImportError:  # pragma: no cover - optional dependency
    cp = None

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
        print("❌ Error: 'geometry' not found in context.")
        return None

    if not isinstance(df, pd.DataFrame):
        print("❌ Error: 'geometry' in context is not a valid DataFrame.")
        return None

    # Extract variable names from the expression (basic regex for word-like tokens)
    tokens = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', expression)
    # Remove known functions like 'abs', 'sin', etc. — we keep only those not in numpy/pandas built-ins
    builtins = {"abs", "log", "sqrt", "sin", "cos", "tan", "exp", "min", "max", "mean", "std", "sum"}
    columns_used = [t for t in tokens if t not in builtins]

    # Check if all used columns exist
    missing = [col for col in columns_used if col not in df.columns]
    if missing:
        print(f"❌ Error: Column(s) not found in geometry: {missing}")
        return None

    try:
        df[header_name] = df.eval(expression)
        return df
    except Exception as e:
        print(f"❌ Failed to create header '{header_name}': {e}")
        return None
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
