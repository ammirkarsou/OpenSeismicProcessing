import os
import numpy as np
import pandas as pd

from ._io import (
    open_segy_data,
    parse_trace_headers,
    parse_text_header
)

def store_geometry_as_parquet(context, file_path, key_geometry="geometry"):
    """
    Stores the geometry DataFrame from the context as a Parquet file.
    
    Parameters:
        context (dict): Dictionary containing the geometry DataFrame.
        file_path (str): The destination file path for storing the Parquet file.
        key_geometry (str): Key in context where the geometry DataFrame is stored (default "geometry").
    
    Returns:
        bool: True if the file is stored successfully, False otherwise.
    """
    # Validate context
    if context is None or not isinstance(context, dict):
        print("❌ Error: Invalid context provided.")
        return False

    # Validate file_path
    if not isinstance(file_path, str) or file_path.strip() == "":
        print("❌ Error: A valid file path must be provided.")
        return False

    # Optionally check if the directory exists (if not, you might want to create it)
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        print(f"❌ Error: The directory '{directory}' does not exist.")
        return False

    # Retrieve the geometry DataFrame
    geometry_df = context.get(key_geometry)
    if geometry_df is None or not isinstance(geometry_df, pd.DataFrame):
        print(f"❌ Error: Geometry DataFrame not found or invalid under key '{key_geometry}'.")
        return False

    try:
        # Store the DataFrame as a Parquet file
        geometry_df.to_parquet(file_path, index=False)
        print(f"✅ Successfully stored geometry DataFrame to '{file_path}'.")
        return True
    except Exception as e:
        print(f"❌ Error: Failed to store geometry DataFrame to '{file_path}': {e}")
        return False

def get_binary_header(context, file_path):
    """
    Opens a SEGY file, extracts its binary header,
    updates the context, and manually closes the file.

    Parameters:
        context (dict): A dictionary to store results.
        file_path (str): Path to the SEGY file.
        key (str): The context key under which to store the binary header (default 'binary_header').

    Returns:
        The binary header if successful, otherwise None.
    """
    # Validate context
    if context is None or not isinstance(context, dict):
        print("❌ Error: Invalid context provided.")
        return None

    # Verify that the file exists
    if not file_path or not os.path.exists(file_path):
        print(f"❌ Error: File '{file_path}' does not exist.")
        return None

    try:
        # Manually open the SEGY file
        segy_file = open_segy_data(file_path,ignore_geometry=True)
    except Exception as e:
        print(f"❌ Error: Failed to open SEGY file '{file_path}': {e}")
        return None

    try:
        # Extract the binary header.
        # In segyio, the binary header is available via segy_file.bin.
        binary_header = segy_file.bin
    except Exception as e:
        print(f"❌ Error: Failed to extract binary header from '{file_path}': {e}")
        binary_header = None
    finally:
        try:
            segy_file.close()
        except Exception as e:
            print(f"❌ Error: Failed to close SEGY file '{file_path}': {e}")

    if binary_header is None:
        return None

    return binary_header

def get_trace_data(context, file_path, ignore_geometry=True):
    """
    Opens a SEGY file or all SEGY files in a folder, extracts trace data, updates the context, 
    and manually closes the files. For folders, data is horizontally stacked.

    Parameters:
        context (dict): A dictionary to store results.
        file_path (str): Path to the SEGY file or folder containing SEGY files.
        ignore_geometry (bool): Whether to ignore geometry when opening SEGY files.

    Returns:
        The extracted trace data (a NumPy array) if successful, otherwise None.
        For folders, returns horizontally stacked data from all SEGY files.
    """
    # Validate context
    if context is None or not isinstance(context, dict):
        print("❌ Error: Invalid context provided.")
        return None

    # Verify that the file/folder exists
    if not file_path or not os.path.exists(file_path):
        print(f"❌ Error: Path '{file_path}' does not exist.")
        return None

    # Handle folder case
    if os.path.isdir(file_path):
        # Find all SEGY files in the folder
        segy_files = []
        for f in os.listdir(file_path):
            if f.lower().endswith(('.sgy', '.segy')):
                segy_files.append(os.path.join(file_path, f))
        
        if not segy_files:
            print(f"❌ Error: No SEGY files found in folder '{file_path}'")
            return None

        # Process each file and stack the data
        trace_data_list = []
        for segy_file_path in segy_files:
            try:
                # Open the SEGY file
                segy_file = open_segy_data(segy_file_path,ignore_geometry=ignore_geometry)
                # Extract trace data and add to list
                trace_data = np.ascontiguousarray(segy_file.trace.raw[:].T)
                trace_data_list.append(trace_data)
            except Exception as e:
                print(f"❌ Error: Failed to process SEGY file '{segy_file_path}': {e}")
                continue
            finally:
                try:
                    if 'segy_file' in locals():
                        segy_file.close()
                except Exception as e:
                    print(f"❌ Error: Failed to close SEGY file '{segy_file_path}': {e}")

        if not trace_data_list:
            print(f"❌ Error: Could not read any SEGY files in folder '{file_path}'")
            return None

        # Horizontally stack all trace data
        try:
            return np.hstack(trace_data_list)
        except Exception as e:
            print(f"❌ Error: Failed to stack trace data from folder '{file_path}': {e}")
            return None

    # Handle single file case (original functionality)
    else:
        try:
            # Open the SEGY file manually (without using a context manager)
            segy_file = open_segy_data(file_path,ignore_geometry=ignore_geometry)
        except Exception as e:
            print(f"❌ Error: Failed to open SEGY file '{file_path}': {e}")
            return None

        try:
            # Extract the trace data from the file.
            trace_data = np.ascontiguousarray(segy_file.trace.raw[:].T)
        except Exception as e:
            print(f"❌ Error: Failed to extract trace data from '{file_path}': {e}")
            trace_data = None
        finally:
            try:
                segy_file.close()
            except Exception as e:
                print(f"❌ Error: Failed to close SEGY file '{file_path}': {e}")

        return trace_data

def get_trace_header(context, file_path, ignore_geometry=True):
    """
    Opens a SEGY file or all SEGY files in a folder, extracts trace headers using parse_trace_headers,
    and updates the context. For folders, headers are combined into a single DataFrame.
    
    Parameters:
        context (dict): A dictionary to store results.
        file_path (str): Path to the SEGY file or folder containing SEGY files.
        ignore_geometry (bool): Whether to ignore geometry when opening SEGY files.
    
    Returns:
        The parsed trace headers (combined DataFrame if folder) if successful, otherwise None.
    """
    # Validate context
    if context is None or not isinstance(context, dict):
        print("❌ Error: Invalid context provided.")
        return None

    # Verify that the file/folder exists
    if not file_path or not os.path.exists(file_path):
        print(f"❌ Error: Path '{file_path}' does not exist.")
        return None

    # Handle folder case
    if os.path.isdir(file_path):
        # Find all SEGY files in the folder
        segy_files = []
        for f in os.listdir(file_path):
            if f.lower().endswith(('.sgy', '.segy')):
                segy_files.append(os.path.join(file_path, f))
        
        if not segy_files:
            print(f"❌ Error: No SEGY files found in folder '{file_path}'")
            return None

        # Process each file and combine DataFrames
        combined_df = None
        for i, segy_file_path in enumerate(segy_files):
            try:
                # Open the SEGY file
                segy_file = open_segy_data(segy_file_path, ignore_geometry=ignore_geometry)
                # Extract trace headers
                headers_df = parse_trace_headers(segy_file)
                
                # Add file identifier column if combining multiple files
                if len(segy_files) > 1:
                    headers_df['source_file'] = os.path.basename(segy_file_path)
                    headers_df['file_index'] = i
                
                # Combine DataFrames
                if combined_df is None:
                    combined_df = headers_df
                else:
                    combined_df = pd.concat([combined_df, headers_df], ignore_index=True)
                    
            except Exception as e:
                print(f"❌ Error: Failed to process SEGY file '{segy_file_path}': {e}")
                continue
            finally:
                try:
                    if 'segy_file' in locals():
                        segy_file.close()
                except Exception as e:
                    print(f"❌ Error: Failed to close SEGY file '{segy_file_path}': {e}")

        if combined_df is None or combined_df.empty:
            print(f"❌ Error: Could not read headers from any SEGY files in folder '{file_path}'")
            return None

        return combined_df

    # Handle single file case
    else:
        try:
            segy_file = open_segy_data(file_path, ignore_geometry=ignore_geometry)
        except Exception as e:
            print(f"❌ Error: Failed to open SEGY file '{file_path}': {e}")
            return None

        try:
            trace_header = parse_trace_headers(segy_file)
        except Exception as e:
            print(f"❌ Error: Failed to parse trace header from '{file_path}': {e}")
            trace_header = None
        finally:
            try:
                segy_file.close()
            except Exception as e:
                print(f"❌ Error: Failed to close SEGY file '{file_path}': {e}")

        return trace_header

def get_text_header(context, file_path):
    """
    Opens a SEGY file, extracts its text header, and updates the context.

    Parameters:
        context (dict): A dictionary to store results.
        file_path (str): Path to the SEGY file.
        key (str): The context key under which to store the text header (default 'text_header').

    Returns:
        The parsed text header if successful, otherwise None.
    """
    # Validate context
    if context is None or not isinstance(context, dict):
        print("❌ Error: Invalid context provided.")
        return None

    # Verify that the file exists
    if not file_path or not os.path.exists(file_path):
        print(f"❌ Error: File '{file_path}' does not exist.")
        return None

    try:
        segy_file = open_segy_data(file_path)
    except Exception as e:
        print(f"❌ Error: Failed to open SEGY file '{file_path}': {e}")
        return None

    try:
        text_headers = parse_text_header(segy_file)
    except Exception as e:
        print(f"❌ Error: Failed to parse text header from '{file_path}': {e}")
        return None

    #close segy
    segy_file.close()

    return text_headers

def read_data(context, file_path, format="binary", order='C', shape=None):
    if not os.path.isfile(file_path):
        print(f"❌ Error: File not found — {file_path}")
        return None

    if format not in ["binary", "npy"]:
        print(f"❌ Error: Format must be 'binary' or 'npy'. Got: {format}")
        return None

    try:
        if format == "npy":
            data = np.load(file_path)
        elif format == "binary":
            if shape is None:
                print("❌ Error: 'shape' must be provided for binary files.")
                return None
            data = np.fromfile(file_path, dtype=np.float32).reshape(shape, order=order)
        else:
            return None

        print(f"✅ Data loaded: {file_path} — shape: {data.shape}")
        return data

    except Exception as e:
        print(f"❌ Failed to read data: {e}")
        return None
    

def write_data(context, file_folder, file_name, format="binary"):
    data = context.get("data")

    if data is None or not isinstance(data, np.ndarray):
        print("❌ Error: 'data' not found or is not a NumPy array.")
        return None

    if not os.path.isdir(file_folder):
        print(f"❌ Error: Directory does not exist — {file_folder}")
        return None

    if format not in ["binary", "npy"]:
        print(f"❌ Error: Format must be 'binary' or 'npy'. Got: {format}")
        return None

    try:
        shape_str = ""
        if format == "binary":
            # Determine shape string for filename when writing binary files.
            if data.ndim == 1:
                shape_str = f"_{len(data)}Samples"
            elif data.ndim == 2:
                shape_str = f"_{data.shape[1]}x{data.shape[0]}Samples"
            else:
                print(f"❌ Error: Only 1D or 2D arrays are supported for binary output. Got shape: {data.shape}")
                return None

        # Determine full file path
        ext = ".bin" if format == "binary" else ".npy"
        full_name = f"{file_name}{shape_str}{ext}"
        file_path = os.path.join(file_folder, full_name)

        if format == "binary":
            if data.ndim > 1:
                data = data.T
            data = np.ascontiguousarray(data.astype(np.float32))
            data.tofile(file_path)
        else:
            np.save(file_path, data.astype(np.float32))

        print(f"✅ Data written: {file_path}")
        return file_path

    except Exception as e:
        print(f"❌ Failed to write data: {e}")
        return None
    

def import_npy_mmap(file_in, mode="r"):
    valid_modes = [None, "r", "r+", "w+", "c"]

    if mode not in valid_modes:
        print(f"❌ Error: Invalid mode '{mode}'. Valid modes are: {valid_modes}")
        return None

    if not os.path.isfile(file_in):
        print(f"❌ Error: File '{file_in}' does not exist.")
        return None

    try:
        return np.load(file_in, mmap_mode=mode)
    except Exception as e:
        print(f"❌ Failed to load file '{file_in}' with mode '{mode}': {e}")
        return None
    
def import_parquet_file(file_in):
    if not os.path.isfile(file_in):
        print(f"❌ Error: File '{file_in}' does not exist.")
        return None

    try:
        return pd.read_parquet(file_in)
    except Exception as e:
        print(f"❌ Failed to read parquet file '{file_in}': {e}")
        return None
