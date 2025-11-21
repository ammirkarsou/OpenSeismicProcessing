import cv2
import numpy as np
import numba as nb
from scipy.interpolate import CubicHermiteSpline
import os
import ctypes
import pandas as pd
import importlib.resources as resources

try:
    import cupy as cp
except ImportError:  # pragma: no cover - optional dependency
    cp = None


def _require_cupy() -> None:
    if cp is None:
        raise ImportError(
            "CuPy is required for GPU-based migration routines. Install the 'openseismicprocessing[gpu]' extra or a matching CuPy wheel for your CUDA version."
        )

try:
    import nvidia.dali as dali  # type: ignore
    from nvidia.dali import pipeline_def, fn  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    dali = None
    pipeline_def = None
    fn = None
from numba.typed import Dict
from numba.types import Tuple, int64

def _load_shared_library() -> ctypes.CDLL:
    """
    Locate the packaged shared library and return a loaded CDLL handle.
    """
    try:
        with resources.path("openseismicprocessing.lib", "libEikonal.so") as lib_path:
            return ctypes.CDLL(str(lib_path))
    except FileNotFoundError as exc:  # pragma: no cover - defensive
        raise FileNotFoundError("Shared library 'libEikonal.so' not found in openseismicprocessing.lib package data.") from exc


# Load the shared library
fsm_lib = _load_shared_library()

# Declare argument types
fsm_lib.fast_sweeping_method.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
    ctypes.c_float, ctypes.c_float,  # sx, sz
    ctypes.c_float, ctypes.c_float,  # dx, dz
    ctypes.c_int, ctypes.c_int       # nx, nz
]
fsm_lib.fast_sweeping_method.restype = ctypes.POINTER(ctypes.c_float)

# --------------------------
# Declare migrate_constant_velocity
# --------------------------
_c_void_p = ctypes.c_void_p

fsm_lib.migrate_constant_velocity.argtypes = [
    _c_void_p,  # data pointer (device)
    _c_void_p,  # cdp pointer (device)
    _c_void_p,  # offsets pointer (device)
    ctypes.c_float,  # v (velocity)
    ctypes.c_float,  # dt (time sampling interval)
    ctypes.c_float,  # dx (lateral sampling interval)
    ctypes.c_float,  # dz (depth sampling interval)
    ctypes.c_int,    # nsmp (number of time samples)
    ctypes.c_int,    # ntraces (number of traces)
    ctypes.c_int,    # nx (output image lateral dimension)
    ctypes.c_int,    # nz (output image depth dimension)
    _c_void_p        # output pointer R (device)
]
fsm_lib.migrate_constant_velocity.restype = None

# --------------------------
# Declare migrate_variable_velocity
# --------------------------
fsm_lib.init_cuda_with_mapped_host()

# Define the argument types for the function.
fsm_lib.migrate_variable_velocity.argtypes = [
    _c_void_p,  # const float* data (device)
    _c_void_p,  # const float* cdp (device)
    _c_void_p,  # const float* offsets (device)
    _c_void_p,  # const float* eikonal positions (device)
    _c_void_p,  # const int* segments (device)
    ctypes.c_float,    # float v
    ctypes.c_float,    # float dt
    ctypes.c_float,    # float dx_fine
    ctypes.c_float,    # float dz_fine
    ctypes.c_float,    # float dx_coarse
    ctypes.c_float,    # float dz_coarse
    ctypes.c_int,      # int nsmp
    ctypes.c_int,      # int ntraces
    ctypes.c_int,      # int nx_coarse
    ctypes.c_int,      # int nz_coarse
    ctypes.c_int,      # int nx_fine
    ctypes.c_int,      # int nz_fine
    _c_void_p,         # float* R (device)
    ctypes.c_char_p,   # const char* traveltime_filename
    ctypes.c_char_p,   # const char* gradient_filename
    ctypes.c_int,      # int num_segments
    ctypes.c_int       # int num_eikonals
]

# The function returns void.
fsm_lib.migrate_variable_velocity.restype = None

# --------------------------
# Declare free_eikonal function
# --------------------------
fsm_lib.free_eikonal.argtypes = [ctypes.POINTER(ctypes.c_float)]
fsm_lib.free_eikonal.restype = None


@nb.njit
def compute_trace_segments(src, rec):
    n = src.shape[0]
    if n == 0:
        return np.empty(0, dtype=np.int64)
    
    # Allocate an output array with maximum possible size (n)
    segs = np.empty(n, dtype=np.int64)
    seg_count = 0
    count = 1
    prev_src = src[0]
    prev_rec = rec[0]
    
    for i in range(1, n):
        # Check if both source and receiver are identical to previous.
        if src[i] == prev_src and rec[i] == prev_rec:
            count += 1
        else:
            segs[seg_count] = count
            seg_count += 1
            count = 1
            prev_src = src[i]
            prev_rec = rec[i]
    
    segs[seg_count] = count
    seg_count += 1
    
    # Return the array up to the number of segments found.
    return segs[:seg_count]


def compute_traveltime_field(Vp, sx, sz, dx, dz, nx, nz):

    Vp_trans = np.ascontiguousarray(Vp.T, dtype=np.float32)
    # Vp = np.asfortranarray(Vp, dtype=np.float32)
    result_ptr = fsm_lib.fast_sweeping_method(Vp_trans, sx, sz, dx, dz, nx, nz)
    result_view = np.ctypeslib.as_array(result_ptr, shape=(nz * nx,))
    traveltime = np.copy(result_view).reshape((nz, nx), order='F')  # or 'C' based on CUDA layout
    fsm_lib.free_eikonal(result_ptr)
    return traveltime

def free_gpu_memory(func):
    if cp is None:
        raise ImportError("cupy is required to manage GPU memory. Install openseismicprocessing with the 'gpu' extra to enable this feature.")

    def wrapper_func(*args, **kwargs):
        retval = func(*args, **kwargs)
        cp._default_memory_pool.free_all_blocks()
        return retval
    return wrapper_func

def resample_lanczos(input_array, dx_old, dy_old, dx_new, dy_new):
    """
    Resample a 2D array using Lanczos interpolation, and check that physical dimensions match.
    
    Parameters:
    - input_array: 2D array (shape [Nz, Nx])
    - dx_old, dy_old: Original grid spacing
    - dx_new, dy_new: New grid spacing
    
    Returns:
    - Resampled array (with shape based on physical size)
    """
    Nz, Nx = input_array.shape

    # Physical dimensions (in meters)
    dim_x = (Nx - 1) * dx_old
    dim_z = (Nz - 1) * dy_old

    # Expected new sizes to preserve physical dimensions
    Nx_new = int(round(dim_x / dx_new)) + 1
    Nz_new = int(round(dim_z / dy_new)) + 1

    # Calculate actual physical dimensions from new grid
    dim_x_new = (Nx_new - 1) * dx_new
    dim_z_new = (Nz_new - 1) * dy_new

    # Check if they match the original physical dimensions
    tol = 1e-3  # Tolerance in meters
    if abs(dim_x_new - dim_x) > tol or abs(dim_z_new - dim_z) > tol:
        print(f"[⚠️ Warning] New grid does not match original physical dimensions.")
        print(f"  Original: ({dim_z:.2f} m, {dim_x:.2f} m)")
        print(f"  New:      ({dim_z_new:.2f} m, {dim_x_new:.2f} m)")

        # Suggest corrected dx_new and dy_new
        dx_suggest = dim_x / (Nx_new - 1)
        dy_suggest = dim_z / (Nz_new - 1)

        print(f"[💡 Suggestion] To match physical size exactly, use:")
        print(f"  dx_new = {dx_suggest:.6f}")
        print(f"  dy_new = {dy_suggest:.6f}")

    # Perform the resampling using OpenCV (Lanczos)
    output_array = cv2.resize(input_array, (Nx_new, Nz_new), interpolation=cv2.INTER_LANCZOS4)

    return output_array

def collect_geometry_near_eikonal_points(df, begin, end, spacing=300):
    """
    Return a DataFrame of all traces where SourceX or GroupX is within `radius`
    of any reference point spaced every `spacing` meters.
    
    Adds two columns:
    - 'EikonalPosition': the x_ref reference point assigned to the row
    - 'DistanceToEikonal': the absolute distance (in meters) from that point
    
    The returned DataFrame preserves the original index.
    """


    reference_points = np.arange(begin, end + 1, spacing)
    return reference_points
    # selected_rows = []

    # for x_ref in reference_points:
    #     mask = ((df[columns[0]] - x_ref).abs() <= radius) | \
    #            ((df[columns[1]] - x_ref).abs() <= radius)

    #     if not mask.any():
    #         continue

    #     subset = df[mask].copy()  # retains original indices

    #     source_dist = (subset[columns[0]] - x_ref).abs()
    #     group_dist = (subset[columns[1]] - x_ref).abs()
    #     closest_dist = np.minimum(source_dist, group_dist)

    #     subset["EikonalPosition"] = x_ref
    #     subset["DistanceToEikonal"] = closest_dist

    #     selected_rows.append(subset)

    # if selected_rows:
    #     result_df = pd.concat(selected_rows, axis=0)
    #     return result_df
    # else:
    #     return pd.DataFrame(columns=list(df.columns) + ["EikonalPosition", "DistanceToEikonal"])




@nb.njit(parallel=True, fastmath=True)
def migrate_constant_velocity_numba(data, cdp_x, offsets, v, dx, dz, dt, nx, nz, aperture):
    """
    Optimized Kirchhoff migration for pre-stack data sorted by (CDP_X, offset),
    with NaN protection and numerical stability.

    Parameters:
        data : np.ndarray
            2D seismic data of shape (nsamples, ntraces)
        cdp_x : np.ndarray
            CDP x-location per trace (shape: ntraces,)
        offsets : np.ndarray
            Offset per trace (shape: ntraces,)
        v : float
            Constant velocity (scalar)
        dx, dz : float
            Horizontal and vertical sampling in the output image
        dt : float
            Time sampling interval
        nx, nz : int
            Output image dimensions in x and z

    Returns:
        R : np.ndarray
            Migrated image of shape (nx, nz)
    """
    nsmp, ntraces = data.shape
    R = np.zeros((nz, nx), dtype=np.float32)
    epsilon = 1e-10  # small number to avoid divide-by-zero

    for itrace in range(ntraces):
        cdp = cdp_x[itrace]
        h = offsets[itrace] * 0.5

        init_x = int(np.floor((cdp-aperture)/dx))
        end_x = int(np.ceil((cdp+aperture)/dx))

        percentage = (itrace+1)/ntraces * 100.

        if percentage % 10 == 0:
            print(percentage,"'%' pronto")

        if init_x < 0:
            init_x=0
        
        if end_x >= nx:
            end_x = nx
        
        if h < aperture:
            for iz in nb.prange(1,nz):
                z = iz * dz
                
                for ix in range(init_x,end_x):
                    x = ix * dx

                    dxs = x - (cdp - h)
                    dxg = x - (cdp + h)

                    rs = np.sqrt(dxs * dxs + z * z)
                    rr = np.sqrt(dxg * dxg + z * z)

                    # Stability fix
                    rs = max(rs, epsilon)
                    rr = max(rr, epsilon)

                    t = (rs + rr) / v
                    it = int(t / dt)

                    if 0 <= it < nsmp:
                        sqrt_rs_rr = np.sqrt(rs / rr)
                        sqrt_rr_rs = 1.0 / sqrt_rs_rr
                        wco = (z / rs * sqrt_rs_rr + z / rr * sqrt_rr_rs) / v

                        # if not np.isnan(wco):
                        R[iz, ix] -= data[it, itrace] * wco * 0.3989422804  #  1/sqrt(2π)

    return R




def compute_and_store_traveltime_fields(Vp, shot_positions, depth_positions, dx, dz, nx, nz, output_folder,output_filename):
    """
    Compute the traveltime field for each shot (or CDP) on a coarse grid,
    and store the results in a single NPY file.
    
    Parameters:
      Vp              : 2D numpy array representing the velocity model.
      shot_positions  : 1D numpy array of shot (or CDP) x positions (e.g., every 300 m).
      depth_positions : 1D numpy array of corresponding shot depths.
      dx, dz          : spatial sampling intervals for the coarse grid.
      nx, nz          : dimensions of the coarse traveltime grid.
      output_folder   : string path to the folder where the file will be stored.
      output_filename : string file name for the saved file (e.g., "tt_fields.npy").

    Returns:
      filepath: The full path to the saved NPY file.
    """
    nshots = shot_positions.shape[0]
    # Allocate array for traveltime.
    
    filepath = os.path.join(output_folder, output_filename)
    # Open the file in binary write mode once.
    file_traveltime = open(filepath, 'wb')

    for i in range(nshots):
        sx = shot_positions[i]
        sz = depth_positions[i]
        # Compute traveltime field for this shot.
        Traveltime = compute_traveltime_field(Vp, sx, sz, dx, dz, nx, nz)

        # Convert to a contiguous array of type float32.
        data = np.ascontiguousarray(Traveltime.astype(np.float32))
        file_traveltime.write(data.tobytes())
    
    return filepath

def compute_and_store_traveltime_derivatives_from_file(traveltime_filepath, s_coords, nz, nx, output_folder, output_filename):
    """
    Read the traveltime fields from a binary file and compute the shot-derivative (dT/ds)
    using finite differences. The derivative is computed for each shot on a coarse grid,
    and the results are written in binary format into a separate file.

    Parameters:
      traveltime_filepath: Path to the binary file containing traveltime fields.
                           The file is assumed to have data written shot-by-shot with shape (Ns, nz, nx).
      s_coords           : 1D numpy array of shot positions (length Ns).
      nz, nx             : Spatial dimensions of each traveltime field.
      output_folder      : Folder to store the output gradient file.
      output_filename    : Name of the gradient file (e.g. "tt_gradients.bin").

    Returns:
      filepath: The full path to the saved binary gradient file.
    """
    # Determine the number of shots from s_coords.
    Ns = s_coords.shape[0]
    
    # Load the traveltime data from file. It is stored as float32.
    data = np.fromfile(traveltime_filepath, dtype=np.float32)
    expected_elements = Ns * nz * nx
    if data.size != expected_elements:
        raise ValueError("Number of elements in the file (%d) does not match expected (%d)." % (data.size, expected_elements))
    
    # Reshape into a 3D array with dimensions (Ns, nz, nx).
    traveltimes = data.reshape((Ns, nz, nx))
    
    # Allocate an array for the derivative field (same shape).
    dT_ds = np.empty_like(traveltimes)
    
    # Compute the derivative along the shot axis:
    # For the first shot, use forward difference.
    dT_ds[0, :, :] = (traveltimes[1, :, :] - traveltimes[0, :, :]) / (s_coords[1] - s_coords[0])
    
    # For interior shots, use centered difference.
    for i in range(1, Ns - 1):
        dT_ds[i, :, :] = (traveltimes[i + 1, :, :] - traveltimes[i - 1, :, :]) / (s_coords[i + 1] - s_coords[i - 1])
    
    # For the last shot, use backward difference.
    dT_ds[-1, :, :] = (traveltimes[-1, :, :] - traveltimes[-2, :, :]) / (s_coords[-1] - s_coords[-2])
    
    # Save the derivative field to a separate binary file.
    gradient_filepath = os.path.join(output_folder, output_filename)
    with open(gradient_filepath, 'wb') as f:
        # Ensure the data is contiguous and of type float32, then write as bytes.
        f.write(np.ascontiguousarray(dT_ds.astype(np.float32)).tobytes())
    
    return gradient_filepath


def migrate_constant_velocity_cuda(data, cdp_x, offsets, v, dx, dz, dt, nx, nz):
    _require_cupy()
    """
    Fully vectorized GPU Kirchhoff migration using CuPy with trace‐batching to limit memory usage.
    
    Instead of processing all traces at once (which can exceed available GPU memory), the code
    processes traces in batches so that the maximum allocated memory does not surpass 4GB.
    
    Parameters:
      data    : np.ndarray
                2D seismic data on CPU of shape (nsmp, ntraces) (float32).
      cdp_x   : np.ndarray
                CDP X locations per trace (ntraces,).
      offsets : np.ndarray
                Offset per trace (ntraces,).
      v       : float
                Constant velocity.
      dx, dz, dt : float
                Spatial (lateral and depth) and time sampling intervals.
      nx, nz  : int
                Output image dimensions (lateral and depth samples).
                
    Returns:
      R       : np.ndarray
                Migrated image on CPU (shape: (nx, nz)).
    """

    nsmp, ntraces = data.shape

    # Ensure arrays are contiguous on the host in the expected order (trace-major)
    data_contig = np.ascontiguousarray(data.astype(np.float32).T)
    cdp_x_contig = np.ascontiguousarray(cdp_x.astype(np.float32))
    offsets_contig = np.ascontiguousarray(offsets.astype(np.float32))

    data_gpu = cp.asarray(data_contig).ravel()
    cdp_gpu = cp.asarray(cdp_x_contig).ravel()
    offsets_gpu = cp.asarray(offsets_contig).ravel()
    R_gpu = cp.zeros(nx * nz, dtype=cp.float32)

    # Launch kernel via shared library
    start_event = cp.cuda.Event()
    end_event = cp.cuda.Event()
    start_event.record()

    fsm_lib.migrate_constant_velocity(
        ctypes.c_void_p(int(data_gpu.data.ptr)),
        ctypes.c_void_p(int(cdp_gpu.data.ptr)),
        ctypes.c_void_p(int(offsets_gpu.data.ptr)),
        np.float32(v), np.float32(dt),
        np.float32(dx), np.float32(dz),
        ctypes.c_int(nsmp),
        ctypes.c_int(ntraces),
        ctypes.c_int(nx),
        ctypes.c_int(nz),
        ctypes.c_void_p(int(R_gpu.data.ptr)),
    )

    end_event.record()
    end_event.synchronize()
    elapsed_time_ms = cp.cuda.get_elapsed_time(start_event, end_event)
    print("Kernel execution time (ms):", elapsed_time_ms * 0.001)

    migrated_image = -cp.asnumpy(R_gpu).reshape((nz, nx))
    return migrated_image

if pipeline_def is not None and fn is not None:

    @pipeline_def(batch_size=1, num_threads=1, device_id=0)
    def pipe_traveltime(file_root, files, Nx, Nz):
        traveltime = fn.readers.numpy(device="gpu", file_root=file_root, files=files)

        traveltime_resized = dali.fn.resize(
            traveltime,
            size=[Nz, Nx],
            interp_type=dali.types.DALIInterpType.INTERP_LANCZOS3
        )

        return traveltime_resized
else:  # pragma: no cover - optional dependency

    def pipe_traveltime(*_args, **_kwargs):
        raise ImportError("nvidia.dali is required to build GPU traveltime pipelines. Install openseismicprocessing[gpu] to enable this feature.")


def migrate_variable_velocity_cuda(data, Geometry_Dataframe, segments, eikonal_positions, v, dx_fine, dz_fine, dx_coarse, dz_coarse, dt, nx_coarse, nz_coarse, 
                                   nx_fine, nz_fine, traveltime_path, gradient_path, key_cdp = "CDP_X",key_offset='offset'):
    _require_cupy()
    """
    Fully vectorized GPU Kirchhoff migration using CuPy with trace‐batching to limit memory usage.
    
    Instead of processing all traces at once (which can exceed available GPU memory), the code
    processes traces in batches so that the maximum allocated memory does not surpass 4GB.
    
    Parameters:
      data    : np.ndarray
                2D seismic data on CPU of shape (nsmp, ntraces) (float32).
      cdp_x   : np.ndarray
                CDP X locations per trace (ntraces,).
      offsets : np.ndarray
                Offset per trace (ntraces,).
      v       : float
                Constant velocity.
      dx, dz, dt : float
                Spatial (lateral and depth) and time sampling intervals.
      nx, nz  : int
                Output image dimensions (lateral and depth samples).
                
    Returns:
      R       : np.ndarray
                Migrated image on CPU (shape: (nx, nz)).
    """
    
    # pipe = pipe_traveltime(npy_folder,npy_filenames, nx, nz)
    # pipe.build()
    
    nsmp, ntraces = data.shape

    cdp_x = Geometry_Dataframe[key_cdp].to_numpy()
    offsets = Geometry_Dataframe[key_offset].to_numpy()

    # Ensure arrays are contiguous:
    data_contig = np.ascontiguousarray(data.astype(np.float32).T)
    cdp_x_contig = np.ascontiguousarray(cdp_x.astype(np.float32))
    offsets_contig = np.ascontiguousarray(offsets.astype(np.float32))
    eikonal_positions_contig = np.ascontiguousarray(eikonal_positions.astype(np.float32))
    num_segments = len(segments)
    segments_contig = np.ascontiguousarray(segments.astype(np.int32))

    data_gpu = cp.asarray(data_contig).ravel()
    cdp_gpu = cp.asarray(cdp_x_contig).ravel()
    offsets_gpu = cp.asarray(offsets_contig).ravel()
    eikonal_gpu = cp.asarray(eikonal_positions_contig).ravel()
    segments_gpu = cp.asarray(segments_contig).ravel()
    R_gpu = cp.zeros(nx_fine * nz_fine, dtype=cp.float32)

    # Convert to bytes for ctypes (c_char_p expects bytes)
    clean_path = traveltime_path.strip()
    # Convert to bytes for ctypes (c_char_p expects a null-terminated byte string)
    traveltime_path_bytes = clean_path.encode('utf-8')

    # Convert to bytes for ctypes (c_char_p expects bytes)
    clean_path = gradient_path.strip()
    # Convert to bytes for ctypes (c_char_p expects a null-terminated byte string)
    gradient_path_bytes = clean_path.encode('utf-8')

    # fsm_lib.init_cuda_with_mapped_host()

    fsm_lib.migrate_variable_velocity(
        ctypes.c_void_p(int(data_gpu.data.ptr)),
        ctypes.c_void_p(int(cdp_gpu.data.ptr)),
        ctypes.c_void_p(int(offsets_gpu.data.ptr)),
        ctypes.c_void_p(int(eikonal_gpu.data.ptr)),
        ctypes.c_void_p(int(segments_gpu.data.ptr)),
        np.float32(v), np.float32(dt),
        np.float32(dx_fine), np.float32(dz_fine),
        np.float32(dx_coarse), np.float32(dz_coarse),
        ctypes.c_int(nsmp),
        ctypes.c_int(ntraces),
        ctypes.c_int(nx_coarse), ctypes.c_int(nz_coarse),
        ctypes.c_int(nx_fine), ctypes.c_int(nz_fine),
        ctypes.c_void_p(int(R_gpu.data.ptr)),
        traveltime_path_bytes,
        gradient_path_bytes,
        ctypes.c_int(num_segments),
        ctypes.c_int(len(eikonal_positions)),
    )

    migrated_image = -cp.asnumpy(R_gpu).reshape((nz_fine, nx_fine))
    cp.get_default_memory_pool().free_all_blocks()

    return migrated_image


@nb.njit(parallel=True, fastmath=True)
def _migrate_trace_local(data_trace, it_field, valid_mask, cdp, h, dx_out, dz_out, Vp, nx, nz):
    """
    Compute the contribution of a single trace using the provided fine-grid time sample index field.
    
    Parameters:
      data_trace : 1D array (nsmp,) of seismic trace amplitudes.
      it_field   : 2D int array (nz, nx) of time sample indices (note: here rows = depth, cols = lateral)
      valid_mask : 2D bool array (nz, nx) indicating which indices are valid.
      cdp        : float, effective common depth point (computed as (SourceX+GroupX)/2).
      h          : float, half-offset for this trace.
      dx_out, dz_out : float, output grid sampling intervals.
      Vp         : 2D array (nz, nx) of local velocities on the output grid.
      nx, nz     : ints, dimensions of the output image.
      
    Returns:
      R_trace    : 2D float array (nz, nx) representing the trace’s contribution.
    """
    R_trace = np.zeros((nz, nx), dtype=np.float32)
    sqrt2pi = 1.0 / np.sqrt(2.0 * np.pi)
    for iz in nb.prange(nz):
        z_val = iz * dz_out
        for ix in range(nx):
            if valid_mask[iz, ix]:
                x_val = ix * dx_out
                it = it_field[iz, ix]
                if it >= 0 and it < data_trace.shape[0]:
                    rs = np.sqrt((x_val - (cdp - h))**2 + z_val**2)
                    rr = np.sqrt((x_val - (cdp + h))**2 + z_val**2)
                    if rs < 1e-10:
                        rs = 1e-10
                    if rr < 1e-10:
                        rr = 1e-10
                    # local_v = Vp[iz, ix]
                    weight = ((z_val / rs) * np.sqrt(rs / rr) + (z_val / rr) * np.sqrt(rr / rs)) #/ local_v
                    weight *= sqrt2pi
                    R_trace[iz, ix] += data_trace[it] * weight
    return R_trace

def migrate_kirchhoff(data, geometry, Vp, image_dims,
                                                dx_model, dz_model, dx_output, dz_output, dt,
                                                unique_positions, traveltime_mmap):
    """
    Perform Kirchhoff migration using precomputed traveltime fields (and derivatives) for both source and receiver.
    
    The precomputed traveltime fields are stored in a single dictionary tt_eikonal_dict,
    computed at positions given by eikonal_positions. For each trace, the traveltime field
    for the source (based on SourceX) and for the receiver (based on GroupX) are retrieved
    directly (since every possible position was computed exactly). Their interpolated values
    are then summed to yield the total traveltime.
    
    Parameters:
      data          : 2D seismic data, shape (nsmp, ntraces)
      geometry      : structured array or DataFrame with 'SourceX', 'GroupX', 'CDP_X', and 'offset'
      Vp            : 2D velocity model on the output grid, shape (nz, nx) [rows=depth, cols=lateral]
      image_dims    : tuple (nx, nz) for the migrated image (fine grid)
      dx_model, dz_model : spacing for the coarse model grid (used in traveltime precomputation)
      dx_output, dz_output : spacing for the output (fine) grid
      dt            : time sampling interval
      tt_eikonal_dict: dict mapping positions (floats) to a tuple (T_coarse, dT_dx, dT_dz)
      eikonal_positions: 1D numpy array of positions at which traveltime fields were computed
      
    Returns:
      R             : Migrated image of shape (nz, nx)
    """
    # Vp is assumed to be defined on the fine grid with shape (nz, nx).
    nz, nx = Vp.shape
    nx_image, nz_image = image_dims  # (nx_image, nz_image) should match (nx, nz)
    nsmp, ntraces = data.shape
    R = np.zeros((nz_image, nx_image), dtype=np.float32)  # migrated image: (nz, nx)

    for itrace in range(ntraces):
        sx = geometry['SourceX'][itrace]
        gx = geometry['GroupX'][itrace]
        h = geometry['offset'][itrace] * 0.5
        cdp = 0.5 * (sx + gx)
        
        # Directly index the precomputed traveltime dictionary:
        index = np.argmin(np.abs(unique_positions - sx))
        T_coarse_src = traveltime_mmap[index, :, :]
        T_coarse_rec = traveltime_mmap[index, :, :]
        
        T_source_fine = resample_lanczos(T_coarse_src, dx_output, dz_output, dx_model, dz_model)
        T_receiver_fine = resample_lanczos(T_coarse_rec, dx_output, dz_output, dx_model, dz_model)
        
        # Total traveltime is the sum.
        tt_field = T_source_fine + T_receiver_fine
        
        # Convert total traveltime to time sample indices.
        it_field = np.floor(tt_field / dt).astype(np.int32)
        valid_mask = (it_field >= 0) & (it_field < nsmp)
        
        data_trace = data[:, itrace]
        
        # Compute migration contribution from this trace using the Numba-accelerated inner loop.
        R_trace = _migrate_trace_local(data_trace, it_field, valid_mask,
                                       cdp, h, dx_output, dz_output, Vp, nx_image, nz_image)
        R += R_trace

    return R

# kirchhoff_2d_flat_nz_nx.py
# 2D Kirchhoff migration using FSM (coarse) + paraxial (fine)
# SHAPES: all fields (velocity, T, px, pz, H*) are (nz, nx). Data is (nt, ntr).

import cupy as cp
from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Dict

# ---------------- Grid & small utils ----------------

@dataclass
class Grid2D:
    x0: float; z0: float; dx: float; dz: float; nx: int; nz: int

def assert_nz_nx(A: cp.ndarray, grid: Grid2D, name: str):
    assert A.shape == (grid.nz, grid.nx), f"{name} must be (nz,nx); got {A.shape}"

def gaussian_blur3(a: cp.ndarray) -> cp.ndarray:
    out = a.astype(cp.float32, copy=True)
    tmp = cp.empty_like(out)
    tmp[1:-1, :] = (out[:-2, :] + out[1:-1, :] + out[2:, :]) / 3.0
    tmp[0,     :] = (out[0, :] + out[1, :]) / 2.0
    tmp[-1,    :] = (out[-2, :] + out[-1, :]) / 2.0
    out[:, 1:-1] = (tmp[:, :-2] + tmp[:, 1:-1] + tmp[:, 2:]) / 3.0
    out[:, 0]    = (tmp[:, 0] + tmp[:, 1]) / 2.0
    out[:, -1]   = (tmp[:, -2] + tmp[:, -1]) / 2.0
    return out

def downsample_avg(a: cp.ndarray, sx: int, sz: int) -> cp.ndarray:
    nz, nx = a.shape
    nz_c, nx_c = nz//sz, nx//sx
    a = a[:nz_c*sz, :nx_c*sx]
    return a.reshape(nz_c, sz, nx_c, sx).mean(axis=(1,3))

# ---------------- Derivatives & Hessian (nz,nx) ----------------

def _cdx(f: cp.ndarray, dx: float) -> cp.ndarray:
    o = cp.empty_like(f)
    o[:,1:-1] = (f[:,2:] - f[:,:-2])/(2*dx)
    o[:,0]    = (f[:,1] - f[:,0])/dx
    o[:,-1]   = (f[:,-1]- f[:,-2])/dx
    return o

def _cdz(f: cp.ndarray, dz: float) -> cp.ndarray:
    o = cp.empty_like(f)
    o[1:-1,:] = (f[2:,:] - f[:-2,:])/(2*dz)
    o[0,   :] = (f[1,:]  - f[0 ,:])/dz
    o[-1,  :] = (f[-1,:] - f[-2,:])/dz
    return o

def _osx(f: cp.ndarray, dx: float, forward=True) -> cp.ndarray:
    o = cp.empty_like(f)
    if forward:
        o[:, :-1] = (f[:,1:] - f[:,:-1])/dx; o[:,-1] = (f[:,-1]-f[:,-2])/dx
    else:
        o[:, 1:]  = (f[:,1:] - f[:,:-1])/dx; o[:, 0]  = (f[:,1]-f[:,0])/dx
    return o

def _osz(f: cp.ndarray, dz: float, forward=True) -> cp.ndarray:
    o = cp.empty_like(f)
    if forward:
        o[:-1,:] = (f[1:,:] - f[:-1,:])/dz; o[-1,:] = (f[-1,:]-f[-2,:])/dz
    else:
        o[1: ,:] = (f[1:,:] - f[:-1,:])/dz; o[0 ,:] = (f[1,:] - f[0 ,:])/dz
    return o

def estimate_gradients_upwind(Tc: cp.ndarray, grid: Grid2D):
    # upwind-aware ∂T/∂x and ∂T/∂z
    dxf = _osx(Tc, grid.dx, True); dxb = _osx(Tc, grid.dx, False)
    px  = cp.where(cp.abs(dxb) <= cp.abs(dxf), dxb, dxf)
    dzf = _osz(Tc, grid.dz, True); dzb = _osz(Tc, grid.dz, False)
    pz  = cp.where(cp.abs(dzb) <= cp.abs(dzf), dzb, dzf)
    # gentle blend with centered where smooth
    cx, cz = _cdx(Tc, grid.dx), _cdz(Tc, grid.dz)
    smooth = (cp.abs(cx-px) < 0.25*cp.abs(px)+1e-9) & (cp.abs(cz-pz) < 0.25*cp.abs(pz)+1e-9)
    px = cp.where(smooth, 0.5*(px+cx), px)
    pz = cp.where(smooth, 0.5*(pz+cz), pz)
    return px, pz

def build_hessian(px: cp.ndarray, pz: cp.ndarray, grid: Grid2D):
    H11 = _cdx(px, grid.dx)
    H22 = _cdz(pz, grid.dz)
    Hxz = _cdz(px, grid.dz)
    Hzx = _cdx(pz, grid.dx)
    H12 = 0.5*(Hxz + Hzx)
    # light stabilization on H (NOT on T)
    H11 = gaussian_blur3(H11); H22 = gaussian_blur3(H22); H12 = gaussian_blur3(H12)
    return H11, H22, H12

# ---------------- Paraxial field (nz,nx) ----------------

@dataclass
class ParaxialField2D:
    grid: Grid2D
    T:  cp.ndarray   # (nz,nx)
    px: cp.ndarray   # (nz,nx)
    pz: cp.ndarray   # (nz,nx)
    H11: cp.ndarray  # (nz,nx)
    H22: cp.ndarray  # (nz,nx)
    H12: cp.ndarray  # (nz,nx)

    @staticmethod
    def from_T(Tc: cp.ndarray, grid: Grid2D, px: Optional[cp.ndarray]=None, pz: Optional[cp.ndarray]=None):
        assert_nz_nx(Tc, grid, "Tc")
        if px is None or pz is None:
            px, pz = estimate_gradients_upwind(Tc, grid)
        H11, H22, H12 = build_hessian(px, pz, grid)
        return ParaxialField2D(grid, Tc, px, pz, H11, H22, H12)

def _unit_g(px, pz):
    n = cp.sqrt(px*px + pz*pz) + 1e-12
    return px/n, pz/n, n

def paraxial_eval(field: ParaxialField2D, xq: cp.ndarray, zq: cp.ndarray):
    g = field.grid
    ix = cp.rint((xq - g.x0)/g.dx).astype(cp.int32)
    iz = cp.rint((zq - g.z0)/g.dz).astype(cp.int32)
    ix = cp.clip(ix, 1, g.nx-2); iz = cp.clip(iz, 1, g.nz-2)
    x0 = g.x0 + ix.astype(xq.dtype)*g.dx
    z0 = g.z0 + iz.astype(zq.dtype)*g.dz
    dx = xq - x0; dz = zq - z0
    T0  = field.T [iz, ix]
    px0 = field.px[iz, ix]
    pz0 = field.pz[iz, ix]
    H11 = field.H11[iz, ix]
    H22 = field.H22[iz, ix]
    H12 = field.H12[iz, ix]
    Tq = T0 + (px0*dx + pz0*dz) + 0.5*(H11*dx*dx + 2.0*H12*dx*dz + H22*dz*dz)
    gx, gz, _ = _unit_g(px0, pz0)
    kappa = (-gz)*(H11*(-gz) + H12*gx) + gx*(H12*(-gz) + H22*gx)  # t=[-gz, gx]
    ghat = cp.stack([gx, gz], axis=-1)
    return Tq, ghat, kappa

def true_amp_weight_2d(cos_s, cos_r, kappa_s, kappa_r, eps=1e-6):
    return cp.sqrt((cp.abs(cos_s)*cp.abs(cos_r)) / cp.maximum(cp.abs(kappa_s + kappa_r), eps))

def prepare_coarse_velocity(v_fine: cp.ndarray, grid_f: Grid2D, dx_c: float, dz_c: float, smooth_sigma_cells=0.75):
    assert_nz_nx(v_fine, grid_f, "v_fine")
    sx = max(1, int(round(dx_c / grid_f.dx)))
    sz = max(1, int(round(dz_c / grid_f.dz)))
    v_avg = cp.asarray(resample_lanczos(cp.asnumpy(v_fine), grid_f.dx, grid_f.dz, dx_c, dz_c))
    # if smooth_sigma_cells > 0: v_avg = gaussian_blur3(v_avg)
    grid_c = Grid2D(grid_f.x0, grid_f.z0, sx*grid_f.dx, sz*grid_f.dz, v_avg.shape[1], v_avg.shape[0])
    return v_avg, grid_c

def interp_linear(trace: cp.ndarray, t: cp.ndarray, dt: float, t0: float=0.0) -> cp.ndarray:
    idx = (t - t0)/dt
    i0 = cp.floor(idx).astype(cp.int32)
    w  = idx - i0
    i0 = cp.clip(i0, 0, trace.size-2)
    return (1-w)*trace[i0] + w*trace[i0+1]

# ---------------- Flat-geometry Kirchhoff (data = (nt, ntr)) ----------------

def kirchhoff_2d_flat(
    data: cp.ndarray,           # (nt, ntr)
    src_x: cp.ndarray,          # (ntr,)
    rec_x: cp.ndarray,          # (ntr,)
    dt: float, t0: float,
    grid_f: Grid2D,             # fine imaging grid (nz,nx)
    v_fine: cp.ndarray,         # (nz,nx)
    fsm_solver: Callable[[cp.ndarray, Grid2D, Tuple[float,float]], cp.ndarray],
    rec_z: float = 0.0,
    dx_coarse: float = 300.0, dz_coarse: float = 300.0,
    use_true_amplitude: bool = True,
    aperture_deg: float = 70.0,
    batch_pixels: int = 250_000
) -> cp.ndarray:
    nt, ntr = data.shape
    assert src_x.shape == (ntr,) and rec_x.shape == (ntr,), "src_x/rec_x must be (ntr,)"
    assert_nz_nx(v_fine, grid_f, "v_fine")

    v_coarse, grid_c = prepare_coarse_velocity(v_fine, grid_f, dx_coarse, dz_coarse, 0.75)

    xs = grid_f.x0 + cp.arange(grid_f.nx, dtype=cp.float32)*grid_f.dx
    zs = grid_f.z0 + cp.arange(grid_f.nz, dtype=cp.float32)*grid_f.dz
    Xf, Zf = cp.meshgrid(xs, zs)  # -> (nz,nx)
    xf, zf = Xf.ravel(), Zf.ravel()
    N = xf.size

    cos_cut = cp.cos(cp.deg2rad(aperture_deg))
    img = cp.zeros(N, dtype=cp.float32)

    # caches
    src_cache: Dict[float, ParaxialField2D] = {}
    rec_cache: Dict[float, ParaxialField2D] = {}

    # group traces by source x (CPU for simplicity)
    sx_np = cp.asnumpy(src_x); rx_np = cp.asnumpy(rec_x)
    import numpy as np
    uniq_sx, inv = np.unique(sx_np, return_inverse=True)
    traces_by_src = [np.where(inv==k)[0] for k in range(uniq_sx.size)]

    for sx_val, idxs in zip(uniq_sx, traces_by_src):
        if sx_val not in src_cache:
            Tc = fsm_solver(v_coarse, grid_c, (float(sx_val), 0.0))
            assert Tc.shape == (grid_c.nz, grid_c.nx), "fsm_solver must return (nz,nx)"
            src_cache[sx_val] = ParaxialField2D.from_T(Tc, grid_c)
        Fs = src_cache[sx_val]

        uniq_rx = np.unique(rx_np[idxs])
        for xr_val in uniq_rx:
            if xr_val not in rec_cache:
                Tr = fsm_solver(v_coarse, grid_c, (float(xr_val), float(rec_z)))
                assert Tr.shape == (grid_c.nz, grid_c.nx), "fsm_solver must return (nz,nx)"
                rec_cache[xr_val] = ParaxialField2D.from_T(Tr, grid_c)

        k0 = 0
        while k0 < N:
            k1 = min(N, k0 + batch_pixels)
            xq, zq = xf[k0:k1], zf[k0:k1]

            Ts, ghat_s, kappa_s = paraxial_eval(Fs, xq, zq)
            cos_s = cp.abs(ghat_s[:,1])   # |gz|
            mask_s = (cos_s >= cos_cut)

            acc = cp.zeros_like(xq, dtype=cp.float32)

            for tr in idxs:
                xr_val = rx_np[tr]
                Fr = rec_cache[xr_val]
                Trv, ghat_r, kappa_r = paraxial_eval(Fr, xq, zq)
                cos_r = cp.abs(ghat_r[:,1])
                mask = mask_s & (cos_r >= cos_cut)
                if not cp.any(mask): continue

                tau = Ts + Trv
                trace = data[:, tr]  # (nt,)
                samp = interp_linear(trace, tau, dt, t0)

                if use_true_amplitude:
                    w = true_amp_weight_2d(cos_s, cos_r, kappa_s, kappa_r, 1e-6)
                else:
                    w = cp.sqrt(cos_s * cos_r)

                acc += w * samp * mask

            img[k0:k1] += acc
            k0 = k1

    return img.reshape(grid_f.nz, grid_f.nx)
