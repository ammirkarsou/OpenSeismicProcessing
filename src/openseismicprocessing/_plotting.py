import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
import pandas as pd
import io
import base64
from IPython.display import HTML, display
from matplotlib.widgets import Slider

def plot_acquisition(
    context: dict,
    key_acquisition_geometry: str = 'acquisition geometry',
    key_model_geometry: str = 'model geometry',
    dataset_acquisition: str | None = None,
    dataset_model: str | None = None,
):
    """
    Plots the acquisition geometry (sources and receivers) and, if available,
    overlays the model geometry as a polygon. Geometry can be supplied either
    as pandas DataFrames stored inside ``context`` or as dictionaries of header
    arrays (for example, the output of ``load_zarr_datasets`` or
    ``slice_zarr_by_expression``).

    Parameters
    ----------
    context : dict
        Global context that should contain the geometry data. When
        ``dataset_acquisition`` or ``dataset_model`` is provided, the
        corresponding entry in ``context`` must map headers ('SourceX', etc.)
        to array-like objects (NumPy or Zarr arrays).
    key_acquisition_geometry : str
        Fallback context key containing an acquisition DataFrame. Used when
        ``dataset_acquisition`` is not specified.
    key_model_geometry : str
        Fallback context key containing a model-geology DataFrame. Used when
        ``dataset_model`` is not specified.
    dataset_acquisition : str, optional
        Context key pointing to a dictionary of header arrays extracted from a
        Zarr store. When provided, it takes precedence over
        ``key_acquisition_geometry``.
    dataset_model : str, optional
        Context key pointing to a dictionary of header arrays for the model
        geometry. When ``None`` the model overlay is omitted unless a DataFrame
        exists under ``key_model_geometry``.
    """
    # Validate context
    if context is None or not isinstance(context, dict):
        print("❌ Error: Invalid context provided.")
        return

    sx_title = "SourceX"
    sy_title = "SourceY"
    gx_title = "GroupX"
    gy_title = "GroupY"
    required_headers = [sx_title, sy_title, gx_title, gy_title]

    def _materialize(value, key_name):
        if value is None:
            return None
        if isinstance(value, pd.DataFrame):
            return value
        if isinstance(value, dict):
            data = {}
            for header in required_headers:
                if header not in value:
                    continue
                column = value[header]
                if hasattr(column, 'oindex') and hasattr(column, '__len__'):
                    data[header] = np.asarray(column[:])
                else:
                    try:
                        data[header] = np.asarray(column)
                    except Exception:
                        raise ValueError(f"Unable to convert header '{header}' from dataset '{key_name}'")
            if data:
                return pd.DataFrame(data)
            return None
        raise ValueError(f"Unsupported geometry type for key '{key_name}'")

    if dataset_acquisition:
        acq_obj = context.get(dataset_acquisition)
    else:
        acq_obj = context.get(key_acquisition_geometry)

    if dataset_model:
        model_obj = context.get(dataset_model)
    else:
        model_obj = context.get(key_model_geometry)

    acq_df = _materialize(acq_obj, dataset_acquisition or key_acquisition_geometry)
    model_df = _materialize(model_obj, dataset_model or key_model_geometry)

    # Check if at least one of the geometry dataframes is available
    if acq_df is None and model_df is None:
        print("❌ Error: Neither acquisition geometry nor model geometry found in context.")
        return

    plt.figure()

    # Plot acquisition geometry if provided and valid
    if acq_df is not None:
        if not isinstance(acq_df, pd.DataFrame):
            print("❌ Error: Acquisition geometry is not a valid DataFrame.")
        else:
            missing_acq = [col for col in [sx_title, sy_title, gx_title, gy_title] if col not in acq_df.columns]
            if missing_acq:
                print(f"❌ Error: The following required acquisition columns are missing: {missing_acq}")
            else:
                plt.plot(acq_df[sx_title], acq_df[sy_title], 'r*', label='Sources')
                plt.plot(acq_df[gx_title], acq_df[gy_title], 'b.', label='Receivers')

    # Plot model geometry if provided and valid
    if model_df is not None:
        if not isinstance(model_df, pd.DataFrame):
            print("❌ Error: Model geometry is not a valid DataFrame.")
        else:
            missing_model = [col for col in [sx_title, sy_title] if col not in model_df.columns]
            if missing_model:
                print(f"❌ Error: The following required model geometry columns are missing: {missing_model}")
            else:
                # Compute corner coordinates for a rectangle (polygon)
                corner_x = [np.min(model_df[sx_title]), np.max(model_df[sx_title]),
                            np.min(model_df[sx_title]), np.max(model_df[sx_title])]
                corner_y = [np.min(model_df[sy_title]), np.min(model_df[sy_title]),
                            np.max(model_df[sy_title]), np.max(model_df[sy_title])]
                # Plot the four edges of the rectangle
                plt.plot([corner_x[0], corner_x[1]], [corner_y[0], corner_y[1]], 'k-', label='Model Polygon')
                plt.plot([corner_x[0], corner_x[2]], [corner_y[0], corner_y[2]], 'k-')
                plt.plot([corner_x[1], corner_x[3]], [corner_y[1], corner_y[3]], 'k-')
                plt.plot([corner_x[2], corner_x[3]], [corner_y[2], corner_y[3]], 'k-')

    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.title("Acquisition and Model Geometry")
    plt.legend()
    # display_centered(plt.gcf())
    # plt.tight_layout()
    try:
        mgr = plt.get_current_fig_manager()
        # Get screen width/height
        w = mgr.window.winfo_screenwidth()
        h = mgr.window.winfo_screenheight()

        # Get figure width/height
        win_w = 800
        win_h = 600
        x = int((w - win_w) / 2)
        y = int((h - win_h) / 2)

        mgr.window.geometry(f"{win_w}x{win_h}+{x}+{y}")
    except Exception as e:
        print("⚠️ Could not center window:", e)

    plt.show()
    

def display_centered(fig):
    """
    Saves the given matplotlib figure to a PNG in memory,
    then displays it centered in the Jupyter Notebook output cell.
    """
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    html = f'<div style="text-align: center;"><img src="data:image/png;base64,{img_b64}"/></div>'
    display(HTML(html))

def plot_seismic_image(context, xlabel, ylabel, y_spacing, x_header, perc=None,
                       key_data="data", key_geometry="geometry",
                       xlim=None, ylim=None, figure_dims=(7,9), cmap='gray_r'):
    """
    Plots a seismic image based on provided data and geometry.
    
    Parameters:
        context (dict): Contains the data and geometry DataFrame.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        y_spacing (int or float): Spacing multiplier for y-axis.
        x_header (str): Column name in geometry DataFrame for x-axis values.
        perc (int or float, optional): Percentile value for symmetric clipping.
                                       If not provided, vmin and vmax are set to the min and max of the data.
        key_data (str): Key in context for the seismic data (default "data").
        key_geometry (str): Key in context for the geometry DataFrame (default "geometry").
        xlim (tuple, optional): Limits for x-axis.
        ylim (tuple, optional): Limits for y-axis.
        figure_dims (tuple): Figure dimensions (width, height) for the plot.
        cmap (str): Colormap for displaying the seismic image.
        
    Returns:
        None. Displays the seismic image.
    """
    def _get_amplitude(data_obj, key_name):
        if data_obj is None:
            return None
        if isinstance(data_obj, dict):
            amp = data_obj.get("amplitude")
            if amp is None:
                raise ValueError(f"Dictionary '{key_name}' does not contain an 'amplitude' entry.")
            return amp
        return data_obj

    def _materialize_geometry(geom_obj, key_name):
        if geom_obj is None:
            return None
        if isinstance(geom_obj, pd.DataFrame):
            return geom_obj
        if isinstance(geom_obj, dict):
            required = [x_header, "SourceX", "SourceY", "GroupX", "GroupY"]
            data = {}
            for header in required:
                if header not in geom_obj:
                    continue
                column = geom_obj[header]
                if hasattr(column, "oindex") and hasattr(column, "__len__"):
                    data[header] = np.asarray(column[:])
                else:
                    data[header] = np.asarray(column)
            if not data:
                raise ValueError(f"Cannot build geometry DataFrame from '{key_name}'.")
            return pd.DataFrame(data)
        raise ValueError(f"Unsupported geometry type for key '{key_name}'.")

    data_obj = context.get(key_data)
    geom_obj = context.get(key_geometry)

    data = _get_amplitude(data_obj, key_data)
    df = _materialize_geometry(geom_obj, key_geometry) if geom_obj is not None else None

    if data is None or not hasattr(data, "shape"):
        if hasattr(data, "oindex"):
            data = data[:]
        else:
            print("❌ Error: 'data' not found or invalid in context.")
            return

    if not isinstance(y_spacing, (int, float)) or y_spacing <= 0:
        print(f"❌ Error: y_spacing must be a positive number. Got: {y_spacing}")
        return

    # Verify the colormap is valid
    if cmap not in plt.colormaps():
        print(f"❌ Error: Colormap '{cmap}' is not valid. Using default 'gray_r'.")
        cmap = 'gray_r'

    try:
        if hasattr(data, "oindex"):
            data = data[:]

        if data.ndim == 2:
            if df is None or not isinstance(df, pd.DataFrame):
                print("❌ Error: 'geometry' not found or invalid in context.")
                return

            if x_header not in df.columns:
                print(f"❌ Error: Column '{x_header}' not found in geometry.")
                return

            num_samples, num_traces = data.shape
            x_values = df[x_header].to_numpy()

            if len(x_values) != num_traces:
                print("❌ Error: Length of x_header values does not match number of traces in data.")
                return

            y_values = np.arange(num_samples) * y_spacing
            extent = [x_values[0], x_values[-1], y_values[-1], y_values[0]]

            if perc is not None and isinstance(perc, (int, float)) and perc > 0:
                clip_value = np.percentile(data, perc)
                vmin = -clip_value
                vmax = clip_value
            else:
                vmin = np.min(data)
                vmax = np.max(data)

            plt.figure(figsize=figure_dims)
            plt.imshow(data, aspect='auto', cmap=cmap, 
                       vmin=vmin, vmax=vmax, extent=extent, interpolation="none")
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.title("Seismic Image")
            plt.colorbar(label="Amplitude")

            if xlim:
                plt.xlim(xlim)
            if ylim:
                plt.ylim(ylim)

            plt.tight_layout()
            # display_centered(plt.gcf())
            plt.show()

        elif data.ndim == 1:
            y = np.arange(len(data)) * y_spacing

            plt.figure(figsize=figure_dims)
            plt.plot(data, y, color='black', linewidth=1)
            plt.gca().invert_yaxis()  # Time increases downward
            plt.xlabel("Amplitude")
            plt.ylabel(ylabel)
            plt.title("Stacked Seismic Trace")
            plt.grid(True)

            if ylim:
                plt.ylim(ylim[::-1])  # Reverse because time is vertical

            plt.tight_layout()
            # display_centered(plt.gcf())
            plt.show()
        else:
            print(f"❌ Error: 'data' must be 1D or 2D. Got shape: {data.shape}")

    except Exception as e:
        print(f"❌ Failed to plot seismic data: {e}")

def plot_seismic_image_interactive(context, xlabel, ylabel, y_spacing, x_header, sort_header, perc=None, 
                                   key_data="data", key_geometry="geometry", 
                                   xlim=None, ylim=None, figure_dims=(7,9), cmap='gray_r'):
    """
    Plots a seismic image interactively. The function uses a slider to let the user scroll
    through each unique value of sort_header from the geometry DataFrame. For each slider position,
    the seismic data is filtered to show only the traces corresponding to that unique value.
    
    Parameters:
        context (dict): Contains seismic data (2D NumPy array) and a geometry DataFrame.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        y_spacing (float): Spacing multiplier for the y-axis.
        x_header (str): Column in the geometry DataFrame used for x-axis values.
        sort_header (str): Column in the geometry DataFrame used for filtering the traces.
        perc (float, optional): Percentile for symmetric clipping. If provided, vmin and vmax
                                are set to ±clip_value. Otherwise, min and max of the data are used.
        key_data (str): Key in context for seismic data (default "data").
        key_geometry (str): Key in context for the geometry DataFrame (default "geometry").
        xlim (tuple, optional): x-axis limits.
        ylim (tuple, optional): y-axis limits.
        figure_dims (tuple): Figure dimensions (width, height).
        cmap (str): Colormap to use (must be a valid colormap for plt.imshow).
        
    Returns:
        None. Opens an interactive window.
    """
    # Retrieve data and geometry.
    data = context.get(key_data)
    df = context.get(key_geometry)
    
    if data is None or not hasattr(data, "shape"):
        print("❌ Error: 'data' not found or invalid in context.")
        return
    if df is None or not isinstance(df, pd.DataFrame):
        print("❌ Error: 'geometry' not found or invalid in context.")
        return
    if x_header not in df.columns:
        print(f"❌ Error: Column '{x_header}' not found in geometry.")
        return
    if sort_header not in df.columns:
        print(f"❌ Error: sort_header '{sort_header}' not found in geometry.")
        return
    
    # Get the unique sorted values for the sort header.
    unique_vals = np.sort(df[sort_header].unique())
    
    # Create the figure and main axis.
    fig, ax = plt.subplots(figsize=figure_dims)
    plt.subplots_adjust(bottom=0.15)  # leave space for the slider.
    
    # Create a slider below the main axis.
    slider_ax = fig.add_axes([0.15, 0.05, 0.7, 0.03])
    slider = Slider(slider_ax, sort_header, 0, len(unique_vals)-1, valinit=0, valfmt='%0.0f')
    
    def update_plot(val):
        idx = int(slider.val)
        current_val = unique_vals[idx]
        print(f"Showing {sort_header} = {current_val}")
        
        # Filter the geometry for rows matching the current value.
        sub_df = df[df[sort_header] == current_val]
        if sub_df.empty:
            print("No traces for this value.")
            ax.clear()
            fig.canvas.draw_idle()
            return
        
        # Assume that each column of data corresponds to a row in sub_df.
        trace_indices = sub_df.index.to_numpy()
        sub_data = data[:, trace_indices]
        num_samples, num_traces = sub_data.shape
        
        # Get x-values for the traces from the geometry.
        x_values = sub_df[x_header].to_numpy()
        if len(x_values) != num_traces:
            print("Mismatch in number of traces and x-values.")
            return
        
        # Construct y-values based on sample index and y_spacing.
        y_values = np.arange(num_samples) * y_spacing
        
        # Optionally, you could create an extent. If not, imshow will use array indices.
        # extent = [x_values[0], x_values[-1], y_values[-1], y_values[0]]
        
        if perc is not None and isinstance(perc, (int, float)) and perc > 0:
            clip_value = np.percentile(sub_data, perc)
            vmin = -clip_value
            vmax = clip_value
        else:
            vmin = np.min(sub_data)
            vmax = np.max(sub_data)
        
        ax.clear()
        im = ax.imshow(sub_data, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax, interpolation="bilinear")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if xlim:
            ax.set_xlim(xlim)
        if ylim:
            ax.set_ylim(ylim)
        fig.canvas.draw_idle()

    # Initialize the plot.
    update_plot(0)
    
    slider.on_changed(update_plot)
    
    
    
    # Use plt.show(block=False) for interactive mode.
    plt.show(block=False)

    return slider

def plot_seismic_comparison_with_trace(context, key1, key2, xlabel, ylabel, y_spacing, x_header, perc, xlim=None, ylim=None):
    data1 = context.get(key1)
    data2 = context.get(key2)
    df = context.get("geometry")

    # Validate inputs
    for key, data in zip([key1, key2], [data1, data2]):
        if data is None or not isinstance(data, np.ndarray) or data.ndim != 2:
            print(f"❌ Error: '{key}' not found or not a valid 2D NumPy array in context.")
            return

    if df is None or x_header not in df.columns:
        print(f"❌ Error: Geometry or header '{x_header}' not found in context.")
        return

    if not isinstance(y_spacing, (int, float)) or y_spacing <= 0:
        print(f"❌ Error: y_spacing must be a positive number. Got: {y_spacing}")
        return

    try:
        num_samples, num_traces = data1.shape
        x_values = df[x_header].to_numpy()
        y_values = np.arange(num_samples) * y_spacing
        extent = [x_values[0], x_values[-1], y_values[-1], y_values[0]]

        center_trace = num_traces // 2
        trace1 = data1[:, center_trace]
        trace2 = data2[:, center_trace]

        combined = np.hstack([data1, data2])
        clip = np.percentile(np.abs(combined), perc)

        fig, axes = plt.subplots(1, 3, figsize=(16, 5), gridspec_kw={'width_ratios': [1.2, 1.2, 0.8]})

        # Image 1
        im1 = axes[0].imshow(data1, aspect='auto', cmap='gray_r',
                             vmin=-clip, vmax=clip, extent=extent)
        axes[0].set_title(f"{key1}")
        axes[0].set_xlabel(xlabel)
        axes[0].set_ylabel(ylabel)
        if xlim: axes[0].set_xlim(xlim)
        if ylim: axes[0].set_ylim(ylim)

        # Image 2
        im2 = axes[1].imshow(data2, aspect='auto', cmap='gray_r',
                             vmin=-clip, vmax=clip, extent=extent)
        axes[1].set_title(f"{key2}")
        axes[1].set_xlabel(xlabel)
        axes[1].set_ylabel(ylabel)
        if xlim: axes[1].set_xlim(xlim)
        if ylim: axes[1].set_ylim(ylim)

        # Trace comparison
        t = y_values
        axes[2].plot(trace1, t, label=key1, color='blue')
        axes[2].plot(trace2, t, label=key2, color='red', linestyle='--')
        axes[2].invert_yaxis()
        axes[2].set_xlabel("Amplitude")
        axes[2].set_title("Central Trace Comparison")
        axes[2].legend()
        axes[2].grid(True)
        if ylim: axes[2].set_ylim(ylim[::-1])

        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"❌ Failed to plot comparison: {e}")

def plot_spectrum(context, key="data", dt=1.0):
    wavelet = context.get(key)
    dt=dt/1000.
    if wavelet is None or not isinstance(wavelet, np.ndarray):
        print(f"❌ Error: '{key}' not found or not a valid NumPy array in context.")
        return

    try:
        if wavelet.ndim == 1:
            t = np.arange(len(wavelet)) * dt
            spectrum = np.fft.rfft(wavelet)
            magnitude = np.abs(spectrum)
            phase = np.angle(spectrum) * (180 / np.pi)
            freqs = np.fft.rfftfreq(len(wavelet), d=dt)
        elif wavelet.ndim == 2:
            n_samples, n_traces = wavelet.shape
            t = np.arange(n_samples) * dt
            spectrum = np.fft.rfft(wavelet, axis=0)  # FFT along vertical (time) axis
            magnitude = np.sum(np.abs(spectrum), axis=1)
            phase = np.angle(np.sum(spectrum, axis=1)) * (180 / np.pi)
            freqs = np.fft.rfftfreq(n_samples, d=dt)
        else:
            print(f"❌ Error: Unsupported array dimension: {wavelet.ndim}")
            return

        fig = plt.figure(figsize=(12, 6))
        gs = gridspec.GridSpec(2, 2, width_ratios=[2, 1], height_ratios=[1, 1])

        # Time domain plot
        ax0 = plt.subplot(gs[:, 0])
        if wavelet.ndim == 1:
            ax0.plot(t, wavelet, color='b', label="Wavelet")
        else:
            ax0.imshow(wavelet, aspect='auto', cmap='gray_r', extent=[0, wavelet.shape[1], t[-1], t[0]])
        ax0.set_xlabel('Time' if wavelet.ndim == 1 else 'Trace')
        ax0.set_ylabel('Amplitude' if wavelet.ndim == 1 else 'Time (ms)')
        ax0.set_title('Wavelet in Time Domain')
        ax0.grid()
        ax0.legend(["Wavelet"])

        # Magnitude spectrum
        ax1 = plt.subplot(gs[0, 1])
        ax1.plot(freqs, magnitude, label='Magnitude Spectrum')
        ax1.set_ylabel('Magnitude')
        ax1.set_title('Wavelet Spectrum')
        ax1.grid()
        ax1.legend()

        # Phase spectrum
        ax2 = plt.subplot(gs[1, 1])
        ax2.plot(freqs, phase, label='Phase Spectrum', color='r')
        ax2.set_xlabel('Frequency')
        ax2.set_ylabel('Phase (degrees)')
        ax2.set_title('Wavelet Phase')
        ax2.grid()
        ax2.legend()

        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"❌ Failed to generate wavelet spectrum and phase plot: {e}")
