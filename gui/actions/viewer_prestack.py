from pathlib import Path
import json

import pandas as pd
import zarr
import numpy as np
from PyQt6 import QtCore, QtWidgets
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
from openseismicprocessing import processing
from openseismicprocessing._plotting import plot_seismic_image
from openseismicprocessing.catalog import list_projects
from boundary_utils import footprint_polygon


def _list_manifests(survey_path: str | Path) -> list[Path]:
    bin_dir = Path(survey_path) / "Binaries"
    if not bin_dir.exists():
        return []
    return sorted(bin_dir.glob("*.manifest.json"), key=lambda p: p.name.lower())


def _dataset_type(manifest: Path) -> str:
    try:
        data = json.loads(manifest.read_text())
        return str(data.get("dataset_type", "")).lower()
    except Exception:
        return ""


class PrestackViewer(QtWidgets.QWidget):
    def __init__(self, parent, manifests, boundary=None):
        super().__init__(parent)
        self.manifests = manifests
        self.boundary = boundary
        self.current_manifest = None
        self.manifest_meta = {}
        self.geom_df = None
        self.amp = None
        self._current_subset_data: np.ndarray | None = None
        self._current_subset_df: pd.DataFrame | None = None
        self._current_header1: str | None = None
        self._current_header2: str | None = None
        self._current_target = None

        layout = QtWidgets.QHBoxLayout(self)

        self.dataset_list = QtWidgets.QListWidget()
        self.dataset_list.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.NoSelection)
        for m in self.manifests:
            name = m.stem.replace(".zarr", "").replace(".manifest", "")
            item = QtWidgets.QListWidgetItem(name)
            item.setFlags(item.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(QtCore.Qt.CheckState.Unchecked)
            item.setData(QtCore.Qt.ItemDataRole.UserRole, m)
            self.dataset_list.addItem(item)
        self.dataset_list.itemChanged.connect(self.on_item_changed)

        left = QtWidgets.QVBoxLayout()
        left.addWidget(QtWidgets.QLabel("Datasets"))
        left.addWidget(self.dataset_list, 2)
        self.map_fig = Figure(figsize=(4, 3))
        self.map_canvas = FigureCanvas(self.map_fig)
        self.map_toolbar = NavigationToolbar2QT(self.map_canvas, self)
        left.addWidget(self.map_toolbar)
        left.addWidget(self.map_canvas, 1)
        left.addStretch()
        layout.addLayout(left, 1)

        right = QtWidgets.QVBoxLayout()
        top_row = QtWidgets.QHBoxLayout()

        top_row.addWidget(QtWidgets.QLabel("Header 1"))
        self.header1_combo = QtWidgets.QComboBox()
        self.header1_combo.addItem("Select header")
        top_row.addWidget(self.header1_combo)

        top_row.addWidget(QtWidgets.QLabel("Header 2"))
        self.header2_combo = QtWidgets.QComboBox()
        self.header2_combo.addItem("Select header")
        top_row.addWidget(self.header2_combo)

        self.apply_btn = QtWidgets.QPushButton("Apply")
        top_row.addWidget(self.apply_btn)

        top_row.addWidget(QtWidgets.QLabel("Percentile"))
        self.perc_spin = QtWidgets.QDoubleSpinBox()
        self.perc_spin.setRange(0.0, 100.0)
        self.perc_spin.setDecimals(2)
        self.perc_spin.setValue(99.0)
        top_row.addWidget(self.perc_spin)

        self.spectrum_chk = QtWidgets.QCheckBox("Spectrum")
        self.spectrum_chk.setChecked(False)
        top_row.addWidget(self.spectrum_chk)
        self.autocorr_chk = QtWidgets.QCheckBox("Autocorrelation")
        self.autocorr_chk.setChecked(False)
        top_row.addWidget(self.autocorr_chk)

        right.addLayout(top_row)

        self.slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider.setMinimum(0)
        self.slider.valueChanged.connect(self.on_slider_changed)
        self.slider_min_lbl = QtWidgets.QLabel("-")
        self.slider_cur_lbl = QtWidgets.QLabel("-")
        self.slider_max_lbl = QtWidgets.QLabel("-")
        slider_row = QtWidgets.QHBoxLayout()
        slider_row.addWidget(self.slider_min_lbl)
        slider_row.addWidget(self.slider, 1)
        slider_row.addWidget(self.slider_max_lbl)
        slider_row.addWidget(QtWidgets.QLabel("Current:"))
        slider_row.addWidget(self.slider_cur_lbl)
        right.addLayout(slider_row)

        self.placeholder = QtWidgets.QLabel("Select a dataset to configure headers.")
        self.placeholder.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        right.addWidget(self.placeholder)

        self.figure = Figure(figsize=(10, 8), constrained_layout=True)
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)

        self.spectrum_fig = Figure(figsize=(6, 3), constrained_layout=True)
        self.spectrum_canvas = FigureCanvas(self.spectrum_fig)
        self.spectrum_canvas.setVisible(False)

        plots_row = QtWidgets.QHBoxLayout()
        main_plot_col = QtWidgets.QVBoxLayout()
        main_plot_col.addWidget(self.toolbar)
        main_plot_col.addWidget(self.canvas, 1)
        plots_row.addLayout(main_plot_col, 5)
        plots_row.addWidget(self.spectrum_canvas, 1)
        self.autocorr_fig = Figure(figsize=(6, 3), constrained_layout=True)
        self.autocorr_canvas = FigureCanvas(self.autocorr_fig)
        self.autocorr_canvas.setVisible(False)
        plots_row.addWidget(self.autocorr_canvas, 1)
        right.addLayout(plots_row)

        layout.addLayout(right, 3)

        self.apply_btn.clicked.connect(self.apply_selection)
        self.perc_spin.valueChanged.connect(self.on_limits_changed)
        self.spectrum_chk.stateChanged.connect(self.on_spectrum_toggle)
        self.autocorr_chk.stateChanged.connect(self.on_autocorr_toggle)

    def on_item_changed(self, item: QtWidgets.QListWidgetItem):
        if item.checkState() == QtCore.Qt.CheckState.Checked:
            for i in range(self.dataset_list.count()):
                other = self.dataset_list.item(i)
                if other is not item and other.checkState() == QtCore.Qt.CheckState.Checked:
                    other.setCheckState(QtCore.Qt.CheckState.Unchecked)
            self.current_manifest = item.data(QtCore.Qt.ItemDataRole.UserRole)
            self._populate_headers()
        else:
            if self.current_manifest == item.data(QtCore.Qt.ItemDataRole.UserRole):
                self.current_manifest = None
                self.header1_combo.clear()
                self.header2_combo.clear()
                self.header1_combo.addItem("Select header")
                self.header2_combo.addItem("Select header")
                self.placeholder.setText("Select a dataset to configure headers.")
                self.geom_df = None
                self.amp = None
                self.slider.blockSignals(True)
                self.slider.setMaximum(0)
                self.slider.setValue(0)
                self.slider.blockSignals(False)
                self._current_subset_data = None
                self._current_subset_df = None
                self._current_header1 = None
                self._current_header2 = None
                self._current_target = None
                self._clear_map()

    def _populate_headers(self):
        meta = self._read_manifest(self.current_manifest)
        geom_path = meta.get("geometry_parquet")
        if not geom_path:
            QtWidgets.QMessageBox.warning(self, "Pre-stack Viewer", "Geometry path missing in manifest.")
            return
        try:
            df = pd.read_parquet(geom_path)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Pre-stack Viewer", f"Failed to read geometry:\n{exc}")
            return
        self.geom_df = df
        headers = [str(c) for c in df.columns]
        self.header1_combo.blockSignals(True)
        self.header2_combo.blockSignals(True)
        self.header1_combo.clear()
        self.header2_combo.clear()
        self.header1_combo.addItems(headers)
        self.header2_combo.addItems(headers)
        default_h1 = "fldr" if "fldr" in df.columns else headers[0] if headers else ""
        default_h2 = "tracf" if "tracf" in df.columns else headers[1] if len(headers) > 1 else default_h1
        if default_h1:
            idx1 = self.header1_combo.findText(default_h1)
            self.header1_combo.setCurrentIndex(idx1 if idx1 >= 0 else 0)
        if default_h2:
            idx2 = self.header2_combo.findText(default_h2)
            self.header2_combo.setCurrentIndex(idx2 if idx2 >= 0 else 0)
        self.header1_combo.blockSignals(False)
        self.header2_combo.blockSignals(False)
        name = self.current_manifest.stem.replace(".zarr", "").replace(".manifest", "")
        self.placeholder.setText(f"Selected: {name}\nHeaders ready. Click Apply to load data.")

    def _read_manifest(self, manifest: Path) -> dict:
        try:
            data = json.loads(manifest.read_text())
            self.manifest_meta[manifest] = data
            return data
        except Exception:
            return {}

    def apply_selection(self):
        if not self.current_manifest:
            QtWidgets.QMessageBox.information(self, "Pre-stack Viewer", "Select a dataset first.")
            return
        meta = self.manifest_meta.get(self.current_manifest) or self._read_manifest(self.current_manifest)
        geom_path = meta.get("geometry_parquet")
        zarr_path = meta.get("zarr_store")
        header1 = self.header1_combo.currentText()
        header2 = self.header2_combo.currentText()
        selected_headers = meta.get("selected_headers", {}) or {}
        if not geom_path or not zarr_path:
            QtWidgets.QMessageBox.warning(self, "Pre-stack Viewer", "Manifest is missing required paths.")
            return
        try:
            self.geom_df = pd.read_parquet(geom_path)
            store = zarr.open(zarr_path, mode="r")
            self.amp = store["amplitude"]
            trace_ids_ds = store.get("trace_ids")
            if trace_ids_ds is not None:
                ids = np.asarray(trace_ids_ds[:], dtype=np.int64)
                if "trace_id" in self.geom_df.columns:
                    try:
                        self.geom_df = (
                            self.geom_df.set_index("trace_id")
                            .loc[ids]
                            .reset_index()
                        )
                    except Exception:
                        # fallback to iloc if loc fails
                        self.geom_df = self.geom_df.iloc[ids]
                else:
                    self.geom_df = self.geom_df.iloc[ids]
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Pre-stack Viewer", f"Failed to load data:\n{exc}")
            return
        self.z_start = float(selected_headers.get("z_start", 0.0) or 0.0)
        self.z_inc = float(selected_headers.get("z_increment", 1.0) or 1.0)
        self.source_x_header = selected_headers.get("source_x_header")
        self.source_y_header = selected_headers.get("source_y_header")
        self.group_x_header = selected_headers.get("group_x_header")
        self.group_y_header = selected_headers.get("group_y_header")
        try:
            context = {"geometry": self.geom_df.copy(), "data": self.amp[:]}
            processing.sort(context, header1, header2)
            self.geom_df = context["geometry"]
            self.amp = context["data"]
            self._header1_values = np.unique(self.geom_df[header1].to_numpy())
            self.slider.blockSignals(True)
            max_idx = len(self._header1_values) - 1 if len(self._header1_values) > 0 else 0
            self.slider.setMaximum(max_idx)
            self.slider.setValue(0)
            self.slider.blockSignals(False)
            self._update_slider_labels()
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Pre-stack Viewer", f"Failed to sort data:\n{exc}")
            return
        name = self.current_manifest.stem.replace(".zarr", "").replace(".manifest", "")
        self.placeholder.setText(
            f"Loaded dataset: {name}\nHeader1: {header1}, Header2: {header2}\n(Plotting not implemented yet.)"
        )
        # Display first slice
        self.on_slider_changed(0)

    def on_slider_changed(self, idx: int):
        if getattr(self, "_header1_values", None) is None:
            return
        if self.geom_df is None or self.amp is None:
            return
        if len(self._header1_values) == 0:
            return
        if idx < 0 or idx >= len(self._header1_values):
            return
        header1 = self.header1_combo.currentText()
        header2 = self.header2_combo.currentText()
        target = self._header1_values[idx]
        self._update_slider_labels(current=target)
        mask = self.geom_df[header1] == target
        subset = self.geom_df[mask]
        trace_indices = subset.index.to_numpy()
        if len(trace_indices) == 0:
            return
        try:
            subset_data = self.amp[:, trace_indices]
            self._plot_gather(subset_data, subset.copy(), header1, header2, target)
            self.placeholder.setText(
                f"Selected {header1}={target} with {len(trace_indices)} traces."
            )
            self._current_subset_data = subset_data
            self._current_subset_df = subset.copy()
            self._current_header1 = header1
            self._current_header2 = header2
            self._current_target = target
            self._plot_map(subset)
            self._update_spectrum_plot()
            self._update_autocorr_plot()
        except Exception:
            self.placeholder.setText(
                f"Selected {header1}={target} with {len(trace_indices)} traces (plot failed)."
            )

    def _plot_gather(self, data: np.ndarray, geom_df: pd.DataFrame, header1: str, header2: str, target):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        if header2 not in geom_df.columns:
            geom_df = geom_df.copy()
            geom_df["trace_index"] = np.arange(data.shape[1])
            header2 = "trace_index"
        context = {"data": data, "geometry": geom_df}
        # Auto-scale using user-selected percentile to dampen outliers
        perc = self.perc_spin.value()
        vmin = vmax = None
        if perc > 0:
            clip = float(np.nanpercentile(data, perc))
            vmin = -clip
            vmax = clip
        y_spacing = getattr(self, "z_inc", 1.0)
        y_start = getattr(self, "z_start", 0.0)
        y_end = y_start + (data.shape[0] - 1) * y_spacing
        extent = [geom_df[header2].iloc[0], geom_df[header2].iloc[-1], y_end, y_start]
        im = ax.imshow(
            data,
            aspect="auto",
            cmap="gray_r",
            vmin=vmin,
            vmax=vmax,
            extent=extent,
            origin="upper",
            interpolation="none",
        )
        ax.margins(0)
        ax.set_xlabel(header2)
        ax.set_ylabel("Time (ms)")
        ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label="Amplitude")
        ax.set_title(f"{header1} = {target}")
        try:
            self.figure.tight_layout(pad=0.3)
        except Exception:
            pass
        self.canvas.draw_idle()

    def _clear_map(self):
        self.map_fig.clear()
        self.map_canvas.draw_idle()
        self._update_slider_labels(clear=True)

    def _find_col(self, df: pd.DataFrame, names: list[str]) -> str | None:
        for n in names:
            if n in df.columns:
                return n
        lowmap = {c.lower(): c for c in df.columns}
        for n in names:
            if n.lower() in lowmap:
                return lowmap[n.lower()]
        return None

    def _plot_map(self, subset: pd.DataFrame):
        self.map_fig.clear()
        ax = self.map_fig.add_subplot(111)
        # Survey frontier from project metadata (if available)
        plotted = False
        x_min = x_max = y_min = y_max = None
        poly = footprint_polygon(self.boundary)
        if poly is not None:
            xs, ys, x_min, x_max, y_min, y_max = poly
            ax.plot(xs, ys, color="green", linewidth=1.5, linestyle="--", label="Survey frontier")
            plotted = True
        if subset is None or subset.empty:
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            if plotted:
                ax.legend(loc="best")
            ax.grid(True, linestyle="--", alpha=0.3)
            self.map_canvas.draw_idle()
            return
        sx_col = self._find_col(subset, ["SourceX", "sx"])
        sy_col = self._find_col(subset, ["SourceY", "sy"])
        gx_col = self._find_col(subset, ["GroupX", "gx"])
        gy_col = self._find_col(subset, ["GroupY", "gy"])
        if sx_col and sy_col:
            ax.scatter(subset[sx_col], subset[sy_col], s=18, c="red", alpha=0.7, marker="*", label="Sources", rasterized=True)
            plotted = True
        if gx_col and gy_col:
            ax.scatter(subset[gx_col], subset[gy_col], s=3, c="blue", alpha=0.6, label="Receivers", rasterized=True)
            plotted = True
        if x_min is not None and x_max is not None and y_min is not None and y_max is not None:
            pad_x = 0.05 * max(1.0, x_max - x_min)
            pad_y = 0.05 * max(1.0, y_max - y_min)
            ax.set_xlim(x_min - pad_x, x_max + pad_x)
            ax.set_ylim(y_min - pad_y, y_max + pad_y)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        if plotted:
            ax.legend(loc="upper right", bbox_to_anchor=(1.15, 1.0))
        ax.grid(True, linestyle="--", alpha=0.3)
        self.map_canvas.draw_idle()

    def _clear_spectrum(self):
        self.spectrum_fig.clear()
        self.spectrum_canvas.draw_idle()
        self.spectrum_canvas.setVisible(False)

    def _clear_autocorr(self):
        self.autocorr_fig.clear()
        self.autocorr_canvas.draw_idle()
        self.autocorr_canvas.setVisible(False)

    def on_spectrum_toggle(self):
        self._update_spectrum_plot()

    def on_autocorr_toggle(self):
        self._update_autocorr_plot()

    def _plot_spectrum(self, data: np.ndarray):
        self.spectrum_fig.clear()
        ax = self.spectrum_fig.add_subplot(111)
        dt_sec = (getattr(self, "z_inc", 1.0) or 1.0) / 1000.0
        try:
            n_samples = data.shape[0]
            spectrum = np.fft.rfft(data, axis=0)
            magnitude = np.sum(np.abs(spectrum), axis=1)
            freqs = np.fft.rfftfreq(n_samples, d=dt_sec)
            ax.plot(freqs, magnitude, label="Amplitude spectrum")
            ax.set_xlabel("Frequency (Hz)")
            ax.set_ylabel("Magnitude")
            ax.grid(True, linestyle="--", alpha=0.3)
            ax.legend()
        except Exception as exc:
            ax.text(0.5, 0.5, f"Failed to compute spectrum:\n{exc}", ha="center", va="center")
        self.spectrum_canvas.draw_idle()

    def _plot_autocorr(self, data: np.ndarray):
        self.autocorr_fig.clear()
        ax = self.autocorr_fig.add_subplot(111)
        dt_sec = (getattr(self, "z_inc", 1.0) or 1.0) / 1000.0
        try:
            # Autocorrelation per trace along Z
            n_samples, n_traces = data.shape
            corrs = np.empty((2 * n_samples - 1, n_traces))
            for i in range(n_traces):
                trace = data[:, i]
                corr = np.correlate(trace, trace, mode="full")
                corrs[:, i] = corr
            lags = np.arange(-(n_samples - 1), n_samples) * dt_sec
            extent = [0, n_traces - 1, lags[-1], lags[0]]
            perc = self.perc_spin.value()
            vmin = vmax = None
            if perc > 0:
                clip = float(np.nanpercentile(corrs, perc))
                vmin = -clip
                vmax = clip
            im = ax.imshow(
                corrs,
                aspect="auto",
                cmap="gray_r",
                vmin=vmin,
                vmax=vmax,
                extent=extent,
                origin="upper",
                interpolation="none",
            )
            ax.set_xlabel("Trace")
            ax.set_ylabel("Lag (ms)")
            ax.set_title("Autocorrelation")
            self.autocorr_fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Amplitude")
        except Exception as exc:
            ax.text(0.5, 0.5, f"Failed to compute autocorrelation:\n{exc}", ha="center", va="center")
        self.autocorr_canvas.draw_idle()

    def _update_autocorr_plot(self):
        if not self.autocorr_chk.isChecked():
            self._clear_autocorr()
            return
        if self._current_subset_data is None:
            self._clear_autocorr()
            return
        self.autocorr_canvas.setVisible(True)
        self._plot_autocorr(self._current_subset_data)

    def _format_val(self, val):
        try:
            ival = int(round(float(val)))
            return str(ival)
        except Exception:
            return str(val)

    def _update_slider_labels(self, current=None, clear: bool = False):
        if clear or getattr(self, "_header1_values", None) is None or len(self._header1_values) == 0:
            self.slider_min_lbl.setText("-")
            self.slider_max_lbl.setText("-")
            self.slider_cur_lbl.setText("-")
            return
        self.slider_min_lbl.setText(self._format_val(self._header1_values[0]))
        self.slider_max_lbl.setText(self._format_val(self._header1_values[-1]))
        if current is None:
            idx = self.slider.value()
            if 0 <= idx < len(self._header1_values):
                current = self._header1_values[idx]
        if current is not None:
            self.slider_cur_lbl.setText(self._format_val(current))

    def _update_spectrum_plot(self):
        if not self.spectrum_chk.isChecked():
            self._clear_spectrum()
            return
        if self._current_subset_data is None:
            self._clear_spectrum()
            return
        self.spectrum_canvas.setVisible(True)
        self._plot_spectrum(self._current_subset_data)

    def apply_global_limits(self):
        # Deprecated placeholder; kept for compatibility if invoked elsewhere.
        if self._current_subset_data is None:
            return

    def on_limits_changed(self):
        if self._current_subset_data is None or self._current_subset_df is None:
            return
        self._plot_gather(
            self._current_subset_data,
            self._current_subset_df,
            self._current_header1 or "",
            self._current_header2 or "",
            self._current_target,
        )

def show_prestack_viewer(window):
    """Add the pre-stack viewer tab for the current survey."""
    if not window.currentSurveyPath:
        QtWidgets.QMessageBox.warning(window, "Warning", "Please select a survey first.")
        return
    manifests_all = [p for p in _list_manifests(window.currentSurveyPath) if p.exists()]
    manifests = [p for p in manifests_all if "post" not in _dataset_type(p)]
    if not manifests:
        QtWidgets.QMessageBox.information(window, "Pre-stack Viewer", "No pre-stack datasets found for this survey.")
        return
    boundary = None
    if window.currentSurveyName:
        try:
            project = next(p for p in list_projects() if p["name"] == window.currentSurveyName)
            metadata = project.get("metadata", {}) or {}
            boundary = metadata.get("boundary", metadata)
        except Exception:
            boundary = None
    widget = PrestackViewer(window, manifests, boundary=boundary)
    window.tabWidget.addTab(widget, "Pre-stack Viewer")
    window.tabWidget.setCurrentWidget(widget)
