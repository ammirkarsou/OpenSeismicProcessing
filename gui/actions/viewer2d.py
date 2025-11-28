from pathlib import Path
import json

import numpy as np
import pandas as pd
import zarr
from PyQt6 import QtCore, QtWidgets
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
from openseismicprocessing.catalog import list_projects


class Viewer2D(QtWidgets.QWidget):
    def __init__(self, parent, manifests, boundary=None):
        super().__init__(parent)
        self.manifests = manifests
        self.boundary = boundary
        self.geom_df = None
        self.amp = None
        self.inline_col = None
        self.xline_col = None
        self.current_orientation = "inline"

        layout = QtWidgets.QHBoxLayout(self)

        # Left panel with datasets and map
        left_panel = QtWidgets.QVBoxLayout()
        left_panel.addWidget(QtWidgets.QLabel("Datasets"))
        self.list = QtWidgets.QListWidget()
        self.list.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.NoSelection)
        for m in self.manifests:
            name = m.stem.replace(".zarr", "").replace(".manifest", "")
            item = QtWidgets.QListWidgetItem(name)
            item.setFlags(item.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(QtCore.Qt.CheckState.Unchecked)
            item.setData(QtCore.Qt.ItemDataRole.UserRole, m)
            self.list.addItem(item)
        self.list.itemChanged.connect(self.on_item_changed)
        left_panel.addWidget(self.list, 2)
        self.map_fig = Figure(figsize=(4, 3))
        self.map_canvas = FigureCanvas(self.map_fig)
        self.map_toolbar = NavigationToolbar2QT(self.map_canvas, self)
        left_panel.addWidget(self.map_toolbar)
        left_panel.addWidget(self.map_canvas, 1)
        left_panel.addStretch()

        layout.addLayout(left_panel, 1)

        main_right = QtWidgets.QVBoxLayout()

        top_row = QtWidgets.QHBoxLayout()
        top_row.addWidget(QtWidgets.QLabel("Orientation"))
        self.orient_combo = QtWidgets.QComboBox()
        self.orient_combo.addItems(["inline", "crossline"])
        top_row.addWidget(self.orient_combo)

        top_row.addWidget(QtWidgets.QLabel("Colormap"))
        self.cmap_combo = QtWidgets.QComboBox()
        self.cmap_combo.addItems(sorted(plt.colormaps()))
        jet_idx = self.cmap_combo.findText("jet")
        if jet_idx >= 0:
            self.cmap_combo.setCurrentIndex(jet_idx)
        top_row.addWidget(self.cmap_combo)

        top_row.addWidget(QtWidgets.QLabel("Min"))
        self.vmin_spin = QtWidgets.QDoubleSpinBox()
        self.vmin_spin.setRange(-1e9, 1e9)
        self.vmin_spin.setDecimals(6)
        self.vmin_spin.setValue(-1.0)
        top_row.addWidget(self.vmin_spin)

        top_row.addWidget(QtWidgets.QLabel("Max"))
        self.vmax_spin = QtWidgets.QDoubleSpinBox()
        self.vmax_spin.setRange(-1e9, 1e9)
        self.vmax_spin.setDecimals(6)
        self.vmax_spin.setValue(1.0)
        top_row.addWidget(self.vmax_spin)

        self.apply_global_btn = QtWidgets.QPushButton("Apply Min/Max")
        top_row.addWidget(self.apply_global_btn)

        right_side = QtWidgets.QVBoxLayout()
        right_side.addLayout(top_row)

        self.slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider.setMinimum(0)
        self.slider.valueChanged.connect(self.update_section)
        self.slider_min_lbl = QtWidgets.QLabel("-")
        self.slider_cur_lbl = QtWidgets.QLabel("-")
        self.slider_max_lbl = QtWidgets.QLabel("-")
        slider_row = QtWidgets.QHBoxLayout()
        slider_row.addWidget(self.slider_min_lbl)
        slider_row.addWidget(self.slider, 1)
        slider_row.addWidget(self.slider_max_lbl)
        slider_row.addWidget(QtWidgets.QLabel("Current:"))
        slider_row.addWidget(self.slider_cur_lbl)
        right_side.addLayout(slider_row)

        self.figure = Figure(figsize=(8, 6))
        self.canvas = FigureCanvas(self.figure)
        right_side.addWidget(self.canvas)

        main_right.addLayout(right_side)
        layout.addLayout(main_right, 3)

        self.orient_combo.currentIndexChanged.connect(self.change_orientation)
        self.cmap_combo.currentIndexChanged.connect(self.update_section)
        self.vmin_spin.valueChanged.connect(self.update_section)
        self.vmax_spin.valueChanged.connect(self.update_section)
        self.apply_global_btn.clicked.connect(self.apply_global_limits)

        self.current_manifest = None
        self._current_subset_df = None

    def on_item_changed(self, item):
        if item.checkState() == QtCore.Qt.CheckState.Checked:
            for i in range(self.list.count()):
                it = self.list.item(i)
                if it is not item and it.checkState() == QtCore.Qt.CheckState.Checked:
                    it.setCheckState(QtCore.Qt.CheckState.Unchecked)
            self.current_manifest = item.data(QtCore.Qt.ItemDataRole.UserRole)
            self.load_dataset()
        else:
            if self.current_manifest == item.data(QtCore.Qt.ItemDataRole.UserRole):
                self.current_manifest = None
                self._current_subset_df = None
                self._clear_map()

    def load_dataset(self):
        manifest_path = self.current_manifest
        if not manifest_path:
            return
        try:
            data = json.loads(Path(manifest_path).read_text())
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "2D Viewer", f"Failed to read manifest:\n{exc}")
            return
        geom_path = data.get("geometry_parquet")
        zarr_path = data.get("zarr_store")
        selected_headers = data.get("selected_headers", {}) or {}
        dtype = str(data.get("dataset_type", "")).lower()
        if "post" not in dtype:
            QtWidgets.QMessageBox.information(self, "2D Viewer", "Pre-stack datasets are not shown here.")
            return
        try:
            self.geom_df = pd.read_parquet(geom_path) if geom_path else None
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "2D Viewer", f"Failed to read geometry:\n{exc}")
            return
        try:
            store = zarr.open(zarr_path, mode="r") if zarr_path else None
            self.amp = store["amplitude"] if store is not None else None
            if store is not None and "trace_ids" in store:
                ids = np.asarray(store["trace_ids"][:], dtype=np.int64)
                if self.geom_df is not None:
                    if "trace_id" in self.geom_df.columns:
                        try:
                            self.geom_df = self.geom_df.set_index("trace_id").loc[ids].reset_index()
                        except Exception:
                            self.geom_df = self.geom_df.iloc[ids]
                    else:
                        self.geom_df = self.geom_df.iloc[ids]
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "2D Viewer", f"Failed to open Zarr store:\n{exc}")
            return

        def find_col(df, names):
            for n in names:
                if n in df.columns:
                    return n
            lowmap = {c.lower(): c for c in df.columns}
            for n in names:
                if n.lower() in lowmap:
                    return lowmap[n.lower()]
            return None

        self.inline_col = selected_headers.get("inline_header") or find_col(self.geom_df, ["iline", "inline"])
        self.xline_col = selected_headers.get("xline_header") or find_col(self.geom_df, ["xline", "crossline"])
        self.z_start = float(selected_headers.get("z_start", 0.0) or 0.0)
        self.z_inc = float(selected_headers.get("z_increment", 1.0) or 1.0)
        if self.inline_col is None or self.xline_col is None:
            QtWidgets.QMessageBox.warning(self, "2D Viewer", "Inline/Crossline headers not found.")
            return
        self.compute_global_limits(force=True)
        self.change_orientation()

    def change_orientation(self):
        self.current_orientation = self.orient_combo.currentText()
        if self.geom_df is None:
            return
        if self.current_orientation == "inline":
            values = np.unique(self.geom_df[self.inline_col].to_numpy())
        else:
            values = np.unique(self.geom_df[self.xline_col].to_numpy())
        self.section_values = values
        self._update_slider_labels()
        if len(values) == 0:
            self.slider.setMaximum(0)
        else:
            self.slider.setMaximum(len(values) - 1)
        self.slider.setValue(0)
        self.update_section()

    def update_section(self):
        if self.geom_df is None or self.amp is None:
            return
        if self.slider.maximum() < 0:
            return
        idx = self.slider.value()
        if self.current_orientation == "inline":
            target = self.section_values[idx]
            mask = self.geom_df[self.inline_col] == target
            order_col = self.xline_col
        else:
            target = self.section_values[idx]
            mask = self.geom_df[self.xline_col] == target
            order_col = self.inline_col
        subset = self.geom_df[mask].copy()
        if subset.empty:
            return
        subset = subset.sort_values(order_col)
        trace_ids = subset["trace_id"].to_numpy()
        try:
            data = self.amp.oindex[:, trace_ids]
        except Exception:
            data = self.amp[:, trace_ids]
        img = data
        vmin = self.vmin_spin.value()
        vmax = self.vmax_spin.value()
        cmap = self.cmap_combo.currentText()
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        im = ax.imshow(img, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax, origin="upper")
        y_spacing = getattr(self, "z_inc", 1.0)
        y_start = getattr(self, "z_start", 0.0)
        y_end = y_start + (img.shape[0] - 1) * y_spacing
        ax.images[0].set_extent([subset[order_col].iloc[0], subset[order_col].iloc[-1], y_end, y_start])
        ax.set_xlabel(order_col)
        ax.set_ylabel("Z")
        try:
            target_disp = int(round(float(target)))
        except Exception:
            target_disp = target
        ax.set_title(f"{self.current_orientation.title()} {target_disp}")
        self.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        self.canvas.draw_idle()
        self._current_subset_df = subset
        self._plot_map(subset)
        self._update_slider_labels(current=target)

    def apply_global_limits(self):
        self.compute_global_limits(force=True)
        self.update_section()

    def compute_global_limits(self, force: bool = False):
        if self.amp is None:
            return
        try:
            if not force and hasattr(self, "_global_vmin") and hasattr(self, "_global_vmax"):
                return
            arr = self.amp[:]
            vmin = float(np.nanmin(arr))
            vmax = float(np.nanmax(arr))
            self._global_vmin = vmin
            self._global_vmax = vmax
            self.vmin_spin.blockSignals(True)
            self.vmax_spin.blockSignals(True)
            self.vmin_spin.setValue(vmin)
            self.vmax_spin.setValue(vmax)
            self.vmin_spin.blockSignals(False)
            self.vmax_spin.blockSignals(False)
        except Exception:
            pass

    def _clear_map(self):
        self.map_fig.clear()
        self.map_canvas.draw_idle()

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
        plotted = False
        x_min = x_max = y_min = y_max = None
        if self.boundary and "x_range" in self.boundary and "y_range" in self.boundary:
            x_min, x_max = self.boundary["x_range"]
            y_min, y_max = self.boundary["y_range"]
            rect_x = [x_min, x_max, x_max, x_min, x_min]
            rect_y = [y_min, y_min, y_max, y_max, y_min]
            ax.plot(rect_x, rect_y, color="green", linewidth=1.5, linestyle="--", label="Survey footprint")
            plotted = True
        if subset is not None and not subset.empty:
            sx_col = self._find_col(subset, ["SourceX", "sx"])
            sy_col = self._find_col(subset, ["SourceY", "sy"])
            gx_col = self._find_col(subset, ["GroupX", "gx"])
            gy_col = self._find_col(subset, ["GroupY", "gy"])
            x_col = sx_col or gx_col
            y_col = sy_col or gy_col
            if x_col and y_col:
                xs = subset[x_col].to_numpy(dtype=float, copy=False)
                ys = subset[y_col].to_numpy(dtype=float, copy=False)
                ax.plot(xs, ys, color="orange", linewidth=2, label="Slice path")
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

    def _update_slider_labels(self, current=None):
        if getattr(self, "section_values", None) is None or len(self.section_values) == 0:
            self.slider_min_lbl.setText("-")
            self.slider_max_lbl.setText("-")
            self.slider_cur_lbl.setText("-")
            return
        fmt = lambda v: str(int(round(float(v)))) if self._is_number(v) else str(v)
        self.slider_min_lbl.setText(fmt(self.section_values[0]))
        self.slider_max_lbl.setText(fmt(self.section_values[-1]))
        if current is None:
            idx = self.slider.value()
            if 0 <= idx < len(self.section_values):
                current = self.section_values[idx]
        if current is not None:
            self.slider_cur_lbl.setText(fmt(current))

    def _is_number(self, v):
        try:
            float(v)
            return True
        except Exception:
            return False


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


def show_2d_viewer(window):
    """Add the 2D viewer tab for the current survey."""
    if not window.currentSurveyPath:
        QtWidgets.QMessageBox.warning(window, "Warning", "Please select a survey first.")
        return
    manifests_all = [p for p in _list_manifests(window.currentSurveyPath) if p.exists()]
    manifests = [p for p in manifests_all if "post" in _dataset_type(p)]
    if not manifests:
        QtWidgets.QMessageBox.information(window, "2D Viewer", "No post-stack datasets found for this survey.")
        return
    boundary = None
    if window.currentSurveyName:
        try:
            project = next(p for p in list_projects() if p["name"] == window.currentSurveyName)
            metadata = project.get("metadata", {}) or {}
            boundary = metadata.get("boundary", metadata)
        except Exception:
            boundary = None
    widget = Viewer2D(window, manifests, boundary=boundary)
    window.tabWidget.addTab(widget, "2D Viewer")
    window.tabWidget.setCurrentWidget(widget)
