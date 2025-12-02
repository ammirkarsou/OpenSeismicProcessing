from pathlib import Path

import numpy as np
import pandas as pd
from PyQt6 import QtCore, QtWidgets
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
from matplotlib.figure import Figure
from openseismicprocessing.catalog import list_projects
from boundary_utils import footprint_polygon


def _infer_dtype(geom_path: Path, survey_root: Path | None) -> str:
    if geom_path is None or survey_root is None:
        return ""
    base = geom_path.name.split(".geometry")[0]
    manifest = survey_root / "Binaries" / f"{base}.zarr.manifest.json"
    if manifest.exists():
        try:
            import json

            data = json.loads(manifest.read_text())
            return str(data.get("dataset_type", "")).lower()
        except Exception:
            return ""
    return ""


class BasemapDialog(QtWidgets.QWidget):
    def __init__(self, parent, files, boundary, survey_root: Path | None, survey_name: str | None):
        super().__init__(parent)
        self.files = files
        self.boundary = boundary
        self.survey_root = survey_root
        self.survey_name = survey_name
        self.setWindowTitle("Basemap")
        self.resize(1100, 700)
        self.list = QtWidgets.QListWidget()
        self.list.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.NoSelection)

        def base_label(p: Path) -> str:
            return p.name.replace(".geometry.parquet", "").replace(".geometry.csv", "")

        # Group by type with headers
        prestack_header = QtWidgets.QListWidgetItem("Pre-stack")
        prestack_header.setFlags(QtCore.Qt.ItemFlag.ItemIsEnabled)
        prestack_header.setData(QtCore.Qt.ItemDataRole.UserRole, None)
        poststack_header = QtWidgets.QListWidgetItem("Post-stack")
        poststack_header.setFlags(QtCore.Qt.ItemFlag.ItemIsEnabled)
        poststack_header.setData(QtCore.Qt.ItemDataRole.UserRole, None)

        self.list.addItem(prestack_header)
        for path in self.files:
            dtype = _infer_dtype(path, self.survey_root)
            if "post" in dtype:
                continue
            item = QtWidgets.QListWidgetItem(f"  {base_label(path)}")
            item.setFlags(item.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(QtCore.Qt.CheckState.Unchecked)
            item.setData(QtCore.Qt.ItemDataRole.UserRole, path)
            self.list.addItem(item)

        self.list.addItem(poststack_header)
        for path in self.files:
            dtype = _infer_dtype(path, self.survey_root)
            if "post" not in dtype:
                continue
            item = QtWidgets.QListWidgetItem(f"  {base_label(path)}")
            item.setFlags(item.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(QtCore.Qt.CheckState.Unchecked)
            item.setData(QtCore.Qt.ItemDataRole.UserRole, path)
            self.list.addItem(item)

        self.list.itemChanged.connect(self.update_plot)

        self.showSourcesChk = QtWidgets.QCheckBox("Sources")
        self.showSourcesChk.setChecked(True)
        self.showReceiversChk = QtWidgets.QCheckBox("Receivers")
        self.showReceiversChk.setChecked(True)
        self.showFoldChk = QtWidgets.QCheckBox("Fold map")
        self.showFoldChk.setChecked(False)
        for chk in (self.showSourcesChk, self.showReceiversChk, self.showFoldChk):
            chk.stateChanged.connect(self.update_plot)

        self.figure = Figure(figsize=(8, 6))
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)

        left_layout = QtWidgets.QVBoxLayout()
        left_layout.addWidget(QtWidgets.QLabel("Datasets"))
        left_layout.addWidget(self.list)
        left_layout.addWidget(self.showSourcesChk)
        left_layout.addWidget(self.showReceiversChk)
        left_layout.addWidget(self.showFoldChk)
        left_layout.addStretch()

        right_layout = QtWidgets.QVBoxLayout()
        right_layout.addWidget(self.toolbar)
        right_layout.addWidget(self.canvas)

        main_layout = QtWidgets.QHBoxLayout(self)
        main_layout.addLayout(left_layout, 1)
        main_layout.addLayout(right_layout, 3)

        self.update_plot()

    def read_geom(self, path: Path) -> pd.DataFrame | None:
        try:
            if path.suffix.lower() == ".csv":
                return pd.read_csv(path)
            return pd.read_parquet(path)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Basemap Error", f"Failed to read geometry {path.name}:\n{exc}")
            return None

    def find_col(self, df, names):
        for n in names:
            if n in df.columns:
                return n
        lowmap = {c.lower(): c for c in df.columns}
        for n in names:
            if n.lower() in lowmap:
                return lowmap[n.lower()]
        return None

    def update_plot(self):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        selected = []
        for i in range(self.list.count()):
            item = self.list.item(i)
            path = item.data(QtCore.Qt.ItemDataRole.UserRole)
            if path is None:
                continue
            if item.checkState() == QtCore.Qt.CheckState.Checked:
                selected.append(path)
        plotted = False
        max_points = 500_000

        for path in selected:
            df = self.read_geom(path)
            if df is None or df.empty:
                continue
            dtype = _infer_dtype(path, self.survey_root)
            is_post = "post" in dtype
            sx_col = self.find_col(df, ["sx"])
            sy_col = self.find_col(df, ["sy"])
            gx_col = self.find_col(df, ["gx"])
            gy_col = self.find_col(df, ["gy"])
            if not is_post and sx_col and sy_col and gx_col and gy_col:
                sx = df[sx_col].to_numpy(dtype=float, copy=False)
                sy = df[sy_col].to_numpy(dtype=float, copy=False)
                gx = df[gx_col].to_numpy(dtype=float, copy=False)
                gy = df[gy_col].to_numpy(dtype=float, copy=False)
                if len(sx) > max_points:
                    idx = np.linspace(0, len(sx) - 1, max_points, dtype=int)
                    sx, sy, gx, gy = sx[idx], sy[idx], gx[idx], gy[idx]
                base_label = path.name.replace(".geometry.parquet", "").replace(".geometry.csv", "")
                if self.showFoldChk.isChecked():
                    try:
                        bins = 200
                        H, xedges, yedges = np.histogram2d(gx, gy, bins=bins)
                        X, Y = np.meshgrid(xedges, yedges)
                        pcm = ax.pcolormesh(X, Y, H.T, cmap="plasma", shading="auto", rasterized=True)
                        cbar = self.figure.colorbar(pcm, ax=ax, fraction=0.046, pad=0.04)
                        cbar.set_label("Fold")
                        pcm.set_label(f"{base_label} fold")
                        plotted = True
                    except Exception:
                        pass
                if self.showReceiversChk.isChecked():
                    ax.scatter(gx, gy, s=2, c="blue", alpha=0.5, rasterized=True, label=f"{base_label} Receivers")
                    plotted = True
                if self.showSourcesChk.isChecked():
                    ax.scatter(sx, sy, s=2, c="red", alpha=0.5, rasterized=True, label=f"{base_label} Sources")
                    plotted = True
            elif is_post and sx_col and sy_col:
                x_vals = df[sx_col].to_numpy(dtype=float, copy=False)
                y_vals = df[sy_col].to_numpy(dtype=float, copy=False)
                x_min, x_max = np.min(x_vals), np.max(x_vals)
                y_min, y_max = np.min(y_vals), np.max(y_vals)
                rect_x = [x_min, x_max, x_max, x_min, x_min]
                rect_y = [y_min, y_min, y_max, y_max, y_min]
                base_label = path.name.replace(".geometry.parquet", "").replace(".geometry.csv", "")
                ax.plot(rect_x, rect_y, linewidth=1.5, linestyle="-", label=f"{base_label} footprint")
                plotted = True

        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_title(f"Basemap: {self.survey_name or ''}")
        ax.grid(True, linestyle="--", alpha=0.3)
        # Draw survey footprint using boundary + azimuth (if available)
        poly = footprint_polygon(self.boundary)
        if poly is not None:
            xs, ys, _, _, _, _ = poly
            ax.plot(xs, ys, color="green", linewidth=1.5, linestyle="--", label="Survey footprint", zorder=1)
            plotted = True
        if plotted:
            handles, labels = ax.get_legend_handles_labels()
            paired = list(zip(labels, handles))
            paired.sort(key=lambda x: (0 if "Survey footprint" in x[0] else 1, x[0]))
            labels_sorted, handles_sorted = zip(*paired)
            ax.legend(handles_sorted, labels_sorted, markerscale=2, loc="upper right")
        self.canvas.draw_idle()


def show_basemap(window):
    """Open the basemap tab for the active survey."""
    if not window.currentSurveyPath:
        QtWidgets.QMessageBox.warning(window, "Warning", "Please select a survey first.")
        return
    geom_dir = Path(window.currentSurveyPath) / "Geometry"
    if not geom_dir.exists():
        QtWidgets.QMessageBox.warning(window, "Warning", "No geometry folder found for the survey.")
        return
    geometry_files = sorted(
        list(geom_dir.glob("*.parquet")) + list(geom_dir.glob("*.csv")),
        key=lambda p: p.name.lower(),
    )
    if not geometry_files:
        QtWidgets.QMessageBox.warning(window, "Warning", "No geometry file found in the survey (Geometry folder).")
        return

    boundary = None
    if window.currentSurveyName:
        try:
            project = next(p for p in list_projects() if p["name"] == window.currentSurveyName)
            metadata = project.get("metadata", {}) or {}
            boundary = metadata.get("boundary", metadata)
        except StopIteration:
            boundary = None

    widget = BasemapDialog(
        window,
        geometry_files,
        boundary,
        Path(window.currentSurveyPath) if window.currentSurveyPath else None,
        window.currentSurveyName,
    )
    window.tabWidget.addTab(widget, "Basemap")
    window.tabWidget.setCurrentWidget(widget)
