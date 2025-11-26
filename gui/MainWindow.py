from SurveyDialogs import NewSurveyDialog, DialogBox, ImportSEGYDialog
from PyQt6 import QtWidgets, uic, QtCore
from PyQt6.QtGui import QIcon, QAction
import sys
import os
import numpy as np
from openseismicprocessing.catalog import init_db, get_workspace_root, set_workspace_root, list_projects
import pandas as pd
from pathlib import Path
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT

class OpenSeismicProcessingWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi("golem.ui", self)  # Load main window UI
        
        init_db()
        stored_root = get_workspace_root()
        # Initialize variables
        self.rootFolderPath = stored_root if stored_root and os.path.isdir(stored_root) else ""
        self.currentSurveyPath = None
        self.currentSurveyName = None
        
        # Find the QAction by its objectName in Qt Designer
        self.actionSetRootFolder = self.findChild(QAction, "action_Set_Root_Folder")
        self.actionSetupSurvey = self.findChild(QAction, "action_Select_Setup")  # Add your QAction here
        self.actionLoadSegy = self.findChild(QAction, "action_Seg_y_file")
        self.actionBaseMap = self.findChild(QAction, "action_Base_Map")
        self.actionHeaders = self.findChild(QAction, "action_Headers")

        # Connect QAction to their respective functions
        self.actionSetRootFolder.triggered.connect(self.SelectRootFolder)
        self.actionSetupSurvey.triggered.connect(self.SetupSurvey)  # Call the dialog
        if self.actionLoadSegy:
            self.actionLoadSegy.triggered.connect(self.LoadSegyFiles)
        if self.actionBaseMap:
            self.actionBaseMap.triggered.connect(self.ShowBasemap)
        if self.actionHeaders:
            self.actionHeaders.triggered.connect(self.ShowHeaders)

        # ✅ Set up an initial empty visualization
        # self.init_empty_visualization()    

    def SelectRootFolder(self):
        """Opens a dialog to select a folder."""
        selected = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Folder")
        if selected:
            self.rootFolderPath = selected
            set_workspace_root(selected)

    def SetupSurvey(self):
        """Opens the custom dialog."""
        
        if not self.rootFolderPath:
            QtWidgets.QMessageBox.warning(self, "Warning", "Please select a root folder first!")
            return
        # Get the list of folder names from the root directory
        self.folder_list = sorted([
           folder for folder in os.listdir(self.rootFolderPath)
           if os.path.isdir(os.path.join(self.rootFolderPath, folder))
       ], key=str.lower)

        dialog = DialogBox(self, selected_survey=self.currentSurveyName)  # Instantiate the custom dialog
        if dialog.exec():  # Show the dialog and wait for it to close
            selected = dialog.GetSelectedSurvey()
            if selected:
                self.currentSurveyName = selected
                self.currentSurveyPath = os.path.join(self.rootFolderPath, selected)
           
        else:
            print("Dialog Closed!")
    
    def LoadSegyFiles(self):
        if not self.rootFolderPath:
            QtWidgets.QMessageBox.warning(self, "Warning", "Please select a root folder first!")
            return
        boundary = None
        if self.currentSurveyName:
            try:
                project = next(p for p in list_projects() if p["name"] == self.currentSurveyName)
                metadata = project.get("metadata", {}) or {}
                boundary = metadata.get("boundary", metadata)
            except StopIteration:
                boundary = None
        dialog = ImportSEGYDialog(boundary=boundary, survey_root=self.currentSurveyPath)
        result = dialog.exec()
        try:
            if result == QtWidgets.QDialog.DialogCode.Accepted:
                self.last_dataset_type = dialog.dataset_type()
        except Exception:
            pass

    def _geometry_files_for_type(self, dataset_type: str) -> list[Path]:
        if not self.currentSurveyPath:
            return []
        geom_dir = Path(self.currentSurveyPath) / "Geometry"
        if not geom_dir.exists():
            return []
        patterns = [
            f"*.{dataset_type}.geometry.parquet",
            f"*.{dataset_type}.geometry.csv",
            "*.geometry.parquet",
            "*.geometry.csv",
            "*.parquet",
            "*.csv",
        ]
        seen = []
        for pat in patterns:
            seen.extend(list(geom_dir.glob(pat)))
        return list({p.resolve() for p in seen})

    def _latest_geometry_file(self) -> Path | None:
        if not self.currentSurveyPath:
            return None
        geom_dir = Path(self.currentSurveyPath) / "Geometry"
        if not geom_dir.exists():
            return None
        candidates = list(geom_dir.glob("*.geometry.parquet")) + list(geom_dir.glob("*.geometry.csv"))
        if not candidates:
            candidates = list(geom_dir.glob("*.parquet")) + list(geom_dir.glob("*.csv"))
        if not candidates:
            return None
        return max(candidates, key=lambda p: p.stat().st_mtime)

    def _read_geometry(self, path: Path) -> pd.DataFrame | None:
        try:
            if path.suffix.lower() == ".csv":
                return pd.read_csv(path)
            return pd.read_parquet(path)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Basemap Error", f"Failed to read geometry file:\n{exc}")
            return None

    def ShowHeaders(self):
        if not self.currentSurveyPath:
            QtWidgets.QMessageBox.warning(self, "Warning", "Please select a survey first.")
            return
        geom_dir = Path(self.currentSurveyPath) / "Geometry"
        if not geom_dir.exists():
            QtWidgets.QMessageBox.warning(self, "Warning", "No geometry folder found for the survey.")
            return
        geometry_files = sorted(
            list(geom_dir.glob("*.parquet")) + list(geom_dir.glob("*.csv")),
            key=lambda p: p.name.lower(),
        )
        if not geometry_files:
            QtWidgets.QMessageBox.warning(self, "Warning", "No geometry file found in the survey (Geometry folder).")
            return

        class HeadersDialog(QtWidgets.QDialog):
            def __init__(self, parent, files):
                super().__init__(parent)
                self.files = files
                self.selected_file = None
                def base_label(p: Path) -> str:
                    return p.name.replace(".geometry.parquet", "").replace(".geometry.csv", "")

                self.setWindowTitle("Select Geometry Dataset")
                self.resize(400, 200)

                layout = QtWidgets.QVBoxLayout(self)
                layout.addWidget(QtWidgets.QLabel("Select dataset:"))
                self.combo = QtWidgets.QComboBox()
                for path in self.files:
                    self.combo.addItem(base_label(path), userData=path)
                layout.addWidget(self.combo)
                layout.addWidget(QtWidgets.QLabel("Rows to show:"))
                self.rowSpin = QtWidgets.QSpinBox()
                self.rowSpin.setRange(1, 1_000_000)
                self.rowSpin.setValue(1000)
                self.rowSpin.setButtonSymbols(QtWidgets.QAbstractSpinBox.ButtonSymbols.NoButtons)
                layout.addWidget(self.rowSpin)
                button_box = QtWidgets.QDialogButtonBox(
                    QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel
                )
                button_box.accepted.connect(self.accept)
                button_box.rejected.connect(self.reject)
                layout.addWidget(button_box)

            def get_selection(self):
                if self.exec() == QtWidgets.QDialog.DialogCode.Accepted:
                    return self.combo.currentData(), int(self.rowSpin.value())
                return None, None

        dlg = HeadersDialog(self, geometry_files)
        geom_path, n_rows = dlg.get_selection()
        if not geom_path:
            return
        try:
            df = pd.read_csv(geom_path) if geom_path.suffix.lower() == ".csv" else pd.read_parquet(geom_path)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Headers Error", f"Failed to read geometry {geom_path.name}:\n{exc}")
            return

        view_df = df.head(n_rows)
        table = QtWidgets.QTableWidget()
        table.setRowCount(len(view_df))
        table.setColumnCount(len(view_df.columns))
        table.setHorizontalHeaderLabels([str(c) for c in view_df.columns])
        for i in range(len(view_df)):
            for j, col in enumerate(view_df.columns):
                table.setItem(i, j, QtWidgets.QTableWidgetItem(str(view_df.iat[i, j])))
        table.resizeColumnsToContents()

        viewer = QtWidgets.QDialog(self)
        viewer.setWindowTitle(f"Headers: {geom_path.name.replace('.geometry.parquet','').replace('.geometry.csv','')}")
        v_layout = QtWidgets.QVBoxLayout(viewer)
        v_layout.addWidget(table)
        viewer.resize(900, 700)
        viewer.exec()


    def ShowBasemap(self):
        if not self.currentSurveyPath:
            QtWidgets.QMessageBox.warning(self, "Warning", "Please select a survey first.")
            return
        geom_dir = Path(self.currentSurveyPath) / "Geometry"
        if not geom_dir.exists():
            QtWidgets.QMessageBox.warning(self, "Warning", "No geometry folder found for the survey.")
            return
        geometry_files = sorted(
            list(geom_dir.glob("*.parquet")) + list(geom_dir.glob("*.csv")),
            key=lambda p: p.name.lower(),
        )
        if not geometry_files:
            QtWidgets.QMessageBox.warning(self, "Warning", "No geometry file found in the survey (Geometry folder).")
            return

        boundary = None
        if self.currentSurveyName:
            try:
                project = next(p for p in list_projects() if p["name"] == self.currentSurveyName)
                metadata = project.get("metadata", {}) or {}
                boundary = metadata.get("boundary", metadata)
            except StopIteration:
                boundary = None

        class BasemapDialog(QtWidgets.QDialog):
            def __init__(self, parent, files, boundary):
                super().__init__(parent)
                self.files = files
                self.boundary = boundary
                self.setWindowTitle("Basemap")
                self.resize(1100, 700)
                self.list = QtWidgets.QListWidget()
                self.list.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.NoSelection)
                def base_label(p: Path) -> str:
                    return p.name.replace(".geometry.parquet", "").replace(".geometry.csv", "")
                for path in self.files:
                    item = QtWidgets.QListWidgetItem(base_label(path))
                    item.setFlags(item.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
                    item.setCheckState(QtCore.Qt.CheckState.Checked)
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
                selected = [
                    self.list.item(i).data(QtCore.Qt.ItemDataRole.UserRole)
                    for i in range(self.list.count())
                    if self.list.item(i).checkState() == QtCore.Qt.CheckState.Checked
                ]
                plotted = False
                max_points = 500_000
                survey_root = Path(self.parent().currentSurveyPath) if self.parent().currentSurveyPath else None
                def infer_dtype(geom_path: Path) -> str:
                    if geom_path is None:
                        return ""
                    base = geom_path.name.split(".geometry")[0]
                    if survey_root:
                        manifest = survey_root / "Binaries" / f"{base}.zarr.manifest.json"
                        if manifest.exists():
                            try:
                                import json
                                data = json.loads(manifest.read_text())
                                return str(data.get("dataset_type", "")).lower()
                            except Exception:
                                return ""
                    return ""
                for path in selected:
                    df = self.read_geom(path)
                    if df is None or df.empty:
                        continue
                    dtype = infer_dtype(path)
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
                ax.set_title(f"Basemap: {self.parent().currentSurveyName or ''}")
                ax.grid(True, linestyle="--", alpha=0.3)
                if self.boundary and "x_range" in self.boundary and "y_range" in self.boundary:
                    x_min, x_max = self.boundary["x_range"]
                    y_min, y_max = self.boundary["y_range"]
                    rect_x = [x_min, x_max, x_max, x_min, x_min]
                    rect_y = [y_min, y_min, y_max, y_max, y_min]
                    ax.plot(rect_x, rect_y, color="green", linewidth=1.5, linestyle="--", label="Survey footprint", zorder=1)
                    plotted = True
                if plotted:
                    handles, labels = ax.get_legend_handles_labels()
                    # make survey footprint first if present
                    paired = list(zip(labels, handles))
                    paired.sort(key=lambda x: (0 if "Survey footprint" in x[0] else 1, x[0]))
                    labels_sorted, handles_sorted = zip(*paired)
                    ax.legend(handles_sorted, labels_sorted, markerscale=2, loc="upper right")
                self.canvas.draw_idle()

        dlg = BasemapDialog(self, geometry_files, boundary)
        dlg.exec()
