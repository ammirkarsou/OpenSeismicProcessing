from SurveyDialogs import NewSurveyDialog, DialogBox, ImportSEGYDialog
from PyQt6 import QtWidgets, uic
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

        # Connect QAction to their respective functions
        self.actionSetRootFolder.triggered.connect(self.SelectRootFolder)
        self.actionSetupSurvey.triggered.connect(self.SetupSurvey)  # Call the dialog
        if self.actionLoadSegy:
            self.actionLoadSegy.triggered.connect(self.LoadSegyFiles)
        if self.actionBaseMap:
            self.actionBaseMap.triggered.connect(self.ShowBasemap)

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
        self.folder_list = [
           folder for folder in os.listdir(self.rootFolderPath)
           if os.path.isdir(os.path.join(self.rootFolderPath, folder))
       ]

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

    def ShowBasemap(self):
        if not self.currentSurveyPath:
            QtWidgets.QMessageBox.warning(self, "Warning", "Please select a survey first.")
            return
        dataset_type = getattr(self, "last_dataset_type", None)
        geom_paths = self._geometry_files_for_type(dataset_type) if dataset_type else []
        if not geom_paths:
            geom = self._latest_geometry_file()
            geom_paths = [geom] if geom else []
        dfs = []
        for gpath in geom_paths:
            df = self._read_geometry(gpath)
            if df is not None and not df.empty:
                dfs.append(df)
        if not dfs:
            QtWidgets.QMessageBox.warning(self, "Warning", "No geometry file found in the survey (Geometry folder).")
            return
        df = pd.concat(dfs, ignore_index=True)

        def find_cols(df, names):
            for n in names:
                if n in df.columns:
                    return n
            # case-insensitive
            lowmap = {c.lower(): c for c in df.columns}
            for n in names:
                if n.lower() in lowmap:
                    return lowmap[n.lower()]
            return None

        sx_col = find_cols(df, ["sx"])
        sy_col = find_cols(df, ["sy"])
        gx_col = find_cols(df, ["gx"])
        gy_col = find_cols(df, ["gy"])
        if not all([sx_col, sy_col, gx_col, gy_col]):
            QtWidgets.QMessageBox.warning(self, "Warning", "Geometry file missing required columns (SourceX/Y, GroupX/Y).")
            return

        sx = df[sx_col].to_numpy(dtype=float, copy=False)
        sy = df[sy_col].to_numpy(dtype=float, copy=False)
        gx = df[gx_col].to_numpy(dtype=float, copy=False)
        gy = df[gy_col].to_numpy(dtype=float, copy=False)

        fig = Figure(figsize=(8, 6))
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)

        # Downsample for plotting if huge
        max_points = 500_000
        if len(sx) > max_points:
            idx = np.linspace(0, len(sx) - 1, max_points, dtype=int)
            sx_plot, sy_plot = sx[idx], sy[idx]
            gx_plot, gy_plot = gx[idx], gy[idx]
        else:
            sx_plot, sy_plot, gx_plot, gy_plot = sx, sy, gx, gy

        ax.scatter(gx_plot, gy_plot, s=2, c="blue", alpha=0.5, rasterized=True, label="Receivers")
        ax.scatter(sx_plot, sy_plot, s=2, c="red", alpha=0.5, rasterized=True, label="Sources")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_title(f"Basemap: {self.currentSurveyName or ''}")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.3)

        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Basemap")
        layout = QtWidgets.QVBoxLayout(dlg)
        toolbar = NavigationToolbar2QT(canvas, dlg)
        layout.addWidget(toolbar)
        layout.addWidget(canvas)
        dlg.resize(900, 700)
        dlg.exec()
