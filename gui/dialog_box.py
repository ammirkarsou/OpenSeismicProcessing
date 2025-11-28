import json
import os
import shutil
from pathlib import Path

import numpy as np
from PyQt6 import QtWidgets
from PyQt6.QtCore import QStringListModel, Qt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

from openseismicprocessing.catalog import delete_project, ensure_project, list_projects, rename_project
from SurveyDialogs import ImportSEGYDialog, ManualSurveyDialog, NewSurveyDialog


class DialogBox(QtWidgets.QDialog):
    """Survey selection/setup dialog."""

    def __init__(self, main, selected_survey: str | None = None):
        super().__init__()
        self.main = main
        self.initial_selection = selected_survey
        self.setWindowTitle("Select/Setup Survey")
        self.resize(640, 500)

        self.figure = Figure(figsize=(3, 3))
        self.canvas = FigureCanvas(self.figure)
        self.range_label = QtWidgets.QLabel()
        self.range_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.range_label.setWordWrap(True)

        self.newSurveyButton = QtWidgets.QPushButton("New")
        self.deleteSurveyButton = QtWidgets.QPushButton("Delete")
        self.renameSurveyButton = QtWidgets.QPushButton("Rename")
        button_row = QtWidgets.QHBoxLayout()
        button_row.addWidget(self.newSurveyButton)
        button_row.addWidget(self.renameSurveyButton)
        button_row.addWidget(self.deleteSurveyButton)
        button_row.addStretch()

        self.surveyListView = QtWidgets.QListView()
        self.model = QStringListModel()
        self.model.setStringList(main.folder_list)
        self.surveyListView.setModel(self.model)

        left_layout = QtWidgets.QVBoxLayout()
        left_layout.addLayout(button_row)
        left_layout.addWidget(self.surveyListView)

        right_layout = QtWidgets.QVBoxLayout()
        right_layout.addWidget(self.canvas, stretch=1)
        right_layout.addWidget(self.range_label)

        main_layout = QtWidgets.QHBoxLayout()
        main_layout.addLayout(left_layout, stretch=1)
        main_layout.addLayout(right_layout, stretch=1)

        button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)

        root_layout = QtWidgets.QVBoxLayout(self)
        root_layout.addLayout(main_layout)
        root_layout.addWidget(button_box)

        self.newSurveyButton.clicked.connect(self.CreateNewSurvey)
        self.deleteSurveyButton.clicked.connect(self.DeleteSurvey)
        self.renameSurveyButton.clicked.connect(self.RenameSurvey)
        self.surveyListView.selectionModel().currentChanged.connect(self._on_survey_selected)
        self.UpdateSurveyList(select_name=self.initial_selection)

    def DeleteSurvey(self):
        selected_survey = self.GetSelectedSurvey()

        if selected_survey is not None:
            folder_path = os.path.join(self.main.rootFolderPath, selected_survey)
            reply = QtWidgets.QMessageBox.question(
                self,
                "Confirm Deletion",
                f"Are you sure you want to delete the survey\n{selected_survey}?",
                QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
                QtWidgets.QMessageBox.StandardButton.No,
            )

            if reply == QtWidgets.QMessageBox.StandardButton.Yes:
                if os.path.exists(folder_path) and os.path.isdir(folder_path):
                    shutil.rmtree(folder_path)
                if selected_survey in self.main.folder_list:
                    self.main.folder_list.remove(selected_survey)

                delete_project(name=selected_survey)

                self.UpdateSurveyList()

    def RenameSurvey(self):
        selected_survey = self.GetSelectedSurvey()
        if not selected_survey:
            QtWidgets.QMessageBox.information(self, "Rename Survey", "Please select a survey first.")
            return
        new_name, ok = QtWidgets.QInputDialog.getText(self, "Rename Survey", "New survey name:", text=selected_survey)
        if not ok or not new_name or new_name == selected_survey:
            return
        old_path = os.path.join(self.main.rootFolderPath, selected_survey)
        new_path = os.path.join(self.main.rootFolderPath, new_name)
        if os.path.exists(new_path):
            QtWidgets.QMessageBox.warning(self, "Rename Survey", "A survey with that name already exists.")
            return
        try:
            os.rename(old_path, new_path)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Rename Survey", f"Failed to rename folder:\n{exc}")
            return
        try:
            rename_project(selected_survey, new_name, Path(new_path))
        except Exception as exc:
            try:
                os.rename(new_path, old_path)
            except Exception:
                pass
            QtWidgets.QMessageBox.warning(self, "Rename Survey", f"Failed to update catalog:\n{exc}")
            return
        if selected_survey in self.main.folder_list:
            self.main.folder_list.remove(selected_survey)
        if new_name not in self.main.folder_list:
            self.main.folder_list.append(new_name)
        self.main.folder_list = sorted(self.main.folder_list, key=str.lower)
        if self.main.currentSurveyName == selected_survey:
            self.main.currentSurveyName = new_name
            self.main.currentSurveyPath = new_path
        self._update_manifest_paths(old_path, new_path)
        self.UpdateSurveyList(select_name=new_name)

    def CreateNewFolders(self):
        os.mkdir(self.surveyPath)
        os.mkdir(self.surveyPath + "/Binaries/")
        os.mkdir(self.surveyPath + "/Geometry/")
        os.mkdir(self.surveyPath + "/JSON/")

    def CreateSurveyFolder(self, survey_name, metadata=None):
        if survey_name:
            self.surveyPath = os.path.join(self.main.rootFolderPath, survey_name)
            if not os.path.exists(self.surveyPath):
                self.CreateNewFolders()

                ensure_project(
                    name=survey_name,
                    root_path=Path(self.surveyPath),
                    description="Survey created via GUI",
                    metadata=metadata or {},
                )

                if survey_name not in self.main.folder_list:
                    self.main.folder_list.append(survey_name)
                    self.main.folder_list = sorted(self.main.folder_list, key=str.lower)

                self.UpdateSurveyList(select_name=survey_name)
                QtWidgets.QMessageBox.information(self, "Success", f"Survey '{survey_name}' created successfully!")
            else:
                QtWidgets.QMessageBox.warning(self, "Warning", "Survey with this name already exists!")

    def CreateNewSurvey(self):
        if not self.main.rootFolderPath:
            QtWidgets.QMessageBox.warning(self, "Warning", "Please select a root folder first!")
            return

        dialog = NewSurveyDialog()
        survey_name = dialog.get_survey_name()
        selected_import = dialog.selected_import
        if not survey_name:
            return

        if selected_import == "Import from SEG-Y":
            SEGYImport = ImportSEGYDialog()
            while True:
                if SEGYImport.exec() == QtWidgets.QDialog.DialogCode.Accepted:
                    child_dialog = None
                    mode = SEGYImport.GetAcquisitionType()
                    if mode == "2D Post-stack":
                        child_dialog = TwoDimensionPostStackSEGYDialog(SEGYImport)
                    elif mode == "3D Post-stack":
                        child_dialog = None
                    else:
                        child_dialog = None

                    if child_dialog is not None and child_dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
                        self.CreateSurveyFolder(survey_name)
                        child_dialog.PassChildClassValues(SEGYImport)
                        reply = QtWidgets.QMessageBox.question(
                            self,
                            "Confirm SEG-Y Import",
                            "Do you wish to import SEG-Y binary?",
                            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
                            QtWidgets.QMessageBox.StandardButton.No,
                        )
                        loadBinary = reply == QtWidgets.QMessageBox.StandardButton.Yes
                        break
                    else:
                        continue
                else:
                    break
        elif selected_import == "Enter by hand":
            manual_dialog = ManualSurveyDialog(self)
            if manual_dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
                values = manual_dialog.get_values()
                metadata = {
                    "boundary": {
                        "x_range": [values["x_min"], values["x_max"]],
                        "y_range": [values["y_min"], values["y_max"]],
                        "inline_start": values["inline_start"],
                        "xline_start": values["xline_start"],
                        "inline_increment": values["inline_increment"],
                        "xline_increment": values["xline_increment"],
                        "azimuth_degrees": values["azimuth"],
                    }
                }
                self.CreateSurveyFolder(survey_name, metadata=metadata)

        self.surveyListView.selectionModel().currentChanged.connect(self._on_survey_selected)
        self._on_survey_selected()

    def _on_survey_selected(self, current=None, prev=None):
        if current and current.isValid():
            selected = current.data()
        else:
            selected = self.GetSelectedSurvey()
        if not selected:
            self.figure.clear()
            self.canvas.draw_idle()
            self.range_label.setText("")
            return
        try:
            project = next(p for p in list_projects() if p["name"] == selected)
        except StopIteration:
            self.figure.clear()
            self.canvas.draw_idle()
            self.range_label.setText("")
            return
        metadata = project.get("metadata", {}).get("boundary")
        if not metadata:
            self.figure.clear()
            self.canvas.draw_idle()
            self.range_label.setText("No boundary metadata stored.")
            return
        x_min, x_max = metadata["x_range"]
        y_min, y_max = metadata["y_range"]
        width = x_max - x_min
        height = y_max - y_min
        center_x = (x_min + x_max) / 2.0
        center_y = (y_min + y_max) / 2.0
        theta = np.deg2rad(metadata.get("azimuth_degrees", 0.0))
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        corners = [
            (-width / 2, -height / 2),
            (width / 2, -height / 2),
            (width / 2, height / 2),
            (-width / 2, height / 2),
            (-width / 2, -height / 2),
        ]
        rotated = [
            (center_x + cos_t * dx - sin_t * dy, center_y + sin_t * dx + cos_t * dy)
            for dx, dy in corners
        ]
        xs = [pt[0] for pt in rotated]
        ys = [pt[1] for pt in rotated]
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.plot(xs, ys, color="tab:red")
        ax.set_xlabel("X (UTM)")
        ax.set_ylabel("Y (UTM)")
        ax.set_title(f"{selected} Footprint")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, linestyle="--", alpha=0.5)
        self.canvas.draw_idle()

        inline_start = metadata.get("inline_start")
        xline_start = metadata.get("xline_start")
        inline_inc = metadata.get("inline_increment")
        xline_inc = metadata.get("xline_increment")
        inline_end = inline_start + (height / inline_inc) if inline_start is not None and inline_inc else "?"
        xline_end = xline_start + (width / xline_inc) if xline_start is not None and xline_inc else "?"

        summary = (
            f"X Range: {x_min:.1f} - {x_max:.1f}\n"
            f"Y Range: {y_min:.1f} - {y_max:.1f}\n"
            f"Inline: {inline_start} - {inline_end}\n"
            f"Crossline: {xline_start} - {xline_end}"
        )
        self.range_label.setText(summary)

    def GetSelectedSurvey(self):
        index = self.surveyListView.currentIndex()
        if index.isValid():
            return index.data()
        return None

    def UpdateSurveyList(self, select_name=None):
        self.model.setStringList(self.main.folder_list)
        if select_name and select_name in self.main.folder_list:
            idx = self.main.folder_list.index(select_name)
            self.surveyListView.setCurrentIndex(self.model.index(idx))

    def _update_manifest_paths(self, old_root: str, new_root: str):
        try:
            old_root_path = Path(old_root).resolve()
            new_root_path = Path(new_root).resolve()
            bin_dir = new_root_path / "Binaries"
            if not bin_dir.exists():
                return
            for manifest_path in bin_dir.glob("*.manifest.json"):
                try:
                    data = json.loads(manifest_path.read_text())
                except Exception:
                    continue
                updated = False
                for key in ("geometry_parquet", "zarr_store"):
                    val = data.get(key)
                    if not val or not isinstance(val, str):
                        continue
                    try:
                        p = Path(val).resolve()
                        rel = p.relative_to(old_root_path)
                        new_val = str(new_root_path / rel)
                    except Exception:
                        if old_root in val:
                            new_val = val.replace(old_root, str(new_root_path))
                        else:
                            continue
                    if new_val != val:
                        data[key] = new_val
                        updated = True
                if updated:
                    try:
                        manifest_path.write_text(json.dumps(data, indent=2))
                    except Exception:
                        pass
        except Exception:
            pass
