#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 18:34:03 2025

@author: ammir
"""
from PyQt6 import QtWidgets, uic, QtCore
from PyQt6.QtCore import QStringListModel, Qt
import os
import shutil
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from uuid import uuid4
import json
import zarr
from numcodecs import Blosc
try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except Exception:
    pa = None
    pq = None
from openseismicprocessing.catalog import ensure_project, delete_project, list_projects, rename_project
from openseismicprocessing.constants import TRACE_HEADER_REV0, TRACE_HEADER_REV1
from openseismicprocessing.segy_inspector import SegyInspector, print_revision_summary
from openseismicprocessing.io import read_trace_headers_until, get_text_header
from openseismicprocessing._io import open_segy_data
from openseismicprocessing.zarr_utils import segy_directory_to_zarr
import segyio
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

class ImportSEGYDialog(QtWidgets.QDialog):
    """Custom QDialog class for creating a new survey or adding data to an existing survey."""
    def __init__(self, boundary=None, survey_root: str | None = None, is_new_survey: bool = False):
        """Build the SEG-Y import dialog, optionally seeded with boundary and root info."""
        super().__init__()
        self.setWindowTitle("Load SEG-Y")
        self.resize(620, 800)
        self.survey_boundary = boundary or {}
        self.survey_root = Path(survey_root) if survey_root else None
        self.is_new_survey = is_new_survey

        main_layout = QtWidgets.QVBoxLayout(self)

        # File path row
        path_layout = QtWidgets.QHBoxLayout()
        path_layout.addWidget(QtWidgets.QLabel("File path"))
        self.lineEdit1 = QtWidgets.QLineEdit()
        path_layout.addWidget(self.lineEdit1, stretch=1)
        self.buttonLoad = QtWidgets.QPushButton("Load")
        path_layout.addWidget(self.buttonLoad)
        main_layout.addLayout(path_layout)

        name_layout = QtWidgets.QHBoxLayout()
        name_layout.addWidget(QtWidgets.QLabel("Dataset name"))
        self.datasetNameEdit = QtWidgets.QLineEdit()
        name_layout.addWidget(self.datasetNameEdit, stretch=1)
        main_layout.addLayout(name_layout)

        slider_layout = QtWidgets.QHBoxLayout()
        self.fileSliderLabel = QtWidgets.QLabel("File 1/1")
        self.fileSlider = QtWidgets.QSlider(Qt.Orientation.Horizontal)
        self.fileSlider.setRange(0, 0)
        self.fileSlider.setEnabled(False)
        self.currentFileNameLabel = QtWidgets.QLabel("")
        slider_layout.addWidget(QtWidgets.QLabel("Browse files"))
        slider_layout.addWidget(self.fileSliderLabel)
        slider_layout.addWidget(self.fileSlider, stretch=1)
        slider_layout.addWidget(self.currentFileNameLabel)
        main_layout.addLayout(slider_layout)

        # Acquisition type
        mode_layout = QtWidgets.QHBoxLayout()
        mode_layout.addWidget(QtWidgets.QLabel("Acquisition"))
        self.comboBoxMode = QtWidgets.QComboBox()
        self.comboBoxMode.addItems([ "2D Pre-stack", "2D Post-stack", "3D Pre-stack", "3D Post-stack"])
        mode_layout.addWidget(self.comboBoxMode)
        self.viewHeader = QtWidgets.QPushButton("View Header")
        mode_layout.addWidget(self.viewHeader)
        main_layout.addLayout(mode_layout)

        form_layout = QtWidgets.QGridLayout()
        form_layout.setHorizontalSpacing(8)
        form_layout.setVerticalSpacing(6)
        form_layout.setColumnStretch(1, 3)

        def add_range_row(row, label_text):
            label = QtWidgets.QLabel(label_text)
            line_edit = QtWidgets.QLineEdit()
            line_edit.setReadOnly(True)
            combo = QtWidgets.QComboBox()
            return label, line_edit, combo

        label_z = QtWidgets.QLabel("Z Range")
        self.lineEdit2 = QtWidgets.QLineEdit()
        self.lineEdit2.setReadOnly(True)
        self.lineEdit2.setMinimumWidth(220)
        self.lineEdit2.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)
        self.spinbox1 = QtWidgets.QDoubleSpinBox()
        self.spinbox1.setDecimals(2)
        self.spinbox1.setRange(-1e9, 1e9)
        self.spinbox1.setValue(0.0)
        self.spinbox1.setButtonSymbols(QtWidgets.QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.spinbox1.setMaximumWidth(90)
        self.spinbox2 = QtWidgets.QDoubleSpinBox()
        self.spinbox2.setDecimals(4)
        self.spinbox2.setRange(-1e9, 1e9)
        self.spinbox2.setValue(1.0)
        self.spinbox2.setButtonSymbols(QtWidgets.QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.spinbox2.setMaximumWidth(90)
        self.comboBoxZUnits = QtWidgets.QComboBox()
        self.comboBoxZUnits.addItems(["meters", "milliseconds", "feet"])
        self.comboBoxHeaderUser = QtWidgets.QComboBox()
        form_layout.addWidget(label_z, 0, 0)
        form_layout.addWidget(self.lineEdit2, 0, 1)
        form_layout.addWidget(self.spinbox1, 0, 2)
        form_layout.addWidget(self.spinbox2, 0, 3)
        form_layout.addWidget(self.comboBoxZUnits, 0, 4)
        form_layout.addWidget(self.comboBoxHeaderUser, 0, 5)
        rev_layout = QtWidgets.QHBoxLayout()
        rev_layout.addWidget(QtWidgets.QLabel("SEGY Revision"))
        self.comboBoxHeaderUser_revision = QtWidgets.QComboBox()
        self.comboBoxHeaderUser_revision.addItems(["0", "1", "2.0", "2.1", "User Defined"])
        rev_layout.addWidget(self.comboBoxHeaderUser_revision)
        rev_layout.addWidget(QtWidgets.QLabel("Traces to view"))
        self.numTracesToShow = QtWidgets.QDoubleSpinBox()
        self.numTracesToShow.setDecimals(0)
        self.numTracesToShow.setRange(1, 1e9)
        self.numTracesToShow.setValue(1000)
        self.numTracesToShow.setButtonSymbols(QtWidgets.QAbstractSpinBox.ButtonSymbols.NoButtons)
        rev_layout.addWidget(self.numTracesToShow)
        rev_layout.addWidget(QtWidgets.QLabel("X/Y Units"))
        self.comboBoxXYUnits = QtWidgets.QComboBox()
        self.comboBoxXYUnits.addItems(["meters", "feet"])
        rev_layout.addWidget(self.comboBoxXYUnits)
        rev_layout.addWidget(QtWidgets.QLabel("Property Unit"))
        self.comboBoxPropertyUnit = QtWidgets.QComboBox()
        self.comboBoxPropertyUnit.addItems(
            ["Amplitude", "P-wave", "Density", "S-Wave", "Epsilon", "Delta", "Theta"]
        )
        rev_layout.addWidget(self.comboBoxPropertyUnit)
        rev_layout.addStretch()
        main_layout.addLayout(rev_layout)

        label_samples = QtWidgets.QLabel("N of Samples")
        self.lineEdit3 = QtWidgets.QLineEdit()
        self.lineEdit3.setReadOnly(True)
        form_layout.addWidget(label_samples, 1, 0)
        form_layout.addWidget(self.lineEdit3, 1, 1, 1, 2)

        inline_label = "Inline Range"
        xline_label = "Crossline Range"
        xrange_label = "X Range"
        yrange_label = "Y Range"
        mode = self.GetAcquisitionType() or ""
        if "Pre-stack" in mode:
            inline_label = "Source X Range"
            xline_label = "Source Y Range"
            xrange_label = "Group X Range"
            yrange_label = "Group Y Range"

        (self.inline_range_label, self.lineEdit4, self.comboBox) = add_range_row(2, inline_label)
        form_layout.addWidget(self.inline_range_label, 2, 0)
        form_layout.addWidget(self.lineEdit4, 2, 1)
        form_layout.addWidget(self.comboBox, 2, 2)

        (self.xline_range_label, self.lineEdit5, self.comboBox2) = add_range_row(3, xline_label)
        form_layout.addWidget(self.xline_range_label, 3, 0)
        form_layout.addWidget(self.lineEdit5, 3, 1)
        form_layout.addWidget(self.comboBox2, 3, 2)

        # Source elevation immediately below source ranges
        (self.source_elev_label, self.lineEditSourceElev, self.comboBoxSourceElev) = add_range_row(4, "Source Elevation Range")
        form_layout.addWidget(self.source_elev_label, 4, 0)
        form_layout.addWidget(self.lineEditSourceElev, 4, 1)
        form_layout.addWidget(self.comboBoxSourceElev, 4, 2)

        (self.x_range_label, self.lineEdit6, self.comboBox3) = add_range_row(5, xrange_label)
        form_layout.addWidget(self.x_range_label, 5, 0)
        form_layout.addWidget(self.lineEdit6, 5, 1)
        form_layout.addWidget(self.comboBox3, 5, 2)

        (self.y_range_label, self.lineEdit7, self.comboBox4) = add_range_row(6, yrange_label)
        form_layout.addWidget(self.y_range_label, 6, 0)
        form_layout.addWidget(self.lineEdit7, 6, 1)
        form_layout.addWidget(self.comboBox4, 6, 2)

        # Group elevation below group Y
        (self.group_elev_label, self.lineEditGroupElev, self.comboBoxGroupElev) = add_range_row(7, "Group Elevation Range")
        form_layout.addWidget(self.group_elev_label, 7, 0)
        form_layout.addWidget(self.lineEditGroupElev, 7, 1)
        form_layout.addWidget(self.comboBoxGroupElev, 7, 2)

        main_layout.addLayout(form_layout)

        self.trace_header_spec = TRACE_HEADER_REV0
        self.scale_options = [key for key in ("scalco", "scalel") if key in self.trace_header_spec]
        if not self.scale_options:
            self.scale_options = list(self.trace_header_spec.keys())[:2]

        # Spinboxes row
        spin_layout = QtWidgets.QGridLayout()

        def add_scale_row(row, label):
            lbl = QtWidgets.QLabel(label)
            combo = QtWidgets.QComboBox()
            combo.addItems(self.scale_options)
            checkbox = QtWidgets.QCheckBox("Manual")
            spin = QtWidgets.QDoubleSpinBox()
            spin.setDecimals(6)
            spin.setRange(-1e9, 1e9)
            spin.setValue(1.0)
            spin.setButtonSymbols(QtWidgets.QAbstractSpinBox.ButtonSymbols.NoButtons)
            spin.setEnabled(False)

            def toggle_manual(checked):
                combo.setEnabled(not checked)
                spin.setEnabled(checked)
            checkbox.toggled.connect(toggle_manual)
            toggle_manual(False)

            spin_layout.addWidget(lbl, row, 0)
            spin_layout.addWidget(combo, row, 1)
            spin_layout.addWidget(checkbox, row, 2)
            spin_layout.addWidget(spin, row, 3)
            return lbl, combo, checkbox, spin

        (self.inlineScaleLabel,
         self.inlineScaleCombo,
         self.inlineManualCheck,
         self.inlineManualSpin) = add_scale_row(0, "Inline Scale")
        (self.xlineScaleLabel,
         self.xlineScaleCombo,
         self.xlineManualCheck,
         self.xlineManualSpin) = add_scale_row(1, "Xline Scale")
        (self.sourceElevScaleLabel,
         self.sourceElevScaleCombo,
         self.sourceElevManualCheck,
         self.sourceElevManualSpin) = add_scale_row(2, "Source Elevation Scale")
        (self.xScaleLabel,
         self.xScaleCombo,
         self.xManualCheck,
         self.xManualSpin) = add_scale_row(3, "X Scale")
        (self.yScaleLabel,
         self.yScaleCombo,
         self.yManualCheck,
         self.yManualSpin) = add_scale_row(4, "Y Scale")
        (self.groupElevScaleLabel,
         self.groupElevScaleCombo,
         self.groupElevManualCheck,
         self.groupElevManualSpin) = add_scale_row(5, "Group Elevation Scale")

        for combo in (
            self.inlineScaleCombo,
            self.xlineScaleCombo,
            self.xScaleCombo,
            self.yScaleCombo,
        ):
            idx = combo.findText("scalco")
            if idx >= 0:
                combo.setCurrentIndex(idx)
        for combo in (
            self.sourceElevScaleCombo,
            self.groupElevScaleCombo,
        ):
            idx = combo.findText("scalel")
            if idx >= 0:
                combo.setCurrentIndex(idx)

        self.inlineManualCheck.toggled.connect(lambda _: self.UpdateInline())
        self.inlineManualSpin.valueChanged.connect(lambda _: self.UpdateInline())
        self.inlineScaleCombo.currentTextChanged.connect(lambda _: self.UpdateInline())
        self.xlineManualCheck.toggled.connect(lambda _: self.UpdateXline())
        self.xlineManualSpin.valueChanged.connect(lambda _: self.UpdateXline())
        self.xlineScaleCombo.currentTextChanged.connect(lambda _: self.UpdateXline())
        self.xManualCheck.toggled.connect(lambda _: self.UpdateXRange())
        self.xManualSpin.valueChanged.connect(lambda _: self.UpdateXRange())
        self.xScaleCombo.currentTextChanged.connect(lambda _: self.UpdateXRange())
        self.yManualCheck.toggled.connect(lambda _: self.UpdateYRange())
        self.yManualSpin.valueChanged.connect(lambda _: self.UpdateYRange())
        self.yScaleCombo.currentTextChanged.connect(lambda _: self.UpdateYRange())

        main_layout.addLayout(spin_layout)

        # Plain text
        self.plainText = QtWidgets.QPlainTextEdit()
        self.plainText.setReadOnly(True)
        self.plainText.setStyleSheet("background-color: lightgray; color: black;")
        main_layout.addWidget(QtWidgets.QLabel("Header Preview"))
        main_layout.addWidget(self.plainText, stretch=1)

        # Buttons
        self.buttonBox = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        main_layout.addWidget(self.buttonBox)

        self.spinbox1.setEnabled(False)
        self.spinbox2.setEnabled(False)

        self.SEGY_Dataframe = None
        self.headers_by_file: list[pd.DataFrame] = []
        self.current_file_idx = 0
        self.inspections_by_file = []
        self.segy_files = []
        self.column_names: list[str] = []
        self.load_column_names()
        self._configure_file_slider()

        # ✅ Connect the signal to track selection changes in real-time
        self.comboBox.currentTextChanged.connect(lambda _: self.UpdateInline())
        self.comboBox2.currentTextChanged.connect(lambda _: self.UpdateXline())
        self.comboBox3.currentTextChanged.connect(lambda _: self.UpdateXRange())
        self.comboBox4.currentTextChanged.connect(lambda _: self.UpdateYRange())
        self.comboBoxHeaderUser.currentTextChanged.connect(self.UpdateZRange)
        self.comboBoxHeaderUser.addItems(["From Header", "From User"])
        self.buttonLoad.clicked.connect(self.ImportSEGYFIle)
        self.viewHeader.clicked.connect(self.ShowHeader)
        self.comboBoxMode.currentTextChanged.connect(lambda _: self.load_column_names())
        self.fileSlider.valueChanged.connect(self.on_file_slider_changed)

        self.numTracesToShow.valueChanged.connect(self.PreviewSegYHeaders)
        self.spinbox1.valueChanged.connect(self.UpdateInitAndDt)
        self.spinbox2.valueChanged.connect(self.UpdateInitAndDt)
        self.comboBoxGroupElev.currentTextChanged.connect(self.UpdateGroupElevRange)
        self.comboBoxSourceElev.currentTextChanged.connect(self.UpdateSourceElevRange)
        self.groupElevManualCheck.toggled.connect(lambda _: self.UpdateGroupElevRange())
        self.groupElevManualSpin.valueChanged.connect(lambda _: self.UpdateGroupElevRange())
        self.groupElevScaleCombo.currentTextChanged.connect(lambda _: self.UpdateGroupElevRange())
        self.sourceElevManualCheck.toggled.connect(lambda _: self.UpdateSourceElevRange())
        self.sourceElevManualSpin.valueChanged.connect(lambda _: self.UpdateSourceElevRange())
        self.sourceElevScaleCombo.currentTextChanged.connect(lambda _: self.UpdateSourceElevRange())

    def _ensure_loaded_path(self) -> bool:
        if not self.segy_files:
            # QtWidgets.QMessageBox.warning(self, "Warning", "Please load at least one SEG-Y file first.")
            return False
        for path in self.segy_files:
            if not os.path.exists(path):
                QtWidgets.QMessageBox.warning(self, "Warning", f"File not found: {path}")
                return False
        return True

    def _configure_file_slider(self):
        count = len(self.headers_by_file) if self.headers_by_file else len(self.segy_files)
        max_idx = max(0, count - 1)
        self.fileSlider.blockSignals(True)
        self.fileSlider.setRange(0, max_idx)
        self.fileSlider.setValue(min(self.current_file_idx, max_idx))
        self.fileSlider.setEnabled(count > 1)
        self.fileSlider.blockSignals(False)
        if count:
            safe_idx = min(self.current_file_idx, max_idx)
            self.fileSliderLabel.setText(f"File {safe_idx + 1}/{count}")
            actual_idx = self._active_file_index()
            if actual_idx is not None and self.segy_files and actual_idx < len(self.segy_files):
                name = os.path.basename(self.segy_files[actual_idx])
                self.currentFileNameLabel.setText(name)
            else:
                self.currentFileNameLabel.setText("")
        else:
            self.fileSliderLabel.setText("No file loaded")
            self.currentFileNameLabel.setText("")

    def _active_file_index(self) -> int | None:
        if not self.headers_by_file:
            return None
        frame = self.headers_by_file[min(self.current_file_idx, len(self.headers_by_file) - 1)]
        if isinstance(frame, pd.DataFrame) and "file_index" in frame.columns and not frame.empty:
            try:
                return int(frame["file_index"].iloc[0])
            except Exception:
                return None
        if self.segy_files and self.current_file_idx < len(self.segy_files):
            return self.current_file_idx
        return None

    def _apply_current_file_headers(self):
        if not self.headers_by_file:
            self.SEGY_Dataframe = None
            return
        self.current_file_idx = max(0, min(self.current_file_idx, len(self.headers_by_file) - 1))
        self.fileSlider.blockSignals(True)
        self.fileSlider.setValue(self.current_file_idx)
        self.fileSlider.blockSignals(False)
        self.fileSliderLabel.setText(f"File {self.current_file_idx + 1}/{len(self.headers_by_file)}")
        file_idx = self._active_file_index()
        if file_idx is not None and self.segy_files and file_idx < len(self.segy_files):
            name = os.path.basename(self.segy_files[file_idx])
            self.currentFileNameLabel.setText(name)
        else:
            self.currentFileNameLabel.setText("")
        self.SEGY_Dataframe = self.headers_by_file[self.current_file_idx]
        self.load_column_names(preserve_selection=True)
        self.ShowMinMaxTraceValues()

    def on_file_slider_changed(self, value: int):
        self.current_file_idx = value
        self._apply_current_file_headers()

    def _read_basic_counts(self, file_index: int | None = None):
        if not self._ensure_loaded_path():
            return None
        if file_index is not None and (file_index < 0 or file_index >= len(self.segy_files)):
            return None
        target_paths = self.segy_files if file_index is None else [self.segy_files[file_index]]
        total_traces = 0
        dt = None
        n_samples = None
        for path in target_paths:
            segy_file = None
            try:
                segy_file = open_segy_data(path, ignore_geometry=True)
                total_traces += segy_file.tracecount
                if n_samples is None:
                    n_samples = len(segy_file.samples)
                if dt is None:
                    try:
                        dt = segyio.tools.dt(segy_file) / 1000.0
                    except Exception:
                        dt = 0.0
            except Exception as e:
                QtWidgets.QMessageBox.warning(self, "Error", f"Failed to read SEG-Y metadata from {path}:\n{e}")
                return None
            finally:
                if segy_file is not None:
                    try:
                        segy_file.close()
                    except Exception:
                        pass
        return total_traces, dt or 0.0, n_samples or 0

    def _load_trace_headers(self, max_traces=None, per_file: bool = False):
        if not self._ensure_loaded_path():
            return None

        frames = []
        remaining = max_traces

        for idx, path in enumerate(self.segy_files):
            limit = None
            if remaining is not None and not per_file:
                if remaining <= 0:
                    break
                limit = remaining
            elif max_traces is not None:
                limit = max_traces

            try:
                headers_dict = read_trace_headers_until(path, num_traces=limit)
            except Exception as exc:
                QtWidgets.QMessageBox.warning(
                    self, "Error", f"Failed to read SEG-Y trace headers from {path}:\n{exc}"
                )
                continue

            if not headers_dict:
                continue

            # Use float32 where possible to reduce memory/IO
            headers_df = pd.DataFrame({k: pd.Series(v, dtype="float32") for k, v in headers_dict.items()})
            headers_df["source_file"] = os.path.basename(path)
            headers_df["file_index"] = idx
            frames.append(headers_df)

            if remaining is not None and not per_file:
                remaining -= len(headers_df)

        if not frames:
            QtWidgets.QMessageBox.warning(self, "Error", "Failed to read SEG-Y trace headers.")
            return None

        if per_file:
            return frames
        return pd.concat(frames, ignore_index=True)

    def _current_header_spec(self):
        """Choose the appropriate trace-header spec (rev0/rev1) based on inspection."""
        spec = TRACE_HEADER_REV0
        if self.inspections_by_file and 0 <= self.current_file_idx < len(self.inspections_by_file):
            insp = self.inspections_by_file[self.current_file_idx]
            rev_name = ""
            rev_value = ""
            major = None
            minor = None
            if insp is not None:
                try:
                    rev_obj = getattr(insp, "classified_revision", None)
                    rev_name = getattr(rev_obj, "name", "") or ""
                    rev_value = getattr(rev_obj, "value", "") or ""
                except Exception:
                    rev_name = ""
                    rev_value = ""
                try:
                    major = getattr(insp, "major", None)
                except Exception:
                    major = None
                try:
                    minor = getattr(insp, "minor", None)
                except Exception:
                    minor = None
            name_match = rev_name in ("REV1", "REV2_PLUS")
            value_match = "rev 1" in rev_value.lower() or "rev 2" in rev_value.lower()
            major_minor_match = (major is not None and major >= 1) or (major == 0 and minor == 1)
            if name_match or value_match or major_minor_match:
                spec = TRACE_HEADER_REV1
        self.trace_header_spec = spec
        return spec

    def _unit_factor(self) -> float:
        """Return multiplier to convert X/Y units combo to meters (feet -> meters)."""
        try:
            unit_text = self.comboBoxXYUnits.currentText().lower()
        except Exception:
            unit_text = "meters"
        if "feet" in unit_text or "ft" in unit_text:
            return 0.3048
        return 1.0

    @staticmethod
    def _apply_coordinate_scale(values: pd.Series, scale) -> np.ndarray:
        """Apply SEG-Y style scaler array to a coordinate series."""
        arr = values.to_numpy(dtype=float, copy=False)
        scale_arr = np.asarray(scale, dtype=float)
        if scale_arr.ndim == 0:
            raise ValueError("Scaler must be provided per trace, not as a single value.")
        if scale_arr.shape[0] != arr.shape[0]:
            raise ValueError("Scaler array length does not match value array length.")
        factors = np.ones_like(scale_arr, dtype=float)
        pos = scale_arr > 0
        neg = scale_arr < 0
        factors[pos] = scale_arr[pos]
        factors[neg] = 1.0 / np.abs(scale_arr[neg])
        return arr * factors

    def _resolve_scale_for_file(self, column: str, manual_checkbox, manual_spin, segy_file):
        """Resolve a single scalar for a file from scaler header or manual value."""
        if manual_checkbox.isChecked():
            return manual_spin.value() or 1.0
        # Map known scalar fields
        scalar_map = {
            "scalco": getattr(segyio.TraceField, "SourceGroupScalar", None),
            "scalel": getattr(segyio.TraceField, "ElevationScalar", None),
        }
        tf = scalar_map.get(column.lower())
        if tf is None:
            return 1.0
        try:
            val = float(segy_file.attributes(tf)[0])
        except Exception:
            return 1.0
        return val if val not in (0, None) else 1.0

    def _resolve_scale(self, combo, manual_checkbox, manual_spin):
        """Resolve a per-trace scale array from the selected scaler column/manual."""
        if self.SEGY_Dataframe is None or self.SEGY_Dataframe.empty:
            raise ValueError("No headers loaded to resolve scale.")
        length = len(self.SEGY_Dataframe)
        if manual_checkbox.isChecked():
            val = manual_spin.value() or 1.0
            return np.full(length, val, dtype=float)
        column = combo.currentText()
        if not column or column not in self.SEGY_Dataframe.columns:
            raise ValueError("Selected scale column not present in headers.")
        values = self.SEGY_Dataframe[column].to_numpy(dtype=float, copy=False)
        if values.shape[0] != length:
            raise ValueError("Scaler column length mismatch.")
        return values

    def _resolve_scale_array_for_file(self, column: str, manual_checkbox, manual_spin, segy_file, length: int):
        """Return per-trace scale factors (with SEG-Y negative rule) as an array."""
        def factor_from_value(v):
            if v in (None, 0):
                return 1.0
            return v if v > 0 else 1.0 / abs(v)

        if manual_checkbox.isChecked():
            val = manual_spin.value() or 1.0
            return np.full(length, factor_from_value(val), dtype=np.float32)

        scalar_map = {
            "scalco": getattr(segyio.TraceField, "SourceGroupScalar", None),
            "scalel": getattr(segyio.TraceField, "ElevationScalar", None),
        }
        tf = scalar_map.get(column.lower())
        if tf is None:
            return np.ones(length, dtype=np.float32)
        try:
            vals = np.asarray(segy_file.attributes(tf)[:length], dtype=np.float32)
        except Exception:
            return np.ones(length, dtype=np.float32)
        factors = np.ones_like(vals, dtype=np.float32)
        pos = vals > 0
        neg = vals < 0
        factors[pos] = vals[pos]
        factors[neg] = 1.0 / np.abs(vals[neg])
        return factors

    def _extract_boundary(self):
        """Return (x_min, x_max, y_min, y_max) if available, else None."""
        if not self.survey_boundary:
            return None

        boundary = self.survey_boundary
        if "boundary" in boundary and isinstance(boundary["boundary"], dict):
            boundary = boundary["boundary"]

        def first_present(*keys):
            for key in keys:
                if key in boundary:
                    return boundary[key]
            return None

        x_range = first_present("x_range", "X_range")
        y_range = first_present("y_range", "Y_range")
        x_min = first_present("x_min", "xmin", "X_min")
        x_max = first_present("x_max", "xmax", "X_max")
        y_min = first_present("y_min", "ymin", "Y_min")
        y_max = first_present("y_max", "ymax", "Y_max")

        if x_range and len(x_range) == 2:
            x_min, x_max = x_range
        if y_range and len(y_range) == 2:
            y_min, y_max = y_range

        if None in (x_min, x_max, y_min, y_max):
            return None

        return float(x_min), float(x_max), float(y_min), float(y_max)

    def _scale_array_from_df(self, df: pd.DataFrame, scale_name: str, manual_checkbox, manual_spin) -> np.ndarray:
        """Return per-trace scaling factors from a dataframe column or manual value."""
        n = len(df)
        factors = np.ones(n, dtype=float)

        def to_factor(val: float) -> float:
            if val in (None, 0):
                return 1.0
            return val if val > 0 else 1.0 / abs(val)

        if manual_checkbox.isChecked():
            val = manual_spin.value() or 1.0
            factors.fill(to_factor(val))
            return factors

        if not scale_name or scale_name not in df.columns:
            return factors

        arr = df[scale_name].to_numpy(dtype=float, copy=False)
        if len(arr) != n:
            return factors
        pos = arr > 0
        neg = arr < 0
        factors[pos] = arr[pos]
        factors[neg] = 1.0 / np.abs(arr[neg])
        return factors

    def _coord_scale_map(self, segy_file) -> dict[str, float]:
        """Determine scale per selected coordinate header for the given SEG-Y file."""
        mode = self.GetAcquisitionType() or ""
        # In pre-stack, Source X/Y and Group X/Y are all coordinates that can be scaled.
        if "Pre-stack" in mode:
            cols_and_scales = [
                (self.comboBox, self.inlineScaleCombo, self.inlineManualCheck, self.inlineManualSpin),   # Source X
                (self.comboBox2, self.xlineScaleCombo, self.xlineManualCheck, self.xlineManualSpin),     # Source Y
                (self.comboBox3, self.xScaleCombo, self.xManualCheck, self.xManualSpin),                 # Group X
                (self.comboBox4, self.yScaleCombo, self.yManualCheck, self.yManualSpin),                 # Group Y
            ]
        else:
            # In post-stack, inline/crossline are not scaled; only X/Y coordinates use scalers.
            cols_and_scales = [
                (self.comboBox3, self.xScaleCombo, self.xManualCheck, self.xManualSpin),  # X
                (self.comboBox4, self.yScaleCombo, self.yManualCheck, self.yManualSpin),  # Y
            ]
        out: dict[str, float] = {}
        for combo, scale_combo, chk, spin in cols_and_scales:
            col = self._current_data(combo)
            if not col:
                continue
            raw_val = self._resolve_scale_for_file(scale_combo.currentText(), chk, spin, segy_file)
            out[col] = self._scalar_factor(raw_val)
        return out

    def _coord_scale_arrays(self, segy_file, length: int) -> dict[str, np.ndarray]:
        """Per-trace scale factors for selected coordinate headers."""
        mode = self.GetAcquisitionType() or ""
        if "Pre-stack" in mode:
            cols_and_scales = [
                (self.comboBox, self.inlineScaleCombo, self.inlineManualCheck, self.inlineManualSpin),   # Source X
                (self.comboBox2, self.xlineScaleCombo, self.xlineManualCheck, self.xlineManualSpin),     # Source Y
                (self.comboBox3, self.xScaleCombo, self.xManualCheck, self.xManualSpin),                 # Group X
                (self.comboBox4, self.yScaleCombo, self.yManualCheck, self.yManualSpin),                 # Group Y
            ]
        else:
            # Post-stack: only X/Y get scaling.
            cols_and_scales = [
                (self.comboBox3, self.xScaleCombo, self.xManualCheck, self.xManualSpin),  # X
                (self.comboBox4, self.yScaleCombo, self.yManualCheck, self.yManualSpin),  # Y
            ]
        out: dict[str, np.ndarray] = {}
        for combo, scale_combo, chk, spin in cols_and_scales:
            col = self._current_data(combo)
            if not col:
                continue
            scale_arr = self._resolve_scale_array_for_file(
                scale_combo.currentText(), chk, spin, segy_file, length
            )
            out[col] = scale_arr
        return out

    def _count_prestack_outside(self, frames: list[pd.DataFrame] | None = None, collect_inside: bool = False):
        """Return (outside_count, total[, inside_trace_ids]) or None."""
        bounds = self._extract_boundary()
        if bounds is None:
            QtWidgets.QMessageBox.information(
                self, "Boundary Missing", "No survey bounding box is available to check traces."
            )
            return None
        x_min, x_max, y_min, y_max = bounds

        # Map combos to columns/scales: Source = inline/xline combos, Group = x/y combos in pre-stack mode
        cols_and_scales = [
            (self.comboBox, self.inlineScaleCombo, self.inlineManualCheck, self.inlineManualSpin),
            (self.comboBox2, self.xlineScaleCombo, self.xlineManualCheck, self.xlineManualSpin),
            (self.comboBox3, self.xScaleCombo, self.xManualCheck, self.xManualSpin),
            (self.comboBox4, self.yScaleCombo, self.yManualCheck, self.yManualSpin),
        ]

        tf_map = {
            "sx": getattr(segyio.TraceField, "SourceX", None),
            "sy": getattr(segyio.TraceField, "SourceY", None),
            "gx": getattr(segyio.TraceField, "GroupX", None),
            "gy": getattr(segyio.TraceField, "GroupY", None),
        }

        total_outside = 0
        total_traces = 0
        inside_ids = [] if collect_inside else None
        unit_factor = self._unit_factor()

        import time
        progress = QtWidgets.QProgressDialog("Checking traces against bounding box...", "Cancel", 0, 0, self)
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setAutoClose(True)
        progress.setAutoReset(True)
        processed = 0
        total_count = 0

        for path in self.segy_files:
            try:
                print(f"[TraceCheck] Opening {path} for boundary check...")
                with open_segy_data(path, ignore_geometry=True) as f:
                    n_traces = f.tracecount
                    total_count += n_traces
                    progress.setMaximum(max(total_count, 1))
                    chunk = 50000
                    # Determine scales per file
                    scales = []
                    for _, scale_combo, chk, spin in cols_and_scales:
                        scales.append(self._resolve_scale_for_file(scale_combo.currentText(), chk, spin, f))

                    for start in range(0, n_traces, chunk):
                        end = min(start + chunk, n_traces)
                        loop_start = time.time()
                        series_list = []
                        for idx, (combo, _, _, _) in enumerate(cols_and_scales):
                            # Use raw header name (UserRole), not the decorated label,
                            # so we can match 'sx','sy','gx','gy' correctly.
                            col_name = self._current_data(combo).lower()
                            tf = tf_map.get(col_name)
                            if tf is None:
                                QtWidgets.QMessageBox.warning(
                                    self,
                                    "Header Missing",
                                    f"Column '{col_name}' not supported for pre-stack check.",
                                )
                                progress.close()
                                return None
                            try:
                                vals = np.asarray(f.attributes(tf)[start:end], dtype=float)
                            except Exception:
                                QtWidgets.QMessageBox.warning(
                                    self, "Header Read Error", f"Failed to read column '{col}' from SEG-Y."
                                )
                                progress.close()
                                return None
                            scale = scales[idx]
                            if scale in (None, 0):
                                scaled = vals
                            elif scale > 0:
                                scaled = vals * scale
                            else:
                                scaled = vals / abs(scale)
                            if unit_factor != 1.0:
                                scaled = scaled * unit_factor
                            series_list.append(scaled)

                        sx, sy, gx, gy = series_list
                        mask_out = (
                            (sx < x_min) | (sx > x_max)
                            | (sy < y_min) | (sy > y_max)
                            | (gx < x_min) | (gx > x_max)
                            | (gy < y_min) | (gy > y_max)
                        )
                        total_outside += int(mask_out.sum())
                        total_traces += len(sx)
                        processed += len(sx)
                        if collect_inside:
                            local_inside = np.nonzero(~mask_out)[0] + start + (total_traces - len(sx))
                            inside_ids.append(local_inside.astype(np.int64))
                        print(f"[TraceCheck] Processed {end}/{n_traces} traces in {path} (chunk took {time.time()-loop_start:.2f}s)")
                        progress.setValue(processed)
                        QtWidgets.QApplication.processEvents()
                        if progress.wasCanceled():
                            progress.close()
                            return None
            except Exception as exc:
                QtWidgets.QMessageBox.warning(self, "Error", f"Failed to process {path}:\n{exc}")
                progress.close()
                return None

        progress.close()
        if collect_inside:
            concat_ids = np.concatenate(inside_ids) if inside_ids else np.array([], dtype=np.int64)
            return total_outside, total_traces, concat_ids
        return total_outside, total_traces

    def _all_headers(self) -> list[str]:
        """Return the list of header names to store based on current revision spec."""
        spec = self._current_header_spec()
        return list(spec.keys())

    def _dataset_type_slug(self) -> str:
        """Return a short slug like '3d-pre' or '2d-post' from the acquisition combo."""
        mode = (self.GetAcquisitionType() or "").lower()
        dim = "2d" if "2d" in mode else "3d" if "3d" in mode else "unknown"
        stack = "pre" if "pre" in mode else "post" if "post" in mode else "unknown"
        return f"{dim}-{stack}"

    def dataset_type(self) -> str:
        """Public accessor for the dataset type slug."""
        return self._dataset_type_slug()

    def _ensure_output_paths(self) -> tuple[Path, Path]:
        """Build/create Binaries/Geometry directories and base filenames for this import."""
        base_dir = self.survey_root
        if base_dir is None:
            dlg = QtWidgets.QFileDialog(self, "Select output folder for SEG-Y import")
            dlg.setFileMode(QtWidgets.QFileDialog.FileMode.Directory)
            dlg.setOptions(QtWidgets.QFileDialog.Option.DontUseNativeDialog | QtWidgets.QFileDialog.Option.ShowDirsOnly)
            if not dlg.exec():
                return None, None
            sel = dlg.selectedFiles()
            if not sel:
                return None, None
            base_dir = Path(sel[0])
        # If a target root was provided but doesn't exist yet, create it
        base_dir.mkdir(parents=True, exist_ok=True)
        binaries_dir = base_dir / "Binaries"
        geometry_dir = base_dir / "Geometry"
        binaries_dir.mkdir(parents=True, exist_ok=True)
        geometry_dir.mkdir(parents=True, exist_ok=True)
        dataset_type = self._dataset_type_slug()
        user_name = (self.datasetNameEdit.text() or "").strip()
        base_name = user_name if user_name else (dataset_type if dataset_type else "segy_import")
        zarr_name = f"{base_name}.zarr"
        geom_name = f"{base_name}.geometry.parquet"
        return binaries_dir / zarr_name, geometry_dir / geom_name

    def _write_subset_to_zarr(self, trace_ids: np.ndarray, zarr_path: Path, geom_path: Path) -> bool:
        """Write a subset of traces and corresponding geometry into a new Zarr store."""
        headers = self._all_headers()
        unit_factor = self._unit_factor()
        if pa is None or pq is None:
            geom_path = geom_path.with_suffix(".csv")
        try:
            ns = None
            # Determine ns from first file
            with open_segy_data(self.segy_files[0], ignore_geometry=True) as f0:
                ns = len(f0.samples)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Import Error", f"Failed to read sample count:\n{exc}")
            return False
        compressor = None
        try:
            compressor = Blosc(cname="zstd", clevel=5, shuffle=Blosc.SHUFFLE)
        except Exception:
            compressor = None
        # scales for selected coord headers (per file)
        try:
            with open_segy_data(self.segy_files[0], ignore_geometry=True) as ftmp:
                coord_scales = self._coord_scale_map(ftmp)
        except Exception:
            coord_scales = {}

        root = zarr.open(zarr_path, mode="w-")
        root.attrs.update(
            {
                "description": "Subset of SEG-Y traces",
                "dataset_type": self._dataset_type_slug(),
                "created_at": datetime.utcnow().isoformat() + "Z",
            }
        )
        amp_out = root.create_dataset(
            "amplitude",
            shape=(ns, len(trace_ids)),
            chunks=(ns, 2048),
            dtype="float32",
            compressor=compressor,
        )
        amp_out.attrs["_ARRAY_DIMENSIONS"] = ["sample", "trace"]

        arrow_writer = None
        pending_frames: list[pd.DataFrame] = []

        sorted_ids = np.asarray(np.sort(trace_ids), dtype=np.int64)
        offset_out = 0
        cumulative = 0
        lower_map = {h.lower(): h for h in headers}

        for file_id, path in enumerate(self.segy_files):
            with open_segy_data(path, ignore_geometry=True) as f:
                ntr = f.tracecount
                mask = (sorted_ids >= cumulative) & (sorted_ids < cumulative + ntr)
                if not np.any(mask):
                    cumulative += ntr
                    continue
                local_ids = sorted_ids[mask] - cumulative
                # read traces
                data = np.asarray(f.trace.raw[local_ids], dtype=np.float32)
                end_out = offset_out + data.shape[0]
                amp_out[:, offset_out:end_out] = data.T

                geom = {}
                spec = self._current_header_spec()
                for name in headers:
                    if name not in spec:
                        continue
                    offset_bytes, _ = spec[name]
                    try:
                        vals = f.attributes(offset_bytes)[local_ids]
                    except Exception:
                        continue
                    geom[name] = np.asarray(vals, dtype=np.float32)

                trace_id = sorted_ids[mask]
                trace_in_file = local_ids.astype(np.int32)
                file_ids = np.full(len(local_ids), file_id, dtype=np.int32)
                data_dict = {"trace_id": trace_id, "file_id": file_ids, "trace_in_file": trace_in_file}
                def _scale_array(val_arr, length: int):
                    base = np.ones(length, dtype=np.float32)
                    if val_arr is None:
                        return base
                    v = np.asarray(val_arr, dtype=np.float32).flatten()
                    if v.size == 1:
                        scalar = v[0]
                        if scalar == 0:
                            return base
                        return np.full(length, scalar if scalar > 0 else 1.0 / abs(scalar), dtype=np.float32)
                    if v.size != length:
                        v = np.resize(v, length)
                    pos = v > 0
                    neg = v < 0
                    base[pos] = v[pos]
                    base[neg] = 1.0 / np.abs(v[neg])
                    return base

            scale_map = self._coord_scale_arrays(f, len(local_ids))
            scale_map_lower = {str(k).lower(): v for k, v in (scale_map or {}).items()}
            for hname, vals in geom.items():
                vals_arr = np.asarray(vals, dtype=np.float32)
                key = str(hname).lower()
                factors = _scale_array(scale_map_lower.get(key), len(vals_arr))
                vals_arr = vals_arr * factors
                # Apply unit factor only to coordinate-like columns
                if unit_factor != 1.0 and key in ("sx", "sy", "gx", "gy", "x", "y", "cdpx", "cdpy"):
                    vals_arr = vals_arr * unit_factor
                data_dict[hname] = vals_arr

        if pa is not None and pq is not None:
            table = pa.Table.from_pydict(data_dict)
            if arrow_writer is None:
                arrow_writer = pq.ParquetWriter(geom_path, table.schema)
            arrow_writer.write_table(table)
        else:
            pending_frames.append(pd.DataFrame(data_dict))

        offset_out = end_out
        cumulative += ntr

        if arrow_writer is not None:
            arrow_writer.close()
        elif pending_frames:
            df_all = pd.concat(pending_frames, ignore_index=True)
            try:
                df_all.to_parquet(geom_path)
            except Exception:
                csv_path = geom_path.with_suffix(".csv")
                df_all.to_csv(csv_path, index=False)
                geom_path = csv_path
        return True

    def _load_and_save_traces(self, trace_ids: np.ndarray | None = None):
        """Main import routine: stream SEG-Y to Zarr + geometry (full or subset)."""
        zarr_path, geom_path = self._ensure_output_paths()
        if not zarr_path or not geom_path:
            return False
        progress = QtWidgets.QProgressDialog("Importing SEG-Y into Zarr...", "Cancel", 0, 100, self)
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setAutoClose(True)
        progress.setAutoReset(True)
        progress.setValue(0)
        progress_max = {"total": 0}
        progress.show()

        def progress_cb(processed: int, total: int, current_file: str | None):
            if total and progress_max["total"] != total:
                progress.setMaximum(total)
                progress_max["total"] = total
            progress.setValue(processed)
            if current_file:
                progress.setLabelText(f"Importing {Path(current_file).name} ({processed}/{max(total,1)})")
            QtWidgets.QApplication.processEvents()
            if progress.wasCanceled():
                raise RuntimeError("Import cancelled by user")

        try:
            if trace_ids is None:
                coord_scales_func = lambda path, f: self._coord_scale_arrays(f, f.tracecount)
                segy_directory_to_zarr(
                    {},
                    segy_input=self.segy_files,
                    zarr_out=zarr_path,
                    headers=self._all_headers(),
                    chunk_trace=2048,
                    geometry_out=geom_path,
                    dataset_type=self._dataset_type_slug(),
                    allow_overwrite=True,
                    allow_append=False,
                    unit_factor=self._unit_factor(),
                    coord_scales=coord_scales_func,
                    selected_headers=self._selected_headers_metadata(),
                    progress_cb=progress_cb,
                )
            else:
                if not self._write_subset_to_zarr(trace_ids, zarr_path, geom_path):
                    return False
                manifest = {
                    "dataset_id": str(uuid4()),
                    "parent_store": "",
                    "zarr_store": str(zarr_path.resolve()),
                    "geometry_parquet": str(geom_path.resolve()),
                    "dataset_type": self._dataset_type_slug(),
                    "trace_count": int(len(trace_ids)),
                    "created_at": datetime.utcnow().isoformat() + "Z",
                    "selected_headers": self._selected_headers_metadata(),
                }
                manifest_path = Path(str(zarr_path) + ".manifest.json")
                manifest_path.write_text(json.dumps(manifest, indent=2))
        except RuntimeError:
            progress.close()
            QtWidgets.QMessageBox.information(self, "Import Cancelled", "SEG-Y import was cancelled by the user.")
            return False
        except Exception as exc:
            progress.close()
            QtWidgets.QMessageBox.warning(self, "Import Error", f"Failed to store SEG-Y into Zarr:\n{exc}")
            return False
        progress.close()
        QtWidgets.QMessageBox.information(self, "SEG-Y Imported", f"Saved traces to:\n{zarr_path}\nGeometry to:\n{geom_path}")
        return True

    def _selected_headers_metadata(self) -> dict:
        """Collect selected headers, scaled ranges and z sampling into a metadata dict."""
        def scaled_range(column_attr: str, scale_attr: str):
            col = getattr(self, column_attr, None)
            scale = getattr(self, scale_attr, None)
            if col and hasattr(self, "SEGY_Dataframe") and isinstance(self.SEGY_Dataframe, pd.DataFrame):
                try:
                    scaled = self._apply_coordinate_scale(self.SEGY_Dataframe[col], scale)
                    return [float(np.nanmin(scaled)), float(np.nanmax(scaled))]
                except Exception:
                    return None
            return None

        meta = {
            "inline_header": getattr(self, "selected_inline_column", None),
            "xline_header": getattr(self, "selected_xline_column", None),
            "x_header": getattr(self, "selected_xrange_column", None),
            "y_header": getattr(self, "selected_yrange_column", None),
            "inline_scaler_header": self.inlineScaleCombo.currentText() if hasattr(self, "inlineScaleCombo") else None,
            "xline_scaler_header": self.xlineScaleCombo.currentText() if hasattr(self, "xlineScaleCombo") else None,
            "x_scaler_header": self.xScaleCombo.currentText() if hasattr(self, "xScaleCombo") else None,
            "y_scaler_header": self.yScaleCombo.currentText() if hasattr(self, "yScaleCombo") else None,
        }
        xr = scaled_range("selected_xrange_column", "xScale")
        yr = scaled_range("selected_yrange_column", "yScale")
        ir = scaled_range("selected_inline_column", "ilineScale") if hasattr(self, "ilineScale") else None
        xrng = scaled_range("selected_xline_column", "xlineScale") if hasattr(self, "xlineScale") else None
        if xr:
            meta["x_range"] = xr
        if yr:
            meta["y_range"] = yr
        if ir:
            meta["inline_range"] = ir
        if xrng:
            meta["xline_range"] = xrng
        z_start = getattr(self, "init", 0.0)
        z_inc = getattr(self, "dt", 1.0)
        meta["z_start"] = float(z_start)
        meta["z_increment"] = float(z_inc)
        if hasattr(self, "n_samples") and self.n_samples:
            meta["z_end"] = float(z_start + (self.n_samples - 1) * z_inc)

        mode = self.GetAcquisitionType() if hasattr(self, "GetAcquisitionType") else ""
        is_pre = "Pre-stack" in mode
        is_post = "Post-stack" in mode
        meta["source_x_header"] = getattr(self, "selected_inline_column", None) if is_pre else None
        meta["source_y_header"] = getattr(self, "selected_xline_column", None) if is_pre else None
        meta["group_x_header"] = getattr(self, "selected_xrange_column", None) if is_pre else None
        meta["group_y_header"] = getattr(self, "selected_yrange_column", None) if is_pre else None
        meta["post_inline_header"] = getattr(self, "selected_inline_column", None) if is_post else None
        meta["post_xline_header"] = getattr(self, "selected_xline_column", None) if is_post else None
        meta["post_x_header"] = getattr(self, "selected_xrange_column", None) if is_post else None
        meta["post_y_header"] = getattr(self, "selected_yrange_column", None) if is_post else None
        return meta

    def GetAcquisitionType(self):
        """Return acquisition type string from combo (e.g. '3D Post-stack')."""
        return self.comboBoxMode.currentText()

    @staticmethod
    def _label_with_bytes(name: str, spec: dict) -> str:
        """Format a header name with its byte range, e.g. 'sx (Bytes 73-76)'."""
        if name in spec:
            start, length = spec[name]
            end = start + length - 1
            return f"{name} (Bytes {start}-{end})"
        return name

    @staticmethod
    def _set_combo_items(combo: QtWidgets.QComboBox, columns, spec: dict):
        combo.clear()
        for c in columns:
            label = ImportSEGYDialog._label_with_bytes(c, spec)
            combo.addItem(label, userData=c)

    @staticmethod
    def _set_current_by_data(combo: QtWidgets.QComboBox, target: str):
        if target is None:
            return
        for i in range(combo.count()):
            if combo.itemData(i, QtCore.Qt.ItemDataRole.UserRole) == target:
                combo.setCurrentIndex(i)
                return

    @staticmethod
    def _current_data(combo: QtWidgets.QComboBox) -> str:
        data = combo.currentData(QtCore.Qt.ItemDataRole.UserRole)
        return data if data else combo.currentText()

    @staticmethod
    def _scalar_factor(val: float) -> float:
        """Return scalar using SEG-Y rule (negative => reciprocal)."""
        if val in (None, 0):
            return 1.0
        return val if val > 0 else 1.0 / abs(val)

    def UpdateInline(self):
        """Handle inline/source X header change and update range preview (scaled for pre-stack)."""
        self.selected_inline_column = self._current_data(self.comboBox)

        if self.SEGY_Dataframe is not None and isinstance(self.SEGY_Dataframe, pd.DataFrame):
            values = self.SEGY_Dataframe[self.selected_inline_column]
            mode = self.GetAcquisitionType() or ""
            try:
                if "Pre-stack" in mode:
                    scale_arr = self._resolve_scale(self.inlineScaleCombo, self.inlineManualCheck, self.inlineManualSpin)
                    scaled = self._apply_coordinate_scale(values, scale_arr)
                    init = float(np.nanmin(scaled))
                    end = float(np.nanmax(scaled))
                    text = "%.2f - %.2f" % (init, end)
                else:
                    init = np.min(values)
                    end = np.max(values)
                    text = "%i - %i" % (init, end)
            except Exception:
                init = np.min(values)
                end = np.max(values)
                text = "%i - %i" % (init, end)
            self.lineEdit4.setText(text)

    def UpdateXline(self):
        """Handle crossline/source Y header change and update range preview (scaled for pre-stack)."""
        self.selected_xline_column = self._current_data(self.comboBox2)

        if self.SEGY_Dataframe is not None and isinstance(self.SEGY_Dataframe, pd.DataFrame):
            values = self.SEGY_Dataframe[self.selected_xline_column]
            mode = self.GetAcquisitionType() or ""
            try:
                if "Pre-stack" in mode:
                    scale_arr = self._resolve_scale(self.xlineScaleCombo, self.xlineManualCheck, self.xlineManualSpin)
                    scaled = self._apply_coordinate_scale(values, scale_arr)
                    init = float(np.nanmin(scaled))
                    end = float(np.nanmax(scaled))
                    text = "%.2f - %.2f" % (init, end)
                else:
                    init = np.min(values)
                    end = np.max(values)
                    text = "%i - %i" % (init, end)
            except Exception:
                init = np.min(values)
                end = np.max(values)
                text = "%i - %i" % (init, end)
            self.lineEdit5.setText(text)
    
    def UpdateXRange(self):
        self.selected_xrange_column = self._current_data(self.comboBox3)
        self.xScale = self._resolve_scale(self.xScaleCombo, self.xManualCheck, self.xManualSpin)

        if self.SEGY_Dataframe is not None and isinstance(self.SEGY_Dataframe, pd.DataFrame):
            scaled = self._apply_coordinate_scale(self.SEGY_Dataframe[self.selected_xrange_column], self.xScale)
            init = np.min(scaled)
            end = np.max(scaled)
            text = "%.2f - %.2f" % (init,end)
            self.lineEdit6.setText(text)

    def UpdateYRange(self):
        self.selected_yrange_column = self._current_data(self.comboBox4)
        self.yScale = self._resolve_scale(self.yScaleCombo, self.yManualCheck, self.yManualSpin)

        if self.SEGY_Dataframe is not None and isinstance(self.SEGY_Dataframe, pd.DataFrame):
            scaled = self._apply_coordinate_scale(self.SEGY_Dataframe[self.selected_yrange_column], self.yScale)
            init = np.min(scaled)
            end = np.max(scaled)
            text = "%.2f - %.2f" % (init,end)
            self.lineEdit7.setText(text)

    def UpdateGroupElevRange(self):
        self.selected_group_elev_column = self._current_data(self.comboBoxGroupElev)
        self.groupElevScale = self._resolve_scale(
            self.groupElevScaleCombo, self.groupElevManualCheck, self.groupElevManualSpin
        )
        if self.SEGY_Dataframe is not None and isinstance(self.SEGY_Dataframe, pd.DataFrame):
            vals = self.SEGY_Dataframe[self.selected_group_elev_column]
            scaled = self._apply_coordinate_scale(vals, self.groupElevScale)
            init = np.nanmin(scaled)
            end = np.nanmax(scaled)
            text = "%.2f - %.2f" % (init, end)
            self.lineEditGroupElev.setText(text)

    def UpdateSourceElevRange(self):
        self.selected_source_elev_column = self._current_data(self.comboBoxSourceElev)
        self.sourceElevScale = self._resolve_scale(
            self.sourceElevScaleCombo, self.sourceElevManualCheck, self.sourceElevManualSpin
        )
        if self.SEGY_Dataframe is not None and isinstance(self.SEGY_Dataframe, pd.DataFrame):
            vals = self.SEGY_Dataframe[self.selected_source_elev_column]
            scaled = self._apply_coordinate_scale(vals, self.sourceElevScale)
            init = np.nanmin(scaled)
            end = np.nanmax(scaled)
            text = "%.2f - %.2f" % (init, end)
            self.lineEditSourceElev.setText(text)

    def UpdateInitAndDt(self):
        self.init = self.spinbox1.value()
        self.dt = self.spinbox2.value()
        text = "%i - %.2f - %.2f" % (self.init,self.init + (self.n_samples-1)*self.dt,self.dt) 
        self.lineEdit2.setText(text)
    
    def UpdateZRange(self):
        choice = self.comboBoxHeaderUser.currentText()

        counts = self._read_basic_counts(file_index=self._active_file_index())
        if counts is None:
            return
        self.n_traces, self.dt, self.n_samples = counts
        
        if choice == "From Header":
            self.spinbox1.setEnabled(False)
            self.spinbox2.setEnabled(False)
            
            text = "%i - %.2f - %.2f" % (0,(self.n_samples-1)*self.dt,self.dt)    
            self.lineEdit2.setText(text)
        
        elif choice == "From User":
            
            self.spinbox1.setEnabled(True)
            self.spinbox2.setEnabled(True)

            self.UpdateInitAndDt()
            text = "%i - %.2f - %.2f" % (self.init,self.init + (self.n_samples-1)*self.dt,self.dt) 
            self.lineEdit2.setText(text)

    def ShowHeader(self):
        if self.SEGY_Dataframe is not None and isinstance(self.SEGY_Dataframe, pd.DataFrame):
            dialog = QtWidgets.QDialog(self)
            dialog.setWindowTitle("DataFrame Contents")
            layout = QtWidgets.QVBoxLayout()

            # Create a table widget to display the DataFrame
            table = QtWidgets.QTableWidget()
            table.setRowCount(self.SEGY_Dataframe.shape[0])
            table.setColumnCount(self.SEGY_Dataframe.shape[1])
            table.setHorizontalHeaderLabels(self.SEGY_Dataframe.columns)

            # Populate the table with DataFrame values
            for i in range(self.SEGY_Dataframe.shape[0]):
                for j in range(self.SEGY_Dataframe.shape[1]):
                    table.setItem(i, j, QtWidgets.QTableWidgetItem(str(self.SEGY_Dataframe.iat[i, j])))

            # Resize columns to fit content
            table.resizeColumnsToContents()
            table.resizeRowsToContents()

            layout.addWidget(table)
            dialog.setLayout(layout)
            dialog.resize(900, 600)
            dialog.exec()


    def ShowMinMaxTraceValues(self):
        counts = self._read_basic_counts(file_index=self._active_file_index())
        if counts is None:
            return
        self.n_traces, self.dt, self.n_samples = counts

        # for z range
        text = "%i - %.2f - %.2f" % (0,(self.n_samples-1)*self.dt,self.dt)
        self.lineEdit2.setText(text)

        # for n samples
        text = "%i (%i traces)" % (self.n_samples,self.n_traces)
        self.lineEdit3.setText(text)
        
        if self.comboBox.count():
            self.UpdateInline()
        if self.comboBox2.count():
            self.UpdateXline()
        if self.comboBox3.count():
            self.UpdateXRange()
        if self.comboBox4.count():
            self.UpdateYRange()
        if self.comboBoxGroupElev.count():
            self.UpdateGroupElevRange()
        if self.comboBoxSourceElev.count():
            self.UpdateSourceElevRange()

    def load_column_names(self, preserve_selection: bool = False):
        """Populate combo boxes with sensible defaults based on acquisition type."""
        if self.comboBoxMode is None:
            return
        mode = self.GetAcquisitionType() or ""
        spec = self._current_header_spec()

        columns = []
        prev_selection = (
            self._current_data(self.comboBox),
            self._current_data(self.comboBox2),
            self._current_data(self.comboBox3),
            self._current_data(self.comboBox4),
            self._current_data(self.comboBoxGroupElev),
            self._current_data(self.comboBoxSourceElev),
        ) if preserve_selection else (None, None, None, None, None, None)

        if self.SEGY_Dataframe is not None and not self.SEGY_Dataframe.empty:
            columns = list(dict.fromkeys(self.SEGY_Dataframe.columns))
        else:
            def keys_for_mode(names):
                return [name for name in names if name in spec]
            if "3D Pre-stack" in mode or "2D Pre-stack" in mode:
                columns = keys_for_mode(["sx", "sy", "gx", "gy"])
            else:
                columns = keys_for_mode(["cdp", "cdpt", "cdpx", "cdpy"])
            if not columns:
                columns = list(spec.keys())[:4]

        # Ensure elevation headers appear even before SEG-Y is loaded
        if "Pre-stack" in mode:
            if "gelev" not in columns:
                columns.insert(0, "gelev")
            if "selev" not in columns:
                columns.insert(0, "selev")

        self.comboBox.blockSignals(True)
        self.comboBox2.blockSignals(True)
        self.comboBox3.blockSignals(True)
        self.comboBox4.blockSignals(True)
        self.comboBoxGroupElev.blockSignals(True)
        self.comboBoxSourceElev.blockSignals(True)
        self.sourceElevScaleCombo.blockSignals(True)
        self.groupElevScaleCombo.blockSignals(True)
        # Populate combos with labels including byte ranges
        spec = self.trace_header_spec
        self._set_combo_items(self.comboBox, columns, spec)
        self._set_combo_items(self.comboBox2, columns, spec)
        self._set_combo_items(self.comboBox3, columns, spec)
        self._set_combo_items(self.comboBox4, columns, spec)
        self._set_combo_items(self.comboBoxGroupElev, columns, spec)
        self._set_combo_items(self.comboBoxSourceElev, columns, spec)
        defaults = [0, 1, 2, 3, 0, 0]
        combos = [
            self.comboBox,
            self.comboBox2,
            self.comboBox3,
            self.comboBox4,
            self.comboBoxGroupElev,
            self.comboBoxSourceElev,
        ]
        pre_defaults = ["sx", "sy", "gx", "gy", "gelev", "selev"]
        for idx, combo in enumerate(combos):
            if preserve_selection and prev_selection[idx] in columns:
                self._set_current_by_data(combo, prev_selection[idx])
                continue
            if "3D Pre-stack" in mode or "2D Pre-stack" in mode:
                target = pre_defaults[idx]
                if target in columns:
                    self._set_current_by_data(combo, target)
                    continue
            if combo.count() > defaults[idx]:
                combo.setCurrentIndex(defaults[idx])
            elif combo.count() > 0:
                combo.setCurrentIndex(0)
        # Explicit defaults for elevation headers when available
        if "gelev" in columns:
            self._set_current_by_data(self.comboBoxGroupElev, "gelev")
        if "selev" in columns:
            self._set_current_by_data(self.comboBoxSourceElev, "selev")
        # Force selev when in pre-stack modes even if a previous selection was kept
        if ("Pre-stack" in mode) and ("selev" in columns):
            self._set_current_by_data(self.comboBoxSourceElev, "selev")

        if "3D Pre-stack" in mode or "2D Pre-stack" in mode:
            range_labels = ("Source X Range", "Source Y Range", "Group X Range", "Group Y Range")
            scale_labels = ("Source X Scale", "Source Y Scale", "Group X Scale", "Group Y Scale")
        else:
            range_labels = ("Inline Range", "Crossline Range", "X Range", "Y Range")
            scale_labels = ("Inline Scale", "Crossline Scale", "CDP X Scale", "CDP Y Scale")

        for label_widget, text in zip(
            (self.inline_range_label, self.xline_range_label, self.x_range_label, self.y_range_label),
            range_labels,
        ):
            label_widget.setText(text)
        # Elevation rows visible only for pre-stack
        is_prestack = "Pre-stack" in mode
        for widget in (
            self.group_elev_label,
            self.lineEditGroupElev,
            self.comboBoxGroupElev,
            self.source_elev_label,
            self.lineEditSourceElev,
            self.comboBoxSourceElev,
        ):
            widget.setVisible(is_prestack)

        for label_widget, text in zip(
            (self.inlineScaleLabel, self.xlineScaleLabel, self.xScaleLabel, self.yScaleLabel),
            scale_labels,
        ):
            label_widget.setText(text)

        # Hide inline/xline scales for 3D post-stack
        is_3d_post = "3D Post-stack" in mode
        for widget in (
            self.inlineScaleLabel,
            self.inlineScaleCombo,
            self.inlineManualCheck,
            self.inlineManualSpin,
            self.xlineScaleLabel,
            self.xlineScaleCombo,
            self.xlineManualCheck,
            self.xlineManualSpin,
        ):
            widget.setVisible(not is_3d_post)

        # Elevation scales visible only for pre-stack
        for widget in (
            self.sourceElevScaleLabel,
            self.sourceElevScaleCombo,
            self.sourceElevManualCheck,
            self.sourceElevManualSpin,
            self.groupElevScaleLabel,
            self.groupElevScaleCombo,
            self.groupElevManualCheck,
            self.groupElevManualSpin,
        ):
            widget.setVisible(is_prestack)

        self.comboBox.blockSignals(False)
        self.comboBox2.blockSignals(False)
        self.comboBox3.blockSignals(False)
        self.comboBox4.blockSignals(False)
        self.comboBoxGroupElev.blockSignals(False)
        self.comboBoxSourceElev.blockSignals(False)
        self.sourceElevScaleCombo.blockSignals(False)
        self.groupElevScaleCombo.blockSignals(False)
        self.column_names = columns
        
    def ShowSEGYTextHeader(self):
        if not self._ensure_loaded_path():
            return
        header_dict = get_text_header({}, self.segy_files[0])
        if not header_dict:
            self.plainText.setPlainText("Unable to read text header.")
            return
        formatted_text = "\n".join(str(value) for value in header_dict.values())
        self.plainText.setPlainText(formatted_text)

    def _set_revision_from_inspection(self, inspection) -> bool:
        """Map SegyInspector result to the revision combo (0 or 1, fallback to 2.* or blank)."""
        try:
            classified = getattr(inspection, "classified_revision", None)
            classified_name = getattr(classified, "name", "") or ""
            classified_value = getattr(classified, "value", "") or ""

            def set_to(text: str) -> bool:
                for idx in range(self.comboBoxHeaderUser_revision.count()):
                    if self.comboBoxHeaderUser_revision.itemText(idx).startswith(text):
                        self.comboBoxHeaderUser_revision.setCurrentIndex(idx)
                        return True
                return False

            if "REV1" in classified_name or "REV 1" in classified_value.upper():
                return set_to("1")
            if "REV0" in classified_name or "REV 0" in classified_value.upper():
                return set_to("0")
            if "REV2" in classified_name or "REV 2" in classified_value.upper():
                return set_to("2")

            textual = getattr(inspection, "textual_revision", "") or ""
        except Exception:
            textual = ""
        txt = textual.strip().lower()
        try:
            if txt.startswith("1") or txt.startswith("rev 1"):
                return set_to("1")
            if txt.startswith("0") or txt.startswith("rev 0"):
                return set_to("0")
            if txt.startswith("2") or txt.startswith("rev 2"):
                return set_to("2")
        except Exception:
            pass
        return False

    def ImportSEGYFIle(self):
        self.SEGY_Dataframe = None
        dlg = QtWidgets.QFileDialog(self, "Select SEG-Y Files")
        dlg.setFileMode(QtWidgets.QFileDialog.FileMode.ExistingFiles)
        dlg.setNameFilter("SEG-Y Files (*.sgy *.segy)")
        dlg.setOptions(QtWidgets.QFileDialog.Option.DontUseNativeDialog)
        paths = dlg.selectedFiles() if dlg.exec() else []

        if not paths:
            return

        self.segy_files = paths
        self.lineEdit1.setText("; ".join(paths))
        self.comboBoxHeaderUser_revision.setCurrentIndex(-1)
        self.current_file_idx = 0
        self._configure_file_slider()

        revision_set = False
        self.inspections_by_file = []
        for segy_path in self.segy_files:
            try:
                inspection = SegyInspector.inspect(segy_path)
                print_revision_summary(inspection)
                self.inspections_by_file.append(inspection)
                revision_set = self._set_revision_from_inspection(inspection) or revision_set
            except Exception as exc:
                print(f"[SEG-Y Inspector] Failed to inspect {segy_path}: {exc}")
                self.inspections_by_file.append(None)
        if not revision_set:
            self.comboBoxHeaderUser_revision.setCurrentIndex(-1)

        self.load_column_names(preserve_selection=True)

        self.ShowSEGYTextHeader()
        preview_limit = min(self.GetNumOfTracesToShow(), 200)
        header_frames = self._load_trace_headers(max_traces=preview_limit, per_file=True)
        if header_frames:
            self.headers_by_file = header_frames
            self._configure_file_slider()
            self._apply_current_file_headers()
        else:
            self.headers_by_file = []
            self.SEGY_Dataframe = None
                
    def GetNumOfTracesToShow(self):
        return int(self.numTracesToShow.value())
    
    def PreviewSegYHeaders(self):
        numTraces = self.GetNumOfTracesToShow()
        prev_idx = self.current_file_idx
        frames = self._load_trace_headers(max_traces=numTraces, per_file=True)
        if not frames:
            return
        self.headers_by_file = frames
        self.current_file_idx = min(prev_idx, len(frames) - 1)
        self._configure_file_slider()
        self._apply_current_file_headers()

    def accept(self):
        mode = self.GetAcquisitionType() or ""

        # Pre-stack data
        if "Pre-stack" in mode:
            # New survey: derive boundary from scaled source/group coordinates and import all traces
            if self.is_new_survey:
                headers = self._load_trace_headers(max_traces=None, per_file=False)
                if headers is None or headers.empty:
                    QtWidgets.QMessageBox.warning(self, "Trace Check", "Failed to read headers for boundary check.")
                    return
                sx_col = getattr(self, "selected_inline_column", None)
                sy_col = getattr(self, "selected_xline_column", None)
                gx_col = getattr(self, "selected_xrange_column", None)
                gy_col = getattr(self, "selected_yrange_column", None)
                try:
                    sx_vals = headers[sx_col].to_numpy(dtype=float, copy=False)
                    sy_vals = headers[sy_col].to_numpy(dtype=float, copy=False)
                    gx_vals = headers[gx_col].to_numpy(dtype=float, copy=False)
                    gy_vals = headers[gy_col].to_numpy(dtype=float, copy=False)
                except Exception:
                    QtWidgets.QMessageBox.warning(self, "Trace Check", "Missing Source/Group X/Y columns.")
                    return
                # Apply scalers to all four coordinate sets
                sx_scale = self._scale_array_from_df(
                    headers, self.inlineScaleCombo.currentText(), self.inlineManualCheck, self.inlineManualSpin
                )
                sy_scale = self._scale_array_from_df(
                    headers, self.xlineScaleCombo.currentText(), self.xlineManualCheck, self.xlineManualSpin
                )
                gx_scale = self._scale_array_from_df(
                    headers, self.xScaleCombo.currentText(), self.xManualCheck, self.xManualSpin
                )
                gy_scale = self._scale_array_from_df(
                    headers, self.yScaleCombo.currentText(), self.yManualCheck, self.yManualSpin
                )
                sx_scaled = sx_vals * sx_scale
                sy_scaled = sy_vals * sy_scale
                gx_scaled = gx_vals * gx_scale
                gy_scaled = gy_vals * gy_scale

                # Overall survey extents from both source and group coordinates
                x_min = float(np.nanmin([sx_scaled.min(), gx_scaled.min()]))
                x_max = float(np.nanmax([sx_scaled.max(), gx_scaled.max()]))
                y_min = float(np.nanmin([sy_scaled.min(), gy_scaled.min()]))
                y_max = float(np.nanmax([sy_scaled.max(), gy_scaled.max()]))

                # Store boundary in the same metadata structure used elsewhere
                self.survey_boundary = {
                    "boundary": {
                        "x_range": [x_min, x_max],
                        "y_range": [y_min, y_max],
                        "x_min": x_min,
                        "x_max": x_max,
                        "y_min": y_min,
                        "y_max": y_max,
                    }
                }
                if not self._load_and_save_traces():
                    return
                super().accept()
                return

            # Existing survey: run pre-stack boundary check against stored bounds
            import time
            t0 = time.time()
            print("[TraceCheck] Starting pre-stack boundary check...")
            result = self._count_prestack_outside()
            if result is None:
                print("[TraceCheck] Boundary check aborted (missing bounds or cancelled).")
                return
            outside, total = result
            print(f"[TraceCheck] Completed in {time.time()-t0:.2f}s. Outside={outside}, Total={total}")
            if outside == 0:
                QtWidgets.QMessageBox.information(
                    self, "Trace Check", "All traces are inside the survey bounding box."
                )
                if not self._load_and_save_traces():
                    return
            else:
                reply = QtWidgets.QMessageBox.question(
                    self,
                    "Trace Check",
                    f"{outside} of {total} traces are outside the survey bounding box.\n"
                    f"Load the {total - outside} inside traces?",
                    QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
                    QtWidgets.QMessageBox.StandardButton.No,
                )
                if reply != QtWidgets.QMessageBox.StandardButton.Yes:
                    return
                # Re-run collecting inside ids
                result_inside = self._count_prestack_outside(collect_inside=True)
                if result_inside is None:
                    return
                _, _, inside_ids = result_inside
                if inside_ids.size == 0:
                    QtWidgets.QMessageBox.information(self, "Trace Check", "No traces inside the bounding box to load.")
                    return
                if not self._load_and_save_traces(trace_ids=inside_ids):
                    return
            super().accept()
            return


        # Post-stack: boundary check using sx/sy only
        if not self.is_new_survey:
            bounds = self._extract_boundary()
            if bounds is None:
                QtWidgets.QMessageBox.information(self, "Boundary Missing", "No survey bounding box is available to check traces.")
                return
            x_min, x_max, y_min, y_max = bounds
            # load headers to check
            headers = self._load_trace_headers(max_traces=None, per_file=False)
            if headers is None or headers.empty:
                QtWidgets.QMessageBox.warning(self, "Trace Check", "Failed to read headers for boundary check.")
                return
            sx_col = self.selected_xrange_column
            sy_col = self.selected_yrange_column
            try:
                sx_vals = headers[sx_col].to_numpy(dtype=float, copy=False)
                sy_vals = headers[sy_col].to_numpy(dtype=float, copy=False)
            except Exception:
                QtWidgets.QMessageBox.warning(self, "Trace Check", "Missing X/Y columns for boundary check.")
                return
            sx_scale = self._scale_array_from_df(headers, self.xScaleCombo.currentText(), self.xManualCheck, self.xManualSpin)
            sy_scale = self._scale_array_from_df(headers, self.yScaleCombo.currentText(), self.yManualCheck, self.yManualSpin)
            sx_scaled = sx_vals * sx_scale
            sy_scaled = sy_vals * sy_scale
            mask_out = (sx_scaled < x_min) | (sx_scaled > x_max) | (sy_scaled < y_min) | (sy_scaled > y_max)
            outside = int(mask_out.sum())
            total = len(sx_vals)
            if outside == 0:
                QtWidgets.QMessageBox.information(self, "Trace Check", "All traces are inside the survey bounding box.")
                if not self._load_and_save_traces():
                    return
            else:
                reply = QtWidgets.QMessageBox.question(
                    self,
                    "Trace Check",
                    f"{outside} of {total} traces are outside the survey bounding box.\n"
                    f"Load the {total - outside} inside traces?",
                    QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
                    QtWidgets.QMessageBox.StandardButton.No,
                )
                if reply != QtWidgets.QMessageBox.StandardButton.Yes:
                    return
                inside_ids = np.nonzero(~mask_out)[0].astype(np.int64)
                if inside_ids.size == 0:
                    QtWidgets.QMessageBox.information(self, "Trace Check", "No traces inside the bounding box to load.")
                    return
                if not self._load_and_save_traces(trace_ids=inside_ids):
                    return
        else:

            # load headers to check
            headers = self._load_trace_headers(max_traces=None, per_file=False)
            if headers is None or headers.empty:
                QtWidgets.QMessageBox.warning(self, "Trace Check", "Failed to read headers for boundary check.")
                return
            sx_col = self.selected_xrange_column
            sy_col = self.selected_yrange_column
            try:
                sx_vals = headers[sx_col].to_numpy(dtype=float, copy=False)
                sy_vals = headers[sy_col].to_numpy(dtype=float, copy=False)
            except Exception:
                QtWidgets.QMessageBox.warning(self, "Trace Check", "Missing X/Y columns for boundary check.")
                return
            sx_scale = self._scale_array_from_df(headers, self.xScaleCombo.currentText(), self.xManualCheck, self.xManualSpin)
            sy_scale = self._scale_array_from_df(headers, self.yScaleCombo.currentText(), self.yManualCheck, self.yManualSpin)
            sx_scaled = sx_vals * sx_scale
            sy_scaled = sy_vals * sy_scale

            x_min = sx_scaled.min()
            x_max = sx_scaled.max()
            y_min = sy_scaled.min()
            y_max = sy_scaled.max()

            # A better center for a rotated survey: mean of all points
            center_x = float(np.nanmean(sx_scaled))
            center_y = float(np.nanmean(sy_scaled))

            # Derive inline/xline ranges and azimuth if headers are present
            inline_range = None
            xline_range = None
            inline_inc = None
            xline_inc = None
            azimuth_deg = None
            
            try:
                il_col = getattr(self, "selected_inline_column", None)
                xl_col = getattr(self, "selected_xline_column", None)

                il_vals = (
                    headers[il_col].to_numpy(dtype=float, copy=False)
                    if il_col is not None and il_col in headers
                    else None
                )
                xl_vals = (
                    headers[xl_col].to_numpy(dtype=float, copy=False)
                    if xl_col is not None and xl_col in headers
                    else None
                )

                # Need both inline and xline to find corners
                if il_vals is not None and xl_vals is not None and il_vals.size and xl_vals.size:
                    valid = (
                        ~np.isnan(il_vals)
                        & ~np.isnan(xl_vals)
                        & ~np.isnan(sx_scaled)
                        & ~np.isnan(sy_scaled)
                    )
                    ilv = il_vals[valid]
                    xlv = xl_vals[valid]
                    sxv = sx_scaled[valid]
                    syv = sy_scaled[valid]

                    # Inline/xline numeric ranges and increments
                    if ilv.size:
                        inline_range = [float(np.min(ilv)), float(np.max(ilv))]
                        uniq_il = np.unique(ilv)
                        if uniq_il.size > 1:
                            inline_inc = float(
                                np.median(np.diff(np.sort(uniq_il)))
                            )
                    if xlv.size:
                        xline_range = [float(np.min(xlv)), float(np.max(xlv))]
                        uniq_xl = np.unique(xlv)
                        if uniq_xl.size > 1:
                            xline_inc = float(
                                np.median(np.diff(np.sort(uniq_xl)))
                            )

                    # --- 2. Build inline/xline -> (sx, sy) centroids ---
                    df = pd.DataFrame(
                        {"il": ilv, "xl": xlv, "sx": sxv, "sy": syv}
                    )
                    cent = (
                        df.groupby(["il", "xl"])[["sx", "sy"]]
                        .mean()
                        .reset_index()
                    )

                    il_min = float(np.min(ilv))
                    il_max = float(np.max(ilv))
                    xl_min = float(np.min(xlv))
                    xl_max = float(np.max(xlv))

                    def _corner(ili: float, xli: float) -> tuple[float, float]:
                        sub = cent[(cent["il"] == ili) & (cent["xl"] == xli)]
                        if sub.empty:
                            raise ValueError(
                                f"Missing corner at inline={ili}, xline={xli}"
                            )
                        return float(sub["sx"].iloc[0]), float(sub["sy"].iloc[0])

                    # Corners in (inline,xline) index space:
                    # (il_min, xl_min), (il_max, xl_min), (il_max, xl_max), (il_min, xl_max)
                    p00 = _corner(il_min, xl_min)
                    p10 = _corner(il_max, xl_min)
                    p11 = _corner(il_max, xl_max)
                    p01 = _corner(il_min, xl_max)

                    corners = [p00, p10, p11, p01]  # consistent ordering around the cube

                    # Override center with average of corners (more robust for tilted survey)
                    center_x = (p00[0] + p10[0] + p11[0] + p01[0]) / 4.0
                    center_y = (p00[1] + p10[1] + p11[1] + p01[1]) / 4.0

                    # --- 3. Inline direction from corners -> azimuth ---
                    # Inline edges: p00->p10 and p01->p11 (same direction ideally)
                    dE_il = 0.5 * ((p10[0] - p00[0]) + (p11[0] - p01[0]))
                    dN_il = 0.5 * ((p10[1] - p00[1]) + (p11[1] - p01[1]))

                    # Azimuth (0° = North, clockwise), using UTM (E,N):
                    # az = atan2(ΔE, ΔN)
                    if dE_il != 0.0 or dN_il != 0.0:
                        az = np.degrees(np.arctan2(dE_il, dN_il))
                        if az < 0.0:
                            az += 360.0
                        azimuth_deg = float(az)

            except Exception:
                # If anything fails above, we still keep x/y ranges and center
                pass

            # --- 4. Store everything in survey_boundary, including corners ---
            self.survey_boundary = {
                "boundary": {
                    "x_range": [x_min, x_max],
                    "y_range": [y_min, y_max],
                    "x_min": x_min,
                    "x_max": x_max,
                    "y_min": y_min,
                    "y_max": y_max,
                    "center_x": center_x,
                    "center_y": center_y,
                    "inline_range": inline_range,
                    "xline_range": xline_range,
                    "inline_increment": inline_inc,
                    "xline_increment": xline_inc,
                    "corners": corners,              # list of 4 (x, y)
                    "azimuth_degrees": azimuth_deg,  # inline azimuth, 0° = North CW
                }
            }
            
            if not self._load_and_save_traces():
                    return

        super().accept()



class ManualSurveyDialog(QtWidgets.QDialog):
    """Dialog that lets the user input survey bounds and preview the footprint."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Manual Survey Definition")
        self.setMinimumSize(900, 600)
        self._build_ui()
        self._connect_signals()
        self.update_plot()

    def _build_ui(self):
        self.x_min = self._spinbox(-1e6, 1e6, 0.0, "X Min")
        self.x_max = self._spinbox(-1e6, 1e6, 1000.0, "X Max")
        self.y_min = self._spinbox(-1e6, 1e6, 0.0, "Y Min")
        self.y_max = self._spinbox(-1e6, 1e6, 1000.0, "Y Max")
        self.inline_start = self._spinbox(-1e4, 1e4, 1000.0, "Inline Start", decimals=0)
        self.xline_start = self._spinbox(-1e4, 1e4, 2000.0, "Xline Start", decimals=0)
        self.inline_increment = self._spinbox(0.01, 1e4, 25.0, "Inline Increment (m)")
        self.xline_increment = self._spinbox(0.01, 1e4, 25.0, "Xline Increment (m)")
        self.azimuth = self._spinbox(-180.0, 180.0, 0.0, "Azimuth (deg)")

        form_layout = QtWidgets.QFormLayout()
        for label, widget in [
            ("X Min (UTM)", self.x_min),
            ("X Max (UTM)", self.x_max),
            ("Y Min (UTM)", self.y_min),
            ("Y Max (UTM)", self.y_max),
            ("Inline Start", self.inline_start),
            ("Xline Start", self.xline_start),
            ("Inline Increment (m)", self.inline_increment),
            ("Xline Increment (m)", self.xline_increment),
            ("Azimuth (deg)", self.azimuth),
        ]:
            form_layout.addRow(label, widget)

        form_widget = QtWidgets.QWidget()
        form_widget.setLayout(form_layout)
        form_widget.setMaximumWidth(300)

        self.figure = Figure(figsize=(5, 5))
        self.canvas = FigureCanvas(self.figure)

        button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)

        content_layout = QtWidgets.QHBoxLayout()
        content_layout.addWidget(form_widget)
        content_layout.addWidget(self.canvas, stretch=1)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(content_layout)
        layout.addWidget(button_box)

    def _spinbox(self, min_val, max_val, default, label, decimals=2):
        box = QtWidgets.QDoubleSpinBox()
        box.setRange(min_val, max_val)
        box.setDecimals(decimals)
        box.setButtonSymbols(QtWidgets.QAbstractSpinBox.ButtonSymbols.NoButtons)
        box.setMaximumWidth(180)
        if default is not None:
            box.setValue(default)
        else:
            box.setValue(min_val)
            box.lineEdit().clear()
            box.lineEdit().setPlaceholderText("Enter value")
        return box

    def _connect_signals(self):
        for widget in (
            self.x_min,
            self.x_max,
            self.y_min,
            self.y_max,
            self.inline_start,
            self.xline_start,
            self.inline_increment,
            self.xline_increment,
            self.azimuth,
        ):
            widget.valueChanged.connect(self.update_plot)

    def update_plot(self):
        x_min, x_max = self.x_min.value(), self.x_max.value()
        y_min, y_max = self.y_min.value(), self.y_max.value()
        az = np.deg2rad(self.azimuth.value())

        width = max(0.0, x_max - x_min)
        height = max(0.0, y_max - y_min)
        center_x = (x_min + x_max) / 2.0
        center_y = (y_min + y_max) / 2.0

        corners = [
            (-width / 2, -height / 2),
            (width / 2, -height / 2),
            (width / 2, height / 2),
            (-width / 2, height / 2),
            (-width / 2, -height / 2),
        ]

        cos_t, sin_t = np.cos(az), np.sin(az)
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
        ax.set_title("Survey Footprint")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, linestyle="--", alpha=0.5)

        corners_xy = rotated[:-1]
        for idx, (cx, cy) in enumerate(corners_xy, start=1):
            inline_num, xline_num = self.compute_inline_xline(cx, cy)
            ax.scatter(cx, cy, color="tab:blue")
            ax.annotate(
                f"Corner {idx}\nInline={inline_num}\nXline={xline_num}",
                xy=(cx, cy),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=9,
                color="black",
            )
        # ax.legend(loc="upper right")
        self.canvas.draw_idle()

    def compute_inline_xline(self, x: float, y: float) -> tuple[int, int]:
        x_min = self.x_min.value()
        x_max = self.x_max.value()
        y_min = self.y_min.value()
        y_max = self.y_max.value()
        width = x_max - x_min
        height = y_max - y_min
        center_x = (x_min + x_max) / 2.0
        center_y = (y_min + y_max) / 2.0

        theta = np.deg2rad(self.azimuth.value())
        cos_t, sin_t = np.cos(theta), np.sin(theta)

        # Global coordinates of the origin corner (local -width/2, -height/2)
        origin_dx = -width / 2
        origin_dy = -height / 2
        origin_x = center_x + cos_t * origin_dx - sin_t * origin_dy
        origin_y = center_y + sin_t * origin_dx + cos_t * origin_dy

        # Basis vectors (unit) along crossline (local +x) and inline (local +y)
        vec_x = (cos_t * width, sin_t * width)
        vec_y = (-sin_t * height, cos_t * height)
        len_x = np.hypot(*vec_x)
        len_y = np.hypot(*vec_y)
        basis_x = (vec_x[0] / len_x, vec_x[1] / len_x) if len_x else (1.0, 0.0)
        basis_y = (vec_y[0] / len_y, vec_y[1] / len_y) if len_y else (0.0, 1.0)

        vx = x - origin_x
        vy = y - origin_y
        local_x = vx * basis_x[0] + vy * basis_x[1]
        local_y = vx * basis_y[0] + vy * basis_y[1]

        inline_start = self.inline_start.value()
        xline_start = self.xline_start.value()
        inline_inc = self.inline_increment.value()
        xline_inc = self.xline_increment.value()

        inline_num = inline_start + (local_y / inline_inc) if inline_inc else inline_start
        xline_num = xline_start + (local_x / xline_inc) if xline_inc else xline_start
        return round(inline_num), round(xline_num)

    def get_values(self):
        return {
            "x_min": self.x_min.value(),
            "x_max": self.x_max.value(),
            "y_min": self.y_min.value(),
            "y_max": self.y_max.value(),
            "inline_start": self.inline_start.value(),
            "xline_start": self.xline_start.value(),
            "inline_increment": self.inline_increment.value(),
            "xline_increment": self.xline_increment.value(),
            "azimuth": self.azimuth.value(),
        }


class NewSurveyDialog(QtWidgets.QDialog):
    """Custom QDialog class for creating a new survey."""
    def __init__(self):
        super().__init__()
        uic.loadUi("newSurveyDialog.ui", self)  # Load the UI
        
        options=['Scan X/Y Ranges','Import from SEG-Y','Enter by hand']

        # Find the QPlainTextEdit and QDialogButtonBox
        self.plainTextEdit = self.findChild(QtWidgets.QPlainTextEdit, "plainTextEdit")
        self.buttonBox = self.findChild(QtWidgets.QDialogButtonBox, "newSurveyDialogButtons")
        self.listView = self.findChild(QtWidgets.QListView, "listViewOptions")
        
        # Populate the QListWidget with options
        self.model = QStringListModel()
        self.model.setStringList(options)  # Populate model with options
        self.listView.setModel(self.model)  # Attach model to QListView
        
        
        # ✅ Get the OK button from QDialogButtonBox
        self.ok_button = self.buttonBox.button(QtWidgets.QDialogButtonBox.StandardButton.Ok)
        if self.ok_button:
            self.ok_button.setEnabled(False)  # Disable OK initially

        self.selected_import = None

        # Connect QDialogButtonBox signals
        if self.buttonBox:
            self.buttonBox.accepted.connect(self.process_selection)  # OK Button
            self.buttonBox.rejected.connect(self.reject)  # Cancel Button
            
        # ✅ Detect clicks on list items and enable OK instantly
        self.listView.clicked.connect(self.enable_ok_button)  # Works instantly on click
            
    def enable_ok_button(self):
        """Enables OK button when an item is selected."""
        selected_indexes = self.listView.selectedIndexes()
        if self.ok_button:
            self.ok_button.setEnabled(bool(selected_indexes))  # Enable only if something is selected

            
    def process_selection(self):
        """Called when OK is clicked. Stores selected values and closes dialog."""
    
        selected_indexes = self.listView.selectedIndexes()
        self.selected_import = selected_indexes[0].data() if selected_indexes else None
        
        print(f"Import Option: {self.selected_import}")  # Debugging
        self.accept()  # ✅ Close the dialog after processing

    def get_survey_name(self):
        """Returns the text entered in QPlainTextEdit when OK is clicked."""
        if self.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            return self.plainTextEdit.toPlainText().strip()  # Get the text input
        return None
    
    def GetSelectedImportOption(self):
        """Returns the selected survey from the QListView when OK is clicked."""
        selected_indexes = self.listView.selectedIndexes()
        if selected_indexes:  # Ensure something is selected
            selected_import = selected_indexes[0].data()  # Get the text of the selected item
            return selected_import
        return None
