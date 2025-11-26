#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 18:34:03 2025

@author: ammir
"""
from PyQt6 import QtWidgets, uic
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
from openseismicprocessing.catalog import ensure_project, delete_project, list_projects
from openseismicprocessing.constants import TRACE_HEADER_REV0, TRACE_HEADER_REV1
from openseismicprocessing.segy_inspector import SegyInspector, print_revision_summary
from openseismicprocessing.io import read_trace_headers_until, get_text_header
from openseismicprocessing._io import open_segy_data
from openseismicprocessing.zarr_utils import segy_directory_to_zarr
import segyio
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

class TwoDimensionPostStackSEGYDialog(QtWidgets.QDialog):
    """Custom QDialog class for creating a new survey."""
    def __init__(self, parentClass):
        super().__init__()
        uic.loadUi("2DPostStackDialog.ui", self)  # Load the UI

        self.dsbNx = self.findChild(QtWidgets.QDoubleSpinBox, "doubleSpinBox")
        self.dsbNz = self.findChild(QtWidgets.QDoubleSpinBox, "doubleSpinBox_2")
        self.dsbDx = self.findChild(QtWidgets.QDoubleSpinBox, "doubleSpinBox_3")
        self.dsbDz = self.findChild(QtWidgets.QDoubleSpinBox, "doubleSpinBox_4")


        self.dsbNx.setButtonSymbols(QtWidgets.QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.dsbNz.setButtonSymbols(QtWidgets.QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.dsbDx.setButtonSymbols(QtWidgets.QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.dsbDz.setButtonSymbols(QtWidgets.QAbstractSpinBox.ButtonSymbols.NoButtons)

        self.n_traces = parentClass.n_traces
        self.n_samples = parentClass.n_samples

        x=parentClass.SEGY_Dataframe[parentClass.selected_xrange_column].to_numpy()/parentClass.xScale - parentClass.xTranslation
        y=parentClass.SEGY_Dataframe[parentClass.selected_yrange_column].to_numpy()/parentClass.yScale - parentClass.yTranslation

        self.Nx = parentClass.n_traces
        self.Nz = parentClass.n_samples
        self.dx = np.sqrt((x[1]-x[0])**2 + (y[1]-y[1])**2)
        self.dz = parentClass.dt

        
        self.dsbNx.setValue(parentClass.n_traces)
        self.dsbNz.setValue(parentClass.n_samples)
        self.dsbDx.setValue(self.dx)
        self.dsbDz.setValue(parentClass.dt)

        self.dsbNx.valueChanged.connect(self.UpdateNx)
        self.dsbNz.valueChanged.connect(self.UpdateNz)
        self.dsbDx.valueChanged.connect(self.UpdateDx)
        self.dsbDz.valueChanged.connect(self.UpdateDz)

        

    def UpdateDx(self, value):
        self.dx = value

    def UpdateDz(self, value):
        self.dz = value

    def UpdateNx(self, value):
        self.Nx = value

    def UpdateNz(self, value):
        self.Nz = value

    # ✅ Getter method to retrieve dx and dz after the dialog closes
    def PassChildClassValues(self, parentClass):
        parentClass.dx = self.dx
        parentClass.dz = self.dz

    # ✅ Override accept() to validate input before closing
    def accept(self):
        # Calculate expected traces
        expected_traces = self.Nz * self.Nx

        # Validate against the parent's n_traces
        if expected_traces != self.n_traces * self.n_samples:
            # Show warning and prevent dialog closure
            QtWidgets.QMessageBox.warning(
                self,
                "Validation Error",
                f"Invalid dimensions: Nz * Nx = {expected_traces}, expected {self.n_traces * self.n_samples}.\nPlease correct the values."
            )
            return  # Don't call super().accept() to keep the dialog open

        # If validation passes, pass the values and accept the dialog
        super().accept()

class ImportSEGYDialog(QtWidgets.QDialog):
    """Custom QDialog class for creating a new survey."""
    def __init__(self, boundary=None, survey_root: str | None = None):
        super().__init__()
        self.setWindowTitle("Load SEG-Y")
        self.resize(620, 800)
        self.survey_boundary = boundary or {}
        self.survey_root = Path(survey_root) if survey_root else None

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
        self.comboBoxHeaderUser = QtWidgets.QComboBox()
        form_layout.addWidget(label_z, 0, 0)
        form_layout.addWidget(self.lineEdit2, 0, 1)
        form_layout.addWidget(self.spinbox1, 0, 2)
        form_layout.addWidget(self.spinbox2, 0, 3)
        form_layout.addWidget(self.comboBoxHeaderUser, 0, 4)
        rev_layout = QtWidgets.QHBoxLayout()
        rev_layout.addWidget(QtWidgets.QLabel("SEGY Revision"))
        self.comboBoxHeaderUser_revision = QtWidgets.QComboBox()
        self.comboBoxHeaderUser_revision.addItems(["0", "1", "2.0", "2.1", "User Defined"])
        rev_layout.addWidget(self.comboBoxHeaderUser_revision)
        rev_layout.addWidget(QtWidgets.QLabel("Traces to view"))
        self.numTracesToShow = QtWidgets.QDoubleSpinBox()
        self.numTracesToShow.setDecimals(0)
        self.numTracesToShow.setRange(1, 1e9)
        self.numTracesToShow.setValue(200)  # lower default for faster initial loads
        self.numTracesToShow.setButtonSymbols(QtWidgets.QAbstractSpinBox.ButtonSymbols.NoButtons)
        rev_layout.addWidget(self.numTracesToShow)
        rev_layout.addWidget(QtWidgets.QLabel("Units"))
        self.comboBoxUnits = QtWidgets.QComboBox()
        self.comboBoxUnits.addItems(["meters", "feet"])
        rev_layout.addWidget(self.comboBoxUnits)
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

        (self.x_range_label, self.lineEdit6, self.comboBox3) = add_range_row(4, xrange_label)
        form_layout.addWidget(self.x_range_label, 4, 0)
        form_layout.addWidget(self.lineEdit6, 4, 1)
        form_layout.addWidget(self.comboBox3, 4, 2)

        (self.y_range_label, self.lineEdit7, self.comboBox4) = add_range_row(5, yrange_label)
        form_layout.addWidget(self.y_range_label, 5, 0)
        form_layout.addWidget(self.lineEdit7, 5, 1)
        form_layout.addWidget(self.comboBox4, 5, 2)

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
        (self.xScaleLabel,
         self.xScaleCombo,
         self.xManualCheck,
         self.xManualSpin) = add_scale_row(2, "X Scale")
        (self.yScaleLabel,
         self.yScaleCombo,
         self.yManualCheck,
         self.yManualSpin) = add_scale_row(3, "Y Scale")

        for combo in (
            self.inlineScaleCombo,
            self.xlineScaleCombo,
            self.xScaleCombo,
            self.yScaleCombo,
        ):
            idx = combo.findText("scalco")
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
        try:
            unit_text = self.comboBoxUnits.currentText().lower()
        except Exception:
            unit_text = "meters"
        if "feet" in unit_text or "ft" in unit_text:
            return 0.3048
        return 1.0

    @staticmethod
    def _apply_coordinate_scale(values: pd.Series, scale) -> np.ndarray:
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

    def _coord_scale_map(self, segy_file) -> dict[str, float]:
        """Determine scale per selected coordinate header for the given SEG-Y file."""
        cols_and_scales = [
            (self.comboBox, self.inlineScaleCombo, self.inlineManualCheck, self.inlineManualSpin),
            (self.comboBox2, self.xlineScaleCombo, self.xlineManualCheck, self.xlineManualSpin),
            (self.comboBox3, self.xScaleCombo, self.xManualCheck, self.xManualSpin),
            (self.comboBox4, self.yScaleCombo, self.yManualCheck, self.yManualSpin),
        ]
        out = {}
        def scale_factor(val: float) -> float:
            if val in (None, 0):
                return 1.0
            return val if val > 0 else 1.0 / abs(val)
        for combo, scale_combo, chk, spin in cols_and_scales:
            col = combo.currentText()
            if not col:
                continue
            raw_val = self._resolve_scale_for_file(scale_combo.currentText(), chk, spin, segy_file)
            scale_val = scale_factor(raw_val)
            out[col] = scale_val
        return out

    def _coord_scale_arrays(self, segy_file, length: int) -> dict[str, np.ndarray]:
        """Per-trace scale factors for selected coordinate headers."""
        cols_and_scales = [
            (self.comboBox, self.inlineScaleCombo, self.inlineManualCheck, self.inlineManualSpin),
            (self.comboBox2, self.xlineScaleCombo, self.xlineManualCheck, self.xlineManualSpin),
            (self.comboBox3, self.xScaleCombo, self.xManualCheck, self.xManualSpin),
            (self.comboBox4, self.yScaleCombo, self.yManualCheck, self.yManualSpin),
        ]
        out = {}
        for combo, scale_combo, chk, spin in cols_and_scales:
            col = combo.currentText()
            if not col:
                continue
            scale_arr = self._resolve_scale_array_for_file(scale_combo.currentText(), chk, spin, segy_file, length)
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
                            col = combo.currentText().lower()
                            tf = tf_map.get(col)
                            if tf is None:
                                QtWidgets.QMessageBox.warning(
                                    self, "Header Missing", f"Column '{col}' not supported for pre-stack check."
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
        spec = self._current_header_spec()
        return list(spec.keys())

    def _dataset_type_slug(self) -> str:
        mode = (self.GetAcquisitionType() or "").lower()
        dim = "2d" if "2d" in mode else "3d" if "3d" in mode else "unknown"
        stack = "pre" if "pre" in mode else "post" if "post" in mode else "unknown"
        return f"{dim}-{stack}"

    def dataset_type(self) -> str:
        return self._dataset_type_slug()

    def _ensure_output_paths(self) -> tuple[Path, Path]:
        base_dir = self.survey_root
        if base_dir is None or not base_dir.exists():
            dlg_dir = QtWidgets.QFileDialog.getExistingDirectory(self, "Select output folder for SEG-Y import")
            if not dlg_dir:
                return None, None
            base_dir = Path(dlg_dir)
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
                    if unit_factor != 1.0:
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
        zarr_path, geom_path = self._ensure_output_paths()
        if not zarr_path or not geom_path:
            return False
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
                }
                manifest_path = Path(str(zarr_path) + ".manifest.json")
                manifest_path.write_text(json.dumps(manifest, indent=2))
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Import Error", f"Failed to store SEG-Y into Zarr:\n{exc}")
            return False
        QtWidgets.QMessageBox.information(self, "SEG-Y Imported", f"Saved traces to:\n{zarr_path}\nGeometry to:\n{geom_path}")
        return True

    def GetAcquisitionType(self):
        return self.comboBoxMode.currentText()

    def UpdateInline(self):
        self.selected_inline_column = self.comboBox.currentText()
        self.ilineScale = self._resolve_scale(self.inlineScaleCombo, self.inlineManualCheck, self.inlineManualSpin)

        if self.SEGY_Dataframe is not None and isinstance(self.SEGY_Dataframe, pd.DataFrame):
            scaled = self._apply_coordinate_scale(self.SEGY_Dataframe[self.selected_inline_column], self.ilineScale)
            init = np.min(scaled)
            end = np.max(scaled)
            text = "%i - %i" % (init,end)
            self.lineEdit4.setText(text)

    def UpdateXline(self):
        self.selected_xline_column = self.comboBox2.currentText()
        self.xlineScale = self._resolve_scale(self.xlineScaleCombo, self.xlineManualCheck, self.xlineManualSpin)

        if self.SEGY_Dataframe is not None and isinstance(self.SEGY_Dataframe, pd.DataFrame):
            scaled = self._apply_coordinate_scale(self.SEGY_Dataframe[self.selected_xline_column], self.xlineScale)
            init = np.min(scaled)
            end = np.max(scaled)
            text = "%i - %i" % (init,end)
            self.lineEdit5.setText(text)
    
    def UpdateXRange(self):
        self.selected_xrange_column = self.comboBox3.currentText()
        self.xScale = self._resolve_scale(self.xScaleCombo, self.xManualCheck, self.xManualSpin)

        if self.SEGY_Dataframe is not None and isinstance(self.SEGY_Dataframe, pd.DataFrame):
            scaled = self._apply_coordinate_scale(self.SEGY_Dataframe[self.selected_xrange_column], self.xScale)
            init = np.min(scaled)
            end = np.max(scaled)
            text = "%.2f - %.2f" % (init,end)
            self.lineEdit6.setText(text)

    def UpdateYRange(self):
        self.selected_yrange_column = self.comboBox4.currentText()
        self.yScale = self._resolve_scale(self.yScaleCombo, self.yManualCheck, self.yManualSpin)

        if self.SEGY_Dataframe is not None and isinstance(self.SEGY_Dataframe, pd.DataFrame):
            scaled = self._apply_coordinate_scale(self.SEGY_Dataframe[self.selected_yrange_column], self.yScale)
            init = np.min(scaled)
            end = np.max(scaled)
            text = "%.2f - %.2f" % (init,end)
            self.lineEdit7.setText(text)

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

    def load_column_names(self, preserve_selection: bool = False):
        """Populate combo boxes with sensible defaults based on acquisition type."""
        if self.comboBoxMode is None:
            return
        mode = self.GetAcquisitionType() or ""
        spec = self._current_header_spec()

        columns = []
        prev_selection = (
            self.comboBox.currentText(),
            self.comboBox2.currentText(),
            self.comboBox3.currentText(),
            self.comboBox4.currentText(),
        ) if preserve_selection else (None, None, None, None)

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

        self.comboBox.blockSignals(True)
        self.comboBox2.blockSignals(True)
        self.comboBox3.blockSignals(True)
        self.comboBox4.blockSignals(True)
        self.comboBox.clear()
        self.comboBox2.clear()
        self.comboBox3.clear()
        self.comboBox4.clear()
        self.comboBox.addItems(columns)
        self.comboBox2.addItems(columns)
        self.comboBox3.addItems(columns)
        self.comboBox4.addItems(columns)
        defaults = [0, 1, 2, 3]
        combos = [self.comboBox, self.comboBox2, self.comboBox3, self.comboBox4]
        pre_defaults = ["sx", "sy", "gx", "gy"]
        for idx, combo in enumerate(combos):
            if preserve_selection and prev_selection[idx] in columns:
                combo.setCurrentText(prev_selection[idx])
                continue
            if "3D Pre-stack" in mode or "2D Pre-stack" in mode:
                target = pre_defaults[idx]
                if target in columns:
                    combo.setCurrentText(target)
                    continue
            if combo.count() > defaults[idx]:
                combo.setCurrentIndex(defaults[idx])
            elif combo.count() > 0:
                combo.setCurrentIndex(0)

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

        for label_widget, text in zip(
            (self.inlineScaleLabel, self.xlineScaleLabel, self.xScaleLabel, self.yScaleLabel),
            scale_labels,
        ):
            label_widget.setText(text)

        self.comboBox.blockSignals(False)
        self.comboBox2.blockSignals(False)
        self.comboBox3.blockSignals(False)
        self.comboBox4.blockSignals(False)
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
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self,
            "Select SEG-Y Files",
            "",
            "SEG-Y Files (*.sgy *.segy)"
        )

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
        # For pre-stack data, perform trace-outside check when pressing OK
        if "Pre-stack" in (self.GetAcquisitionType() or ""):
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

        super().accept()
        
    # def PassChildClassValues(self, parentClass):
    #     parentClass.


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
        

class DialogBox(QtWidgets.QDialog):
    """Custom QDialog class for the secondary dialog."""
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
        button_row = QtWidgets.QHBoxLayout()
        button_row.addWidget(self.newSurveyButton)
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
        self.surveyListView.selectionModel().currentChanged.connect(self._on_survey_selected)
        self.UpdateSurveyList(select_name=self.initial_selection)

    def DeleteSurvey(self):
        selected_survey = self.GetSelectedSurvey()
        
        if selected_survey is not None:
            folder_path = os.path.join(self.main.rootFolderPath, selected_survey)
            # ✅ Show confirmation dialog
            reply = QtWidgets.QMessageBox.question(
                self,
                "Confirm Deletion",
                f"Are you sure you want to delete the survey\n{selected_survey}?",
                QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
                QtWidgets.QMessageBox.StandardButton.No  # Default to "No"
            )

            if reply == QtWidgets.QMessageBox.StandardButton.Yes:
            
                if os.path.exists(folder_path) and os.path.isdir(folder_path):
                    shutil.rmtree(folder_path)
                    print(f"Folder '{folder_path}' deleted successfully!")
                else:
                    print(f"Error: Folder '{folder_path}' does not exist.")

                if selected_survey in self.main.folder_list:
                    self.main.folder_list.remove(selected_survey)

                delete_project(name=selected_survey)

                self.UpdateSurveyList()
    
    def CreateNewFolders(self):
        #create main folder
        os.mkdir(self.surveyPath)

        #create binaries folder
        os.mkdir(self.surveyPath + "/Binaries/")

        #create geometry folder
        os.mkdir(self.surveyPath + "/Geometry/")

        #create JSON folder
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

                # ✅ Refresh the list in surveyListView
                self.UpdateSurveyList(select_name=survey_name)

                # shows that it was sucessfull
                QtWidgets.QMessageBox.information(self, "Success", f"Survey '{survey_name}' created successfully!")
            else:
                QtWidgets.QMessageBox.warning(self, "Warning", "Survey with this name already exists!")

    
    def CreateNewSurvey(self):
        """Opens the NewSurveyDialog, gets user input, and creates a new folder."""
        if not self.main.rootFolderPath:
            QtWidgets.QMessageBox.warning(self, "Warning", "Please select a root folder first!")
            return
    
        # Open the new survey dialog
        dialog = NewSurveyDialog()
        survey_name = dialog.get_survey_name()
        selected_import = dialog.selected_import
        if not survey_name:
            return

        # 'Scan X/Y Ranges','Import from SEG-Y','Enter by hand'
        if selected_import == "Import from SEG-Y":
            
            # ✅ Show the SEG-Y import dialog
            SEGYImport = ImportSEGYDialog()

            while True:
                
                if SEGYImport.exec() == QtWidgets.QDialog.DialogCode.Accepted:  # User pressed OK
                    
                    child_dialog = None

                    mode = SEGYImport.GetAcquisitionType()
                    
                    # ✅ Open the child dialog and pass the SEGYImport dialog
                    if mode == "2D Post-stack":
                        child_dialog = TwoDimensionPostStackSEGYDialog(SEGYImport)

                    # Open the dialog and wait for the user's response
                    if child_dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:  # If user clicks OK in the child dialog
                        
                        # ✅ Create the survey folder
                        self.CreateSurveyFolder(survey_name)

                        child_dialog.PassChildClassValues(SEGYImport)

                        # ✅ Confirm binary import
                        reply = QtWidgets.QMessageBox.question(
                            self,
                            "Confirm SEG-Y Import",
                            "Do you wish to import SEG-Y binary?",
                            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
                            QtWidgets.QMessageBox.StandardButton.No
                        )

                        # ✅ Set the binary loading flag
                        loadBinary = reply == QtWidgets.QMessageBox.StandardButton.Yes
                        break  # ✅ Exit the loop since the process was completed successfully

                    else:
                        # User clicked Cancel in TwoDimensionPostStackSEGYDialog
                        # ✅ Reopen the SEGYImport dialog
                        continue  

                else:
                    # User clicked Cancel in SEGYImport dialog
                    break  # Exit the loop entirely and go back to NewSurveyDialog

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

    def UpdateSurveyList(self, select_name: str | None = None):
        """Updates the QListView with the latest survey names and selects the requested one."""

        self.model.setStringList(self.main.folder_list)  # Update model with new folder list
        self.surveyListView.setModel(self.model)  # Refresh view
        target_index = None
        if select_name and select_name in self.main.folder_list:
            row = self.main.folder_list.index(select_name)
            target_index = self.model.index(row, 0)
        elif self.main.folder_list:
            target_index = self.model.index(0, 0)

        if target_index is not None:
            self.surveyListView.setCurrentIndex(target_index)
            self._on_survey_selected(target_index)
        else:
            self._on_survey_selected()
        
    def GetSelectedSurvey(self):
        """Returns the selected survey from the QListView when OK is clicked."""
        selected_indexes = self.surveyListView.selectedIndexes()
        if selected_indexes:  # Ensure something is selected
            selected_survey = selected_indexes[0].data()  # Get the text of the selected item
            return selected_survey
        return None
