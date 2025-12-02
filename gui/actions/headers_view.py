from pathlib import Path

import pandas as pd
from PyQt6 import QtWidgets

from openseismicprocessing.constants import TRACE_HEADER_REV1


def _label_with_bytes(name: str) -> str:
    spec = TRACE_HEADER_REV1
    if name in spec:
        start, length = spec[name]
        end = start + length - 1
        return f"{name} (Bytes {start}-{end})"
    return name


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


def _build_headers_table(view_df: pd.DataFrame, title: str) -> QtWidgets.QWidget:
    table = QtWidgets.QTableWidget()
    table.setRowCount(len(view_df))
    table.setColumnCount(len(view_df.columns))
    table.setHorizontalHeaderLabels([_label_with_bytes(str(c)) for c in view_df.columns])
    table.setWordWrap(False)
    # Align header font with cell font (non-bold)
    header_font = table.font()
    header_font.setBold(False)
    table.horizontalHeader().setFont(header_font)
    table.verticalHeader().setFont(header_font)
    for i in range(len(view_df)):
        for j, col in enumerate(view_df.columns):
            table.setItem(i, j, QtWidgets.QTableWidgetItem(str(view_df.iat[i, j])))
    header = table.horizontalHeader()
    header.setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
    header.setStretchLastSection(True)
    table.verticalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
    table.resizeColumnsToContents()
    table.resizeRowsToContents()

    widget = QtWidgets.QWidget()
    v_layout = QtWidgets.QVBoxLayout(widget)
    v_layout.addWidget(table)
    widget.resize(900, 700)
    return widget


def show_headers(window):
    """Open the headers viewer tab for the selected survey geometry."""
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

    dlg = HeadersDialog(window, geometry_files)
    geom_path, n_rows = dlg.get_selection()
    if not geom_path:
        return
    try:
        df = pd.read_csv(geom_path) if geom_path.suffix.lower() == ".csv" else pd.read_parquet(geom_path)
    except Exception as exc:
        QtWidgets.QMessageBox.warning(window, "Headers Error", f"Failed to read geometry {geom_path.name}:\n{exc}")
        return

    view_df = df.head(n_rows)
    base_name = geom_path.name.replace(".geometry.parquet", "").replace(".geometry.csv", "")
    title = f"Headers: {base_name}"
    widget = _build_headers_table(view_df, title)
    window.tabWidget.addTab(widget, title)
    window.tabWidget.setCurrentWidget(widget)
