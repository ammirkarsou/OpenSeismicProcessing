import json
from pathlib import Path

import inspect

from PyQt6 import QtWidgets, QtGui, QtCore
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from openseismicprocessing import SignalProcessing


class ProcessingDiagram(QtWidgets.QGraphicsView):
    def __init__(self, parent=None, on_selection_change=None, on_label_change=None):
        super().__init__(parent)
        self.scene = QtWidgets.QGraphicsScene(self)
        self.setScene(self.scene)
        self.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        self._boxes: list[tuple[QtWidgets.QGraphicsRectItem, QtWidgets.QGraphicsTextItem]] = []
        self._arrows: list[QtWidgets.QGraphicsItem] = []
        self._function_names = self._collect_function_names()
        self._selected_box: QtWidgets.QGraphicsRectItem | None = None
        self._box_size = QtCore.QSizeF(200, 60)
        self._vertical_spacing = 120
        self._arrow_gap = 12
        self._on_selection_change = on_selection_change
        self._on_label_change = on_label_change

        # Start with a single empty box
        self._add_box("")
        self._reflow()

    def mouseDoubleClickEvent(self, event: QtGui.QMouseEvent) -> None:
        if not self._boxes:
            return super().mouseDoubleClickEvent(event)
        scene_pos = self.mapToScene(event.pos())
        for box, text in self._boxes:
            if box.rect().translated(box.scenePos()).contains(scene_pos):
                self._set_selected_box(box)
                dlg = self._make_line_edit_dialog(text.toPlainText())
                if dlg.exec() == QtWidgets.QDialog.DialogCode.Accepted:
                    new_label = dlg.line_edit.text().strip()
                    if new_label:
                        text.setPlainText(new_label)
                        rect = box.rect()
                        text_rect = text.boundingRect()
                        text.setPos(
                            rect.center().x() - text_rect.width() / 2,
                            rect.center().y() - text_rect.height() / 2,
                        )
                        if callable(self._on_label_change):
                            self._on_label_change(box, new_label)
                return
        super().mouseDoubleClickEvent(event)

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        scene_pos = self.mapToScene(event.pos())
        hit = None
        for box, _ in self._boxes:
            if box.rect().translated(box.scenePos()).contains(scene_pos):
                hit = box
                break
        self._set_selected_box(hit)
        super().mousePressEvent(event)

    def _set_selected_box(self, box: QtWidgets.QGraphicsRectItem | None):
        self._selected_box = box
        for b, _ in self._boxes:
            pen = QtGui.QPen(QtCore.Qt.GlobalColor.blue if b is box else QtCore.Qt.GlobalColor.black, 2)
            b.setPen(pen)
        if callable(self._on_selection_change):
            self._on_selection_change(box)

    def _collect_function_names(self) -> list[str]:
        try:
            names = list(getattr(SignalProcessing, "__all__", []))
        except Exception:
            names = []
        return sorted({n for n in names if isinstance(n, str)}, key=str.lower)

    def _make_line_edit_dialog(self, initial: str) -> QtWidgets.QDialog:
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Edit Step")
        layout = QtWidgets.QVBoxLayout(dlg)
        layout.addWidget(QtWidgets.QLabel("Step name (processing/io function):"))
        line_edit = QtWidgets.QLineEdit()
        line_edit.setText(initial)
        completer = QtWidgets.QCompleter(self._function_names, self)
        completer.setCaseSensitivity(QtCore.Qt.CaseSensitivity.CaseInsensitive)
        completer.setFilterMode(QtCore.Qt.MatchFlag.MatchContains)
        line_edit.setCompleter(completer)
        layout.addWidget(line_edit)
        button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        layout.addWidget(button_box)
        button_box.accepted.connect(dlg.accept)
        button_box.rejected.connect(dlg.reject)
        dlg.line_edit = line_edit
        return dlg

    def _add_box(self, label: str):
        rect = QtCore.QRectF(QtCore.QPointF(0, 0), self._box_size)
        box = self.scene.addRect(rect, pen=QtGui.QPen(QtCore.Qt.GlobalColor.black, 2), brush=QtGui.QBrush(QtCore.Qt.GlobalColor.white))
        text = self.scene.addText(label)
        text.setDefaultTextColor(QtCore.Qt.GlobalColor.black)
        self._boxes.append((box, text))

    def add_box_below(self):
        self._add_box("")
        self._reflow()

    def delete_selected_box(self):
        if self._selected_box is None or len(self._boxes) <= 1:
            return
        for idx, (b, t) in enumerate(self._boxes):
            if b is self._selected_box:
                self.scene.removeItem(b)
                self.scene.removeItem(t)
                self._boxes.pop(idx)
                break
        self._selected_box = None
        self._reflow()

    def _reflow(self):
        # Position boxes vertically and rebuild arrows
        y = 20.0
        x = 150.0
        for box, text in self._boxes:
            rect = QtCore.QRectF(QtCore.QPointF(x, y), self._box_size)
            box.setRect(rect)
            text_rect = text.boundingRect()
            text.setPos(
                rect.center().x() - text_rect.width() / 2,
                rect.center().y() - text_rect.height() / 2,
            )
            y += self._vertical_spacing
        for arrow in self._arrows:
            self.scene.removeItem(arrow)
        self._arrows.clear()

        def center_bottom(box: QtWidgets.QGraphicsRectItem) -> QtCore.QPointF:
            r = box.rect()
            return QtCore.QPointF(r.center().x(), r.bottom())

        def center_top(box: QtWidgets.QGraphicsRectItem) -> QtCore.QPointF:
            r = box.rect()
            return QtCore.QPointF(r.center().x(), r.top())

        import math

        for i in range(len(self._boxes) - 1):
            top_box = self._boxes[i][0]
            bottom_box = self._boxes[i + 1][0]
            start = center_bottom(top_box) + QtCore.QPointF(0, self._arrow_gap)
            end = center_top(bottom_box) - QtCore.QPointF(0, self._arrow_gap)
            line = self.scene.addLine(QtCore.QLineF(start, end), pen=QtGui.QPen(QtCore.Qt.GlobalColor.black, 2))
            angle = math.atan2(end.y() - start.y(), end.x() - start.x())
            head_size = 10
            dx = math.cos(angle)
            dy = math.sin(angle)
            p1 = end
            p2 = end + QtCore.QPointF(
                -dx * head_size - dy * head_size * 0.5,
                -dy * head_size + dx * head_size * 0.5,
            )
            p3 = end + QtCore.QPointF(
                -dx * head_size + dy * head_size * 0.5,
                -dy * head_size - dx * head_size * 0.5,
            )
            poly = QtGui.QPolygonF([p1, p2, p3])
            head = self.scene.addPolygon(poly, pen=QtGui.QPen(QtCore.Qt.GlobalColor.black, 2), brush=QtGui.QBrush(QtCore.Qt.GlobalColor.black))
            self._arrows.extend([line, head])
        self.scene.setSceneRect(self.scene.itemsBoundingRect().marginsAdded(QtCore.QMarginsF(50, 50, 50, 50)))

    def set_boxes(self, labels: list[str]):
        # Remove existing items
        for b, t in self._boxes:
            self.scene.removeItem(b)
            self.scene.removeItem(t)
        for arrow in self._arrows:
            self.scene.removeItem(arrow)
        self._boxes.clear()
        self._arrows.clear()
        self._selected_box = None
        if not labels:
            labels = [""]
        for lbl in labels:
            self._add_box(lbl)
        self._reflow()
        if self._boxes:
            self._set_selected_box(self._boxes[0][0])


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


class ProcessingPanel(QtWidgets.QWidget):
    def __init__(self, manifests: list[Path]):
        super().__init__()
        layout = QtWidgets.QVBoxLayout(self)
        self._params_by_box: dict[QtWidgets.QGraphicsRectItem, dict[str, str]] = {}
        self._current_dataset: Path | None = None

        job_row = QtWidgets.QHBoxLayout()
        job_row.addWidget(QtWidgets.QLabel("Job Name"))
        self.job_name = QtWidgets.QLineEdit()
        self.job_name.setPlaceholderText("Untitled job")
        job_row.addWidget(self.job_name)
        self.open_btn = QtWidgets.QPushButton("Open")
        self.save_btn = QtWidgets.QPushButton("Save")
        self.save_as_btn = QtWidgets.QPushButton("Save As...")
        job_row.addWidget(self.open_btn)
        job_row.addWidget(self.save_btn)
        job_row.addWidget(self.save_as_btn)
        job_row.addStretch()
        layout.addLayout(job_row)

        top_row = QtWidgets.QHBoxLayout()
        top_row.addWidget(QtWidgets.QLabel("Dataset"))
        self.dataset_combo = QtWidgets.QComboBox()
        self.dataset_combo.setSizeAdjustPolicy(QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.dataset_combo.setMaximumWidth(260)
        model = QtGui.QStandardItemModel(self.dataset_combo)
        def add_header(text: str):
            item = QtGui.QStandardItem(text)
            item.setFlags(QtCore.Qt.ItemFlag.ItemIsEnabled)
            model.appendRow(item)
        def add_entry(label: str, path: Path):
            item = QtGui.QStandardItem(f"  {label}")
            item.setData(path, QtCore.Qt.ItemDataRole.UserRole)
            model.appendRow(item)
        add_header("Pre-stack")
        for m in manifests:
            if "post" in _dataset_type(m):
                continue
            add_entry(m.stem.replace(".zarr", "").replace(".manifest", ""), m)
        add_header("Post-stack")
        for m in manifests:
            if "post" not in _dataset_type(m):
                continue
            add_entry(m.stem.replace(".zarr", "").replace(".manifest", ""), m)
        self.dataset_combo.setModel(model)
        top_row.addWidget(self.dataset_combo)

        self.add_box_btn = QtWidgets.QPushButton("New Box")
        self.del_box_btn = QtWidgets.QPushButton("Delete Box")
        self.run_btn = QtWidgets.QPushButton("Run Job")
        top_row.addWidget(self.add_box_btn)
        top_row.addWidget(self.del_box_btn)
        top_row.addWidget(self.run_btn)
        top_row.addStretch()
        layout.addLayout(top_row)

        layout.addWidget(QtWidgets.QLabel("Processing Flow"))
        flow_row = QtWidgets.QHBoxLayout()
        self.diagram = ProcessingDiagram(
            on_selection_change=self._on_box_selected,
            on_label_change=self._on_box_label_changed,
        )
        flow_row.addWidget(self.diagram, 3)

        self.params_widget = QtWidgets.QWidget()
        self.params_layout = QtWidgets.QFormLayout(self.params_widget)
        self.params_layout.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        self.params_layout.setFieldGrowthPolicy(QtWidgets.QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        # Params with scrolling
        right_col = QtWidgets.QVBoxLayout()
        params_scroll = QtWidgets.QScrollArea()
        params_scroll.setWidgetResizable(True)
        params_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        params_scroll.setMinimumWidth(250)
        params_scroll.setWidget(self.params_widget)
        right_col.addWidget(params_scroll, 1)
        # Output plot now shown in separate window; keep placeholder label for spacing
        right_col.addWidget(QtWidgets.QLabel("Output Plot (opens in new window)"))
        flow_row.addLayout(right_col, 2)

        layout.addLayout(flow_row, 1)
        self.add_box_btn.clicked.connect(self.diagram.add_box_below)
        self.del_box_btn.clicked.connect(self.diagram.delete_selected_box)
        self.run_btn.clicked.connect(self._run_pipeline)
        self._select_first_dataset()
        self._plot_windows: list[QtWidgets.QDialog] = []
        self.save_btn.clicked.connect(self._save_job)
        self.save_as_btn.clicked.connect(lambda: self._save_job(force_path=True))
        self.open_btn.clicked.connect(self._open_job)

    def _on_box_label_changed(self, box, label: str):
        self._params_by_box.setdefault(box, {})
        # If params exist from previous label, keep them; otherwise start empty
        self._render_params(box, label)

    def _on_box_selected(self, box):
        current_label = ""
        for b, t in self.diagram._boxes:
            if b is box:
                current_label = t.toPlainText()
                break
        self._render_params(box, current_label)

    def _clear_params(self):
        while self.params_layout.count():
            item = self.params_layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()

    def _render_params(self, box, label: str):
        self._clear_params()
        if box is None or not label:
            return
        func = getattr(SignalProcessing, label, None)
        if not callable(func):
            return
        try:
            sig = inspect.signature(func)
        except Exception:
            return
        # Skip context arg
        params = [p for p in sig.parameters.values() if p.name != "context"]
        stored = self._params_by_box.setdefault(box, {})
        for p in params:
            le = QtWidgets.QLineEdit()
            if p.name in stored:
                le.setText(stored[p.name])
            elif p.default is not inspect._empty:
                le.setText(str(p.default))
            le.textChanged.connect(lambda text, name=p.name, b=box: self._save_param(b, name, text))
            self.params_layout.addRow(f"{p.name}:", le)

    def _save_param(self, box, name: str, value: str):
        self._params_by_box.setdefault(box, {})[name] = value

    def _select_first_dataset(self):
        model = self.dataset_combo.model()
        for i in range(model.rowCount()):
            item = model.item(i)
            data = item.data(QtCore.Qt.ItemDataRole.UserRole) if item else None
            if isinstance(data, Path):
                self.dataset_combo.setCurrentIndex(i)
                break

    def _selected_manifest(self) -> Path | None:
        idx = self.dataset_combo.currentIndex()
        if idx < 0:
            return None
        model = self.dataset_combo.model()
        item = model.item(idx)
        if not item:
            return None
        data = item.data(QtCore.Qt.ItemDataRole.UserRole)
        return data if isinstance(data, Path) else None

    def _run_pipeline(self):
        manifest = self._selected_manifest()
        if manifest is None or not manifest.exists():
            QtWidgets.QMessageBox.warning(self, "Processing", "Select a dataset first.")
            return
        context = self._load_context(manifest)
        steps = self._gather_steps()
        if not steps:
            QtWidgets.QMessageBox.information(self, "Processing", "No steps defined.")
            return
        # close prior plot windows
        for dlg in self._plot_windows:
            try:
                dlg.close()
            except Exception:
                pass
        self._plot_windows.clear()
        for box, label in steps:
            func = getattr(SignalProcessing, label, None)
            if not callable(func):
                QtWidgets.QMessageBox.warning(self, "Processing", f"Function '{label}' not found.")
                return
            kwargs = {}
            for name, val in self._params_by_box.get(box, {}).items():
                try:
                    import ast

                    kwargs[name] = ast.literal_eval(val)
                except Exception:
                    kwargs[name] = val
            try:
                sig = inspect.signature(func)
                valid_keys = {k for k in sig.parameters.keys() if k != "context"}
                kwargs = {k: v for k, v in kwargs.items() if k in valid_keys}
                if label.startswith("plot_"):
                    if "ax" in sig.parameters:
                        ax, dlg = self._make_plot_window(label)
                        if "show" in sig.parameters:
                            kwargs.setdefault("show", False)
                        kwargs["ax"] = ax
                        func(context, **kwargs)
                        dlg.show()
                        self._plot_windows.append(dlg)
                    else:
                        func(context, **kwargs)
                else:
                    func(context, **kwargs)
            except Exception as exc:
                QtWidgets.QMessageBox.warning(self, "Processing", f"Failed at '{label}':\n{exc}")
                return
        QtWidgets.QMessageBox.information(self, "Processing", "Job finished.")

    def _load_context(self, manifest: Path) -> dict:
        context = {}
        try:
            data = json.loads(manifest.read_text())
            zarr_path = data.get("zarr_store")
            geom_path = data.get("geometry_parquet")
            import pandas as pd, zarr

            if geom_path:
                context["geometry"] = pd.read_parquet(geom_path)
                context["_geometry_path"] = geom_path
            if zarr_path:
                context["data"] = zarr.open(zarr_path, mode="r")["amplitude"][:]
        except Exception:
            pass
        return context

    def _make_plot_window(self, title: str) -> tuple:
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle(f"{title} Output")
        dlg.resize(1080, 900)
        dlg.setSizeGripEnabled(True)
        dlg.setWindowFlags(dlg.windowFlags() | QtCore.Qt.WindowType.WindowMinMaxButtonsHint)
        layout = QtWidgets.QVBoxLayout(dlg)
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

        fig = Figure(figsize=(6, 4))
        canvas = FigureCanvas(fig)
        toolbar = NavigationToolbar(canvas, dlg)
        layout.addWidget(toolbar)
        layout.addWidget(canvas, 1)
        ax = fig.add_subplot(111)
        return ax, dlg

    def _gather_steps(self):
        steps = []
        for box, text in self.diagram._boxes:
            label = text.toPlainText().strip()
            if not label:
                continue
            steps.append((box, label))
        return steps

    def _save_job(self, force_path: bool = False):
        steps = self._gather_steps()
        if not steps:
            QtWidgets.QMessageBox.information(self, "Save Job", "No steps to save.")
            return
        job_data = []
        for box, label in steps:
            job_data.append(
                {
                    "func": label,
                    "params": self._params_by_box.get(box, {}),
                }
            )
        default_name = self.job_name.text().strip() or "job"
        if not force_path and hasattr(self, "_current_job_path") and self._current_job_path:
            path = self._current_job_path
        else:
            path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self,
                "Save Job",
                f"{default_name}.json",
                "JSON Files (*.json);;All Files (*)",
            )
            if not path:
                return
            self._current_job_path = path
        try:
            with open(path, "w") as f:
                json.dump(job_data, f, indent=2)
            QtWidgets.QMessageBox.information(self, "Save Job", f"Job saved to:\n{path}")
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Save Job", f"Failed to save job:\n{exc}")

    def _open_job(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open Job",
            "",
            "JSON Files (*.json);;All Files (*)",
        )
        if not path:
            return
        try:
            with open(path, "r") as f:
                job_data = json.load(f)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Open Job", f"Failed to load job:\n{exc}")
            return
        if not isinstance(job_data, list):
            QtWidgets.QMessageBox.warning(self, "Open Job", "Invalid job format.")
            return
        labels = [entry.get("func", "") for entry in job_data]
        self.diagram.set_boxes(labels)
        self._params_by_box.clear()
        for (box, _), entry in zip(self.diagram._boxes, job_data):
            params = entry.get("params", {})
            if isinstance(params, dict):
                self._params_by_box[box] = {str(k): str(v) for k, v in params.items()}
        if job_data and path:
            self.job_name.setText(Path(path).stem)
            self._current_job_path = path
        # refresh params display
        if self.diagram._boxes:
            self.diagram._set_selected_box(self.diagram._boxes[0][0])


def show_processing(window):
    if not window.currentSurveyPath:
        QtWidgets.QMessageBox.warning(window, "Processing", "Please select a survey first.")
        return
    manifests = [p for p in _list_manifests(window.currentSurveyPath) if p.exists()]
    widget = ProcessingPanel(manifests)
    window.tabWidget.addTab(widget, "Processing")
    window.tabWidget.setCurrentWidget(widget)
