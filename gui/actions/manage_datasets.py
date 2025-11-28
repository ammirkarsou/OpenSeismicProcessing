from pathlib import Path
import json
import shutil

from PyQt6 import QtWidgets, QtCore


class ManageDatasetsDialog(QtWidgets.QDialog):
    def __init__(self, parent, manifests):
        super().__init__(parent)
        self.manifests = manifests
        self.setWindowTitle("Manage Datasets")
        self.resize(500, 300)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(QtWidgets.QLabel("Datasets in survey:"))

        self.list = QtWidgets.QListWidget()
        self._populate_list()
        layout.addWidget(self.list)

        btn_delete = QtWidgets.QPushButton("Delete Selected")
        btn_close = QtWidgets.QPushButton("Close")
        btn_delete.clicked.connect(self.delete_selected)
        btn_close.clicked.connect(self.close)
        btn_row = QtWidgets.QHBoxLayout()
        btn_row.addWidget(btn_delete)
        btn_row.addStretch()
        btn_row.addWidget(btn_close)
        layout.addLayout(btn_row)

    def _populate_list(self):
        self.list.clear()
        pre_label = QtWidgets.QListWidgetItem("Pre-stack")
        pre_label.setFlags(QtCore.Qt.ItemFlag.ItemIsEnabled)
        pre_label.setData(QtCore.Qt.ItemDataRole.UserRole, None)
        post_label = QtWidgets.QListWidgetItem("Post-stack")
        post_label.setFlags(QtCore.Qt.ItemFlag.ItemIsEnabled)
        post_label.setData(QtCore.Qt.ItemDataRole.UserRole, None)
        self.list.addItem(pre_label)
        for m in self.manifests:
            dtype = _dataset_type(m)
            if "post" in dtype:
                continue
            name = m.stem.replace(".zarr", "").replace(".manifest", "")
            item = QtWidgets.QListWidgetItem(f"  {name}")
            item.setData(QtCore.Qt.ItemDataRole.UserRole, m)
            item.setFlags(QtCore.Qt.ItemFlag.ItemIsEnabled | QtCore.Qt.ItemFlag.ItemIsSelectable)
            self.list.addItem(item)
        self.list.addItem(post_label)
        for m in self.manifests:
            dtype = _dataset_type(m)
            if "post" not in dtype:
                continue
            name = m.stem.replace(".zarr", "").replace(".manifest", "")
            item = QtWidgets.QListWidgetItem(f"  {name}")
            item.setData(QtCore.Qt.ItemDataRole.UserRole, m)
            item.setFlags(QtCore.Qt.ItemFlag.ItemIsEnabled | QtCore.Qt.ItemFlag.ItemIsSelectable)
            self.list.addItem(item)

    def delete_selected(self):
        item = self.list.currentItem()
        if item is None:
            return
        manifest = item.data(QtCore.Qt.ItemDataRole.UserRole)
        if manifest is None or not isinstance(manifest, Path):
            return
        reply = QtWidgets.QMessageBox.question(
            self,
            "Confirm Deletion",
            f"Delete dataset '{manifest.stem}'?\nZarr, geometry, and manifest will be removed.",
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.No,
        )
        if reply != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        try:
            data = json.loads(manifest.read_text())
            zarr_path = data.get("zarr_store")
            geom_path = data.get("geometry_parquet")
        except Exception:
            zarr_path = None
            geom_path = None
        # Only delete assets that are not referenced by other manifests
        remaining_manifests = [m for m in self.manifests if m != manifest]
        referenced = set()
        for m in remaining_manifests:
            try:
                md = json.loads(m.read_text())
                if "zarr_store" in md:
                    referenced.add(Path(md["zarr_store"]).resolve())
                if "geometry_parquet" in md:
                    referenced.add(Path(md["geometry_parquet"]).resolve())
            except Exception:
                continue

        for p in [zarr_path, geom_path]:
            if not p:
                continue
            target = Path(p)
            if not target.exists():
                continue
            if target.resolve() in referenced:
                # skip deletion if still referenced
                continue
            try:
                if target.is_dir():
                    shutil.rmtree(target, ignore_errors=True)
                else:
                    target.unlink(missing_ok=True)
            except Exception:
                pass
        try:
            manifest.unlink(missing_ok=True)
        except Exception:
            pass
        row = self.list.row(item)
        self.list.takeItem(row)
        try:
            self.manifests.remove(manifest)
        except ValueError:
            pass
        QtWidgets.QMessageBox.information(self, "Dataset Deleted", "Dataset removed successfully.")


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


def manage_datasets(window):
    """Show the manage datasets dialog for the current survey."""
    if not window.currentSurveyPath:
        QtWidgets.QMessageBox.warning(window, "Warning", "Please select a survey first.")
        return
    manifests = _list_manifests(window.currentSurveyPath)
    if not manifests:
        QtWidgets.QMessageBox.information(window, "Manage Datasets", "No datasets found for this survey.")
        return
    dlg = ManageDatasetsDialog(window, manifests)
    dlg.exec()
