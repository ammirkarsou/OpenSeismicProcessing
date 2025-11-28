from pathlib import Path
import json

import numpy as np
import pandas as pd
import zarr
from PyQt6 import QtCore, QtWidgets
from PyQt6.QtGui import QPixmap, QImage, QIcon

try:
    import vtk
    from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
    from vtk.util import numpy_support
except Exception:  # noqa: BLE001
    vtk = None
    QVTKRenderWindowInteractor = None
    numpy_support = None


def _dataset_type(manifest: Path) -> str:
    try:
        data = json.loads(manifest.read_text())
        return str(data.get("dataset_type", "")).lower()
    except Exception:
        return ""


def _list_poststack_manifests(survey_path: str | Path) -> list[Path]:
    bin_dir = Path(survey_path) / "Binaries"
    if not bin_dir.exists():
        return []
    manifests = []
    for m in bin_dir.glob("*.manifest.json"):
        if "post" in _dataset_type(m):
            manifests.append(m)
    return sorted(manifests, key=lambda p: p.name.lower())


class Viewer3D(QtWidgets.QWidget):
    def __init__(self, survey_path: Path):
        super().__init__()
        self.survey_path = Path(survey_path)

        self.vtk_widget = None
        self.renderer = None
        self.plane_x = None
        self.plane_y = None
        self.plane_z = None
        self.cmap_combo = None
        self._lut = None
        self._window_level = None

        layout = QtWidgets.QHBoxLayout(self)

        # Left: list of post-stack datasets
        left = QtWidgets.QVBoxLayout()
        left.addWidget(QtWidgets.QLabel("Post-stack datasets"))
        self.dataset_list = QtWidgets.QListWidget()
        self.dataset_list.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        left.addWidget(self.dataset_list, 1)
        layout.addLayout(left, 1)

        # Right: colormap controls + VTK canvas + status
        self.view_container = QtWidgets.QVBoxLayout()
        self.placeholder = QtWidgets.QLabel("Select a post-stack dataset to load 3D view.")
        self.placeholder.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

        controls = QtWidgets.QHBoxLayout()
        controls.addWidget(QtWidgets.QLabel("Colormap:"))
        self.cmap_combo = QtWidgets.QComboBox()
        self._populate_cmap_combo()
        controls.addWidget(self.cmap_combo)
        apply_btn = QtWidgets.QPushButton("Apply colormap")
        apply_btn.clicked.connect(self._apply_colormap)
        controls.addWidget(apply_btn)
        controls.addStretch(1)
        self.view_container.addLayout(controls)

        if QVTKRenderWindowInteractor is not None and vtk is not None:
            self.vtk_widget = QVTKRenderWindowInteractor(self)
            self.view_container.addWidget(self.vtk_widget, 4)
            rw = self.vtk_widget.GetRenderWindow()
            self.renderer = vtk.vtkRenderer()
            rw.AddRenderer(self.renderer)
        else:
            self.vtk_widget = None

        # You can show this somewhere if you like; for now we keep it invisible.
        self.placeholder.setVisible(False)
        layout.addLayout(self.view_container, 3)

        self._populate_list()
        self.dataset_list.itemChanged.connect(self._on_item_changed)
        # (Optional) second call, not strictly needed but harmless
        self._populate_cmap_combo()

    @staticmethod
    def _find_header(df: pd.DataFrame, preferred: str | None, fallbacks: list[str]) -> str | None:
        if preferred and preferred in df.columns:
            return preferred
        for c in fallbacks:
            if c in df.columns:
                return c
        lowmap = {c.lower(): c for c in df.columns}
        for c in fallbacks:
            if c.lower() in lowmap:
                return lowmap[c.lower()]
        return None

    def _load_snapshot(self, manifest: Path) -> tuple[np.ndarray, dict]:
        meta = json.loads(manifest.read_text())
        zarr_path = meta.get("zarr_store")
        geom_path = meta.get("geometry_parquet")
        if not zarr_path or not Path(zarr_path).exists():
            raise FileNotFoundError(f"Zarr not found: {zarr_path}")
        if not geom_path or not Path(geom_path).exists():
            raise FileNotFoundError(f"Geometry not found: {geom_path}")

        geom = pd.read_parquet(geom_path)

        # Load fully into NumPy (copy) to avoid mmap/thread issues
        amp = np.array(
            zarr.open(zarr_path, mode="r")["amplitude"][:],
            dtype=np.float32,
            copy=True,
            order="F",
        )

        sel = meta.get("selected_headers", {}) or {}
        inline_col = self._find_header(geom, sel.get("inline_header"), ["iline", "inline"])
        xline_col = self._find_header(geom, sel.get("xline_header"), ["xline", "crossline"])
        if inline_col is None or xline_col is None:
            raise ValueError("Inline/Xline headers not found in geometry.")

        ntr = amp.shape[1]
        if len(geom) != ntr:
            raise ValueError("Geometry length does not match trace count.")

        geom_sorted = geom.sort_values([inline_col, xline_col]).reset_index(drop=True)
        inlines = geom_sorted[inline_col].to_numpy()
        xlines = geom_sorted[xline_col].to_numpy()
        uniq_inline = np.unique(inlines)
        uniq_xline = np.unique(xlines)
        expected = len(uniq_inline) * len(uniq_xline)
        if expected != ntr:
            raise ValueError("Grid is not regular; cannot reshape to cube.")

        order_idx = geom_sorted.index.to_numpy()
        amp_sorted = amp[:, order_idx]
        vol = amp_sorted.reshape((amp.shape[0], len(uniq_inline), len(uniq_xline)))  # (nt, ni, nx)
        snapshot = vol.transpose(1, 2, 0)  # (inline, xline, depth/time)

        return snapshot, {
            "inline_header": inline_col,
            "xline_header": xline_col,
            "traces": ntr,
        }

    def _populate_list(self):
        self.dataset_list.clear()
        for m in _list_poststack_manifests(self.survey_path):
            item = QtWidgets.QListWidgetItem(
                m.stem.replace(".zarr", "").replace(".manifest", "")
            )
            item.setFlags(item.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable | QtCore.Qt.ItemFlag.ItemIsEnabled)
            item.setCheckState(QtCore.Qt.CheckState.Unchecked)
            item.setData(QtCore.Qt.ItemDataRole.UserRole, m)
            self.dataset_list.addItem(item)

    def _populate_cmap_combo(self):
        """Fill the combo with icons that show the colormap strip.
        The actual colormap name is stored in UserRole, and shown as tooltip.
        """
        if self.cmap_combo is None:
            return

        self.cmap_combo.clear()
        self.cmap_combo.setIconSize(QtCore.QSize(120, 16))

        names = ["gray", "viridis", "plasma", "inferno", "magma", "jet", "seismic", "coolwarm"]

        try:
            import matplotlib.cm as cm
        except Exception:
            # No matplotlib: fall back to plain text list
            self.cmap_combo.addItems(names)
            return

        for name in names:
            try:
                cmap = cm.get_cmap(name, 256)
            except Exception:
                continue

            # Build RGBA strip: 256 samples horizontally, 16 px high
            arr = (cmap(np.linspace(0, 1, 256)) * 255).astype(np.uint8)  # (256, 4)
            strip = np.repeat(arr[np.newaxis, :, :], 16, axis=0).copy(order="C")  # (16, 256, 4)

            img = QImage(
                strip.data,
                256,
                16,
                256 * 4,
                QImage.Format.Format_RGBA8888,
            ).scaled(120, 16, QtCore.Qt.AspectRatioMode.IgnoreAspectRatio)

            pix = QPixmap.fromImage(img)
            icon = QIcon(pix)

            idx = self.cmap_combo.count()
            # Add item with only icon (no visible text)
            self.cmap_combo.addItem(icon, "")

            # Store cmap name as UserRole & tooltip
            self.cmap_combo.setItemData(idx, name, QtCore.Qt.ItemDataRole.UserRole)
            self.cmap_combo.setItemData(idx, name, QtCore.Qt.ItemDataRole.ToolTipRole)

    def _on_item_changed(self, item: QtWidgets.QListWidgetItem):
        if item is None:
            return

        if item.checkState() == QtCore.Qt.CheckState.Unchecked:
            return

        # Uncheck others
        for i in range(self.dataset_list.count()):
            other = self.dataset_list.item(i)
            if other is not item and other.checkState() == QtCore.Qt.CheckState.Checked:
                other.setCheckState(QtCore.Qt.CheckState.Unchecked)

        manifest = item.data(QtCore.Qt.ItemDataRole.UserRole)
        if manifest is None or not Path(manifest).exists():
            return

        if self.vtk_widget is None or vtk is None:
            return

        try:
            snapshot, info = self._load_snapshot(Path(manifest))
        except Exception:
            return

        self._render_vtk_slices(snapshot)

    def _render_vtk_slices(self, snapshot: np.ndarray):
        """
        Use VTK image plane widgets for three orthogonal slice planes.
        snapshot shape: (Ni, Nx, Nt) = (inline, xline, time/depth)
        Planes are locked to axes and (where supported) only allowed to slide.
        """
        Ni, Nx, Nt = snapshot.shape

        # Clear previous plane widgets
        if self.plane_x is not None:
            self.plane_x.Off()
            self.plane_x = None
        if self.plane_y is not None:
            self.plane_y.Off()
            self.plane_y = None
        if self.plane_z is not None:
            self.plane_z.Off()
            self.plane_z = None

        # Reset renderer actors
        self.renderer.RemoveAllViewProps()

        # Build vtkImageData, using point scalars
        img = vtk.vtkImageData()
        img.SetDimensions(Ni, Nx, Nt)
        img.SetSpacing(1.0, 1.0, 1.0)
        img.SetOrigin(0.0, 0.0, 0.0)

        data_flat = snapshot.astype(np.float32, copy=False).ravel(order="F")
        vtk_arr = numpy_support.numpy_to_vtk(
            num_array=data_flat,
            deep=True,
            array_type=vtk.VTK_FLOAT,
        )
        img.GetPointData().SetScalars(vtk_arr)

        # Window/level and LUT from data range
        dmin, dmax = float(snapshot.min()), float(snapshot.max())
        if dmax > dmin:
            window = dmax - dmin
            level = 0.5 * (dmax + dmin)
        else:
            window = 1.0
            level = dmin

        lut = self._build_lut(self._current_cmap(), dmin, dmax)
        self._lut = lut
        self._window_level = (window, level, dmin, dmax)

        self.renderer.SetBackground(1.0, 1.0, 1.0)

        rw = self.vtk_widget.GetRenderWindow()
        interactor = rw.GetInteractor()

        # Interactor style for rotating the camera
        style = vtk.vtkInteractorStyleTrackballCamera()
        interactor.SetInteractorStyle(style)

        # Safely get action constants (may be missing in some VTK builds)
        PlaneClass = vtk.vtkImagePlaneWidget
        SLICE_ACTION = getattr(PlaneClass, "VTK_SLICE_MOTION_ACTION", None)
        NO_ACTION = getattr(PlaneClass, "VTK_NO_ACTION", None)

        def configure_plane(plane, orientation: str, slice_idx: int):
            plane.SetInputData(img)
            if orientation == "x":
                plane.SetPlaneOrientationToXAxes()
            elif orientation == "y":
                plane.SetPlaneOrientationToYAxes()
            elif orientation == "z":
                plane.SetPlaneOrientationToZAxes()

            plane.SetSliceIndex(slice_idx)
            plane.SetInteractor(interactor)
            plane.DisplayTextOff()
            plane.SetWindowLevel(window, level)
            plane.SetLookupTable(lut)

            # Keep the plane inside the volume and locked to axes
            plane.RestrictPlaneToVolumeOn()

            # If this VTK build exposes the action constants, use them
            if SLICE_ACTION is not None:
                plane.SetLeftButtonAction(SLICE_ACTION)
            if NO_ACTION is not None:
                plane.SetMiddleButtonAction(NO_ACTION)
                plane.SetRightButtonAction(NO_ACTION)

            plane.On()

        # Create three orthogonal planes
        self.plane_x = vtk.vtkImagePlaneWidget()
        configure_plane(self.plane_x, "x", max(Ni // 2, 0))

        self.plane_y = vtk.vtkImagePlaneWidget()
        configure_plane(self.plane_y, "y", max(Nx // 2, 0))

        self.plane_z = vtk.vtkImagePlaneWidget()
        configure_plane(self.plane_z, "z", max(Nt // 2, 0))

        # Camera / render
        self.renderer.ResetCamera()

        # Flip seismic vertical orientation (180 deg elevation)
        camera = self.renderer.GetActiveCamera()
        camera.Elevation(180)
        self.renderer.ResetCameraClippingRange()

        self.vtk_widget.Initialize()
        rw.Render()

    def _current_cmap(self) -> str:
        """Return current colormap name stored in combo's UserRole."""
        if self.cmap_combo is None:
            return "gray"

        idx = self.cmap_combo.currentIndex()
        if idx < 0:
            return "gray"

        name = self.cmap_combo.itemData(idx, QtCore.Qt.ItemDataRole.UserRole)
        if not name:
            return "gray"
        return str(name)

    def _build_lut(self, cmap_name: str, dmin: float, dmax: float):
        lut = vtk.vtkLookupTable()
        lut.SetNumberOfTableValues(256)
        lut.SetRange(dmin, dmax)
        try:
            import matplotlib.cm as cm
            cmap = cm.get_cmap(cmap_name, 256)
        except Exception:
            cmap = None

        if cmap is None:
            for i in range(256):
                g = i / 255.0
                lut.SetTableValue(i, g, g, g, 1.0)
        else:
            colors = cmap(np.linspace(0, 1, 256))
            for i, (r, g, b, a) in enumerate(colors):
                lut.SetTableValue(i, float(r), float(g), float(b), float(a))
        lut.Build()
        return lut

    def _apply_colormap(self):
        if self._lut is None or self._window_level is None:
            return
        if self.plane_x is None or self.plane_y is None or self.plane_z is None:
            return

        window, level, dmin, dmax = self._window_level
        lut = self._build_lut(self._current_cmap(), dmin, dmax)
        self._lut = lut

        for plane in (self.plane_x, self.plane_y, self.plane_z):
            plane.SetLookupTable(lut)
            plane.SetWindowLevel(window, level)
            plane.UpdatePlacement()

        self.renderer.Modified()
        self.vtk_widget.GetRenderWindow().Render()


def show_viewer3d(window):
    if not getattr(window, "currentSurveyPath", None):
        QtWidgets.QMessageBox.warning(window, "3D Viewer", "Please select a survey first.")
        return
    widget = Viewer3D(Path(window.currentSurveyPath))
    window.tabWidget.addTab(widget, "3D Viewer")
    window.tabWidget.setCurrentWidget(widget)


# Alias expected by MainWindow
show_3d_viewer = show_viewer3d
