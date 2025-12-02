from PyQt6 import QtWidgets
from SurveyDialogs import ImportSEGYDialog
from openseismicprocessing.catalog import list_projects


def load_segy_files(window):
    """Launch the SEGY import dialog and record the last dataset type."""
    if not window.rootFolderPath:
        QtWidgets.QMessageBox.warning(window, "Warning", "Please select a root folder first!")
        return

    boundary = None
    if window.currentSurveyName:
        try:
            project = next(p for p in list_projects() if p["name"] == window.currentSurveyName)
            metadata = project.get("metadata", {}) or {}
            boundary = metadata.get("boundary", metadata)
        except StopIteration:
            boundary = None

    is_new_survey = not bool(getattr(window, "currentSurveyName", None))
    dialog = ImportSEGYDialog(
        boundary=boundary,
        survey_root=window.currentSurveyPath,
        is_new_survey=is_new_survey,
    )
    result = dialog.exec()
    try:
        if result == QtWidgets.QDialog.DialogCode.Accepted:
            window.last_dataset_type = dialog.dataset_type()
    except Exception:
        pass
