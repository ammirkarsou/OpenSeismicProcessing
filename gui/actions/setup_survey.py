import os
from PyQt6 import QtWidgets
from dialog_box import DialogBox


def setup_survey(window):
    """Open the survey selection dialog and update the current survey context."""
    if not window.rootFolderPath:
        QtWidgets.QMessageBox.warning(window, "Warning", "Please select a root folder first!")
        return

    window.folder_list = sorted(
        [
            folder
            for folder in os.listdir(window.rootFolderPath)
            if os.path.isdir(os.path.join(window.rootFolderPath, folder))
        ],
        key=str.lower,
    )

    dialog = DialogBox(window, selected_survey=window.currentSurveyName)
    if dialog.exec():
        selected = dialog.GetSelectedSurvey()
        if selected:
            window.currentSurveyName = selected
            window.currentSurveyPath = os.path.join(window.rootFolderPath, selected)
