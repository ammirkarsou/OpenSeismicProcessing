import os

from PyQt6 import QtWidgets, uic
from PyQt6.QtGui import QAction, QFont

from actions.basemap_view import show_basemap
from actions.headers_view import show_headers
from actions.load_segy import load_segy_files
from actions.manage_datasets import manage_datasets
from actions.select_root import select_root_folder
from actions.setup_survey import setup_survey
from actions.viewer2d import show_2d_viewer
from actions.viewer_prestack import show_prestack_viewer
from actions.viewer3d import show_3d_viewer
from actions.processing_view import show_processing
from openseismicprocessing.catalog import get_workspace_root, init_db


class OpenSeismicProcessingWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi("golem.ui", self)

        init_db()
        stored_root = get_workspace_root()
        self.rootFolderPath = stored_root if stored_root and os.path.isdir(stored_root) else ""
        self.currentSurveyPath = None
        self.currentSurveyName = None

        self.actionSetRootFolder = self.findChild(QAction, "action_Set_Root_Folder")
        self.actionSetupSurvey = self.findChild(QAction, "action_Select_Setup")
        self.actionLoadSegy = self.findChild(QAction, "action_Seg_y_file")
        self.actionBaseMap = self.findChild(QAction, "action_Base_Map")
        self.actionHeaders = self.findChild(QAction, "action_Headers")
        self.action3DViewer = self.findChild(QAction, "action_3D_Viewer")
        self.actionManage = self.findChild(QAction, "action_Manage")
        self.action2DViewer = self.findChild(QAction, "action_2D_Viewer")
        self.actionPreStackViewer = self.findChild(QAction, "action_Pre_stack_Viewer")
        self.menuProcessing = self.findChild(QtWidgets.QMenu, "menu_Processing")
        self.actionProcessing = self.findChild(QAction, "action_Processing")
        if self.actionProcessing is None and self.menuProcessing:
            self.actionProcessing = QAction("Processing", self)
            self.actionProcessing.setObjectName("action_Processing")
            self.menuProcessing.addAction(self.actionProcessing)

        self.tabWidget = QtWidgets.QTabWidget()
        self.tabWidget.setTabsClosable(True)
        self.tabWidget.tabCloseRequested.connect(self.close_tab)
        self.setCentralWidget(self.tabWidget)

        if self.actionSetRootFolder:
            self.actionSetRootFolder.triggered.connect(self.SelectRootFolder)
        if self.actionSetupSurvey:
            self.actionSetupSurvey.triggered.connect(self.SetupSurvey)
        if self.actionLoadSegy:
            self.actionLoadSegy.triggered.connect(self.LoadSegyFiles)
        if self.actionBaseMap:
            self.actionBaseMap.triggered.connect(self.ShowBasemap)
        if self.actionHeaders:
            self.actionHeaders.triggered.connect(self.ShowHeaders)
        if self.action3DViewer:
            self.action3DViewer.triggered.connect(self.Show3DViewer)
        if self.actionManage:
            self.actionManage.triggered.connect(self.ManageDatasets)
        if self.action2DViewer:
            self.action2DViewer.triggered.connect(self.Show2DViewer)
        if self.actionPreStackViewer:
            self.actionPreStackViewer.triggered.connect(self.ShowPreStackViewer)
        if self.actionProcessing:
            self.actionProcessing.triggered.connect(self.ShowProcessing)
        self._harmonize_toolbar_fonts()

    def _harmonize_toolbar_fonts(self):
        # Use the window font (non-bold) as the reference to avoid mismatched weights.
        base_font: QFont = QFont(self.font())
        base_font.setBold(False)
        for toolbar in self.findChildren(QtWidgets.QToolBar):
            toolbar.setFont(base_font)
            for action in toolbar.actions():
                action.setFont(base_font)
            for widget in toolbar.findChildren(QtWidgets.QWidget):
                widget.setFont(base_font)

    def close_tab(self, index):
        widget = self.tabWidget.widget(index)
        self.tabWidget.removeTab(index)
        if widget:
            widget.deleteLater()

    def SelectRootFolder(self):
        select_root_folder(self)

    def SetupSurvey(self):
        setup_survey(self)

    def LoadSegyFiles(self):
        load_segy_files(self)

    def ShowBasemap(self):
        show_basemap(self)

    def ShowHeaders(self):
        show_headers(self)

    def Show3DViewer(self):
        show_3d_viewer(self)

    def ManageDatasets(self):
        manage_datasets(self)

    def Show2DViewer(self):
        show_2d_viewer(self)

    def ShowPreStackViewer(self):
        show_prestack_viewer(self)

    def ShowProcessing(self):
        show_processing(self)
