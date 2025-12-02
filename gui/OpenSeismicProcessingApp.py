#!/usr/bin/env python3
"""Launcher for the Open Seismic Processing GUI."""

import os
import sys

# Avoid DBus theme warnings in sandboxed/portal-less environments.
os.environ.setdefault("QT_QPA_PLATFORMTHEME", "")

from PyQt6.QtWidgets import QApplication

from MainWindow import OpenSeismicProcessingWindow


def main() -> int:
    app = QApplication(sys.argv)
    window = OpenSeismicProcessingWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
