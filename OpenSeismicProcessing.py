#!/usr/bin/env python3
"""Launcher for the Open Seismic Processing GUI from the repo root."""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Point imports at the existing GUI modules.
ROOT = Path(__file__).resolve().parent
GUI_DIR = ROOT / "gui"
sys.path.insert(0, str(GUI_DIR))

# Keep relative Qt asset paths (golem.ui, *.png, *.ui) working.
os.chdir(GUI_DIR)

from OpenSeismicProcessingApp import main  # noqa: E402


if __name__ == "__main__":
    sys.exit(main())
