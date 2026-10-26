# -*- coding: utf-8 -*-
"""Source-tree entry point; equivalent to python -m pyannote_dual.cli."""
from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from pyannote_dual.cli import main  # noqa: E402

if __name__ == "__main__":
    main()
