#!/usr/bin/env python3
"""Run the nPathway comparison-only mode from a repo checkout."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from npathway.cli.compare_gsea import main


if __name__ == "__main__":
    main()
