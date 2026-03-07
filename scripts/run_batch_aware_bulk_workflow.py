#!/usr/bin/env python3
"""Run the batch-aware bulk nPathway workflow from a repo checkout."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from npathway.cli.bulk_workflow import main


if __name__ == "__main__":
    main()
