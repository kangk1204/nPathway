#!/usr/bin/env python3
"""Validate bulk or scRNA input files before running nPathway."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from npathway.cli.validate_inputs import main


if __name__ == "__main__":
    main()
