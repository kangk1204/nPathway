#!/usr/bin/env python3
"""Build a merged public pathway reference stack from a repo checkout."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from npathway.cli.public_references import main


if __name__ == "__main__":
    main()
