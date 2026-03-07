#!/usr/bin/env python3
"""Run the beginner-friendly scRNA easy workflow from a repo checkout."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from npathway.cli.scrna_easy import main


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        print(
            "Hint: start with `python scripts/run_scrna_easy.py --wizard-only ...` to inspect auto-detected columns and eligible cell types.",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc
