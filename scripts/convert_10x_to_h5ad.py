#!/usr/bin/env python3
"""Convert raw 10x inputs into .h5ad from a repo checkout."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from npathway.cli.convert_10x import main


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        print(
            "Hint: run `python scripts/convert_10x_to_h5ad.py --check-only` first, then convert raw 10x data into .h5ad before using run_scrna_easy.py.",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc
