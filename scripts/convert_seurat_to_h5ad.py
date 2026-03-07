#!/usr/bin/env python3
"""Convert a Seurat .rds object into .h5ad from a repo checkout."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from npathway.cli.convert_seurat import main


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        print(
            "Hint: run `python scripts/convert_seurat_to_h5ad.py --check-only` first to verify that R, Seurat, and SeuratDisk are available.",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc
