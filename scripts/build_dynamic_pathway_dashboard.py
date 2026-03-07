#!/usr/bin/env python3
"""Build interactive dashboard + figure/table package from dynamic pathway outputs."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from npathway.reporting import DashboardConfig, build_dynamic_dashboard_package


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", required=True, help="Directory from bulk dynamic run.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output dashboard directory (default: <results-dir>/dashboard).",
    )
    parser.add_argument(
        "--title",
        default="nPathway Dynamic Pathway Dashboard",
        help="Dashboard title.",
    )
    parser.add_argument("--top-k", type=int, default=20, help="Top-K rows used for focused plots.")
    parser.add_argument(
        "--no-pdf",
        action="store_true",
        help="Disable PDF figure export (PNG only).",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable INFO logs.")
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir) if args.output_dir else results_dir / "dashboard"
    cfg = DashboardConfig(
        results_dir=str(results_dir),
        output_dir=str(output_dir),
        title=args.title,
        top_k=max(5, args.top_k),
        include_pdf=not args.no_pdf,
    )
    artifacts = build_dynamic_dashboard_package(cfg)
    print("Dashboard package generated.")
    print(f"- html: {artifacts.html_path}")
    print(f"- figures: {artifacts.figure_dir}")
    print(f"- tables: {artifacts.table_dir}")
    print(f"- summary_table: {artifacts.summary_table_path}")


if __name__ == "__main__":
    main()
