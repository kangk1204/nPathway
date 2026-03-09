"""CLI wrapper for building public pathway reference stacks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from npathway.data.public_references import build_public_reference_stack


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for building the public pathway reference stack."""
    parser = argparse.ArgumentParser(
        description=(
            "Download and merge public pathway collections for nPathway annotation. "
            "Supported sources: Reactome, WikiPathways, Pathway Commons."
        )
    )
    parser.add_argument(
        "--collections",
        default="reactome,wikipathways,pathwaycommons",
        help=(
            "Comma-separated public collections to merge. "
            "Default: reactome,wikipathways,pathwaycommons"
        ),
    )
    parser.add_argument(
        "--species",
        default="human",
        choices=["human", "mouse"],
        help="Reference species for download and harmonization. Default: human",
    )
    parser.add_argument(
        "--output-dir",
        default="data/reference/public",
        help="Directory where the merged GMT and manifest will be written.",
    )
    parser.add_argument(
        "--output-stem",
        default=None,
        help="Optional output filename stem. Default: public_reference_stack_<species>",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Run the public-reference build command and print a JSON summary."""
    args = parse_args(argv)
    collections = tuple(
        item.strip().lower()
        for item in str(args.collections).split(",")
        if item.strip()
    )
    merged = build_public_reference_stack(
        collections=collections,
        species=str(args.species),
        output_dir=args.output_dir,
        output_stem=args.output_stem,
    )
    output_dir = Path(args.output_dir)
    stem = args.output_stem or f"public_reference_stack_{args.species}"
    print(
        json.dumps(
            {
                "output_gmt": str((output_dir / f"{stem}.gmt").resolve()),
                "manifest": str((output_dir / f"{stem}_manifest.json").resolve()),
                "n_gene_sets": len(merged),
                "collections": list(collections),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
