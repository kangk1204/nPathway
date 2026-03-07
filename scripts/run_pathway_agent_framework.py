"""Generate planning artifacts for the pathway multi-agent framework."""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from npathway.agents import build_default_pathway_framework


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Build markdown/json planning outputs for pathway agents.",
    )
    parser.add_argument(
        "--markdown-out",
        type=Path,
        default=None,
        help="Output path for markdown plan.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Output path for JSON plan.",
    )
    return parser.parse_args()


def main() -> None:
    """Run planning output generation."""
    args = parse_args()
    today = date.today()
    stamp = today.strftime("%Y%m%d")

    markdown_out = args.markdown_out or Path(f"docs/pathway_agent_framework_{stamp}.md")
    json_out = args.json_out or Path(f"results/pathway_agent_plan_{stamp}.json")

    framework = build_default_pathway_framework()
    md_path, js_path = framework.write_outputs(
        markdown_path=markdown_out,
        json_path=json_out,
        generated_on=today,
    )

    print("Created pathway framework outputs:")
    print(f"- Markdown: {md_path}")
    print(f"- JSON: {js_path}")


if __name__ == "__main__":
    main()
