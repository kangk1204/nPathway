#!/usr/bin/env python3
"""Validate bulk or scRNA input files before running nPathway."""

from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path

from npathway.pipeline import (
    ValidationReport,
    validate_bulk_input_files,
    validate_scrna_pseudobulk_input,
)


def _add_bulk_subparser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("bulk", help="Validate bulk RNA-seq matrix + metadata files.")
    parser.add_argument("--matrix", required=True, help="Path to bulk matrix CSV/TSV.")
    parser.add_argument("--metadata", required=True, help="Path to metadata CSV/TSV.")
    parser.add_argument("--sample-col", default="sample", help="Metadata sample ID column.")
    parser.add_argument("--group-col", required=True, help="Metadata contrast column.")
    parser.add_argument("--group-a", required=True, help="First group label.")
    parser.add_argument("--group-b", required=True, help="Second group label.")
    parser.add_argument(
        "--batch-col",
        default=None,
        help="Optional known batch column to validate for batch-aware workflows.",
    )
    parser.add_argument(
        "--matrix-orientation",
        default="genes_by_samples",
        choices=["genes_by_samples", "samples_by_genes"],
        help="Input matrix orientation.",
    )
    parser.add_argument(
        "--sep",
        default=None,
        help="Delimiter override for matrix and metadata (default: auto infer).",
    )
    parser.add_argument(
        "--raw-counts",
        action="store_true",
        help="Validate the matrix under raw-count assumptions (non-negative integers).",
    )
    parser.add_argument(
        "--html-out",
        default=None,
        help="Optional path to write a user-friendly HTML validation report.",
    )
    parser.add_argument(
        "--json-out",
        default=None,
        help="Optional path to write the validation summary as JSON.",
    )


def _add_scrna_subparser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("scrna", help="Validate scRNA `.h5ad` input for the pseudobulk route.")
    parser.add_argument("--adata", required=True, help="Path to input AnnData `.h5ad`.")
    parser.add_argument("--sample-col", required=True, help="Sample/donor column in `adata.obs`.")
    parser.add_argument("--group-col", required=True, help="Case/control column in `adata.obs`.")
    parser.add_argument("--group-a", required=True, help="First group label.")
    parser.add_argument("--group-b", required=True, help="Second group label.")
    parser.add_argument("--subset-col", default=None, help="Optional cell subset column.")
    parser.add_argument("--subset-value", default=None, help="Optional cell subset value.")
    parser.add_argument("--layer", default=None, help="Optional AnnData layer to inspect.")
    parser.add_argument(
        "--use-raw",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Expect the workflow to aggregate from `adata.raw` when available (default: true).",
    )
    parser.add_argument(
        "--min-cells-per-sample",
        type=int,
        default=10,
        help="Minimum cells required in each pseudobulk sample.",
    )
    parser.add_argument(
        "--html-out",
        default=None,
        help="Optional path to write a user-friendly HTML validation report.",
    )
    parser.add_argument(
        "--json-out",
        default=None,
        help="Optional path to write the validation summary as JSON.",
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    _add_bulk_subparser(subparsers)
    _add_scrna_subparser(subparsers)
    return parser.parse_args(argv)


def _write_json(report: ValidationReport, path: str | None) -> None:
    """Optionally write a report to disk as JSON."""
    if path is None:
        return
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")


def _build_next_steps(mode: str, *, success: bool) -> list[str]:
    """Return short next-step guidance for the HTML report."""
    if success and mode == "bulk":
        return [
            "Run `python scripts/run_bulk_dynamic_pathway.py ...` with the same input files.",
            "Keep `--raw-counts` enabled if your matrix contains integer count data.",
            "Open `docs/quickstart_input_guide.md` if you need a first-run walkthrough.",
        ]
    if success and mode == "scrna":
        return [
            "Run `python scripts/run_scrna_pseudobulk_dynamic_pathway.py ...` with the same `.h5ad` file.",
            "Check donor/sample labels again if you change `--subset-col` or `--min-cells-per-sample`.",
            "Open `docs/quickstart_input_guide.md` if you need a first-run walkthrough.",
        ]
    if mode == "bulk":
        return [
            "Fix the reported matrix/metadata issue and rerun the validator.",
            "Use `data/templates/` or `data/bulk_demo_case_vs_ctrl/` as a formatting reference.",
            "Keep sample IDs identical between the matrix and metadata.",
        ]
    return [
        "Fix the reported `.h5ad` / `adata.obs` issue and rerun the validator.",
        "Check that each donor/sample maps to exactly one group label.",
        "Reduce `--min-cells-per-sample` only if the biological design still remains defensible.",
    ]


def _write_html_report(
    *,
    mode: str,
    path: str | None,
    success: bool,
    report: ValidationReport | None = None,
    error_message: str | None = None,
) -> None:
    """Write a compact HTML validation report."""
    if path is None:
        return

    title = f"nPathway Input Validation - {mode}"
    status_label = "VALID" if success else "INVALID"
    status_class = "ok" if success else "bad"
    summary_items = ""
    warning_items = ""
    error_block = ""

    if report is not None:
        summary_items = "\n".join(
            f"<tr><th>{html.escape(str(key))}</th><td>{html.escape(str(value))}</td></tr>"
            for key, value in report.summary.items()
        )
        if report.warnings:
            warning_items = "\n".join(
                f"<li>{html.escape(warning)}</li>" for warning in report.warnings
            )
    if error_message is not None:
        error_block = (
            "<section><h2>Error</h2>"
            f"<pre>{html.escape(error_message)}</pre>"
            "</section>"
        )

    next_steps = "\n".join(
        f"<li>{html.escape(step)}</li>"
        for step in _build_next_steps(mode, success=success)
    )
    html_text = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{html.escape(title)}</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f7f5ef;
      --panel: #fffdf8;
      --ink: #1f2421;
      --muted: #5f665f;
      --ok: #1f6d3d;
      --bad: #9f2f25;
      --border: #d9d3c5;
      --accent: #9d6b2f;
    }}
    body {{
      margin: 0;
      padding: 32px;
      font: 16px/1.5 "IBM Plex Sans", "Segoe UI", sans-serif;
      background: linear-gradient(180deg, #f5f2ea 0%, #ede7da 100%);
      color: var(--ink);
    }}
    main {{
      max-width: 920px;
      margin: 0 auto;
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 18px;
      padding: 28px;
      box-shadow: 0 18px 48px rgba(31, 36, 33, 0.08);
    }}
    .status {{
      display: inline-block;
      padding: 6px 12px;
      border-radius: 999px;
      font-weight: 700;
      letter-spacing: 0.04em;
      background: rgba(0, 0, 0, 0.04);
    }}
    .status.ok {{ color: var(--ok); background: rgba(31, 109, 61, 0.12); }}
    .status.bad {{ color: var(--bad); background: rgba(159, 47, 37, 0.12); }}
    h1, h2 {{ margin: 0 0 12px; }}
    p {{ color: var(--muted); }}
    section {{ margin-top: 24px; }}
    table {{
      width: 100%;
      border-collapse: collapse;
      border: 1px solid var(--border);
      border-radius: 12px;
      overflow: hidden;
    }}
    th, td {{
      padding: 10px 12px;
      border-bottom: 1px solid var(--border);
      text-align: left;
      vertical-align: top;
    }}
    th {{
      width: 32%;
      color: var(--accent);
      background: #faf5ea;
    }}
    ul {{ padding-left: 22px; }}
    pre {{
      white-space: pre-wrap;
      background: #f7f1e4;
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 14px;
      margin: 0;
    }}
  </style>
</head>
<body>
  <main>
    <span class="status {status_class}">{status_label}</span>
    <h1>{html.escape(title)}</h1>
    <p>{html.escape('Input looks ready for the next nPathway step.' if success else 'Input needs fixes before the workflow should be run.')}</p>
    {error_block}
    <section>
      <h2>Summary</h2>
      <table>
        <tbody>
          {summary_items if summary_items else '<tr><th>mode</th><td>' + html.escape(mode) + '</td></tr>'}
        </tbody>
      </table>
    </section>
    <section>
      <h2>Warnings</h2>
      {('<ul>' + warning_items + '</ul>') if warning_items else '<p>No warnings.</p>'}
    </section>
    <section>
      <h2>Next Steps</h2>
      <ul>{next_steps}</ul>
    </section>
  </main>
</body>
</html>
"""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html_text, encoding="utf-8")


def _print_report(report: ValidationReport) -> None:
    """Print a user-friendly validation summary."""
    print(f"VALID: {report.mode} input looks ready.")
    for key, value in report.summary.items():
        print(f"- {key}: {value}")
    if report.warnings:
        print("- warnings:")
        for warning in report.warnings:
            print(f"  - {warning}")


def _build_report(args: argparse.Namespace) -> ValidationReport:
    """Run the appropriate validator for the selected mode."""
    if args.mode == "bulk":
        return validate_bulk_input_files(
            matrix_path=args.matrix,
            metadata_path=args.metadata,
            sample_col=args.sample_col,
            group_col=args.group_col,
            group_a=args.group_a,
            group_b=args.group_b,
            matrix_orientation=args.matrix_orientation,
            sep=args.sep,
            raw_counts=bool(args.raw_counts),
            batch_col=args.batch_col,
        )
    return validate_scrna_pseudobulk_input(
        adata_path=args.adata,
        sample_col=args.sample_col,
        group_col=args.group_col,
        group_a=args.group_a,
        group_b=args.group_b,
        subset_col=args.subset_col,
        subset_value=args.subset_value,
        layer=args.layer,
        use_raw=bool(args.use_raw),
        min_cells_per_sample=args.min_cells_per_sample,
    )


def main(argv: list[str] | None = None) -> None:
    """CLI entrypoint."""
    args_for_failure: argparse.Namespace | None = None
    try:
        args_for_failure = parse_args(argv)
        report = _build_report(args_for_failure)
        _write_json(report, args_for_failure.json_out)
        _write_html_report(
            mode=report.mode,
            path=args_for_failure.html_out,
            success=True,
            report=report,
        )
        _print_report(report)
    except SystemExit:
        raise
    except Exception as exc:
        if args_for_failure is not None:
            _write_html_report(
                mode=str(getattr(args_for_failure, "mode", "unknown")),
                path=getattr(args_for_failure, "html_out", None),
                success=False,
                error_message=str(exc),
            )
        print(f"INVALID: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
