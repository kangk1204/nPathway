"""Top-level nPathway CLI for beginner-friendly public workflows."""

from __future__ import annotations

import importlib.metadata
import sys


_TOP_LEVEL_HELP = """nPathway

Use one command and subcommands:

  npathway quickstart
  npathway demo bulk --output-dir results/demo_bulk_case_vs_control
  npathway demo scrna --output-dir results/demo_scrna_case_vs_control
  npathway validate bulk --matrix data/my_bulk_matrix.csv --metadata data/my_bulk_metadata.csv --sample-col sample --group-col condition --group-a case --group-b control
  npathway run bulk --matrix data/my_bulk_matrix.csv --metadata data/my_bulk_metadata.csv --sample-col sample --group-col condition --group-a case --group-b control --output-dir results/my_bulk_run
  npathway run scrna --adata data/my_scrna.h5ad --condition-col condition --case case --control control --output-dir results/my_scrna_run
  npathway compare --ranked-genes ranked.csv --dynamic-gmt dynamic_programs.gmt --curated-gmt curated.gmt
  npathway references build --output-dir data/reference/public
  npathway convert seurat --check-only
  npathway convert 10x --check-only

Short aliases:

  npathway bulk ...
  npathway scrna ...
  npathway check ...
  npathway gsea ...

Installed direct aliases still work:

  npathway-demo
  npathway-validate-inputs
  npathway-bulk-workflow
  npathway-scrna-easy
  npathway-compare-gsea
  npathway-convert-seurat
  npathway-convert-10x
"""

_RUN_HELP = """nPathway run

Run a discovery workflow:

  npathway run bulk --help
  npathway run scrna --help
"""

_CONVERT_HELP = """nPathway convert

Prepare scRNA input files for nPathway:

  npathway convert seurat --check-only
  npathway convert 10x --check-only
"""

_QUICKSTART = """nPathway quickstart

1. Install once:
   bash scripts/install_npathway_easy.sh

2. Activate the local environment:
   source .venv/bin/activate

3. Smoke-test the install:
   npathway demo bulk --output-dir results/demo_bulk_case_vs_control
   npathway demo scrna --output-dir results/demo_scrna_case_vs_control

4. Open the generated dashboards:
   results/demo_bulk_case_vs_control/index.html
   results/demo_scrna_case_vs_control/index.html

5. Validate your own inputs:
   npathway validate bulk --help
   npathway validate scrna --help

6. Run discovery:
   npathway run bulk --help
   npathway run scrna --help

7. Compare against curated GSEA on the same ranking:
  npathway compare --help
  npathway references --help
"""


def _print(message: str) -> None:
    print(message.rstrip())


def _version() -> str:
    try:
        return importlib.metadata.version("npathway")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _dispatch_demo(argv: list[str]) -> None:
    from npathway.cli import demo

    demo.main(argv)


def _dispatch_validate(argv: list[str]) -> None:
    from npathway.cli import validate_inputs

    validate_inputs.main(argv)


def _dispatch_bulk(argv: list[str]) -> None:
    from npathway.cli import bulk_workflow

    bulk_workflow.main(argv)


def _dispatch_scrna(argv: list[str]) -> None:
    from npathway.cli import scrna_easy

    scrna_easy.main(argv)


def _dispatch_compare(argv: list[str]) -> None:
    from npathway.cli import compare_gsea

    compare_gsea.main(argv)


def _dispatch_public_references(argv: list[str]) -> None:
    from npathway.cli import public_references

    public_references.main(argv)


def _dispatch_convert_seurat(argv: list[str]) -> None:
    from npathway.cli import convert_seurat

    convert_seurat.main(argv)


def _dispatch_convert_10x(argv: list[str]) -> None:
    from npathway.cli import convert_10x

    convert_10x.main(argv)


def _handle_run(argv: list[str]) -> None:
    if not argv or argv[0] in {"-h", "--help"}:
        _print(_RUN_HELP)
        raise SystemExit(0)
    if argv[0] == "bulk":
        _dispatch_bulk(argv[1:])
        return
    if argv[0] == "scrna":
        _dispatch_scrna(argv[1:])
        return
    raise SystemExit(f"Unknown workflow '{argv[0]}'. Use `npathway run bulk` or `npathway run scrna`.")


def _handle_convert(argv: list[str]) -> None:
    if not argv or argv[0] in {"-h", "--help"}:
        _print(_CONVERT_HELP)
        raise SystemExit(0)
    if argv[0] == "seurat":
        _dispatch_convert_seurat(argv[1:])
        return
    if argv[0] == "10x":
        _dispatch_convert_10x(argv[1:])
        return
    raise SystemExit(
        f"Unknown conversion target '{argv[0]}'. Use `npathway convert seurat` or `npathway convert 10x`."
    )


def _handle_references(argv: list[str]) -> None:
    if not argv or argv[0] in {"-h", "--help"}:
        _print(
            "nPathway references\n\n"
            "Build broader public pathway stacks for annotation:\n\n"
            "  npathway references build --output-dir data/reference/public\n"
        )
        raise SystemExit(0)
    if argv[0] == "build":
        _dispatch_public_references(argv[1:])
        return
    raise SystemExit(
        f"Unknown references command '{argv[0]}'. Use `npathway references build`."
    )


def main(argv: list[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help", "help"}:
        _print(_TOP_LEVEL_HELP)
        return

    command = args[0]
    rest = args[1:]

    try:
        if command == "quickstart":
            _print(_QUICKSTART)
            return
        if command == "version":
            print(_version())
            return
        if command == "demo":
            _dispatch_demo(rest)
            return
        if command in {"validate", "check"}:
            _dispatch_validate(rest)
            return
        if command == "run":
            _handle_run(rest)
            return
        if command in {"bulk", "bulk-workflow"}:
            _dispatch_bulk(rest)
            return
        if command in {"scrna", "scrna-easy"}:
            _dispatch_scrna(rest)
            return
        if command in {"compare", "gsea"}:
            _dispatch_compare(rest)
            return
        if command == "references":
            _handle_references(rest)
            return
        if command == "convert":
            _handle_convert(rest)
            return
        if command == "seurat":
            _dispatch_convert_seurat(rest)
            return
        if command == "10x":
            _dispatch_convert_10x(rest)
            return
        raise SystemExit(
            f"Unknown command '{command}'. Run `npathway --help` for the public entrypoints."
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        print(
            "Hint: run `npathway quickstart` or `npathway --help` for the beginner-facing commands.",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
