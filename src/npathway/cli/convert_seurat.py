"""Convert a Seurat .rds object into .h5ad for nPathway scRNA workflows."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass
class RPackageStatus:
    """Availability status for one required R package."""
    name: str
    available: bool


_REQUIRED_PACKAGES = ("Seurat", "SeuratDisk")
_R_STATUS_EXPR = "pkgs <- c(%s); cat(paste(pkgs, sapply(pkgs, requireNamespace, quietly=TRUE), sep='='), sep='\\n')" % \
    ",".join(f'\"{pkg}\"' for pkg in _REQUIRED_PACKAGES)
_R_CONVERT_SCRIPT = r'''
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 4) {
  stop("Expected 4 arguments: rds_path, h5seurat_path, h5ad_path, assay")
}
rds_path <- args[[1]]
h5seurat_path <- args[[2]]
h5ad_path <- args[[3]]
assay <- args[[4]]

if (!requireNamespace("Seurat", quietly = TRUE)) {
  stop("R package 'Seurat' is required but not installed.")
}
if (!requireNamespace("SeuratDisk", quietly = TRUE)) {
  stop("R package 'SeuratDisk' is required but not installed.")
}

obj <- readRDS(rds_path)
class_names <- class(obj)
if (!("Seurat" %in% class_names)) {
  stop(paste0("Input .rds is not a Seurat object. Classes: ", paste(class_names, collapse = ", ")))
}

obj <- SeuratObject::UpdateSeuratObject(obj)
if (!identical(assay, "")) {
  if (!(assay %in% names(obj@assays))) {
    stop(paste0("Requested assay '", assay, "' not found. Available assays: ", paste(names(obj@assays), collapse = ", ")))
  }
  Seurat::DefaultAssay(obj) <- assay
}

SeuratDisk::SaveH5Seurat(obj, filename = h5seurat_path, overwrite = TRUE, verbose = FALSE)
converted_path <- SeuratDisk::Convert(h5seurat_path, dest = "h5ad", overwrite = TRUE, verbose = FALSE)
if (!file.exists(converted_path)) {
  alt_path <- sub("\\.h5seurat$", ".h5ad", h5seurat_path)
  if (file.exists(alt_path)) {
    converted_path <- alt_path
  } else {
    stop("SeuratDisk conversion finished but the .h5ad output was not found.")
  }
}
ok <- file.copy(converted_path, h5ad_path, overwrite = TRUE)
if (!ok) {
  stop("Failed to copy converted .h5ad file to the requested output path.")
}
cat(paste0("default_assay=", Seurat::DefaultAssay(obj)), "\n")
cat(paste0("h5ad_path=", h5ad_path), "\n")
'''


def _check_r_packages() -> list[RPackageStatus]:
    try:
        result = subprocess.run(
            ["Rscript", "-e", _R_STATUS_EXPR],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("Rscript was not found. Install R before converting Seurat objects.") from exc
    if result.returncode != 0:
        stderr = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"Failed to query R package availability: {stderr}")
    statuses: list[RPackageStatus] = []
    for line in result.stdout.splitlines():
        if "=" not in line:
            continue
        name, flag = line.strip().split("=", 1)
        statuses.append(RPackageStatus(name=name, available=flag.strip().upper() == "TRUE"))
    return statuses


def _write_status_report(path: Path, *, input_path: Path | None, output_path: Path | None, statuses: list[RPackageStatus]) -> None:
    payload = {
        "input_path": None if input_path is None else str(input_path),
        "output_path": None if output_path is None else str(output_path),
        "packages": [{"name": item.name, "available": item.available} for item in statuses],
        "ready": all(item.available for item in statuses),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for Seurat-to-h5ad conversion."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seurat-rds", default=None, help="Path to the input Seurat .rds object.")
    parser.add_argument("--output-h5ad", default=None, help="Output .h5ad path (default: same stem as --seurat-rds).")
    parser.add_argument("--assay", default="", help="Optional Seurat assay to set as default before conversion.")
    parser.add_argument("--keep-h5seurat", action="store_true", help="Keep the intermediate .h5seurat file.")
    parser.add_argument("--check-only", action="store_true", help="Only check R package availability and write a readiness report.")
    parser.add_argument("--status-json", default=None, help="Optional path to write the readiness/conversion report as JSON.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Run the public Seurat conversion entrypoint."""
    args = parse_args(argv)
    statuses = _check_r_packages()
    ready = all(item.available for item in statuses)

    input_path = None if args.seurat_rds is None else Path(args.seurat_rds)
    output_path = None if args.output_h5ad is None else Path(args.output_h5ad)
    status_json = Path(args.status_json) if args.status_json else None
    if status_json is not None:
        _write_status_report(status_json, input_path=input_path, output_path=output_path, statuses=statuses)

    print("nPathway Seurat conversion readiness")
    for item in statuses:
        print(f"- {item.name}: {'OK' if item.available else 'MISSING'}")

    if args.check_only:
        print(f"- ready: {ready}")
        return

    if args.seurat_rds is None:
        raise ValueError("--seurat-rds is required unless --check-only is used.")
    if not ready:
        raise RuntimeError(
            "Seurat conversion requires R packages Seurat and SeuratDisk. "
            "Install them first, then rerun this command."
        )

    input_path = Path(args.seurat_rds)
    if not input_path.exists():
        raise FileNotFoundError(f"Input Seurat .rds file was not found: {input_path}")
    output_path = Path(args.output_h5ad) if args.output_h5ad else input_path.with_suffix(".h5ad")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="npathway_seurat_convert_") as tmpdir:
        tmpdir_path = Path(tmpdir)
        r_script_path = tmpdir_path / "convert_seurat_to_h5ad.R"
        r_script_path.write_text(_R_CONVERT_SCRIPT, encoding="utf-8")
        h5seurat_path = tmpdir_path / (input_path.stem + ".h5seurat")
        result = subprocess.run(
            [
                "Rscript",
                str(r_script_path),
                str(input_path),
                str(h5seurat_path),
                str(output_path),
                str(args.assay),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            stderr = result.stderr.strip() or result.stdout.strip()
            raise RuntimeError(f"Seurat conversion failed: {stderr}")
        if args.keep_h5seurat:
            kept_path = output_path.with_suffix(".h5seurat")
            kept_path.write_bytes(h5seurat_path.read_bytes())
            print(f"- h5seurat: {kept_path}")

    if status_json is not None:
        payload = json.loads(status_json.read_text(encoding="utf-8"))
        payload.update({"converted": True, "output_h5ad": str(output_path), "assay": args.assay})
        status_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("nPathway Seurat conversion completed.")
    print(f"- input_rds: {input_path}")
    print(f"- output_h5ad: {output_path}")
    print(f"- assay: {args.assay or 'default'}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        print(
            "Hint: run `npathway convert seurat --check-only` first to verify that R, Seurat, and SeuratDisk are available.",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc
