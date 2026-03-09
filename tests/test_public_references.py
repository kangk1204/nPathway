"""Tests for public pathway reference download and harmonization helpers."""

from __future__ import annotations

import gzip
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from npathway.data import public_references as pr


def _write_plain_gmt(path: Path, content: str) -> Path:
    path.write_text(content, encoding="utf-8")
    return path


def _write_gzip_gmt(path: Path, content: str) -> Path:
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write(content)
    return path


def _write_zip_gmt(path: Path, member_name: str, content: str) -> Path:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(member_name, content)
    return path


def test_load_public_reference_gene_sets_normalizes_across_formats(monkeypatch, tmp_path) -> None:
    """Public reference loader should normalize plain, gzip, and zip GMT files."""
    files = {
        "reactome": _write_zip_gmt(
            tmp_path / "reactome.zip",
            "ReactomePathways.gmt",
            "Innate Signaling\tNA\tapoe\tapoe\tTREM2\tTYROBP\n",
        ),
        "wikipathways": _write_plain_gmt(
            tmp_path / "wikipathways.gmt",
            "Alzheimer Disease\tNA\tAPP\tAPOE\tTREM2\n",
        ),
        "pathwaycommons": _write_gzip_gmt(
            tmp_path / "pathwaycommons.gmt.gz",
            "Complement Cascade\tNA\tC1QA\tC1QB\tC3\n",
        ),
    }

    monkeypatch.setattr(
        pr,
        "_download_collection",
        lambda *, collection, species, cache_dir=None: files[collection],
    )

    reactome = pr.load_public_reference_gene_sets("reactome", species="human")
    wikipathways = pr.load_public_reference_gene_sets("wikipathways", species="human")
    pathwaycommons = pr.load_public_reference_gene_sets("pathwaycommons", species="human")

    assert reactome == {"REACTOME::Innate Signaling": ["APOE", "TREM2", "TYROBP"]}
    assert wikipathways == {"WP::Alzheimer Disease": ["APP", "APOE", "TREM2"]}
    assert pathwaycommons == {"PATHWAYCOMMONS::Complement Cascade": ["C1QA", "C1QB", "C3"]}


def test_build_public_reference_stack_writes_manifest(monkeypatch, tmp_path) -> None:
    """Merged public stack should materialize GMT and manifest metadata."""
    files = {
        "reactome": _write_plain_gmt(
            tmp_path / "reactome.gmt",
            "Innate Signaling\tNA\tAPOE\tTREM2\tTYROBP\n",
        ),
        "wikipathways": _write_plain_gmt(
            tmp_path / "wikipathways.gmt",
            "Alzheimer Disease\tNA\tAPP\tAPOE\tTREM2\n",
        ),
    }

    monkeypatch.setattr(
        pr,
        "_download_collection",
        lambda *, collection, species, cache_dir=None: files[collection],
    )

    outdir = tmp_path / "public"
    merged = pr.build_public_reference_stack(
        collections=("reactome", "wikipathways"),
        species="human",
        output_dir=outdir,
        output_stem="stack",
    )

    assert len(merged) == 2
    assert (outdir / "stack.gmt").exists()
    manifest_path = outdir / "stack_manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["species"] == "human"
    assert manifest["total_gene_sets"] == 2
    assert {entry["name"] for entry in manifest["collections"]} == {"reactome", "wikipathways"}


def test_resolve_wikipathways_url_from_listing(monkeypatch) -> None:
    """WikiPathways resolver should find the latest species-specific GMT filename."""
    html = """
    <html><body>
    <a href="wikipathways-20260110-gmt-Homo_sapiens.gmt">older</a>
    <a href="wikipathways-20260210-gmt-Homo_sapiens.gmt">latest</a>
    </body></html>
    """
    monkeypatch.setattr(pr, "_fetch_text", lambda url: html)
    resolved = pr._resolve_wikipathways_url("human")
    assert resolved.endswith("wikipathways-20260210-gmt-Homo_sapiens.gmt")
