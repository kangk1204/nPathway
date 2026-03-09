"""Public pathway/reference collection download and harmonization helpers.

This module extends nPathway beyond MSigDB-centric annotation by adding
official public pathway resources that can be merged into a broader
reference layer for program annotation and dashboard reporting.
"""

from __future__ import annotations

import gzip
import json
import logging
import re
import shutil
import urllib.request
import zipfile
from pathlib import Path

from npathway.utils.gmt_io import write_gmt

logger = logging.getLogger(__name__)

_CACHE_DIR = Path.home() / ".npathway" / "data" / "public_references"

_SUPPORTED_COLLECTIONS = {"reactome", "wikipathways", "pathwaycommons"}
_SUPPORTED_SPECIES = {"human", "mouse"}
_SOURCE_PREFIX = {
    "reactome": "REACTOME",
    "wikipathways": "WP",
    "pathwaycommons": "PATHWAYCOMMONS",
}
_WP_SPECIES_TOKEN = {
    "human": "Homo_sapiens",
    "mouse": "Mus_musculus",
}
_PATHWAYCOMMONS_GMT = {
    "human": "pc-hgnc.gmt.gz",
}


def _ensure_cache_dir() -> Path:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return _CACHE_DIR


def _fetch_text(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "nPathway/0.1"})
    with urllib.request.urlopen(req, timeout=120) as response:
        return response.read().decode("utf-8", errors="replace")


def _download_file(url: str, dest: Path, description: str) -> Path:
    if dest.exists():
        logger.info("Using cached %s at %s", description, dest)
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "nPathway/0.1"})
    with urllib.request.urlopen(req, timeout=180) as response:
        with open(dest, "wb") as handle:
            shutil.copyfileobj(response, handle)
    return dest


def _read_gmt_from_plain(path: Path) -> dict[str, list[str]]:
    gene_sets: dict[str, list[str]] = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.rstrip("\n\r")
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            name = str(parts[0]).strip()
            genes = [str(g).strip() for g in parts[2:] if str(g).strip()]
            if name:
                gene_sets[name] = genes
    return gene_sets


def _read_gmt_from_gzip(path: Path) -> dict[str, list[str]]:
    gene_sets: dict[str, list[str]] = {}
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            line = line.rstrip("\n\r")
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            name = str(parts[0]).strip()
            genes = [str(g).strip() for g in parts[2:] if str(g).strip()]
            if name:
                gene_sets[name] = genes
    return gene_sets


def _read_gmt_from_zip(path: Path) -> dict[str, list[str]]:
    with zipfile.ZipFile(path) as archive:
        gmt_members = [name for name in archive.namelist() if name.lower().endswith(".gmt")]
        if not gmt_members:
            raise ValueError(f"No GMT file found inside archive: {path}")
        member = sorted(gmt_members)[0]
        with archive.open(member) as handle:
            content = handle.read().decode("utf-8", errors="replace").splitlines()
    gene_sets: dict[str, list[str]] = {}
    for line in content:
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        name = str(parts[0]).strip()
        genes = [str(g).strip() for g in parts[2:] if str(g).strip()]
        if name:
            gene_sets[name] = genes
    return gene_sets


def _read_any_gmt(path: Path) -> dict[str, list[str]]:
    suffixes = [s.lower() for s in path.suffixes]
    if suffixes[-2:] == [".gmt", ".gz"] or suffixes[-1:] == [".gz"]:
        return _read_gmt_from_gzip(path)
    if suffixes[-1:] == [".zip"]:
        return _read_gmt_from_zip(path)
    return _read_gmt_from_plain(path)


def _normalize_gene_sets(
    gene_sets: dict[str, list[str]],
    *,
    source_prefix: str,
    uppercase_genes: bool = True,
    min_genes: int = 3,
) -> dict[str, list[str]]:
    normalized: dict[str, list[str]] = {}
    for raw_name, genes in gene_sets.items():
        clean_name = str(raw_name).strip()
        if not clean_name:
            continue
        seen: set[str] = set()
        ordered_genes: list[str] = []
        for gene in genes:
            clean_gene = str(gene).strip()
            if uppercase_genes:
                clean_gene = clean_gene.upper()
            if clean_gene and clean_gene not in seen:
                ordered_genes.append(clean_gene)
                seen.add(clean_gene)
        if len(ordered_genes) < min_genes:
            continue
        normalized[f"{source_prefix}::{clean_name}"] = ordered_genes
    return normalized


def _resolve_reactome_url(species: str) -> str:
    if species != "human":
        raise ValueError("Reactome GMT support is currently limited to human.")
    return "https://reactome.org/download/current/ReactomePathways.gmt.zip"


def _resolve_wikipathways_url(species: str) -> str:
    species_token = _WP_SPECIES_TOKEN.get(species)
    if species_token is None:
        raise ValueError(
            f"WikiPathways species '{species}' is unsupported. "
            f"Choose from: {sorted(_WP_SPECIES_TOKEN)}"
        )
    listing_url = "https://data.wikipathways.org/current/gmt/"
    html = _fetch_text(listing_url)
    pattern = re.compile(
        rf'href="(wikipathways-[^"]+-gmt-{re.escape(species_token)}\.gmt)"',
        flags=re.IGNORECASE,
    )
    matches = pattern.findall(html)
    if not matches:
        raise RuntimeError(f"Could not locate a WikiPathways GMT for species '{species}'.")
    filename = sorted(matches)[-1]
    return listing_url + filename


def _resolve_pathwaycommons_url(species: str) -> str:
    filename = _PATHWAYCOMMONS_GMT.get(species)
    if filename is None:
        raise ValueError(
            f"Pathway Commons species '{species}' is unsupported. "
            f"Choose from: {sorted(_PATHWAYCOMMONS_GMT)}"
        )
    listing_url = "https://download.baderlab.org/PathwayCommons/PC2/"
    html = _fetch_text(listing_url)
    versions = re.findall(r'href="(v\d+)/"', html, flags=re.IGNORECASE)
    if not versions:
        raise RuntimeError("Could not determine the latest Pathway Commons PC2 version.")
    latest = sorted(versions, key=lambda value: int(value.lstrip("v")))[-1]
    return f"{listing_url}{latest}/{filename}"


def _download_collection(
    *,
    collection: str,
    species: str,
    cache_dir: Path | None = None,
) -> Path:
    collection = str(collection).strip().lower()
    species = str(species).strip().lower()
    if collection not in _SUPPORTED_COLLECTIONS:
        raise ValueError(
            f"Unknown public reference collection '{collection}'. "
            f"Choose from: {sorted(_SUPPORTED_COLLECTIONS)}"
        )
    if species not in _SUPPORTED_SPECIES:
        raise ValueError(
            f"Unknown species '{species}'. Choose from: {sorted(_SUPPORTED_SPECIES)}"
        )
    base = cache_dir if cache_dir is not None else (_ensure_cache_dir() / species / collection)
    base.mkdir(parents=True, exist_ok=True)
    if collection == "reactome":
        url = _resolve_reactome_url(species)
    elif collection == "wikipathways":
        url = _resolve_wikipathways_url(species)
    else:
        url = _resolve_pathwaycommons_url(species)
    filename = Path(url).name or f"{collection}_{species}.gmt"
    return _download_file(url, base / filename, f"{collection}:{species}")


def download_public_reference_gmt(
    collection: str,
    species: str = "human",
    cache_dir: Path | None = None,
) -> Path:
    """Download an official public reference gene-set file."""
    return _download_collection(collection=collection, species=species, cache_dir=cache_dir)


def load_public_reference_gene_sets(
    collection: str,
    species: str = "human",
    cache_dir: Path | None = None,
    *,
    uppercase_genes: bool = True,
    min_genes: int = 3,
) -> dict[str, list[str]]:
    """Download, read, and normalize one public reference collection."""
    path = _download_collection(collection=collection, species=species, cache_dir=cache_dir)
    raw = _read_any_gmt(path)
    return _normalize_gene_sets(
        raw,
        source_prefix=_SOURCE_PREFIX[collection],
        uppercase_genes=uppercase_genes,
        min_genes=min_genes,
    )


def build_public_reference_stack(
    *,
    collections: tuple[str, ...] = ("reactome", "wikipathways", "pathwaycommons"),
    species: str = "human",
    cache_dir: Path | None = None,
    output_dir: str | Path | None = None,
    output_stem: str | None = None,
    uppercase_genes: bool = True,
    min_genes: int = 3,
) -> dict[str, list[str]]:
    """Build a merged public reference stack and optionally materialize it."""
    merged: dict[str, list[str]] = {}
    manifest: dict[str, object] = {
        "species": species,
        "collections": [],
        "total_gene_sets": 0,
        "total_unique_genes": 0,
    }
    for collection in collections:
        gene_sets = load_public_reference_gene_sets(
            collection=collection,
            species=species,
            cache_dir=cache_dir,
            uppercase_genes=uppercase_genes,
            min_genes=min_genes,
        )
        merged.update(gene_sets)
        manifest["collections"].append(  # type: ignore[union-attr]
            {
                "name": collection,
                "source_prefix": _SOURCE_PREFIX[collection],
                "n_gene_sets": len(gene_sets),
                "n_unique_genes": len({g for genes in gene_sets.values() for g in genes}),
            }
        )

    manifest["total_gene_sets"] = len(merged)
    manifest["total_unique_genes"] = len({g for genes in merged.values() for g in genes})

    if output_dir is not None:
        outdir = Path(output_dir)
        outdir.mkdir(parents=True, exist_ok=True)
        stem = output_stem or f"public_reference_stack_{species}"
        write_gmt(merged, outdir / f"{stem}.gmt")
        (outdir / f"{stem}_manifest.json").write_text(
            json.dumps(manifest, indent=2),
            encoding="utf-8",
        )
    return merged
