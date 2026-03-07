"""Consistency checks between manuscript claims and benchmark tables."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

IssueLevel = Literal["ERROR", "WARN", "INFO"]


@dataclass(frozen=True)
class ConsistencyIssue:
    """A manuscript consistency issue."""

    code: str
    level: IssueLevel
    message: str


def _safe_float(value: str) -> float:
    return float(value.replace(",", ""))


def _approx_equal(a: float, b: float, atol: float = 1e-3) -> bool:
    return bool(np.isfinite(a) and np.isfinite(b) and abs(a - b) <= atol)


def _read_tables(tables_dir: Path) -> dict[str, pd.DataFrame]:
    required = {
        "benchmark2": "benchmark2_discovery.csv",
        "benchmark3": "benchmark3_power.csv",
        "multi_dataset": "multi_dataset_quality.csv",
        "zeroshot": "fm_advantage_zeroshot.csv",
        "lowdata": "fm_advantage_lowdata.csv",
        "coverage": "fm_advantage_coverage.csv",
        "reprod": "cross_dataset_reproducibility.csv",
    }
    tables: dict[str, pd.DataFrame] = {}
    for key, filename in required.items():
        path = tables_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing required table: {path}")
        tables[key] = pd.read_csv(path)
    return tables


def _find_float(pattern: str, text: str) -> float | None:
    m = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
    if not m:
        return None
    return _safe_float(m.group(1))


def validate_manuscript_consistency(
    manuscript_path: Path,
    tables_dir: Path,
) -> list[ConsistencyIssue]:
    """Validate major manuscript claims against benchmark CSV tables."""
    text = manuscript_path.read_text(encoding="utf-8")
    tables = _read_tables(tables_dir)
    issues: list[ConsistencyIssue] = []

    # 1) Explicit cNMF coverage/genes claim
    m = re.search(
        r"cNMF[^.]*?only\s+([0-9,]+)\s+genes\s+\(([0-9.]+)\\%\)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if m:
        claimed_genes = int(_safe_float(m.group(1)))
        claimed_pct = _safe_float(m.group(2))
        cov_df = tables["benchmark2"]
        row = cov_df[cov_df["method"] == "cNMF"]
        if row.empty:
            issues.append(
                ConsistencyIssue(
                    code="TABLE_MISSING_CNMF",
                    level="ERROR",
                    message="benchmark2_discovery.csv has no cNMF row.",
                )
            )
        else:
            coverage = float(row["coverage"].iloc[0])
            # PBMC3k section in manuscript currently states 1,838 HVGs.
            hvg_match = re.search(
                r"PBMC 3k[^.]*?([0-9,]+)\s+HVGs",
                text,
                flags=re.IGNORECASE | re.DOTALL,
            )
            n_hvg = int(_safe_float(hvg_match.group(1))) if hvg_match else 1838
            expected_genes = int(round(coverage * n_hvg))
            expected_pct = 100.0 * coverage
            if abs(claimed_genes - expected_genes) > 1 or abs(claimed_pct - expected_pct) > 0.2:
                issues.append(
                    ConsistencyIssue(
                        code="CNMF_COVERAGE_MISMATCH",
                        level="ERROR",
                        message=(
                            f"Claimed cNMF coverage {claimed_genes} genes ({claimed_pct:.1f}%) "
                            f"but table implies ~{expected_genes} genes ({expected_pct:.1f}%)."
                        ),
                    )
                )

    # 2) Power claims at FC=2.0 and FC=3.0
    power = tables["benchmark3"]
    for method, fc, pattern in [
        ("nPathway-KMeans", 2.0, r"2-fold enrichment threshold[^.]*?KMeans achieves TPR = ([0-9.]+)"),
        ("Expr-Cluster", 2.0, r"Expr-Cluster: TPR = ([0-9.]+), FPR = ([0-9.]+)"),
        ("nPathway-Leiden", 3.0, r"3-fold threshold[^.]*?Leiden achieves TPR = ([0-9.]+)"),
        ("cNMF", 3.0, r"3-fold threshold[^.]*?cNMF \(TPR = ([0-9.]+), FPR = ([0-9.]+)\)"),
        ("nPathway-Ensemble", 3.0, r"3-fold threshold[^.]*?Ensemble\s*\(TPR = ([0-9.]+), FPR = ([0-9.]+)\)"),
    ]:
        row = power[(power["method"] == method) & np.isclose(power["fold_change"], fc)]
        if row.empty:
            issues.append(
                ConsistencyIssue(
                    code="POWER_TABLE_ROW_MISSING",
                    level="ERROR",
                    message=f"Power table missing row for {method} at FC={fc}.",
                )
            )
            continue
        expected_tpr = float(row["tpr"].iloc[0])
        claimed_tpr = _find_float(pattern, text)
        if claimed_tpr is not None and not _approx_equal(claimed_tpr, expected_tpr, atol=0.02):
            issues.append(
                ConsistencyIssue(
                    code="POWER_TPR_MISMATCH",
                    level="ERROR",
                    message=(
                        f"{method} FC={fc} TPR claim {claimed_tpr:.3f} does not match "
                        f"table {expected_tpr:.3f}."
                    ),
                )
            )

    # 3) Low-data and zero-shot headline checks
    lowdata = tables["lowdata"]
    row_pca_50 = lowdata[(lowdata["embedding"] == "PCA") & (lowdata["n_cells"] == 50)]
    row_gf_50 = lowdata[(lowdata["embedding"] == "Geneformer") & (lowdata["n_cells"] == 50)]
    if not row_pca_50.empty and not row_gf_50.empty:
        expected_pca = float(row_pca_50["mean_jaccard"].iloc[0])
        expected_gf = float(row_gf_50["mean_jaccard"].iloc[0])
        claimed_gf = _find_float(
            r"At \$n = 50\$ cells,\s*Geneformer achieves Jaccard = ([0-9.]+)",
            text,
        )
        claimed_pca = _find_float(
            r"At \$n = 50\$ cells,[^.]*?versus Jaccard = ([0-9.]+) for\s*expression-based PCA",
            text,
        )
        if claimed_gf is not None and not _approx_equal(claimed_gf, expected_gf, atol=0.01):
            issues.append(
                ConsistencyIssue(
                    code="LOWDATA_GF50_MISMATCH",
                    level="ERROR",
                    message=f"Geneformer@50 claim {claimed_gf:.3f} != table {expected_gf:.3f}.",
                )
            )
        if claimed_pca is not None and not _approx_equal(claimed_pca, expected_pca, atol=0.01):
            issues.append(
                ConsistencyIssue(
                    code="LOWDATA_PCA50_MISMATCH",
                    level="ERROR",
                    message=f"PCA@50 claim {claimed_pca:.3f} != table {expected_pca:.3f}.",
                )
            )

    zeroshot = tables["zeroshot"]
    zs_row = zeroshot[
        (zeroshot["model"].str.lower() == "geneformer")
        & (zeroshot["graph_reg"] == "none")
        & (zeroshot["method"] == "kmeans")
    ]
    if not zs_row.empty:
        expected_zs = float(zs_row["mean_jaccard"].iloc[0])
        claimed_zs = _find_float(
            r"Geneformer KMeans achieves Jaccard = ([0-9.]+)\s+without any expression data",
            text,
        )
        if claimed_zs is not None and not _approx_equal(claimed_zs, expected_zs, atol=0.01):
            issues.append(
                ConsistencyIssue(
                    code="ZEROSHOT_MISMATCH",
                    level="ERROR",
                    message=f"Zero-shot Geneformer claim {claimed_zs:.3f} != table {expected_zs:.3f}.",
                )
            )

    # 4) Reproducibility vs random contradiction check
    if re.search(r"all methods substantially exceed random reproducibility", text, re.IGNORECASE):
        reprod = tables["reprod"]
        means = reprod[reprod["dataset_pair"] == "mean"].set_index("method")["similarity"]
        if "Random" in means.index:
            random_mean = float(means["Random"])
            offenders = [
                m for m, v in means.items() if m != "Random" and float(v) <= random_mean
            ]
            if offenders:
                issues.append(
                    ConsistencyIssue(
                        code="REPRO_RANDOM_CONTRADICTION",
                        level="ERROR",
                        message=(
                            "Text says all methods exceed random, but these do not: "
                            + ", ".join(sorted(offenders))
                        ),
                    )
                )

    # 5) PCA all-genes claim must have backing row in coverage table
    pca_all_claim = re.search(
        r"PCA:\s*Jaccard\s*=\s*([0-9.]+)\s*on HVGs\s*vs\.\s*([0-9.]+)\s*on all genes",
        text,
        flags=re.IGNORECASE,
    )
    if pca_all_claim:
        coverage = tables["coverage"]
        pca_all = coverage[
            (coverage["scope"] == "all_genes")
            & (coverage["embedding"].str.lower() == "pca")
        ]
        if pca_all.empty:
            issues.append(
                ConsistencyIssue(
                    code="PCA_ALLGENES_UNBACKED",
                    level="ERROR",
                    message="Manuscript claims PCA all-genes Jaccard but coverage table has no all_genes/PCA row.",
                )
            )

    # 6) Placeholder metadata checks
    if "Author Names" in text or "email@institution.edu" in text:
        issues.append(
            ConsistencyIssue(
                code="PLACEHOLDER_AUTHOR",
                level="WARN",
                message="Manuscript still contains placeholder author metadata.",
            )
        )
    if "github.com/author/" in text:
        issues.append(
            ConsistencyIssue(
                code="PLACEHOLDER_REPO_URL",
                level="WARN",
                message="Manuscript contains placeholder GitHub URL (github.com/author/...).",
            )
        )

    # 7) Derived artifact freshness
    tex_mtime = manuscript_path.stat().st_mtime
    for suffix in (".pdf", ".docx"):
        candidate = manuscript_path.with_suffix(suffix)
        if candidate.exists() and candidate.stat().st_mtime < tex_mtime:
            issues.append(
                ConsistencyIssue(
                    code="STALE_ARTIFACT",
                    level="INFO",
                    message=f"{candidate.name} is older than {manuscript_path.name}.",
                )
            )

    if not issues:
        issues.append(
            ConsistencyIssue(
                code="CONSISTENT",
                level="INFO",
                message="No manuscript/table inconsistencies detected.",
            )
        )
    return issues
