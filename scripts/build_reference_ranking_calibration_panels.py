from __future__ import annotations

import argparse
from pathlib import Path
import textwrap

import matplotlib.pyplot as plt
import pandas as pd

from npathway.evaluation.pathway_annotation import (
    family_interpretation_score,
    reference_relevance_band,
    reference_relevance_score,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "results" / "methods_submission_figures" / "reference_ranking_calibration_20260309"
DEFAULT_DATASETS: tuple[tuple[str, Path], ...] = (
    ("GSE203206 bulk AD", ROOT / "results" / "ad_bulk_GSE203206_public_refs_20260309"),
    ("GSE214921 bulk MDD", ROOT / "results" / "gse214921_mdd_public_refs_20260309"),
)


def _wrap(text: str, width: int = 24) -> str:
    return "\n".join(textwrap.wrap(str(text), width=width, break_long_words=False))


def _load_family_summary(result_dir: Path) -> pd.DataFrame:
    family_df = pd.read_csv(result_dir / "annotation" / "program_reference_family_summary.csv")
    if "disease_priority_score" not in family_df.columns:
        family_df["disease_priority_score"] = family_df["family_key"].astype(str).map(reference_relevance_score)
    if "priority_band" not in family_df.columns:
        family_df["priority_band"] = family_df["family_key"].astype(str).map(reference_relevance_band)
    if "source_count" not in family_df.columns:
        family_df["source_count"] = family_df["sources_display"].fillna("").astype(str).map(
            lambda value: len([part for part in value.split(",") if part.strip()])
        )
    if "interpretation_score" not in family_df.columns:
        family_df["interpretation_score"] = family_df.apply(
            lambda row: family_interpretation_score(
                reference_name=str(row["family_key"]),
                best_jaccard=row["best_jaccard"],
                programs_covered=int(row["programs_covered"]),
                source_count=int(max(row["source_count"], 1)),
            ),
            axis=1,
        )
    return family_df


def _calibration_rows(family_df: pd.DataFrame, top_n: int = 8) -> pd.DataFrame:
    raw_ranked = family_df.sort_values(
        ["best_jaccard", "programs_covered", "references_merged"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    raw_ranked["raw_rank"] = range(1, len(raw_ranked) + 1)

    prioritized = family_df.sort_values(
        ["interpretation_score", "disease_priority_score", "best_jaccard", "programs_covered"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    prioritized["prioritized_rank"] = range(1, len(prioritized) + 1)

    merged = raw_ranked.merge(
        prioritized.loc[:, ["family_key", "prioritized_rank"]],
        on="family_key",
        how="outer",
    )
    merged["raw_top"] = pd.to_numeric(merged["raw_rank"], errors="coerce").le(top_n)
    merged["prioritized_top"] = pd.to_numeric(merged["prioritized_rank"], errors="coerce").le(top_n)
    merged = merged.loc[merged["raw_top"] | merged["prioritized_top"]].copy()
    merged["family_axis"] = merged["family_display"].astype(str).map(_wrap)
    return merged.sort_values(
        ["prioritized_rank", "raw_rank", "interpretation_score", "best_jaccard"],
        ascending=[True, True, False, False],
        na_position="last",
    ).reset_index(drop=True)


def _band_color(band: str) -> str:
    mapping = {
        "Disease-prioritized": "#c0392b",
        "Supportive": "#d68910",
        "Background": "#95a5a6",
    }
    return mapping.get(str(band), "#95a5a6")


def _plot_dataset(ax_raw, ax_prior, dataset_label: str, calib: pd.DataFrame) -> dict[str, object]:
    raw_view = calib.sort_values(["raw_rank", "best_jaccard"], ascending=[True, False]).head(8).copy()
    prior_view = calib.sort_values(
        ["prioritized_rank", "interpretation_score"],
        ascending=[True, False],
    ).head(8).copy()

    raw_colors = ["#7b8794" if not bool(v) else "#4c78a8" for v in raw_view["prioritized_top"]]
    ax_raw.barh(
        raw_view["family_axis"].iloc[::-1],
        raw_view["best_jaccard"].iloc[::-1],
        color=raw_colors[::-1],
        edgecolor="white",
        linewidth=1.0,
    )
    ax_raw.set_xlabel("Best Jaccard")
    ax_raw.set_title(f"{dataset_label}: raw overlap ranking", loc="left", fontsize=11, fontweight="bold")
    ax_raw.grid(axis="x", alpha=0.18)

    prior_colors = [_band_color(band) for band in prior_view["priority_band"]]
    ax_prior.barh(
        prior_view["family_axis"].iloc[::-1],
        prior_view["interpretation_score"].iloc[::-1],
        color=prior_colors[::-1],
        edgecolor="white",
        linewidth=1.0,
    )
    ax_prior.set_xlabel("Interpretation score")
    ax_prior.set_title(f"{dataset_label}: interpretation-prioritized ranking", loc="left", fontsize=11, fontweight="bold")
    ax_prior.grid(axis="x", alpha=0.18)

    moved_in = int(((calib["prioritized_top"]) & (~calib["raw_top"])).sum())
    moved_out = int(((calib["raw_top"]) & (~calib["prioritized_top"])).sum())
    top_family = prior_view.iloc[0] if not prior_view.empty else None
    return {
        "dataset": dataset_label,
        "moved_in": moved_in,
        "moved_out": moved_out,
        "top_family": str(top_family["family_display"]) if top_family is not None else "",
        "top_interpretation_score": float(top_family["interpretation_score"]) if top_family is not None else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build raw-vs-prioritized reference-family calibration panels.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        len(DEFAULT_DATASETS),
        2,
        figsize=(13.2, 4.8 * len(DEFAULT_DATASETS)),
        gridspec_kw={"width_ratios": [1.0, 1.1], "wspace": 0.30, "hspace": 0.48},
    )
    if len(DEFAULT_DATASETS) == 1:
        axes = [axes]

    rows: list[dict[str, object]] = []
    for (dataset_label, result_dir), ax_pair in zip(DEFAULT_DATASETS, axes, strict=False):
        family_df = _load_family_summary(result_dir.resolve())
        calib = _calibration_rows(family_df)
        rows.append(_plot_dataset(ax_pair[0], ax_pair[1], dataset_label, calib))

    fig.suptitle(
        "Supplementary Figure 3. Raw-overlap vs interpretation-prioritized family ranking",
        x=0.06,
        y=0.995,
        ha="left",
        fontsize=14,
        fontweight="bold",
    )
    fig.text(
        0.06,
        0.01,
        "Interpretation prioritization is a downstream readability layer. It does not alter discovery assignments or GSEA statistics; it reorders family labels to reduce domination by generic housekeeping terms.",
        ha="left",
        va="bottom",
        fontsize=9,
        color="#34495e",
    )

    png_path = output_dir / "reference_ranking_calibration_combined.png"
    pdf_path = output_dir / "reference_ranking_calibration_combined.pdf"
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    pd.DataFrame(rows).to_csv(output_dir / "reference_ranking_calibration_summary.csv", index=False)
    (output_dir / "caption_starter.md").write_text(
        "Supplementary Figure 3. Calibration of reference-family ranking before and after interpretation prioritization. "
        "The left column shows families ranked by raw best Jaccard alone. The right column shows the same families ranked by an interpretation score that combines conservative disease relevance, overlap strength, program coverage, and source diversity. "
        "This layer improves readability without changing the underlying discovery or enrichment statistics.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
