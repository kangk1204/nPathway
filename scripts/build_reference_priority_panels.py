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
    source_interpretation_score,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "results" / "methods_submission_figures" / "reference_layers_disease_prioritized_20260309"
DEFAULT_DATASETS: tuple[tuple[str, Path], ...] = (
    ("GSE203206 bulk AD", ROOT / "results" / "ad_bulk_GSE203206_public_refs_20260309"),
    ("GSE214921 bulk MDD", ROOT / "results" / "gse214921_mdd_public_refs_20260309"),
)


def _wrap(text: str, width: int = 30) -> str:
    return "\n".join(textwrap.wrap(str(text), width=width, break_long_words=False))


def _load_prioritized_tables(result_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    source_path = result_dir / "annotation" / "program_reference_source_summary.csv"
    family_path = result_dir / "annotation" / "program_reference_family_summary.csv"
    source_df = pd.read_csv(source_path)
    family_df = pd.read_csv(family_path)

    if "best_priority_score" not in source_df.columns:
        source_df["best_priority_score"] = 0.0
    if "mean_priority_score" not in source_df.columns:
        source_df["mean_priority_score"] = 0.0
    if "interpretation_score" not in source_df.columns:
        source_df["interpretation_score"] = source_df.apply(
            lambda row: source_interpretation_score(
                best_priority_score=row["best_priority_score"],
                best_jaccard=row["best_jaccard"],
                programs_covered=int(row["programs_covered"]),
                references_covered=int(row["references_covered"]),
            ),
            axis=1,
        )

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

    source_df = source_df.sort_values(
        ["interpretation_score", "best_priority_score", "best_jaccard", "programs_covered", "references_covered"],
        ascending=[False, False, False, False, False],
    ).reset_index(drop=True)
    family_df = family_df.sort_values(
        ["interpretation_score", "disease_priority_score", "best_jaccard", "programs_covered", "references_merged"],
        ascending=[False, False, False, False, False],
    ).reset_index(drop=True)
    return source_df, family_df


def _focus_family_rows(family_df: pd.DataFrame, n: int = 6) -> pd.DataFrame:
    focused = family_df[family_df["disease_priority_score"].fillna(0.0) > 0].copy()
    if focused.empty:
        focused = family_df.copy()
    focused = focused.head(n).copy()
    focused["family_axis"] = focused["family_display"].astype(str).map(lambda x: _wrap(x, 28))
    return focused.iloc[::-1].reset_index(drop=True)


def _focus_source_rows(source_df: pd.DataFrame, n: int = 6) -> pd.DataFrame:
    focused = source_df.copy().head(n)
    focused["source_axis"] = focused["source_display"].astype(str)
    return focused.iloc[::-1].reset_index(drop=True)


def _band_color(band: str) -> str:
    mapping = {
        "Disease-prioritized": "#c0392b",
        "Supportive": "#d68910",
        "Background": "#95a5a6",
    }
    return mapping.get(str(band), "#95a5a6")


def _build_panel(dataset_label: str, result_dir: Path, output_dir: Path) -> dict[str, object]:
    source_df, family_df = _load_prioritized_tables(result_dir)
    family_focus = _focus_family_rows(family_df)
    source_focus = _focus_source_rows(source_df)

    fig, (ax_family, ax_source) = plt.subplots(
        1,
        2,
        figsize=(12.6, 4.8),
        gridspec_kw={"width_ratios": [1.8, 1.0], "wspace": 0.28},
    )

    family_colors = [_band_color(band) for band in family_focus["priority_band"].astype(str)]
    ax_family.barh(
        family_focus["family_axis"],
        family_focus["best_jaccard"],
        color=family_colors,
        edgecolor="white",
        linewidth=1.0,
    )
    ax_family.set_xlabel("Best program-reference Jaccard")
    ax_family.set_title("Interpretation-prioritized reference families", loc="left", fontsize=12, fontweight="bold")
    ax_family.grid(axis="x", alpha=0.18)
    for _, row in family_focus.iterrows():
        ax_family.text(
            float(row["best_jaccard"]) + 0.008,
            row["family_axis"],
            f"{row['priority_band']} | score {float(row['interpretation_score']):.2f}",
            va="center",
            fontsize=8,
            color="#34495e",
        )

    scatter = ax_source.scatter(
        source_focus["best_jaccard"],
        source_focus["source_axis"],
        s=source_focus["references_covered"].astype(float).clip(lower=1.0) * 24.0,
        c=source_focus["best_priority_score"],
        cmap="viridis",
        edgecolors="white",
        linewidths=0.9,
    )
    ax_source.set_xlabel("Best Jaccard within source")
    ax_source.set_title("Reference-source support", loc="left", fontsize=12, fontweight="bold")
    ax_source.grid(axis="x", alpha=0.18)
    for _, row in source_focus.iterrows():
        ax_source.text(
            float(row["best_jaccard"]) + 0.008,
            row["source_axis"],
            f"{int(row['programs_covered'])} programs",
            va="center",
            fontsize=8,
            color="#34495e",
        )
    cbar = fig.colorbar(scatter, ax=ax_source, shrink=0.88, pad=0.02)
    cbar.set_label("Best disease-priority score")

    fig.suptitle(dataset_label, x=0.06, y=0.99, ha="left", fontsize=14, fontweight="bold")
    top_family = family_df.iloc[0] if not family_df.empty else None
    footer = (
        f"Top focused family: {top_family['family_display']} "
        f"(interpretation {float(top_family['interpretation_score']):.2f}, "
        f"best Jaccard {float(top_family['best_jaccard']):.3f})"
        if top_family is not None
        else "No family hits available."
    )
    fig.text(0.06, 0.01, footer, ha="left", va="bottom", fontsize=9, color="#34495e")

    stem = dataset_label.lower().replace(" ", "_").replace("-", "_")
    stem = "".join(ch for ch in stem if ch.isalnum() or ch == "_").strip("_")
    png_path = output_dir / f"{stem}_reference_layers_compact.png"
    pdf_path = output_dir / f"{stem}_reference_layers_compact.pdf"
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "dataset": dataset_label,
        "result_dir": str(result_dir),
        "top_family": str(top_family["family_display"]) if top_family is not None else "",
        "top_family_interpretation_score": float(top_family["interpretation_score"]) if top_family is not None else 0.0,
        "top_family_priority_score": float(top_family["disease_priority_score"]) if top_family is not None else 0.0,
        "top_family_best_jaccard": float(top_family["best_jaccard"]) if top_family is not None else 0.0,
        "n_focused_families": int(len(family_focus)),
        "n_sources_shown": int(len(source_focus)),
        "png_path": str(png_path),
        "pdf_path": str(pdf_path),
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build disease-prioritized compact reference-layer panels.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for dataset_label, result_dir in DEFAULT_DATASETS:
        rows.append(_build_panel(dataset_label, result_dir.resolve(), output_dir))

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(output_dir / "reference_layers_compact_summary.csv", index=False)
    (output_dir / "caption_starter.md").write_text(
        "Supplementary Figure 2. Interpretation-prioritized compact summaries of the broadened reference layer. "
        "Families are ranked first by an interpretation score that combines a conservative disease-relevance heuristic, overlap strength, program coverage, and source diversity, "
        "so generic housekeeping pathways do not dominate the interpretation merely because they produce large Jaccard values. "
        "Source-layer support is shown alongside the prioritized family panel to preserve auditability across Hallmark, GO, KEGG, Reactome, WikiPathways, and Pathway Commons where available.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
