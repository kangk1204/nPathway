"""Tests for visualization helpers."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from npathway.utils.visualization import (
    plot_benchmark_summary,
    plot_enrichment_comparison,
    plot_program_overlap_heatmap,
    plot_program_sizes,
)


def test_plot_program_overlap_heatmap_empty_inputs() -> None:
    """Empty inputs should return a figure instead of raising."""
    fig = plot_program_overlap_heatmap({}, {})
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_benchmark_summary_empty_inputs() -> None:
    """Empty benchmark metrics should return a figure instead of raising."""
    fig = plot_benchmark_summary({})
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_program_sizes_empty_inputs() -> None:
    """Empty gene programs should return a figure instead of warnings/errors."""
    fig = plot_program_sizes({})
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_enrichment_comparison_accepts_preranked_gsea_schema() -> None:
    """Default plotting should work with lowercase GSEA output columns."""
    learned = pd.DataFrame(
        [
            {"program": "prog_A", "nes": 1.2, "p_value": 0.01, "fdr": 0.02},
            {"program": "prog_B", "nes": -1.0, "p_value": 0.2, "fdr": 0.2},
        ]
    )
    curated = pd.DataFrame(
        [
            {"program": "hallmark_A", "nes": 1.1, "p_value": 0.02, "fdr": 0.03},
        ]
    )

    fig = plot_enrichment_comparison(learned, curated)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)
