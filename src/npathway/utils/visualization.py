"""
Visualization module for nPathway gene program analysis and benchmarking.

Provides publication-quality plotting functions for gene embeddings, program
overlap heatmaps, enrichment comparisons, benchmark summaries, and
context-specificity analysis. All functions return ``matplotlib.figure.Figure``
objects and optionally save to disk.

Dependencies: matplotlib, seaborn, numpy, pandas, umap-learn (for UMAP plots).
"""

from __future__ import annotations

import logging
from typing import Any

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Style configuration
# ---------------------------------------------------------------------------

_STYLE_PARAMS: dict[str, Any] = {
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
}


def _apply_style() -> None:
    """Apply the nPathway default matplotlib style."""
    plt.rcParams.update(_STYLE_PARAMS)


def _save_if_requested(fig: plt.Figure, save_path: str | None) -> None:
    """Save figure to *save_path* if the path is not ``None``."""
    if save_path is not None:
        from pathlib import Path

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)
        logger.info("Figure saved to %s", save_path)


# ---------------------------------------------------------------------------
# 1. UMAP of gene embeddings
# ---------------------------------------------------------------------------


def plot_embedding_umap(
    embeddings: np.ndarray,
    labels: np.ndarray | None = None,
    gene_names: list[str] | None = None,
    save_path: str | None = None,
    *,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    point_size: float = 5.0,
    alpha: float = 0.7,
    figsize: tuple[float, float] = (8, 6),
    random_state: int = 42,
) -> plt.Figure:
    """Create a UMAP scatter plot of gene embeddings colored by program label.

    Parameters
    ----------
    embeddings : np.ndarray
        Gene embedding matrix of shape ``(n_genes, embedding_dim)``.
    labels : np.ndarray | None, optional
        Integer cluster/program labels of shape ``(n_genes,)``.
        If ``None``, all points are plotted in a single color.
    gene_names : list[str] | None, optional
        Gene symbols; used for hover annotation metadata (stored in the
        figure for downstream interactive use) but not rendered as text
        to avoid clutter.
    save_path : str | None, optional
        Path to save the figure. If ``None``, the figure is not saved.
    n_neighbors : int
        UMAP ``n_neighbors`` parameter.
    min_dist : float
        UMAP ``min_dist`` parameter.
    point_size : float
        Marker size for scatter points.
    alpha : float
        Marker opacity.
    figsize : tuple[float, float]
        Figure size in inches ``(width, height)``.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    matplotlib.figure.Figure
        The UMAP scatter plot figure.

    Raises
    ------
    ValueError
        If *embeddings* is not 2-D or if *labels* length does not match.
    """
    try:
        import umap
    except ImportError as exc:
        raise ImportError(
            "umap-learn is required for plot_embedding_umap. "
            "Install it with: pip install umap-learn"
        ) from exc

    if embeddings.ndim != 2:
        raise ValueError(
            f"embeddings must be a 2-D array, got shape {embeddings.shape}."
        )
    if labels is not None and len(labels) != embeddings.shape[0]:
        raise ValueError(
            f"Length of labels ({len(labels)}) does not match number of "
            f"embeddings ({embeddings.shape[0]})."
        )

    _apply_style()

    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
        n_components=2,
    )
    coords = reducer.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=figsize)

    if labels is not None:
        unique_labels = np.unique(labels)
        n_labels = len(unique_labels)
        palette = sns.color_palette("husl", n_labels)
        color_map = {lab: palette[i] for i, lab in enumerate(unique_labels)}
        colors = [color_map[lab] for lab in labels]

        ax.scatter(
            coords[:, 0],
            coords[:, 1],
            c=colors,
            s=point_size,
            alpha=alpha,
            linewidths=0,
            rasterized=True,
        )

        # Create legend with up to 30 entries to avoid clutter
        if n_labels <= 30:
            handles = [
                mpatches.Patch(color=color_map[lab], label=f"Program {lab}")
                for lab in unique_labels
            ]
            ax.legend(
                handles=handles,
                loc="center left",
                bbox_to_anchor=(1.02, 0.5),
                frameon=False,
                ncol=max(1, n_labels // 15),
            )
    else:
        ax.scatter(
            coords[:, 0],
            coords[:, 1],
            s=point_size,
            alpha=alpha,
            color="steelblue",
            linewidths=0,
            rasterized=True,
        )

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title("Gene Embedding UMAP")
    ax.set_xticks([])
    ax.set_yticks([])
    sns.despine(ax=ax, left=True, bottom=True)

    # Attach gene_names metadata to figure for downstream interactive use
    if gene_names is not None:
        fig._npathway_gene_names = gene_names  # type: ignore[attr-defined]
        fig._npathway_umap_coords = coords  # type: ignore[attr-defined]

    fig.tight_layout()
    _save_if_requested(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# 2. Program overlap heatmap
# ---------------------------------------------------------------------------


def _jaccard_index(set_a: set[str], set_b: set[str]) -> float:
    """Compute Jaccard similarity index between two sets."""
    if not set_a and not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union


def plot_program_overlap_heatmap(
    gene_programs: dict[str, list[str]],
    curated_pathways: dict[str, list[str]],
    save_path: str | None = None,
    top_n: int = 50,
    *,
    figsize: tuple[float, float] | None = None,
    cmap: str = "YlOrRd",
    annot_threshold: float = 0.1,
) -> plt.Figure:
    """Plot a Jaccard overlap heatmap between learned programs and curated pathways.

    The heatmap shows the Jaccard similarity index between each pair of
    learned gene programs (rows) and curated pathways (columns). Programs
    and pathways are sorted by maximum overlap for readability.

    Parameters
    ----------
    gene_programs : dict[str, list[str]]
        Learned gene programs.
    curated_pathways : dict[str, list[str]]
        Reference curated pathways (e.g., KEGG, Reactome).
    save_path : str | None, optional
        Path to save the figure.
    top_n : int
        Maximum number of programs and pathways to display. The top
        entries by maximum Jaccard overlap are selected.
    figsize : tuple[float, float] | None, optional
        Figure size. If ``None``, auto-scaled based on matrix dimensions.
    cmap : str
        Matplotlib colormap name.
    annot_threshold : float
        Jaccard values below this threshold are not annotated in cells.

    Returns
    -------
    matplotlib.figure.Figure
        The heatmap figure.
    """
    _apply_style()

    if not gene_programs or not curated_pathways:
        logger.warning(
            "Cannot plot overlap heatmap with empty input "
            "(gene_programs=%d, curated_pathways=%d).",
            len(gene_programs),
            len(curated_pathways),
        )
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(
            0.5, 0.5,
            "Need non-empty learned programs and curated pathways.",
            transform=ax.transAxes, ha="center", va="center",
        )
        ax.set_axis_off()
        _save_if_requested(fig, save_path)
        return fig

    prog_names = list(gene_programs.keys())
    pw_names = list(curated_pathways.keys())

    prog_sets = {k: set(v) for k, v in gene_programs.items()}
    pw_sets = {k: set(v) for k, v in curated_pathways.items()}

    # Compute full Jaccard matrix
    jaccard_matrix = np.zeros((len(prog_names), len(pw_names)))
    for i, pn in enumerate(prog_names):
        for j, cn in enumerate(pw_names):
            jaccard_matrix[i, j] = _jaccard_index(prog_sets[pn], pw_sets[cn])

    # Select top_n programs and pathways by their maximum overlap
    prog_max = jaccard_matrix.max(axis=1)
    pw_max = jaccard_matrix.max(axis=0)

    top_prog_idx = np.argsort(prog_max)[::-1][:top_n]
    top_pw_idx = np.argsort(pw_max)[::-1][:top_n]

    sub_matrix = jaccard_matrix[np.ix_(top_prog_idx, top_pw_idx)]
    sub_prog_names = [prog_names[i] for i in top_prog_idx]
    sub_pw_names = [pw_names[j] for j in top_pw_idx]

    # Sort rows and columns by hierarchical clustering order
    if sub_matrix.shape[0] > 1 and sub_matrix.shape[1] > 1:
        from scipy.cluster.hierarchy import leaves_list, linkage
        from scipy.spatial.distance import pdist

        if sub_matrix.shape[0] > 2:
            row_order = leaves_list(linkage(pdist(sub_matrix, metric="euclidean"), method="average"))
        else:
            row_order = list(range(sub_matrix.shape[0]))
        if sub_matrix.shape[1] > 2:
            col_order = leaves_list(linkage(pdist(sub_matrix.T, metric="euclidean"), method="average"))
        else:
            col_order = list(range(sub_matrix.shape[1]))

        sub_matrix = sub_matrix[np.ix_(row_order, col_order)]
        sub_prog_names = [sub_prog_names[i] for i in row_order]
        sub_pw_names = [sub_pw_names[j] for j in col_order]

    df_heatmap = pd.DataFrame(sub_matrix, index=sub_prog_names, columns=sub_pw_names)

    n_rows, n_cols = df_heatmap.shape
    if figsize is None:
        figsize = (max(6, n_cols * 0.4 + 3), max(4, n_rows * 0.35 + 2))

    fig, ax = plt.subplots(figsize=figsize)

    # Annotation: only show values above threshold
    annot_data = df_heatmap.copy()
    annot_strings = annot_data.map(
        lambda v: f"{v:.2f}" if v >= annot_threshold else ""
    )

    sns.heatmap(
        df_heatmap,
        ax=ax,
        cmap=cmap,
        vmin=0,
        vmax=max(0.5, float(df_heatmap.values.max())),
        annot=annot_strings,
        fmt="",
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Jaccard Index", "shrink": 0.6},
    )

    ax.set_xlabel("Curated Pathways")
    ax.set_ylabel("Learned Programs")
    ax.set_title("Gene Program -- Curated Pathway Overlap")

    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    fig.tight_layout()
    _save_if_requested(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# 3. Enrichment comparison bar chart
# ---------------------------------------------------------------------------


def plot_enrichment_comparison(
    results_learned: pd.DataFrame,
    results_curated: pd.DataFrame,
    save_path: str | None = None,
    *,
    metric_col: str = "nes",
    pval_col: str = "fdr",
    name_col: str = "program",
    pval_threshold: float = 0.05,
    top_n: int = 20,
    figsize: tuple[float, float] = (10, 6),
) -> plt.Figure:
    """Create a grouped bar chart comparing enrichment results.

    Parameters
    ----------
    results_learned : pd.DataFrame
        Enrichment results for learned gene programs. Must contain columns
        for the metric, p-value, and pathway/program name.
    results_curated : pd.DataFrame
        Enrichment results for curated pathways, same column schema.
    save_path : str | None, optional
        Path to save the figure.
    metric_col : str
        Column name for the enrichment metric (e.g., NES, enrichment score).
    pval_col : str
        Column name for adjusted p-value.
    name_col : str
        Column name for pathway/program name.
    pval_threshold : float
        Significance threshold.
    top_n : int
        Number of top pathways/programs to display per source.
    figsize : tuple[float, float]
        Figure size in inches.

    Returns
    -------
    matplotlib.figure.Figure
        The comparison bar chart figure.
    """
    _apply_style()

    def _resolve_column(
        df: pd.DataFrame,
        preferred: str,
        aliases: tuple[str, ...],
    ) -> str:
        if preferred in df.columns:
            return preferred
        for alias in aliases:
            if alias in df.columns:
                return alias
        raise KeyError(
            f"Required enrichment column '{preferred}' not found. "
            f"Available columns: {list(df.columns)}"
        )

    # Filter significant results and take top_n by absolute metric
    def _filter_top(df: pd.DataFrame, label: str) -> pd.DataFrame:
        metric_name = _resolve_column(
            df, metric_col, ("nes", "NES", "es", "enrichment_score")
        )
        pval_name = _resolve_column(
            df, pval_col, ("fdr", "padj", "p_value", "q_value")
        )
        label_name = _resolve_column(
            df, name_col, ("program", "pathway", "gene_set")
        )
        sig = df[df[pval_name] <= pval_threshold].copy()
        sig["abs_metric"] = sig[metric_name].abs()
        sig["_metric"] = sig[metric_name]
        sig["_name"] = sig[label_name].astype(str)
        sig = sig.nlargest(top_n, "abs_metric")
        sig["source"] = label
        return sig

    top_learned = _filter_top(results_learned, "Learned Programs")
    top_curated = _filter_top(results_curated, "Curated Pathways")

    combined = pd.concat([top_learned, top_curated], ignore_index=True)

    if combined.empty:
        logger.warning("No significant results to plot.")
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(
            0.5, 0.5, "No significant enrichment results",
            transform=ax.transAxes, ha="center", va="center", fontsize=14,
        )
        _save_if_requested(fig, save_path)
        return fig

    # Truncate long pathway names
    combined["display_name"] = combined["_name"].str[:40]

    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=False)

    # Left panel: Learned programs
    ax_l = axes[0]
    learned_data = combined[combined["source"] == "Learned Programs"].sort_values(
        "_metric", ascending=True
    )
    if not learned_data.empty:
        colors_l = [
            "#2196F3" if v >= 0 else "#FF5722" for v in learned_data["_metric"]
        ]
        ax_l.barh(
            learned_data["display_name"],
            learned_data["_metric"],
            color=colors_l,
            edgecolor="white",
            linewidth=0.5,
        )
    ax_l.set_xlabel(metric_col)
    ax_l.set_title("Learned Programs")
    ax_l.axvline(0, color="black", linewidth=0.5)
    sns.despine(ax=ax_l)

    # Right panel: Curated pathways
    ax_r = axes[1]
    curated_data = combined[combined["source"] == "Curated Pathways"].sort_values(
        "_metric", ascending=True
    )
    if not curated_data.empty:
        colors_r = [
            "#4CAF50" if v >= 0 else "#FF9800" for v in curated_data["_metric"]
        ]
        ax_r.barh(
            curated_data["display_name"],
            curated_data["_metric"],
            color=colors_r,
            edgecolor="white",
            linewidth=0.5,
        )
    ax_r.set_xlabel(metric_col)
    ax_r.set_title("Curated Pathways")
    ax_r.axvline(0, color="black", linewidth=0.5)
    sns.despine(ax=ax_r)

    fig.suptitle("Enrichment Comparison", fontsize=13, y=1.02)
    fig.tight_layout()
    _save_if_requested(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# 4. Program size distribution
# ---------------------------------------------------------------------------


def plot_program_sizes(
    gene_programs: dict[str, list[str]],
    save_path: str | None = None,
    *,
    figsize: tuple[float, float] = (8, 5),
    bins: int | str = "auto",
    color: str = "#3F51B5",
) -> plt.Figure:
    """Plot the distribution of gene program sizes (number of genes per program).

    Parameters
    ----------
    gene_programs : dict[str, list[str]]
        Mapping from program name to list of gene symbols.
    save_path : str | None, optional
        Path to save the figure.
    figsize : tuple[float, float]
        Figure size in inches.
    bins : int | str
        Number of histogram bins or ``"auto"`` for automatic.
    color : str
        Histogram bar color.

    Returns
    -------
    matplotlib.figure.Figure
        The histogram figure.
    """
    _apply_style()

    if not gene_programs:
        logger.warning("Cannot plot program sizes: gene_programs is empty.")
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(
            0.5, 0.5, "No gene programs to plot.",
            transform=ax.transAxes, ha="center", va="center",
        )
        ax.set_axis_off()
        _save_if_requested(fig, save_path)
        return fig

    sizes = [len(genes) for genes in gene_programs.values()]

    fig, axes = plt.subplots(1, 2, figsize=figsize, gridspec_kw={"width_ratios": [2, 1]})

    # Histogram
    ax_hist = axes[0]
    ax_hist.hist(sizes, bins=bins, color=color, edgecolor="white", linewidth=0.5, alpha=0.85)
    ax_hist.set_xlabel("Program Size (number of genes)")
    ax_hist.set_ylabel("Frequency")
    ax_hist.set_title("Distribution of Gene Program Sizes")
    ax_hist.axvline(
        np.median(sizes), color="#F44336", linestyle="--", linewidth=1.5,
        label=f"Median = {np.median(sizes):.0f}",
    )
    ax_hist.axvline(
        np.mean(sizes), color="#FF9800", linestyle=":", linewidth=1.5,
        label=f"Mean = {np.mean(sizes):.1f}",
    )
    ax_hist.legend(frameon=False)
    sns.despine(ax=ax_hist)

    # Box plot
    ax_box = axes[1]
    ax_box.boxplot(
        sizes,
        vert=True,
        widths=0.5,
        patch_artist=True,
        boxprops={"facecolor": color, "alpha": 0.6},
        medianprops={"color": "#F44336", "linewidth": 2},
        whiskerprops={"linewidth": 1.2},
        flierprops={"marker": "o", "markersize": 3, "alpha": 0.5},
    )
    ax_box.set_ylabel("Program Size")
    ax_box.set_title("Summary")
    ax_box.set_xticks([])
    sns.despine(ax=ax_box)

    fig.tight_layout()
    _save_if_requested(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# 5. Benchmark summary (multi-panel)
# ---------------------------------------------------------------------------


def plot_benchmark_summary(
    metrics: dict[str, dict[str, float]],
    save_path: str | None = None,
    *,
    figsize: tuple[float, float] = (12, 5),
) -> plt.Figure:
    """Create a multi-panel benchmark comparison figure.

    Renders three sub-panels:
    1. Grouped bar chart of all metrics per method.
    2. Radar / polar chart summarizing each method.
    3. Rank table (heatmap) showing rank of each method per metric.

    Parameters
    ----------
    metrics : dict[str, dict[str, float]]
        Nested dict: outer keys are method names, inner keys are metric
        names, values are metric scores.  Example::

            {
                "nPathway": {"Precision": 0.82, "Recall": 0.75, "F1": 0.78},
                "KEGG": {"Precision": 0.71, "Recall": 0.68, "F1": 0.69},
            }

    save_path : str | None, optional
        Path to save the figure.
    figsize : tuple[float, float]
        Figure size.

    Returns
    -------
    matplotlib.figure.Figure
        The multi-panel benchmark figure.
    """
    _apply_style()

    if not metrics:
        logger.warning("Cannot plot benchmark summary: metrics is empty.")
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(
            0.5, 0.5, "No benchmark metrics to plot.",
            transform=ax.transAxes, ha="center", va="center",
        )
        ax.set_axis_off()
        _save_if_requested(fig, save_path)
        return fig

    df = pd.DataFrame(metrics).T  # methods x metrics
    methods = list(df.index)
    metric_names = list(df.columns)
    n_methods = len(methods)
    n_metrics = len(metric_names)

    palette = sns.color_palette("Set2", n_methods)
    method_colors = {m: palette[i] for i, m in enumerate(methods)}

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(1, 3, width_ratios=[2, 1.5, 1.2], wspace=0.35)

    # Panel 1: Grouped bar chart
    ax_bar = fig.add_subplot(gs[0])
    x = np.arange(n_metrics)
    bar_width = 0.8 / max(n_methods, 1)
    for i, method in enumerate(methods):
        offsets = x + (i - n_methods / 2 + 0.5) * bar_width
        ax_bar.bar(
            offsets,
            df.loc[method].values,
            width=bar_width,
            label=method,
            color=method_colors[method],
            edgecolor="white",
            linewidth=0.5,
        )
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(metric_names, rotation=30, ha="right")
    ax_bar.set_ylabel("Score")
    ax_bar.set_title("Metric Comparison")
    ax_bar.legend(frameon=False, fontsize=8)
    ax_bar.set_ylim(0, min(1.15, float(df.values.max()) * 1.15))
    sns.despine(ax=ax_bar)

    # Panel 2: Radar chart
    ax_radar = fig.add_subplot(gs[1], polar=True)
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    for method in methods:
        values = df.loc[method].values.tolist()
        values += values[:1]
        ax_radar.plot(angles, values, "o-", linewidth=1.5, label=method, color=method_colors[method])
        ax_radar.fill(angles, values, alpha=0.1, color=method_colors[method])

    ax_radar.set_xticks(angles[:-1])
    ax_radar.set_xticklabels(metric_names, fontsize=8)
    ax_radar.set_title("Radar Overview", pad=15)
    ax_radar.set_ylim(0, min(1.0, float(df.values.max()) * 1.1))

    # Panel 3: Rank heatmap
    ax_rank = fig.add_subplot(gs[2])
    rank_df = df.rank(ascending=False, method="min").astype(int)
    sns.heatmap(
        rank_df,
        ax=ax_rank,
        annot=True,
        fmt="d",
        cmap="YlGn_r",
        linewidths=0.5,
        linecolor="white",
        cbar=False,
        vmin=1,
        vmax=n_methods,
    )
    ax_rank.set_title("Rank (1 = best)")
    ax_rank.set_xlabel("")
    ax_rank.set_ylabel("")
    plt.setp(ax_rank.get_xticklabels(), rotation=30, ha="right")

    fig.suptitle("Benchmark Summary", fontsize=14, y=1.03)
    try:
        fig.tight_layout()
    except ValueError:
        # Polar axes are not fully compatible with tight_layout
        fig.subplots_adjust(wspace=0.35)
    _save_if_requested(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# 6. Cross-model consistency heatmap
# ---------------------------------------------------------------------------


def plot_cross_model_consistency(
    similarity_matrix: pd.DataFrame,
    save_path: str | None = None,
    *,
    figsize: tuple[float, float] | None = None,
    cmap: str = "coolwarm",
    vmin: float = 0.0,
    vmax: float = 1.0,
) -> plt.Figure:
    """Plot a heatmap of cross-model consistency (e.g., Jaccard, ARI, NMI).

    Parameters
    ----------
    similarity_matrix : pd.DataFrame
        Square DataFrame where both index and columns are model/program
        identifiers and values are pairwise similarity scores.
    save_path : str | None, optional
        Path to save the figure.
    figsize : tuple[float, float] | None, optional
        Figure size. Auto-scaled if ``None``.
    cmap : str
        Colormap name.
    vmin : float
        Minimum value for the color scale.
    vmax : float
        Maximum value for the color scale.

    Returns
    -------
    matplotlib.figure.Figure
        The cross-model consistency heatmap figure.
    """
    _apply_style()

    n = similarity_matrix.shape[0]
    if figsize is None:
        side = max(5, n * 0.5 + 2)
        figsize = (side, side)

    fig, ax = plt.subplots(figsize=figsize)

    # Create mask for upper triangle (optional aesthetic choice)
    mask = np.triu(np.ones_like(similarity_matrix, dtype=bool), k=1)

    sns.heatmap(
        similarity_matrix,
        ax=ax,
        mask=mask,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        linecolor="white",
        square=True,
        cbar_kws={"label": "Similarity", "shrink": 0.7},
    )

    ax.set_title("Cross-Model Consistency")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    fig.tight_layout()
    _save_if_requested(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# 7. Context-specificity visualization
# ---------------------------------------------------------------------------


def _compute_alluvial_flows(
    context_programs: dict[str, dict[str, list[str]]],
) -> tuple[list[str], list[str], list[str], list[int]]:
    """Compute flow data for an alluvial / Sankey-style diagram.

    Returns lists of source context-program labels, target gene labels,
    context names for coloring, and flow widths (gene counts).
    """
    sources: list[str] = []
    targets: list[str] = []
    contexts: list[str] = []
    widths: list[int] = []

    for context_name, programs in context_programs.items():
        for prog_name, genes in programs.items():
            sources.append(f"{context_name}\n{prog_name}")
            targets.append(prog_name)
            contexts.append(context_name)
            widths.append(len(genes))

    return sources, targets, contexts, widths


def plot_context_specificity(
    context_programs: dict[str, dict[str, list[str]]],
    save_path: str | None = None,
    *,
    figsize: tuple[float, float] = (14, 8),
    top_programs: int = 15,
    top_genes: int = 30,
) -> plt.Figure:
    """Plot gene membership changes across biological contexts.

    Creates an alluvial / Sankey-style visualization showing how genes
    move between programs in different contexts (cell types, tissues,
    disease states). Implemented as a multi-panel layout with:

    1. Left panel: Stacked bar chart of gene program sizes per context.
    2. Center panel: Connectivity ribbons showing shared genes between
       programs across contexts (approximated with bezier curves).
    3. Right panel: Heatmap of gene-program membership changes.

    Parameters
    ----------
    context_programs : dict[str, dict[str, list[str]]]
        Nested dict: ``{context_name: {program_name: [gene_list]}}``.
        Example::

            {
                "T_cells": {"Prog_1": ["CD3D", "CD3E"], "Prog_2": ["IL2", "IFNG"]},
                "B_cells": {"Prog_1": ["CD19", "MS4A1"], "Prog_3": ["CD3D", "PAX5"]},
            }
    save_path : str | None, optional
        Path to save the figure.
    figsize : tuple[float, float]
        Figure size in inches.
    top_programs : int
        Maximum number of programs to display per context.
    top_genes : int
        Maximum number of genes to display in the membership heatmap.

    Returns
    -------
    matplotlib.figure.Figure
        The context-specificity visualization figure.
    """
    _apply_style()

    context_names = list(context_programs.keys())
    n_contexts = len(context_names)

    if n_contexts < 2:
        logger.warning(
            "Context-specificity plot requires at least 2 contexts; got %d.",
            n_contexts,
        )
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(
            0.5, 0.5,
            "At least 2 contexts required for context-specificity plot.",
            transform=ax.transAxes, ha="center", va="center", fontsize=14,
        )
        _save_if_requested(fig, save_path)
        return fig

    # --- Collect all genes and programs across contexts ---
    all_genes: set[str] = set()
    for programs in context_programs.values():
        for genes in programs.values():
            all_genes.update(genes)

    # Find genes that appear in multiple contexts for highlighting
    gene_context_count: dict[str, int] = {}
    for context_name, programs in context_programs.items():
        context_genes: set[str] = set()
        for genes in programs.values():
            context_genes.update(genes)
        for g in context_genes:
            gene_context_count[g] = gene_context_count.get(g, 0) + 1

    shared_genes = sorted(
        [g for g, c in gene_context_count.items() if c >= 2],
        key=lambda g: -gene_context_count[g],
    )
    display_genes = shared_genes[:top_genes] if shared_genes else sorted(all_genes)[:top_genes]

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 2], wspace=0.3)

    # --- Panel 1: Stacked bar chart of program sizes per context ---
    ax_bar = fig.add_subplot(gs[0])

    # Collect all program names across contexts
    all_program_names: set[str] = set()
    for programs in context_programs.values():
        all_program_names.update(programs.keys())
    sorted_program_names = sorted(all_program_names)[:top_programs]

    palette = sns.color_palette("husl", len(sorted_program_names))
    prog_colors = {p: palette[i] for i, p in enumerate(sorted_program_names)}

    x_positions = np.arange(n_contexts)
    bottoms = np.zeros(n_contexts)

    for prog_name in sorted_program_names:
        heights = []
        for ctx in context_names:
            if prog_name in context_programs[ctx]:
                heights.append(len(context_programs[ctx][prog_name]))
            else:
                heights.append(0)
        heights_arr = np.array(heights, dtype=float)
        ax_bar.bar(
            x_positions,
            heights_arr,
            bottom=bottoms,
            label=prog_name,
            color=prog_colors[prog_name],
            edgecolor="white",
            linewidth=0.5,
            width=0.6,
        )
        bottoms += heights_arr

    ax_bar.set_xticks(x_positions)
    ax_bar.set_xticklabels(context_names, rotation=30, ha="right")
    ax_bar.set_ylabel("Number of Genes")
    ax_bar.set_title("Program Sizes by Context")
    if len(sorted_program_names) <= 15:
        ax_bar.legend(
            fontsize=7, frameon=False, loc="upper left",
            bbox_to_anchor=(0, 1),
        )
    sns.despine(ax=ax_bar)

    # --- Panel 2: Gene membership heatmap across contexts ---
    ax_heat = fig.add_subplot(gs[1])

    # Build a matrix: rows = genes, columns = (context, program) pairs
    col_labels: list[str] = []
    membership_data: list[list[int]] = []

    for ctx in context_names:
        progs = context_programs[ctx]
        for prog_name in sorted(progs.keys())[:top_programs]:
            col_labels.append(f"{ctx} | {prog_name}")

    for gene in display_genes:
        row: list[int] = []
        for ctx in context_names:
            progs = context_programs[ctx]
            for prog_name in sorted(progs.keys())[:top_programs]:
                row.append(1 if gene in progs[prog_name] else 0)
        membership_data.append(row)

    if membership_data and col_labels:
        heat_df = pd.DataFrame(
            membership_data,
            index=display_genes,
            columns=col_labels,
        )

        # Create a custom colormap: white for 0, colored for 1
        cmap_binary = mcolors.ListedColormap(["#F5F5F5", "#1976D2"])
        bounds = [-0.5, 0.5, 1.5]
        norm = mcolors.BoundaryNorm(bounds, cmap_binary.N)

        sns.heatmap(
            heat_df,
            ax=ax_heat,
            cmap=cmap_binary,
            norm=norm,
            linewidths=0.3,
            linecolor="white",
            cbar_kws={"label": "Membership", "shrink": 0.3, "ticks": [0, 1]},
            yticklabels=True,
            xticklabels=True,
        )

        ax_heat.set_title("Gene Membership Across Contexts")
        plt.setp(ax_heat.get_xticklabels(), rotation=45, ha="right", fontsize=7)
        plt.setp(ax_heat.get_yticklabels(), fontsize=7)
    else:
        ax_heat.text(
            0.5, 0.5, "No shared genes to display",
            transform=ax_heat.transAxes, ha="center", va="center",
        )

    fig.suptitle("Context-Specific Gene Program Membership", fontsize=14, y=1.02)
    try:
        fig.tight_layout()
    except ValueError:
        fig.subplots_adjust(wspace=0.3)
    _save_if_requested(fig, save_path)
    return fig
