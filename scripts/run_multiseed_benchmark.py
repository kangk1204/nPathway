#!/usr/bin/env python3
"""Run multi-seed benchmark with statistical testing and summary figures.

This script reuses the single-run benchmark workflow in
`scripts/generate_benchmark_report.py`, repeating the full pipeline over
multiple random seeds, then aggregating metrics and computing pairwise
Wilcoxon tests.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import generate_benchmark_report as gbr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats as scipy_stats

from npathway.evaluation.metrics import (
    benjamini_hochberg_fdr_grouped,
    paired_rank_biserial_correlation,
)

logger = logging.getLogger(__name__)


def _ensure_dirs(base_dir: Path) -> tuple[Path, Path]:
    tables_dir = base_dir / "tables"
    figures_dir = base_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    return tables_dir, figures_dir


def _bootstrap_ci_mean(
    values: np.ndarray,
    n_bootstrap: int,
    alpha: float,
    seed: int,
) -> tuple[float, float]:
    if len(values) == 0 or n_bootstrap <= 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_bootstrap, len(values)))
    boot_means = values[idx].mean(axis=1)
    lo = float(np.quantile(boot_means, alpha / 2))
    hi = float(np.quantile(boot_means, 1 - alpha / 2))
    return lo, hi


def _paired_effect_size_dz(delta: np.ndarray) -> float:
    if len(delta) < 2:
        return np.nan
    sd = float(np.std(delta, ddof=1))
    if sd <= 0:
        return np.nan
    return float(np.mean(delta) / sd)


def _method_track(method: str) -> str:
    if method.startswith("Curated-") or method == "starCAT-Ref":
        return "reference_guided"
    return "de_novo"


def run_single_seed(
    seed: int,
    adata: Any,
    gene_embeddings: np.ndarray,
    graph_reg_embeddings: np.ndarray,
    gene_names: list[str],
    hallmark: dict[str, list[str]],
    n_power_trials: int,
) -> dict[str, Any]:
    """Run all benchmarks once at a given seed."""
    logger.info("Running benchmarks for seed=%d", seed)
    gbr.SEED = seed

    all_methods = gbr.discover_all_methods(
        gene_embeddings,
        graph_reg_embeddings,
        gene_names,
        adata,
        hallmark,
    )
    recovery_df = gbr.benchmark_pathway_recovery(all_methods, hallmark, gene_names, adata)
    discovery_df = gbr.benchmark_discovery(all_methods, hallmark, gene_names)
    power_df = gbr.benchmark_power(all_methods, gene_names, adata, n_trials=n_power_trials)
    cross_model_df = gbr.benchmark_cross_model(graph_reg_embeddings, gene_names)

    recovery_df = recovery_df.copy()
    discovery_df = discovery_df.copy()
    power_df = power_df.copy()

    recovery_df["seed"] = seed
    discovery_df["seed"] = seed
    if not power_df.empty:
        power_df["seed"] = seed

    cross_long = (
        cross_model_df.reset_index()
        .rename(columns={"index": "row_model"})
        .melt(id_vars=["row_model"], var_name="col_model", value_name="similarity")
    )
    cross_long["seed"] = seed

    return {
        "methods": all_methods,
        "recovery": recovery_df,
        "discovery": discovery_df,
        "power": power_df,
        "cross_model": cross_long,
    }


def summarize_metrics(
    recovery_all: pd.DataFrame,
    discovery_all: pd.DataFrame,
    power_all: pd.DataFrame,
) -> pd.DataFrame:
    """Compute per-seed per-method summary metrics and aggregate statistics."""
    rec_seed = (
        recovery_all
        .groupby(["seed", "method"], as_index=False)
        .agg(
            recovery_mean_n_sig=("n_sig", "mean"),
            recovery_mean_hallmark_jaccard=("best_hallmark_jaccard", "mean"),
            recovery_mean_best_fdr=("best_fdr", "mean"),
        )
    )

    disc_seed = discovery_all.rename(
        columns={
            "coverage": "discovery_coverage",
            "redundancy": "discovery_redundancy",
            "novelty": "discovery_novelty",
            "specificity": "discovery_specificity",
            "mean_hallmark_alignment": "discovery_mean_hallmark_alignment",
            "mean_program_size": "discovery_mean_program_size",
            "n_programs": "discovery_n_programs",
        }
    )

    merged = rec_seed.merge(disc_seed, on=["seed", "method"], how="outer")

    if not power_all.empty:
        pwr_seed = (
            power_all
            .groupby(["seed", "method"], as_index=False)
            .agg(
                power_mean_tpr=("tpr", "mean"),
                power_mean_fpr=("fpr", "mean"),
            )
        )
        fc2 = power_all[np.isclose(power_all["fold_change"], 2.0)].copy()
        if not fc2.empty:
            fc2 = fc2.rename(
                columns={"tpr": "power_tpr_fc2", "fpr": "power_fpr_fc2"}
            )[["seed", "method", "power_tpr_fc2", "power_fpr_fc2"]]
            pwr_seed = pwr_seed.merge(fc2, on=["seed", "method"], how="left")
        merged = merged.merge(pwr_seed, on=["seed", "method"], how="outer")

    aggregate = (
        merged
        .groupby("method", as_index=False)
        .agg({
            col: ["mean", "std"]
            for col in merged.columns
            if col not in {"seed", "method"}
        })
    )
    aggregate.columns = [
        "method" if c[0] == "method" else f"{c[0]}_{c[1]}"
        for c in aggregate.columns.to_flat_index()
    ]

    return merged, aggregate


def pairwise_stats(
    seed_summary: pd.DataFrame,
    reference_method: str = "cNMF",
    *,
    n_bootstrap: int = 2000,
    bootstrap_alpha: float = 0.05,
) -> pd.DataFrame:
    """Compute pairwise Wilcoxon tests versus a reference method."""
    metrics = [
        ("recovery_mean_n_sig", "greater"),
        ("recovery_mean_hallmark_jaccard", "greater"),
        ("discovery_mean_hallmark_alignment", "greater"),
        ("power_mean_tpr", "greater"),
        ("power_mean_fpr", "less"),
    ]

    methods = sorted(seed_summary["method"].dropna().unique().tolist())
    if reference_method not in methods:
        logger.warning(
            "Reference method '%s' not found; skipping stats.", reference_method
        )
        return pd.DataFrame(
            columns=[
                "method",
                "reference",
                "track",
                "metric",
                "alternative",
                "n_pairs",
                "n_non_ties",
                "method_mean",
                "reference_mean",
                "delta_mean",
                "delta_median",
                "delta_std",
                "effect_size_dz",
                "effect_size_rank_biserial",
                "effect_size_rank_biserial_directional",
                "delta_ci95_low",
                "delta_ci95_high",
                "win_rate",
                "delta_directional",
                "p_value",
                "sign_p_value",
                "fdr_bh",
                "fdr_bh_by_metric",
                "fdr_bh_by_track_metric",
                "sign_fdr_bh",
                "sign_fdr_bh_by_metric",
                "sign_fdr_bh_by_track_metric",
            ]
        )

    rows: list[dict[str, Any]] = []
    for method in methods:
        if method == reference_method:
            continue
        left = seed_summary[seed_summary["method"] == method]
        right = seed_summary[seed_summary["method"] == reference_method]
        pair = left.merge(right, on="seed", suffixes=("_m", "_r"))
        for metric, alternative in metrics:
            col_m = f"{metric}_m"
            col_r = f"{metric}_r"
            if col_m not in pair.columns or col_r not in pair.columns:
                continue
            vals = pair[[col_m, col_r]].dropna()
            n_pairs = len(vals)
            delta = (
                vals[col_m].values - vals[col_r].values
                if n_pairs > 0
                else np.array([])
            )
            if n_pairs < 2:
                p_value = np.nan
            else:
                try:
                    res = scipy_stats.wilcoxon(
                        vals[col_m].values,
                        vals[col_r].values,
                        alternative=alternative,
                        zero_method="wilcox",
                    )
                    p_value = float(res.pvalue)
                except ValueError:
                    p_value = np.nan

            method_mean = float(vals[col_m].mean()) if n_pairs > 0 else np.nan
            ref_mean = float(vals[col_r].mean()) if n_pairs > 0 else np.nan
            delta_mean = (
                method_mean - ref_mean
                if np.isfinite(method_mean) and np.isfinite(ref_mean)
                else np.nan
            )
            delta_median = float(np.median(delta)) if n_pairs > 0 else np.nan
            delta_std = float(np.std(delta, ddof=1)) if n_pairs >= 2 else np.nan
            dz = _paired_effect_size_dz(delta)
            rank_biserial = paired_rank_biserial_correlation(delta)
            ci_low, ci_high = _bootstrap_ci_mean(
                delta.astype(np.float64),
                n_bootstrap=n_bootstrap,
                alpha=bootstrap_alpha,
                seed=17 + n_pairs,
            )

            if n_pairs > 0:
                if alternative == "greater":
                    wins = int(np.sum(delta > 0))
                    losses = int(np.sum(delta < 0))
                else:
                    wins = int(np.sum(delta < 0))
                    losses = int(np.sum(delta > 0))
                n_non_ties = wins + losses
                win_rate = float(wins / n_non_ties) if n_non_ties > 0 else np.nan
                if n_non_ties >= 2:
                    sign_p_value = float(
                        scipy_stats.binomtest(
                            wins,
                            n_non_ties,
                            p=0.5,
                            alternative="greater",
                        ).pvalue
                    )
                else:
                    sign_p_value = np.nan
            else:
                n_non_ties = 0
                win_rate = np.nan
                sign_p_value = np.nan

            rows.append(
                {
                    "method": method,
                    "reference": reference_method,
                    "track": _method_track(method),
                    "metric": metric,
                    "alternative": alternative,
                    "n_pairs": n_pairs,
                    "n_non_ties": n_non_ties,
                    "method_mean": method_mean,
                    "reference_mean": ref_mean,
                    "delta_mean": delta_mean,
                    "delta_median": delta_median,
                    "delta_std": delta_std,
                    "effect_size_dz": dz,
                    "effect_size_rank_biserial": rank_biserial,
                    "effect_size_rank_biserial_directional": (
                        rank_biserial if alternative == "greater" else -rank_biserial
                    ),
                    "delta_ci95_low": ci_low,
                    "delta_ci95_high": ci_high,
                    "win_rate": win_rate,
                    "delta_directional": (
                        delta_mean if alternative == "greater" else -delta_mean
                    ),
                    "p_value": p_value,
                    "sign_p_value": sign_p_value,
                }
            )

    stats_df = pd.DataFrame(rows)
    if stats_df.empty:
        stats_df["fdr_bh"] = []
        stats_df["fdr_bh_by_metric"] = []
        stats_df["fdr_bh_by_track_metric"] = []
        stats_df["sign_fdr_bh"] = []
        stats_df["sign_fdr_bh_by_metric"] = []
        stats_df["sign_fdr_bh_by_track_metric"] = []
        return stats_df

    stats_df["fdr_bh"] = benjamini_hochberg_fdr_grouped(
        stats_df, p_value_col="p_value"
    ).to_numpy(dtype=np.float64)
    stats_df["fdr_bh_by_metric"] = benjamini_hochberg_fdr_grouped(
        stats_df, p_value_col="p_value", group_cols=["metric"]
    ).to_numpy(dtype=np.float64)
    stats_df["fdr_bh_by_track_metric"] = benjamini_hochberg_fdr_grouped(
        stats_df,
        p_value_col="p_value",
        group_cols=["track", "metric", "alternative"],
    ).to_numpy(dtype=np.float64)
    stats_df["sign_fdr_bh"] = benjamini_hochberg_fdr_grouped(
        stats_df, p_value_col="sign_p_value"
    ).to_numpy(dtype=np.float64)
    stats_df["sign_fdr_bh_by_metric"] = benjamini_hochberg_fdr_grouped(
        stats_df, p_value_col="sign_p_value", group_cols=["metric"]
    ).to_numpy(dtype=np.float64)
    stats_df["sign_fdr_bh_by_track_metric"] = benjamini_hochberg_fdr_grouped(
        stats_df,
        p_value_col="sign_p_value",
        group_cols=["track", "metric", "alternative"],
    ).to_numpy(dtype=np.float64)
    return stats_df


def make_figures(
    seed_summary: pd.DataFrame,
    power_all: pd.DataFrame,
    figures_dir: Path,
) -> None:
    """Render multi-seed summary figures."""
    sns.set_theme(style="whitegrid")

    # Figure 1: Recovery metric distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    order = (
        seed_summary
        .groupby("method")["recovery_mean_n_sig"]
        .mean()
        .sort_values(ascending=False)
        .index
        .tolist()
    )

    sns.boxplot(
        data=seed_summary,
        x="method",
        y="recovery_mean_n_sig",
        order=order,
        ax=axes[0],
        color="#90CAF9",
    )
    sns.stripplot(
        data=seed_summary,
        x="method",
        y="recovery_mean_n_sig",
        order=order,
        ax=axes[0],
        color="#0D47A1",
        size=4,
        alpha=0.8,
    )
    axes[0].set_title("Recovery: Mean Significant Programs")
    axes[0].set_xlabel("")
    axes[0].tick_params(axis="x", rotation=35)

    sns.boxplot(
        data=seed_summary,
        x="method",
        y="discovery_mean_hallmark_alignment",
        order=order,
        ax=axes[1],
        color="#A5D6A7",
    )
    sns.stripplot(
        data=seed_summary,
        x="method",
        y="discovery_mean_hallmark_alignment",
        order=order,
        ax=axes[1],
        color="#1B5E20",
        size=4,
        alpha=0.8,
    )
    axes[1].set_title("Discovery: Mean Hallmark Alignment")
    axes[1].set_xlabel("")
    axes[1].tick_params(axis="x", rotation=35)

    fig.tight_layout()
    fig.savefig(figures_dir / "multiseed_boxplots.png", dpi=180)
    plt.close(fig)

    # Figure 2: Power curves (mean +/- std across seeds)
    if not power_all.empty:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        grouped = (
            power_all
            .groupby(["method", "fold_change"], as_index=False)
            .agg(
                tpr_mean=("tpr", "mean"),
                tpr_std=("tpr", "std"),
                fpr_mean=("fpr", "mean"),
                fpr_std=("fpr", "std"),
            )
        )

        method_order = (
            grouped[grouped["fold_change"] == grouped["fold_change"].max()]
            .sort_values("tpr_mean", ascending=False)["method"]
            .tolist()
        )
        for method in method_order:
            sub = grouped[grouped["method"] == method].sort_values("fold_change")
            x = sub["fold_change"].values
            tpr_mean = sub["tpr_mean"].values
            tpr_std = np.nan_to_num(sub["tpr_std"].values, nan=0.0)
            fpr_mean = sub["fpr_mean"].values
            fpr_std = np.nan_to_num(sub["fpr_std"].values, nan=0.0)

            axes[0].plot(x, tpr_mean, marker="o", linewidth=2, label=method)
            axes[0].fill_between(x, np.clip(tpr_mean - tpr_std, 0, 1), np.clip(tpr_mean + tpr_std, 0, 1), alpha=0.15)

            axes[1].plot(x, fpr_mean, marker="o", linewidth=2, label=method)
            axes[1].fill_between(x, np.clip(fpr_mean - fpr_std, 0, 1), np.clip(fpr_mean + fpr_std, 0, 1), alpha=0.15)

        axes[0].set_title("Power Curve: TPR (mean +/- std)")
        axes[0].set_xlabel("Fold Change")
        axes[0].set_ylabel("TPR")
        axes[0].set_ylim(0, 1.02)
        axes[0].legend(fontsize=8)

        axes[1].set_title("Power Curve: FPR (mean +/- std)")
        axes[1].set_xlabel("Fold Change")
        axes[1].set_ylabel("FPR")
        axes[1].set_ylim(0, 0.6)
        axes[1].legend(fontsize=8)

        fig.tight_layout()
        fig.savefig(figures_dir / "multiseed_power_curves.png", dpi=180)
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 43, 44, 45, 46],
        help="Random seeds for repeated full benchmark runs.",
    )
    parser.add_argument(
        "--power-trials",
        type=int,
        default=15,
        help="Number of trials per fold-change in power benchmark.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("results") / "multiseed",
        help="Output directory for multi-seed tables/figures.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    t0 = time.time()
    logger.info("Multi-seed benchmark started")
    logger.info("Seeds: %s", args.seeds)
    logger.info("Power trials per fold-change: %d", args.power_trials)

    tables_dir, figures_dir = _ensure_dirs(args.outdir)

    adata, gene_embeddings, graph_reg_embeddings, gene_names, hallmark = gbr.load_real_data()
    logger.info(
        "Loaded data once: cells=%d genes=%d hallmark_sets=%d",
        adata.n_obs,  # type: ignore[union-attr]
        len(gene_names),
        len(hallmark),
    )

    recovery_all: list[pd.DataFrame] = []
    discovery_all: list[pd.DataFrame] = []
    power_all: list[pd.DataFrame] = []
    cross_all: list[pd.DataFrame] = []

    for seed in args.seeds:
        run = run_single_seed(
            seed=seed,
            adata=adata,
            gene_embeddings=gene_embeddings,
            graph_reg_embeddings=graph_reg_embeddings,
            gene_names=gene_names,
            hallmark=hallmark,
            n_power_trials=args.power_trials,
        )
        recovery_all.append(run["recovery"])
        discovery_all.append(run["discovery"])
        cross_all.append(run["cross_model"])
        if not run["power"].empty:
            power_all.append(run["power"])

    recovery_df = pd.concat(recovery_all, ignore_index=True)
    discovery_df = pd.concat(discovery_all, ignore_index=True)
    power_df = (
        pd.concat(power_all, ignore_index=True)
        if power_all else
        pd.DataFrame(columns=["seed", "method", "fold_change", "tpr", "fpr"])
    )
    cross_df = pd.concat(cross_all, ignore_index=True)

    seed_summary, aggregate_summary = summarize_metrics(
        recovery_df, discovery_df, power_df
    )
    stats_df = pairwise_stats(seed_summary, reference_method="cNMF")

    recovery_df.to_csv(tables_dir / "multiseed_benchmark1_recovery.csv", index=False)
    discovery_df.to_csv(tables_dir / "multiseed_benchmark2_discovery.csv", index=False)
    power_df.to_csv(tables_dir / "multiseed_benchmark3_power.csv", index=False)
    cross_df.to_csv(tables_dir / "multiseed_benchmark4_cross_model.csv", index=False)
    seed_summary.to_csv(tables_dir / "multiseed_seed_summary.csv", index=False)
    aggregate_summary.to_csv(tables_dir / "multiseed_method_summary.csv", index=False)
    stats_df.to_csv(tables_dir / "multiseed_wilcoxon_vs_cnmf.csv", index=False)

    make_figures(seed_summary, power_df, figures_dir)

    elapsed = time.time() - t0
    logger.info("Multi-seed benchmark finished in %.1f sec", elapsed)
    logger.info("Outputs written to %s", args.outdir)
    logger.info("Key tables:")
    logger.info("  %s", tables_dir / "multiseed_method_summary.csv")
    logger.info("  %s", tables_dir / "multiseed_wilcoxon_vs_cnmf.csv")


if __name__ == "__main__":
    main()
