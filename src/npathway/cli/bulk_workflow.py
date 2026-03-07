"""Batch-aware bulk workflow with same-ranked-list curated-vs-dynamic GSEA comparison."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA

from npathway.pipeline import BulkDynamicConfig, run_bulk_dynamic_pipeline, validate_bulk_input_files
from npathway.pipeline.gsea_comparison import compare_curated_vs_dynamic_gsea
from npathway.reporting import DashboardConfig, build_dynamic_dashboard_package

logger = logging.getLogger(__name__)

_AUTO = "__AUTO__"
_NONE = "__NONE__"
_R_BULK_SCRIPT = r'''
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 19) {
  stop("Expected 19 arguments.")
}
matrix_path <- args[[1]]
metadata_path <- args[[2]]
output_unadjusted_matrix_path <- args[[3]]
output_adjusted_matrix_path <- args[[4]]
output_metadata_path <- args[[5]]
output_ranked_path <- args[[6]]
output_de_path <- args[[7]]
sample_col <- args[[8]]
group_col <- args[[9]]
group_a <- args[[10]]
group_b <- args[[11]]
orientation <- args[[12]]
sep_override <- args[[13]]
raw_counts <- identical(args[[14]], "true")
batch_col <- args[[15]]
covariate_cols <- if (identical(args[[16]], "")) character(0) else strsplit(args[[16]], ",", fixed = TRUE)[[1]]
sva_mode <- args[[17]]
sva_max_n_sv <- as.integer(args[[18]])
sva_min_residual_df <- as.integer(args[[19]])

read_table_auto <- function(path, sep_override) {
  conn <- if (grepl("\\.gz$", path, ignore.case = TRUE)) gzfile(path, open = "rt") else path
  if (identical(sep_override, "__AUTO__")) {
    lower <- tolower(path)
    if (grepl("\\.tsv(\\.gz)?$", lower) || grepl("\\.txt(\\.gz)?$", lower)) {
      return(read.delim(conn, check.names = FALSE, stringsAsFactors = FALSE))
    }
    return(read.csv(conn, check.names = FALSE, stringsAsFactors = FALSE))
  }
  read.table(
    conn,
    sep = sep_override,
    header = TRUE,
    check.names = FALSE,
    stringsAsFactors = FALSE,
    comment.char = "",
    quote = "\""
  )
}

as_design_column <- function(x) {
  x_chr <- as.character(x)
  x_num <- suppressWarnings(as.numeric(x_chr))
  numeric_ok <- !all(is.na(x_num)) && !any(is.na(x_num[x_chr != "NA"]))
  if (numeric_ok) {
    return(x_num)
  }
  factor(x_chr)
}

expand_covariates <- function(df, columns) {
  mats <- list()
  for (col in columns) {
    vec <- df[[col]]
    if (is.factor(vec)) {
      mm <- model.matrix(~ vec - 1)
      colnames(mm) <- paste0(col, "__", gsub("^vec", "", colnames(mm)))
      mats[[col]] <- mm
    } else {
      mats[[col]] <- matrix(vec, ncol = 1, dimnames = list(rownames(df), col))
    }
  }
  if (length(mats) == 0) {
    return(NULL)
  }
  do.call(cbind, mats)
}

matrix_raw <- read_table_auto(matrix_path, sep_override)
metadata <- read_table_auto(metadata_path, sep_override)
if (!(sample_col %in% colnames(metadata))) {
  stop(paste0("metadata is missing sample column '", sample_col, "'."))
}
if (!(group_col %in% colnames(metadata))) {
  stop(paste0("metadata is missing group column '", group_col, "'."))
}
if (!identical(batch_col, "__NONE__") && !(batch_col %in% colnames(metadata))) {
  stop(paste0("metadata is missing batch column '", batch_col, "'."))
}
for (col in covariate_cols) {
  if (!(col %in% colnames(metadata))) {
    stop(paste0("metadata is missing covariate column '", col, "'."))
  }
}

rownames(matrix_raw) <- as.character(matrix_raw[[1]])
matrix_df <- matrix_raw[, -1, drop = FALSE]
matrix_df[] <- lapply(matrix_df, function(col) as.numeric(as.character(col)))
metadata[[sample_col]] <- as.character(metadata[[sample_col]])
metadata[[group_col]] <- as.character(metadata[[group_col]])
metadata <- metadata[!duplicated(metadata[[sample_col]]), , drop = FALSE]
rownames(metadata) <- metadata[[sample_col]]

if (identical(orientation, "genes_by_samples")) {
  sample_ids <- colnames(matrix_df)
  common <- metadata[[sample_col]][metadata[[sample_col]] %in% sample_ids]
  counts_gs <- as.matrix(matrix_df[, common, drop = FALSE])
  aligned_meta <- metadata[common, , drop = FALSE]
} else if (identical(orientation, "samples_by_genes")) {
  sample_ids <- rownames(matrix_df)
  common <- metadata[[sample_col]][metadata[[sample_col]] %in% sample_ids]
  counts_gs <- t(as.matrix(matrix_df[common, , drop = FALSE]))
  aligned_meta <- metadata[common, , drop = FALSE]
} else {
  stop("matrix_orientation must be 'genes_by_samples' or 'samples_by_genes'.")
}

if (length(common) == 0) {
  stop("No overlapping sample IDs between matrix and metadata.")
}

keep_samples <- aligned_meta[[group_col]] %in% c(group_a, group_b)
counts_gs <- counts_gs[, keep_samples, drop = FALSE]
aligned_meta <- aligned_meta[keep_samples, , drop = FALSE]
if (sum(aligned_meta[[group_col]] == group_a) < 2 || sum(aligned_meta[[group_col]] == group_b) < 2) {
  stop("Each group must have at least 2 samples after alignment.")
}

aligned_meta$group <- factor(aligned_meta[[group_col]], levels = c(group_b, group_a))
if (!identical(batch_col, "__NONE__")) {
  aligned_meta[[batch_col]] <- factor(as.character(aligned_meta[[batch_col]]))
}
for (col in covariate_cols) {
  aligned_meta[[col]] <- as_design_column(aligned_meta[[col]])
}

base_terms <- character(0)
if (!identical(batch_col, "__NONE__")) {
  base_terms <- c(base_terms, paste0("`", batch_col, "`"))
}
if (length(covariate_cols) > 0) {
  base_terms <- c(base_terms, paste0("`", covariate_cols, "`"))
}
base_terms_with_group <- c(base_terms, "group")
design_formula <- as.formula(paste("~", paste(base_terms_with_group, collapse = " + ")))
design_base <- model.matrix(design_formula, data = aligned_meta)
if (qr(design_base)$rank < ncol(design_base)) {
  stop("Design matrix is rank-deficient. Reduce batch/covariate terms or provide more samples.")
}
coef_names <- grep("^group", colnames(design_base), value = TRUE)
if (length(coef_names) != 1) {
  stop("Unable to identify a unique group coefficient for the requested contrast.")
}
coef_name <- coef_names[[1]]

if (raw_counts) {
  if (!requireNamespace("edgeR", quietly = TRUE)) {
    stop("edgeR is required for raw-count ranking but is not installed.")
  }
  if (!requireNamespace("limma", quietly = TRUE)) {
    stop("limma is required for discovery-matrix adjustment but is not installed.")
  }
  y <- edgeR::DGEList(counts = counts_gs)
  keep_genes <- edgeR::filterByExpr(y, group = aligned_meta$group)
  if (sum(keep_genes) == 0) {
    stop("No genes remained after edgeR::filterByExpr. Check counts and contrast size.")
  }
  y <- y[keep_genes, , keep.lib.sizes = FALSE]
  y <- edgeR::calcNormFactors(y)
  log_expr <- edgeR::cpm(y, log = TRUE, prior.count = 2)
} else {
  if (!requireNamespace("limma", quietly = TRUE)) {
    stop("limma is required for normalized-matrix ranking but is not installed.")
  }
  log_expr <- counts_gs
  keep_genes <- apply(log_expr, 1, function(x) { any(is.finite(x)) && stats::sd(x, na.rm = TRUE) > 0 })
  if (sum(keep_genes) == 0) {
    stop("No genes with non-zero variance remained for limma ranking.")
  }
  log_expr <- log_expr[keep_genes, , drop = FALSE]
}

sva_used <- FALSE
n_surrogate_variables <- 0L
surrogate_variable_columns <- character(0)
sva_message <- "disabled"
if (!identical(sva_mode, "off")) {
  if (!requireNamespace("sva", quietly = TRUE)) {
    if (identical(sva_mode, "on")) {
      stop("SVA was requested but the R package 'sva' is not installed.")
    }
    sva_message <- "sva package not installed; skipped surrogate variable estimation"
  } else {
    mod <- design_base
    if (length(base_terms) > 0) {
      mod0_formula <- as.formula(paste("~", paste(base_terms, collapse = " + ")))
      mod0 <- model.matrix(mod0_formula, data = aligned_meta)
    } else {
      mod0 <- model.matrix(~ 1, data = aligned_meta)
    }
    residual_df <- nrow(aligned_meta) - qr(mod)$rank
    max_sv_allowed <- min(as.integer(sva_max_n_sv), max(0L, as.integer(residual_df) - 1L))
    if (residual_df < as.integer(sva_min_residual_df) || max_sv_allowed < 1L) {
      sva_message <- paste0(
        "residual degrees of freedom too small for guarded SVA (residual_df=",
        residual_df,
        ")"
      )
      if (identical(sva_mode, "on")) {
        stop(paste0("SVA was requested but ", sva_message, "."))
      }
    } else {
      estimated_n_sv <- tryCatch(
        as.integer(sva::num.sv(as.matrix(log_expr), mod, method = "be")),
        error = function(e) NA_integer_
      )
      if (is.na(estimated_n_sv) || estimated_n_sv < 1L) {
        sva_message <- "num.sv estimated zero usable surrogate variables"
        if (identical(sva_mode, "on")) {
          stop("SVA was requested but num.sv estimated zero usable surrogate variables.")
        }
      } else {
        estimated_n_sv <- min(as.integer(estimated_n_sv), as.integer(max_sv_allowed))
        sv_fit <- tryCatch(
          sva::sva(as.matrix(log_expr), mod = mod, mod0 = mod0, n.sv = estimated_n_sv),
          error = function(e) e
        )
        if (inherits(sv_fit, "error")) {
          sva_message <- paste0("sva failed: ", conditionMessage(sv_fit))
          if (identical(sva_mode, "on")) {
            stop(paste0("SVA was requested but ", sva_message))
          }
        } else if (is.null(sv_fit$sv) || ncol(sv_fit$sv) < 1L) {
          sva_message <- "sva returned zero surrogate variables"
          if (identical(sva_mode, "on")) {
            stop("SVA was requested but no surrogate variables were returned.")
          }
        } else {
          sv_matrix <- sv_fit$sv
          surrogate_variable_columns <- paste0("SV", seq_len(ncol(sv_matrix)))
          colnames(sv_matrix) <- surrogate_variable_columns
          for (sv_name in surrogate_variable_columns) {
            aligned_meta[[sv_name]] <- as.numeric(sv_matrix[, sv_name])
          }
          sva_used <- TRUE
          n_surrogate_variables <- ncol(sv_matrix)
          sva_message <- paste0("used ", n_surrogate_variables, " surrogate variable(s)")
        }
      }
    }
  }
}

final_covariate_cols <- c(covariate_cols, surrogate_variable_columns)
final_terms <- base_terms
if (length(surrogate_variable_columns) > 0) {
  final_terms <- c(final_terms, surrogate_variable_columns)
}
final_terms <- c(final_terms, "group")
design_formula_final <- as.formula(paste("~", paste(final_terms, collapse = " + ")))
design <- model.matrix(design_formula_final, data = aligned_meta)
if (qr(design)$rank < ncol(design)) {
  stop("Final design matrix became rank-deficient after adding surrogate variables.")
}
coef_names_final <- grep("^group", colnames(design), value = TRUE)
if (length(coef_names_final) != 1) {
  stop("Unable to identify a unique group coefficient after surrogate-variable adjustment.")
}
coef_name <- coef_names_final[[1]]

if (raw_counts) {
  fit_y <- edgeR::estimateDisp(y, design)
  fit <- edgeR::glmQLFit(fit_y, design, robust = TRUE)
  test <- edgeR::glmQLFTest(fit, coef = which(colnames(design) == coef_name))
  tt <- edgeR::topTags(test, n = Inf, sort.by = "none")$table
  tt$gene <- rownames(tt)
  de_out <- data.frame(
    gene = tt$gene,
    logFC = tt$logFC,
    average_expression = tt$logCPM,
    p_value = tt$PValue,
    fdr = tt$FDR,
    score = sign(tt$logFC) * -log10(pmax(tt$PValue, 1e-300)),
    method = "edgeR_glmQLF"
  )
} else {
  fit <- limma::lmFit(log_expr, design)
  fit <- limma::eBayes(fit, robust = TRUE, trend = TRUE)
  tt <- limma::topTable(fit, coef = coef_name, number = Inf, sort.by = "none")
  tt$gene <- rownames(tt)
  de_out <- data.frame(
    gene = tt$gene,
    logFC = tt$logFC,
    average_expression = tt$AveExpr,
    p_value = tt$P.Value,
    fdr = tt$adj.P.Val,
    score = sign(tt$logFC) * -log10(pmax(tt$P.Value, 1e-300)),
    method = "limma_eBayes"
  )
}

preserve_design <- model.matrix(~ group, data = aligned_meta)
cov_matrix <- expand_covariates(aligned_meta, final_covariate_cols)
if (!identical(batch_col, "__NONE__") || !is.null(cov_matrix)) {
  batch_arg <- if (!identical(batch_col, "__NONE__")) aligned_meta[[batch_col]] else NULL
  adjusted <- limma::removeBatchEffect(
    log_expr,
    batch = batch_arg,
    covariates = cov_matrix,
    design = preserve_design
  )
} else {
  adjusted <- log_expr
}

unadjusted_df <- data.frame(gene = rownames(log_expr), log_expr, check.names = FALSE)
colnames(unadjusted_df)[-1] <- rownames(aligned_meta)
adjusted_df <- data.frame(gene = rownames(adjusted), adjusted, check.names = FALSE)
colnames(adjusted_df)[-1] <- rownames(aligned_meta)
write.csv(unadjusted_df, output_unadjusted_matrix_path, row.names = FALSE)
write.csv(adjusted_df, output_adjusted_matrix_path, row.names = FALSE)
write.csv(aligned_meta[, setdiff(colnames(aligned_meta), "group"), drop = FALSE], output_metadata_path, row.names = FALSE)
write.csv(de_out[order(de_out$p_value, decreasing = FALSE), , drop = FALSE], output_de_path, row.names = FALSE)
write.csv(de_out[, c("gene", "score")][order(de_out$score, decreasing = TRUE), , drop = FALSE], output_ranked_path, row.names = FALSE)
cat(paste0("NPW__ranking_method=", if (raw_counts) "edgeR_glmQLF" else "limma_eBayes"), "\n")
cat(paste0("NPW__sva_mode=", sva_mode), "\n")
cat(paste0("NPW__sva_used=", if (sva_used) "true" else "false"), "\n")
cat(paste0("NPW__n_surrogate_variables=", n_surrogate_variables), "\n")
cat(paste0("NPW__surrogate_variable_columns=", paste(surrogate_variable_columns, collapse = ",")), "\n")
cat(paste0("NPW__sva_message=", gsub("[[:cntrl:]]+", " ", sva_message)), "\n")
'''


@dataclass
class PreparedBulkArtifacts:
    """Intermediate files generated before the main nPathway run."""

    unadjusted_matrix_path: str
    discovery_matrix_path: str
    aligned_metadata_path: str
    ranked_genes_path: str
    de_results_path: str
    batch_qc_dir: str
    ranking_method: str
    discovery_adjustment: str
    surrogate_variable_mode: str
    surrogate_variable_used: bool
    n_surrogate_variables: int
    surrogate_variable_columns: tuple[str, ...]
    surrogate_variable_message: str


def _parse_csv_list(value: str | None) -> tuple[str, ...]:
    if value is None:
        return tuple()
    return tuple(part.strip() for part in str(value).split(",") if part.strip())


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", required=True, help="Path to count or normalized bulk matrix CSV/TSV.")
    parser.add_argument("--metadata", required=True, help="Path to sample metadata CSV/TSV.")
    parser.add_argument("--sample-col", default="sample", help="Metadata sample ID column.")
    parser.add_argument("--group-col", required=True, help="Metadata contrast column.")
    parser.add_argument("--group-a", required=True, help="Case / target group label.")
    parser.add_argument("--group-b", required=True, help="Reference / control group label.")
    parser.add_argument(
        "--matrix-orientation",
        default="genes_by_samples",
        choices=["genes_by_samples", "samples_by_genes"],
        help="Input matrix orientation.",
    )
    parser.add_argument("--sep", default=None, help="Delimiter override for matrix and metadata.")
    parser.add_argument(
        "--raw-counts",
        action="store_true",
        help="Treat the matrix as raw counts and use edgeR for the ranking step.",
    )
    parser.add_argument("--batch-col", default=None, help="Optional known batch column in metadata.")
    parser.add_argument(
        "--covariate-cols",
        default="",
        help="Comma-separated nuisance covariates to regress out for discovery and include in ranking.",
    )
    parser.add_argument(
        "--surrogate-variable-mode",
        default="auto",
        choices=["off", "auto", "on"],
        help="Optional surrogate-variable adjustment: off disables SVA, auto uses it when available and safe, on requires it.",
    )
    parser.add_argument(
        "--sva-max-n-sv",
        type=int,
        default=3,
        help="Maximum number of surrogate variables when SVA is enabled.",
    )
    parser.add_argument(
        "--sva-min-residual-df",
        type=int,
        default=3,
        help="Minimum residual degrees of freedom required before guarded SVA will run.",
    )
    parser.add_argument(
        "--ranked-genes",
        default=None,
        help="Optional external ranked gene table. When provided, nPathway GSEA uses this ranking instead of generating one internally.",
    )
    parser.add_argument("--ranked-gene-col", default="gene", help="Gene column name in --ranked-genes.")
    parser.add_argument("--ranked-score-col", default="score", help="Score column name in --ranked-genes.")
    parser.add_argument("--ranked-sep", default=None, help="Delimiter override for --ranked-genes.")
    parser.add_argument(
        "--curated-gmt",
        default=None,
        help="Optional curated GMT used for program annotation and same-ranked-list GSEA comparison.",
    )
    parser.add_argument(
        "--focus-genes",
        default="",
        help="Comma-separated genes to track in anchor dynamic programs versus curated sets.",
    )
    parser.add_argument(
        "--discovery-method",
        default="kmeans",
        choices=["leiden", "spectral", "kmeans", "hdbscan"],
        help="Program discovery method.",
    )
    parser.add_argument("--n-programs", type=int, default=20, help="Target number of programs for kmeans/spectral.")
    parser.add_argument("--k-neighbors", type=int, default=15, help="kNN neighbors.")
    parser.add_argument("--resolution", type=float, default=1.0, help="Leiden resolution.")
    parser.add_argument("--n-components", type=int, default=30, help="Embedding dimension.")
    parser.add_argument("--n-diffusion-steps", type=int, default=3, help="Diffusion iterations.")
    parser.add_argument("--diffusion-alpha", type=float, default=0.5, help="Diffusion self-weight.")
    parser.add_argument(
        "--de-test",
        default="welch",
        choices=["welch", "mwu"],
        help="Internal DE fallback used for claim tables only when no external ranking is supplied.",
    )
    parser.add_argument("--de-alpha", type=float, default=0.05, help="DE FDR threshold.")
    parser.add_argument(
        "--min-abs-logfc-for-claim",
        type=float,
        default=0.2,
        help="Minimum mean abs(logFC) gate for claim support.",
    )
    parser.add_argument("--gsea-n-perm", type=int, default=1000, help="GSEA permutations.")
    parser.add_argument(
        "--min-genes-per-program-claim",
        type=int,
        default=10,
        help="Minimum program size gate for claim support.",
    )
    parser.add_argument("--n-bootstrap", type=int, default=0, help="Bootstrap runs for stability estimation (0 disables).")
    parser.add_argument(
        "--min-stability-for-claim",
        type=float,
        default=0.25,
        help="Minimum stability gate when bootstrap is enabled.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--annotate-programs",
        action="store_true",
        help="Annotate programs with closest reference sets (MSigDB/custom GMT).",
    )
    parser.add_argument(
        "--annotation-collections",
        default="hallmark,go_bp,kegg",
        help="Comma-separated MSigDB collections for annotation.",
    )
    parser.add_argument(
        "--annotation-species",
        default="human",
        choices=["human", "mouse"],
        help="MSigDB species for annotation.",
    )
    parser.add_argument("--annotation-gmt", default=None, help="Optional custom GMT file for program naming annotation.")
    parser.add_argument(
        "--annotation-topk-per-program",
        type=int,
        default=15,
        help="Top reference matches saved per program for heatmap/reporting.",
    )
    parser.add_argument(
        "--annotation-min-jaccard-for-label",
        type=float,
        default=0.03,
        help="Minimum Jaccard to use reference-derived label (else Unmatched).",
    )
    parser.add_argument("--output-dir", default=None, help="Workflow output directory.")
    parser.add_argument("--with-dashboard", action="store_true", help="Build interactive dashboard package after the run.")
    parser.add_argument(
        "--dashboard-output-dir",
        default=None,
        help="Dashboard output directory (default: <output-dir>/dashboard).",
    )
    parser.add_argument("--dashboard-top-k", type=int, default=20, help="Top-K rows used in dashboard focused plots.")
    parser.add_argument("--dashboard-no-pdf", action="store_true", help="Dashboard figures PNG only (skip PDF).")
    parser.add_argument("--verbose", action="store_true", help="Enable INFO logs.")
    return parser.parse_args(argv)


def _require_int_at_least(name: str, value: int, minimum: int) -> None:
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}.")


def _require_float_range(
    name: str,
    value: float,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    min_inclusive: bool = True,
    max_inclusive: bool = True,
) -> None:
    if minimum is not None:
        if min_inclusive and value < minimum:
            raise ValueError(f"{name} must be >= {minimum}.")
        if not min_inclusive and value <= minimum:
            raise ValueError(f"{name} must be > {minimum}.")
    if maximum is not None:
        if max_inclusive and value > maximum:
            raise ValueError(f"{name} must be <= {maximum}.")
        if not max_inclusive and value >= maximum:
            raise ValueError(f"{name} must be < {maximum}.")


def _validate_cli_args(args: argparse.Namespace) -> None:
    if args.group_a == args.group_b:
        raise ValueError("--group-a and --group-b must be different labels.")
    _require_int_at_least("--n-programs", args.n_programs, 1)
    _require_int_at_least("--k-neighbors", args.k_neighbors, 1)
    _require_float_range("--resolution", args.resolution, minimum=0.0, min_inclusive=False)
    _require_int_at_least("--n-components", args.n_components, 1)
    _require_int_at_least("--n-diffusion-steps", args.n_diffusion_steps, 0)
    _require_float_range("--diffusion-alpha", args.diffusion_alpha, minimum=0.0, maximum=1.0)
    _require_float_range("--de-alpha", args.de_alpha, minimum=0.0, maximum=1.0, min_inclusive=False)
    _require_float_range("--min-abs-logfc-for-claim", args.min_abs_logfc_for_claim, minimum=0.0)
    _require_int_at_least("--gsea-n-perm", args.gsea_n_perm, 1)
    _require_int_at_least("--min-genes-per-program-claim", args.min_genes_per_program_claim, 1)
    _require_int_at_least("--n-bootstrap", args.n_bootstrap, 0)
    _require_int_at_least("--sva-max-n-sv", args.sva_max_n_sv, 1)
    _require_int_at_least("--sva-min-residual-df", args.sva_min_residual_df, 1)
    _require_float_range("--min-stability-for-claim", args.min_stability_for_claim, minimum=0.0, maximum=1.0)
    _require_int_at_least("--annotation-topk-per-program", args.annotation_topk_per_program, 1)
    _require_float_range(
        "--annotation-min-jaccard-for-label",
        args.annotation_min_jaccard_for_label,
        minimum=0.0,
        maximum=1.0,
    )
    _require_int_at_least("--dashboard-top-k", args.dashboard_top_k, 1)


def _detect_r_dependencies(*, raw_counts: bool, surrogate_variable_mode: str = "off") -> dict[str, bool]:
    pkgs = ["limma"] + (["edgeR"] if raw_counts else [])
    if surrogate_variable_mode != "off":
        pkgs.append("sva")
    expr = "pkgs <- c(%s); cat(paste(pkgs, sapply(pkgs, requireNamespace, quietly=TRUE), sep='='), sep='\\n')" % \
        ",".join(f'\"{pkg}\"' for pkg in pkgs)
    try:
        result = subprocess.run(
            ["Rscript", "-e", expr],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("Rscript is required for the batch-aware bulk workflow but was not found.") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"Failed to query R package availability: {exc.stderr.strip()}") from exc

    availability: dict[str, bool] = {}
    for line in result.stdout.splitlines():
        if "=" not in line:
            continue
        name, flag = line.strip().split("=", 1)
        availability[name] = flag.strip().upper() == "TRUE"
    missing_required = [pkg for pkg in (["limma"] + (["edgeR"] if raw_counts else [])) if not availability.get(pkg, False)]
    missing_optional = []
    if surrogate_variable_mode == "on" and not availability.get("sva", False):
        missing_required.append("sva")
    elif surrogate_variable_mode == "auto" and not availability.get("sva", False):
        missing_optional.append("sva")
    missing = missing_required
    if missing:
        missing_str = ", ".join(missing)
        raise RuntimeError(
            "This workflow requires R packages that are not installed: "
            f"{missing_str}. Install them before running the batch-aware workflow."
        )
    if missing_optional:
        logger.warning(
            "Optional R package(s) missing for surrogate-variable auto mode: %s. "
            "The workflow will fall back to known batch/covariate adjustment only.",
            ", ".join(missing_optional),
        )
    return availability


def _parse_r_preparation_metadata(stdout: str) -> dict[str, str]:
    metadata: dict[str, str] = {}
    for line in stdout.splitlines():
        line = line.strip()
        if not line.startswith("NPW__") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        metadata[key.replace("NPW__", "", 1)] = value
    return metadata


def _read_expression_csv(path: Path, sample_ids: list[str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "gene" not in df.columns:
        raise ValueError(f"Expression matrix is missing a 'gene' column: {path}")
    missing = [sample for sample in sample_ids if sample not in df.columns]
    if missing:
        raise ValueError(f"Expression matrix is missing aligned samples: {missing[:5]}")
    expr = df.set_index("gene")[sample_ids].apply(pd.to_numeric, errors="coerce")
    if expr.isna().any().any():
        raise ValueError(f"Expression matrix contains non-numeric values after coercion: {path}")
    return expr


def _summarise_batch_correlation(corr: pd.DataFrame, metadata: pd.DataFrame, *, sample_col: str, batch_col: str | None) -> dict[str, float | None]:
    if batch_col is None or batch_col not in metadata.columns:
        return {"mean_within_batch_corr": None, "mean_between_batch_corr": None}
    meta = metadata[[sample_col, batch_col]].copy()
    meta[sample_col] = meta[sample_col].astype(str)
    meta[batch_col] = meta[batch_col].astype(str)
    meta = meta.drop_duplicates(subset=[sample_col]).set_index(sample_col)
    if corr.shape[0] < 2:
        return {"mean_within_batch_corr": None, "mean_between_batch_corr": None}
    within: list[float] = []
    between: list[float] = []
    samples = [sample for sample in corr.index if sample in meta.index]
    for i, left in enumerate(samples):
        for right in samples[i + 1 :]:
            value = float(corr.loc[left, right])
            if meta.loc[left, batch_col] == meta.loc[right, batch_col]:
                within.append(value)
            else:
                between.append(value)
    return {
        "mean_within_batch_corr": float(np.mean(within)) if within else None,
        "mean_between_batch_corr": float(np.mean(between)) if between else None,
    }


def _write_pca_plot(
    coords: pd.DataFrame,
    *,
    sample_col: str,
    group_col: str,
    batch_col: str | None,
    title: str,
    out_path: Path,
) -> None:
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(7.8, 5.8))
    scatter_kwargs: dict[str, object] = {"x": "PC1", "y": "PC2", "hue": group_col, "palette": "Set2", "s": 90}
    if batch_col is not None and batch_col in coords.columns:
        scatter_kwargs["style"] = batch_col
    sns.scatterplot(data=coords, ax=ax, **scatter_kwargs)
    for row in coords.itertuples(index=False):
        ax.text(
            float(getattr(row, "PC1")) + 0.02,
            float(getattr(row, "PC2")) + 0.02,
            str(getattr(row, sample_col)),
            fontsize=8,
            alpha=0.85,
        )
    pc1 = float(coords.attrs.get("variance_ratio_pc1", 0.0)) * 100.0
    pc2 = float(coords.attrs.get("variance_ratio_pc2", 0.0)) * 100.0
    ax.set_xlabel(f"PC1 ({pc1:.1f}%)")
    ax.set_ylabel(f"PC2 ({pc2:.1f}%)")
    ax.set_title(title)
    ax.legend(loc="best", frameon=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _write_correlation_heatmap(
    corr: pd.DataFrame,
    *,
    title: str,
    out_path: Path,
) -> None:
    sns.set_theme(style="white")
    fig, ax = plt.subplots(figsize=(7.8, 6.6))
    sns.heatmap(
        corr,
        cmap="mako",
        vmin=-1.0,
        vmax=1.0,
        square=True,
        linewidths=0.35,
        cbar_kws={"label": "Pearson correlation"},
        ax=ax,
    )
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _write_batch_qc_artifacts(
    *,
    unadjusted_matrix_path: Path,
    discovery_matrix_path: Path,
    aligned_metadata_path: Path,
    sample_col: str,
    group_col: str,
    batch_col: str | None,
    qc_dir: Path,
    discovery_adjustment: str,
    surrogate_variable_mode: str,
    surrogate_variable_used: bool,
    n_surrogate_variables: int,
    surrogate_variable_message: str,
) -> None:
    metadata = pd.read_csv(aligned_metadata_path)
    metadata[sample_col] = metadata[sample_col].astype(str)
    order_cols = [col for col in [batch_col, group_col, sample_col] if col and col in metadata.columns]
    if order_cols:
        metadata = metadata.sort_values(order_cols).reset_index(drop=True)
    sample_ids = metadata[sample_col].tolist()
    expr_before = _read_expression_csv(unadjusted_matrix_path, sample_ids)
    expr_after = _read_expression_csv(discovery_matrix_path, sample_ids)

    def _coords(expr: pd.DataFrame) -> tuple[pd.DataFrame, list[float]]:
        matrix = expr.T.to_numpy(dtype=float)
        n_components = min(2, matrix.shape[0], matrix.shape[1])
        if n_components < 1:
            raise ValueError("At least one component is required to compute PCA coordinates.")
        pca = PCA(n_components=n_components, svd_solver="full")
        transformed = pca.fit_transform(matrix)
        coords = metadata.copy()
        coords["PC1"] = transformed[:, 0]
        coords["PC2"] = transformed[:, 1] if n_components > 1 else 0.0
        explained = pca.explained_variance_ratio_.tolist()
        if len(explained) < 2:
            explained = explained + [0.0] * (2 - len(explained))
        coords.attrs["variance_ratio_pc1"] = explained[0]
        coords.attrs["variance_ratio_pc2"] = explained[1]
        return coords, explained

    qc_dir.mkdir(parents=True, exist_ok=True)
    before_coords, before_explained = _coords(expr_before)
    after_coords, after_explained = _coords(expr_after)
    before_corr = pd.DataFrame(np.corrcoef(expr_before.T.to_numpy(dtype=float)), index=sample_ids, columns=sample_ids)
    after_corr = pd.DataFrame(np.corrcoef(expr_after.T.to_numpy(dtype=float)), index=sample_ids, columns=sample_ids)

    before_coords.to_csv(qc_dir / "pca_before.csv", index=False)
    after_coords.to_csv(qc_dir / "pca_after.csv", index=False)
    before_corr.to_csv(qc_dir / "correlation_before.csv", index=True)
    after_corr.to_csv(qc_dir / "correlation_after.csv", index=True)

    _write_pca_plot(
        before_coords,
        sample_col=sample_col,
        group_col=group_col,
        batch_col=batch_col,
        title="Before Batch Adjustment: Pseudobulk PCA",
        out_path=qc_dir / "pca_before.png",
    )
    _write_pca_plot(
        after_coords,
        sample_col=sample_col,
        group_col=group_col,
        batch_col=batch_col,
        title="After Batch Adjustment: Pseudobulk PCA",
        out_path=qc_dir / "pca_after.png",
    )
    _write_correlation_heatmap(
        before_corr,
        title="Before Batch Adjustment: Sample Correlation",
        out_path=qc_dir / "correlation_before.png",
    )
    _write_correlation_heatmap(
        after_corr,
        title="After Batch Adjustment: Sample Correlation",
        out_path=qc_dir / "correlation_after.png",
    )

    summary = {
        "sample_col": sample_col,
        "group_col": group_col,
        "batch_col": batch_col,
        "n_samples": len(sample_ids),
        "discovery_adjustment": discovery_adjustment,
        "surrogate_variable_mode": surrogate_variable_mode,
        "surrogate_variable_used": surrogate_variable_used,
        "n_surrogate_variables": n_surrogate_variables,
        "surrogate_variable_message": surrogate_variable_message,
        "variance_ratio_before": {"pc1": before_explained[0], "pc2": before_explained[1]},
        "variance_ratio_after": {"pc1": after_explained[0], "pc2": after_explained[1]},
        "correlation_before": _summarise_batch_correlation(before_corr, metadata, sample_col=sample_col, batch_col=batch_col),
        "correlation_after": _summarise_batch_correlation(after_corr, metadata, sample_col=sample_col, batch_col=batch_col),
        "artifacts": {
            "pca_before_csv": str(qc_dir / "pca_before.csv"),
            "pca_after_csv": str(qc_dir / "pca_after.csv"),
            "pca_before_png": str(qc_dir / "pca_before.png"),
            "pca_after_png": str(qc_dir / "pca_after.png"),
            "correlation_before_csv": str(qc_dir / "correlation_before.csv"),
            "correlation_after_csv": str(qc_dir / "correlation_after.csv"),
            "correlation_before_png": str(qc_dir / "correlation_before.png"),
            "correlation_after_png": str(qc_dir / "correlation_after.png"),
        },
    }
    (qc_dir / "pca_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def _prepare_bulk_artifacts(args: argparse.Namespace, outdir: Path) -> PreparedBulkArtifacts:
    inputs_dir = outdir / "prepared_inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)

    unadjusted_matrix_path = inputs_dir / "discovery_matrix_unadjusted.csv"
    discovery_matrix_path = inputs_dir / "discovery_matrix_adjusted.csv"
    aligned_metadata_path = inputs_dir / "aligned_metadata.csv"
    ranked_genes_path = inputs_dir / "ranked_genes_external.csv"
    de_results_path = inputs_dir / "de_results_external.csv"
    batch_qc_dir = inputs_dir / "qc"

    _detect_r_dependencies(
        raw_counts=bool(args.raw_counts),
        surrogate_variable_mode=str(args.surrogate_variable_mode),
    )
    with tempfile.TemporaryDirectory(prefix="npathway_bulk_workflow_") as tmpdir:
        script_path = Path(tmpdir) / "bulk_batch_workflow.R"
        script_path.write_text(_R_BULK_SCRIPT, encoding="utf-8")
        cmd = [
            "Rscript",
            str(script_path),
            str(args.matrix),
            str(args.metadata),
            str(unadjusted_matrix_path),
            str(discovery_matrix_path),
            str(aligned_metadata_path),
            str(ranked_genes_path),
            str(de_results_path),
            str(args.sample_col),
            str(args.group_col),
            str(args.group_a),
            str(args.group_b),
            str(args.matrix_orientation),
            args.sep if args.sep is not None else _AUTO,
            "true" if args.raw_counts else "false",
            args.batch_col if args.batch_col else _NONE,
            ",".join(_parse_csv_list(args.covariate_cols)),
            str(args.surrogate_variable_mode),
            str(args.sva_max_n_sv),
            str(args.sva_min_residual_df),
        ]
        try:
            completed = subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as exc:
            stderr = exc.stderr.strip()
            stdout = exc.stdout.strip()
            message = stderr or stdout or "Unknown R workflow failure."
            raise RuntimeError(f"Batch-aware preparation failed: {message}") from exc
    prep_meta = _parse_r_preparation_metadata(completed.stdout)
    ranking_method = prep_meta.get("ranking_method", "edgeR_glmQLF" if args.raw_counts else "limma_eBayes")
    sva_used = prep_meta.get("sva_used", "false").lower() == "true"
    n_surrogate_variables = int(prep_meta.get("n_surrogate_variables", "0") or "0")
    surrogate_variable_columns = tuple(
        item for item in prep_meta.get("surrogate_variable_columns", "").split(",") if item
    )
    discovery_adjustment = "none"
    if args.batch_col or _parse_csv_list(args.covariate_cols):
        discovery_adjustment = "limma_removeBatchEffect"
    if sva_used:
        discovery_adjustment = (
            "sva_only" if discovery_adjustment == "none" else f"{discovery_adjustment}+sva"
        )

    manifest = {
        "batch_col": args.batch_col,
        "covariate_cols": list(_parse_csv_list(args.covariate_cols)),
        "ranking_method": ranking_method,
        "discovery_adjustment": discovery_adjustment,
        "surrogate_variable_mode": str(args.surrogate_variable_mode),
        "surrogate_variable_used": sva_used,
        "n_surrogate_variables": n_surrogate_variables,
        "surrogate_variable_columns": list(surrogate_variable_columns),
        "surrogate_variable_message": prep_meta.get("sva_message", "not reported"),
    }
    _write_batch_qc_artifacts(
        unadjusted_matrix_path=unadjusted_matrix_path,
        discovery_matrix_path=discovery_matrix_path,
        aligned_metadata_path=aligned_metadata_path,
        sample_col=str(args.sample_col),
        group_col=str(args.group_col),
        batch_col=str(args.batch_col) if args.batch_col else None,
        qc_dir=batch_qc_dir,
        discovery_adjustment=discovery_adjustment,
        surrogate_variable_mode=str(args.surrogate_variable_mode),
        surrogate_variable_used=sva_used,
        n_surrogate_variables=n_surrogate_variables,
        surrogate_variable_message=prep_meta.get("sva_message", "not reported"),
    )
    manifest["unadjusted_matrix_path"] = str(unadjusted_matrix_path)
    manifest["batch_qc_dir"] = str(batch_qc_dir)
    (inputs_dir / "preparation_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return PreparedBulkArtifacts(
        unadjusted_matrix_path=str(unadjusted_matrix_path),
        discovery_matrix_path=str(discovery_matrix_path),
        aligned_metadata_path=str(aligned_metadata_path),
        ranked_genes_path=str(ranked_genes_path),
        de_results_path=str(de_results_path),
        batch_qc_dir=str(batch_qc_dir),
        ranking_method=str(manifest["ranking_method"]),
        discovery_adjustment=str(manifest["discovery_adjustment"]),
        surrogate_variable_mode=str(manifest["surrogate_variable_mode"]),
        surrogate_variable_used=bool(manifest["surrogate_variable_used"]),
        n_surrogate_variables=int(manifest["n_surrogate_variables"]),
        surrogate_variable_columns=tuple(str(x) for x in manifest["surrogate_variable_columns"]),
        surrogate_variable_message=str(manifest["surrogate_variable_message"]),
    )


def _resolve_ranked_genes(args: argparse.Namespace, prepared: PreparedBulkArtifacts) -> tuple[str, str, str, str]:
    if args.ranked_genes is None:
        return (
            prepared.ranked_genes_path,
            "gene",
            "score",
            prepared.ranking_method,
        )

    target = Path(prepared.ranked_genes_path)
    if Path(args.ranked_genes).resolve() != target.resolve():
        shutil.copyfile(args.ranked_genes, target)
    return (
        str(target),
        str(args.ranked_gene_col),
        str(args.ranked_score_col),
        "external_user_table",
    )


def run_batch_aware_bulk_workflow(args: argparse.Namespace) -> dict[str, object]:
    """Run the batch-aware bulk workflow and return structured outputs."""
    report = validate_bulk_input_files(
        matrix_path=args.matrix,
        metadata_path=args.metadata,
        sample_col=args.sample_col,
        group_col=args.group_col,
        group_a=args.group_a,
        group_b=args.group_b,
        matrix_orientation=args.matrix_orientation,
        sep=args.sep,
        raw_counts=bool(args.raw_counts),
    )
    for warning in report.warnings:
        logging.warning("Input validation: %s", warning)

    outdir = Path(args.output_dir) if args.output_dir else Path("results") / (
        f"bulk_batch_workflow_{date.today().strftime('%Y%m%d')}"
    )
    outdir.mkdir(parents=True, exist_ok=True)

    prepared = _prepare_bulk_artifacts(args, outdir)
    ranked_path, ranked_gene_col, ranked_score_col, ranking_source = _resolve_ranked_genes(args, prepared)

    annotation_gmt = args.annotation_gmt
    annotate_programs = bool(args.annotate_programs)
    if args.curated_gmt and annotation_gmt is None:
        annotation_gmt = args.curated_gmt
        annotate_programs = True

    result = run_bulk_dynamic_pipeline(
        config=BulkDynamicConfig(
            matrix_path=prepared.discovery_matrix_path,
            metadata_path=prepared.aligned_metadata_path,
            group_col=args.group_col,
            group_a=args.group_a,
            group_b=args.group_b,
            sample_col=args.sample_col,
            matrix_orientation="genes_by_samples",
            sep=",",
            raw_counts=False,
            discovery_method=args.discovery_method,
            n_programs=args.n_programs,
            k_neighbors=args.k_neighbors,
            resolution=args.resolution,
            n_components=args.n_components,
            n_diffusion_steps=args.n_diffusion_steps,
            diffusion_alpha=args.diffusion_alpha,
            de_test=args.de_test,
            de_alpha=args.de_alpha,
            min_abs_logfc_for_claim=args.min_abs_logfc_for_claim,
            gsea_n_perm=args.gsea_n_perm,
            min_genes_per_program_claim=args.min_genes_per_program_claim,
            n_bootstrap=args.n_bootstrap,
            min_stability_for_claim=args.min_stability_for_claim,
            random_seed=args.seed,
            annotate_programs=annotate_programs,
            annotation_collections=tuple(
                x.strip() for x in str(args.annotation_collections).split(",") if x.strip()
            ),
            annotation_species=args.annotation_species,
            annotation_gmt_path=annotation_gmt,
            annotation_topk_per_program=args.annotation_topk_per_program,
            annotation_min_jaccard_for_label=args.annotation_min_jaccard_for_label,
            ranked_genes_path=ranked_path,
            ranked_genes_sep=args.ranked_sep if args.ranked_genes else ",",
            ranked_gene_col=ranked_gene_col,
            ranked_score_col=ranked_score_col,
        ),
        output_dir=outdir,
    )

    comparison_result = None
    if args.curated_gmt:
        comparison_result = compare_curated_vs_dynamic_gsea(
            ranked_genes_path=ranked_path,
            dynamic_gmt_path=outdir / "dynamic_programs.gmt",
            curated_gmt_path=args.curated_gmt,
            output_dir=outdir / "comparison",
            gene_col=ranked_gene_col,
            score_col=ranked_score_col,
            sep=args.ranked_sep if args.ranked_genes else ",",
            n_perm=args.gsea_n_perm,
            seed=args.seed,
            focus_genes=[g for g in _parse_csv_list(args.focus_genes)],
        )

    workflow_manifest = {
        "prepared": asdict(prepared),
        "ranking_source": ranking_source,
        "curated_gmt": args.curated_gmt,
        "focus_genes": list(_parse_csv_list(args.focus_genes)),
        "result": asdict(result),
        "comparison": asdict(comparison_result) if comparison_result is not None else None,
    }
    (outdir / "workflow_manifest.json").write_text(json.dumps(workflow_manifest, indent=2), encoding="utf-8")

    dashboard_html_path = None
    if args.with_dashboard:
        dashboard_dir = (
            Path(args.dashboard_output_dir)
            if args.dashboard_output_dir
            else Path(result.output_dir) / "dashboard"
        )
        artifacts = build_dynamic_dashboard_package(
            DashboardConfig(
                results_dir=result.output_dir,
                output_dir=str(dashboard_dir),
                title=f"nPathway Dashboard: {args.group_a} vs {args.group_b}",
                top_k=args.dashboard_top_k,
                include_pdf=not args.dashboard_no_pdf,
            )
        )
        dashboard_html_path = artifacts.html_path
    return {
        "prepared": prepared,
        "result": result,
        "comparison_result": comparison_result,
        "ranking_source": ranking_source,
        "ranked_path": ranked_path,
        "ranked_gene_col": ranked_gene_col,
        "ranked_score_col": ranked_score_col,
        "manifest_path": str(outdir / "workflow_manifest.json"),
        "dashboard_html_path": dashboard_html_path,
    }


def main(argv: list[str] | None = None) -> None:
    """CLI entrypoint."""
    args = parse_args(argv)
    _validate_cli_args(args)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    run = run_batch_aware_bulk_workflow(args)
    prepared = run["prepared"]
    result = run["result"]
    comparison_result = run["comparison_result"]
    ranking_source = run["ranking_source"]
    ranked_path = run["ranked_path"]

    print("nPathway batch-aware bulk workflow completed.")
    print(f"- output_dir: {result.output_dir}")
    print(f"- discovery_matrix: {prepared.discovery_matrix_path}")
    print(f"- aligned_metadata: {prepared.aligned_metadata_path}")
    print(f"- ranked_genes: {ranked_path}")
    print(f"- ranking_source: {ranking_source}")
    print(f"- batch_qc_dir: {prepared.batch_qc_dir}")
    print(f"- surrogate_variable_mode: {prepared.surrogate_variable_mode}")
    print(f"- surrogate_variable_used: {prepared.surrogate_variable_used}")
    print(f"- n_surrogate_variables: {prepared.n_surrogate_variables}")
    print(f"- surrogate_variable_message: {prepared.surrogate_variable_message}")
    print(f"- n_samples: {result.n_samples}")
    print(f"- n_genes: {result.n_genes}")
    print(f"- n_programs: {result.n_programs}")
    print(f"- n_sig_de_genes: {result.n_sig_de_genes}")
    if comparison_result is not None:
        print(f"- comparison_dir: {comparison_result.output_dir}")
        print(f"- anchor_program: {comparison_result.anchor_program}")
        print(f"- anchor_reference: {comparison_result.anchor_reference}")
        print(f"- anchor_jaccard: {comparison_result.anchor_jaccard}")
    if run["dashboard_html_path"] is not None:
        print(f"- dashboard_html: {run['dashboard_html_path']}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        print(
            "Hint: validate the input first with `npathway-validate-inputs bulk ...` and "
            "keep batch/covariate columns in the metadata table.",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc
