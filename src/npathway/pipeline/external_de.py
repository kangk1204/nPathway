"""Helpers for external bulk differential-expression backends."""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path


def run_external_de_ranking(
    *,
    root_dir: str | Path,
    method: str,
    counts_path: str | Path,
    metadata_path: str | Path,
    outdir: str | Path,
    sample_col: str,
    group_col: str,
    group_a: str,
    group_b: str,
    covariate_cols: list[str],
    local_r_lib: str | Path | None = None,
) -> tuple[Path, Path]:
    """Run a common bulk DE engine and emit a ranked table plus DE results."""
    root = Path(root_dir)
    out_path = Path(outdir)
    out_path.mkdir(parents=True, exist_ok=True)
    ranked_path = out_path / "ranked_genes_external.csv"
    de_path = out_path / "de_results_external.csv"
    method_label = "limma_voom" if method == "limma_voom" else "DESeq2_Wald"
    covariate_r = ", ".join(f'"{col}"' for col in covariate_cols) if covariate_cols else ""
    script = f"""
suppressPackageStartupMessages({{
  if (!requireNamespace("limma", quietly = TRUE)) {{
    stop("limma is required for the external DE workflow.")
  }}
  if (!requireNamespace("edgeR", quietly = TRUE)) {{
    stop("edgeR is required for the external DE workflow.")
  }}
  if ("{method}" == "deseq2" && !requireNamespace("DESeq2", quietly = TRUE)) {{
    stop("DESeq2 is not installed in this R environment.")
  }}
}})

counts <- read.csv("{Path(counts_path).as_posix()}", row.names = 1, check.names = FALSE)
metadata <- read.csv("{Path(metadata_path).as_posix()}", check.names = FALSE, stringsAsFactors = FALSE)
sample_ids <- metadata[["{sample_col}"]]
counts <- counts[, sample_ids, drop = FALSE]
metadata[["{group_col}"]] <- factor(metadata[["{group_col}"]], levels = c("{group_b}", "{group_a}"))
covariate_cols <- c({covariate_r})
covariate_cols <- covariate_cols[nzchar(covariate_cols)]
design_terms <- c("{group_col}", covariate_cols)
design_formula <- as.formula(paste("~", paste(design_terms, collapse = " + ")))

if ("{method}" == "limma_voom") {{
  dge <- edgeR::DGEList(counts = as.matrix(counts))
  keep <- edgeR::filterByExpr(dge, group = metadata[["{group_col}"]])
  if (!any(keep)) {{
    stop("filterByExpr removed every gene.")
  }}
  dge <- dge[keep, , keep.lib.sizes = FALSE]
  dge <- edgeR::calcNormFactors(dge)
  design <- model.matrix(design_formula, data = metadata)
  coef_name <- paste0("{group_col}", "{group_a}")
  v <- limma::voom(dge, design, plot = FALSE)
  fit <- limma::lmFit(v, design)
  fit <- limma::eBayes(fit, robust = TRUE, trend = TRUE)
  res <- limma::topTable(fit, coef = coef_name, number = Inf, sort.by = "none")
  res$gene <- rownames(res)
  out <- data.frame(
    gene = res$gene,
    logFC = res$logFC,
    average_expression = res$AveExpr,
    p_value = res$P.Value,
    fdr = res$adj.P.Val,
    score = res$t,
    method = "{method_label}",
    stringsAsFactors = FALSE
  )
}} else {{
  if (!all(abs(counts - round(counts)) < 1e-6)) {{
    stop("DESeq2 mode requires integer-like counts. The supplied matrix is not suitable.")
  }}
  dds <- DESeq2::DESeqDataSetFromMatrix(
    countData = round(as.matrix(counts)),
    colData = metadata,
    design = design_formula
  )
  keep <- rowSums(DESeq2::counts(dds)) > 0
  if (!any(keep)) {{
    stop("All genes have zero counts after filtering.")
  }}
  dds <- dds[keep, ]
  dds <- DESeq2::DESeq(dds, quiet = TRUE)
  res <- as.data.frame(DESeq2::results(dds, contrast = c("{group_col}", "{group_a}", "{group_b}")))
  res$gene <- rownames(res)
  out <- data.frame(
    gene = res$gene,
    logFC = res$log2FoldChange,
    average_expression = res$baseMean,
    p_value = res$pvalue,
    fdr = res$padj,
    score = res$stat,
    method = "{method_label}",
    stringsAsFactors = FALSE
  )
}}

out <- out[!is.na(out$gene) & !is.na(out$score), ]
out <- out[is.finite(out$score), ]
out <- out[order(out$score, decreasing = TRUE), ]
write.csv(out, "{de_path.as_posix()}", row.names = FALSE, quote = TRUE)
write.csv(out[, c("gene", "score")], "{ranked_path.as_posix()}", row.names = FALSE, quote = TRUE)
"""

    with tempfile.NamedTemporaryFile("w", suffix=".R", delete=False, encoding="utf-8") as handle:
        handle.write(script)
        script_path = Path(handle.name)
    try:
        env = os.environ.copy()
        if local_r_lib is not None:
            existing_r_lib = env.get("R_LIBS_USER", "")
            env["R_LIBS_USER"] = (
                str(local_r_lib) if not existing_r_lib else f"{local_r_lib}{os.pathsep}{existing_r_lib}"
            )
        proc = subprocess.run(
            ["Rscript", str(script_path)],
            cwd=root,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
    finally:
        script_path.unlink(missing_ok=True)

    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip() or "R external DE ranking failed.")
    return ranked_path, de_path
