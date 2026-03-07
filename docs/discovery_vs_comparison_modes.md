# nPathway Usage Modes

nPathway is best understood as a pathway-specialized system with **two official operating modes**.

## Mode 1. Discovery Mode

Use this mode when you want nPathway to **discover dynamic gene programs from expression data**.

Inputs:

- bulk matrix + metadata
- or scRNA `.h5ad` through the pseudobulk route
- optional external ranked gene table for fair downstream GSEA comparison

What nPathway does:

- builds or ingests an analyzable expression matrix
- discovers dynamic gene programs
- aligns those programs to curated references
- writes claim-gated tables, GMTs, and dashboard outputs

Use this when the scientific question is:

- what dynamic programs exist in this dataset?
- which curated pathways do they align to?
- are there disease-linked genes inside the dynamic program that are absent from curated definitions?

Installed commands:

- `npathway-bulk-workflow`
- `npathway-demo bulk`
- `npathway-demo scrna`

Repository-local scripts:

- `scripts/run_batch_aware_bulk_workflow.py`
- `scripts/run_bulk_dynamic_pathway.py`
- `scripts/run_scrna_pseudobulk_dynamic_pathway.py`

## Mode 2. Comparison Mode

Use this mode when you already have:

- a **full ranked gene table** from DESeq2, dream, limma, edgeR, or another upstream DE engine
- an nPathway dynamic GMT
- a curated GMT

This mode does **not** discover new programs. It exists to answer a narrower and review-friendly question:

> On the **same ranked gene list**, how do curated pathways compare with nPathway dynamic programs?

What comparison mode produces:

- dynamic-program GSEA on the supplied ranking
- curated-pathway GSEA on the supplied ranking
- side-by-side comparison tables
- dynamic-versus-curated overlap tables
- focus-gene membership summaries

Installed command:

- `npathway-compare-gsea`

Repository-local script:

- `scripts/run_curated_vs_dynamic_gsea.py`

## Which mode should you use?

Choose **Discovery Mode** if you still need nPathway to identify the gene programs.

Choose **Comparison Mode** if you already have:

- the ranking you trust
- the dynamic programs you want to defend
- a curated pathway collection to benchmark against

## Recommended product framing

nPathway should not be framed as a preprocessing suite or a universal DE engine.

The strongest framing is:

- upstream normalization / batch correction / DE are user-selectable
- nPathway is the **pathway-specialized layer**
- the system's core value is **dynamic program discovery, pathway grounding, and fair comparison against curated pathways**

## Minimal ranked-gene table format

Comparison mode expects a ranked-gene table with at least:

- `gene`
- `score`

Examples of defensible `score` values:

- DESeq2 Wald statistic
- limma moderated t-statistic
- edgeR quasi-likelihood statistic transformed to a signed ranking
- `sign(logFC) * -log10(p)`

Do not pass only the significant DEGs. Use the **full ranked list**.
