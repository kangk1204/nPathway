# nPathway 5-Minute Input Guide

This guide is for users who want to answer one question first:

> "Is my input data shaped correctly for nPathway?"

## 1. Pick the right mode

### Discovery mode

Use this when nPathway still needs to discover the dynamic gene programs from expression data.

### Bulk RNA-seq

Use this when you already have a sample-level expression matrix and a metadata table.

- Input 1: matrix CSV/TSV
- Input 2: metadata CSV/TSV
- Use: `scripts/run_bulk_dynamic_pathway.py`

### scRNA-seq

Use this when you have a `.h5ad` file and want a case/control comparison.

- Input: `.h5ad` AnnData
- Required `adata.obs` columns:
  - sample / donor ID
  - group / condition label
- Recommended route: pseudobulk by sample, optionally within one cell type
- Easiest use: `npathway-scrna-easy`
- Expert use: `scripts/run_scrna_pseudobulk_dynamic_pathway.py`
- If you start from Seurat `.rds`, convert it first with `npathway-convert-seurat`
- If you start from raw 10x output, convert it first with `npathway-convert-10x`

### Comparison mode

Use this when you already have:

- a **full ranked gene list**
- an nPathway `dynamic_programs.gmt`
- a curated GMT

Installed command:

- `npathway-compare-gsea`

Repository-local script:

- `scripts/run_curated_vs_dynamic_gsea.py`

## 2. Start from the templates

Template files live in `data/templates/`.

- `bulk_matrix_template.csv`
- `bulk_metadata_template.csv`
- `scrna_obs_template.csv`

These are not example datasets to analyze directly. They are shape templates so you can match your own column names and layout before running the pipeline.

If you want to confirm that your environment is working before touching your own data, use one of the bundled demo datasets:

- bulk: `data/bulk_demo_case_vs_ctrl/`
- scRNA: `data/scrna_demo_case_vs_ctrl/`

If you installed the package with `pip install -e .`, you can also run:

```bash
npathway-demo bulk
npathway-demo scrna
```

If you want the simplest local setup path from a fresh checkout:

```bash
bash scripts/install_npathway_easy.sh
```

## 3. Validate before you run

### Validate bulk input

```bash
npathway-validate-inputs bulk \
  --matrix data/my_bulk_matrix.csv \
  --metadata data/my_bulk_metadata.csv \
  --sample-col sample \
  --group-col condition \
  --group-a case \
  --group-b control \
  --html-out results/my_bulk_validation.html
```

What this checks:

- matrix layout
- metadata columns
- overlapping sample IDs
- group sizes
- non-numeric matrix values

If validation succeeds, the HTML report is a portable handoff artifact you can attach to issue reports or review notes.

### Validate scRNA input

```bash
npathway-validate-inputs scrna \
  --adata data/my_scrna.h5ad \
  --sample-col donor_id \
  --group-col condition \
  --group-a case \
  --group-b control \
  --subset-col cell_type \
  --subset-value CD4_T
```

What this checks:

- required `adata.obs` columns
- one group label per donor/sample
- retained pseudobulk sample counts after `--min-cells-per-sample`
- whether the requested contrast is actually analyzable

## 4. Minimum sample rules

### Bulk

- Hard minimum: **2 samples per group**
- Recommended: **3 or more per group**

### scRNA pseudobulk

- Hard minimum: **2 retained pseudobulk samples per group**
- Recommended: **3 or more donors/samples per group**

## 5. Run the analysis

### Fastest first run with bundled demo data

```bash
npathway-demo bulk --output-dir results/demo_bulk_case_vs_control
```

Repository-local equivalent:

```bash
python scripts/validate_npathway_inputs.py bulk \
  --matrix data/bulk_demo_case_vs_ctrl/bulk_matrix_case_ctrl_demo.csv \
  --metadata data/bulk_demo_case_vs_ctrl/bulk_metadata_case_ctrl_demo.csv \
  --sample-col sample \
  --group-col condition \
  --group-a case \
  --group-b control \
  --html-out results/demo_bulk_validation.html

python scripts/run_bulk_dynamic_pathway.py \
  --matrix data/bulk_demo_case_vs_ctrl/bulk_matrix_case_ctrl_demo.csv \
  --metadata data/bulk_demo_case_vs_ctrl/bulk_metadata_case_ctrl_demo.csv \
  --sample-col sample \
  --group-col condition \
  --group-a case \
  --group-b control \
  --n-programs 8 \
  --n-components 8 \
  --gsea-n-perm 50 \
  --annotate-programs \
  --annotation-gmt data/bulk_demo_case_vs_ctrl/bulk_reference_demo.gmt \
  --output-dir results/demo_bulk_case_vs_control
```

The bundled demo matrix is already normalized/log-scale compatible, so do not add `--raw-counts` for that dataset.

### Fastest scRNA first run with bundled demo data

```bash
npathway-demo scrna --output-dir results/demo_scrna_case_vs_control
```

### Fastest beginner scRNA run with auto-detection

```bash
npathway-scrna-easy \
  --adata data/my_scrna.h5ad \
  --condition-col condition \
  --case case \
  --control control \
  --output-dir results/my_scrna_easy_run
```

What this mode adds on top of the older pseudobulk CLI:

- auto-detection of sample / donor, cell type, batch, and common covariate columns
- preflight HTML report before execution
- automatic cell-type eligibility checks
- top-level HTML index linking all completed cell-type runs
- automatic use of the batch-aware backend when it is available
- guarded SVA in `auto` mode when the `sva` R package is installed and the design is large enough
- batch-QC artifacts with before/after PCA and correlation heatmaps on donor-level pseudobulk samples
- a `figure_ready/` package that copies the main figures, comparison tables, and batch-QC panels into one place

Important strategy note:

- For scRNA-seq, nPathway does **not** use Harmony-like integration as the ranking engine. The default route is donor-level pseudobulk plus known batch/covariates and optional guarded SVA. Harmony/BBKNN/scVI remain useful upstream for exploratory embeddings, not as the core pathway-ranking engine for nPathway.

### Seurat `.rds` to `.h5ad`

```bash
npathway-convert-seurat \
  --seurat-rds data/my_seurat_object.rds \
  --output-h5ad data/my_scrna.h5ad
```

If you only want to check whether your machine is ready for this conversion:

```bash
npathway-convert-seurat --check-only
```

### Raw 10x to `.h5ad`

Single 10x input:

```bash
npathway-convert-10x \
  --matrix-dir data/sample_A_10x/ \
  --sample-id sample_A \
  --output-h5ad data/sample_A.h5ad
```

Multiple 10x inputs:

```bash
npathway-convert-10x \
  --manifest data/tenx_manifest.csv \
  --output-h5ad data/combined_10x.h5ad
```

The manifest should contain:

- `input_path`
- `sample_id`
- any sample-level metadata columns you want copied into `adata.obs`

Repository-local equivalent:

```bash
python scripts/validate_npathway_inputs.py scrna \
  --adata data/scrna_demo_case_vs_ctrl/demo_scrna_case_ctrl.h5ad \
  --sample-col donor_id \
  --group-col condition \
  --group-a case \
  --group-b control \
  --subset-col cell_type \
  --subset-value CD4_T \
  --html-out results/demo_scrna_validation.html

python scripts/run_scrna_pseudobulk_dynamic_pathway.py \
  --adata data/scrna_demo_case_vs_ctrl/demo_scrna_case_ctrl.h5ad \
  --sample-col donor_id \
  --group-col condition \
  --group-a case \
  --group-b control \
  --subset-col cell_type \
  --subset-value CD4_T \
  --min-cells-per-sample 10 \
  --n-programs 4 \
  --n-components 3 \
  --gsea-n-perm 50 \
  --annotate-programs \
  --annotation-gmt data/scrna_demo_case_vs_ctrl/scrna_reference_demo.gmt \
  --output-dir results/demo_scrna_case_vs_control
```

### Bulk

```bash
python scripts/run_bulk_dynamic_pathway.py \
  --matrix data/my_bulk_matrix.csv \
  --metadata data/my_bulk_metadata.csv \
  --sample-col sample \
  --group-col condition \
  --group-a case \
  --group-b control \
  --raw-counts \
  --output-dir results/my_bulk_run
```

### Batch-aware bulk discovery

```bash
npathway-bulk-workflow \
  --matrix data/my_bulk_counts.csv \
  --metadata data/my_bulk_metadata.csv \
  --sample-col sample \
  --group-col condition \
  --group-a case \
  --group-b control \
  --raw-counts \
  --batch-col batch \
  --covariate-cols age,sex,RIN \
  --curated-gmt data/my_curated_pathways.gmt \
  --focus-genes TREM2,TYROBP,BIN1,PICALM \
  --output-dir results/my_bulk_batch_run
```

### scRNA pseudobulk

```bash
python scripts/run_scrna_pseudobulk_dynamic_pathway.py \
  --adata data/my_scrna.h5ad \
  --sample-col donor_id \
  --group-col condition \
  --group-a case \
  --group-b control \
  --subset-col cell_type \
  --subset-value CD4_T \
  --min-cells-per-sample 20 \
  --output-dir results/my_scrna_run
```

### scRNA preflight-only mode

```bash
npathway-scrna-easy \
  --adata data/my_scrna.h5ad \
  --condition-col condition \
  --case case \
  --control control \
  --wizard-only \
  --output-dir results/my_scrna_preflight
```

### Comparison-only mode

```bash
npathway-compare-gsea \
  --ranked-genes results/my_bulk_run/ranked_genes_for_gsea.csv \
  --dynamic-gmt results/my_bulk_run/dynamic_programs.gmt \
  --curated-gmt data/reference/alzheimer_msigdb_curated_20260306.gmt \
  --focus-genes TREM2,TYROBP,BIN1,PICALM,CLU,ABCA7 \
  --output-dir results/my_bulk_run/comparison
```

## 6. Common failure modes

### "No overlapping sample IDs"

- Matrix sample names and metadata sample names do not match exactly.

### "Each group must have at least 2 samples"

- One contrast group is undersized after filtering.

### "Each pseudobulk sample must map to exactly one group label"

- The same donor/sample appears under multiple conditions in `adata.obs`.

### "No cells matched subset"

- The requested cell type label does not exist in `adata.obs[subset_col]`.

## 7. What you get back

After a successful discovery-mode run, nPathway writes:

- dynamic gene programs
- pathway matching tables
- context-aware gene rankings
- claim-gated enrichment outputs
- an optional interactive dashboard

After a successful comparison-mode run, nPathway writes:

- `dynamic_gsea.csv`
- `curated_gsea.csv`
- `gsea_comparison_combined.csv`
- `dynamic_curated_overlap.csv`
- `focus_gene_membership.csv`
- `summary.md`
