# nPathway scRNA Demo Dataset

This folder contains a small bundled scRNA-seq dataset for validating your installation and running a first end-to-end pseudobulk case/control demo.

## Files

- `demo_scrna_case_ctrl.h5ad`
  - cells x genes AnnData object
  - `adata.obs['donor_id']` defines pseudobulk samples
  - `adata.obs['condition']` defines the `case` vs `control` contrast
  - `adata.obs['cell_type']` is set to `CD4_T` for every cell in this demo
- `demo_scrna_obs_preview.csv`
  - preview of the key `adata.obs` columns
- `scrna_reference_demo.gmt`
  - optional demo reference sets for program annotation

## Important note

This bundled scRNA demo is raw-count compatible and includes `adata.raw`, so the default pseudobulk route works without extra flags.

## First-run commands

Installed command:

```bash
npathway-demo scrna --output-dir results/demo_scrna_case_vs_control
```

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
