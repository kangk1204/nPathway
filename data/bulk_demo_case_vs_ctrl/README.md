# nPathway Bulk Demo Dataset

This folder contains a small bundled dataset for validating your installation and running a first end-to-end bulk demo.

## Files

- `bulk_matrix_case_ctrl_demo.csv`
  - expression matrix
  - first column is `gene`
  - remaining columns are samples `S1`-`S12`
- `bulk_metadata_case_ctrl_demo.csv`
  - one row per sample
  - `sample` column matches the matrix sample IDs
  - `condition` defines the `case` vs `control` contrast
- `bulk_reference_demo.gmt`
  - optional demo reference sets for program annotation

## Important note

The bundled demo matrix is already normalized / log-scale compatible.

- Do not use `--raw-counts` with this demo dataset.

## First-run commands

Validate and create an HTML report:

```bash
python scripts/validate_npathway_inputs.py bulk \
  --matrix data/bulk_demo_case_vs_ctrl/bulk_matrix_case_ctrl_demo.csv \
  --metadata data/bulk_demo_case_vs_ctrl/bulk_metadata_case_ctrl_demo.csv \
  --sample-col sample \
  --group-col condition \
  --group-a case \
  --group-b control \
  --html-out results/demo_bulk_validation.html
```

Run the bulk dynamic workflow:

```bash
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
