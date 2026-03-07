# nPathway Input Templates

These files are minimal starting points for preparing user data before running nPathway.

If you want a ready-to-run example instead of a blank template, use:

- `data/bulk_demo_case_vs_ctrl/`
- `data/scrna_demo_case_vs_ctrl/`

If you installed the package with `pip install -e .`, you can also use:

- `npathway-validate-inputs`
- `npathway-demo bulk`
- `npathway-demo scrna`

## Files

- `bulk_matrix_template.csv`
  - Default bulk layout for `--matrix-orientation genes_by_samples`
  - First column is gene ID, remaining columns are samples
- `bulk_metadata_template.csv`
  - One row per sample
  - Includes the default `sample` and `condition` columns used by the CLI examples
- `scrna_obs_template.csv`
  - Example schema for the key `adata.obs` fields used by the scRNA pseudobulk workflow
  - The actual input is still a `.h5ad` file, not a CSV

## Recommended preparation flow

1. Copy the closest template and rename columns only if necessary.
2. Run the validator before analysis:

```bash
npathway-validate-inputs bulk \
  --matrix data/my_matrix.csv \
  --metadata data/my_metadata.csv \
  --group-col condition \
  --group-a case \
  --group-b control \
  --html-out results/my_bulk_validation.html
```

```bash
npathway-validate-inputs scrna \
  --adata data/my_data.h5ad \
  --sample-col donor_id \
  --group-col condition \
  --group-a case \
  --group-b control \
  --subset-col cell_type \
  --subset-value CD4_T
```
