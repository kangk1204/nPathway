# nPathway

nPathway discovers **dynamic gene programs** from transcriptomic data, aligns them to curated pathways, and returns reviewer-friendly outputs such as enrichment tables, GMT files, and dashboards.

This is the practical distinction from standard GSEA:

- GSEA asks: does a **predefined** pathway score highly in this ranked gene list?
- nPathway asks: what **data-driven** gene program is present here, how does it map to known pathways, and which relevant genes fall outside the original static gene-set definition?

That makes nPathway useful when you want to show something like:

- a curated Alzheimer's pathway is enriched by GSEA
- the nPathway Alzheimer's-like program overlaps that pathway
- the dynamic program also contains additional disease-linked genes that are biologically relevant but not present in the curated GMT definition

## What You Need To Know First

- nPathway does **not** require a precomputed DEG-only table for the main workflow.
- The official discovery workflows start from an **expression matrix plus metadata**.
- If you already have a full ranked gene table from `DESeq2`, `limma`, `edgeR`, or `dream`, you can reuse it in **comparison mode**.
- The core product is:
  - dynamic gene program discovery
  - pathway alignment against curated references
  - interpretable outputs for review and reporting

## Fastest Start

### Option 1. One-click install

```bash
bash scripts/install_npathway_easy.sh
```

This creates a local virtual environment, installs nPathway, and smoke-checks the installed commands.
The installer auto-prefers `python3.11`, `python3.12`, or `python3.10` when available.

After that:

```bash
source .venv/bin/activate
npathway-demo bulk --output-dir results/demo_bulk_case_vs_control
npathway-demo scrna --output-dir results/demo_scrna_case_vs_control
```

### Option 2. Editable install

Base install:

```bash
pip install -e .
```

Developer install:

```bash
pip install -e ".[dev]"
```

Optional model extras:

```bash
pip install -e ".[scgpt]"
pip install -e ".[geneformer]"
pip install -e ".[scbert]"
pip install -e ".[all-models]"
```

Installed commands:

- `npathway-validate-inputs`
- `npathway-demo`
- `npathway-bulk-workflow`
- `npathway-scrna-easy`
- `npathway-compare-gsea`
- `npathway-convert-seurat`
- `npathway-convert-10x`

## The 3 Main Ways To Use nPathway

### 1. Discovery mode for bulk RNA-seq

Use this when you have a sample-level expression matrix and metadata.

Input files:

- matrix: CSV/TSV
- metadata: CSV/TSV

Supported matrix layouts:

- `genes_by_samples`: first column is gene ID, remaining columns are samples
- `samples_by_genes`: first column is sample ID, remaining columns are genes

Minimum rule:

- Hard minimum: **2 samples per group**
- Recommended: **3 or more samples per group**

Example matrix:

```csv
gene,S1,S2,S3,S4
TREM2,120,110,80,75
BIN1,55,51,90,95
APOE,200,180,130,120
```

Example metadata:

```csv
sample,condition
S1,case
S2,case
S3,control
S4,control
```

Validate first:

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

Run the beginner-friendly bulk workflow:

```bash
npathway-bulk-workflow \
  --matrix data/my_bulk_matrix.csv \
  --metadata data/my_bulk_metadata.csv \
  --sample-col sample \
  --group-col condition \
  --group-a case \
  --group-b control \
  --raw-counts \
  --output-dir results/my_bulk_run
```

Notes:

- use `--raw-counts` for count matrices
- omit `--raw-counts` for already normalized or log-scale compatible matrices
- the bulk workflow can also reuse an external ranked gene table with `--ranked-genes`

### 2. Discovery mode for scRNA-seq

Use this when you have an annotated `AnnData (.h5ad)` object.

Input:

- one `.h5ad` file

Required `adata.obs` fields:

- sample or donor column
- condition or group column

Important rule:

- each donor/sample must map to **exactly one** group label

Official scRNA route:

- `npathway-scrna-easy` for the easiest path
- `scripts/run_scrna_pseudobulk_dynamic_pathway.py` for the expert path
- the official pseudobulk case/control CLI for scRNA-seq runs case/control comparisons through donor-level pseudobulk

Minimum rule after filtering and pseudobulk:

- Hard minimum: **2 retained pseudobulk samples per group**
- Recommended: **3 or more retained donors/samples per group**

Validate first:

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

Run the easy scRNA workflow:

```bash
npathway-scrna-easy \
  --adata data/my_scrna.h5ad \
  --condition-col condition \
  --case case \
  --control control \
  --output-dir results/my_scrna_run
```

If you start from Seurat or raw 10x:

```bash
npathway-convert-seurat --check-only
npathway-convert-10x --check-only
```

### 3. Comparison mode against curated GSEA

Use this when you already have:

- a **full ranked gene table**
- an nPathway dynamic GMT
- a curated GMT

This mode does **not** discover new programs. It compares curated pathways and dynamic programs on the **same ranking**.

This is the cleanest mode for reviewer-facing analyses such as:

- fair `same ranked list` comparisons against GSEA/MSigDB
- comparing nPathway vs curated Alzheimer-like pathways
- testing whether dynamic programs capture additional disease-linked genes outside the static set definition

Run it with:

```bash
npathway-compare-gsea --help
```

Repository-local equivalent:

```bash
python scripts/run_curated_vs_dynamic_gsea.py --help
```

## Quick Demo Paths

### Bulk demo

```bash
npathway-demo bulk --output-dir results/demo_bulk_case_vs_control
```

Bundled demo files:

- `data/bulk_demo_case_vs_ctrl/bulk_matrix_case_ctrl_demo.csv`
- `data/bulk_demo_case_vs_ctrl/bulk_metadata_case_ctrl_demo.csv`
- `data/bulk_demo_case_vs_ctrl/bulk_reference_demo.gmt`

### scRNA demo

```bash
npathway-demo scrna --output-dir results/demo_scrna_case_vs_control
```

Bundled demo files:

- `data/scrna_demo_case_vs_ctrl/demo_scrna_case_ctrl.h5ad`
- `data/scrna_demo_case_vs_ctrl/demo_scrna_obs_preview.csv`
- `data/scrna_demo_case_vs_ctrl/scrna_reference_demo.gmt`

### Input templates

Templates for your own data live in:

- `data/templates/bulk_matrix_template.csv`
- `data/templates/bulk_metadata_template.csv`
- `data/templates/scrna_obs_template.csv`

The 5-minute setup guide is here:

- `docs/quickstart_input_guide.md`

The discovery-vs-comparison guide is here:

- `docs/discovery_vs_comparison_modes.md`

## What nPathway Produces

Typical outputs include:

- `dynamic_programs.gmt`
- `de_results.csv`
- `ranked_genes_for_gsea.csv`
- pathway comparison tables
- dashboard artifacts
- batch-QC figures in the batch-aware workflow
- `figure_ready/` bundles in the easy scRNA workflow

## How nPathway Differs From Standard GSEA

| Question | Standard GSEA | nPathway |
| --- | --- | --- |
| Starting point | Static curated gene sets | Dynamic programs discovered from the data |
| Main question | Is this known pathway enriched? | What program exists here, and how does it align to known pathways? |
| Can it add genes beyond the original pathway definition? | No | Yes |
| Can it reuse a DESeq2/limma/edgeR ranked list? | Yes | Yes |
| Main strength | Simple curated interpretation | Dynamic discovery plus curated grounding |

nPathway should not be framed as a replacement for every upstream statistical method. A safer framing is:

- upstream DE/ranking can come from `DESeq2`, `limma`, `edgeR`, or `dream`
- nPathway adds the dynamic gene program layer and the pathway interpretation layer

## Beginner Checklist

1. Install with `bash scripts/install_npathway_easy.sh` or `pip install -e .`.
2. Run `npathway-demo bulk` once before touching your own data.
3. Match your files to the templates in `data/templates/`.
4. Validate with `npathway-validate-inputs`.
5. Run `npathway-bulk-workflow` or `npathway-scrna-easy`.
6. Use `npathway-compare-gsea` when you want a direct curated-vs-dynamic comparison on the same ranking.

## Advanced Utilities

Public repository utilities kept in this checkout:

- `scripts/run_bulk_dynamic_pathway.py`
- `scripts/run_scrna_pseudobulk_dynamic_pathway.py`
- `scripts/run_batch_aware_bulk_workflow.py`
- `scripts/run_scrna_easy.py`
- `scripts/run_curated_vs_dynamic_gsea.py`
- `scripts/validate_npathway_inputs.py`
- `scripts/convert_seurat_to_h5ad.py`
- `scripts/convert_10x_to_h5ad.py`

Private manuscript materials, submission figures, and large generated results are intentionally kept out of the public GitHub snapshot.

Maintainer note:

- before pushing a public snapshot, run `bash scripts/prepare_public_github_snapshot.sh --dry-run`

## License

MIT
