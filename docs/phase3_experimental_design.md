# Phase 3: Experimental Design

## nPathway -- Foundation Model-Derived Gene Programs for Context-Aware Gene Set Enrichment Analysis

---

## Overview

This document specifies the experimental design for five benchmarks evaluating nPathway-derived gene programs against curated pathway databases and alternative data-driven methods. Each benchmark is designed to assess a specific dimension of gene program quality. Together, they provide a comprehensive evaluation that addresses all four research questions from Phase 1.

### Baselines

All benchmarks compare nPathway gene programs against the following baselines:

| Baseline | Source | Description |
|----------|--------|-------------|
| KEGG | MSigDB C2:CP:KEGG | Curated metabolic and signalling pathways |
| Reactome | MSigDB C2:CP:REACTOME | Curated molecular pathways |
| GO BP | MSigDB C5:BP | Gene Ontology Biological Process terms (15--500 genes) |
| Hallmark | MSigDB H | 50 well-defined biological states/processes |
| WGCNA modules | Computed per dataset | Weighted gene co-expression network analysis |
| cNMF programs | Computed per dataset | Consensus NMF gene expression programs |
| SCENIC regulons | Computed per dataset | SCENIC transcription factor regulons |

### nPathway Variants

We evaluate multiple nPathway configurations:

| Variant | Embedding source | Discovery method |
|---------|-----------------|------------------|
| nPathway-scGPT-Leiden | scGPT (universal) | Leiden clustering |
| nPathway-scGPT-ETM | scGPT (universal) | Embedded Topic Model |
| nPathway-scGPT-Attn | scGPT (attention matrices) | Attention network |
| nPathway-GF-Leiden | Geneformer (universal) | Leiden clustering |
| nPathway-GF-ETM | Geneformer (universal) | Embedded Topic Model |
| nPathway-scBERT-Leiden | scBERT (universal) | Leiden clustering |
| nPathway-Ensemble | All models | Ensemble consensus |
| nPathway-Context-* | Context-specific embeddings | Leiden/ETM per cell type |

---

## Benchmark 1: Recovery of Known Biology

### Objective

Evaluate whether nPathway gene programs correctly identify biological processes known to be perturbed in controlled experimental systems. This is a sanity check demonstrating that the learned programs capture genuine biology.

### Datasets

| Dataset | Description | Perturbations | Source |
|---------|-------------|---------------|--------|
| Replogle et al. 2022 | Genome-scale Perturb-seq in K562 cells | ~5,000 single-gene CRISPRi knockdowns | GEO: GSE169314 |
| Adamson et al. 2016 | CRISPRi screen with scRNA-seq readout | 9 target genes + controls | GEO: GSE90063 |
| Norman et al. 2019 | Combinatorial CRISPR screen | 112 pairwise perturbations | GEO: GSE133344 |
| sci-Plex (Srivatsan et al. 2020) | Drug response profiling | 188 compounds at 4 doses | GEO: GSE139944 |

### Experimental Procedure

```
For each dataset D and each perturbation p:
  1. Compute differential expression: perturbed vs. control cells
     - Method: Wilcoxon rank-sum test (Scanpy rank_genes_groups)
     - Output: ranked gene list by -log10(p) * sign(logFC)

  2. Define ground truth:
     - For CRISPRi: expected pathway = curated pathway(s) containing the
       target gene (from KEGG, Reactome, or known biology)
     - For drugs: expected pathway = known drug target pathway (from
       DGIdb, DrugBank, or CMap annotations)

  3. Run enrichment analysis with each gene set collection:
     - Preranked GSEA (1000 permutations, weighted_score_type=1)
     - Using: nPathway programs, KEGG, Reactome, Hallmark, GO BP,
       WGCNA modules, cNMF programs, SCENIC regulons

  4. Evaluate recovery:
     - For each perturbation, check whether the expected ground-truth
       pathway is among the significant results (FDR < 0.05)
     - Record rank of the ground-truth pathway among all tested sets
```

### Metrics

| Metric | Definition | Interpretation |
|--------|-----------|----------------|
| Recovery rate | Fraction of perturbations where the ground-truth pathway is significant (FDR < 0.05) | Higher = better sensitivity |
| Mean reciprocal rank (MRR) | 1/K averaged over perturbations, where K = rank of ground-truth pathway | Higher = ground truth ranked higher |
| Precision@K | Fraction of top-K enriched programs that overlap with ground truth | Higher = fewer false positives |
| AUROC | Area under ROC for classifying true vs. false pathway associations | Higher = better discrimination |
| AUPRC | Area under precision-recall curve | Higher = better for imbalanced classes |

### Statistical Tests

- McNemar's test for paired comparison of recovery rates between nPathway and each baseline.
- Paired Wilcoxon signed-rank test for MRR and AUROC differences.
- Bonferroni correction for multiple baseline comparisons.

### Expected Figures

**Figure 3a:** Bar chart comparing recovery rates across gene set collections for the Replogle Perturb-seq dataset. X-axis: gene set collection. Y-axis: fraction of perturbations where ground-truth pathway recovered (FDR < 0.05). Error bars: 95% bootstrap confidence intervals.

**Figure 3b:** ROC curves (sensitivity vs. 1-specificity) for each gene set collection, aggregated across all perturbations. One curve per collection.

**Figure 3c:** Heatmap showing per-perturbation enrichment significance (-log10 FDR) for a subset of 50 representative perturbations (rows) across the top 30 programs from nPathway and KEGG (columns).

**Figure 3d:** Scatter plot of MRR (nPathway) vs. MRR (best curated baseline) for each perturbation. Points above the diagonal indicate nPathway outperformance.

**Supplementary Figure S1:** Same analysis for sci-Plex drug perturbations.

---

## Benchmark 2: Novel Discovery Potential

### Objective

Identify gene programs enriched in disease states that are not captured by any existing curated pathway, and validate their biological coherence through orthogonal evidence.

### Datasets

| Dataset | Description | Conditions | Source |
|---------|-------------|------------|--------|
| COVID-19 PBMC atlas (Wilk et al. 2020) | scRNA-seq of PBMCs from COVID-19 patients | Healthy, Mild, Severe | GEO: GSE150728 |
| TCGA pan-cancer (bulk + deconvolved) | 33 cancer types, >10,000 samples | Tumour vs. normal | TCGA Data Portal |
| Alzheimer's snRNA-seq (Mathys et al. 2019) | Prefrontal cortex snRNA-seq | AD pathology vs. no pathology | Synapse: syn18485175 |

### Experimental Procedure

```
For each dataset D and each disease comparison:
  1. Compute differential expression: disease vs. control

  2. Run enrichment analysis with nPathway programs

  3. Identify novel programs:
     novel_programs = {p : max_Jaccard(p, all_curated) < 0.1 AND FDR < 0.05}

  4. Validate each novel program via:
     a. PPI enrichment:
        - Map program genes to STRING database (v12.0)
        - Test whether the number of intra-program PPI edges exceeds
          expectation for a random gene set of the same size
        - Metric: PPI enrichment p-value (hypergeometric), connectivity ratio
     b. Regulatory evidence:
        - Query ENCODE and ChIP-Atlas for shared TF binding among
          program genes
        - Test for enrichment of shared regulatory elements
     c. Co-expression validation:
        - In an independent dataset (not used for embedding extraction),
          compute mean pairwise Pearson correlation among program genes
        - Compare to correlation among random gene sets of matched size
     d. Literature mining:
        - For each program, query PubMed for co-occurrence of gene pairs
          in abstracts
        - Compute literature co-citation score
     e. Independent cohort replication:
        - Apply nPathway to an independent dataset of the same disease
        - Measure Jaccard similarity of novel programs across cohorts
```

### Metrics

| Metric | Definition |
|--------|-----------|
| Novelty score | Fraction of program genes absent from all curated pathways |
| PPI connectivity ratio | Actual / expected intra-program PPI edges |
| Mean co-expression | Mean Pearson correlation among program gene pairs in independent data |
| TF enrichment | Number of TFs whose targets are significantly enriched in the program |
| Replication Jaccard | Jaccard similarity of the same novel program across independent cohorts |

### Statistical Tests

- Permutation test (10,000 permutations) for PPI connectivity: compare observed intra-program edges to distribution under random gene sets of matched size.
- Empirical p-value for co-expression: compare mean pairwise correlation to null distribution from 10,000 random gene sets.
- Hypergeometric test for TF target enrichment.

### Expected Figures

**Figure 5a:** Volcano plot of enrichment results for COVID-19 severe vs. healthy, with novel programs highlighted in colour and curated-matching programs in grey.

**Figure 5b:** Network visualisation of a selected novel program showing PPI connections (STRING) among member genes, with node colour indicating expression fold-change.

**Figure 5c:** Bar chart of PPI connectivity ratios for top 20 novel programs vs. 20 randomly selected curated pathways, with significance thresholds marked.

**Figure 5d:** Heatmap of program-program Jaccard similarity between novel programs discovered in COVID-19 cohort 1 and cohort 2 (replication analysis).

---

## Benchmark 3: Context-Specificity

### Objective

Demonstrate that nPathway gene programs reorganise across biological contexts (cell types, tissues), reflecting genuine context-dependent regulatory rewiring, in contrast to static curated pathways.

### Datasets

| Dataset | Description | Contexts | Source |
|---------|-------------|----------|--------|
| Tabula Sapiens (Tabula Sapiens Consortium, 2022) | Multi-tissue scRNA-seq atlas | 24 tissues, >400 cell types | CZI CELLxGENE |
| Human Cell Atlas PBMC | PBMC reference | 8 major cell types | HCA Data Portal |
| GTEx (bulk, deconvolved) | Multi-tissue bulk RNA-seq | 54 tissues | GTEx Portal |

### Experimental Procedure

```
1. Extract context-specific embeddings:
   For each cell type t in the Tabula Sapiens atlas:
     E_t = extractor.extract_context_embeddings(adata, cell_type_key="cell_type")[t]

2. Discover context-specific programs:
   For each cell type t:
     P_t = ClusteringProgramDiscovery(method="leiden").fit(E_t, gene_names).get_programs()

3. Measure program divergence across contexts:
   For each pair of cell types (t1, t2):
     overlap = compute_overlap_matrix(P_t1, P_t2)
     best_match_similarity = max over columns for each row
     mean_divergence = 1 - mean(best_match_similarity)

4. Identify context-switching genes:
   For each gene g:
     Record which program it belongs to in each cell type
     If g changes program membership across cell types:
       g is a "context-switching gene"
   Compute: fraction of genes that switch programs

5. Validate context-switching genes:
   - Test enrichment for known tissue-specific TFs
   - Test enrichment for genes with tissue-specific expression (GTEx)
   - Test enrichment for genes with cell-type-specific eQTLs

6. Compare with curated pathways:
   For each curated pathway p in KEGG:
     p is identical across all cell types (by definition)
     Compute the actual expression correlation among p's genes
     in each cell type -> measure context-dependent coherence
```

### Metrics

| Metric | Definition |
|--------|-----------|
| Program divergence | 1 - mean best-match Jaccard between programs of two cell types |
| Context-switching fraction | Fraction of genes changing program membership across cell types |
| Context-dependent coherence | Mean pairwise expression correlation of program genes within specific cell types |
| TF specificity enrichment | Enrichment of known cell-type-specific TFs among hub genes in context-specific programs |

### Statistical Tests

- Permutation test for program divergence: shuffle cell type labels and recompute divergence to establish a null distribution.
- Fisher's exact test for enrichment of tissue-specific TFs among context-switching genes.
- Paired t-test comparing context-dependent coherence of nPathway programs vs. curated pathways within each cell type.

### Expected Figures

**Figure 4a:** Heatmap of program divergence (1 - best-match Jaccard) across 10 selected cell types from Tabula Sapiens. Rows and columns are cell types; colour intensity indicates divergence.

**Figure 4b:** Alluvial/Sankey diagram showing how genes move between programs across 3 cell types (e.g., T cells, hepatocytes, neurons). Width of flows proportional to number of switching genes.

**Figure 4c:** Box plot comparing context-dependent coherence (mean pairwise correlation of program genes within each cell type) between nPathway context-specific programs and KEGG pathways. Separate boxes for each cell type.

**Figure 4d:** UMAP of gene embeddings coloured by program membership, shown side-by-side for T cells and hepatocytes, illustrating reorganisation of the embedding landscape.

---

## Benchmark 4: Statistical Power in Enrichment Analysis

### Objective

Compare the sensitivity and specificity of enrichment analysis using nPathway gene programs vs. curated pathways, under controlled conditions with known ground truth.

### 4.1 Simulation Study

```
Algorithm: Statistical Power Simulation
Input: Real expression dataset D (background), curated pathway P_true,
       effect sizes S = {0.1, 0.2, 0.5, 1.0, 2.0} (log2 fold-change)

For each effect size s in S:
  For each replicate r = 1, ..., 100:
    1. Sample n_case=50 and n_control=50 cells from D
    2. Spike in signal: for genes in P_true in case cells,
       add s * std(gene) to expression values
    3. Compute differential expression (Wilcoxon rank-sum)
    4. Generate ranked gene list
    5. Run GSEA with each gene set collection:
       - nPathway programs (all variants)
       - KEGG, Reactome, Hallmark, GO BP
    6. Record:
       - Whether any program significantly overlapping P_true
         is detected (FDR < 0.05) -> true positive
       - Whether any non-overlapping program is falsely detected
         (FDR < 0.05) -> false positive
    7. Compute TPR and FPR for this replicate

Aggregate across replicates:
  - Compute mean TPR and FPR at each effect size
  - Compute AUROC and AUPRC
```

### 4.2 Real Data Comparison

```
For each case-control dataset:
  1. Compute differential expression
  2. Run GSEA with all gene set collections
  3. For each collection, record:
     a. Number of significant programs (FDR < 0.05)
     b. Biological coherence of significant programs
        (PPI connectivity, GO enrichment depth)
     c. Redundancy among significant programs
        (mean pairwise Jaccard among top 20 programs)
     d. Leading edge overlap:
        Fraction of top DE genes that appear in any
        significant program's leading edge
```

### Datasets for Real Data Comparison

| Dataset | Comparison | Source |
|---------|-----------|--------|
| PBMC COVID-19 | Severe vs. Healthy | GEO: GSE150728 |
| IPF lung (Habermann et al. 2020) | IPF vs. Control | GEO: GSE135893 |
| AML (van Galen et al. 2019) | AML vs. normal BM | GEO: GSE116256 |

### Metrics

| Metric | Definition | Interpretation |
|--------|-----------|----------------|
| Sensitivity (TPR) | TP / (TP + FN) at FDR < 0.05 | Fraction of true signals detected |
| Specificity (1 - FPR) | TN / (TN + FP) | Fraction of null signals correctly rejected |
| AUROC | Area under ROC across effect sizes | Overall discrimination |
| AUPRC | Area under precision-recall curve | Performance under class imbalance |
| Effective redundancy | Mean Jaccard among top-K significant programs | Lower = more informative results |
| Leading edge coverage | Fraction of top DE genes in any leading edge | Higher = better gene-level interpretability |

### Statistical Tests

- DeLong test for comparing AUROCs between nPathway and each baseline.
- Paired Wilcoxon signed-rank test for TPR differences at fixed effect sizes.
- Bootstrap confidence intervals (1000 resamples) for all metrics.

### Expected Figures

**Figure S2a:** Power curves: TPR vs. effect size for each gene set collection. Separate panels for each spiked-in pathway. Shaded regions: 95% CI across simulation replicates.

**Figure S2b:** ROC curves (aggregated across effect sizes) for each collection.

**Figure S2c:** Bar chart comparing number of significant pathways (FDR < 0.05), redundancy (mean Jaccard), and leading edge coverage for COVID-19 real data comparison.

**Figure S2d:** Scatter plot: biological coherence (PPI connectivity) vs. statistical significance (-log10 FDR) for all significant programs across all collections, colour-coded by collection.

---

## Benchmark 5: Cross-Model Robustness

### Objective

Assess whether gene programs derived from different foundation models (scGPT, Geneformer, scBERT) converge on the same biological structure, and quantify the added value of multi-model ensemble programs.

### Experimental Procedure

```
1. Extract gene embeddings from each of 3 models:
   E_scgpt = ScGPTExtractor.extract(adata)
   E_gf = GeneformerExtractor.extract(adata)
   E_scbert = ScBERTExtractor.extract(adata)

2. Discover programs from each model independently:
   P_scgpt = ClusteringProgramDiscovery("leiden").fit(E_scgpt)
   P_gf = ClusteringProgramDiscovery("leiden").fit(E_gf)
   P_scbert = ClusteringProgramDiscovery("leiden").fit(E_scbert)

3. Compute pairwise consistency:
   For each pair of models (A, B):
     a. Align gene universes to common genes
     b. Convert programs to label vectors:
        L_A = programs_to_labels(P_A, common_genes)
        L_B = programs_to_labels(P_B, common_genes)
     c. Compute ARI(L_A, L_B)
     d. Compute NMI(L_A, L_B)
     e. Compute best-match Jaccard matrix:
        J = compute_overlap_matrix(P_A, P_B)
        mean_best_match = mean(max(J, axis=1))

4. Identify consensus and model-specific programs:
   consensus = EnsembleProgramDiscovery(
       [P_scgpt, P_gf, P_scbert],
       consensus_method="leiden"
   ).fit(E_ensemble, gene_names)

   For each model M:
     specific_M = programs in P_M with max Jaccard < 0.2
                  with any program from other models

5. Evaluate consensus vs. model-specific programs:
   a. Enrichment analysis on perturbation data:
      Compare recovery rates of consensus programs vs. individual
      model programs vs. model-specific programs
   b. Biological coherence:
      Compare PPI connectivity and GO enrichment of consensus vs.
      model-specific programs
   c. Stability:
      Subsample 80% of cells, re-extract embeddings, re-discover
      programs, and measure Jaccard stability with full-data programs
      (repeat 10 times)
```

### Datasets

| Dataset | Purpose |
|---------|---------|
| PBMC 10X (Zheng et al. 2017) | Primary dataset for cross-model comparison |
| Tabula Muris | Replication in a different organism |
| Replogle Perturb-seq | Evaluation of consensus programs on functional benchmark |

### Metrics

| Metric | Definition | Expected range |
|--------|-----------|----------------|
| Adjusted Rand Index (ARI) | Agreement between two clustering solutions, adjusted for chance | [-1, 1]; >0.3 = moderate; >0.6 = strong |
| Normalised Mutual Information (NMI) | Information-theoretic agreement | [0, 1]; >0.3 = moderate; >0.6 = strong |
| Mean best-match Jaccard | For each program in A, max Jaccard with any program in B; averaged | [0, 1]; >0.3 = substantial overlap |
| Consensus fraction | Fraction of genes in consensus programs out of total assigned genes | [0, 1]; higher = more agreement |
| Stability (subsampling) | Mean Jaccard between full-data and subsampled programs | [0, 1]; >0.7 = highly stable |

### Statistical Tests

- Permutation test for ARI/NMI: shuffle gene labels to establish null distributions.
- Bootstrap confidence intervals for mean best-match Jaccard.
- Paired comparison of consensus vs. individual model recovery rates using McNemar's test.

### Expected Figures

**Figure 6a:** Heatmap of pairwise ARI and NMI between all model pairs (scGPT-GF, scGPT-scBERT, GF-scBERT) for each discovery method.

**Figure 6b:** Best-match Jaccard matrix (programs from model A as rows, programs from model B as columns) shown as a clustered heatmap with dendrogram.

**Figure 6c:** Venn diagram showing the overlap of genes covered by consensus programs vs. scGPT-specific, Geneformer-specific, and scBERT-specific programs.

**Figure 6d:** Bar chart comparing perturbation recovery rate (Benchmark 1 metric) for consensus programs, individual model programs, and curated pathways.

**Figure 6e:** Stability analysis: box plot of subsampling Jaccard across 10 resamples for each model and for consensus programs.

---

## Summary of All Expected Figures and Tables

### Main Figures

| Figure | Title | Benchmark |
|--------|-------|-----------|
| Fig 1 | Overview schematic and gene embedding landscape | Methods overview |
| Fig 2 | Gene program discovery and characterisation | Methods + characterisation |
| Fig 3 | Perturbation recovery benchmark | Benchmark 1 |
| Fig 4 | Context-specificity demonstration | Benchmark 3 |
| Fig 5 | Novel biology discovery (COVID-19 case study) | Benchmark 2 |
| Fig 6 | Cross-model consistency analysis | Benchmark 5 |

### Supplementary Figures

| Figure | Title | Benchmark |
|--------|-------|-----------|
| Fig S1 | Drug perturbation recovery (sci-Plex) | Benchmark 1 |
| Fig S2 | Statistical power analysis | Benchmark 4 |
| Fig S3 | Discovery method comparison (Leiden vs. ETM vs. Attention vs. Ensemble) | Methods |
| Fig S4 | Hyperparameter sensitivity analysis (resolution, K, k_neighbors) | Methods |
| Fig S5 | Layer selection analysis (embedding quality vs. transformer layer) | Methods |

### Tables

| Table | Content |
|-------|---------|
| Table 1 | Summary statistics of discovered gene programs (size, coverage, redundancy, novelty) |
| Table 2 | Benchmark 1 results: recovery rates and MRR across all collections |
| Table 3 | Benchmark 4 results: AUROC and TPR at fixed FPR across collections |
| Table 4 | Benchmark 5 results: cross-model consistency metrics |
| Table S1 | Full gene program catalogue with functional annotations |
| Table S2 | Novel programs with validation evidence |

---

## Potential Pitfalls and Mitigations

| Pitfall | Mitigation |
|---------|-----------|
| Foundation model vocabulary may not cover all genes in the dataset | Report coverage statistics; restrict analysis to shared genes |
| Gene embedding quality may vary across models | Compare multiple models; use ensemble as default |
| Choice of clustering resolution heavily influences program number and size | Systematic resolution sweep with stability analysis |
| Curated pathway ground truth is itself incomplete | Use perturbation data with mechanistic ground truth; acknowledge limitation |
| Computational cost of embedding extraction for large datasets | Batch processing; cell subsampling with convergence analysis |
| Overfitting of ETM to small expression datasets | Use synthetic BOW from embeddings; cross-validation |
| Permutation-based GSEA p-values have limited resolution | Use >=10,000 permutations for final results; report exact permutation count |

---

## References

1. Habermann, A. C., et al. (2020). Single-cell RNA sequencing reveals profibrotic roles of distinct epithelial and mesenchymal lineages in pulmonary fibrosis. *Science Advances*, 6(28), eaba1972.
2. Mathys, H., et al. (2019). Single-cell transcriptomic analysis of Alzheimer's disease. *Nature*, 570(7761), 332--337.
3. Norman, T. M., et al. (2019). Exploring genetic interaction manifolds constructed from rich single-cell phenotypes. *Science*, 365(6455), 786--793.
4. Replogle, J. M., et al. (2022). Mapping information-rich genotype-phenotype landscapes with genome-scale Perturb-seq. *Cell*, 185(14), 2559--2575.
5. Srivatsan, S. R., et al. (2020). Massively multiplex chemical transcriptomics at single-cell resolution. *Science*, 367(6473), 45--51.
6. Tabula Sapiens Consortium. (2022). The Tabula Sapiens: a multiple-organ, single-cell transcriptomic atlas of humans. *Science*, 376(6594), eabl4896.
7. van Galen, P., et al. (2019). Single-cell RNA-seq reveals AML hierarchies relevant to disease progression and immunity. *Cell*, 176(6), 1265--1281.
8. Wilk, A. J., et al. (2020). A single-cell atlas of the peripheral immune response in patients with severe COVID-19. *Nature Medicine*, 26(7), 1070--1076.
9. Zheng, G. X., et al. (2017). Massively parallel digital transcriptional profiling of single cells. *Nature Communications*, 8, 14049.
