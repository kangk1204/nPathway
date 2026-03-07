# Phase 1: Literature Review and Research Design

## nPathway -- Foundation Model-Derived Gene Programs for Context-Aware Gene Set Enrichment Analysis

---

## 1. Limitations of Curated Pathway Databases

### 1.1 Annotation Bias

Curated pathway databases -- KEGG (Kanehisa et al., 2023), Reactome (Gillespie et al., 2022), Gene Ontology (Gene Ontology Consortium, 2021), and MSigDB (Liberzon et al., 2015) -- are the products of decades of manual literature curation. While invaluable, they exhibit systematic annotation bias toward well-studied genes and organisms. Haynes et al. (2018) demonstrated that gene annotations in GO are strongly correlated with publication count: genes studied in more publications carry more annotations, creating a self-reinforcing cycle in which enrichment analyses preferentially detect already-known biology. Approximately 30% of human protein-coding genes remain functionally unannotated or poorly annotated in GO (Kustatscher et al., 2022), and these "dark genes" are systematically excluded from enrichment analysis regardless of their biological importance.

KEGG pathways are heavily skewed toward metabolic and signalling cascades characterised in model organisms (primarily *E. coli*, yeast, and mouse), with comparatively sparse coverage of cell-type-specific regulatory programmes, non-coding RNA-mediated regulation, and recently discovered functional modules. Reactome provides more granular molecular detail but shares the same fundamental dependence on published experimental evidence, leading to coverage gaps for understudied cell types, tissues, and disease states.

### 1.2 Static Definitions and Lack of Context-Specificity

A critical limitation is that curated pathways are defined as fixed gene sets, independent of biological context. The KEGG "MAPK signalling pathway" contains the same gene members whether the analysis concerns hepatocytes, neurons, or T cells -- yet extensive evidence demonstrates that pathway topology and membership are dynamically rewired across cell types, developmental stages, and disease states (Lachmann et al., 2016; Szklarczyk et al., 2023). Single-cell RNA-seq data have revealed that the same signalling pathway operates through different effector genes in different cell types (Regev et al., 2017), rendering static gene set definitions at best imprecise and at worst misleading.

This problem is particularly acute for enrichment analysis of single-cell data, where cell-type composition effects can confound pathway-level inference. When a GSEA is performed using a gene set that conflates the regulatory programmes of multiple cell types, the resulting enrichment scores reflect an average over biologically distinct contexts, reducing both sensitivity and interpretability.

### 1.3 Incomplete Coverage and Redundancy

MSigDB, the most widely used gene set collection for GSEA, contains over 33,000 gene sets across its various sub-collections (Hallmark, C1-C8). Despite this apparent comprehensiveness, substantial gaps remain. Gene sets are heavily redundant: MSigDB's canonical pathway collection contains numerous near-duplicate entries from different source databases (e.g., KEGG, Reactome, WikiPathways, PID) that describe the same biological process with slightly different gene memberships. This redundancy inflates the multiple testing burden and complicates interpretation of enrichment results -- a researcher may observe 50 "significant" pathways that effectively describe the same underlying biology.

Conversely, many functional gene modules identified by recent large-scale experimental approaches -- including CRISPR screens (Replogle et al., 2022), spatial transcriptomics (Rao et al., 2021), and multi-omic perturbation assays -- have no representation in curated databases. The lag between experimental discovery and database incorporation can span years, during which enrichment analyses against outdated gene sets miss genuine biological signals.

### 1.4 Sensitivity to Gene Set Size and Composition

The statistical properties of enrichment analysis are sensitive to gene set size. Standard GSEA performs optimally with gene sets of 15--500 members (Subramanian et al., 2005), yet curated databases contain many gene sets outside this range. Small GO terms (fewer than 10 genes) lack statistical power, while large terms (GO:0005515 "protein binding," with over 7,000 annotations) are so broadly defined as to be uninformative. Curated databases provide no principled mechanism for controlling gene set granularity, and users must resort to ad hoc size filters that discard potentially informative gene sets.

---

## 2. Survey of Data-Driven Gene Set Discovery Methods

### 2.1 WGCNA (Weighted Gene Co-expression Network Analysis)

WGCNA (Langfelder & Horvath, 2008) was among the earliest systematic approaches to data-driven gene module discovery. It constructs a weighted gene co-expression network from a gene expression matrix, applies a soft thresholding power to enforce scale-free topology, and identifies modules via hierarchical clustering with dynamic tree cutting. WGCNA modules have been widely used as gene sets for downstream enrichment analysis.

**Limitations:** WGCNA modules are derived from individual datasets and therefore reflect the co-expression structure of a specific experimental condition, tissue, and technology platform. They do not generalise across datasets without re-computation. The method also relies on correlation-based similarity, which captures linear associations but misses nonlinear regulatory relationships. WGCNA scales poorly to single-cell data with high sparsity and large cell numbers, and its modules tend to be large and heterogeneous.

### 2.2 cNMF (Consensus Non-negative Matrix Factorisation)

cNMF (Kotliar et al., 2019) applies non-negative matrix factorisation to single-cell expression matrices to discover gene expression programs. By running NMF multiple times with different random initialisations and retaining consensus solutions, cNMF identifies robust programs that capture sources of biological variation. Each program is defined by a set of genes with high loadings, analogous to the topic-word distribution in topic models.

**Limitations:** cNMF programs are dataset-specific: they capture variation present in the analysed dataset but may not generalise to unseen conditions. The number of programs (K) must be specified a priori, and the method lacks a principled criterion for K selection. Gene programs from cNMF are defined on individual datasets, so cross-dataset comparison requires post hoc alignment.

### 2.3 Spectra

Spectra (Kedzierska et al., 2023) extends the topic modelling framework by incorporating prior biological knowledge (e.g., known gene sets from databases) as soft constraints on the factorisation. This yields gene programs that are informed by curated pathways while allowing discovery of novel modules from the data. Spectra operates on single-cell data and can incorporate cell-type labels to discover cell-type-specific programs.

**Limitations:** Spectra's reliance on prior gene sets means that its programs are anchored to existing knowledge, limiting its capacity for genuinely novel discovery. The method also operates on individual datasets and requires careful hyperparameter tuning.

### 2.4 SCENIC and SCENIC+

SCENIC (Aibar et al., 2017) and its successor SCENIC+ (Bravo Gonzalez-Blas et al., 2023) discover gene regulatory networks by combining expression-based co-expression analysis with transcription factor binding motif enrichment. SCENIC identifies "regulons" -- sets of genes co-regulated by a common transcription factor -- providing mechanistic interpretability. SCENIC+ extends this to integrate chromatin accessibility data from single-cell ATAC-seq.

**Limitations:** SCENIC regulons are restricted to transcription factor-target relationships and therefore capture only one layer of gene regulation. Post-transcriptional regulation, signalling cascades, metabolic modules, and other functional groupings are not represented. The method is also computationally expensive and requires motif databases that are species-specific and incomplete.

### 2.5 GenKI (Gene Knockout Inference)

GenKI (Xu et al., 2023) uses graph neural networks trained on gene regulatory networks to infer the downstream effects of gene knockouts. While not a gene set discovery method per se, GenKI demonstrates that learned neural network representations can capture functional gene relationships beyond what is encoded in curated databases.

**Limitations:** GenKI requires pre-defined gene regulatory networks as input and is designed for perturbation inference rather than systematic gene program discovery. It does not produce gene sets compatible with standard enrichment analysis frameworks.

### 2.6 Summary of Gaps

All existing data-driven methods share one or more of the following limitations: (i) they operate on individual datasets rather than leveraging compressed cross-dataset knowledge; (ii) they require substantial hyperparameter tuning with limited principled guidance; (iii) they capture only specific types of gene relationships (co-expression, TF-target); (iv) their outputs are not directly compatible with standard GSEA workflows. There is a clear need for a method that combines the generalisability of pre-trained foundation models with the flexibility of data-driven discovery and the rigour of established enrichment analysis frameworks.

---

## 3. How scRNA-seq Foundation Models Learn Gene Representations

### 3.1 Architecture and Pre-training Objectives

Single-cell RNA-seq foundation models are transformer-based neural networks pre-trained on large corpora of single-cell transcriptomes. Three prominent models illustrate different approaches:

**scGPT** (Cui et al., 2024) adopts a generative pre-training approach inspired by GPT. Genes within each cell are treated as tokens, and the model is trained to predict masked gene expression values conditioned on the remaining genes. scGPT's encoder produces per-gene token embeddings that capture the co-expression context of each gene within a cell. The model was pre-trained on over 33 million cells from the CELLxGENE repository, enabling it to learn gene-gene relationships across a vast diversity of cell types, tissues, and conditions.

**Geneformer** (Theodoris et al., 2023) represents gene expression as a rank-ordered sequence, where genes within each cell are sorted by their expression rank. This rank-value encoding is processed by a BERT-like transformer trained with masked token prediction. Geneformer was pre-trained on approximately 30 million cells and has demonstrated strong transfer learning performance for cell type classification, gene dosage sensitivity prediction, and chromatin dynamics.

**scBERT** (Yang et al., 2022) applies the BERT architecture to single-cell gene expression data using a masked language model objective. Genes are tokenised with expression binning, and the model learns contextual gene embeddings through pre-training on large-scale single-cell datasets.

### 3.2 Biological Information Encoded in Embeddings

Several studies have demonstrated that the internal representations of scRNA-seq foundation models encode biologically meaningful gene-gene relationships:

**Gene embedding structure mirrors functional ontology.** Cui et al. (2024) showed that gene embeddings from scGPT cluster according to known functional categories when visualised with UMAP: genes involved in the same biological processes (e.g., ribosomal biogenesis, immune signalling, cell cycle regulation) occupy contiguous regions of the embedding space. Theodoris et al. (2023) demonstrated similar structure in Geneformer embeddings, with genes in the same regulon or pathway exhibiting high cosine similarity.

**Attention patterns capture regulatory relationships.** The multi-head attention mechanism in transformer models learns to attend to functionally related genes. Analysis of scGPT's attention matrices reveals that attention weights between gene pairs correlate with known protein-protein interactions, co-expression relationships, and shared pathway membership (Cui et al., 2024). This suggests that attention matrices encode an implicit gene regulatory network.

**Context-dependent embeddings reflect cell-type-specific regulation.** Because foundation model embeddings are context-dependent -- the representation of a gene depends on the other genes co-expressed in the same cell -- the same gene can have different embeddings in different cell types. This property is precisely what is needed for context-specific pathway definitions: gene programs derived from T cell-specific embeddings will differ from those derived from hepatocyte-specific embeddings, reflecting genuine differences in regulatory wiring.

**Embeddings generalise across datasets.** Unlike co-expression modules derived from individual datasets, foundation model embeddings encode relationships learned across millions of cells from diverse experimental conditions. This provides a form of biological prior knowledge that is complementary to curated databases: where databases encode expert-curated knowledge, embeddings encode statistically learned co-regulation patterns from large-scale data.

### 3.3 Unexploited Potential

Despite these promising properties, no existing method systematically extracts gene programs from foundation model embeddings and deploys them within a rigorous enrichment analysis framework. Existing applications of scRNA-seq foundation models have focused on cell-level tasks (cell type annotation, perturbation prediction, multi-omic integration) rather than gene-level functional annotation. The gene-level knowledge encoded in these models represents a rich, untapped resource for pathway analysis.

---

## 4. Novelty Statement

**nPathway introduces a new paradigm for gene set enrichment analysis by systematically extracting gene programs from the embedding spaces and attention layers of pre-trained single-cell RNA-seq foundation models, replacing or augmenting static curated pathway databases with context-aware, data-driven gene sets that improve the sensitivity and specificity of enrichment analysis.** Unlike existing data-driven approaches (WGCNA, cNMF, Spectra, SCENIC), which derive gene modules from individual datasets, nPathway leverages the compressed cross-dataset biological knowledge embedded in foundation models pre-trained on tens of millions of cells, producing gene programs that are simultaneously general (drawing on broad pre-training data) and context-specific (conditioned on particular cell types or tissues through contextual embeddings).

**Why now?** The convergence of three developments makes this work timely and feasible: (i) the recent maturation of scRNA-seq foundation models (scGPT, Geneformer, scBERT) pre-trained on sufficiently large and diverse corpora to capture meaningful gene-gene relationships; (ii) accumulating evidence that these models encode biologically interpretable gene representations in their embedding spaces; and (iii) the growing recognition in the genomics community that static curated pathways are a bottleneck for single-cell functional genomics. nPathway fills the gap between foundation model representation learning and practical pathway analysis.

---

## 5. Research Questions

### RQ1: Do gene embeddings from foundation models encode functional pathway-like structures?

**Hypothesis:** Gene embeddings extracted from scGPT, Geneformer, and scBERT cluster into groups that significantly overlap with known curated pathways (KEGG, Reactome, GO Biological Process), as measured by Jaccard similarity, overlap coefficient, and hypergeometric enrichment p-values.

**Testable prediction:** Clustering the gene embedding space will produce gene programs in which at least 60% of programs show significant enrichment (FDR < 0.05) for at least one GO Biological Process term, and the mean maximum Jaccard similarity between discovered programs and KEGG pathways will exceed that of size-matched random gene sets by at least 5-fold.

### RQ2: Are foundation model-derived gene programs more context-specific than curated pathways?

**Hypothesis:** Gene programs derived from cell-type-specific embeddings (obtained by conditioning the foundation model on cells of a particular type) differ systematically across cell types, reflecting known tissue-specific pathway rewiring, whereas curated pathways provide identical gene sets regardless of context.

**Testable prediction:** For at least 30% of gene programs, the Jaccard similarity between the same program in two different cell types (e.g., T cells vs. hepatocytes) will be below 0.5, indicating substantial context-dependent rewiring. Genes that switch program membership across cell types will be enriched for known context-dependent regulators (e.g., tissue-specific transcription factors).

### RQ3: Do foundation model-derived gene programs improve the sensitivity and specificity of enrichment analysis compared to curated gene sets?

**Hypothesis:** When applied to perturbation datasets with known ground-truth affected pathways (e.g., Perturb-seq with targeted gene knockouts), enrichment analysis using nPathway-derived gene programs will achieve higher sensitivity (true positive rate) and comparable or higher specificity (true negative rate) compared to enrichment analysis using KEGG, Reactome, Hallmark, and MSigDB canonical pathways.

**Testable prediction:** In a simulation study with spiked-in pathway signals at varying effect sizes, nPathway gene programs will achieve at least 10% higher AUROC than the best-performing curated gene set collection. In real Perturb-seq data, nPathway will recover the expected target pathway for at least 20% more perturbations than KEGG or Hallmark.

### RQ4: Can foundation model-derived gene programs reveal novel functional modules not captured by existing databases?

**Hypothesis:** A substantial fraction of discovered gene programs will have low overlap with all curated pathways (novelty score > 0.5), and these novel programs will be biologically coherent as assessed by protein-protein interaction density, co-expression correlation, regulatory evidence, and independent experimental validation.

**Testable prediction:** At least 15% of discovered gene programs will have a maximum Jaccard similarity below 0.1 with any KEGG, Reactome, or GO term, yet will show significantly higher PPI connectivity (STRING database) than size-matched random gene sets (p < 0.01, permutation test) and will replicate across independent datasets (cross-dataset Jaccard > 0.3 for consensus programs).

---

## References

1. Aibar, S., Gonzalez-Blas, C. B., Moerman, T., Huynh-Thu, V. A., Imrichova, H., Hulselmans, G., ... & Aerts, S. (2017). SCENIC: single-cell regulatory network inference and clustering. *Nature Methods*, 14(11), 1083--1086.
2. Bravo Gonzalez-Blas, C., De Winter, S., Hulselmans, G., Rebolleda-Gomez, N., Maniero, I., Olofsson, D., ... & Aerts, S. (2023). SCENIC+: single-cell multiomic inference of enhancers and gene regulatory networks. *Nature Methods*, 20(9), 1355--1367.
3. Cui, H., Wang, C., Maan, H., Pang, K., Luo, F., Duan, N., & Wang, B. (2024). scGPT: toward building a foundation model for single-cell multi-omics using generative AI. *Nature Methods*, 21(8), 1470--1480.
4. Gene Ontology Consortium. (2021). The Gene Ontology resource: enriching a GOld mine. *Nucleic Acids Research*, 49(D1), D325--D334.
5. Gillespie, M., Jassal, B., Stephan, R., Milacic, M., Rothfels, K., Senber, A., ... & D'Eustachio, P. (2022). The reactome pathway knowledgebase 2022. *Nucleic Acids Research*, 50(D1), D588--D592.
6. Haynes, W. A., Tomczak, A., & Khatri, P. (2018). Gene annotation bias impedes biomedical research. *Scientific Reports*, 8, 1362.
7. Kanehisa, M., Furumichi, M., Sato, Y., Kawashima, M., & Ishiguro-Watanabe, M. (2023). KEGG for taxonomy-based analysis of pathways and genomes. *Nucleic Acids Research*, 51(D1), D587--D592.
8. Kedzierska, K. Z., Crawford, L., Amini, A. P., & Lu, M. (2023). Assessing the limits of zero-shot foundation models in single-cell biology. *bioRxiv*.
9. Kotliar, D., Veres, A., Nagy, M. A., Tabrizi, S., Hodber, E., Melton, D. A., & Sabeti, P. C. (2019). Identifying gene expression programs of cell-type identity and cellular activity with single-cell RNA-Seq. *eLife*, 8, e43803.
10. Kustatscher, G., Collins, T., Gingras, A. C., Guo, T., Hanash, S. M., Kluger, Y., ... & Rappsilber, J. (2022). Understudied proteins: opportunities and challenges for functional proteomics. *Nature Methods*, 19(7), 774--779.
11. Lachmann, A., Giorgi, F. M., Lopez, G., & Califano, A. (2016). ARACNe-AP: gene network reverse engineering through adaptive partitioning inference of mutual information. *Bioinformatics*, 32(14), 2233--2235.
12. Langfelder, P., & Horvath, S. (2008). WGCNA: an R package for weighted correlation network analysis. *BMC Bioinformatics*, 9, 559.
13. Liberzon, A., Birger, C., Thorvaldsdottir, H., Ghandi, M., Mesirov, J. P., & Tamayo, P. (2015). The Molecular Signatures Database Hallmark gene set collection. *Cell Systems*, 1(6), 417--425.
14. Rao, A., Barkley, D., Franca, G. S., & Yanai, I. (2021). Exploring tissue architecture using spatial transcriptomics. *Nature*, 596(7871), 211--220.
15. Regev, A., Teichmann, S. A., Lander, E. S., Amit, I., Benoist, C., Birney, E., ... & Human Cell Atlas Meeting Participants. (2017). Science forum: the human cell atlas. *eLife*, 6, e27041.
16. Replogle, J. M., Saunders, R. A., Pogson, A. N., Hussmann, J. A., Lenail, A., Guna, A., ... & Weissman, J. S. (2022). Mapping information-rich genotype-phenotype landscapes with genome-scale Perturb-seq. *Cell*, 185(14), 2559--2575.
17. Subramanian, A., Tamayo, P., Mootha, V. K., Mukherjee, S., Ebert, B. L., Gillette, M. A., ... & Mesirov, J. P. (2005). Gene set enrichment analysis: a knowledge-based approach for interpreting genome-wide expression profiles. *Proceedings of the National Academy of Sciences*, 102(43), 15545--15550.
18. Szklarczyk, D., Kirsch, R., Koutrouli, M., Nastou, K., Mehryary, F., Hachilif, R., ... & von Mering, C. (2023). The STRING database in 2023: protein-protein association networks and functional enrichment analyses for any sequenced genome of interest. *Nucleic Acids Research*, 51(D1), D99--D105.
19. Theodoris, C. V., Xiao, L., Chopra, A., Chaffin, M. D., Al Sayed, Z. R., Hill, M. C., ... & Ellinor, P. T. (2023). Transfer learning enables predictions in network biology. *Nature*, 618(7965), 616--624.
20. Xu, Y., et al. (2023). GenKI: Gene Knockout Inference via graph neural networks. *Bioinformatics*, 39(1), btac785.
21. Yang, F., Wang, W., Wang, F., Fang, Y., Tang, D., Huang, J., ... & Yao, J. (2022). scBERT as a large-scale pretrained deep language model for cell type annotation of single-cell RNA-seq data. *Nature Machine Intelligence*, 4(10), 852--866.
