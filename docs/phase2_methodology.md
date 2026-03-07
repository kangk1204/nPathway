# Phase 2: Methodology Design

## nPathway -- Foundation Model-Derived Gene Programs for Context-Aware Gene Set Enrichment Analysis

---

## 1. Gene Embedding Extraction Strategies

### 1.1 Overview

The first stage of the nPathway pipeline extracts gene-level vector representations from pre-trained single-cell RNA-seq foundation models. These embeddings encode the functional relationships among genes learned during pre-training. We support three extraction strategies, one for each foundation model architecture.

All extractors implement the `BaseEmbeddingExtractor` interface, which provides:
- `load_model(model_path)` -- load a pre-trained checkpoint
- `extract_gene_embeddings(adata, layer)` -- universal embeddings
- `extract_context_embeddings(adata, cell_type_key, layer)` -- context-specific embeddings
- `save_embeddings()` / `load_embeddings()` -- disk persistence

### 1.2 scGPT Embedding Extraction

**Model architecture:** scGPT uses a transformer encoder with gene tokens as input. Each gene in a cell is represented as a token embedding summed with an expression-value embedding. The transformer processes all gene tokens in parallel (masked attention during pre-training, full attention during inference).

**Extraction strategy:**

```
Algorithm: scGPT Gene Embedding Extraction
Input: Pre-trained scGPT model M, AnnData object D, layer index L
Output: Gene embedding matrix E of shape (n_genes, d)

1. Load model M and identify vocabulary V (gene name -> token ID)
2. Identify overlapping genes G = V ∩ D.var_names
3. For each cell c_i in D:
   a. Tokenise cell: map expressed genes to token IDs
   b. Forward pass through M, collecting hidden states at layer L
   c. Extract per-gene hidden states h_{i,g} for each gene g in G
4. Compute universal embedding for each gene g:
   E[g] = (1/N_g) * sum_{i: g expressed in c_i} h_{i,g}
   where N_g = number of cells expressing gene g
5. Return E, gene_names(G)
```

**Layer selection:** We extract from the final transformer layer by default (layer = -1), as this captures the highest-level contextual representations. Intermediate layers can be specified for analysis of the representation hierarchy.

**Context-specific embeddings:** To obtain cell-type-specific embeddings, we partition cells by `cell_type_key` in `adata.obs` and average gene hidden states within each partition:

```
For each cell type t:
    E_t[g] = (1/N_{g,t}) * sum_{i in cells(t): g expressed in c_i} h_{i,g}
```

This produces a dictionary `{cell_type: embedding_matrix}`.

### 1.3 Geneformer Embedding Extraction

**Model architecture:** Geneformer uses a BERT-like transformer with rank-value encoding. Genes within each cell are sorted by their expression rank, and this ordered sequence is processed by the transformer. The model was pre-trained with masked token prediction on rank-ordered gene sequences.

**Extraction strategy:**

```
Algorithm: Geneformer Gene Embedding Extraction
Input: Pre-trained Geneformer model M (HuggingFace), AnnData D, layer L
Output: Gene embedding matrix E of shape (n_genes, d)

1. Load model M from HuggingFace hub
2. For each cell c_i in D:
   a. Rank genes by expression level (descending)
   b. Encode as rank-ordered token sequence (top-K genes)
   c. Forward pass through M with output_hidden_states=True
   d. Collect hidden states at layer L for each gene token
3. Average per-gene hidden states across cells (same as scGPT)
4. Return E, gene_names
```

**Rank-value encoding considerations:** Geneformer's rank-based tokenisation means that the same gene receives different positional context depending on its relative expression in each cell. This provides an intrinsic form of context-dependence that enriches the extracted embeddings.

### 1.4 scBERT Embedding Extraction

**Model architecture:** scBERT applies BERT's masked language model objective to single-cell gene expression data. Gene expression values are discretised into bins, and each gene is represented as a token with a binned expression embedding.

**Extraction strategy:**

```
Algorithm: scBERT Gene Embedding Extraction
Input: Pre-trained scBERT model M, AnnData D, layer L
Output: Gene embedding matrix E of shape (n_genes, d)

1. Load model M
2. Discretise expression values into bins per scBERT's tokenisation scheme
3. For each cell c_i in D:
   a. Construct input tokens (gene ID + expression bin)
   b. Forward pass through M, collecting hidden states at layer L
   c. Extract per-gene hidden states
4. Average per-gene hidden states across cells
5. Return E, gene_names
```

### 1.5 Universal vs. Context-Specific Embeddings

We define two embedding modes:

- **Universal embeddings:** Averaged across all cells in the dataset, capturing the global co-regulation structure learned by the model. Suitable for general-purpose gene program discovery.

- **Context-specific embeddings:** Averaged within cell-type or tissue partitions, capturing condition-specific co-regulation. Suitable for discovering gene programs that reflect tissue-specific pathway rewiring.

Both modes produce an embedding matrix of shape `(n_genes, embedding_dim)`, where `embedding_dim` is model-dependent (typically 256--512).

---

## 2. Gene Program Discovery from Embeddings

### 2.1 Approach 1: Clustering-Based Discovery

**Rationale:** If the gene embedding space encodes functional relationships, then genes in similar regions of the space should participate in similar biological processes. Clustering the embedding space directly partitions genes into programs.

#### 2.1.1 Leiden Community Detection

```
Algorithm: Leiden-Based Gene Program Discovery
Input: Gene embeddings E (n_genes x d), k_neighbors, resolution
Output: Gene programs P = {program_id: [gene_list]}

1. L2-normalise embeddings: E_norm = E / ||E||
2. Build kNN graph G with k = k_neighbors using cosine distance:
   For each gene g, find k nearest neighbours in E_norm
   Edge weight w(g_i, g_j) = 1 - cosine_distance(g_i, g_j)
3. Symmetrise: w(i,j) = max(w(i,j), w(j,i))
4. Apply Leiden algorithm with RBConfigurationVertexPartition:
   partition = leiden(G, resolution=resolution)
5. Assign genes to programs based on partition membership
6. Score genes within each program by cosine similarity to centroid:
   centroid_k = mean(E[genes in program k])
   score(g, k) = cosine_sim(E[g], centroid_k)
7. Return P, scores
```

**Hyperparameter selection:**
- `k_neighbors`: Default 15. Can be tuned via modularity maximisation.
- `resolution`: Controls granularity. Higher values yield more programs. Default 1.0; sweep [0.5, 2.0] for sensitivity analysis.

#### 2.1.2 Spectral Clustering

Spectral clustering on the kNN affinity matrix provides an alternative that explicitly uses the eigenstructure of the graph Laplacian:

1. Build kNN adjacency matrix A (same as Leiden step 2).
2. Compute the normalised graph Laplacian L = I - D^{-1/2} A D^{-1/2}.
3. Extract the first K eigenvectors of L.
4. Apply k-means to the spectral embedding.

**K selection:** If K is not specified, we perform silhouette analysis over a range [K_min, K_max] using k-means on the spectral embedding.

#### 2.1.3 K-Means Clustering

Direct k-means in the embedding space:

1. L2-normalise embeddings.
2. Run k-means with K clusters, n_init=10, max_iter=300.
3. Auto-select K via silhouette score if not specified.

#### 2.1.4 HDBSCAN

Density-based clustering that does not require K:

1. Apply HDBSCAN with `min_cluster_size` (default 5).
2. Genes assigned label -1 are classified as noise.
3. Noise genes are retained in a separate "noise" program.

**Advantages:** Automatically determines the number of programs and can identify programs of varying size and density.

### 2.2 Approach 2: Topic Model-Based Discovery

**Rationale:** The Embedded Topic Model (ETM; Dieng et al., 2020) treats cells as documents and genes as words, discovering latent topics (gene programs) that explain the co-occurrence of genes across cells. By incorporating pre-trained gene embeddings as a prior on the topic-gene distribution, the ETM leverages foundation model knowledge while discovering programs from expression data.

```
Algorithm: ETM-Based Gene Program Discovery
Input: Gene embeddings E (n_genes x d), expression matrix X (n_cells x n_genes),
       n_topics K, n_epochs, learning_rate
Output: Gene programs P, topic-gene weight matrix W (K x n_genes)

1. Normalise expression matrix X to bag-of-words (row-normalised)
   If X is not provided, generate synthetic BOW from embedding similarities
2. Initialise ETM:
   - Encoder: MLP mapping BOW -> (mu, log_sigma) in topic space (VAE)
   - Topic embeddings: T (K x d), randomly initialised
   - Gene embeddings: E_g (n_genes x d), initialised from pre-trained E
     optionally frozen (freeze_gene_embeddings=True)
3. Training loop (n_epochs iterations):
   For each mini-batch of cells:
     a. Encode: (mu, log_sigma) = Encoder(BOW)
     b. Sample: z ~ N(mu, exp(log_sigma)^2) via reparameterisation trick
     c. Topic proportions: theta = softmax(z)
     d. Topic-gene logits: L = normalize(T) @ normalize(E_g)^T
     e. Reconstruction: recon = theta @ log_softmax(L)
     f. Loss = -sum(BOW * recon) + KL(N(mu, sigma^2) || N(0, I))
     g. Backpropagate, clip gradients, update parameters
4. Extract topic-gene weights: W = softmax(L, dim=-1)
5. For each topic k:
   Select top-N genes by W[k, :] -> program_k
   Score genes by their topic weight
6. Return P, W
```

**Key design decisions:**
- **Frozen vs. trainable gene embeddings:** Freezing preserves the pre-trained structure while allowing topic embeddings to learn; fine-tuning allows the model to adapt gene representations to the expression data.
- **Synthetic BOW fallback:** When only embeddings (not expression data) are available, we generate synthetic cells by sampling random directions in embedding space and computing gene similarity scores, enabling an embedding-only workflow.
- **Number of topics K:** Evaluated via perplexity, topic coherence (NPMI), and downstream enrichment performance.

### 2.3 Approach 3: Attention Network-Based Discovery

**Rationale:** Transformer attention matrices encode pairwise gene-gene relationships that may capture regulatory interactions beyond what is encoded in the embedding space alone. We extract attention matrices and construct gene-gene networks for community detection.

```
Algorithm: Attention Network Gene Program Discovery
Input: Attention matrices A (n_heads x n_genes x n_genes) or embeddings E,
       aggregation method, threshold_quantile, resolution
Output: Gene programs P with centrality scores

1. If attention matrices are provided:
   a. Handle shape variants:
      - (n_layers, n_heads, G, G): average across layers first
      - (n_heads, G, G): use directly
      - (G, G): treat as single-head
   b. Aggregate across heads:
      - "mean": A_agg = mean(A, axis=heads)
      - "max": A_agg = max(A, axis=heads)
   Else:
   a. Construct cosine similarity proxy from embeddings:
      A_agg = max(0, E_norm @ E_norm^T)

2. Symmetrise: A_sym = (A_agg + A_agg^T) / 2
   Zero diagonal: A_sym[i,i] = 0

3. Threshold edges:
   cutoff = quantile(A_sym[upper_triangle], threshold_quantile)
   A_thresh[i,j] = A_sym[i,j] if A_sym[i,j] >= cutoff, else 0

4. Community detection (Leiden):
   Build igraph from A_thresh
   partition = leiden(G, resolution=resolution)

5. For each community (program):
   Extract sub-adjacency matrix
   Compute gene centrality:
   - PageRank: iterative power method on column-stochastic sub-matrix
   - Eigenvector centrality: leading eigenvector via power iteration
   Normalise scores to [0, 1]

6. Return P, centrality scores
```

**Advantages over clustering:** Attention-based programs are derived from the model's learned pairwise interaction strengths rather than embedding proximity, potentially capturing directional regulatory relationships and long-range gene-gene dependencies.

### 2.4 Approach 4: Ensemble / Consensus Discovery

**Rationale:** Different discovery methods capture different aspects of gene-gene relationships. An ensemble approach aggregates the outputs of multiple methods to identify high-confidence consensus programs.

```
Algorithm: Ensemble Consensus Gene Program Discovery
Input: Gene embeddings E, gene_names, list of M discovery methods,
       consensus_method, resolution, threshold_quantile, min_program_size
Output: Consensus gene programs P with confidence scores

1. Run each of M methods independently:
   For m = 1, ..., M:
     P_m = method_m.fit(E, gene_names).get_programs()

2. Build co-occurrence matrix C (n_genes x n_genes):
   For each method m and each program p in P_m:
     For each pair of genes (g_i, g_j) in p:
       C[i, j] += 1
   Normalise: C = C / M

3. Consensus clustering on C:
   If consensus_method == "leiden":
     Threshold C: keep edges where C[i,j] > quantile(C, threshold_quantile)
     Build graph, apply Leiden with given resolution
   If consensus_method == "hierarchical":
     Convert C to distance: D = max(C) - C
     Apply agglomerative clustering with n_programs clusters

4. Merge small programs:
   For each program with < min_program_size genes:
     Merge into the nearest neighbour program by mean co-occurrence

5. Compute confidence scores:
   For each gene g in consensus program k:
     confidence(g, k) = fraction of M methods in which g co-occurs
                        with at least one other member of k in the same
                        original program

6. Return P, confidence scores
```

**Advantages:** Consensus programs are robust to the idiosyncrasies of individual methods. The confidence score provides a natural weight for use in weighted enrichment analysis.

---

## 3. Gene Program Characterisation

### 3.1 Automated Functional Annotation

Each discovered gene program is automatically annotated using:

1. **GO enrichment analysis:** Fisher's exact test for over-representation of GO Biological Process terms among program genes, with BH-FDR correction.
2. **KEGG/Reactome enrichment:** Same hypergeometric test against curated pathway databases.
3. **Transcription factor enrichment:** Test whether program genes are enriched for targets of specific transcription factors (ENCODE, ChEA databases).

The annotation pipeline assigns to each program the most significant term (lowest FDR) from each database, providing a multi-faceted functional description.

### 3.2 Overlap and Divergence from Curated Pathways

We compute the pairwise Jaccard similarity matrix between all discovered programs and all curated pathways:

```
overlap_matrix[i, j] = |program_i ∩ pathway_j| / |program_i ∪ pathway_j|
```

From this matrix, we derive:
- **Best-match similarity:** For each program, the maximum Jaccard with any curated pathway.
- **Coverage:** Fraction of curated pathway genes present in at least one discovered program.
- **Redundancy:** Mean pairwise Jaccard among all discovered programs.

### 3.3 Novelty Quantification

The novelty score measures the fraction of gene-program assignments not found in any curated pathway:

```
novelty = |{g in program: g not in any curated pathway}| / |program|
```

Aggregated across all programs, this yields a global novelty measure indicating how much new biology the discovered programs capture.

### 3.4 Cross-Model Consistency

To assess whether the same programs emerge from different foundation models, we compute:

1. **Adjusted Rand Index (ARI):** Between the label vectors produced by programs from model A and model B over a shared gene universe.
2. **Normalised Mutual Information (NMI):** Same comparison, using information-theoretic agreement.
3. **Pairwise Jaccard:** Between each program from model A and all programs from model B, identifying best-matching program pairs.
4. **Consensus programs:** Programs identified by the ensemble method using extractors from multiple models.

---

## 4. Integration with GSEA Framework

### 4.1 GMT/GMX Export

All discovery methods export programs in standard GMT format:

```
program_name<TAB>description<TAB>gene1<TAB>gene2<TAB>...
```

This ensures compatibility with:
- **fgsea** (R): fast preranked GSEA
- **GSVA** (R): gene set variation analysis
- **ssGSEA**: single-sample GSEA
- **AUCell** (R/Python): area under the curve-based enrichment
- **nPathway's built-in enrichment engine**

### 4.2 Binary vs. Weighted Membership

**Binary membership** (standard GMT): Each gene is either in or out of a program. Compatible with all standard GSEA tools.

**Weighted membership** (extended GMT): Each gene carries an association weight w(g, k) in [0, 1] reflecting the strength of its membership in program k. Weights are derived from:
- Cosine similarity to cluster centroid (clustering methods)
- Topic loading (topic models)
- Centrality score (attention networks)
- Confidence score (ensemble methods)

Weighted membership is exported in an extended GMT format:

```
program_name<TAB>na<TAB>gene1,0.95<TAB>gene2,0.87<TAB>...
```

### 4.3 Weighted Enrichment Score

For weighted gene programs, we introduce a modified enrichment score that incorporates both the ranking metric (e.g., log fold-change) and the membership weight:

```
Standard GSEA hit step:
  P_hit(i) += |r_i|^p / sum_{g in S} |r_g|^p    if gene_i in S

Weighted nPathway hit step:
  P_hit(i) += |r_i| * w(g_i, k) / sum_{g in S} |r_g| * w(g, k)
```

where `r_i` is the ranking metric for gene i, `w(g_i, k)` is the membership weight of gene i in program k, and p is the weight exponent.

This formulation ensures that:
- Genes with strong program membership contribute more to the enrichment score.
- Genes with weak membership (peripheral members) contribute less, reducing noise.
- The denominator normalises by the total weighted contribution, preserving the KS-like distributional interpretation.

### 4.4 Enrichment Methods Implemented

nPathway provides four enrichment analysis methods:

| Method | Input | Output | Use case |
|--------|-------|--------|----------|
| Fisher's exact test | Gene list + programs | p-values, odds ratios, FDR | Over-representation analysis |
| Preranked GSEA | Ranked gene list + programs | ES, NES, p-values, FDR, leading edge | Standard GSEA workflow |
| ssGSEA | Expression matrix + programs | Per-cell enrichment scores | Single-sample scoring, cell-level analysis |
| Weighted GSEA | Ranked list + weighted programs | ES, NES, p-values, FDR | Exploiting probabilistic membership |

All methods return tidy DataFrames with Benjamini-Hochberg FDR correction for multiple testing.

### 4.5 Pseudocode: Complete Pipeline

```
Algorithm: Complete nPathway Pipeline
Input: AnnData object D, foundation model path, curated GMT file,
       discovery method(s), enrichment method
Output: Enrichment results DataFrame, gene programs (GMT)

# Stage 1: Embedding Extraction
extractor = get_extractor("scgpt")
extractor.load_model(model_path)
E = extractor.extract_gene_embeddings(D, layer=-1)
G = extractor.get_gene_names()

# Stage 2: Gene Program Discovery
discovery = ClusteringProgramDiscovery(method="leiden", resolution=1.0)
discovery.fit(E, G)
programs = discovery.get_programs()
scores = discovery.get_program_scores()

# Stage 3: Characterisation
curated = read_gmt(curated_gmt_file)
overlap = compute_overlap_matrix(programs, curated)
novelty = novelty_score(programs, curated)
redundancy = program_redundancy(programs)

# Stage 4: Enrichment Analysis
ranked_genes = compute_differential_expression(D, condition_key)
results = preranked_gsea(ranked_genes, programs, n_perm=1000)
significant = results[results["fdr"] < 0.05]

# Stage 5: Export
write_gmt(programs, "npathway_programs.gmt")
discovery.to_gmt("npathway_programs.gmt")
```

---

## 5. Computational Considerations

### 5.1 Scalability

| Component | Time complexity | Memory | Typical runtime |
|-----------|----------------|--------|-----------------|
| Embedding extraction (scGPT) | O(N * G * d) | O(G * d) | 5-30 min (10K cells, GPU) |
| kNN graph construction | O(G^2 * d) | O(G * k) | 1-5 min (20K genes) |
| Leiden clustering | O(E * iterations) | O(G + E) | < 1 min |
| ETM training | O(epochs * N * G * K) | O(G * d + K * d) | 5-20 min (GPU) |
| Preranked GSEA (1000 perm) | O(n_perm * G * P) | O(G) | 1-10 min per comparison |

where N = cells, G = genes, d = embedding dim, K = programs, E = edges, P = number of programs.

### 5.2 Recommended Hardware

- **GPU:** Required for embedding extraction and ETM training. NVIDIA GPU with 16+ GB VRAM recommended. Apple MPS supported.
- **CPU:** Sufficient for clustering, enrichment analysis, and benchmarking.
- **Memory:** 32 GB RAM recommended for large datasets (> 100K cells).

---

## References

1. Blei, D. M., Ng, A. Y., & Jordan, M. I. (2003). Latent Dirichlet allocation. *Journal of Machine Learning Research*, 3, 993--1022.
2. Cui, H., et al. (2024). scGPT: toward building a foundation model for single-cell multi-omics using generative AI. *Nature Methods*, 21(8), 1470--1480.
3. Dieng, A. B., Ruiz, F. J., & Blei, D. M. (2020). Topic modeling in embedding spaces. *Transactions of the Association for Computational Linguistics*, 8, 439--453.
4. Korotkevich, G., et al. (2021). Fast gene set enrichment analysis. *bioRxiv*, 060012.
5. Subramanian, A., et al. (2005). Gene set enrichment analysis. *PNAS*, 102(43), 15545--15550.
6. Theodoris, C. V., et al. (2023). Transfer learning enables predictions in network biology. *Nature*, 618(7965), 616--624.
7. Traag, V. A., Waltman, L., & van Eck, N. J. (2019). From Louvain to Leiden: guaranteeing well-connected communities. *Scientific Reports*, 9, 5233.
8. Yang, F., et al. (2022). scBERT as a large-scale pretrained deep language model for cell type annotation. *Nature Machine Intelligence*, 4(10), 852--866.
