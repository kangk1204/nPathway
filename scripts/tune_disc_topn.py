"""Quick sweep: find optimal top_n for DiscRefined to maximize Hallmark alignment."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from npathway.data.datasets import filter_gene_sets_to_adata, load_msigdb_gene_sets, load_pbmc3k
from npathway.data.preprocessing import build_graph_regularized_embeddings
from npathway.discovery.clustering import ClusteringProgramDiscovery


def compute_hallmark_alignment(programs, hallmark_sets):
    """Compute mean best-match Jaccard across all programs."""
    from npathway.evaluation.metrics import jaccard_similarity
    scores = []
    for prog_genes in programs.values():
        prog_set = set(prog_genes)
        if not prog_set:
            continue
        best = max(
            jaccard_similarity(prog_set, set(h)) for h in hallmark_sets.values()
        )
        scores.append(best)
    return float(np.mean(scores)) if scores else 0.0


print("Loading PBMC 3k...")
adata = load_pbmc3k(preprocessed=True)

print("Building graph-regularized embeddings...")
embeddings, gene_names = build_graph_regularized_embeddings(
    adata, n_components=50, k_neighbors=15, n_diffusion_steps=3, alpha=0.5
)

print("Loading Hallmark sets...")
hallmark_raw = load_msigdb_gene_sets(collection="hallmark")
hallmark_sets = filter_gene_sets_to_adata(hallmark_raw, adata, min_genes=3)
print(f"  {len(hallmark_sets)} Hallmark sets")

print("\n--- Sweep: K=20 (baseline) ---")
km20 = ClusteringProgramDiscovery(method="kmeans", n_programs=20, random_state=42)
km20.fit(embeddings, gene_names)

for top_n in [10, 15, 20, 25, 30, 40, 50, 70]:
    progs = km20.get_discriminative_programs(top_n=top_n)
    align = compute_hallmark_alignment(progs, hallmark_sets)
    print(f"  K=20, top_n={top_n:3d}: hallmark_alignment={align:.4f}")

print("\n--- Sweep: K=30 ---")
km30 = ClusteringProgramDiscovery(method="kmeans", n_programs=30, random_state=42)
km30.fit(embeddings, gene_names)

for top_n in [10, 15, 20, 25, 30, 40, 50]:
    progs = km30.get_discriminative_programs(top_n=top_n)
    align = compute_hallmark_alignment(progs, hallmark_sets)
    print(f"  K=30, top_n={top_n:3d}: hallmark_alignment={align:.4f}")

print("\n--- Sweep: K=40 ---")
km40 = ClusteringProgramDiscovery(method="kmeans", n_programs=40, random_state=42)
km40.fit(embeddings, gene_names)

for top_n in [10, 15, 20, 25, 30]:
    progs = km40.get_discriminative_programs(top_n=top_n)
    align = compute_hallmark_alignment(progs, hallmark_sets)
    print(f"  K=40, top_n={top_n:3d}: hallmark_alignment={align:.4f}")

print("\n--- Reference ---")
print("  cNMF baseline: hallmark_alignment=0.0729")
