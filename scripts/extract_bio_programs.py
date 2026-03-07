"""Extract biological gene programs from PBMC 3k and annotate with Hallmark sets.

Outputs:
  results/tables/bio_programs_kmeans.csv  -- top genes per program + Hallmark match
  results/tables/bio_programs_ensemble.csv
  results/tables/bio_case_study.csv       -- top 3 programs with full gene lists
"""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from npathway.data.datasets import filter_gene_sets_to_adata, load_msigdb_gene_sets, load_pbmc3k
from npathway.data.preprocessing import (
    build_graph_regularized_embeddings,
)
from npathway.discovery.clustering import ClusteringProgramDiscovery
from npathway.discovery.ensemble import EnsembleProgramDiscovery
from npathway.evaluation.metrics import jaccard_similarity

OUTDIR = os.path.join(os.path.dirname(__file__), "..", "results", "tables")
os.makedirs(OUTDIR, exist_ok=True)


def annotate_programs(programs: dict, hallmark_sets: dict, top_n: int = 10):
    """Annotate each program with best-matching Hallmark set and top genes."""
    rows = []
    for prog_name, gene_list in programs.items():
        prog_set = set(gene_list)

        # Best Hallmark match
        best_hall = None
        best_jacc = 0.0
        for hall_name, hall_genes in hallmark_sets.items():
            j = jaccard_similarity(prog_set, set(hall_genes))
            if j > best_jacc:
                best_jacc = j
                best_hall = hall_name

        # Clean Hallmark name
        hall_short = best_hall.replace("HALLMARK_", "").replace("_", " ").title() if best_hall else "Novel"

        rows.append({
            "program": prog_name,
            "n_genes": len(gene_list),
            "hallmark_match": hall_short,
            "hallmark_jaccard": round(best_jacc, 4),
            "top_genes": ", ".join(gene_list[:top_n]),
        })

    return pd.DataFrame(rows).sort_values("hallmark_jaccard", ascending=False)


def main():
    print("Loading PBMC 3k (preprocessed)...")
    adata = load_pbmc3k(preprocessed=True)

    print("Building graph-regularized embeddings...")
    embeddings, gene_names = build_graph_regularized_embeddings(
        adata, n_components=50, k_neighbors=15, n_diffusion_steps=3, alpha=0.5
    )
    print(f"  Embeddings: {embeddings.shape}")

    # Load Hallmark gene sets (filtered to adata genes)
    print("Loading Hallmark gene sets...")
    hallmark_raw = load_msigdb_gene_sets(collection="hallmark")
    hallmark_sets = filter_gene_sets_to_adata(hallmark_raw, adata, min_genes=3)
    print(f"  {len(hallmark_sets)} Hallmark gene sets loaded (filtered to adata genes)")

    # --- KMeans ---
    print("Running nPathway-KMeans (K=20)...")
    km = ClusteringProgramDiscovery(method="kmeans", n_programs=20, random_state=42)
    km.fit(embeddings, gene_names)
    km_programs = km.get_programs()
    print(f"  {len(km_programs)} programs, {sum(len(v) for v in km_programs.values())} genes total")

    km_df = annotate_programs(km_programs, hallmark_sets, top_n=15)
    km_df.to_csv(os.path.join(OUTDIR, "bio_programs_kmeans.csv"), index=False)
    print("  Saved bio_programs_kmeans.csv")

    # --- Leiden ---
    print("Running nPathway-Leiden...")
    lei = ClusteringProgramDiscovery(method="leiden", random_state=42)
    lei.fit(embeddings, gene_names)
    lei_programs = lei.get_programs()
    print(f"  {len(lei_programs)} programs")

    lei_df = annotate_programs(lei_programs, hallmark_sets, top_n=15)
    lei_df.to_csv(os.path.join(OUTDIR, "bio_programs_leiden.csv"), index=False)
    print("  Saved bio_programs_leiden.csv")

    # --- Ensemble ---
    print("Running nPathway-Ensemble (KMeans + Leiden + Spectral)...")
    methods = [
        ClusteringProgramDiscovery(method="kmeans", n_programs=20, random_state=42),
        ClusteringProgramDiscovery(method="leiden", random_state=42),
        ClusteringProgramDiscovery(method="spectral", n_programs=20, random_state=42),
    ]
    ens = EnsembleProgramDiscovery(methods=methods, consensus_method="leiden",
                                    resolution=1.0, random_state=42)
    ens.fit(embeddings, gene_names)
    ens_programs = ens.get_programs()
    print(f"  {len(ens_programs)} consensus programs")

    ens_df = annotate_programs(ens_programs, hallmark_sets, top_n=15)
    ens_df.to_csv(os.path.join(OUTDIR, "bio_programs_ensemble.csv"), index=False)
    print("  Saved bio_programs_ensemble.csv")

    # --- Biological case study: top 6 programs by Hallmark Jaccard ---
    print("\nBuilding biological case study table...")
    top6 = km_df.head(6).copy()

    # Full gene lists for top 6 programs (up to 30 genes)
    top6["full_gene_list"] = top6["program"].apply(
        lambda p: ", ".join(km_programs[p][:30])
    )
    top6.to_csv(os.path.join(OUTDIR, "bio_case_study.csv"), index=False)
    print("  Saved bio_case_study.csv")

    # Print summary for paper writing
    print("\n" + "="*70)
    print("TOP PROGRAMS FOR PAPER:")
    print("="*70)
    for _, row in top6.iterrows():
        print(f"\n[{row['program']}] ({row['n_genes']} genes)")
        print(f"  Best Hallmark: {row['hallmark_match']} (Jaccard={row['hallmark_jaccard']})")
        print(f"  Top genes: {row['top_genes']}")

    print("\n" + "="*70)
    print("ALL KMEANS PROGRAMS SUMMARY:")
    print("="*70)
    print(km_df[["program", "n_genes", "hallmark_match", "hallmark_jaccard"]].to_string(index=False))


if __name__ == "__main__":
    main()
