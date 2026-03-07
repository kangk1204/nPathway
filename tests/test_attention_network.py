"""Tests for attention-network program discovery utilities."""

from __future__ import annotations

import numpy as np

from npathway.discovery.attention_network import AttentionNetworkProgramDiscovery


def test_threshold_uses_positive_edges_only() -> None:
    """High quantile threshold should prune most sparse positive edges."""
    rng = np.random.default_rng(42)
    n = 80

    upper = np.zeros((n, n), dtype=np.float64)
    tri_i, tri_j = np.triu_indices(n, k=1)
    keep_mask = rng.random(len(tri_i)) < 0.05
    upper[tri_i[keep_mask], tri_j[keep_mask]] = rng.random(keep_mask.sum())
    adj = upper + upper.T

    model = AttentionNetworkProgramDiscovery(threshold_quantile=0.9)
    thresholded = model._threshold(adj)

    before = int((adj[np.triu_indices(n, k=1)] > 0).sum())
    after = int((thresholded[np.triu_indices(n, k=1)] > 0).sum())

    assert before > 0
    # After 0.9 quantile thresholding on positive edges, only top ~10%
    # should remain (allow ties and small-sample variation).
    assert after < before
    assert after <= max(1, int(np.ceil(before * 0.2)))


def test_threshold_handles_no_positive_edges() -> None:
    """Thresholding should return an all-zero adjacency when no edges exist."""
    adj = np.zeros((6, 6), dtype=np.float64)
    np.fill_diagonal(adj, 1.0)

    model = AttentionNetworkProgramDiscovery(threshold_quantile=0.9)
    thresholded = model._threshold(adj)

    assert thresholded.shape == adj.shape
    assert np.allclose(thresholded, 0.0)


def test_pagerank_convergence_uses_final_iterate() -> None:
    """Converged PageRank should return the updated vector, even at loose tolerance."""
    adj = np.array(
        [
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )

    scores = AttentionNetworkProgramDiscovery._pagerank(adj, tol=10.0)
    assert scores == [0.0, 1.0, 0.0]
