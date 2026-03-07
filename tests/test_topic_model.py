"""Tests for topic model-based gene program discovery."""

from __future__ import annotations

import numpy as np
import pytest

from npathway.discovery.topic_model import (
    TopicModelProgramDiscovery,
    TrainingHistory,
)


def _make_topic_data(
    n_genes: int = 50,
    n_dims: int = 32,
    seed: int = 42,
) -> tuple[np.ndarray, list[str]]:
    """Generate synthetic gene embeddings for topic model testing."""
    rng = np.random.default_rng(seed)
    embeddings = rng.standard_normal((n_genes, n_dims)).astype(np.float32)
    gene_names = [f"GENE_{i}" for i in range(n_genes)]
    return embeddings, gene_names


def _make_expression_matrix(
    n_cells: int = 200,
    n_genes: int = 50,
    seed: int = 42,
) -> np.ndarray:
    """Generate synthetic expression matrix with realistic structure."""
    rng = np.random.default_rng(seed)
    # Simulate count-like data with some structure
    base = rng.poisson(lam=5, size=(n_cells, n_genes)).astype(np.float32)
    # Add some highly variable genes
    base[:, :10] = rng.poisson(lam=20, size=(n_cells, 10)).astype(np.float32)
    return base


# ======================================================================
# Original tests (preserved)
# ======================================================================


def test_topic_model_fit() -> None:
    """TopicModelProgramDiscovery should fit without errors."""
    embeddings, gene_names = _make_topic_data(n_genes=50, n_dims=16)
    model = TopicModelProgramDiscovery(
        n_topics=5,
        hidden_dim=32,
        n_epochs=10,
        batch_size=64,
        top_n_genes=20,
        device="cpu",
        random_state=42,
    )
    result = model.fit(embeddings, gene_names)
    assert result is model  # Should return self


def test_topic_model_programs() -> None:
    """get_programs should return the correct number of topics with gene lists."""
    embeddings, gene_names = _make_topic_data(n_genes=50, n_dims=16)
    model = TopicModelProgramDiscovery(
        n_topics=5,
        hidden_dim=32,
        n_epochs=10,
        batch_size=64,
        top_n_genes=20,
        device="cpu",
        random_state=42,
    )
    model.fit(embeddings, gene_names)

    programs = model.get_programs()
    assert isinstance(programs, dict)
    assert len(programs) == 5

    for topic_name, genes in programs.items():
        assert topic_name.startswith("topic_")
        assert isinstance(genes, list)
        assert len(genes) == 20  # top_n_genes
        for g in genes:
            assert g in gene_names


def test_topic_model_configurable_k() -> None:
    """Different n_topics should produce correspondingly many programs."""
    embeddings, gene_names = _make_topic_data(n_genes=50, n_dims=16)

    for k in [3, 7]:
        model = TopicModelProgramDiscovery(
            n_topics=k,
            hidden_dim=32,
            n_epochs=5,
            batch_size=64,
            top_n_genes=15,
            device="cpu",
            random_state=42,
        )
        model.fit(embeddings, gene_names)
        programs = model.get_programs()
        assert len(programs) == k, f"Expected {k} topics, got {len(programs)}"


def test_topic_gene_weights() -> None:
    """get_topic_gene_weights should return a (n_topics, n_genes) matrix."""
    n_genes = 50
    n_topics = 5
    embeddings, gene_names = _make_topic_data(n_genes=n_genes, n_dims=16)
    model = TopicModelProgramDiscovery(
        n_topics=n_topics,
        hidden_dim=32,
        n_epochs=10,
        batch_size=64,
        top_n_genes=20,
        device="cpu",
        random_state=42,
    )
    model.fit(embeddings, gene_names)

    weights = model.get_topic_gene_weights()
    assert weights.shape == (n_topics, n_genes)

    # Weights should be non-negative (softmax output)
    assert np.all(weights >= 0.0)

    # Each row (topic) should roughly sum to 1 (softmax)
    row_sums = weights.sum(axis=1)
    np.testing.assert_allclose(row_sums, 1.0, atol=1e-5)

    # get_program_scores should return scored genes
    scores = model.get_program_scores()
    assert isinstance(scores, dict)
    assert len(scores) == n_topics
    for topic_name, gene_score_list in scores.items():
        assert len(gene_score_list) == 20  # top_n_genes
        for gene, score in gene_score_list:
            assert isinstance(gene, str)
            assert isinstance(score, float)


# ======================================================================
# New tests: expression matrix handling
# ======================================================================


def test_fit_with_expression_matrix() -> None:
    """Fitting with a real expression matrix should use it as BOW."""
    n_genes = 50
    embeddings, gene_names = _make_topic_data(n_genes=n_genes, n_dims=16)
    expr = _make_expression_matrix(n_cells=100, n_genes=n_genes)

    model = TopicModelProgramDiscovery(
        n_topics=5,
        hidden_dim=32,
        n_epochs=10,
        batch_size=64,
        top_n_genes=20,
        device="cpu",
        random_state=42,
    )
    result = model.fit(embeddings, gene_names, expression_matrix=expr)
    assert result is model
    programs = model.get_programs()
    assert len(programs) == 5


def test_expression_preprocessing() -> None:
    """_preprocess_expression should normalise and log-transform."""
    n_cells, n_genes = 100, 50
    expr = _make_expression_matrix(n_cells=n_cells, n_genes=n_genes)

    processed, hvg_mask = TopicModelProgramDiscovery._preprocess_expression(expr)

    assert processed.shape == (n_cells, n_genes)
    assert hvg_mask is None
    # Should be log1p-transformed, so no values should be negative
    assert np.all(processed >= 0)
    # Values should be finite
    assert np.all(np.isfinite(processed))


def test_expression_hvg_selection() -> None:
    """HVG selection should subset genes correctly."""
    n_cells, n_genes = 100, 50
    n_hvg = 20
    expr = _make_expression_matrix(n_cells=n_cells, n_genes=n_genes)

    processed, hvg_mask = TopicModelProgramDiscovery._preprocess_expression(
        expr, n_top_hvg=n_hvg
    )

    assert processed.shape == (n_cells, n_hvg)
    assert hvg_mask is not None
    assert hvg_mask.sum() == n_hvg


def test_fit_with_hvg_selection() -> None:
    """Fitting with HVG selection should work end to end."""
    n_genes = 50
    n_hvg = 30
    embeddings, gene_names = _make_topic_data(n_genes=n_genes, n_dims=16)
    expr = _make_expression_matrix(n_cells=100, n_genes=n_genes)

    model = TopicModelProgramDiscovery(
        n_topics=3,
        hidden_dim=32,
        n_epochs=5,
        batch_size=64,
        top_n_genes=10,
        n_top_hvg=n_hvg,
        device="cpu",
        random_state=42,
    )
    model.fit(embeddings, gene_names, expression_matrix=expr)
    programs = model.get_programs()
    assert len(programs) == 3
    # Gene names should be a subset of original
    all_genes = set()
    for genes in programs.values():
        all_genes.update(genes)
    assert all_genes.issubset(set(gene_names))


def test_expression_matrix_shape_mismatch() -> None:
    """Mismatched expression matrix columns should raise ValueError."""
    embeddings, gene_names = _make_topic_data(n_genes=50, n_dims=16)
    bad_expr = np.ones((100, 30), dtype=np.float32)  # wrong n_genes

    model = TopicModelProgramDiscovery(
        n_topics=3, hidden_dim=32, n_epochs=5, device="cpu"
    )
    with pytest.raises(ValueError, match="genes"):
        model.fit(embeddings, gene_names, expression_matrix=bad_expr)


# ======================================================================
# New tests: training diagnostics
# ======================================================================


def test_training_history() -> None:
    """get_training_history should return per-epoch diagnostics."""
    embeddings, gene_names = _make_topic_data(n_genes=50, n_dims=16)
    n_epochs = 15
    model = TopicModelProgramDiscovery(
        n_topics=5,
        hidden_dim=32,
        n_epochs=n_epochs,
        batch_size=64,
        top_n_genes=20,
        early_stopping_patience=0,  # disable early stopping
        val_fraction=0.0,  # no validation
        device="cpu",
        random_state=42,
    )
    model.fit(embeddings, gene_names)

    history = model.get_training_history()
    assert isinstance(history, TrainingHistory)
    assert len(history.elbo) == n_epochs
    assert len(history.reconstruction_loss) == n_epochs
    assert len(history.kl_divergence) == n_epochs
    assert len(history.val_perplexity) == 0  # no val split
    assert history.stopped_epoch is None  # no early stop

    # All loss values should be finite
    for loss in history.elbo:
        assert np.isfinite(loss)
    for nll in history.reconstruction_loss:
        assert np.isfinite(nll)
    for kl in history.kl_divergence:
        assert np.isfinite(kl)


def test_training_history_with_validation() -> None:
    """Validation perplexity should be tracked when val_fraction > 0."""
    embeddings, gene_names = _make_topic_data(n_genes=50, n_dims=16)
    expr = _make_expression_matrix(n_cells=200, n_genes=50)

    model = TopicModelProgramDiscovery(
        n_topics=3,
        hidden_dim=32,
        n_epochs=10,
        batch_size=32,
        top_n_genes=15,
        val_fraction=0.2,
        early_stopping_patience=0,  # disable early stopping
        device="cpu",
        random_state=42,
    )
    model.fit(embeddings, gene_names, expression_matrix=expr)

    history = model.get_training_history()
    assert len(history.val_perplexity) == 10
    for ppl in history.val_perplexity:
        assert ppl > 0  # perplexity must be positive


def test_early_stopping() -> None:
    """Early stopping should stop training before n_epochs."""
    embeddings, gene_names = _make_topic_data(n_genes=50, n_dims=16)

    model = TopicModelProgramDiscovery(
        n_topics=3,
        hidden_dim=32,
        n_epochs=500,  # very high; early stopping should kick in
        batch_size=64,
        top_n_genes=15,
        early_stopping_patience=5,
        val_fraction=0.1,
        device="cpu",
        random_state=42,
    )
    model.fit(embeddings, gene_names)

    history = model.get_training_history()
    # Should have stopped before 500 epochs
    assert len(history.elbo) < 500
    # OR it went all 500 but at least the mechanism was invoked
    # (in case loss kept improving, which is unlikely for 500 epochs)


def test_training_history_not_fitted() -> None:
    """get_training_history should raise RuntimeError if not fitted."""
    model = TopicModelProgramDiscovery(n_topics=3, device="cpu")
    with pytest.raises(RuntimeError):
        model.get_training_history()


# ======================================================================
# New tests: architecture improvements
# ======================================================================


def test_configurable_hidden_layers() -> None:
    """Model should work with different numbers of hidden layers."""
    embeddings, gene_names = _make_topic_data(n_genes=50, n_dims=16)

    for n_layers in [1, 3, 4]:
        model = TopicModelProgramDiscovery(
            n_topics=3,
            hidden_dim=32,
            n_hidden_layers=n_layers,
            n_epochs=5,
            batch_size=64,
            top_n_genes=15,
            device="cpu",
            random_state=42,
        )
        model.fit(embeddings, gene_names)
        assert len(model.get_programs()) == 3


def test_batch_norm_toggle() -> None:
    """Model should work with and without batch normalisation."""
    embeddings, gene_names = _make_topic_data(n_genes=50, n_dims=16)

    for use_bn in [True, False]:
        model = TopicModelProgramDiscovery(
            n_topics=3,
            hidden_dim=32,
            n_epochs=5,
            batch_size=64,
            top_n_genes=15,
            use_batch_norm=use_bn,
            device="cpu",
            random_state=42,
        )
        model.fit(embeddings, gene_names)
        assert len(model.get_programs()) == 3


def test_lr_scheduler() -> None:
    """Model should work with and without LR scheduler."""
    embeddings, gene_names = _make_topic_data(n_genes=50, n_dims=16)

    for use_sched in [True, False]:
        model = TopicModelProgramDiscovery(
            n_topics=3,
            hidden_dim=32,
            n_epochs=10,
            batch_size=64,
            top_n_genes=15,
            use_lr_scheduler=use_sched,
            device="cpu",
            random_state=42,
        )
        model.fit(embeddings, gene_names)
        assert len(model.get_programs()) == 3


# ======================================================================
# New tests: topic-gene scoring
# ======================================================================


def test_decoder_topic_gene_weights() -> None:
    """get_decoder_topic_gene_weights should return valid weight matrix."""
    n_genes = 50
    n_topics = 5
    embeddings, gene_names = _make_topic_data(n_genes=n_genes, n_dims=16)
    model = TopicModelProgramDiscovery(
        n_topics=n_topics,
        hidden_dim=32,
        n_epochs=10,
        batch_size=64,
        top_n_genes=20,
        device="cpu",
        random_state=42,
    )
    model.fit(embeddings, gene_names)

    dw = model.get_decoder_topic_gene_weights()
    assert dw.shape == (n_topics, n_genes)
    assert np.all(dw >= 0)
    np.testing.assert_allclose(dw.sum(axis=1), 1.0, atol=1e-5)


def test_topic_coherence() -> None:
    """get_topic_coherence should return a score for each topic."""
    n_topics = 5
    embeddings, gene_names = _make_topic_data(n_genes=50, n_dims=16)
    model = TopicModelProgramDiscovery(
        n_topics=n_topics,
        hidden_dim=32,
        n_epochs=10,
        batch_size=64,
        top_n_genes=20,
        device="cpu",
        random_state=42,
    )
    model.fit(embeddings, gene_names)

    coherence = model.get_topic_coherence()
    assert isinstance(coherence, dict)
    assert len(coherence) == n_topics
    for topic_name, score in coherence.items():
        assert topic_name.startswith("topic_")
        assert isinstance(score, float)
        assert np.isfinite(score)
        # NPMI range is roughly [-1, 1]
        assert -1.5 <= score <= 1.5


def test_topic_coherence_not_fitted() -> None:
    """get_topic_coherence should raise RuntimeError if not fitted."""
    model = TopicModelProgramDiscovery(n_topics=3, device="cpu")
    with pytest.raises(RuntimeError):
        model.get_topic_coherence()


def test_coherence_filtering() -> None:
    """Setting a very high coherence threshold should filter topics."""
    embeddings, gene_names = _make_topic_data(n_genes=50, n_dims=16)

    # Use a very high threshold that should filter out some/all topics
    model = TopicModelProgramDiscovery(
        n_topics=5,
        hidden_dim=32,
        n_epochs=10,
        batch_size=64,
        top_n_genes=20,
        coherence_threshold=999.0,  # impossibly high
        device="cpu",
        random_state=42,
    )
    model.fit(embeddings, gene_names)
    programs = model.get_programs()
    # All topics should be filtered
    assert len(programs) == 0


# ======================================================================
# New tests: ablation support
# ======================================================================


def test_embedding_init_random() -> None:
    """embedding_init='random' should ignore pretrained embeddings."""
    embeddings, gene_names = _make_topic_data(n_genes=50, n_dims=16)
    model = TopicModelProgramDiscovery(
        n_topics=3,
        hidden_dim=32,
        n_epochs=5,
        batch_size=64,
        top_n_genes=15,
        embedding_init="random",
        device="cpu",
        random_state=42,
    )
    model.fit(embeddings, gene_names)
    assert len(model.get_programs()) == 3


def test_embedding_init_pretrained() -> None:
    """embedding_init='pretrained' should use provided embeddings."""
    embeddings, gene_names = _make_topic_data(n_genes=50, n_dims=16)
    model = TopicModelProgramDiscovery(
        n_topics=3,
        hidden_dim=32,
        n_epochs=5,
        batch_size=64,
        top_n_genes=15,
        embedding_init="pretrained",
        device="cpu",
        random_state=42,
    )
    model.fit(embeddings, gene_names)
    assert len(model.get_programs()) == 3


def test_freeze_embeddings() -> None:
    """freeze_gene_embeddings=True should keep embeddings fixed."""
    embeddings, gene_names = _make_topic_data(n_genes=50, n_dims=16)

    model = TopicModelProgramDiscovery(
        n_topics=3,
        hidden_dim=32,
        n_epochs=10,
        batch_size=64,
        top_n_genes=15,
        freeze_gene_embeddings=True,
        embedding_init="pretrained",
        device="cpu",
        random_state=42,
    )
    model.fit(embeddings, gene_names)

    # Gene embeddings should match the original (they were frozen)
    import torch
    with torch.no_grad():
        fitted_emb = model._model.gene_embeddings.cpu().numpy()  # type: ignore
    np.testing.assert_allclose(fitted_emb, embeddings, atol=1e-6)


def test_freeze_vs_unfreeze_differ() -> None:
    """Frozen and unfrozen embeddings should give different results."""
    embeddings, gene_names = _make_topic_data(n_genes=50, n_dims=16)

    model_frozen = TopicModelProgramDiscovery(
        n_topics=3,
        hidden_dim=32,
        n_epochs=20,
        batch_size=64,
        top_n_genes=15,
        freeze_gene_embeddings=True,
        use_decoder_weights=False,
        device="cpu",
        random_state=42,
    )
    model_frozen.fit(embeddings, gene_names)

    model_free = TopicModelProgramDiscovery(
        n_topics=3,
        hidden_dim=32,
        n_epochs=20,
        batch_size=64,
        top_n_genes=15,
        freeze_gene_embeddings=False,
        use_decoder_weights=False,
        device="cpu",
        random_state=42,
    )
    model_free.fit(embeddings, gene_names)

    w_frozen = model_frozen.get_topic_gene_weights()
    w_free = model_free.get_topic_gene_weights()
    # Weights should differ (different training dynamics)
    assert not np.allclose(w_frozen, w_free, atol=1e-4)


def test_pretrained_vs_random_differ() -> None:
    """Pretrained vs random embedding init should give different results."""
    embeddings, gene_names = _make_topic_data(n_genes=50, n_dims=16)

    model_pt = TopicModelProgramDiscovery(
        n_topics=3,
        hidden_dim=32,
        n_epochs=20,
        batch_size=64,
        top_n_genes=15,
        embedding_init="pretrained",
        use_decoder_weights=False,
        device="cpu",
        random_state=42,
    )
    model_pt.fit(embeddings, gene_names)

    model_rnd = TopicModelProgramDiscovery(
        n_topics=3,
        hidden_dim=32,
        n_epochs=20,
        batch_size=64,
        top_n_genes=15,
        embedding_init="random",
        use_decoder_weights=False,
        device="cpu",
        random_state=42,
    )
    model_rnd.fit(embeddings, gene_names)

    w_pt = model_pt.get_topic_gene_weights()
    w_rnd = model_rnd.get_topic_gene_weights()
    assert not np.allclose(w_pt, w_rnd, atol=1e-4)


# ======================================================================
# Edge cases
# ======================================================================


def test_small_dataset() -> None:
    """Model should handle very small datasets (few cells/genes)."""
    embeddings, gene_names = _make_topic_data(n_genes=10, n_dims=8)
    model = TopicModelProgramDiscovery(
        n_topics=2,
        hidden_dim=16,
        n_epochs=5,
        batch_size=64,
        top_n_genes=5,
        device="cpu",
        random_state=42,
    )
    model.fit(embeddings, gene_names)
    assert len(model.get_programs()) == 2


def test_fit_with_expression_and_validation() -> None:
    """Full pipeline: expression matrix + validation + early stopping."""
    n_genes = 50
    embeddings, gene_names = _make_topic_data(n_genes=n_genes, n_dims=16)
    expr = _make_expression_matrix(n_cells=200, n_genes=n_genes)

    model = TopicModelProgramDiscovery(
        n_topics=5,
        hidden_dim=32,
        n_epochs=20,
        batch_size=32,
        top_n_genes=15,
        val_fraction=0.15,
        early_stopping_patience=5,
        use_lr_scheduler=True,
        device="cpu",
        random_state=42,
    )
    model.fit(embeddings, gene_names, expression_matrix=expr)

    # All getters should work
    programs = model.get_programs()
    scores = model.get_program_scores()
    weights = model.get_topic_gene_weights()
    history = model.get_training_history()
    coherence = model.get_topic_coherence()
    decoder_w = model.get_decoder_topic_gene_weights()

    assert len(programs) <= 5
    assert len(scores) <= 5
    assert weights.shape[0] == 5
    assert len(history.elbo) > 0
    assert len(coherence) == 5
    assert decoder_w.shape[0] == 5


def test_val_fraction_must_be_strictly_less_than_one() -> None:
    """Validation split must leave at least one training sample."""
    import pytest

    with pytest.raises(ValueError, match="val_fraction must be in"):
        TopicModelProgramDiscovery(val_fraction=1.0)
