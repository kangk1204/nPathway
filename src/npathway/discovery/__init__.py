"""Gene program discovery subpackage.

This subpackage implements multiple strategies for discovering data-driven
gene programs from gene embeddings extracted by foundation models.

Modules
-------
base
    Abstract base class defining the discovery interface.
clustering
    Clustering-based gene program discovery (Leiden, spectral, k-means, HDBSCAN).
topic_model
    Neural topic model-based discovery (Embedded Topic Model) using
    foundation model embedding priors.
attention_network
    Attention matrix analysis and community detection for program discovery.
ensemble
    Consensus gene program discovery combining multiple methods.
"""

from __future__ import annotations

from npathway.discovery.attention_network import AttentionNetworkProgramDiscovery
from npathway.discovery.base import BaseProgramDiscovery
from npathway.discovery.clustering import ClusteringProgramDiscovery
from npathway.discovery.ensemble import EnsembleProgramDiscovery
from npathway.discovery.topic_model import TopicModelProgramDiscovery

__all__: list[str] = [
    "BaseProgramDiscovery",
    "ClusteringProgramDiscovery",
    "TopicModelProgramDiscovery",
    "AttentionNetworkProgramDiscovery",
    "EnsembleProgramDiscovery",
]
