"""semantic_clusterer - Lightweight, high-performance semantic text clustering."""

from importlib.metadata import version as _version

try:
    __version__ = _version("semantic_clusterer")
except Exception:
    __version__ = "0.1.0"

__all__ = [
    "SemanticClusterer",
    "SemanticKSplit",
    "SemanticClustererConfig",
    "SemanticKSplitConfig",
    "ClusteringReport",
    "FittedState",
    "ClusterStats",
    "SUPPORTED_DIM_BANDS",
    "normalize_embedding_model",
    "validate_embeddings",
    "__version__",
    # Backward-compat aliases
    "ClustererConfig",
]


def __getattr__(name: str):
    if name == "SemanticKSplit":
        from semantic_clusterer.k_split import SemanticKSplit
        return SemanticKSplit
    if name == "SemanticClusterer":
        from semantic_clusterer.core import SemanticClusterer
        return SemanticClusterer
    if name == "SemanticClustererConfig":
        from semantic_clusterer.config import SemanticClustererConfig
        return SemanticClustererConfig
    if name == "SemanticKSplitConfig":
        from semantic_clusterer.config import SemanticKSplitConfig
        return SemanticKSplitConfig
    if name == "ClustererConfig":
        from semantic_clusterer.config import ClustererConfig
        return ClustererConfig
    if name == "ClusteringReport":
        from semantic_clusterer.report import ClusteringReport
        return ClusteringReport
    if name == "SUPPORTED_DIM_BANDS":
        from semantic_clusterer.dim_bands import SUPPORTED_DIM_BANDS
        return SUPPORTED_DIM_BANDS
    if name == "normalize_embedding_model":
        from semantic_clusterer.embedding.adapters import normalize_embedding_model
        return normalize_embedding_model
    if name == "validate_embeddings":
        from semantic_clusterer.embedding.adapters import validate_embeddings
        return validate_embeddings
    if name == "FittedState":
        from semantic_clusterer.persistence import FittedState
        return FittedState
    if name == "ClusterStats":
        from semantic_clusterer.persistence import ClusterStats
        return ClusterStats
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
