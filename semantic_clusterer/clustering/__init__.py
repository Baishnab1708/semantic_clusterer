"""Clustering algorithms and strategies."""

__all__ = ["CentroidFallback"]


def __getattr__(name: str):
    if name == "CentroidFallback":
        from semantic_clusterer.clustering.centroid_fallback import CentroidFallback
        return CentroidFallback
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
