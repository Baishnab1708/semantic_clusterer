"""Output formatting for clustering results."""

from typing import List, Optional, TypedDict, Union

import numpy as np

from semantic_clusterer.utils.helpers import (
    compute_centroid,
    compute_confidence,
    find_representative,
    get_cluster_indices,
)


class ClusterResult(TypedDict, total=False):
    """TypedDict for detailed cluster output."""
    cluster_id: int
    representative: str
    items: List[str]
    size: int
    centroid: Optional[List[float]]
    confidence: Optional[float]
    keywords: Optional[List[str]]       # Top topic keywords (c-TF-IDF)
    topic_label: Optional[str]          # Short human-readable topic label


class OutputFormatter:
    """Format clustering results for output.
    
    Supports two output formats:
    - "simple": List[List[str]] - just grouped texts
    - "detailed": List[ClusterResult] - with metadata
    
    Attributes:
        include_centroid: Include centroid in detailed output.
        include_confidence: Include confidence in detailed output.
        exclude_noise: Whether to exclude noise cluster from output.
    """

    def __init__(
        self,
        include_centroid: bool = False,
        include_confidence: bool = True,
        exclude_noise: bool = False,
    ):
        """Initialize output formatter.
        
        Args:
            include_centroid: Include cluster centroid vectors.
            include_confidence: Include cluster confidence scores.
            exclude_noise: Whether to exclude noise points (label -1).
        """
        self.include_centroid = include_centroid
        self.include_confidence = include_confidence
        self.exclude_noise = exclude_noise

    def format_simple(
        self,
        texts: List[str],
        labels: np.ndarray,
    ) -> List[List[str]]:
        """Format results as simple grouped lists.
        
        Args:
            texts: Original text strings.
            labels: Cluster labels.
            
        Returns:
            List of clusters, each a list of text strings.
        """
        cluster_indices = get_cluster_indices(labels)

        clusters = []
        for label in sorted(cluster_indices.keys()):
            if self.exclude_noise and label == -1:
                continue

            indices = cluster_indices[label]
            cluster_texts = [texts[i] for i in indices]
            clusters.append(cluster_texts)

        return clusters

    def format_detailed(
        self,
        texts: List[str],
        embeddings: np.ndarray,
        labels: np.ndarray,
    ) -> List[ClusterResult]:
        """Format results with detailed metadata.
        
        Args:
            texts: Original text strings.
            embeddings: Embedding vectors.
            labels: Cluster labels.
            
        Returns:
            List of ClusterResult dictionaries.
        """
        cluster_indices = get_cluster_indices(labels)

        results: List[ClusterResult] = []

        for label in sorted(cluster_indices.keys()):
            if self.exclude_noise and label == -1:
                continue

            indices = cluster_indices[label]
            mask = np.isin(np.arange(len(texts)), indices)

            cluster_texts = [texts[i] for i in indices]

            # Find representative
            representative = find_representative(texts, embeddings, mask)

            result: ClusterResult = {
                "cluster_id": int(label),
                "representative": representative,
                "items": cluster_texts,
                "size": len(cluster_texts),
            }

            # Optional: include centroid
            if self.include_centroid:
                centroid = compute_centroid(embeddings, mask)
                result["centroid"] = centroid.tolist()

            # Optional: include confidence
            if self.include_confidence:
                confidence = float(np.clip(compute_confidence(embeddings, mask), 0.0, 1.0))
                result["confidence"] = round(confidence, 4)

            results.append(result)

        return results

    def enrich_with_keywords(
        self,
        results: "List[ClusterResult]",
        texts: List[str],
        labels: "np.ndarray",
        top_n: int = 10,
    ) -> "List[ClusterResult]":
        """Enrich detailed results with c-TF-IDF keywords and topic labels.

        This is a pure post-processing step called after format_detailed.
        It has zero effect on cluster assignments.

        Args:
            results: Output of format_detailed().
            texts: The texts that were clustered (aligned with labels).
            labels: Cluster labels aligned with texts.
            top_n: Number of keywords to extract per cluster.

        Returns:
            The same results list, mutated in-place with keywords and
            topic_label added to each non-noise cluster. Returns results
            unchanged if keyword extraction fails for any reason.
        """
        try:
            from semantic_clusterer.representation.keywords import (
                extract_cluster_keywords,
                generate_topic_label,
            )

            cluster_keywords = extract_cluster_keywords(
                texts, labels, top_n=top_n
            )

            for result in results:
                cid = result.get("cluster_id", -1)
                if cid < 0 or cid not in cluster_keywords:
                    continue
                kw_pairs = cluster_keywords[cid]
                result["keywords"] = [word for word, _ in kw_pairs]
                result["topic_label"] = generate_topic_label(kw_pairs)

        except Exception:
            pass  # Graceful degradation — never break clustering output

        return results

    def format(
        self,
        texts: List[str],
        embeddings: np.ndarray,
        labels: np.ndarray,
        return_format: str = "simple",
    ) -> Union[List[List[str]], List[ClusterResult]]:
        """Format results based on specified format.
        
        Args:
            texts: Original text strings.
            embeddings: Embedding vectors.
            labels: Cluster labels.
            return_format: "simple" or "detailed".
            
        Returns:
            Formatted clustering results.
            
        Raises:
            ValueError: If unknown format specified.
        """
        if return_format == "simple":
            return self.format_simple(texts, labels)
        elif return_format == "detailed":
            return self.format_detailed(texts, embeddings, labels)
        else:
            raise ValueError(f"Unknown return_format: {return_format}")
