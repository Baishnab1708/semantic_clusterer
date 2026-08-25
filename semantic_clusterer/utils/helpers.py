"""Helper functions for clustering analysis."""

from typing import List, Optional

import numpy as np

from semantic_clusterer.utils.similarity import normalize_vectors


def compute_centroid(embeddings: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Compute the centroid (mean embedding) of a cluster.
    
    Args:
        embeddings: Array of shape (n_samples, n_features).
        mask: Optional boolean mask of shape (n_samples,) to select cluster members.
            If None, all samples are used.
        
    Returns:
        Centroid vector of shape (n_features,).
    """
    if mask is not None:
        cluster_embeddings = embeddings[mask]
    else:
        cluster_embeddings = embeddings

    if len(cluster_embeddings) == 0:
        return np.zeros(embeddings.shape[1])

    return np.mean(cluster_embeddings, axis=0)


def find_representative(
    texts: List[str],
    embeddings: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> str:
    """Find the most representative text in a cluster.
    
    The representative is the item closest to the cluster centroid.
    
    Args:
        texts: List of text strings.
        embeddings: Array of shape (n_samples, n_features).
        mask: Optional boolean mask of shape (n_samples,) to select cluster members.
            If None, all samples are used.
        
    Returns:
        The most representative text string.
    """
    if mask is not None:
        indices = np.where(mask)[0]
        cluster_texts = [texts[i] for i in indices]
        cluster_embeddings = embeddings[mask]
    else:
        indices = np.arange(len(texts))
        cluster_texts = texts
        cluster_embeddings = embeddings

    if len(cluster_texts) == 0:
        return ""

    if len(cluster_texts) == 1:
        return cluster_texts[0]

    # Compute centroid
    centroid = compute_centroid(cluster_embeddings)

    # Find closest item to centroid
    centroid_normalized = normalize_vectors(centroid.reshape(1, -1))
    embeddings_normalized = normalize_vectors(cluster_embeddings)
    similarities = np.dot(embeddings_normalized, centroid_normalized.T).flatten()

    best_idx = np.argmax(similarities)
    return cluster_texts[best_idx]


def compute_confidence(
    embeddings: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> float:
    """Compute cluster cohesion score (confidence).
    
    This is based on the average cosine similarity of all items to the centroid.
    Higher values indicate more cohesive clusters.
    
    Args:
        embeddings: Array of shape (n_samples, n_features).
        mask: Optional boolean mask of shape (n_samples,) to select cluster members.
            If None, all samples are used.
        
    Returns:
        Confidence score in [0, 1].
    """
    if mask is not None:
        cluster_embeddings = embeddings[mask]
    else:
        cluster_embeddings = embeddings

    if len(cluster_embeddings) <= 1:
        return 1.0  # Perfect confidence for single-item clusters

    # Compute centroid
    centroid = compute_centroid(cluster_embeddings)

    # Compute average similarity to centroid
    centroid_normalized = normalize_vectors(centroid.reshape(1, -1))
    embeddings_normalized = normalize_vectors(cluster_embeddings)
    similarities = np.dot(embeddings_normalized, centroid_normalized.T).flatten()

    score = float(np.mean(similarities))
    return float(np.clip(score, 0.0, 1.0))


def get_cluster_indices(labels: np.ndarray) -> dict:
    """Group sample indices by cluster label.
    
    Args:
        labels: Array of cluster labels of shape (n_samples,).
        
    Returns:
        Dictionary mapping cluster label to list of sample indices.
    """
    cluster_indices = {}
    for idx, label in enumerate(labels):
        if label not in cluster_indices:
            cluster_indices[label] = []
        cluster_indices[label].append(idx)
    return cluster_indices


def compute_noise_ratio(labels: np.ndarray) -> float:
    """Compute the ratio of noise points (label == -1).
    
    Args:
        labels: Array of cluster labels of shape (n_samples,).
        
    Returns:
        Noise ratio in [0, 1].
    """
    if len(labels) == 0:
        return 0.0

    n_noise = np.sum(labels == -1)
    return float(n_noise / len(labels))
