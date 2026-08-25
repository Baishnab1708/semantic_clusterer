"""Vectorized cosine similarity functions."""

import numpy as np


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    """L2-normalize vectors along the last axis.
    
    Args:
        vectors: Array of shape (..., n_features).
        
    Returns:
        Normalized array of the same shape.
    """
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    # Avoid division by zero and tiny values
    norms = np.clip(norms, a_min=1e-10, a_max=None)
    return vectors / norms


def cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Compute pairwise cosine similarity matrix.
    
    Args:
        embeddings: Array of shape (n_samples, n_features).
        
    Returns:
        Similarity matrix of shape (n_samples, n_samples).
    """
    normalized = normalize_vectors(embeddings)
    return np.dot(normalized, normalized.T)


def pairwise_cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute pairwise cosine similarity between two sets of vectors.
    
    Args:
        a: Array of shape (n_samples_a, n_features).
        b: Array of shape (n_samples_b, n_features).
        
    Returns:
        Similarity matrix of shape (n_samples_a, n_samples_b).
    """
    a_normalized = normalize_vectors(a)
    b_normalized = normalize_vectors(b)
    return np.dot(a_normalized, b_normalized.T)


def cosine_distance_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Compute pairwise cosine distance matrix.
    
    Args:
        embeddings: Array of shape (n_samples, n_features).
        
    Returns:
        Distance matrix of shape (n_samples, n_samples). Values in [0, 2].
    """
    return 1 - cosine_similarity_matrix(embeddings)
