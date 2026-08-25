"""Centroid-based fallback for noise recovery."""

import numpy as np
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
from semantic_clusterer.utils.helpers import compute_noise_ratio

class CentroidFallback:
    """Nearest Centroid fallback for HDBSCAN noise points.
    
    When HDBSCAN classifies points as noise (label == -1), this fallback
    assigns each noise point to the nearest valid cluster centroid. This
    ensures all data is clustered (0 noise) while maintaining the structural
    integrity and purity of the original HDBSCAN clusters.
    
    Attributes:
        noise_threshold: Minimum noise ratio to trigger fallback.
        metric: Distance metric to use for centroid assignment.
    """

    def __init__(
        self,
        noise_threshold: float = 0.0,
        metric: str = "euclidean",
    ):
        """Initialize centroid fallback.
        
        Args:
            noise_threshold: Ratio of noise points to trigger fallback.
            metric: Distance metric ('euclidean' or 'cosine').
        """
        self.noise_threshold = noise_threshold
        self.metric = metric

    def should_apply(self, labels: np.ndarray) -> bool:
        """Check if fallback should be applied.
        
        Args:
            labels: Cluster labels from HDBSCAN.
            
        Returns:
            True if noise ratio > threshold and there is at least one valid cluster.
        """
        noise_ratio = compute_noise_ratio(labels)
        has_valid_clusters = np.any(labels >= 0)
        return noise_ratio > self.noise_threshold and has_valid_clusters

    def recluster_noise(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
    ) -> np.ndarray:
        """Assign noise points to nearest valid cluster centroids.
        
        Args:
            embeddings: Full embedding array.
            labels: Original cluster labels.
            
        Returns:
            Updated cluster labels with all noise points assigned.
        """
        if not self.should_apply(labels):
            return labels

        labels = labels.copy()
        noise_mask = labels == -1
        valid_mask = labels >= 0

        # If everything is noise or nothing is noise, return as-is
        if not np.any(noise_mask) or not np.any(valid_mask):
            return labels

        # Compute centroids for all valid clusters
        unique_valid_labels = np.unique(labels[valid_mask])
        n_clusters = len(unique_valid_labels)
        
        centroids = np.zeros((n_clusters, embeddings.shape[1]), dtype=np.float64)
        label_to_centroid_idx = {}
        centroid_idx_to_label = {}
        
        for idx, cluster_id in enumerate(unique_valid_labels):
            cluster_points = embeddings[labels == cluster_id]
            centroids[idx] = np.mean(cluster_points, axis=0)
            label_to_centroid_idx[cluster_id] = idx
            centroid_idx_to_label[idx] = cluster_id
            
        # Optional: re-normalize centroids if using cosine
        if self.metric == "cosine":
            norms = np.linalg.norm(centroids, axis=1, keepdims=True)
            # Avoid division by zero
            norms[norms == 0] = 1e-10
            centroids = centroids / norms

        # Get noise embeddings
        noise_indices = np.where(noise_mask)[0]
        noise_embeddings = embeddings[noise_mask]

        # Calculate distances from each noise point to all centroids
        if self.metric == "cosine":
            distances = cosine_distances(noise_embeddings, centroids)
        else:
            distances = euclidean_distances(noise_embeddings, centroids)

        # Assign each noise point to the nearest centroid
        nearest_centroid_indices = np.argmin(distances, axis=1)
        
        for i, noise_idx in enumerate(noise_indices):
            centroid_idx = nearest_centroid_indices[i]
            labels[noise_idx] = centroid_idx_to_label[centroid_idx]

        return labels
