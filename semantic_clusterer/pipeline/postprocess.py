"""Post-processing operations for medium and large pipelines.

This module implements confidence-gated noise recovery and structural
repair, following the patterns from the small pipeline.
"""

import numpy as np
from typing import Dict, Any, Optional
from semantic_clusterer.utils.similarity import normalize_vectors
from semantic_clusterer.optional_deps import _seeded_global_numpy


def recover_noise_with_confidence(
    embeddings: np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    """Recover confidently assignable noise points using adaptive thresholds.
    
    Args:
        embeddings: Shape (N, D) array of embeddings.
        labels: Shape (N,) array of cluster labels (-1 for noise).
    
    Returns:
        Updated labels with confidently assignable noise reassigned.
    """
    from semantic_clusterer.pipeline.quality import compute_cluster_stats
    
    labels = labels.copy()
    noise_indices = np.where(labels == -1)[0]

    if len(noise_indices) == 0:
        return labels

    stats = compute_cluster_stats(embeddings, labels)
    valid_labels = stats["labels"]
    
    if len(valid_labels) == 0:
        return labels

    noise_embeddings = embeddings[noise_indices]
    distances = 1.0 - np.clip(np.dot(noise_embeddings, stats["centroids"].T), -1.0, 1.0)
    best_indices = np.argmin(distances, axis=1)
    best_distances = distances[np.arange(len(noise_indices)), best_indices]

    core_max_distances = np.zeros(len(valid_labels))
    for i, cluster_id in enumerate(valid_labels):
        cluster_distances = 1.0 - np.clip(np.dot(embeddings[labels == cluster_id], stats["centroids"][i]), -1.0, 1.0)
        core_max_distances[i] = np.max(cluster_distances) if len(cluster_distances) > 0 else 0.0

    assignable_thresholds = np.maximum(core_max_distances * 1.5, 0.40)
    best_thresholds = assignable_thresholds[best_indices]

    if len(valid_labels) >= 2:
        top_two = np.partition(distances, kth=min(1, distances.shape[1] - 1), axis=1)[:, :min(2, distances.shape[1])]
        if top_two.shape[1] >= 2:
            second_best = np.max(top_two, axis=1)
        else:
            second_best = best_distances + 0.1
        
        margin = second_best - best_distances
        confident = (best_distances <= best_thresholds) & (margin > 0.01)
    else:
        confident = best_distances <= best_thresholds

    for row_idx, noise_idx in enumerate(noise_indices):
        if confident[row_idx]:
            labels[noise_idx] = int(valid_labels[best_indices[row_idx]])

    return labels


def split_oversized_clusters(
    embeddings: np.ndarray,
    labels: np.ndarray,
    quality_scorer,
    umap_cls=None,
    hdbscan_cls=None,
    max_split_passes: int = 1,
    max_clusters_to_split: int = 2,
    size_threshold_ratio: float = 0.18,
    min_quality_gain: float = 1e-3,
    random_state: int = 42,
    band: Optional[str] = None,
) -> np.ndarray:
    """Split weak oversized clusters only when split improves quality.
    
    Args:
        embeddings: Shape (N, D) array of embeddings.
        labels: Shape (N,) array of cluster labels.
        quality_scorer: Function to compute quality metrics.
        umap_cls: UMAP class for sub-clustering (optional).
        hdbscan_cls: HDBSCAN class for sub-clustering (optional).
        max_split_passes: Maximum refinement passes.
        max_clusters_to_split: Max clusters to split per pass.
        random_state: Random seed for sub-UMAP and sub-HDBSCAN calls.
        band: Dim-band name (e.g. "low", "mid", "high", "xhigh"). When
            supplied, min-cluster-size candidates are drawn from
            ``get_band_grid(band, "small").hdbscan_min_cluster_size_ratios``
            instead of the hardcoded formula.
    
    Returns:
        Updated labels with weak oversized clusters split.
    """
    refined = labels.copy()
    valid_labels, counts = np.unique(refined[refined >= 0], return_counts=True)
    
    if len(valid_labels) == 0:
        return refined

    baseline_metrics = quality_scorer(embeddings, refined)
    baseline_score = baseline_metrics["score"]

    next_label = int(np.max(valid_labels)) + 1
    median_cluster_size = float(np.median(counts)) if len(counts) else 0.0

    for split_pass in range(max_split_passes):
        clusters_split_this_pass = 0
        
        for cluster_id, cluster_size in sorted(
            zip(valid_labels, counts),
            key=lambda item: item[1],
            reverse=True,
        ):
            if clusters_split_this_pass >= max_clusters_to_split:
                break

            cluster_mask = refined == cluster_id
            cluster_points = embeddings[cluster_mask]
            centroid = normalize_vectors(np.mean(cluster_points, axis=0, keepdims=True))[0]
            cohesion = float(np.mean(np.clip(np.dot(cluster_points, centroid), -1.0, 1.0)))

            oversized_threshold = max(
                12,
                int(len(embeddings) * size_threshold_ratio),
                int(max(1.0, median_cluster_size) * 2.0),
            )

            # Don't split small or already-cohesive clusters
            if cluster_size < oversized_threshold or cohesion >= 0.90:
                continue

            # Skip if UMAP/HDBSCAN not available
            if umap_cls is None or hdbscan_cls is None:
                continue

            try:
                split_applied = False
                neighbor_candidates = sorted(
                    list(
                        {
                            int(np.clip(np.sqrt(cluster_size) * 2.0, 8, cluster_size - 1)),
                            int(np.clip(np.log2(max(8, cluster_size)) * 6.0, 8, cluster_size - 1)),
                        }
                    )
                )
                component_candidates = sorted(
                    list(
                        {
                            int(np.clip(6, 3, cluster_points.shape[1])),
                            int(np.clip(10, 3, cluster_points.shape[1])),
                        }
                    )
                )
                min_cluster_candidates = sorted(
                    list(
                        {
                            int(np.clip(cluster_size // 10, 3, cluster_size - 1)),
                            int(np.clip(cluster_size // 6, 5, cluster_size - 1)),
                        }
                    )
                )

                if band is not None:
                    from semantic_clusterer.pipeline.tuning import get_band_grid
                    grid = get_band_grid(band, "small")
                    min_cluster_candidates = sorted(list({
                        max(3, int(ratio * cluster_size))
                        for ratio in grid.hdbscan_min_cluster_size_ratios
                    }))

                cluster_indices = np.where(cluster_mask)[0]

                for sub_neighbors in neighbor_candidates:
                    if sub_neighbors < 5:
                        continue

                    for sub_components in component_candidates:
                        reducer = umap_cls(
                            n_neighbors=sub_neighbors,
                            n_components=sub_components,
                            min_dist=0.0,
                            metric="cosine",
                            n_jobs=-1,
                        )
                        reduced = reducer.fit_transform(cluster_points)

                        for sub_min_cluster_size in min_cluster_candidates:
                            sub_min_samples = max(1, min(sub_min_cluster_size // 2, sub_min_cluster_size - 1))

                            clusterer = hdbscan_cls(
                                min_cluster_size=sub_min_cluster_size,
                                min_samples=sub_min_samples,
                                metric="euclidean",
                                cluster_selection_method="leaf",
                                cluster_selection_epsilon=0.0,
                                gen_min_span_tree=False,
                            )
                            with _seeded_global_numpy(random_state):
                                sub_labels = clusterer.fit_predict(reduced)
                            valid_sub_labels = np.unique(sub_labels[sub_labels >= 0])

                            if len(valid_sub_labels) < 2:
                                continue

                            sub_counts = np.array(
                                [np.sum(sub_labels == label) for label in valid_sub_labels],
                                dtype=np.int32,
                            )
                            if len(sub_counts) == 0 or np.max(sub_counts) >= int(cluster_size * 0.90):
                                continue

                            candidate = refined.copy()
                            largest_sub_label = int(valid_sub_labels[np.argmax(sub_counts)])
                            remap: Dict[int, int] = {largest_sub_label: int(cluster_id)}
                            for sub_label in valid_sub_labels:
                                sub_label = int(sub_label)
                                if sub_label == largest_sub_label:
                                    continue
                                remap[sub_label] = next_label
                                next_label += 1

                            for local_idx, global_idx in enumerate(cluster_indices):
                                sub_label = int(sub_labels[local_idx])
                                if sub_label >= 0:
                                    candidate[global_idx] = remap[sub_label]
                                else:
                                    candidate[global_idx] = int(cluster_id)

                            candidate_metrics = quality_scorer(embeddings, candidate)
                            if candidate_metrics["score"] > (baseline_score + min_quality_gain):
                                refined = candidate
                                baseline_score = candidate_metrics["score"]
                                clusters_split_this_pass += 1
                                split_applied = True
                                break
                        if split_applied:
                            break
                    if split_applied:
                        break
                if split_applied:
                    continue
            except Exception:
                continue
        
        if clusters_split_this_pass == 0:
            break
        
        valid_labels, counts = np.unique(refined[refined >= 0], return_counts=True)

    return refined


def merge_near_duplicate_clusters(
    embeddings: np.ndarray,
    labels: np.ndarray,
    similarity_threshold: Optional[float] = None,
) -> np.ndarray:
    """Merge only near-duplicate clusters.
    
    Clusters with very high centroid similarity and compatible sizes are merged.
    If similarity_threshold is None, it uses an adaptive threshold based on
    the mean pairwise centroid similarity.
    
    Args:
        embeddings: Shape (N, D) array of embeddings.
        labels: Shape (N,) array of cluster labels.
        similarity_threshold: Merge threshold for centroid similarity.
    
    Returns:
        Updated labels with near-duplicates merged.
    """
    from semantic_clusterer.pipeline.quality import compute_cluster_stats
    
    merged = labels.copy()
    stats = compute_cluster_stats(embeddings, merged)
    valid_labels = stats["labels"]

    if len(valid_labels) < 2:
        return merged

    similarities = np.clip(np.dot(stats["centroids"], stats["centroids"].T), -1.0, 1.0)
    upper = similarities[np.triu_indices_from(similarities, k=1)]
    
    if len(upper) == 0:
        return merged
        
    if similarity_threshold is None:
        mean_sim = float(np.mean(upper))
        similarity_threshold = max(0.88, mean_sim + 0.03)
    
    parent = list(range(len(valid_labels)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(len(valid_labels)):
        for j in range(i + 1, len(valid_labels)):
            if similarities[i, j] >= similarity_threshold:
                union(i, j)

    for i in range(len(valid_labels)):
        root_idx = find(i)
        if root_idx != i:
            label_from = int(valid_labels[i])
            label_to = int(valid_labels[root_idx])
            merged[merged == label_from] = label_to

    return merged


def compact_labels(labels: np.ndarray) -> np.ndarray:
    """Remap non-noise labels to a compact 0..K-1 range.
    
    Args:
        labels: Shape (N,) array of cluster labels.
    
    Returns:
        Compacted labels.
    """
    compacted = labels.copy().astype(np.int32)
    valid_labels = [int(label) for label in np.unique(labels) if label >= 0]
    label_map = {label: idx for idx, label in enumerate(valid_labels)}

    for old_label, new_label in label_map.items():
        compacted[labels == old_label] = new_label

    compacted[labels == -1] = -1
    return compacted


def merge_clusters_by_centroid_similarity(
    embeddings: np.ndarray,
    labels: np.ndarray,
    threshold: float,
) -> tuple:
    """Merge clusters whose centroids exceed a cosine similarity threshold.

    Runs union-find iteratively until no more merges occur. More aggressive
    than merge_near_duplicate_clusters because the threshold is lower.

    Args:
        embeddings: (N, D) L2-normalised embeddings.
        labels: (N,) int32 cluster labels. -1 (noise) untouched.
        threshold: Cosine similarity above which two clusters are merged.

    Returns:
        (new_labels, n_merges) — compacted labels and count of pairs merged.
    """
    from semantic_clusterer.pipeline.quality import compute_cluster_stats

    merged = labels.copy()
    total_merges = 0

    # Iterative merge until convergence
    while True:
        stats = compute_cluster_stats(embeddings, merged)
        valid_labels = stats["labels"]
        if len(valid_labels) < 2:
            break

        centroids = stats["centroids"]
        similarities = np.clip(centroids @ centroids.T, -1.0, 1.0)

        # Union-find
        parent = list(range(len(valid_labels)))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> bool:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra
                return True
            return False

        any_merge = False
        for i in range(len(valid_labels)):
            for j in range(i + 1, len(valid_labels)):
                if similarities[i, j] >= threshold:
                    if union(i, j):
                        any_merge = True

        if not any_merge:
            break

        # Apply the merges
        for i in range(len(valid_labels)):
            root_idx = find(i)
            if root_idx != i:
                label_from = int(valid_labels[i])
                label_to = int(valid_labels[root_idx])
                merged[merged == label_from] = label_to
                total_merges += 1

    # Compact
    return compact_labels(merged), total_merges
