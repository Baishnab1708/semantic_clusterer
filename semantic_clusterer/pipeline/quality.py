from __future__ import annotations

import numpy as np
from typing import Dict, Mapping, Optional, Tuple
from semantic_clusterer.utils.similarity import normalize_vectors


def clip01(value: float) -> float:
    """Clamp a floating-point value to [0, 1]."""
    return float(np.clip(value, 0.0, 1.0))


# Score weights — kept explicit so pipelines don't hide tradeoffs in local formulas.
DEFAULT_CLUSTER_SCORE_WEIGHTS: Dict[str, float] = {
    "density": 0.15,
    "coverage": 0.15,
    "cohesion": 0.20,
    "separation": 0.22,
    "stability": 0.10,
    "fragmentation_penalty": 0.10,
    "largest_cluster_penalty": 0.10,
}
DEFAULT_LARGEST_CLUSTER_BASELINE = 0.10


def resolve_cluster_score_weights(
    weights: Optional[Mapping[str, float]] = None,
) -> Dict[str, float]:
    """Merge caller overrides with the shared default scoring weights."""
    resolved = dict(DEFAULT_CLUSTER_SCORE_WEIGHTS)
    if weights:
        resolved.update({key: float(value) for key, value in weights.items()})
    return resolved


def compute_cluster_stats(
    embeddings: np.ndarray,
    labels: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Compute centroid, cohesion, and size stats for non-noise clusters.
    
    Args:
        embeddings: Shape (N, D) array of embeddings.
        labels: Shape (N,) array of cluster labels (-1 for noise).
    
    Returns:
        Dictionary with:
        - 'labels': Unique non-noise cluster IDs
        - 'centroids': Cluster centroids
        - 'sizes': Cluster sizes
        - 'cohesions': Mean within-cluster similarity
        - 'thresholds': Adaptive per-cluster assignment thresholds
    """
    valid_labels = np.unique(labels[labels >= 0]).astype(np.int32)

    if len(valid_labels) == 0:
        return {
            "labels": valid_labels,
            "centroids": np.empty((0, embeddings.shape[1]), dtype=np.float32),
            "sizes": np.empty(0, dtype=np.int32),
            "cohesions": np.empty(0, dtype=np.float32),
            "thresholds": np.empty(0, dtype=np.float32),
        }

    centroids = np.zeros((len(valid_labels), embeddings.shape[1]), dtype=np.float32)
    sizes = np.zeros(len(valid_labels), dtype=np.int32)
    cohesions = np.zeros(len(valid_labels), dtype=np.float32)
    thresholds = np.zeros(len(valid_labels), dtype=np.float32)

    for idx, cluster_id in enumerate(valid_labels):
        cluster_points = embeddings[labels == cluster_id]
        sizes[idx] = len(cluster_points)

        centroid = normalize_vectors(np.mean(cluster_points, axis=0, keepdims=True))[0]
        centroids[idx] = centroid.astype(np.float32)

        similarities = np.clip(np.dot(cluster_points, centroid), -1.0, 1.0)
        distances = 1.0 - similarities

        cohesions[idx] = float(np.mean(similarities))
        thresholds[idx] = float(np.clip(np.percentile(distances, 90) + 0.03, 0.05, 0.35))

    return {
        "labels": valid_labels,
        "centroids": centroids,
        "sizes": sizes,
        "cohesions": cohesions,
        "thresholds": thresholds,
    }


def score_clustering(
    embeddings: np.ndarray,
    labels: np.ndarray,
    density_score: Optional[float] = None,
    weights: Optional[Mapping[str, float]] = None,
    largest_cluster_baseline: float = DEFAULT_LARGEST_CLUSTER_BASELINE,
) -> Dict[str, float]:
    """Score a clustering with intrinsic quality metrics.
    
    Args:
        embeddings: Shape (N, D) array of embeddings.
        labels: Shape (N,) array of cluster labels (-1 for noise).
        density_score: Optional HDBSCAN relative_validity score.
            None means density is unavailable and its weight is redistributed.
    
    Returns:
        Dictionary with individual scores and composite score.
    """
    n_samples = len(labels)
    stats = compute_cluster_stats(embeddings, labels)
    valid_labels = stats["labels"]
    sizes = stats["sizes"]

    if n_samples == 0 or len(valid_labels) == 0:
        return {
            "score": 0.0,
            "density": 0.0,
            "coverage": 0.0,
            "cohesion": 0.0,
            "separation": 0.0,
            "stability": 0.0,
            "largest_ratio": 1.0,
            "fragmentation": 1.0,
            "n_clusters": 0,
            "noise_ratio": 1.0,
        }

    # Compute individual metrics
    noise_ratio = float(np.mean(labels == -1))
    coverage = 1.0 - noise_ratio

    resolved_weights = resolve_cluster_score_weights(weights)

    # Density: use DBCV when available, otherwise redistribute weight
    if density_score is not None and density_score >= -1.0:
        density = clip01((float(density_score) + 1.0) / 2.0)
    else:
        density = 0.0
        # Redistribute density weight proportionally to other positive terms
        density_w = resolved_weights["density"]
        resolved_weights["density"] = 0.0
        other_pos_keys = [k for k, v in resolved_weights.items()
                         if not k.endswith("penalty") and k != "density"]
        other_pos_sum = sum(resolved_weights[k] for k in other_pos_keys)
        if other_pos_sum > 0:
            for k in other_pos_keys:
                resolved_weights[k] += density_w * (resolved_weights[k] / other_pos_sum)

    weighted_cohesion = float(np.average(stats["cohesions"], weights=sizes))
    cohesion = clip01((weighted_cohesion + 1.0) / 2.0)

    if len(valid_labels) >= 2:
        centroid_similarities = np.clip(np.dot(stats["centroids"], stats["centroids"].T), -1.0, 1.0)
        upper = centroid_similarities[np.triu_indices_from(centroid_similarities, k=1)]
        separation = clip01(float(np.mean(1.0 - upper))) if len(upper) else 0.0

        # Compensate for cosine concentration in high-dimensional spaces.
        # High-dim embeddings naturally have higher centroid similarities,
        # which compresses separation into a narrow low range. Rescale to
        # give high-dim embeddings a fair comparison.
        D_eff = embeddings.shape[1]
        if D_eff >= 1024:
            # High-dim: centroids are naturally close. Scale separation up.
            sep_scale = min(2.5, 1.0 + (D_eff - 512) / 1024.0)
            separation = clip01(separation * sep_scale)
        elif D_eff >= 512:
            sep_scale = 1.0 + (D_eff - 512) / 2048.0
            separation = clip01(separation * sep_scale)

        # Gentle size-uniformity dampening: raw separation rewards merging
        # everything into a few giant blobs (fewer centroids = more spread).
        # Dampen when the largest cluster is >2× the median — only blob-
        # dominated solutions are affected; even-sized clusters are untouched.
        if len(sizes) >= 2:
            median_size = float(np.median(sizes))
            largest_size = float(np.max(sizes))
            if largest_size > 2.0 * median_size and median_size > 0:
                dampening = float(np.sqrt(median_size / largest_size))
                separation = clip01(separation * dampening)
                
        # Zero out separation for K <= 3 on larger datasets (to reward correct fragmentation),
        # but keep it active for tiny datasets (N <= 150) where K=2 or K=3 are highly natural.
        if len(valid_labels) <= 3 and n_samples > 150:
            separation = 0.0

        # Dampen separation for K far below the expected cluster count on
        # larger datasets.  Very few clusters trivially achieve high
        # separation (centroids are naturally far apart when everything is
        # merged into a handful of blobs).  This mirrors the fragmentation
        # penalty symmetrically — over-fragmentation penalised above,
        # over-merging dampened here.
        if len(valid_labels) >= 4 and n_samples > 500:
            _count_ref = max(2, min(22, int(np.sqrt(max(1, n_samples) / 3.0))))
            k_threshold = max(4, _count_ref // 2)
            if len(valid_labels) < k_threshold:
                sep_scale = len(valid_labels) / float(k_threshold)
                separation *= sep_scale
    else:
        separation = 0.0

    # Stability: lower variance in cluster sizes indicates more stable structure
    if len(sizes) > 1:
        size_cv = float(np.std(sizes) / np.mean(sizes))
        stability = clip01(1.0 - min(size_cv, 2.0) / 2.0)
    else:
        stability = 0.5

    largest_ratio = float(np.max(sizes) / max(1, n_samples))

    # Fragmentation: combine size-relative and count-relative signals so
    # balanced/many-small-clusters get penalised even when no single
    # cluster is below the size-relative micro-threshold.
    n_clusters = max(1, len(valid_labels))
    expected_cluster_size = n_samples / n_clusters
    micro_threshold = max(3, int(expected_cluster_size * 0.25))
    size_fragmentation = float(np.mean(sizes <= micro_threshold))

    # Count-based fragmentation: K > baseline starts incurring a penalty.
    # baseline = min(22, sqrt(N/3)):
    #   N=116  → baseline≈6   (tiny: K=8 OK, K=20 penalised — expected)
    #   N=1500 → baseline≈22  (small: K=20 penalty-free ✓)
    #   N=15000→ baseline≈22  (medium: K=22 penalty-free, K=30+ penalised ✓)
    # The cap at 22 prevents the baseline from growing unboundedly on
    # very large corpora, keeping K>22 penalised for balanced granularity.
    count_baseline = max(2, min(22, int(np.sqrt(max(1, n_samples) / 3.0))))
    count_fragmentation = clip01((n_clusters - count_baseline) / max(1.0, count_baseline))
    fragmentation = max(size_fragmentation, count_fragmentation)

    largest_cluster_penalty = max(0.0, largest_ratio - largest_cluster_baseline)

    # Blob penalty: penalize clusters with very low internal cohesion (garbage clusters)
    blob_penalty = 0.0
    if len(valid_labels) > 0:
        min_cohesion = float(np.min(stats["cohesions"]))
        if min_cohesion < 0.15:
            blob_penalty = 0.15 - min_cohesion

    # Composite score: weighted objective
    score = (
        resolved_weights["density"] * density
        + resolved_weights["coverage"] * coverage
        + resolved_weights["cohesion"] * cohesion
        + resolved_weights["separation"] * separation
        + resolved_weights["stability"] * stability
        - resolved_weights["fragmentation_penalty"] * fragmentation
        - resolved_weights["largest_cluster_penalty"] * largest_cluster_penalty
        - blob_penalty
    )

    score = clip01(score)

    return {
        "score": float(score),
        "density": float(density),
        "coverage": float(coverage),
        "cohesion": float(cohesion),
        "separation": float(separation),
        "stability": float(stability),
        "largest_ratio": float(largest_ratio),
        "fragmentation": float(fragmentation),
        "n_clusters": int(len(valid_labels)),
        "noise_ratio": float(noise_ratio),
    }


def should_trigger_refinement(
    metrics: Dict[str, float],
    thresholds: Dict[str, float],
) -> Tuple[bool, str]:
    """Determine whether to trigger a refinement pass.
    
    Args:
        metrics: Clustering metrics from score_clustering.
        thresholds: Threshold dictionary from tuning module.
    
    Returns:
        (should_refine, reason)
    """
    reasons = []

    if metrics["coverage"] < thresholds["min_coverage"]:
        reasons.append("low_coverage")

    if metrics["noise_ratio"] > thresholds["max_noise"]:
        reasons.append("high_noise")

    if metrics["largest_ratio"] > thresholds["max_giant_ratio"]:
        reasons.append("giant_cluster")

    if metrics.get("stability", 0.5) < thresholds.get("min_stability", 0.35):
        reasons.append("low_stability")

    if reasons:
        return True, ",".join(reasons)
    else:
        return False, ""


def compare_clustering_pair(
    embedding1: np.ndarray,
    labels1: np.ndarray,
    embedding2: np.ndarray,
    labels2: np.ndarray,
) -> float:
    """Compare two clusterings using normalized mutual information.

    Returns:
        NMI score in [0, 1] where 1 = identical clustering.
    """
    from sklearn.metrics import normalized_mutual_info_score

    if len(labels1) != len(labels2):
        return 0.0

    mask = (labels1 >= 0) & (labels2 >= 0)
    if not np.any(mask):
        return 0.0

    try:
        nmi = normalized_mutual_info_score(labels1[mask], labels2[mask])
        return float(nmi)
    except Exception:
        return 0.0
