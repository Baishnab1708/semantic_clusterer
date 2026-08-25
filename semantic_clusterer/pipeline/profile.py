"""Dataset profiling utilities for adaptive parameter tuning.

This module computes a bounded statistical profile of an embedding dataset
to drive parameter selection in medium and large pipelines.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class DatasetProfile:
    """Statistical profile of an embedding dataset."""

    n_samples: int
    n_features: int
    effective_rank: int
    variance_decay_ratio: float
    local_density_mean: float
    local_density_cv: float
    distance_concentration: float
    duplicate_ratio: float
    near_duplicate_ratio: float
    cluster_tendency: float
    imbalance_tendency: float
    memory_pressure: float

    def effective_rank_ratio(self) -> float:
        """Ratio of effective rank to total features."""
        return float(self.effective_rank) / max(1, self.n_features)


def compute_dataset_profile(
    embeddings: np.ndarray,
    sample_size: int = None,
    rng: Optional[np.random.Generator] = None,
) -> DatasetProfile:
    """Compute a statistical profile of the embedding dataset.

    The implementation is deliberately bounded: it samples the dataset and
    avoids constructing large pairwise distance matrices.

    Parameters
    ----------
    embeddings:
        2-D float32 array of shape (N, D).
    sample_size:
        Optional override for the number of rows to sample.  When ``None``
        the size is chosen automatically as a sub-linear function of N.
    rng:
        A ``numpy.random.Generator`` instance (e.g. from
        ``np.random.default_rng(seed)``).  When ``None`` a default generator
        seeded with 42 is constructed so that legacy callers continue to
        receive deterministic results.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    if embeddings.ndim != 2:
        raise ValueError("embeddings must be a 2D array")

    N, D = embeddings.shape
    if N == 0:
        return DatasetProfile(
            n_samples=0,
            n_features=D,
            effective_rank=0,
            variance_decay_ratio=0.0,
            local_density_mean=0.0,
            local_density_cv=0.0,
            distance_concentration=0.5,
            duplicate_ratio=0.0,
            near_duplicate_ratio=0.0,
            cluster_tendency=0.5,
            imbalance_tendency=0.5,
            memory_pressure=0.0,
        )

    if sample_size is None:
        # Scale sublinearly with N while keeping profiling comfortably bounded.
        sample_size = int(min(N, max(1024, min(4096, 16 * np.sqrt(N)))))

    sample_size = max(1, min(sample_size, N))

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    normalized = embeddings / norms

    if sample_size < N:
        indices = rng.choice(N, size=sample_size, replace=False)
        sample = normalized[indices]
        sample_full = embeddings[indices]
    else:
        sample = normalized
        sample_full = embeddings

    effective_rank = _estimate_effective_rank(sample_full, min(sample_size, 256), rng=rng)
    variance_decay_ratio = _estimate_variance_decay(sample_full, effective_rank, rng=rng)
    local_density_mean, local_density_cv = _estimate_local_density(sample, k=20)
    distance_concentration = _estimate_distance_concentration(sample, pair_count=8192, rng=rng)
    duplicate_ratio, near_duplicate_ratio = _estimate_duplicate_ratios(sample)
    cluster_tendency = _estimate_cluster_tendency(sample, rng=rng)
    imbalance_tendency = _estimate_imbalance_tendency(sample, rng=rng)
    memory_pressure = _estimate_memory_pressure(N, D)

    return DatasetProfile(
        n_samples=N,
        n_features=D,
        effective_rank=int(effective_rank),
        variance_decay_ratio=float(variance_decay_ratio),
        local_density_mean=float(local_density_mean),
        local_density_cv=float(local_density_cv),
        distance_concentration=float(distance_concentration),
        duplicate_ratio=float(duplicate_ratio),
        near_duplicate_ratio=float(near_duplicate_ratio),
        cluster_tendency=float(cluster_tendency),
        imbalance_tendency=float(imbalance_tendency),
        memory_pressure=float(memory_pressure),
    )


def _estimate_effective_rank(
    embeddings: np.ndarray,
    max_components: int,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """Estimate effective rank using randomized SVD."""
    if rng is None:
        rng = np.random.default_rng(42)

    n_components = min(max_components, embeddings.shape[0] - 1, embeddings.shape[1])
    if n_components < 1:
        return float(embeddings.shape[1])

    try:
        from sklearn.decomposition import TruncatedSVD

        svd = TruncatedSVD(
            n_components=n_components,
            random_state=int(rng.integers(0, 2**32)),
        )
        svd.fit(embeddings)
        cumsum = np.cumsum(svd.explained_variance_ratio_)
        rank = int(np.searchsorted(cumsum, 0.90)) + 1
        return float(min(rank, n_components))
    except Exception:
        return float(min(max(1, embeddings.shape[1] // 2), embeddings.shape[1]))


def _estimate_variance_decay(
    embeddings: np.ndarray,
    effective_rank: int,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """Estimate ratio of captured variance at effective rank."""
    if rng is None:
        rng = np.random.default_rng(42)

    try:
        from sklearn.decomposition import TruncatedSVD

        n_comp = min(int(effective_rank), embeddings.shape[0] - 1, embeddings.shape[1])
        if n_comp < 1:
            return 0.0

        svd = TruncatedSVD(
            n_components=n_comp,
            random_state=int(rng.integers(0, 2**32)),
        )
        svd.fit(embeddings)
        return float(np.sum(svd.explained_variance_ratio_))
    except Exception:
        return 0.5


def _estimate_local_density(normalized_embeddings: np.ndarray, k: int = 20) -> Tuple[float, float]:
    """Estimate local density using k-NN distances without full pairwise matrices."""
    n = len(normalized_embeddings)
    if n < 2:
        return 0.0, 0.0

    k = min(k, n - 1)
    if k < 1:
        return 0.0, 0.0

    try:
        from sklearn.neighbors import NearestNeighbors

        nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine", algorithm="brute")
        nn.fit(normalized_embeddings)
        distances, _ = nn.kneighbors(normalized_embeddings)
        kth_distances = distances[:, -1]
    except Exception:
        similarities = np.clip(normalized_embeddings @ normalized_embeddings.T, -1.0, 1.0)
        distances = 1.0 - similarities
        np.fill_diagonal(distances, np.inf)
        kth_distances = np.partition(distances, kth=k - 1, axis=1)[:, k - 1]

    mean_density = float(np.mean(kth_distances))
    std_density = float(np.std(kth_distances))
    cv = std_density / max(mean_density, 1e-8)
    return mean_density, min(cv, 2.0)


def _estimate_distance_concentration(
    normalized_embeddings: np.ndarray,
    pair_count: int = 8192,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """Estimate concentration of pairwise cosine distances from sampled pairs."""
    if rng is None:
        rng = np.random.default_rng(42)

    n = len(normalized_embeddings)
    if n < 10:
        return 0.5

    pair_count = min(pair_count, n * (n - 1) // 2)
    if pair_count <= 0:
        return 0.5

    idx_a = rng.integers(0, n, size=pair_count)
    idx_b = rng.integers(0, n, size=pair_count)
    different = idx_a != idx_b
    if not np.any(different):
        return 0.5

    idx_a = idx_a[different]
    idx_b = idx_b[different]
    dots = np.einsum("ij,ij->i", normalized_embeddings[idx_a], normalized_embeddings[idx_b])
    distances = 1.0 - np.clip(dots, -1.0, 1.0)

    min_d = float(np.min(distances))
    max_d = float(np.max(distances))
    mean_d = float(np.mean(distances))
    if max_d - min_d < 1e-8:
        return 0.5

    concentration = 1.0 - (mean_d - min_d) / (max_d - min_d)
    return float(np.clip(concentration, 0.0, 1.0))


def _estimate_duplicate_ratios(normalized_embeddings: np.ndarray) -> Tuple[float, float]:
    """Estimate fraction of duplicates and near-duplicates."""
    n = len(normalized_embeddings)
    if n < 2:
        return 0.0, 0.0

    try:
        from sklearn.neighbors import NearestNeighbors

        nn = NearestNeighbors(n_neighbors=2, metric="cosine", algorithm="brute")
        nn.fit(normalized_embeddings)
        distances, _ = nn.kneighbors(normalized_embeddings)
        nn_distances = distances[:, 1]
    except Exception:
        similarities = np.clip(normalized_embeddings @ normalized_embeddings.T, -1.0, 1.0)
        distances = 1.0 - similarities
        np.fill_diagonal(distances, np.inf)
        nn_distances = np.min(distances, axis=1)

    duplicate_ratio = float(np.mean(nn_distances < 1e-4))
    near_duplicate_ratio = float(np.mean(nn_distances < 1e-2))
    return duplicate_ratio, near_duplicate_ratio


def _estimate_cluster_tendency(
    normalized_embeddings: np.ndarray,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """Estimate clusterability with a bounded Hopkins-style proxy."""
    if rng is None:
        rng = np.random.default_rng(42)

    n = len(normalized_embeddings)
    if n < 10:
        return 0.5

    sample_size = min(n, 256)
    indices = rng.choice(n, size=sample_size, replace=False)
    sample = normalized_embeddings[indices]
    k = min(5, sample_size - 1)
    if k < 1:
        return 0.5

    real_mean, _ = _estimate_local_density(sample, k=k)

    random_data = rng.standard_normal(size=(sample_size, normalized_embeddings.shape[1]))
    random_norms = np.linalg.norm(random_data, axis=1, keepdims=True)
    random_data = random_data / np.maximum(random_norms, 1e-8)
    random_mean, _ = _estimate_local_density(random_data, k=k)

    ratio = random_mean / max(real_mean, 1e-8)
    tendency = min(ratio, 2.0) / 2.0
    return float(np.clip(tendency, 0.0, 1.0))


def _estimate_imbalance_tendency(
    normalized_embeddings: np.ndarray,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """Estimate cluster imbalance using a small KMeans probe."""
    if rng is None:
        rng = np.random.default_rng(42)

    n = len(normalized_embeddings)
    if n < 20:
        return 0.5

    sample_size = min(n, 300)
    indices = rng.choice(n, size=sample_size, replace=False)
    sample = normalized_embeddings[indices]

    try:
        from sklearn.cluster import KMeans

        n_clusters = max(2, min(sample_size, int(np.sqrt(sample_size))))
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=int(rng.integers(0, 2**32)),
            n_init=2,
        )
        labels = kmeans.fit_predict(sample)

        sizes = np.bincount(labels)
        sizes = sizes[sizes > 0]
        probs = sizes / np.sum(sizes)
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        max_entropy = np.log(len(sizes))
        norm_entropy = entropy / max(max_entropy, 1e-8)
        imbalance = 1.0 - norm_entropy
        return float(np.clip(imbalance, 0.0, 1.0))
    except Exception:
        return 0.5


def _estimate_memory_pressure(n_samples: int, n_features: int) -> float:
    """Estimate memory pressure as a function of dataset size and dimensionality."""
    embedding_bytes = n_samples * n_features * 4  # float32
    reduced_bytes = n_samples * min(max(48, n_features // 2), 256) * 4
    working_bytes = embedding_bytes + reduced_bytes + (n_samples * 64)

    # Keep the heuristic conservative: once expected working set approaches
    # a few gigabytes, prefer more memory-aware execution paths.
    budget_bytes = 6e9
    pressure = working_bytes / budget_bytes
    return float(np.clip(pressure, 0.0, 1.0))
