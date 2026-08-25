"""Adaptive parameter tuning for medium and large pipelines.

This module generates parameter candidates and search spaces based on
dataset profiles, following the principles from the small pipeline but
adapted for medium and large scale.

It also exposes ``get_band_grid(band, tier) -> BandGrid`` which is the
single source of truth for all per-(band, tier) parameter grids used by
every pipeline tier.
"""

import numpy as np
from typing import List, Dict, Literal
from semantic_clusterer.pipeline.profile import DatasetProfile
from semantic_clusterer.dim_bands import BandGrid, DimBand


# ---------------------------------------------------------------------------
# Concrete grids per (band, tier)
# ---------------------------------------------------------------------------

# Tier: "small"
_SMALL_GRIDS: Dict[str, BandGrid] = {
    "low": BandGrid(
        pca_targets=[384],  # placeholder for D; pipeline substitutes actual D at runtime
        umap_n_neighbors=[20, 25, 30, 40],
        umap_n_components=[8, 10, 12],
        hdbscan_min_cluster_size_ratios=[0.020, 0.035, 0.06, 0.10, 0.15],
        hdbscan_min_samples=[1, 2, 4],
        hdbscan_methods=["eom", "leaf"],
    ),
    "mid": BandGrid(
        pca_targets=[256, 384],
        umap_n_neighbors=[15, 20, 25, 30],
        umap_n_components=[6, 8, 9],
        hdbscan_min_cluster_size_ratios=[0.020, 0.04, 0.07, 0.11],
        hdbscan_min_samples=[1, 2, 4],
        hdbscan_methods=["eom", "leaf"],
    ),
    "high": BandGrid(
        pca_targets=[128, 192, 256],
        umap_n_neighbors=[15, 20, 25],
        umap_n_components=[6, 8],
        hdbscan_min_cluster_size_ratios=[0.020, 0.04, 0.07],
        hdbscan_min_samples=[1, 2, 4],
        hdbscan_methods=["eom", "leaf"],
    ),
    "xhigh": BandGrid(
        pca_targets=[64, 96, 128],
        umap_n_neighbors=[15, 20],
        umap_n_components=[6, 8],
        hdbscan_min_cluster_size_ratios=[0.025, 0.05, 0.08],
        hdbscan_min_samples=[1, 2, 4],
        hdbscan_methods=["eom", "leaf"],
    ),
}

# Tier: "medium"
_MEDIUM_GRIDS: Dict[str, BandGrid] = {
    "low": BandGrid(
        pca_targets=[320, 384],
        umap_n_neighbors=[35],
        umap_n_components=[10],
        hdbscan_min_cluster_size_ratios=[0.0020, 0.0040, 0.0060],
        hdbscan_min_samples=[1, 2],
        hdbscan_methods=["eom", "leaf"],
    ),
    "mid": BandGrid(
        pca_targets=[256, 320],
        umap_n_neighbors=[30],
        umap_n_components=[8],
        hdbscan_min_cluster_size_ratios=[0.0020, 0.0040, 0.0060],
        hdbscan_min_samples=[1, 2],
        hdbscan_methods=["eom", "leaf"],
    ),
    "high": BandGrid(
        pca_targets=[192, 256],
        umap_n_neighbors=[25],
        umap_n_components=[8],
        hdbscan_min_cluster_size_ratios=[0.0008, 0.0015, 0.0030],
        hdbscan_min_samples=[1, 2],
        hdbscan_methods=["eom", "leaf"],
    ),
    "xhigh": BandGrid(
        pca_targets=[96, 128],
        umap_n_neighbors=[20],
        umap_n_components=[8],
        hdbscan_min_cluster_size_ratios=[0.0008, 0.0015, 0.0030],
        hdbscan_min_samples=[1, 2],
        hdbscan_methods=["eom", "leaf"],
    ),
}

# Tier: "large"
# umap_n_neighbors and umap_n_components reuse the medium values for the same band
# (they are needed by BandGrid but the large pipeline uses them for coarse-partition UMAP).
_LARGE_GRIDS: Dict[str, BandGrid] = {
    "low": BandGrid(
        pca_targets=[192, 256],
        umap_n_neighbors=[25],
        umap_n_components=[10],
        hdbscan_min_cluster_size_ratios=[0.0010, 0.0020],
        hdbscan_min_samples=[2, 4],
        hdbscan_methods=["eom", "leaf"],
    ),
    "mid": BandGrid(
        pca_targets=[128, 192],
        umap_n_neighbors=[20],
        umap_n_components=[8],
        hdbscan_min_cluster_size_ratios=[0.0010, 0.0020],
        hdbscan_min_samples=[2, 4],
        hdbscan_methods=["eom", "leaf"],
    ),
    "high": BandGrid(
        pca_targets=[96, 128],
        umap_n_neighbors=[15],
        umap_n_components=[6],
        hdbscan_min_cluster_size_ratios=[0.0010, 0.0020],
        hdbscan_min_samples=[2, 4],
        hdbscan_methods=["eom", "leaf"],
    ),
    "xhigh": BandGrid(
        pca_targets=[64, 96],
        umap_n_neighbors=[15],
        umap_n_components=[6],
        hdbscan_min_cluster_size_ratios=[0.0010, 0.0020],
        hdbscan_min_samples=[2, 4],
        hdbscan_methods=["eom", "leaf"],
    ),
}

# Tier: "tiny"
# tiny_k_grid is the same across all bands; umap fields are inherited from small
# (they are not used by the tiny pipeline but BandGrid requires them).
_TINY_GRIDS: Dict[str, BandGrid] = {
    "low": BandGrid(
        pca_targets=[384],               # same as small/low placeholder
        umap_n_neighbors=[20, 25, 30, 40],  # same as small/low
        umap_n_components=[8, 10, 12],      # same as small/low
        hdbscan_min_cluster_size_ratios=[0.035, 0.06, 0.10, 0.15],
        hdbscan_min_samples=[1, 2, 4],
        hdbscan_methods=["eom", "leaf"],
        tiny_k_grid=[2, 3, 5, 8, 12],
    ),
    "mid": BandGrid(
        pca_targets=[256, 384],
        umap_n_neighbors=[15, 20, 25, 30],
        umap_n_components=[6, 8, 9],
        hdbscan_min_cluster_size_ratios=[0.020, 0.04, 0.07, 0.11],
        hdbscan_min_samples=[1, 2, 4],
        hdbscan_methods=["eom", "leaf"],
        tiny_k_grid=[2, 3, 5, 8, 12],
    ),
    "high": BandGrid(
        pca_targets=[128, 192, 256],
        umap_n_neighbors=[15, 20, 25],
        umap_n_components=[6, 8],
        hdbscan_min_cluster_size_ratios=[0.020, 0.04, 0.07],
        hdbscan_min_samples=[1, 2, 4],
        hdbscan_methods=["eom", "leaf"],
        tiny_k_grid=[2, 3, 5, 8, 12],
    ),
    "xhigh": BandGrid(
        pca_targets=[64, 96, 128],
        umap_n_neighbors=[15, 20],
        umap_n_components=[6, 8],
        hdbscan_min_cluster_size_ratios=[0.025, 0.05, 0.08],
        hdbscan_min_samples=[1, 2, 4],
        hdbscan_methods=["eom", "leaf"],
        tiny_k_grid=[2, 3, 5, 8, 12],
    ),
}

_ALL_GRIDS: Dict[str, Dict[str, BandGrid]] = {
    "small": _SMALL_GRIDS,
    "medium": _MEDIUM_GRIDS,
    "large": _LARGE_GRIDS,
    "tiny": _TINY_GRIDS,
}

_VALID_BANDS = frozenset({"low", "mid", "high", "xhigh"})
_VALID_TIERS = frozenset({"small", "medium", "large", "tiny"})


def get_band_grid(band: str, tier: str) -> BandGrid:
    """Return the concrete parameter grid for a given (band, tier) pair.

    This is the single source of truth for all per-(band, tier) parameter
    grids used by every pipeline tier.  Every pipeline should call this
    function rather than hard-coding candidate lists.

    Args:
        band: One of ``"low"``, ``"mid"``, ``"high"``, ``"xhigh"``.
        tier: One of ``"small"``, ``"medium"``, ``"large"``, ``"tiny"``.

    Returns:
        A frozen :class:`~semantic_clusterer.dim_bands.BandGrid` instance
        whose every list field is non-empty.

    Raises:
        ValueError: If ``band`` or ``tier`` is not one of the supported values.

    Example::

        from semantic_clusterer.dim_bands import resolve_dim_band
        from semantic_clusterer.pipeline.tuning import get_band_grid

        band = resolve_dim_band(embeddings.shape[1])
        grid = get_band_grid(band, "medium")
        for mcs_ratio in grid.hdbscan_min_cluster_size_ratios:
            mcs = max(5, int(mcs_ratio * N))
            ...
    """
    if band not in _VALID_BANDS:
        raise ValueError(
            f"Unknown band {band!r}. Must be one of {sorted(_VALID_BANDS)}."
        )
    if tier not in _VALID_TIERS:
        raise ValueError(
            f"Unknown tier {tier!r}. Must be one of {sorted(_VALID_TIERS)}."
        )
    return _ALL_GRIDS[tier][band]


def compute_medium_reduction_dimension(
    profile: DatasetProfile,
    base_anchor_fn=None,
) -> int:
    """Compute target reduction dimension for medium pipeline.
    
    Uses the base anchor (config.get_reduction_components) and adjusts
    based on dataset profile characteristics.
    
    Args:
        profile: DatasetProfile of the dataset.
        base_anchor_fn: Function that computes base dimension.
            If None, uses a simple default.
    
    Returns:
        Target reduction dimension.
    """
    if base_anchor_fn is None:
        # Simple fallback: sqrt(D) * log2(N)^0.7
        D = profile.n_features
        N = profile.n_samples
        base_dim = int(np.sqrt(D) * (np.log2(max(10, N)) ** 0.7))
    else:
        base_dim = base_anchor_fn(profile.n_features, profile.n_samples)
    
    # Adjustments based on profile
    effective_rank_ratio = profile.effective_rank_ratio()
    
    # Increase dimensions if effective rank is high or distance concentration is poor
    if effective_rank_ratio > 0.35 or profile.distance_concentration < 0.12:
        base_dim = int(base_dim * 1.20)
    
    # Reduce dimensions if variance is concentrated or duplicates dominate
    if profile.near_duplicate_ratio > 0.10 or effective_rank_ratio < 0.12:
        base_dim = int(base_dim * 0.85)
    
    # Clip to sensible range
    min_dim = 96 if profile.n_features <= 512 else 48
    max_dim = min(profile.n_features, 384)
    
    # Round to multiple of 8 for alignment
    result = int(np.round(base_dim / 8) * 8)
    result = np.clip(result, min_dim, max_dim)
    
    return int(result)


def compute_large_reduction_dimension(
    profile: DatasetProfile,
    base_anchor_fn=None,
) -> int:
    """Compute target reduction dimension for large pipeline.
    
    Uses more aggressive reduction than medium to prioritize speed.
    
    Args:
        profile: DatasetProfile of the dataset.
        base_anchor_fn: Function that computes base dimension.
    
    Returns:
        Target reduction dimension.
    """
    if base_anchor_fn is None:
        D = profile.n_features
        N = profile.n_samples
        base_dim = int(np.sqrt(D) * (np.log2(max(10, N)) ** 0.6))  # Slightly lower exponent
    else:
        base_dim = base_anchor_fn(profile.n_features, profile.n_samples)
    
    effective_rank_ratio = profile.effective_rank_ratio()
    
    # Increase dimensions if effective rank is high
    if effective_rank_ratio > 0.35:
        base_dim = int(base_dim * 1.15)
    
    # Reduce dimensions for memory pressure or duplicates
    if profile.memory_pressure > 0.5 or profile.near_duplicate_ratio > 0.10:
        base_dim = int(base_dim * 0.80)
    
    # Tighter range for large: prioritize speed
    min_dim = 48
    max_dim = min(profile.n_features, 256)
    
    result = int(np.round(base_dim / 8) * 8)
    result = np.clip(result, min_dim, max_dim)
    
    return int(result)


def compute_reduction_candidates(
    profile: DatasetProfile,
    base_dimension: int,
    num_candidates: int = 3,
) -> List[int]:
    """Generate reduction dimension candidates around a base dimension.
    
    Args:
        profile: DatasetProfile of the dataset.
        base_dimension: Center dimension for candidates.
        num_candidates: Number of candidates to generate (default 3).
    
    Returns:
        List of unique dimension candidates.
    """
    candidates = []
    
    if num_candidates >= 3:
        candidates.append(int(base_dimension * 0.80))
    
    candidates.append(base_dimension)
    
    if num_candidates >= 2:
        candidates.append(int(base_dimension * 1.20))
    
    # Clip to valid range
    min_dim = 48
    max_dim = min(profile.n_features, 384)
    
    # Align to multiples of 8 and deduplicate
    candidates = [int(np.round(c / 8) * 8) for c in candidates]
    candidates = [np.clip(c, min_dim, max_dim) for c in candidates]
    candidates = sorted(list(set(candidates)))
    
    return candidates


def compute_medium_hdbscan_candidates(
    profile: DatasetProfile,
) -> Dict[str, List]:
    """Generate HDBSCAN parameter candidates for medium pipeline.
    
    Args:
        profile: DatasetProfile of the dataset.
    
    Returns:
        Dictionary with keys:
        - 'min_cluster_sizes': List of min_cluster_size candidates
        - 'min_samples_sets': List of min_samples candidates for different scenarios
        - 'methods': List of cluster selection methods to try
    """
    N = profile.n_samples
    D = profile.n_features
    
    # We use high floor sizes across all models to prefer generalized macro-level
    # topics and prevent over-fragmentation (micro-clusters) on powerful models.
    min_cluster_sizes = [
        max(10, int(0.0010 * N)),
        max(15, int(0.0020 * N)),
        max(20, int(0.0040 * N)),
        max(30, int(0.0060 * N)),
    ]

    # Deduplicate and sort
    min_cluster_sizes = sorted(list(set(min_cluster_sizes)))

    # Adaptive min_samples based on density profile
    if profile.local_density_cv < 0.35:
        min_samples = [1, 2, 4]
    elif profile.local_density_cv > 0.80:
        min_samples = [4, 8, 12]
    else:
        min_samples = [2, 4, 8]

    # Cluster selection methods
    methods = ["eom"]
    
    # Add 'leaf' if data might have blob risk or local density variation
    if profile.cluster_tendency > 0.65 or profile.local_density_cv > 0.60:
        methods.append("leaf")
    
    return {
        "min_cluster_sizes": min_cluster_sizes,
        "min_samples": min_samples,
        "methods": methods,
    }


def compute_large_target_shard_size(
    profile: DatasetProfile,
) -> int:
    """Compute target shard size for the large pipeline."""
    D = profile.n_features
    target_shard_size = 1500
    effective_rank_ratio = profile.effective_rank_ratio()

    if D > 1536 or effective_rank_ratio > 0.35:
        target_shard_size = 1000

    if profile.near_duplicate_ratio > 0.10 or profile.local_density_cv < 0.35:
        target_shard_size = 2500

    if profile.memory_pressure > 0.6:
        target_shard_size = min(target_shard_size, 1200)

    return int(np.clip(target_shard_size, 800, 4000))


def compute_large_coarse_partition_count(
    profile: DatasetProfile,
) -> int:
    """Compute target number of coarse K-Means partitions."""
    N = profile.n_samples
    if N <= 1:
        return max(1, N)

    target_shard_size = compute_large_target_shard_size(profile)
    n_coarse = int(np.ceil(N / max(1, target_shard_size)))
    return int(np.clip(n_coarse, 2, min(256, N)))


def compute_large_hdbscan_candidates(
    shard_size: int,
    embedding_dim: int = 768,
) -> List[int]:
    """Generate HDBSCAN min_cluster_size candidates for a shard.
    
    Args:
        shard_size: Number of points in the shard.
        embedding_dim: Original embedding dimensionality.
    
    Returns:
        List of min_cluster_size candidates.
    """
    candidates = [
        max(10, int(0.010 * shard_size)),
        max(15, int(0.020 * shard_size)),
        max(20, int(0.030 * shard_size)),
    ]
    
    return sorted(list(set(candidates)))


def compute_large_min_samples_candidates(
    profile: DatasetProfile,
    min_cluster_size: int,
) -> List[int]:
    """Generate min_samples candidates for a large-pipeline shard."""
    if profile.local_density_cv < 0.35:
        values = [1, max(2, min_cluster_size // 4)]
    elif profile.local_density_cv > 0.80:
        values = [max(2, min_cluster_size // 4), max(4, min_cluster_size // 2)]
    else:
        values = [max(1, min_cluster_size // 5), max(2, min_cluster_size // 3)]

    return sorted(list(set(int(max(1, value)) for value in values)))


def compute_umap_parameters(
    profile: DatasetProfile,
    reduction_dim: int,
) -> Dict[str, List]:
    """Generate UMAP parameters for medium pipeline.
    
    Args:
        profile: DatasetProfile of the dataset.
        reduction_dim: Dimensionality after PCA reduction.
    
    Returns:
        Dictionary with:
        - 'n_neighbors_candidates': List of n_neighbors values
        - 'n_components_candidates': List of n_components values
    """
    N = profile.n_samples
    
    # Adaptive n_neighbors based on dataset size
    base_neighbors = max(15, min(200, int(np.log2(N) * 15)))
    
    # Generate candidates
    n_neighbors_candidates = [
        int(base_neighbors * 0.75),
        base_neighbors,
        int(base_neighbors * 1.35),
    ]
    n_neighbors_candidates = [max(5, min(N - 1, nn)) for nn in n_neighbors_candidates]
    n_neighbors_candidates = sorted(list(set(n_neighbors_candidates)))
    
    # n_components candidates
    n_components_candidates = [
        int(reduction_dim * 0.8),
        reduction_dim,
    ]
    
    # Deduplicate and clip
    n_components_candidates = [
        max(3, min(reduction_dim, nc))
        for nc in n_components_candidates
    ]
    n_components_candidates = sorted(list(set(n_components_candidates)))
    
    return {
        "n_neighbors_candidates": n_neighbors_candidates,
        "n_components_candidates": n_components_candidates,
    }


def compute_refinement_trigger_thresholds(
    profile: DatasetProfile,
) -> Dict[str, float]:
    """Compute thresholds that trigger refinement passes.
    
    Args:
        profile: DatasetProfile of the dataset.
    
    Returns:
        Dictionary of threshold values.
    """
    thresholds = {
        "min_coverage": 0.65,           # Trigger if coverage below this
        "max_noise": 0.30,              # Trigger if noise above this
        "max_giant_ratio": 0.25,        # Trigger if largest cluster > this fraction
        "min_stability": 0.35,          # Trigger if stability below this
    }
    
    # Adapt based on profile
    if profile.cluster_tendency > 0.75:
        thresholds["min_coverage"] = 0.70
        thresholds["max_noise"] = 0.25
    
    if profile.imbalance_tendency > 0.60:
        thresholds["max_giant_ratio"] = 0.20
    
    return thresholds
