import numpy as np
from typing import List
from semantic_clusterer.pipeline.utils import _unique_int_candidates

def compute_optimal_umap_components(embedding_dim: int, n_samples: int) -> int:
    """Compute optimal UMAP dimensions based on information theory priors.
    
    Uses a log-scaling formula to ensure enough dimensions to preserve
    the topology of K expected clusters, where K ~ sqrt(N/8).
    """
    k_expected = np.sqrt(n_samples / 8.0)
    # Target dim = base + log2(k) * multiplier
    target = 4 + int(np.round(np.log2(max(2, k_expected)) * 2.5))
    return int(np.clip(target, 5, min(15, embedding_dim)))

def adaptive_small_umap_components(n_features: int) -> List[int]:
    """Return UMAP component candidates for the small pipeline.
    
    Low-dim models (<=512) are already semantically compact; aggressive
    UMAP compression to 5-7 dims creates artificial micro-clusters.
    We use higher components for these models to preserve structure.
    """
    max_components = max(2, min(15, n_features))

    if n_features >= 1536:
        candidates = [8, 10, 12]
    elif n_features >= 768:
        candidates = [6, 8, 9]
    elif n_features >= 384:
        # Low-dim models: use higher components to avoid over-fragmentation
        candidates = [8, 10, 12]
    else:
        center = max(4, min(max_components, 4 + (n_features // 128)))
        candidates = [center - 1, center, center + 1]

    return _unique_int_candidates(candidates, max_components)

def compute_optimal_umap_neighbors(n_samples: int) -> int:
    """Compute theoretically-grounded UMAP neighbors based on dataset size.

    Formula: clip(round(log2(N) * 2.2), 10, 40)
    Ensures local/global balance scales with the dataset volume.
    """
    target = int(np.round(np.log2(n_samples) * 2.2))
    return int(np.clip(target, 10, min(40, n_samples - 1)))
