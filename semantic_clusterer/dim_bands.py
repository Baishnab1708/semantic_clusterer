"""Dim-band definitions and parameter-grid resolution."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Dict, List, Literal, Tuple

DimBand = Literal["low", "mid", "high", "xhigh"]

# Closed inclusive integer ranges. Read-only at module level.
_RAW_BANDS: Dict[DimBand, Tuple[int, int]] = {
    "low":   (256, 511),
    "mid":   (512, 1023),
    "high":  (1024, 2047),
    "xhigh": (2048, 16384),
}
SUPPORTED_DIM_BANDS = MappingProxyType(_RAW_BANDS)  # public, read-only


def resolve_dim_band(D: int) -> DimBand:
    """Resolve embedding dim D into one of the four supported bands.

    Raises ValueError for D < 1.
    Emits a UserWarning and returns 'low' for 1 <= D < 256.
    Emits a UserWarning and returns 'xhigh' for D > 16384.
    """
    if D < 1:
        raise ValueError(f"embedding_dim must be >= 1, got {D}")
    if D < 256:
        warnings.warn(
            f"Embedding dim {D} is below low band lower bound 256; "
            f"falling back to 'low' band parameters.",
            UserWarning,
            stacklevel=2,
        )
        return "low"
    if D > 16384:
        warnings.warn(
            f"Embedding dim {D} is above xhigh band upper bound 16384; "
            f"falling back to 'xhigh' band parameters.",
            UserWarning,
            stacklevel=2,
        )
        return "xhigh"
    for name, (lo, hi) in _RAW_BANDS.items():
        if lo <= D <= hi:
            return name
    # Defensive: unreachable for the contiguous ranges above
    raise ValueError(f"could not resolve dim band for D={D}")


@dataclass(frozen=True)
class BandGrid:
    """Per-(band, tier) parameter grid.

    Every list is non-empty and every value satisfies the documented bounds.
    """
    # Reduction
    pca_targets: List[int]
    # UMAP
    umap_n_neighbors: List[int]
    umap_n_components: List[int]
    umap_min_dists: List[float] = field(default_factory=lambda: [0.0, 0.05])
    # HDBSCAN
    hdbscan_min_cluster_size_ratios: List[float] = field(
        default_factory=lambda: [0.020, 0.040, 0.060]
    )
    hdbscan_min_samples: List[int] = field(default_factory=lambda: [1, 2, 4])
    hdbscan_methods: List[str] = field(default_factory=lambda: ["eom", "leaf"])
    # Tiny pipeline only
    tiny_k_grid: List[int] = field(default_factory=lambda: [2, 3, 5, 8, 12])
