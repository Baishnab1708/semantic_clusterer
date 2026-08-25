"""Configuration dataclasses for semantic-clusterer v0.1.0.

Two public config classes, one for each entry point:
  - SemanticClustererConfig  — density-based variable-K clustering
  - SemanticKSplitConfig     — fixed-K partitioning

Both extend _BaseConfig which holds the shared infrastructure knobs.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Literal, Optional

_PUBLIC_CLUSTERER_FIELDS = frozenset({
    "cluster_granularity",
    "min_cluster_size",
    "min_samples",
    "max_samples",
    "extract_keywords",
    "keywords_top_n",
    "batch_size",
    "normalize_embeddings",
    "verbose",
    "random_state",
})

_PUBLIC_KSPLIT_FIELDS = frozenset({
    "quality",
    "max_samples",
    "extract_keywords",
    "keywords_top_n",
    "batch_size",
    "normalize_embeddings",
    "verbose",
    "random_state",
})

# Backward-compat alias — old code that uses _PUBLIC_CONFIG_FIELDS still works
_PUBLIC_CONFIG_FIELDS = _PUBLIC_CLUSTERER_FIELDS | _PUBLIC_KSPLIT_FIELDS | frozenset({
    "min_cluster_size",
    "min_samples",
    # Deprecated alias retained for one release
    "allow_oversized_datasets",
})


def _adaptive_reduction_components(
    n_samples: int,
    n_features: int,
    min_dim: int = 64,
    max_dim: int = 512,
) -> int:
    n_samples = max(10, n_samples)
    n_features = max(1, n_features)
    score = math.sqrt(n_features) * (math.log10(n_samples) ** 0.7)
    min_score = math.sqrt(384) * (math.log10(5_000) ** 0.7)
    max_score = math.sqrt(3072) * (math.log10(200_000) ** 0.7)
    if max_score - min_score > 0:
        norm = (score - min_score) / (max_score - min_score)
    else:
        norm = 0.5
    norm = max(0.0, min(1.0, norm))
    target_dim = min_dim + norm * (max_dim - min_dim)
    return int(min(target_dim, n_features))


@dataclass
class _BaseConfig:
    """Shared configuration fields for both clusterer classes."""
    batch_size: int = 64
    normalize_embeddings: bool = True
    verbose: bool = False
    random_state: int = 42
    max_samples: Optional[int] = 200_000
    extract_keywords: bool = True
    keywords_top_n: int = 10

    # Size thresholds for auto strategy selection (internal)
    _tiny_threshold: int = field(default=150, repr=False)
    _small_threshold: int = field(default=5000, repr=False)
    _medium_threshold: int = field(default=50000, repr=False)
    _large_threshold: int = field(default=200000, repr=False)

    def _validate_base(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if not isinstance(self.random_state, int) or isinstance(self.random_state, bool):
            raise ValueError(f"random_state must be an int, got {type(self.random_state).__name__!r}")
        if not (0 <= self.random_state <= 2**32 - 1):
            raise ValueError(f"random_state must be in [0, 2**32 - 1], got {self.random_state}")
        if self.max_samples is not None:
            if not isinstance(self.max_samples, int) or isinstance(self.max_samples, bool):
                raise ValueError(f"max_samples must be an int or None")
            if self.max_samples < 1:
                raise ValueError(f"max_samples must be >= 1")
        if not isinstance(self.extract_keywords, bool):
            raise ValueError("extract_keywords must be a bool")
        if not isinstance(self.keywords_top_n, int) or isinstance(self.keywords_top_n, bool):
            raise ValueError("keywords_top_n must be an int")
        if self.keywords_top_n < 1:
            raise ValueError(f"keywords_top_n must be >= 1")

    def get_strategy_for_size(self, n_samples: int) -> Literal["tiny", "small", "medium", "large"]:
        if n_samples <= self._tiny_threshold:
            return "tiny"
        elif n_samples <= self._small_threshold:
            return "small"
        elif n_samples <= self._medium_threshold:
            return "medium"
        else:
            return "large"

    def get_reduction_for_strategy(self, strategy):
        if strategy in ("tiny", "small"):
            return None
        return "pca"

    def get_reduction_components(self, n_features: int, n_samples: int = 10000) -> int:
        if n_features < 1:
            raise ValueError("n_features must be >= 1")
        reduction_components = getattr(self, "_reduction_components", None)
        if reduction_components is not None:
            return max(1, min(reduction_components, n_features))
        return _adaptive_reduction_components(n_samples=n_samples, n_features=n_features, min_dim=64, max_dim=512)


@dataclass
class SemanticClustererConfig(_BaseConfig):
    """Configuration for SemanticClusterer (density-based variable-K).

    Key knob
    --------
    cluster_granularity : "fine" | "balanced" | "coarse"
        Controls how many clusters the library targets and how
        aggressively it merges near-duplicate clusters:

        "fine"     — current HDBSCAN defaults, no forced merging.
                     Produces the most clusters; good for topic discovery.
        "balanced" — (default) moderate mcs floor + centroid merge pass.
                     Produces clean, usable clusters for most corpora.
        "coarse"   — aggressive mcs floor + hard merge pass.
                     Produces few large clusters; good for broad grouping.
    """
    cluster_granularity: Literal["fine", "balanced", "coarse"] = "balanced"
    min_cluster_size: Optional[int] = None
    min_samples: Optional[int] = None
    _reduction_components: Optional[int] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self._validate_base()
        if self.cluster_granularity not in ("fine", "balanced", "coarse"):
            raise ValueError(f"cluster_granularity must be 'fine', 'balanced', or 'coarse', got {self.cluster_granularity!r}")
        if self.min_cluster_size is not None:
            if not isinstance(self.min_cluster_size, int) or isinstance(self.min_cluster_size, bool):
                raise ValueError("min_cluster_size must be an int or None")
            if self.min_cluster_size < 2:
                raise ValueError(f"min_cluster_size must be >= 2, got {self.min_cluster_size}")
        if self.min_samples is not None:
            if not isinstance(self.min_samples, int) or isinstance(self.min_samples, bool):
                raise ValueError("min_samples must be an int or None")
            if self.min_samples < 1:
                raise ValueError(f"min_samples must be >= 1, got {self.min_samples}")

    @property
    def _min_cluster_size(self) -> int:
        return self.min_cluster_size if self.min_cluster_size is not None else 5

    @property
    def _min_samples(self) -> int:
        return self.min_samples if self.min_samples is not None else 5


@dataclass
class SemanticKSplitConfig(_BaseConfig):
    """Configuration for SemanticKSplit (fixed-K partitioning).

    Key knob
    --------
    quality : "fast" | "balanced" | "best"
        Controls how many restarts the algorithm performs and how
        rigorously the best partition is selected:

        "fast"     — single pass, first result returned.
                     Fastest; good for quick iteration.
        "balanced" — (default) 3–5 restarts, best by composite score.
                     Good output for most production uses.
        "best"     — 8–12 restarts, strict selection.
                     Best possible partition; use when quality is critical.
    """
    quality: Literal["fast", "balanced", "best"] = "balanced"

    def __post_init__(self) -> None:
        self._validate_base()
        if self.quality not in ("fast", "balanced", "best"):
            raise ValueError(f"quality must be 'fast', 'balanced', or 'best', got {self.quality!r}")



# ---------------------------------------------------------------------------
# Backward-compat: ClustererConfig alias
# ---------------------------------------------------------------------------

@dataclass
class ClustererConfig(_BaseConfig):
    """Configuration for SemanticClusterer (backward-compatible).

    .. deprecated::
        Use :class:`SemanticClustererConfig` directly. This class is an
        alias kept for backward compatibility and will be updated to be
        equivalent in the next major release.
    """

    # ------------------------------------------------------------------
    # Core public fields
    # ------------------------------------------------------------------
    cluster_granularity: Literal["fine", "balanced", "coarse"] = "balanced"

    # ------------------------------------------------------------------
    # Promoted in v0.3.0
    # ------------------------------------------------------------------
    min_cluster_size: Optional[int] = None
    min_samples: Optional[int] = None

    # ------------------------------------------------------------------
    # Deprecated; kept for one release
    # ------------------------------------------------------------------
    allow_oversized_datasets: bool = False

    # ------------------------------------------------------------------
    # Internal parameters (not exposed to users, optimized defaults)
    # ------------------------------------------------------------------
    _noise_threshold: float = field(default=0.0, repr=False)
    _reduction_components: Optional[int] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self._validate_base()

        # Validate promoted fields
        if self.min_cluster_size is not None:
            if not isinstance(self.min_cluster_size, int) or isinstance(self.min_cluster_size, bool):
                raise ValueError(
                    "min_cluster_size must be an int or None, got "
                    f"{type(self.min_cluster_size).__name__!r}"
                )
            if self.min_cluster_size < 2:
                raise ValueError(
                    f"min_cluster_size must be >= 2, got {self.min_cluster_size}"
                )

        if self.min_samples is not None:
            if not isinstance(self.min_samples, int) or isinstance(self.min_samples, bool):
                raise ValueError(
                    "min_samples must be an int or None, got "
                    f"{type(self.min_samples).__name__!r}"
                )
            if self.min_samples < 1:
                raise ValueError(
                    f"min_samples must be >= 1, got {self.min_samples}"
                )

        # Validate allow_oversized_datasets and reconcile with max_samples.
        if not isinstance(self.allow_oversized_datasets, bool):
            raise ValueError(
                "allow_oversized_datasets must be a bool, got "
                f"{type(self.allow_oversized_datasets).__name__!r}"
            )
        if self.allow_oversized_datasets:
            warnings.warn(
                "allow_oversized_datasets is deprecated and will be removed in "
                "v0.4.0. Use max_samples=None to disable the cap, or pass an "
                "integer to set a custom cap.",
                DeprecationWarning,
                stacklevel=3,
            )
            if self.max_samples == 200_000:
                self.max_samples = None

    @property
    def _min_cluster_size(self) -> int:
        return self.min_cluster_size if self.min_cluster_size is not None else 5

    @property
    def _min_samples(self) -> int:
        return self.min_samples if self.min_samples is not None else 5


# Keep the old name as an alias for pipeline internals that still reference it.
# Will be removed in a future cleanup pass.
# SemanticClustererConfig = ClustererConfig (done via direct subclass above)


def _validate_config_dict(config_dict: dict, allowed_fields: frozenset = None) -> None:
    """Validate that a config dict only contains allowed public fields."""
    if allowed_fields is None:
        allowed_fields = _PUBLIC_CONFIG_FIELDS
    invalid_fields = set(config_dict.keys()) - allowed_fields
    if invalid_fields:
        allowed = ", ".join(sorted(allowed_fields))
        invalid = ", ".join(sorted(invalid_fields))
        raise ValueError(
            f"Invalid config field(s): {invalid}. "
            f"Allowed fields are: {allowed}"
        )
