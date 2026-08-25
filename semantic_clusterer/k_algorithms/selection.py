"""Algorithm selection for SemanticKSplit.

Maps (Pipeline_Tier, K) pairs to the appropriate clustering algorithm.
"""

from typing import TYPE_CHECKING, Literal

# String-literal type enumerating every valid Algorithm_Used value.
# Mirrors the Algorithm_Used glossary entry.
AlgorithmUsed = Literal[
    "agglomerative-cut-k",
    "bisecting-kmeans",
    "spherical-kmeans",
    "spectral-cosine",
    "constrained-kmeans",
    "balanced-kmeans",
    "minibatch-kmeans-assign",
    "identical-embeddings-tiebreak",
]

if TYPE_CHECKING:
    from semantic_clusterer.config import ClustererConfig


def _select_k_algorithm(
    tier: Literal["tiny", "small", "medium", "large"],
    k: int,
    config: "ClustererConfig",
) -> str:
    """Resolve (tier, k) -> algorithm string.

    The selection matrix is:
    - tiny,   k == 2          -> "bisecting-kmeans"
    - tiny,   3 <= k < 10     -> "agglomerative-cut-k"
    - tiny,   k >= 10         -> "balanced-kmeans"
    - small,  k == 2          -> "bisecting-kmeans"
    - small,  3 <= k <= 10    -> "spectral-cosine"
    - small,  k > 10          -> "balanced-kmeans"
    - medium  (any k)         -> "balanced-kmeans"
    - large   (any k)         -> "minibatch-kmeans-assign"

    Notes:
        - ``config.strategy`` overrides affect which tier is passed in;
          once the tier is resolved, this function applies the matrix above.
        - ``"spherical-kmeans"`` and ``"constrained-kmeans"`` are fallback
          labels set at runtime by individual algorithm wrappers; they are
          never returned by this selection function.
        - ``"identical-embeddings-tiebreak"`` is set by the caller when all
          embeddings are identical; it is never returned by this function.

    Args:
        tier: The Pipeline_Tier resolved by
            ``ClustererConfig.get_strategy_for_size(N_Unique)``.
        k: The requested cluster count (``Requested_K``).
        config: The active ``ClustererConfig`` instance (reserved for
            future per-config overrides; not used in current logic).

    Returns:
        One of the eight ``AlgorithmUsed`` literal strings.

    Raises:
        ValueError: If ``tier`` is not one of the four recognised values.
    """
    if tier == "tiny":
        # 5.2
        if k == 2:
            return "bisecting-kmeans"
        if k >= 10:
            return "balanced-kmeans"
        return "agglomerative-cut-k"

    if tier == "small":
        # 5.4, 5.5
        if k == 2:
            return "bisecting-kmeans"
        if 3 <= k <= 10:
            return "spectral-cosine"
        return "balanced-kmeans"

    if tier == "medium":
        return "balanced-kmeans"

    if tier == "large":
        return "minibatch-kmeans-assign"

    raise ValueError(
        f"Unknown pipeline tier: {tier!r}. "
        "Expected one of 'tiny', 'small', 'medium', 'large'."
    )
