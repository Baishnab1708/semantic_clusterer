"""Property-based tests for algorithm selection matrix (Property 5).


Algorithm Selection Matrix
--------------------------------------
Force a specific Pipeline_Tier by sizing the input to land in the correct
tier under auto-routing:

    tiny   → N ≤ 150
    small  → 151 ≤ N ≤ 5000
    medium → 5001 ≤ N ≤ 50000
    large  → N > 50000

    tiny  + k == 2          → "bisecting-kmeans"
    tiny  + k == 3          → "agglomerative-cut-k"
    small + k == 2          → "bisecting-kmeans"
    small + k == 5          → "spectral-cosine"
    small + k == 15         → "balanced-kmeans"
    medium + k == 2         → "balanced-kmeans"
    large  + k == 2         → "minibatch-kmeans-assign"
    constant embedder       → "identical-embeddings-tiebreak"

Also asserts that the ``"algorithm_used"`` key is present in
``chosen_params`` for every run and that the recorded
value is one of the legal ``Algorithm_Used`` literals.
"""

from __future__ import annotations

from typing import List

import numpy as np
import pytest

from semantic_clusterer import SemanticKSplit
from semantic_clusterer.k_algorithms.selection import AlgorithmUsed

# ---------------------------------------------------------------------------
# Legal algorithm identifier set (mirrors the AlgorithmUsed Literal type)
# ---------------------------------------------------------------------------

_VALID_ALGORITHM_IDS = frozenset(
    AlgorithmUsed.__args__  # type: ignore[attr-defined]
)

# ---------------------------------------------------------------------------
# Stub embedder utilities
# ---------------------------------------------------------------------------


class _Sha256Embedder:
    """Fast deterministic embedder — no ONNX required."""

    DIM: int = 32  # keeps tests fast; resolves to "low" dim-band

    def __init__(self, dim: int = DIM):
        self._dim = dim

    def embed(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        import hashlib

        out = np.empty((len(texts), self._dim), dtype=np.float32)
        for i, text in enumerate(texts):
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            seed = int.from_bytes(digest[:8], byteorder="little")
            rng = np.random.default_rng(seed)
            vec = rng.standard_normal(self._dim).astype(np.float32)
            norm = float(np.linalg.norm(vec))
            out[i] = vec / max(norm, 1e-8)
        return out


class _ConstantEmbedder:
    """Embedder that returns the *same* vector for every text."""

    DIM: int = 32

    def __init__(self, dim: int = DIM):
        self._dim = dim

    def embed(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        vec = np.ones(self._dim, dtype=np.float32)
        vec /= np.linalg.norm(vec)
        return np.tile(vec, (len(texts), 1))


# ---------------------------------------------------------------------------
# Helper: build a text list of length N with distinct, stable strings
# ---------------------------------------------------------------------------


def _texts(n: int, prefix: str = "text") -> List[str]:
    return [f"{prefix} {i:06d}" for i in range(n)]


# ---------------------------------------------------------------------------
# Helper: run split_with_report and return the algorithm_used field
# Tier is forced purely by N (auto-routing).
# ---------------------------------------------------------------------------


def _get_algorithm_used(
    embedder, texts: List[str], k: int
) -> str:
    """Construct a SemanticKSplit, call split_with_report, return algorithm_used."""
    ks = SemanticKSplit(
        embedding_model=embedder,
        k=k,
        random_state=42,
    )
    _, report = ks.split_with_report(texts)
    return report.chosen_params["algorithm_used"]


# ---------------------------------------------------------------------------
# Parametrised tests: (tier, k) → expected algorithm
# Tier is forced by N (auto-routing):
#   "tiny"   → N ≤ 150
#   "small"  → 151 ≤ N ≤ 5000
#   "medium" → 5001 ≤ N ≤ 50000
#   "large"  → N > 50000
# ---------------------------------------------------------------------------

# Columns: (test_id, N, k, expected_algorithm)
_MATRIX_CASES = [
    # --- tiny tier (N ≤ 150) ---
    ("tiny_k2",  20,    2,  "bisecting-kmeans"),
    ("tiny_k3",  20,    3,  "agglomerative-cut-k"),
    # --- small tier (151 ≤ N ≤ 5000) ---
    ("small_k2",  200,  2,  "bisecting-kmeans"),
    ("small_k5",  200,  5,  "spectral-cosine"),
    ("small_k15", 200, 15,  "balanced-kmeans"),
    # --- medium tier (5001 ≤ N ≤ 50000) ---
    ("medium_k2", 5100, 2,  "balanced-kmeans"),
    # --- large tier (N > 50000) ---
    ("large_k2",  50100, 2, "minibatch-kmeans-assign"),
]


@pytest.mark.parametrize(
    "n, k, expected_algorithm",
    [(n, k, e) for (_, n, k, e) in _MATRIX_CASES],
    ids=[test_id for (test_id, *_rest) in _MATRIX_CASES],
)
def test_algorithm_selection_matrix(
    n: int, k: int, expected_algorithm: str
) -> None:
    """algorithm_used matches the (tier, k) selection matrix.

    """
    embedder = _Sha256Embedder()
    texts = _texts(n)
    algo = _get_algorithm_used(embedder, texts, k=k)
    assert algo == expected_algorithm, (
        f"Expected algorithm_used={expected_algorithm!r} for "
        f"k={k}, N={n}; got {algo!r}"
    )


# ---------------------------------------------------------------------------
# Test: algorithm_used key is always present in chosen_params (Req 11.2)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "n, k",
    [(n, k) for (_, n, k, _e) in _MATRIX_CASES],
    ids=[test_id for (test_id, *_rest) in _MATRIX_CASES],
)
def test_algorithm_used_key_always_present(n: int, k: int) -> None:
    """Chosen_params must contain the 'algorithm_used' key."""
    embedder = _Sha256Embedder()
    ks = SemanticKSplit(
        embedding_model=embedder,
        k=k,
        random_state=42,
    )
    _, report = ks.split_with_report(_texts(n))
    assert "algorithm_used" in report.chosen_params, (
        "report.chosen_params is missing the 'algorithm_used' key"
    )


# ---------------------------------------------------------------------------
# Test: algorithm_used value is always a valid Algorithm_Used literal
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "n, k",
    [(n, k) for (_, n, k, _e) in _MATRIX_CASES],
    ids=[test_id for (test_id, *_rest) in _MATRIX_CASES],
)
def test_algorithm_used_is_valid_literal(n: int, k: int) -> None:
    """algorithm_used must be one of the legal Algorithm_Used strings."""
    embedder = _Sha256Embedder()
    ks = SemanticKSplit(
        embedding_model=embedder,
        k=k,
        random_state=42,
    )
    _, report = ks.split_with_report(_texts(n))
    algo = report.chosen_params["algorithm_used"]
    assert algo in _VALID_ALGORITHM_IDS, (
        f"algorithm_used={algo!r} is not a valid AlgorithmUsed literal. "
        f"Valid values: {sorted(_VALID_ALGORITHM_IDS)}"
    )


# ---------------------------------------------------------------------------
# Test: constant embedder triggers "identical-embeddings-tiebreak"
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("k", [2, 3, 5])
def test_constant_embedder_triggers_tiebreak(k: int) -> None:
    """All-identical embeddings -> 'identical-embeddings-tiebreak'."""
    embedder = _ConstantEmbedder()
    texts = _texts(20)
    ks = SemanticKSplit(
        embedding_model=embedder,
        k=k,
        random_state=42,
    )
    _, report = ks.split_with_report(texts)
    algo = report.chosen_params["algorithm_used"]
    assert algo == "identical-embeddings-tiebreak", (
        f"Expected 'identical-embeddings-tiebreak' for constant embedder with k={k}; "
        f"got {algo!r}"
    )


def test_constant_embedder_tiebreak_in_warnings(k: int = 3) -> None:
    """The 'identical-embeddings-tiebreak' warning string appears in report.warnings."""
    embedder = _ConstantEmbedder()
    ks = SemanticKSplit(
        embedding_model=embedder,
        k=k,
        random_state=42,
    )
    _, report = ks.split_with_report(_texts(20))
    assert "identical-embeddings-tiebreak" in report.warnings, (
        f"Expected 'identical-embeddings-tiebreak' in report.warnings; "
        f"got {report.warnings!r}"
    )


# ---------------------------------------------------------------------------
# Test: tier boundary — tiny k >= 3 gives agglomerative
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("k", [3, 5, 9])
def test_tiny_tier_k_ge_3_gives_agglomerative(k: int) -> None:
    """Tiny tier (N ≤ 150), 3 <= k < 10 → agglomerative-cut-k."""
    embedder = _Sha256Embedder()
    texts = _texts(50)  # N=50 → tiny tier
    algo = _get_algorithm_used(embedder, texts, k=k)
    assert algo == "agglomerative-cut-k", (
        f"Expected 'agglomerative-cut-k' for tiny tier, k={k}; got {algo!r}"
    )

def test_tiny_tier_k_ge_10_gives_balanced_kmeans() -> None:
    """Tiny tier (N ≤ 150), k >= 10 → balanced-kmeans."""
    embedder = _Sha256Embedder()
    texts = _texts(50)  # N=50 → tiny tier
    algo = _get_algorithm_used(embedder, texts, k=10)
    assert algo == "balanced-kmeans", (
        f"Expected 'balanced-kmeans' for tiny tier, k=10; got {algo!r}"
    )


# ---------------------------------------------------------------------------
# Test: small tier k boundary — k=10 (spectral) vs k=11 (balanced)
# ---------------------------------------------------------------------------


def test_small_tier_k10_gives_spectral() -> None:
    """Small tier, k=10 (boundary) → spectral-cosine."""
    embedder = _Sha256Embedder()
    texts = _texts(200)  # N=200 → small tier
    algo = _get_algorithm_used(embedder, texts, k=10)
    assert algo == "spectral-cosine", (
        f"Expected 'spectral-cosine' for small tier, k=10; got {algo!r}"
    )


def test_small_tier_k11_gives_balanced() -> None:
    """Small tier, k=11 (just over boundary) → balanced-kmeans."""
    embedder = _Sha256Embedder()
    texts = _texts(200)  # N=200 → small tier
    algo = _get_algorithm_used(embedder, texts, k=11)
    assert algo == "balanced-kmeans", (
        f"Expected 'balanced-kmeans' for small tier, k=11; got {algo!r}"
    )


# ---------------------------------------------------------------------------
# Test: strategy override tests — now verify auto-routing by N instead
# ---------------------------------------------------------------------------


def test_large_tier_via_n_uses_minibatch() -> None:
    """Large tier (N > 50000) → minibatch-kmeans-assign."""
    embedder = _Sha256Embedder()
    texts = _texts(50100)  # N > 50000 → large tier
    algo = _get_algorithm_used(embedder, texts, k=2)
    assert algo == "minibatch-kmeans-assign", (
        f"Expected 'minibatch-kmeans-assign' for large tier; got {algo!r}"
    )


def test_small_tier_k2_gives_bisecting() -> None:
    """Small tier (151 ≤ N ≤ 5000), k=2 → bisecting-kmeans."""
    embedder = _Sha256Embedder()
    texts = _texts(200)  # N=200 → small tier
    algo = _get_algorithm_used(embedder, texts, k=2)
    assert algo == "bisecting-kmeans", (
        f"Expected 'bisecting-kmeans' for small tier, k=2; got {algo!r}"
    )


def test_medium_tier_k2_gives_balanced_kmeans() -> None:
    """Medium tier (5001 ≤ N ≤ 50000), k=2 → balanced-kmeans."""
    embedder = _Sha256Embedder()
    texts = _texts(5100)  # N=5100 → medium tier
    algo = _get_algorithm_used(embedder, texts, k=2)
    assert algo == "balanced-kmeans", (
        f"Expected 'balanced-kmeans' for medium tier; got {algo!r}"
    )
