"""Unit tests for k-aware clustering algorithm wrappers.

Covers:
- _agglomerative_cut_k  (agglomerative.py)
- _bisecting_kmeans     (bisecting.py)
- _spectral_cosine      (spectral.py)
- _balanced_kmeans      (balanced.py)
- _minibatch_kmeans_assign (minibatch_assign.py)

For each wrapper the tests assert:
  1. dtype == np.int32
  2. shape == (N_Unique,)
  3. values in [0, k-1]
  4. bit-identical labels under repeated calls with the same seed

Both k == 2 and k >= 3 paths are covered for bisecting and balanced.

A deterministic sha256-derived fake embedder is used to keep tests fast:
  - hash each index as bytes
  - decode as float32 vector
  - L2-normalize

"""

from __future__ import annotations

import hashlib
from types import SimpleNamespace
from typing import Optional

import numpy as np
import pytest

from semantic_clusterer.k_algorithms.agglomerative import _agglomerative_cut_k
from semantic_clusterer.k_algorithms.bisecting import _bisecting_kmeans
from semantic_clusterer.k_algorithms.spectral import _spectral_cosine
from semantic_clusterer.k_algorithms.balanced import _balanced_kmeans
from semantic_clusterer.k_algorithms.minibatch_assign import _minibatch_kmeans_assign


# ---------------------------------------------------------------------------
# Deterministic sha256-derived fake embedder
# ---------------------------------------------------------------------------

def _sha256_embedding(index: int, dim: int = 64) -> np.ndarray:
    """Build a deterministic L2-normalised float32 vector from an index.

    Algorithm:
    1. Hash the index as bytes with sha256 to get a 32-byte digest.
    2. Interpret the first 8 bytes of the digest as a uint64 seed.
    3. Use that seed to initialise a numpy RNG and draw ``dim`` standard
       normal float32 values — this guarantees no NaN/Inf bit patterns.
    4. L2-normalise the resulting vector.
    """
    digest = hashlib.sha256(str(index).encode()).digest()
    # Extract a uint64 seed from the first 8 bytes of the digest.
    seed = int.from_bytes(digest[:8], byteorder="little")
    rng = np.random.default_rng(seed)
    raw = rng.standard_normal(dim).astype(np.float32)
    norm = np.linalg.norm(raw)
    if norm > 0:
        raw /= norm
    return raw


def make_embeddings(n: int, dim: int = 64) -> np.ndarray:
    """Build an (n, dim) float32 embedding matrix using sha256-derived vectors."""
    return np.stack([_sha256_embedding(i, dim) for i in range(n)], axis=0)


# ---------------------------------------------------------------------------
# Helper: fake trace object for spectral_cosine
# ---------------------------------------------------------------------------

def _make_trace() -> SimpleNamespace:
    """Minimal trace stub with chosen_params dict."""
    return SimpleNamespace(chosen_params={})


# ---------------------------------------------------------------------------
# Helper: validate common assertions
# ---------------------------------------------------------------------------

def _assert_labels_valid(labels: np.ndarray, n: int, k: int) -> None:
    """Assert dtype, shape, and value range for a label array."""
    assert labels.dtype == np.int32, f"Expected int32, got {labels.dtype}"
    assert labels.shape == (n,), f"Expected shape ({n},), got {labels.shape}"
    assert np.all(labels >= 0), "Labels contain negative values"
    assert np.all(labels < k), f"Labels contain values >= k={k}"


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

# (n_samples, k, seed)
COMMON_CASES = [
    (20, 2, 42),
    (20, 3, 42),
    (30, 4, 7),
    (40, 5, 99),
]

SEED = 42
N = 30
DIM = 64


# ===========================================================================
# _agglomerative_cut_k
# ===========================================================================

class TestAgglomerativeCutK:
    """Tests for _agglomerative_cut_k."""

    def test_dtype_int32(self):
        """Output dtype must be np.int32."""
        emb = make_embeddings(N, DIM)
        labels = _agglomerative_cut_k(emb, k=3)
        assert labels.dtype == np.int32

    def test_shape_equals_n(self):
        """Output shape must be (N,)."""
        emb = make_embeddings(N, DIM)
        labels = _agglomerative_cut_k(emb, k=3)
        assert labels.shape == (N,)

    def test_values_in_range_k3(self):
        """All label values must be in [0, k-1] for k=3."""
        emb = make_embeddings(N, DIM)
        labels = _agglomerative_cut_k(emb, k=3)
        _assert_labels_valid(labels, N, k=3)

    def test_values_in_range_k5(self):
        """All label values must be in [0, k-1] for k=5."""
        emb = make_embeddings(N, DIM)
        labels = _agglomerative_cut_k(emb, k=5)
        _assert_labels_valid(labels, N, k=5)

    def test_deterministic_repeated_calls(self):
        """Agglomerative clustering is deterministic; repeated calls produce identical labels."""
        emb = make_embeddings(N, DIM)
        labels1 = _agglomerative_cut_k(emb, k=3)
        labels2 = _agglomerative_cut_k(emb, k=3)
        assert np.array_equal(labels1, labels2), "Labels differ on repeated calls"

    def test_deterministic_k2(self):
        """Determinism also holds for k=2."""
        emb = make_embeddings(N, DIM)
        labels1 = _agglomerative_cut_k(emb, k=2)
        labels2 = _agglomerative_cut_k(emb, k=2)
        assert np.array_equal(labels1, labels2)

    @pytest.mark.parametrize("n,k", [(20, 2), (20, 3), (30, 4), (40, 5)])
    def test_parametrized_valid_output(self, n, k):
        """Parametrized check: dtype, shape, and value range for various (n, k)."""
        emb = make_embeddings(n, DIM)
        labels = _agglomerative_cut_k(emb, k=k)
        _assert_labels_valid(labels, n, k)

    def test_minimum_k2(self):
        """k=2 with small sample still produces valid labels."""
        emb = make_embeddings(10, DIM)
        labels = _agglomerative_cut_k(emb, k=2)
        _assert_labels_valid(labels, 10, k=2)


# ===========================================================================
# _bisecting_kmeans
# ===========================================================================

class TestBisectingKMeans:
    """Tests for _bisecting_kmeans."""

    # --- k == 2 path (n_restarts = 5) -------------------------------------

    def test_dtype_int32_k2(self):
        """dtype must be np.int32 for k=2."""
        emb = make_embeddings(N, DIM)
        labels = _bisecting_kmeans(emb, k=2, seed=SEED)
        assert labels.dtype == np.int32

    def test_shape_k2(self):
        """Shape must be (N,) for k=2."""
        emb = make_embeddings(N, DIM)
        labels = _bisecting_kmeans(emb, k=2, seed=SEED)
        assert labels.shape == (N,)

    def test_values_in_range_k2(self):
        """All label values must be in [0, 1] for k=2."""
        emb = make_embeddings(N, DIM)
        labels = _bisecting_kmeans(emb, k=2, seed=SEED)
        _assert_labels_valid(labels, N, k=2)

    def test_deterministic_k2_same_seed(self):
        """Repeated calls with the same seed produce bit-identical labels (k=2)."""
        emb = make_embeddings(N, DIM)
        labels1 = _bisecting_kmeans(emb, k=2, seed=SEED)
        labels2 = _bisecting_kmeans(emb, k=2, seed=SEED)
        assert np.array_equal(labels1, labels2), (
            "k=2 bisecting: labels differ on repeated calls with same seed"
        )

    def test_different_seeds_k2(self):
        """Different seeds may produce different label arrays (k=2)."""
        emb = make_embeddings(N, DIM)
        labels1 = _bisecting_kmeans(emb, k=2, seed=0)
        labels2 = _bisecting_kmeans(emb, k=2, seed=999)
        # Both must at minimum be valid
        _assert_labels_valid(labels1, N, k=2)
        _assert_labels_valid(labels2, N, k=2)

    # --- k >= 3 path (n_restarts = 3) -------------------------------------

    def test_dtype_int32_k3(self):
        """dtype must be np.int32 for k=3."""
        emb = make_embeddings(N, DIM)
        labels = _bisecting_kmeans(emb, k=3, seed=SEED)
        assert labels.dtype == np.int32

    def test_shape_k3(self):
        """Shape must be (N,) for k=3."""
        emb = make_embeddings(N, DIM)
        labels = _bisecting_kmeans(emb, k=3, seed=SEED)
        assert labels.shape == (N,)

    def test_values_in_range_k3(self):
        """All label values must be in [0, 2] for k=3."""
        emb = make_embeddings(N, DIM)
        labels = _bisecting_kmeans(emb, k=3, seed=SEED)
        _assert_labels_valid(labels, N, k=3)

    def test_deterministic_k3_same_seed(self):
        """Repeated calls with the same seed produce bit-identical labels (k=3)."""
        emb = make_embeddings(N, DIM)
        labels1 = _bisecting_kmeans(emb, k=3, seed=SEED)
        labels2 = _bisecting_kmeans(emb, k=3, seed=SEED)
        assert np.array_equal(labels1, labels2), (
            "k=3 bisecting: labels differ on repeated calls with same seed"
        )

    def test_values_in_range_k4(self):
        """All label values must be in [0, 3] for k=4."""
        emb = make_embeddings(40, DIM)
        labels = _bisecting_kmeans(emb, k=4, seed=SEED)
        _assert_labels_valid(labels, 40, k=4)

    def test_deterministic_k4_same_seed(self):
        """Repeated calls with the same seed produce bit-identical labels (k=4)."""
        emb = make_embeddings(40, DIM)
        labels1 = _bisecting_kmeans(emb, k=4, seed=SEED)
        labels2 = _bisecting_kmeans(emb, k=4, seed=SEED)
        assert np.array_equal(labels1, labels2)

    @pytest.mark.parametrize("n,k,seed", COMMON_CASES)
    def test_parametrized_valid_output(self, n, k, seed):
        """Parametrized check: dtype, shape, and value range."""
        emb = make_embeddings(n, DIM)
        labels = _bisecting_kmeans(emb, k=k, seed=seed)
        _assert_labels_valid(labels, n, k)

    @pytest.mark.parametrize("n,k,seed", COMMON_CASES)
    def test_parametrized_determinism(self, n, k, seed):
        """Parametrized determinism: same seed -> same labels."""
        emb = make_embeddings(n, DIM)
        labels1 = _bisecting_kmeans(emb, k=k, seed=seed)
        labels2 = _bisecting_kmeans(emb, k=k, seed=seed)
        assert np.array_equal(labels1, labels2)


# ===========================================================================
# _spectral_cosine
# ===========================================================================

class TestSpectralCosine:
    """Tests for _spectral_cosine."""

    def test_dtype_int32_k3(self):
        """dtype must be np.int32 for k=3."""
        emb = make_embeddings(N, DIM)
        trace = _make_trace()
        labels = _spectral_cosine(emb, k=3, seed=SEED, trace=trace)
        assert labels.dtype == np.int32

    def test_shape_k3(self):
        """Shape must be (N,) for k=3."""
        emb = make_embeddings(N, DIM)
        trace = _make_trace()
        labels = _spectral_cosine(emb, k=3, seed=SEED, trace=trace)
        assert labels.shape == (N,)

    def test_values_in_range_k3(self):
        """All label values must be in [0, k-1] for k=3."""
        emb = make_embeddings(N, DIM)
        trace = _make_trace()
        labels = _spectral_cosine(emb, k=3, seed=SEED, trace=trace)
        _assert_labels_valid(labels, N, k=3)

    def test_values_in_range_k2(self):
        """All label values must be in [0, 1] for k=2."""
        emb = make_embeddings(N, DIM)
        trace = _make_trace()
        labels = _spectral_cosine(emb, k=2, seed=SEED, trace=trace)
        _assert_labels_valid(labels, N, k=2)

    def test_deterministic_k3_same_seed(self):
        """Repeated calls with same seed produce bit-identical labels (k=3)."""
        emb = make_embeddings(N, DIM)
        trace1 = _make_trace()
        trace2 = _make_trace()
        labels1 = _spectral_cosine(emb, k=3, seed=SEED, trace=trace1)
        labels2 = _spectral_cosine(emb, k=3, seed=SEED, trace=trace2)
        assert np.array_equal(labels1, labels2), (
            "spectral k=3: labels differ on repeated calls with same seed"
        )

    def test_deterministic_k2_same_seed(self):
        """Repeated calls with same seed produce bit-identical labels (k=2)."""
        emb = make_embeddings(N, DIM)
        trace1 = _make_trace()
        trace2 = _make_trace()
        labels1 = _spectral_cosine(emb, k=2, seed=SEED, trace=trace1)
        labels2 = _spectral_cosine(emb, k=2, seed=SEED, trace=trace2)
        assert np.array_equal(labels1, labels2)

    def test_values_in_range_k5(self):
        """All label values must be in [0, 4] for k=5."""
        emb = make_embeddings(N, DIM)
        trace = _make_trace()
        labels = _spectral_cosine(emb, k=5, seed=SEED, trace=trace)
        _assert_labels_valid(labels, N, k=5)

    def test_trace_has_chosen_params(self):
        """After the call, trace.chosen_params should exist (may remain empty or be updated)."""
        emb = make_embeddings(N, DIM)
        trace = _make_trace()
        _spectral_cosine(emb, k=3, seed=SEED, trace=trace)
        assert isinstance(trace.chosen_params, dict)

    @pytest.mark.parametrize("n,k,seed", COMMON_CASES)
    def test_parametrized_valid_output(self, n, k, seed):
        """Parametrized check: dtype, shape, and value range."""
        emb = make_embeddings(n, DIM)
        trace = _make_trace()
        labels = _spectral_cosine(emb, k=k, seed=seed, trace=trace)
        _assert_labels_valid(labels, n, k)

    @pytest.mark.parametrize("n,k,seed", COMMON_CASES)
    def test_parametrized_determinism(self, n, k, seed):
        """Parametrized determinism: same seed -> same labels."""
        emb = make_embeddings(n, DIM)
        labels1 = _spectral_cosine(emb, k=k, seed=seed, trace=_make_trace())
        labels2 = _spectral_cosine(emb, k=k, seed=seed, trace=_make_trace())
        assert np.array_equal(labels1, labels2)


# ===========================================================================
# _balanced_kmeans
# ===========================================================================

class TestBalancedKMeans:
    """Tests for _balanced_kmeans."""

    # --- k == 2 path -------------------------------------------------------

    def test_dtype_int32_k2(self):
        """dtype must be np.int32 for k=2."""
        emb = make_embeddings(N, DIM)
        labels = _balanced_kmeans(emb, k=2, seed=SEED)
        assert labels.dtype == np.int32

    def test_shape_k2(self):
        """Shape must be (N,) for k=2."""
        emb = make_embeddings(N, DIM)
        labels = _balanced_kmeans(emb, k=2, seed=SEED)
        assert labels.shape == (N,)

    def test_values_in_range_k2(self):
        """All label values must be in [0, 1] for k=2."""
        emb = make_embeddings(N, DIM)
        labels = _balanced_kmeans(emb, k=2, seed=SEED)
        _assert_labels_valid(labels, N, k=2)

    def test_deterministic_k2_same_seed(self):
        """Repeated calls with the same seed produce bit-identical labels (k=2)."""
        emb = make_embeddings(N, DIM)
        labels1 = _balanced_kmeans(emb, k=2, seed=SEED)
        labels2 = _balanced_kmeans(emb, k=2, seed=SEED)
        assert np.array_equal(labels1, labels2), (
            "balanced_kmeans k=2: labels differ on repeated calls with same seed"
        )

    # --- k >= 3 path -------------------------------------------------------

    def test_dtype_int32_k3(self):
        """dtype must be np.int32 for k=3."""
        emb = make_embeddings(N, DIM)
        labels = _balanced_kmeans(emb, k=3, seed=SEED)
        assert labels.dtype == np.int32

    def test_shape_k3(self):
        """Shape must be (N,) for k=3."""
        emb = make_embeddings(N, DIM)
        labels = _balanced_kmeans(emb, k=3, seed=SEED)
        assert labels.shape == (N,)

    def test_values_in_range_k3(self):
        """All label values must be in [0, 2] for k=3."""
        emb = make_embeddings(N, DIM)
        labels = _balanced_kmeans(emb, k=3, seed=SEED)
        _assert_labels_valid(labels, N, k=3)

    def test_deterministic_k3_same_seed(self):
        """Repeated calls with the same seed produce bit-identical labels (k=3)."""
        emb = make_embeddings(N, DIM)
        labels1 = _balanced_kmeans(emb, k=3, seed=SEED)
        labels2 = _balanced_kmeans(emb, k=3, seed=SEED)
        assert np.array_equal(labels1, labels2), (
            "balanced_kmeans k=3: labels differ on repeated calls with same seed"
        )

    def test_values_in_range_k5(self):
        """All label values must be in [0, 4] for k=5."""
        emb = make_embeddings(40, DIM)
        labels = _balanced_kmeans(emb, k=5, seed=SEED)
        _assert_labels_valid(labels, 40, k=5)

    def test_deterministic_k5_same_seed(self):
        """Repeated calls with the same seed produce bit-identical labels (k=5)."""
        emb = make_embeddings(40, DIM)
        labels1 = _balanced_kmeans(emb, k=5, seed=SEED)
        labels2 = _balanced_kmeans(emb, k=5, seed=SEED)
        assert np.array_equal(labels1, labels2)

    def test_custom_n_restarts_k2(self):
        """n_restarts parameter is forwarded correctly (k=2)."""
        emb = make_embeddings(N, DIM)
        labels = _balanced_kmeans(emb, k=2, seed=SEED, n_restarts=5)
        _assert_labels_valid(labels, N, k=2)

    def test_n_restarts_1_still_valid(self):
        """With n_restarts=1, a valid partition is still returned."""
        emb = make_embeddings(N, DIM)
        labels = _balanced_kmeans(emb, k=3, seed=SEED, n_restarts=1)
        _assert_labels_valid(labels, N, k=3)

    @pytest.mark.parametrize("n,k,seed", COMMON_CASES)
    def test_parametrized_valid_output(self, n, k, seed):
        """Parametrized check: dtype, shape, and value range."""
        emb = make_embeddings(n, DIM)
        labels = _balanced_kmeans(emb, k=k, seed=seed)
        _assert_labels_valid(labels, n, k)

    @pytest.mark.parametrize("n,k,seed", COMMON_CASES)
    def test_parametrized_determinism(self, n, k, seed):
        """Parametrized determinism: same seed -> same labels."""
        emb = make_embeddings(n, DIM)
        labels1 = _balanced_kmeans(emb, k=k, seed=seed)
        labels2 = _balanced_kmeans(emb, k=k, seed=seed)
        assert np.array_equal(labels1, labels2)


# ===========================================================================
# _minibatch_kmeans_assign
# ===========================================================================

class TestMinibatchKMeansAssign:
    """Tests for _minibatch_kmeans_assign."""

    def test_dtype_int32_k2(self):
        """dtype must be np.int32 for k=2."""
        emb = make_embeddings(N, DIM)
        labels = _minibatch_kmeans_assign(emb, k=2, seed=SEED)
        assert labels.dtype == np.int32

    def test_shape_k2(self):
        """Shape must be (N,) for k=2."""
        emb = make_embeddings(N, DIM)
        labels = _minibatch_kmeans_assign(emb, k=2, seed=SEED)
        assert labels.shape == (N,)

    def test_values_in_range_k2(self):
        """All label values must be in [0, 1] for k=2."""
        emb = make_embeddings(N, DIM)
        labels = _minibatch_kmeans_assign(emb, k=2, seed=SEED)
        _assert_labels_valid(labels, N, k=2)

    def test_dtype_int32_k3(self):
        """dtype must be np.int32 for k=3."""
        emb = make_embeddings(N, DIM)
        labels = _minibatch_kmeans_assign(emb, k=3, seed=SEED)
        assert labels.dtype == np.int32

    def test_shape_k3(self):
        """Shape must be (N,) for k=3."""
        emb = make_embeddings(N, DIM)
        labels = _minibatch_kmeans_assign(emb, k=3, seed=SEED)
        assert labels.shape == (N,)

    def test_values_in_range_k3(self):
        """All label values must be in [0, 2] for k=3."""
        emb = make_embeddings(N, DIM)
        labels = _minibatch_kmeans_assign(emb, k=3, seed=SEED)
        _assert_labels_valid(labels, N, k=3)

    def test_deterministic_k2_same_seed(self):
        """Repeated calls with the same seed produce bit-identical labels (k=2)."""
        emb = make_embeddings(N, DIM)
        labels1 = _minibatch_kmeans_assign(emb, k=2, seed=SEED)
        labels2 = _minibatch_kmeans_assign(emb, k=2, seed=SEED)
        assert np.array_equal(labels1, labels2), (
            "minibatch k=2: labels differ on repeated calls with same seed"
        )

    def test_deterministic_k3_same_seed(self):
        """Repeated calls with the same seed produce bit-identical labels (k=3)."""
        emb = make_embeddings(N, DIM)
        labels1 = _minibatch_kmeans_assign(emb, k=3, seed=SEED)
        labels2 = _minibatch_kmeans_assign(emb, k=3, seed=SEED)
        assert np.array_equal(labels1, labels2), (
            "minibatch k=3: labels differ on repeated calls with same seed"
        )

    def test_values_in_range_k5(self):
        """All label values must be in [0, 4] for k=5."""
        emb = make_embeddings(40, DIM)
        labels = _minibatch_kmeans_assign(emb, k=5, seed=SEED)
        _assert_labels_valid(labels, 40, k=5)

    def test_deterministic_k5_same_seed(self):
        """Repeated calls with the same seed produce bit-identical labels (k=5)."""
        emb = make_embeddings(40, DIM)
        labels1 = _minibatch_kmeans_assign(emb, k=5, seed=SEED)
        labels2 = _minibatch_kmeans_assign(emb, k=5, seed=SEED)
        assert np.array_equal(labels1, labels2)

    def test_large_n_k2(self):
        """Handles larger N without errors (k=2)."""
        emb = make_embeddings(200, DIM)
        labels = _minibatch_kmeans_assign(emb, k=2, seed=SEED)
        _assert_labels_valid(labels, 200, k=2)

    def test_large_n_k5(self):
        """Handles larger N without errors (k=5)."""
        emb = make_embeddings(200, DIM)
        labels = _minibatch_kmeans_assign(emb, k=5, seed=SEED)
        _assert_labels_valid(labels, 200, k=5)

    @pytest.mark.parametrize("n,k,seed", COMMON_CASES)
    def test_parametrized_valid_output(self, n, k, seed):
        """Parametrized check: dtype, shape, and value range."""
        emb = make_embeddings(n, DIM)
        labels = _minibatch_kmeans_assign(emb, k=k, seed=seed)
        _assert_labels_valid(labels, n, k)

    @pytest.mark.parametrize("n,k,seed", COMMON_CASES)
    def test_parametrized_determinism(self, n, k, seed):
        """Parametrized determinism: same seed -> same labels."""
        emb = make_embeddings(n, DIM)
        labels1 = _minibatch_kmeans_assign(emb, k=k, seed=seed)
        labels2 = _minibatch_kmeans_assign(emb, k=k, seed=seed)
        assert np.array_equal(labels1, labels2)
