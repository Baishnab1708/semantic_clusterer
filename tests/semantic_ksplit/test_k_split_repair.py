"""Unit tests for _repair_empty_clusters, _all_identical, and _round_robin_labels.

Covers:
- _repair_empty_clusters: targets largest cluster, ties broken by lowest id,
  all k labels present after repair, warning appended exactly once.
- _all_identical: True for constant-embedding fixtures, False otherwise.
- _round_robin_labels: produces [i % k for i in range(N)].

"""

from __future__ import annotations

import types

import numpy as np
import pytest

from semantic_clusterer.k_algorithms.repair import _repair_empty_clusters
from semantic_clusterer.k_algorithms.degenerate import _all_identical, _round_robin_labels


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_trace():
    """Minimal trace object whose warn() appends to trace.warnings."""
    trace = types.SimpleNamespace(warnings=[])
    trace.warn = lambda msg: trace.warnings.append(msg)
    return trace


def _make_emb(n: int, d: int = 16, seed: int = 0) -> np.ndarray:
    """Return L2-normalised random embeddings of shape (n, d)."""
    rng = np.random.default_rng(seed)
    emb = rng.standard_normal((n, d)).astype(np.float32)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    return emb / np.where(norms == 0, 1.0, norms)


# ---------------------------------------------------------------------------
# Tests: _repair_empty_clusters
# ---------------------------------------------------------------------------


class TestRepairEmptyClusters:
    """Unit tests for _repair_empty_clusters."""

    # --- no-op when already complete -----------------------------------------

    def test_no_repair_needed_leaves_labels_unchanged(self):
        """If all k labels are already present, labels are not modified."""
        k = 3
        emb = _make_emb(9)
        labels = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=np.int32)
        labels_before = labels.copy()
        trace = _make_trace()

        _repair_empty_clusters(emb, labels, k, seed=42, trace=trace)

        assert np.array_equal(labels, labels_before)

    def test_no_repair_needed_no_warning_emitted(self):
        """No warning when repair was not needed."""
        k = 2
        emb = _make_emb(10)
        labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=np.int32)
        trace = _make_trace()

        _repair_empty_clusters(emb, labels, k, seed=42, trace=trace)

        assert "empty-cluster-repaired" not in trace.warnings

    # --- basic repair ---------------------------------------------------------

    def test_repair_fills_missing_label(self):
        """After repair, all k labels in [0, k-1] are present."""
        k = 3
        # Labels 0 and 1 assigned; label 2 is missing.
        emb = _make_emb(12, seed=0)
        labels = np.array([0] * 4 + [1] * 8, dtype=np.int32)
        trace = _make_trace()

        _repair_empty_clusters(emb, labels, k, seed=42, trace=trace)

        present = set(int(x) for x in np.unique(labels))
        assert present == set(range(k))

    def test_result_values_in_valid_range(self):
        """All labels after repair are in [0, k-1]."""
        k = 3
        emb = _make_emb(15, seed=1)
        labels = np.array([0] * 5 + [1] * 10, dtype=np.int32)
        trace = _make_trace()

        result = _repair_empty_clusters(emb, labels, k, seed=42, trace=trace)

        assert np.all(result >= 0)
        assert np.all(result < k)

    def test_result_dtype_is_int32(self):
        """Returned labels array has dtype int32."""
        k = 3
        emb = _make_emb(12, seed=2)
        labels = np.array([0] * 4 + [1] * 8, dtype=np.int32)
        trace = _make_trace()

        result = _repair_empty_clusters(emb, labels, k, seed=42, trace=trace)

        assert result.dtype == np.int32

    def test_return_value_is_same_array(self):
        """The returned object is the in-place modified input labels array."""
        k = 3
        emb = _make_emb(12, seed=3)
        labels = np.array([0] * 4 + [1] * 8, dtype=np.int32)
        trace = _make_trace()

        result = _repair_empty_clusters(emb, labels, k, seed=42, trace=trace)

        assert result is labels

    # --- targets largest cluster ----------------------------

    def test_repair_targets_largest_cluster(self):
        """The largest cluster loses points to the missing label.

        Largest cluster is targeted; lowest id on ties.
        Cluster 0 has 4 points (smaller), cluster 1 has 8 points (larger).
        Label 2 is missing.  The repair must bisect cluster 1, leaving
        cluster 0 intact.
        """
        k = 3
        emb = _make_emb(12, seed=10)
        labels = np.array([0] * 4 + [1] * 8, dtype=np.int32)
        trace = _make_trace()

        _repair_empty_clusters(emb, labels, k, seed=42, trace=trace)

        # Cluster 0 (smaller, 4 pts) was NOT the target — must be untouched.
        assert np.sum(labels == 0) == 4
        # Cluster 2 received points from the bisected cluster 1.
        assert np.sum(labels == 2) >= 1

    # --- tie-break by lowest cluster id --------------------

    def test_repair_ties_broken_by_lowest_cluster_id(self):
        """On a size tie the cluster with the lower id is targeted.

        Ties broken by lowest cluster id.
        Clusters 0 and 1 each have 6 points (tied).  Label 2 is missing.
        The repair must bisect cluster 0 (lower id), leaving cluster 1 intact.
        """
        k = 3
        emb = _make_emb(12, seed=20)
        labels = np.array([0] * 6 + [1] * 6, dtype=np.int32)
        trace = _make_trace()

        _repair_empty_clusters(emb, labels, k, seed=42, trace=trace)

        # Cluster 1 (higher id) was NOT the target — must be untouched.
        assert np.sum(labels == 1) == 6
        # Cluster 0 lost some points to cluster 2.
        assert np.sum(labels == 0) < 6
        assert np.sum(labels == 2) >= 1

    # --- warning emitted exactly once ----------------------

    def test_repair_warning_appended_exactly_once(self):
        """Warning 'empty-cluster-repaired' is appended exactly once per call.


        """
        k = 3
        emb = _make_emb(12, seed=30)
        labels = np.array([0] * 4 + [1] * 8, dtype=np.int32)
        trace = _make_trace()

        _repair_empty_clusters(emb, labels, k, seed=42, trace=trace)

        assert trace.warnings.count("empty-cluster-repaired") == 1

    def test_repair_warning_exactly_once_with_multiple_missing_labels(self):
        """Warning is appended exactly once even when several repair iterations run.

        A single call appends 'empty-cluster-repaired' once
        regardless of how many iterations were required.
        """
        k = 4
        # All 20 points in cluster 0; labels 1, 2, 3 all missing.
        emb = _make_emb(20, seed=40)
        labels = np.zeros(20, dtype=np.int32)
        trace = _make_trace()

        _repair_empty_clusters(emb, labels, k, seed=42, trace=trace)

        assert set(int(x) for x in np.unique(labels)) == set(range(k))
        assert trace.warnings.count("empty-cluster-repaired") == 1

    # --- determinism ----------------------------------------------------------

    def test_repair_is_deterministic(self):
        """Two calls with the same (emb, labels, k, seed) produce identical results."""
        k = 3
        emb = _make_emb(12, seed=50)

        labels1 = np.array([0] * 4 + [1] * 8, dtype=np.int32)
        labels2 = labels1.copy()
        trace1, trace2 = _make_trace(), _make_trace()

        _repair_empty_clusters(emb, labels1, k, seed=42, trace=trace1)
        _repair_empty_clusters(emb, labels2, k, seed=42, trace=trace2)

        assert np.array_equal(labels1, labels2)


# ---------------------------------------------------------------------------
# Tests: _all_identical
# ---------------------------------------------------------------------------


class TestAllIdentical:
    """Unit tests for _all_identical."""

    def test_constant_rows_returns_true(self):
        """Returns True when all rows are identical."""
        row = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        emb = np.tile(row, (10, 1))
        assert _all_identical(emb) is True

    def test_single_row_returns_true(self):
        """A single-row matrix is trivially identical."""
        emb = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        assert _all_identical(emb) is True

    def test_all_zeros_returns_true(self):
        """An all-zeros matrix is trivially identical."""
        emb = np.zeros((8, 6), dtype=np.float32)
        assert _all_identical(emb) is True

    def test_different_rows_returns_false(self):
        """Returns False when rows differ by more than default tolerance."""
        emb = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        assert _all_identical(emb) is False

    def test_random_embeddings_return_false(self):
        """Randomly generated embeddings are not identical (with overwhelming probability)."""
        rng = np.random.default_rng(0)
        emb = rng.standard_normal((20, 16)).astype(np.float32)
        assert _all_identical(emb) is False

    def test_within_default_tolerance_returns_true(self):
        """Perturbations smaller than 1e-9 are within the default tolerance."""
        base = np.array([0.5, 0.5, 0.5], dtype=np.float64)
        emb = np.tile(base, (5, 1))
        emb[2] += 5e-10  # strictly within 1e-9
        assert _all_identical(emb) is True

    def test_outside_default_tolerance_returns_false(self):
        """Perturbation larger than 1e-9 causes the function to return False."""
        base = np.array([0.5, 0.5, 0.5], dtype=np.float64)
        emb = np.tile(base, (5, 1))
        emb[3, 1] += 2e-9  # strictly outside 1e-9
        assert _all_identical(emb) is False

    def test_custom_tolerance_true(self):
        """Custom tolerance of 0.1 allows a difference of 0.05."""
        emb = np.array([[1.0, 0.0], [1.05, 0.0]], dtype=np.float32)
        assert _all_identical(emb, tol=0.1) is True

    def test_custom_tolerance_false(self):
        """Custom tolerance of 0.01 rejects a difference of 0.05."""
        emb = np.array([[1.0, 0.0], [1.05, 0.0]], dtype=np.float32)
        assert _all_identical(emb, tol=0.01) is False

    def test_returns_python_bool(self):
        """Return value is a plain Python bool, not a numpy bool."""
        emb = np.ones((5, 4), dtype=np.float32)
        result = _all_identical(emb)
        assert isinstance(result, bool)

    def test_near_identical_multi_row(self):
        """Returns True when a large matrix is built from tiles of the same row."""
        row = np.array([0.3, -0.1, 0.7, 0.0], dtype=np.float32)
        emb = np.tile(row, (100, 1))
        assert _all_identical(emb) is True


# ---------------------------------------------------------------------------
# Tests: _round_robin_labels
# ---------------------------------------------------------------------------


class TestRoundRobinLabels:
    """Unit tests for _round_robin_labels."""

    def test_basic_round_robin_pattern(self):
        """Labels equal [i % k for i in range(N)]."""
        N, k = 7, 3
        result = _round_robin_labels(N, k)
        expected = np.array([i % k for i in range(N)], dtype=np.int32)
        assert np.array_equal(result, expected)

    def test_n_equals_k(self):
        """When N == k, labels are exactly [0, 1, ..., k-1]."""
        k = 5
        result = _round_robin_labels(k, k)
        expected = np.arange(k, dtype=np.int32)
        assert np.array_equal(result, expected)

    def test_n_exact_multiple_of_k(self):
        """Pattern repeats perfectly when N is a multiple of k."""
        N, k = 12, 4
        result = _round_robin_labels(N, k)
        expected = np.array([i % k for i in range(N)], dtype=np.int32)
        assert np.array_equal(result, expected)

    def test_n_not_multiple_of_k(self):
        """Truncated last cycle when N is not a multiple of k."""
        N, k = 10, 3  # pattern: 0,1,2,0,1,2,0,1,2,0
        result = _round_robin_labels(N, k)
        expected = np.array([i % k for i in range(N)], dtype=np.int32)
        assert np.array_equal(result, expected)

    def test_k_equals_2_alternates(self):
        """k=2 produces 0,1,0,1,... alternating pattern."""
        N, k = 8, 2
        result = _round_robin_labels(N, k)
        expected = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int32)
        assert np.array_equal(result, expected)

    def test_all_labels_covered_when_n_ge_k(self):
        """Every label in [0, k-1] appears at least once when N >= k."""
        N, k = 20, 7
        result = _round_robin_labels(N, k)
        assert set(result.tolist()) == set(range(k))

    def test_shape_is_n(self):
        """Output shape is (N,)."""
        N, k = 15, 4
        result = _round_robin_labels(N, k)
        assert result.shape == (N,)

    def test_dtype_is_int32(self):
        """Output dtype is np.int32."""
        result = _round_robin_labels(10, 3)
        assert result.dtype == np.int32

    def test_n_equals_1(self):
        """N=1 always produces label 0 (1 % k == 0 for any k >= 1)."""
        result = _round_robin_labels(1, 2)
        assert np.array_equal(result, np.array([0], dtype=np.int32))

    def test_large_n(self):
        """Correct output for large N."""
        N, k = 1000, 7
        result = _round_robin_labels(N, k)
        expected = np.array([i % k for i in range(N)], dtype=np.int32)
        assert np.array_equal(result, expected)

    def test_deterministic(self):
        """Two calls with the same (N, k) return identical arrays."""
        r1 = _round_robin_labels(15, 4)
        r2 = _round_robin_labels(15, 4)
        assert np.array_equal(r1, r2)

    def test_values_within_valid_range(self):
        """All output values are in [0, k-1]."""
        N, k = 50, 6
        result = _round_robin_labels(N, k)
        assert np.all(result >= 0)
        assert np.all(result < k)
