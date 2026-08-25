"""Unit tests for _oversized_subsample_and_assign_k.

Covers:
1. _OVERSIZED_ERROR_MSG_FMT message is byte-for-byte identical to what
   SemanticClusterer raises.
2. "oversized-subsampled" fires exactly once and is appended to
   trace.warnings.
3. Subsample index set is deterministic across runs sharing the same seed
.
4. Final label array satisfies shape (N,) and values in [0, k-1]
.

Tests that require N > 200_000 patch ``_OVERSIZED_LIMIT`` in
``semantic_clusterer.k_algorithms.oversized`` to a small value (50) so
the test data remains tiny and runs fast.

"""

from __future__ import annotations

import types
from unittest.mock import patch

import numpy as np
import pytest

from semantic_clusterer.k_algorithms.oversized import _oversized_subsample_and_assign_k
from semantic_clusterer.core import _OVERSIZED_ERROR_MSG_FMT, _OVERSIZED_WARN_MSG_FMT


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Patched limit used in tests so data stays small
_TEST_LIMIT = 50


def _make_trace():
    """Minimal trace whose warn() appends to trace.warnings."""
    trace = types.SimpleNamespace(warnings=[])
    trace.warn = lambda msg: trace.warnings.append(msg)
    return trace


def _make_emb(n: int, d: int = 8, seed: int = 0) -> np.ndarray:
    """Return L2-normalised random embeddings of shape (n, d)."""
    rng = np.random.default_rng(seed)
    emb = rng.standard_normal((n, d)).astype(np.float32)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    return emb / np.where(norms == 0, 1.0, norms)


# ---------------------------------------------------------------------------
# Tests: gating ValueError message
# ---------------------------------------------------------------------------


class TestOversizedErrorMessageFormat:
    """_OVERSIZED_ERROR_MSG_FMT produces the exact text SemanticClusterer raises."""

    def test_error_msg_exact_wording(self):
        """_OVERSIZED_ERROR_MSG_FMT.format(N=n) produces the expected exact string."""
        n = 300_000
        msg = _OVERSIZED_ERROR_MSG_FMT.format(N=n)
        expected = (
            f"Dataset size {n} exceeds the hard limit of 200_000. "
            "Set allow_oversized_datasets=True to enable subsampling."
        )
        assert msg == expected

    def test_error_msg_contains_hard_limit_phrase(self):
        """Error message contains the required key phrases."""
        msg = _OVERSIZED_ERROR_MSG_FMT.format(N=250_000)
        assert "exceeds the hard limit of 200_000" in msg
        assert "allow_oversized_datasets=True" in msg

    def test_error_msg_embeds_n_value(self):
        """The {N} placeholder is replaced by the actual dataset size."""
        for n in (200_001, 350_000, 1_000_000):
            msg = _OVERSIZED_ERROR_MSG_FMT.format(N=n)
            assert str(n) in msg

    def test_warn_msg_exact_wording(self):
        """_OVERSIZED_WARN_MSG_FMT.format(N=n) produces the expected exact string."""
        n = 300_000
        msg = _OVERSIZED_WARN_MSG_FMT.format(N=n)
        expected = f"Dataset size {n} exceeds 200_000; subsampling to 200_000 points."
        assert msg == expected

    def test_warn_msg_contains_subsampling_phrase(self):
        """Warning message contains the subsampling phrase."""
        msg = _OVERSIZED_WARN_MSG_FMT.format(N=250_000)
        assert "subsampling to 200_000 points" in msg

    def test_semantic_clusterer_raises_with_matching_message(self):
        """SemanticClusterer raises ValueError whose text matches _OVERSIZED_ERROR_MSG_FMT.

        Skipped when hdbscan is unavailable.  This pins the behavior:
        both SemanticClusterer and SemanticKSplit format from the same constant.
        """
        pytest.importorskip("hdbscan")
        from semantic_clusterer.core import SemanticClusterer

        sc = SemanticClusterer()
        n = 300_000
        expected_msg = _OVERSIZED_ERROR_MSG_FMT.format(N=n)

        with pytest.raises(ValueError) as exc_info:
            # Pre-flight check fires before any preprocessing — fast path.
            sc._run_clustering(["text"] * n, trace=None)

        assert str(exc_info.value) == expected_msg


# ---------------------------------------------------------------------------
# Tests: warning fires exactly once
# ---------------------------------------------------------------------------


class TestOversizedWarning:
    """'oversized-subsampled' appended to trace.warnings exactly once."""

    def test_warning_appended_exactly_once(self):
        """One call -> exactly one 'oversized-subsampled' in trace.warnings."""
        N, D, k = 100, 8, 3
        emb = _make_emb(N, D, seed=0)
        trace = _make_trace()

        with patch("semantic_clusterer.k_algorithms.oversized._OVERSIZED_LIMIT", _TEST_LIMIT):
            _oversized_subsample_and_assign_k(emb, k, seed=42, trace=trace)

        assert trace.warnings.count("oversized-subsampled") == 1

    def test_warning_is_the_exact_literal_string(self):
        """The exact string 'oversized-subsampled' is appended, not a variant."""
        N, D, k = 100, 8, 2
        emb = _make_emb(N, D, seed=1)
        trace = _make_trace()

        with patch("semantic_clusterer.k_algorithms.oversized._OVERSIZED_LIMIT", _TEST_LIMIT):
            _oversized_subsample_and_assign_k(emb, k, seed=0, trace=trace)

        assert "oversized-subsampled" in trace.warnings

    def test_warning_not_duplicated_in_single_call(self):
        """No duplicate warning entries for a single call."""
        N, D, k = 100, 8, 2
        emb = _make_emb(N, D, seed=2)
        trace = _make_trace()

        with patch("semantic_clusterer.k_algorithms.oversized._OVERSIZED_LIMIT", _TEST_LIMIT):
            _oversized_subsample_and_assign_k(emb, k, seed=0, trace=trace)

        oversized_warnings = [w for w in trace.warnings if w == "oversized-subsampled"]
        assert len(oversized_warnings) == 1

    def test_warning_present_for_k2(self):
        """Warning fires for k=2 as well as larger k values."""
        N, D, k = 100, 8, 2
        emb = _make_emb(N, D, seed=3)
        trace = _make_trace()

        with patch("semantic_clusterer.k_algorithms.oversized._OVERSIZED_LIMIT", _TEST_LIMIT):
            _oversized_subsample_and_assign_k(emb, k, seed=42, trace=trace)

        assert "oversized-subsampled" in trace.warnings

    def test_warning_present_for_k5(self):
        """Warning fires for k=5 — multi-cluster case."""
        N, D, k = 100, 8, 5
        emb = _make_emb(N, D, seed=4)
        trace = _make_trace()

        with patch("semantic_clusterer.k_algorithms.oversized._OVERSIZED_LIMIT", _TEST_LIMIT):
            _oversized_subsample_and_assign_k(emb, k, seed=42, trace=trace)

        assert trace.warnings.count("oversized-subsampled") == 1


# ---------------------------------------------------------------------------
# Tests: subsample index determinism
# ---------------------------------------------------------------------------


class TestSubsampleDeterminism:
    """Subsample index set is deterministic across runs sharing the same seed."""

    def test_same_seed_same_labels(self):
        """Two calls with the same seed and embeddings return identical label arrays."""
        N, D, k = 100, 8, 3
        emb = _make_emb(N, D, seed=0)
        trace1, trace2 = _make_trace(), _make_trace()

        with patch("semantic_clusterer.k_algorithms.oversized._OVERSIZED_LIMIT", _TEST_LIMIT):
            labels1 = _oversized_subsample_and_assign_k(emb, k, seed=42, trace=trace1)
            labels2 = _oversized_subsample_and_assign_k(emb, k, seed=42, trace=trace2)

        assert np.array_equal(labels1, labels2)

    def test_same_seed_repeated_calls_are_identical(self):
        """Multiple repeated calls with the same seed produce bit-identical output."""
        N, D, k = 100, 8, 2
        emb = _make_emb(N, D, seed=7)
        results = []

        with patch("semantic_clusterer.k_algorithms.oversized._OVERSIZED_LIMIT", _TEST_LIMIT):
            for _ in range(3):
                trace = _make_trace()
                labels = _oversized_subsample_and_assign_k(emb, k, seed=99, trace=trace)
                results.append(labels.copy())

        for lab in results[1:]:
            assert np.array_equal(results[0], lab)

    def test_rng_choice_index_determinism(self):
        """np.random.default_rng(seed).choice produces the same indices across calls.

        This is the low-level guarantee that makes oversized dataset handling work.
        """
        seed = 42
        N = 200
        LIMIT = 50

        rng_a = np.random.default_rng(seed)
        idx_a = rng_a.choice(N, size=LIMIT, replace=False)
        idx_a.sort()

        rng_b = np.random.default_rng(seed)
        idx_b = rng_b.choice(N, size=LIMIT, replace=False)
        idx_b.sort()

        assert np.array_equal(idx_a, idx_b)

    def test_different_seeds_may_differ(self):
        """Two calls with different seeds can produce different assignments.

        Shape and dtype must always be correct regardless of seed.
        """
        N, D, k = 100, 8, 3
        emb = _make_emb(N, D, seed=0)
        trace1, trace2 = _make_trace(), _make_trace()

        with patch("semantic_clusterer.k_algorithms.oversized._OVERSIZED_LIMIT", _TEST_LIMIT):
            labels1 = _oversized_subsample_and_assign_k(emb, k, seed=0, trace=trace1)
            labels2 = _oversized_subsample_and_assign_k(emb, k, seed=999, trace=trace2)

        assert labels1.shape == (N,)
        assert labels2.shape == (N,)
        assert labels1.dtype == np.int32
        assert labels2.dtype == np.int32


# ---------------------------------------------------------------------------
# Tests: final label array properties
# ---------------------------------------------------------------------------


class TestLabelArrayProperties:
    """Final label array satisfies shape (N,) and values in [0, k-1]."""

    def test_output_shape_matches_input_rows(self):
        """Output shape is (N,) where N is the number of input rows."""
        N, D, k = 100, 8, 3
        emb = _make_emb(N, D, seed=0)
        trace = _make_trace()

        with patch("semantic_clusterer.k_algorithms.oversized._OVERSIZED_LIMIT", _TEST_LIMIT):
            labels = _oversized_subsample_and_assign_k(emb, k, seed=42, trace=trace)

        assert labels.shape == (N,)

    def test_output_dtype_is_int32(self):
        """Output dtype is np.int32."""
        N, D, k = 100, 8, 2
        emb = _make_emb(N, D, seed=1)
        trace = _make_trace()

        with patch("semantic_clusterer.k_algorithms.oversized._OVERSIZED_LIMIT", _TEST_LIMIT):
            labels = _oversized_subsample_and_assign_k(emb, k, seed=42, trace=trace)

        assert labels.dtype == np.int32

    def test_no_negative_labels(self):
        """No -1 labels — the oversized function assigns every row to a cluster."""
        N, D, k = 100, 8, 3
        emb = _make_emb(N, D, seed=2)
        trace = _make_trace()

        with patch("semantic_clusterer.k_algorithms.oversized._OVERSIZED_LIMIT", _TEST_LIMIT):
            labels = _oversized_subsample_and_assign_k(emb, k, seed=42, trace=trace)

        assert not np.any(labels == -1)

    def test_values_in_range_0_to_k_minus_1(self):
        """All label values satisfy 0 <= label <= k-1."""
        N, D, k = 100, 8, 4
        emb = _make_emb(N, D, seed=3)
        trace = _make_trace()

        with patch("semantic_clusterer.k_algorithms.oversized._OVERSIZED_LIMIT", _TEST_LIMIT):
            labels = _oversized_subsample_and_assign_k(emb, k, seed=42, trace=trace)

        assert np.all(labels >= 0)
        assert np.all(labels < k)

    def test_all_k_labels_present(self):
        """Every label in [0, k-1] appears at least once."""
        N, D, k = 100, 8, 3
        emb = _make_emb(N, D, seed=4)
        trace = _make_trace()

        with patch("semantic_clusterer.k_algorithms.oversized._OVERSIZED_LIMIT", _TEST_LIMIT):
            labels = _oversized_subsample_and_assign_k(emb, k, seed=42, trace=trace)

        assert set(int(x) for x in labels) == set(range(k))

    def test_shape_and_range_for_k2(self):
        """Shape and value range correct for k=2 (binary split)."""
        N, D, k = 100, 8, 2
        emb = _make_emb(N, D, seed=5)
        trace = _make_trace()

        with patch("semantic_clusterer.k_algorithms.oversized._OVERSIZED_LIMIT", _TEST_LIMIT):
            labels = _oversized_subsample_and_assign_k(emb, k, seed=42, trace=trace)

        assert labels.shape == (N,)
        assert labels.dtype == np.int32
        assert set(int(x) for x in labels) == {0, 1}

    def test_shape_and_range_for_various_k(self):
        """Shape (N,) and full label coverage holds for k in {2, 3, 5}."""
        N, D = 100, 8
        emb = _make_emb(N, D, seed=6)

        for k in (2, 3, 5):
            trace = _make_trace()
            with patch(
                "semantic_clusterer.k_algorithms.oversized._OVERSIZED_LIMIT", _TEST_LIMIT
            ):
                labels = _oversized_subsample_and_assign_k(emb, k, seed=42, trace=trace)

            assert labels.shape == (N,), f"shape mismatch for k={k}"
            assert labels.dtype == np.int32, f"dtype mismatch for k={k}"
            assert np.all(labels >= 0) and np.all(labels < k), f"out-of-range for k={k}"
            assert set(int(x) for x in labels) == set(range(k)), f"missing labels for k={k}"

    def test_out_of_sample_rows_all_assigned(self):
        """Rows not in the subsample (out-of-sample) receive valid cluster labels."""
        N, D, k = 100, 8, 3
        emb = _make_emb(N, D, seed=7)
        trace = _make_trace()

        with patch("semantic_clusterer.k_algorithms.oversized._OVERSIZED_LIMIT", _TEST_LIMIT):
            labels = _oversized_subsample_and_assign_k(emb, k, seed=42, trace=trace)

        # All N rows (including out-of-sample) must have a valid assignment
        assert labels.shape == (N,)
        assert np.all(labels >= 0)
        assert np.all(labels < k)
