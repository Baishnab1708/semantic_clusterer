"""Unit tests for SemanticKSplit split_labels edge cases.

Uses a fast sha256-derived fake embedder injected via ``embedding_model``
to avoid the slow ONNX model.

Covers:
- Empty list input
- All-None input
- k > N_Unique raises ValueError with correct message
- k == N_Unique warning text contains the specified phrase
- All-identical embeddings → tiebreak path with Algorithm_Used == "identical-embeddings-tiebreak"
- TypeError for non-string elements
- Label dtype and shape contract

              12.1, 12.2, 12.3, 12.4, 12.5, 12.6
"""

from __future__ import annotations

import hashlib
import warnings
from typing import List, Sequence

import numpy as np
import pytest

from semantic_clusterer.k_split import SemanticKSplit


# ---------------------------------------------------------------------------
# Fast deterministic fake embedder (sha256-derived, no ONNX)
# ---------------------------------------------------------------------------


def _sha256_embedding(index: int, dim: int = 64) -> np.ndarray:
    """Build a deterministic L2-normalised float32 vector from an index.

    Same algorithm used by test_k_split_algorithms.py:
    1. sha256(str(index)) → 32-byte digest
    2. First 8 bytes interpreted as uint64 seed for numpy RNG
    3. Draw ``dim`` standard-normal floats, then L2-normalise.
    """
    digest = hashlib.sha256(str(index).encode()).digest()
    seed = int.from_bytes(digest[:8], byteorder="little")
    rng = np.random.default_rng(seed)
    raw = rng.standard_normal(dim).astype(np.float32)
    norm = np.linalg.norm(raw)
    if norm > 0:
        raw /= norm
    return raw


class _Sha256Embedder:
    """Fake embedder that returns sha256-derived vectors indexed by position.

    Each call to `embed` maps the i-th text to `_sha256_embedding(i)`.
    This gives deterministic, distinct vectors without any real model.
    """

    def __init__(self, dim: int = 64) -> None:
        self._dim = dim

    def embed(self, texts: Sequence[str], batch_size: int = 64) -> np.ndarray:
        out = np.stack(
            [_sha256_embedding(i, self._dim) for i in range(len(texts))],
            axis=0,
        )
        return out.astype(np.float32)


class _ConstantEmbedder:
    """Fake embedder that returns the *same* vector for every input text.

    Used to trigger the all-identical-embeddings tiebreak path (Req 12.3).
    """

    def __init__(self, dim: int = 64) -> None:
        self._dim = dim
        self._vec = _sha256_embedding(0, dim)  # any fixed vector

    def embed(self, texts: Sequence[str], batch_size: int = 64) -> np.ndarray:
        return np.tile(self._vec, (len(texts), 1)).astype(np.float32)


# Shared instance for tests that need a fake embedder
_FAKE_EMBEDDER = _Sha256Embedder(dim=64)
_CONST_EMBEDDER = _ConstantEmbedder(dim=64)


def _make_ks(k: int, embedder=None, **kwargs) -> SemanticKSplit:
    """Construct a SemanticKSplit with the fake embedder by default."""
    if embedder is None:
        embedder = _FAKE_EMBEDDER
    return SemanticKSplit(k=k, embedding_model=embedder, random_state=42, **kwargs)


# ===========================================================================
# Empty list input
# ===========================================================================


class TestEmptyListInput:
    """split_labels([]) must return an empty int32 array; split([]) must return []."""

    def test_split_labels_empty_returns_empty_array(self):
        """split_labels([]) must return an empty int32 numpy array."""
        ks = _make_ks(k=2)
        result = ks.split_labels([])
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.int32
        assert result.shape == (0,)

    def test_split_empty_returns_empty_list(self):
        """split([]) must return an empty list without raising."""
        ks = _make_ks(k=2)
        result = ks.split([])
        assert result == []

    def test_split_labels_empty_does_not_raise(self):
        """split_labels([]) must not raise any exception."""
        ks = _make_ks(k=2)
        ks.split_labels([])  # must not raise


# ===========================================================================
# All-None input
# ===========================================================================


class TestAllNoneInput:
    """When every element is None / NaN / empty, labels are all -1."""

    def test_all_none_labels_are_minus_one(self):
        """split_labels([None, None, None]) must return an array of all -1."""
        ks = _make_ks(k=2)
        result = ks.split_labels([None, None, None])
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.int32
        assert result.shape == (3,)
        assert np.all(result == -1)

    def test_all_nan_labels_are_minus_one(self):
        """split_labels([float('nan'), float('nan')]) must return all -1."""
        ks = _make_ks(k=2)
        result = ks.split_labels([float("nan"), float("nan")])
        assert np.all(result == -1)

    def test_all_empty_string_labels_are_minus_one(self):
        """split_labels(['', '  ', '']) must return all -1."""
        ks = _make_ks(k=2)
        result = ks.split_labels(["", "  ", ""])
        assert np.all(result == -1)

    def test_split_all_none_returns_empty_list(self):
        """split([None, None]) must return [] without raising."""
        ks = _make_ks(k=2)
        result = ks.split([None, None])
        assert result == []

    def test_all_none_shape_preserved(self):
        """Shape (N,) must be preserved even when all labels are -1."""
        ks = _make_ks(k=2)
        texts = [None] * 5
        result = ks.split_labels(texts)
        assert result.shape == (5,)

    def test_mixed_none_and_valid_gives_minus_one_for_none(self):
        """None positions receive -1; valid-text positions receive ∈ [0, k-1]."""
        texts = ["hello world", None, "foo bar baz", None]
        ks = _make_ks(k=2)
        result = ks.split_labels(texts)
        assert result[1] == -1
        assert result[3] == -1
        assert result[0] >= 0
        assert result[2] >= 0


# ===========================================================================
# K > N_Unique raises ValueError with correct message
# ===========================================================================


class TestKGreaterThanNUnique:
    """k > N_Unique must raise ValueError containing k and N_Unique. (Req 12.1)"""

    def test_k_greater_than_n_unique_raises(self):
        """k > N_Unique must raise ValueError."""
        # 2 distinct texts, k=3 → k > N_Unique
        ks = _make_ks(k=3)
        with pytest.raises(ValueError):
            ks.split_labels(["hello", "world"])

    def test_k_greater_raises_with_requested_k_in_message(self):
        """Error message must contain the requested_k value."""
        ks = _make_ks(k=5)
        with pytest.raises(ValueError, match="5"):
            ks.split_labels(["alpha", "beta", "gamma"])

    def test_k_greater_raises_with_n_unique_in_message(self):
        """Error message must contain the N_Unique value."""
        # 3 unique inputs, k=5 → error should mention 3
        ks = _make_ks(k=5)
        with pytest.raises(ValueError, match="3"):
            ks.split_labels(["alpha", "beta", "gamma"])

    def test_k_greater_raises_with_phrase(self):
        """Error message must contain the phrase 'k cannot exceed'."""
        ks = _make_ks(k=4)
        with pytest.raises(ValueError, match="k cannot exceed"):
            ks.split_labels(["a sentence", "another sentence", "third sentence"])

    def test_k_greater_than_n_unique_after_dedup(self):
        """Duplicates collapse N_Unique; k > N_Unique still raises."""
        # 4 texts but only 2 unique → k=3 > N_Unique=2
        ks = _make_ks(k=3)
        with pytest.raises(ValueError):
            ks.split_labels(
                ["cat cat cat", "dog dog dog", "cat cat cat", "dog dog dog"]
            )

    def test_k_equal_n_unique_does_not_raise(self):
        """k == N_Unique must NOT raise (it only warns). (Req 12.2)"""
        ks = _make_ks(k=2)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = ks.split_labels(["first text here", "second text here"])
        assert result is not None

    def test_k_less_than_n_unique_does_not_raise(self):
        """k < N_Unique must not raise."""
        ks = _make_ks(k=2)
        result = ks.split_labels(
            ["text one", "text two", "text three", "text four"]
        )
        assert result is not None


# ===========================================================================
# K == N_Unique warning text
# ===========================================================================


class TestKEqualsNUniqueWarning:
    """k == N_Unique must emit a UserWarning with the specified phrase. (Req 12.2)"""

    def test_k_equals_n_unique_emits_user_warning(self):
        """A UserWarning must be emitted when k == N_Unique."""
        ks = _make_ks(k=2)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ks.split_labels(["hello world text", "foo bar baz qux"])
        user_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
        assert len(user_warnings) >= 1

    def test_k_equals_n_unique_warning_text_contains_required_phrase(self):
        """Warning message must contain 'k equals the number of unique inputs'."""
        ks = _make_ks(k=2)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ks.split_labels(["hello world text", "foo bar baz qux"])
        messages = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
        assert any("k equals the number of unique inputs" in msg for msg in messages), (
            f"Expected warning phrase not found. Got: {messages}"
        )

    def test_k_equals_n_unique_warning_text_contains_single_point_phrase(self):
        """Warning message must contain 'each cluster will contain a single point'."""
        ks = _make_ks(k=3)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ks.split_labels(
                ["apple is a fruit", "banana is yellow", "cherry is red"]
            )
        messages = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
        assert any("each cluster will contain a single point" in msg for msg in messages), (
            f"Expected phrase not found in warnings. Got: {messages}"
        )

    def test_k_equals_n_unique_produces_k_clusters(self):
        """When k == N_Unique, the result must have exactly k clusters."""
        ks = _make_ks(k=3)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            labels = ks.split_labels(
                ["apple is a fruit", "banana is yellow", "cherry is red"]
            )
        valid_labels = labels[labels >= 0]
        assert set(valid_labels.tolist()) == {0, 1, 2}

    def test_k_not_equal_n_unique_no_warning(self):
        """When k < N_Unique, no 'k equals' warning must be emitted."""
        ks = _make_ks(k=2)
        texts = ["alpha text", "beta text", "gamma text", "delta text"]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ks.split_labels(texts)
        messages = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
        assert not any("k equals the number of unique inputs" in msg for msg in messages)


# ===========================================================================
# All-identical embeddings tiebreak path
# ===========================================================================


class TestAllIdenticalEmbeddingsTiebreak:
    """All-identical embeddings must trigger round-robin with Algorithm_Used set. (Req 12.3)"""

    def test_identical_embeddings_algorithm_used_in_report(self):
        """split_with_report with constant embedder sets algorithm_used to
        'identical-embeddings-tiebreak' in the report."""
        ks = SemanticKSplit(k=2, embedding_model=_CONST_EMBEDDER, random_state=42)
        labels, report = ks.split_with_report(
            ["same text alpha", "same text beta", "same text gamma", "same text delta"]
        )
        assert report.chosen_params.get("algorithm_used") == "identical-embeddings-tiebreak", (
            f"Expected 'identical-embeddings-tiebreak' but got "
            f"{report.chosen_params.get('algorithm_used')!r}"
        )

    def test_identical_embeddings_warning_in_report(self):
        """'identical-embeddings-tiebreak' must appear in report.warnings."""
        ks = SemanticKSplit(k=2, embedding_model=_CONST_EMBEDDER, random_state=42)
        _, report = ks.split_with_report(
            ["same text alpha", "same text beta", "same text gamma", "same text delta"]
        )
        assert "identical-embeddings-tiebreak" in report.warnings, (
            f"Expected warning string in {report.warnings!r}"
        )

    def test_identical_embeddings_produces_k_non_empty_clusters(self):
        """Round-robin assignment must produce exactly k non-empty clusters."""
        n = 6
        k = 3
        ks = SemanticKSplit(k=k, embedding_model=_CONST_EMBEDDER, random_state=42)
        texts = [f"identical text {i}" for i in range(n)]
        labels = ks.split_labels(texts)
        valid = labels[labels >= 0]
        assert set(valid.tolist()) == set(range(k)), (
            f"Expected labels {set(range(k))}, got {set(valid.tolist())}"
        )

    def test_identical_embeddings_round_robin_assignment(self):
        """Round-robin means text i goes to cluster i % k in input order."""
        k = 3
        n = 9
        ks = SemanticKSplit(k=k, embedding_model=_CONST_EMBEDDER, random_state=42)
        texts = [f"text number {i}" for i in range(n)]
        labels = ks.split_labels(texts)
        # All texts are unique pre-processed strings → N_Unique = n
        # Expected: labels[i] == i % k for i in range(n)
        expected = np.array([i % k for i in range(n)], dtype=np.int32)
        assert np.array_equal(labels, expected), (
            f"Expected round-robin {expected.tolist()}, got {labels.tolist()}"
        )

    def test_identical_embeddings_k2_round_robin(self):
        """Round-robin for k=2: alternates 0, 1, 0, 1, ..."""
        k = 2
        texts = [f"word{i}" for i in range(6)]
        ks = SemanticKSplit(k=k, embedding_model=_CONST_EMBEDDER, random_state=42)
        labels = ks.split_labels(texts)
        expected = np.array([i % k for i in range(6)], dtype=np.int32)
        assert np.array_equal(labels, expected), (
            f"Expected {expected.tolist()}, got {labels.tolist()}"
        )

    def test_identical_embeddings_dtype_int32(self):
        """Labels from tiebreak path must have dtype np.int32. (Req 8.1)"""
        ks = SemanticKSplit(k=2, embedding_model=_CONST_EMBEDDER, random_state=42)
        labels = ks.split_labels(["abc def ghi", "xyz uvw rst"])
        assert labels.dtype == np.int32

    def test_identical_embeddings_shape(self):
        """Labels from tiebreak path must have shape (N_Input,). (Req 8.1)"""
        texts = [f"text {i}" for i in range(8)]
        ks = SemanticKSplit(k=2, embedding_model=_CONST_EMBEDDER, random_state=42)
        labels = ks.split_labels(texts)
        assert labels.shape == (len(texts),)


# ===========================================================================
# TypeError for non-string elements
# ===========================================================================


class TestTypeErrorForNonStringElements:
    """Non-string, non-None, non-NaN elements must raise TypeError. (Req 12.6)"""

    def test_int_element_raises_type_error(self):
        """An integer element in texts must raise TypeError."""
        ks = _make_ks(k=2)
        with pytest.raises(TypeError):
            ks.split_labels(["valid string", 42, "another string"])

    def test_list_element_raises_type_error(self):
        """A list element in texts must raise TypeError."""
        ks = _make_ks(k=2)
        with pytest.raises(TypeError):
            ks.split_labels(["valid", ["nested", "list"]])

    def test_dict_element_raises_type_error(self):
        """A dict element in texts must raise TypeError."""
        ks = _make_ks(k=2)
        with pytest.raises(TypeError):
            ks.split_labels(["valid", {"key": "value"}])

    def test_bool_element_raises_type_error(self):
        """A bool element (not a string, None or NaN) must raise TypeError."""
        ks = _make_ks(k=2)
        with pytest.raises(TypeError):
            ks.split_labels(["valid", True])

    def test_bytes_element_raises_type_error(self):
        """A bytes element in texts must raise TypeError."""
        ks = _make_ks(k=2)
        with pytest.raises(TypeError):
            ks.split_labels(["valid", b"bytes string"])

    def test_none_does_not_raise(self):
        """None is an allowed missing value and must not raise TypeError."""
        ks = _make_ks(k=2)
        # needs enough valid strings so k <= N_Unique
        texts = ["hello world alpha", "goodbye world beta", None]
        result = ks.split_labels(texts)
        assert result[2] == -1

    def test_nan_does_not_raise(self):
        """float('nan') is an allowed missing value and must not raise TypeError."""
        ks = _make_ks(k=2)
        texts = ["hello world alpha", "goodbye world beta", float("nan")]
        result = ks.split_labels(texts)
        assert result[2] == -1


# ===========================================================================
# 8.3, 8.4 — label dtype, shape, and value contract
# ===========================================================================


class TestLabelContract:
    """Label array must have dtype int32, shape (N,), values in {-1} ∪ [0, k-1]."""

    def test_dtype_is_int32(self):
        """Output dtype must be np.int32. (Req 8.1)"""
        texts = [f"sentence number {i}" for i in range(5)]
        ks = _make_ks(k=2)
        labels = ks.split_labels(texts)
        assert labels.dtype == np.int32

    def test_shape_equals_n_input(self):
        """Output shape must be (len(texts),). (Req 8.1)"""
        texts = [f"sentence {i}" for i in range(7)]
        ks = _make_ks(k=3)
        labels = ks.split_labels(texts)
        assert labels.shape == (7,)

    def test_valid_labels_in_range(self):
        """Valid labels (>= 0) must be in [0, k-1]. (Req 8.2)"""
        texts = [f"item {i}" for i in range(8)]
        k = 3
        ks = _make_ks(k=k)
        labels = ks.split_labels(texts)
        valid = labels[labels >= 0]
        assert np.all(valid < k), f"Labels contain value >= k={k}: {valid}"
        assert np.all(valid >= 0), "Labels contain negative value besides -1"

    def test_filtered_rows_are_minus_one(self):
        """None rows must have label -1. (Req 8.3)"""
        texts = ["valid one here", None, "valid two here", None]
        ks = _make_ks(k=2)
        labels = ks.split_labels(texts)
        assert labels[1] == -1
        assert labels[3] == -1

    def test_valid_rows_not_minus_one(self):
        """Valid rows must not have label -1. (Req 8.4)"""
        texts = [f"good text {i}" for i in range(6)]
        ks = _make_ks(k=2)
        labels = ks.split_labels(texts)
        assert np.all(labels != -1), f"Valid rows received label -1: {labels}"

    def test_all_k_clusters_covered(self):
        """The set of valid labels must equal {0, ..., k-1}. (Req 8.5)"""
        texts = [f"text item {i}" for i in range(10)]
        k = 4
        ks = _make_ks(k=k)
        labels = ks.split_labels(texts)
        valid = set(labels[labels >= 0].tolist())
        assert valid == set(range(k)), f"Expected {set(range(k))}, got {valid}"
