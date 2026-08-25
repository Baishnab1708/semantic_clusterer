"""Unit tests for SemanticKSplit.split() — simple and detailed return formats.

Covers:
- simple return_format returns List[List[str]] of length exactly k (Req 9.1)
- detailed return_format returns list of length exactly k with required dict keys
  (cluster_id, representative, items, size, confidence) (Req 9.2)
- cluster_id at position c equals c (Req 9.3)
- items at position c contains original input strings in original input order (Req 9.4)
- No valid rows after preprocessing → return [] without raising (Req 9.5)
- Invalid return_format raises ValueError with exact wording (Req 2.2, 2.3)

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
    """Build a deterministic L2-normalised float32 vector from an index."""
    digest = hashlib.sha256(str(index).encode()).digest()
    seed = int.from_bytes(digest[:8], byteorder="little")
    rng = np.random.default_rng(seed)
    raw = rng.standard_normal(dim).astype(np.float32)
    norm = np.linalg.norm(raw)
    if norm > 0:
        raw /= norm
    return raw


class _Sha256Embedder:
    """Fake embedder returning sha256-derived vectors indexed by position."""

    def __init__(self, dim: int = 64) -> None:
        self._dim = dim

    def embed(self, texts: Sequence[str], batch_size: int = 64) -> np.ndarray:
        out = np.stack(
            [_sha256_embedding(i, self._dim) for i in range(len(texts))],
            axis=0,
        )
        return out.astype(np.float32)


class _ConstantEmbedder:
    """Fake embedder that returns the same vector for all inputs."""

    def __init__(self, dim: int = 64) -> None:
        self._dim = dim
        self._vec = _sha256_embedding(0, dim)

    def embed(self, texts: Sequence[str], batch_size: int = 64) -> np.ndarray:
        return np.tile(self._vec, (len(texts), 1)).astype(np.float32)


_FAKE_EMBEDDER = _Sha256Embedder(dim=64)
_CONST_EMBEDDER = _ConstantEmbedder(dim=64)


def _make_ks(k: int, embedder=None, **kwargs) -> SemanticKSplit:
    if embedder is None:
        embedder = _FAKE_EMBEDDER
    # Use random_state from kwargs if provided, otherwise default to 42
    kwargs.setdefault("random_state", 42)
    return SemanticKSplit(k=k, embedding_model=embedder, **kwargs)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DETAILED_REQUIRED_KEYS = {"cluster_id", "representative", "items", "size", "confidence"}


# ===========================================================================
# Invalid return_format raises ValueError with exact wording (Req 2.2, 2.3)
# ===========================================================================


class TestInvalidReturnFormat:
    """Invalid return_format must raise ValueError with the exact message. (Req 2.2, 2.3)"""

    def test_invalid_format_raises_value_error(self):
        """split(..., return_format='wrong') must raise ValueError."""
        ks = _make_ks(k=2)
        with pytest.raises(ValueError):
            ks.split(["text one", "text two", "text three"], return_format="wrong")

    def test_invalid_format_exact_wording(self):
        """The error message must be exactly the specified string (Req 2.2/2.3)."""
        ks = _make_ks(k=2)
        with pytest.raises(
            ValueError,
            match="return_format must be either 'simple' or 'detailed'",
        ):
            ks.split(["a", "b", "c"], return_format="list")

    def test_invalid_format_none_raises(self):
        """return_format=None must raise ValueError."""
        ks = _make_ks(k=2)
        with pytest.raises(ValueError):
            ks.split(["text a", "text b", "text c"], return_format=None)  # type: ignore[arg-type]

    def test_invalid_format_uppercase_raises(self):
        """return_format='Simple' (wrong case) must raise ValueError."""
        ks = _make_ks(k=2)
        with pytest.raises(ValueError):
            ks.split(["text a", "text b", "text c"], return_format="Simple")

    def test_invalid_format_empty_string_raises(self):
        """return_format='' must raise ValueError."""
        ks = _make_ks(k=2)
        with pytest.raises(ValueError):
            ks.split(["text a", "text b", "text c"], return_format="")


# ===========================================================================
# Simple return_format returns List[List[str]] of length k
# ===========================================================================


class TestSimpleReturnFormat:
    """split(..., return_format='simple') returns List[List[str]] of length k. (Req 9.1)"""

    def test_simple_returns_list_of_lists(self):
        """Result must be a list of lists of strings."""
        texts = [f"sentence about topic {i}" for i in range(6)]
        ks = _make_ks(k=2)
        result = ks.split(texts, return_format="simple")
        assert isinstance(result, list)
        for bucket in result:
            assert isinstance(bucket, list)
            for item in bucket:
                assert isinstance(item, str)

    def test_simple_length_equals_k(self):
        """The outer list length must equal k. (Req 9.1)"""
        k = 3
        texts = [f"text item {i}" for i in range(9)]
        ks = _make_ks(k=k)
        result = ks.split(texts, return_format="simple")
        assert len(result) == k

    def test_simple_length_equals_k_for_k2(self):
        """Length == k=2."""
        k = 2
        texts = [f"doc {i}" for i in range(6)]
        ks = _make_ks(k=k)
        result = ks.split(texts, return_format="simple")
        assert len(result) == k

    def test_simple_length_equals_k_for_k4(self):
        """Length == k=4."""
        k = 4
        texts = [f"entry {i}" for i in range(12)]
        ks = _make_ks(k=k)
        result = ks.split(texts, return_format="simple")
        assert len(result) == k

    def test_simple_all_items_are_original_strings(self):
        """Every string in the result must appear in the original texts list."""
        texts = [f"unique phrase {i}" for i in range(8)]
        ks = _make_ks(k=2)
        result = ks.split(texts, return_format="simple")
        flat = [item for bucket in result for item in bucket]
        for item in flat:
            assert item in texts, f"Item {item!r} not in original texts"

    def test_simple_no_bucket_is_empty(self):
        """Each bucket must contain at least one item (Req 7.3)."""
        k = 3
        texts = [f"item {i}" for i in range(9)]
        ks = _make_ks(k=k)
        result = ks.split(texts, return_format="simple")
        for c, bucket in enumerate(result):
            assert len(bucket) >= 1, f"Bucket {c} is empty"

    def test_simple_total_count_equals_valid_inputs(self):
        """The total number of items across all buckets must equal the valid input count."""
        texts = ["apple fruit", None, "banana fruit", "car vehicle", None, "truck vehicle"]
        k = 2
        ks = _make_ks(k=k)
        result = ks.split(texts, return_format="simple")
        valid_count = sum(1 for t in texts if t is not None)
        total = sum(len(b) for b in result)
        assert total == valid_count

    def test_simple_default_format_is_simple(self):
        """Default return_format is 'simple' (no argument needed)."""
        texts = [f"word {i}" for i in range(6)]
        ks = _make_ks(k=2)
        result = ks.split(texts)  # no return_format argument
        assert isinstance(result, list)
        assert len(result) == 2
        for bucket in result:
            assert isinstance(bucket, list)


# ===========================================================================
# Detailed return_format returns list of length k with required keys
# ===========================================================================


class TestDetailedReturnFormat:
    """split(..., return_format='detailed') returns list of dicts of length k. (Req 9.2)"""

    def test_detailed_returns_list(self):
        """Result must be a list."""
        texts = [f"content {i}" for i in range(6)]
        ks = _make_ks(k=2)
        result = ks.split(texts, return_format="detailed")
        assert isinstance(result, list)

    def test_detailed_length_equals_k(self):
        """The list length must equal k. (Req 9.2)"""
        k = 3
        texts = [f"sample text {i}" for i in range(9)]
        ks = _make_ks(k=k)
        result = ks.split(texts, return_format="detailed")
        assert len(result) == k

    def test_detailed_elements_are_dicts(self):
        """Each element must be a dict (or TypedDict)."""
        texts = [f"item {i}" for i in range(6)]
        ks = _make_ks(k=2)
        result = ks.split(texts, return_format="detailed")
        for elem in result:
            assert isinstance(elem, dict), f"Expected dict, got {type(elem)}"

    def test_detailed_has_all_required_keys(self):
        """Each dict must have cluster_id, representative, items, size, confidence. (Req 9.2)"""
        texts = [f"document {i}" for i in range(8)]
        ks = _make_ks(k=2)
        result = ks.split(texts, return_format="detailed")
        for c, elem in enumerate(result):
            missing = _DETAILED_REQUIRED_KEYS - elem.keys()
            assert not missing, f"Cluster {c} missing keys: {missing}"

    def test_detailed_items_field_is_list_of_strings(self):
        """The 'items' field must be a list of strings."""
        texts = [f"phrase {i}" for i in range(6)]
        ks = _make_ks(k=2)
        result = ks.split(texts, return_format="detailed")
        for elem in result:
            assert isinstance(elem["items"], list)
            for item in elem["items"]:
                assert isinstance(item, str)

    def test_detailed_representative_is_string(self):
        """The 'representative' field must be a string."""
        texts = [f"text {i}" for i in range(6)]
        ks = _make_ks(k=2)
        result = ks.split(texts, return_format="detailed")
        for elem in result:
            assert isinstance(elem["representative"], str)

    def test_detailed_size_equals_items_length(self):
        """The 'size' field must equal len(items)."""
        texts = [f"row {i}" for i in range(8)]
        ks = _make_ks(k=3)
        result = ks.split(texts, return_format="detailed")
        for elem in result:
            assert elem["size"] == len(elem["items"])

    def test_detailed_confidence_is_float_in_range(self):
        """The 'confidence' field must be a float in [0, 1]."""
        texts = [f"entry {i}" for i in range(6)]
        ks = _make_ks(k=2)
        result = ks.split(texts, return_format="detailed")
        for elem in result:
            conf = elem["confidence"]
            assert isinstance(conf, float), f"confidence not float: {conf!r}"
            assert 0.0 <= conf <= 1.0, f"confidence out of [0,1]: {conf}"

    def test_detailed_no_bucket_is_empty(self):
        """Each items list must contain at least one element (Req 7.3)."""
        texts = [f"item {i}" for i in range(9)]
        ks = _make_ks(k=3)
        result = ks.split(texts, return_format="detailed")
        for c, elem in enumerate(result):
            assert len(elem["items"]) >= 1, f"Cluster {c} has empty items"


# ===========================================================================
# Cluster_id at position c equals c
# ===========================================================================


class TestDetailedClusterId:
    """cluster_id at position c must equal c. (Req 9.3)"""

    def test_cluster_id_equals_position_k2(self):
        """For k=2: cluster_id[0]==0, cluster_id[1]==1."""
        texts = [f"text {i}" for i in range(6)]
        ks = _make_ks(k=2)
        result = ks.split(texts, return_format="detailed")
        for c, elem in enumerate(result):
            assert elem["cluster_id"] == c, (
                f"Position {c}: expected cluster_id={c}, got {elem['cluster_id']}"
            )

    def test_cluster_id_equals_position_k3(self):
        """For k=3: cluster_id[c]==c for each c in {0,1,2}."""
        texts = [f"phrase {i}" for i in range(9)]
        ks = _make_ks(k=3)
        result = ks.split(texts, return_format="detailed")
        for c, elem in enumerate(result):
            assert elem["cluster_id"] == c

    def test_cluster_id_equals_position_k4(self):
        """For k=4: cluster_id[c]==c for each c in {0,1,2,3}."""
        texts = [f"entry {i}" for i in range(12)]
        ks = _make_ks(k=4)
        result = ks.split(texts, return_format="detailed")
        for c, elem in enumerate(result):
            assert elem["cluster_id"] == c

    def test_cluster_ids_are_ints(self):
        """cluster_id values must be Python ints."""
        texts = [f"sample {i}" for i in range(6)]
        ks = _make_ks(k=2)
        result = ks.split(texts, return_format="detailed")
        for elem in result:
            assert isinstance(elem["cluster_id"], int)

    def test_cluster_id_matches_simple_format_ordering(self):
        """The cluster at position c in detailed must contain the same texts
        as bucket c in simple format (cluster_id == c confirms identity)."""
        texts = [f"item {i}" for i in range(8)]
        ks = _make_ks(k=3)
        simple = ks.split(texts, return_format="simple")
        detailed = ks.split(texts, return_format="detailed")
        for c in range(3):
            detail_items = set(detailed[c]["items"])
            simple_items = set(simple[c])
            assert detail_items == simple_items, (
                f"Cluster {c}: detailed items {detail_items} != simple items {simple_items}"
            )


# ===========================================================================
# Items preserve original input order
# ===========================================================================


class TestItemsOriginalInputOrder:
    """items at position c must contain original strings in original input order. (Req 9.4)"""

    def test_items_preserve_original_order_in_each_bucket(self):
        """Items within each bucket must be in the same relative order as in texts."""
        k = 2
        texts = [f"sentence {i}" for i in range(8)]
        ks = _make_ks(k=k)
        labels = ks.split_labels(texts)
        result = ks.split(texts, return_format="simple")

        for c in range(k):
            # Expected: original texts assigned to cluster c, in original order
            expected = [texts[i] for i in range(len(texts)) if labels[i] == c]
            assert result[c] == expected, (
                f"Cluster {c}: expected {expected}, got {result[c]}"
            )

    def test_items_order_with_mixed_none(self):
        """None rows are excluded; valid rows maintain original order."""
        k = 2
        texts = ["apple fruit", None, "banana fruit", None, "carrot veg", "daikon veg"]
        ks = _make_ks(k=k)
        labels = ks.split_labels(texts)
        result = ks.split(texts, return_format="simple")

        for c in range(k):
            expected = [texts[i] for i in range(len(texts))
                        if texts[i] is not None and labels[i] == c]
            assert result[c] == expected, (
                f"Cluster {c}: expected {expected}, got {result[c]}"
            )

    def test_detailed_items_preserve_original_order(self):
        """Detailed format items must also preserve original input order."""
        k = 2
        texts = [f"doc {i}" for i in range(8)]
        ks = _make_ks(k=k)
        labels = ks.split_labels(texts)
        result = ks.split(texts, return_format="detailed")

        for elem in result:
            c = elem["cluster_id"]
            expected = [texts[i] for i in range(len(texts)) if labels[i] == c]
            assert elem["items"] == expected, (
                f"Cluster {c}: expected {expected}, got {elem['items']}"
            )

    def test_items_are_original_strings_not_preprocessed(self):
        """The items field must contain original (un-preprocessed) input strings."""
        # Use uppercase text to verify original case is preserved
        texts = [
            "Hello World First",
            "Python Programming Second",
            "Machine Learning Third",
            "Data Science Fourth",
            "Natural Language Fifth",
            "Deep Learning Sixth",
        ]
        ks = _make_ks(k=2)
        result = ks.split(texts, return_format="simple")
        flat = [item for bucket in result for item in bucket]
        # Items must be exactly the original strings (not lowercased)
        for item in flat:
            assert item in texts, f"Got preprocessed/modified item: {item!r}"

    def test_simple_and_detailed_items_match_split_labels(self):
        """items in both formats must exactly match what split_labels assigns."""
        k = 3
        texts = [f"text number {i}" for i in range(9)]
        ks = _make_ks(k=k)
        labels = ks.split_labels(texts)
        simple = ks.split(texts, return_format="simple")
        detailed = ks.split(texts, return_format="detailed")

        for c in range(k):
            expected = [texts[i] for i in range(len(texts)) if labels[i] == c]
            assert set(simple[c]) == set(expected)
            assert set(detailed[c]["items"]) == set(expected)


# ===========================================================================
# No valid rows → return [] without raising
# ===========================================================================


class TestNoValidRowsReturnsEmpty:
    """When no valid rows survive preprocessing, split must return []. (Req 9.5)"""

    def test_all_none_returns_empty_list(self):
        """split([None, None, None]) must return []."""
        ks = _make_ks(k=2)
        result = ks.split([None, None, None])
        assert result == []

    def test_all_none_returns_empty_list_detailed(self):
        """split([None, None], return_format='detailed') must return []."""
        ks = _make_ks(k=2)
        result = ks.split([None, None], return_format="detailed")
        assert result == []

    def test_all_nan_returns_empty_list(self):
        """split([nan, nan]) must return []."""
        ks = _make_ks(k=2)
        result = ks.split([float("nan"), float("nan")])
        assert result == []

    def test_all_empty_strings_returns_empty_list(self):
        """split(['', '  ', '']) must return []."""
        ks = _make_ks(k=2)
        result = ks.split(["", "  ", ""])
        assert result == []

    def test_mixed_invalid_returns_empty_list(self):
        """A mix of None/nan/empty with no valid strings returns []."""
        ks = _make_ks(k=2)
        result = ks.split([None, float("nan"), "", "   ", None])
        assert result == []

    def test_empty_texts_returns_empty_list(self):
        """split([]) must return []."""
        ks = _make_ks(k=2)
        result = ks.split([])
        assert result == []

    def test_no_valid_rows_does_not_raise(self):
        """split with all-invalid input must not raise any exception."""
        ks = _make_ks(k=2)
        try:
            result = ks.split([None, None, float("nan"), ""])
            assert result == []
        except Exception as exc:
            pytest.fail(f"Expected no exception, got {exc!r}")

    def test_no_valid_rows_detailed_does_not_raise(self):
        """split(..., return_format='detailed') with all-invalid input must not raise."""
        ks = _make_ks(k=2)
        try:
            result = ks.split([None, None], return_format="detailed")
            assert result == []
        except Exception as exc:
            pytest.fail(f"Expected no exception, got {exc!r}")


# ===========================================================================
# Integration: simple and detailed consistency
# ===========================================================================


class TestSimpleDetailedConsistency:
    """simple and detailed formats must be consistent with each other and split_labels."""

    def test_simple_detailed_same_cluster_count(self):
        """Both formats must return a list of the same length k."""
        texts = [f"item {i}" for i in range(10)]
        k = 3
        ks = _make_ks(k=k)
        simple = ks.split(texts, return_format="simple")
        detailed = ks.split(texts, return_format="detailed")
        assert len(simple) == k
        assert len(detailed) == k

    def test_simple_detailed_same_items_per_cluster(self):
        """Items in each cluster must match between simple and detailed formats."""
        texts = [f"content {i}" for i in range(8)]
        k = 2
        ks = _make_ks(k=k)
        simple = ks.split(texts, return_format="simple")
        detailed = ks.split(texts, return_format="detailed")
        for c in range(k):
            assert set(simple[c]) == set(detailed[c]["items"]), (
                f"Cluster {c} items differ between simple and detailed"
            )

    def test_consistent_with_split_labels(self):
        """split(simple) must exactly match the bucketing from split_labels."""
        texts = [f"unique string {i}" for i in range(8)]
        k = 3
        ks = _make_ks(k=k)
        labels = ks.split_labels(texts)
        result = ks.split(texts, return_format="simple")
        for c in range(k):
            expected_set = {texts[i] for i in range(len(texts)) if labels[i] == c}
            assert set(result[c]) == expected_set, (
                f"Cluster {c}: split() set {set(result[c])} != split_labels() set {expected_set}"
            )

    def test_with_none_values_simple_excludes_none(self):
        """None values must not appear in any bucket in simple format."""
        texts = ["alpha beta", None, "gamma delta", "epsilon zeta", None]
        ks = _make_ks(k=2)
        result = ks.split(texts, return_format="simple")
        for bucket in result:
            assert None not in bucket

    def test_with_none_values_detailed_excludes_none(self):
        """None values must not appear in items of any cluster in detailed format."""
        texts = ["alpha beta", None, "gamma delta", "epsilon zeta", None]
        ks = _make_ks(k=2)
        result = ks.split(texts, return_format="detailed")
        for elem in result:
            assert None not in elem["items"]


# ===========================================================================
# split_with_report — focused tests
#               11.1, 11.2, 11.3, 11.4, 11.5, 11.6, 11.7
# ===========================================================================


class TestSplitWithReportChosenParams:
    """split_with_report populates chosen_params with required keys. (Req 11.1, 11.2)"""

    def test_chosen_params_contains_requested_k(self):
        """K_Run_Report.chosen_params must contain key 'requested_k'. (Req 11.1)"""
        k = 3
        texts = [f"phrase {i}" for i in range(9)]
        ks = _make_ks(k=k)
        _, report = ks.split_with_report(texts)
        assert "requested_k" in report.chosen_params, (
            "chosen_params missing 'requested_k'"
        )

    def test_chosen_params_requested_k_value_equals_k(self):
        """chosen_params['requested_k'] must equal the constructor k. (Req 11.1)"""
        k = 4
        texts = [f"item {i}" for i in range(12)]
        ks = _make_ks(k=k)
        _, report = ks.split_with_report(texts)
        assert report.chosen_params["requested_k"] == k

    def test_chosen_params_contains_algorithm_used(self):
        """chosen_params must contain 'algorithm_used'. (Req 11.2)"""
        texts = [f"text {i}" for i in range(8)]
        ks = _make_ks(k=2)
        _, report = ks.split_with_report(texts)
        assert "algorithm_used" in report.chosen_params, (
            "chosen_params missing 'algorithm_used'"
        )

    def test_chosen_params_algorithm_used_is_valid_literal(self):
        """algorithm_used must be one of the enumerated literal strings. (Req 11.2)"""
        _VALID_ALGORITHMS = {
            "agglomerative-cut-k",
            "bisecting-kmeans",
            "spherical-kmeans",
            "spectral-cosine",
            "constrained-kmeans",
            "balanced-kmeans",
            "minibatch-kmeans-assign",
            "identical-embeddings-tiebreak",
        }
        texts = [f"sentence {i}" for i in range(8)]
        ks = _make_ks(k=2)
        _, report = ks.split_with_report(texts)
        assert report.chosen_params["algorithm_used"] in _VALID_ALGORITHMS, (
            f"algorithm_used {report.chosen_params['algorithm_used']!r} not in valid set"
        )

    def test_chosen_params_contains_pipeline_tier(self):
        """chosen_params must contain 'pipeline_tier'. (Req 11.7)"""
        texts = [f"doc {i}" for i in range(6)]
        ks = _make_ks(k=2)
        _, report = ks.split_with_report(texts)
        assert "pipeline_tier" in report.chosen_params, (
            "chosen_params missing 'pipeline_tier'"
        )

    def test_chosen_params_contains_embedding_dim(self):
        """chosen_params must contain 'embedding_dim'."""
        texts = [f"word {i}" for i in range(6)]
        ks = _make_ks(k=2)
        _, report = ks.split_with_report(texts)
        assert "embedding_dim" in report.chosen_params, (
            "chosen_params missing 'embedding_dim'"
        )

    def test_chosen_params_embedding_dim_is_positive_int(self):
        """embedding_dim must be a positive integer."""
        texts = [f"entry {i}" for i in range(6)]
        ks = _make_ks(k=2)
        _, report = ks.split_with_report(texts)
        dim = report.chosen_params["embedding_dim"]
        assert isinstance(dim, int) and dim > 0, (
            f"embedding_dim must be positive int, got {dim!r}"
        )

    def test_chosen_params_contains_dim_band(self):
        """chosen_params must contain 'dim_band'."""
        texts = [f"text {i}" for i in range(6)]
        ks = _make_ks(k=2)
        _, report = ks.split_with_report(texts)
        assert "dim_band" in report.chosen_params, (
            "chosen_params missing 'dim_band'"
        )

    def test_chosen_params_all_five_required_keys(self):
        """All five required chosen_params keys must be present simultaneously."""
        _REQUIRED = {"requested_k", "algorithm_used", "pipeline_tier", "embedding_dim", "dim_band"}
        texts = [f"content {i}" for i in range(8)]
        ks = _make_ks(k=3)
        _, report = ks.split_with_report(texts)
        missing = _REQUIRED - report.chosen_params.keys()
        assert not missing, f"chosen_params missing keys: {missing}"


class TestSplitWithReportIntrinsicMetrics:
    """split_with_report populates intrinsic_metrics with required keys. (Req 11.3, 11.4)"""

    def test_intrinsic_metrics_contains_silhouette(self):
        """intrinsic_metrics must contain key 'silhouette'. (Req 11.3)"""
        texts = [f"phrase {i}" for i in range(9)]
        ks = _make_ks(k=3)
        _, report = ks.split_with_report(texts)
        assert "silhouette" in report.intrinsic_metrics, (
            "intrinsic_metrics missing 'silhouette'"
        )

    def test_intrinsic_metrics_contains_davies_bouldin(self):
        """intrinsic_metrics must contain key 'davies_bouldin'. (Req 11.3)"""
        texts = [f"item {i}" for i in range(9)]
        ks = _make_ks(k=3)
        _, report = ks.split_with_report(texts)
        assert "davies_bouldin" in report.intrinsic_metrics, (
            "intrinsic_metrics missing 'davies_bouldin'"
        )

    def test_intrinsic_metrics_contains_per_cluster_size(self):
        """intrinsic_metrics must contain key 'per_cluster_size'. (Req 11.3)"""
        texts = [f"sample {i}" for i in range(9)]
        ks = _make_ks(k=3)
        _, report = ks.split_with_report(texts)
        assert "per_cluster_size" in report.intrinsic_metrics, (
            "intrinsic_metrics missing 'per_cluster_size'"
        )

    def test_intrinsic_metrics_contains_per_cluster_cohesion(self):
        """intrinsic_metrics must contain key 'per_cluster_cohesion'. (Req 11.3)"""
        texts = [f"entry {i}" for i in range(9)]
        ks = _make_ks(k=3)
        _, report = ks.split_with_report(texts)
        assert "per_cluster_cohesion" in report.intrinsic_metrics, (
            "intrinsic_metrics missing 'per_cluster_cohesion'"
        )

    def test_all_four_required_intrinsic_keys_present(self):
        """All four required intrinsic_metrics keys must be present. (Req 11.3)"""
        _REQUIRED = {"silhouette", "davies_bouldin", "per_cluster_size", "per_cluster_cohesion"}
        texts = [f"doc {i}" for i in range(8)]
        ks = _make_ks(k=2)
        _, report = ks.split_with_report(texts)
        missing = _REQUIRED - report.intrinsic_metrics.keys()
        assert not missing, f"intrinsic_metrics missing keys: {missing}"

    def test_per_cluster_size_length_equals_k(self):
        """per_cluster_size must be a list of length exactly k. (Req 11.4)"""
        k = 4
        texts = [f"text {i}" for i in range(12)]
        ks = _make_ks(k=k)
        _, report = ks.split_with_report(texts)
        sizes = report.intrinsic_metrics["per_cluster_size"]
        assert isinstance(sizes, list), f"per_cluster_size not a list: {type(sizes)}"
        assert len(sizes) == k, f"Expected length {k}, got {len(sizes)}"

    def test_per_cluster_size_sum_equals_valid_input_count(self):
        """Sum of per_cluster_size must equal the number of valid (non-filtered) rows. (Req 11.4)"""
        texts = [f"row {i}" for i in range(9)]
        k = 3
        ks = _make_ks(k=k)
        _, report = ks.split_with_report(texts)
        sizes = report.intrinsic_metrics["per_cluster_size"]
        assert sum(sizes) == len(texts), (
            f"Sum of sizes {sum(sizes)} != total valid rows {len(texts)}"
        )

    def test_per_cluster_size_each_positive(self):
        """Every entry in per_cluster_size must be >= 1 (no empty cluster). (Req 7.3, 11.4)"""
        k = 3
        texts = [f"item {i}" for i in range(9)]
        ks = _make_ks(k=k)
        _, report = ks.split_with_report(texts)
        sizes = report.intrinsic_metrics["per_cluster_size"]
        for c, size in enumerate(sizes):
            assert size >= 1, f"Cluster {c} has empty size"

    def test_per_cluster_cohesion_length_equals_k(self):
        """per_cluster_cohesion must be a list of length exactly k."""
        k = 3
        texts = [f"word {i}" for i in range(9)]
        ks = _make_ks(k=k)
        _, report = ks.split_with_report(texts)
        cohesions = report.intrinsic_metrics["per_cluster_cohesion"]
        assert isinstance(cohesions, list), (
            f"per_cluster_cohesion not a list: {type(cohesions)}"
        )
        assert len(cohesions) == k, (
            f"Expected length {k}, got {len(cohesions)}"
        )

    def test_per_cluster_cohesion_values_in_range(self):
        """Each cohesion value must be a float in [-1, 1]."""
        k = 2
        texts = [f"entry {i}" for i in range(6)]
        ks = _make_ks(k=k)
        _, report = ks.split_with_report(texts)
        for c, val in enumerate(report.intrinsic_metrics["per_cluster_cohesion"]):
            assert isinstance(val, float), f"Cluster {c} cohesion is not float: {val!r}"
            assert -1.0 <= val <= 1.0, f"Cluster {c} cohesion {val} out of [-1, 1]"

    def test_silhouette_is_numeric(self):
        """silhouette must be a numeric value (float or nan)."""
        import math
        texts = [f"text {i}" for i in range(8)]
        ks = _make_ks(k=2)
        _, report = ks.split_with_report(texts)
        sil = report.intrinsic_metrics["silhouette"]
        assert isinstance(sil, float), f"silhouette is not float: {type(sil)}"

    def test_davies_bouldin_is_numeric(self):
        """davies_bouldin must be a numeric value (float >= 0 or inf)."""
        texts = [f"text {i}" for i in range(8)]
        ks = _make_ks(k=2)
        _, report = ks.split_with_report(texts)
        db = report.intrinsic_metrics["davies_bouldin"]
        assert isinstance(db, float), f"davies_bouldin is not float: {type(db)}"
        assert db >= 0.0, f"davies_bouldin must be >= 0, got {db}"


class TestSplitWithReportReportLevelFields:
    """split_with_report populates top-level report fields correctly.
    (Req 11.5, 11.6, 11.7)"""

    def test_n_clusters_equals_k(self):
        """report.n_clusters must equal the constructor k. (Req 11.5)"""
        k = 3
        texts = [f"sentence {i}" for i in range(9)]
        ks = _make_ks(k=k)
        _, report = ks.split_with_report(texts)
        assert report.n_clusters == k, (
            f"Expected n_clusters={k}, got {report.n_clusters}"
        )

    def test_n_clusters_equals_k_for_k2(self):
        """n_clusters == k=2. (Req 11.5)"""
        k = 2
        texts = [f"item {i}" for i in range(6)]
        ks = _make_ks(k=k)
        _, report = ks.split_with_report(texts)
        assert report.n_clusters == k

    def test_n_clusters_equals_k_for_k4(self):
        """n_clusters == k=4. (Req 11.5)"""
        k = 4
        texts = [f"entry {i}" for i in range(12)]
        ks = _make_ks(k=k)
        _, report = ks.split_with_report(texts)
        assert report.n_clusters == k

    def test_n_noise_equals_filtered_row_count(self):
        """report.n_noise must equal (labels == -1).sum(). (Req 11.6)"""
        texts = ["apple fruit", None, "banana split", None, "cherry pie", "date cake"]
        k = 2
        ks = _make_ks(k=k)
        labels, report = ks.split_with_report(texts)
        expected_noise = int((labels == -1).sum())
        assert report.n_noise == expected_noise, (
            f"Expected n_noise={expected_noise}, got {report.n_noise}"
        )

    def test_n_noise_zero_when_no_filtered_rows(self):
        """n_noise == 0 when all rows are valid. (Req 11.6)"""
        texts = [f"valid text {i}" for i in range(6)]
        k = 2
        ks = _make_ks(k=k)
        labels, report = ks.split_with_report(texts)
        assert report.n_noise == 0
        assert int((labels == -1).sum()) == 0

    def test_n_noise_counts_none_rows(self):
        """n_noise must count None rows as noise. (Req 11.6)"""
        texts = [None, "valid one", None, "valid two", "valid three", "valid four"]
        k = 2
        ks = _make_ks(k=k)
        labels, report = ks.split_with_report(texts)
        assert report.n_noise == 2  # two None entries
        assert int((labels == -1).sum()) == 2

    def test_n_noise_consistent_with_labels(self):
        """n_noise must always equal (labels == -1).sum() for mixed inputs."""
        texts = ["text a", None, "text b", float("nan"), "", "text c"]
        k = 2
        ks = _make_ks(k=k)
        labels, report = ks.split_with_report(texts)
        assert report.n_noise == int((labels == -1).sum())

    def test_pipeline_tier_is_valid_string(self):
        """report.pipeline_tier must be one of 'tiny', 'small', 'medium', 'large'. (Req 11.7)"""
        _VALID_TIERS = {"tiny", "small", "medium", "large"}
        texts = [f"phrase {i}" for i in range(8)]
        ks = _make_ks(k=2)
        _, report = ks.split_with_report(texts)
        assert report.pipeline_tier in _VALID_TIERS, (
            f"pipeline_tier {report.pipeline_tier!r} not in {_VALID_TIERS}"
        )

    def test_pipeline_tier_matches_chosen_params_tier(self):
        """report.pipeline_tier must match chosen_params['pipeline_tier']."""
        texts = [f"text {i}" for i in range(8)]
        ks = _make_ks(k=2)
        _, report = ks.split_with_report(texts)
        assert report.pipeline_tier == report.chosen_params.get("pipeline_tier"), (
            "pipeline_tier mismatch between report field and chosen_params"
        )

    def test_random_state_equals_constructor_value(self):
        """report.random_state must equal the random_state passed to the constructor."""
        seed = 99
        texts = [f"sentence {i}" for i in range(8)]
        ks = _make_ks(k=2, random_state=seed)
        _, report = ks.split_with_report(texts)
        assert report.random_state == seed, (
            f"Expected random_state={seed}, got {report.random_state}"
        )

    def test_random_state_default_is_42(self):
        """Default random_state=42 must appear in the report."""
        texts = [f"text {i}" for i in range(6)]
        ks = _make_ks(k=2)  # default random_state=42
        _, report = ks.split_with_report(texts)
        assert report.random_state == 42

    def test_random_state_zero_is_propagated(self):
        """random_state=0 must be propagated to the report."""
        texts = [f"word {i}" for i in range(6)]
        ks = _make_ks(k=2, random_state=0)
        _, report = ks.split_with_report(texts)
        assert report.random_state == 0

    def test_warnings_is_a_list(self):
        """report.warnings must be a list. (Req 11.2)"""
        texts = [f"item {i}" for i in range(6)]
        ks = _make_ks(k=2)
        _, report = ks.split_with_report(texts)
        assert isinstance(report.warnings, list), (
            f"warnings is not a list: {type(report.warnings)}"
        )

    def test_warnings_contains_only_strings(self):
        """Every element in report.warnings must be a string."""
        texts = [f"text {i}" for i in range(6)]
        ks = _make_ks(k=2)
        _, report = ks.split_with_report(texts)
        for w in report.warnings:
            assert isinstance(w, str), f"Non-string in warnings: {w!r}"

    def test_warnings_no_noise_warning_for_normal_run(self):
        """No algorithmic noise warnings should appear for a normal valid run."""
        texts = [f"entry {i}" for i in range(8)]
        k = 2
        ks = _make_ks(k=k)
        _, report = ks.split_with_report(texts)
        # Warnings list may be empty — that's fine
        assert isinstance(report.warnings, list)

    def test_identical_embeddings_warning_in_warnings_list(self):
        """identical-embeddings-tiebreak must be appended to warnings when triggered. (Req 12.3)"""
        k = 2
        # Use constant embedder to trigger the identical-embeddings path
        ks = _make_ks(k=k, embedder=_CONST_EMBEDDER)
        texts = [f"same text {i}" for i in range(6)]
        _, report = ks.split_with_report(texts)
        assert "identical-embeddings-tiebreak" in report.warnings, (
            f"Expected 'identical-embeddings-tiebreak' in warnings, got {report.warnings}"
        )
        assert report.chosen_params.get("algorithm_used") == "identical-embeddings-tiebreak"


class TestSplitWithReportLabelsContract:
    """split_with_report returns (labels, report) matching the split_labels contract."""

    def test_split_with_report_returns_tuple_of_two(self):
        """split_with_report must return a 2-tuple (labels, report)."""
        texts = [f"text {i}" for i in range(6)]
        ks = _make_ks(k=2)
        result = ks.split_with_report(texts)
        assert isinstance(result, tuple) and len(result) == 2

    def test_labels_dtype_int32(self):
        """The returned labels array must have dtype == np.int32."""
        texts = [f"phrase {i}" for i in range(8)]
        ks = _make_ks(k=3)
        labels, _ = ks.split_with_report(texts)
        assert labels.dtype == np.int32, f"Expected int32, got {labels.dtype}"

    def test_labels_shape_equals_input_length(self):
        """The labels array shape must equal (len(texts),)."""
        texts = [f"entry {i}" for i in range(10)]
        k = 3
        ks = _make_ks(k=k)
        labels, _ = ks.split_with_report(texts)
        assert labels.shape == (len(texts),), (
            f"Expected shape ({len(texts)},), got {labels.shape}"
        )

    def test_labels_values_cover_all_k_clusters(self):
        """Valid rows must cover exactly {0, …, k-1} in the label array."""
        k = 3
        texts = [f"word {i}" for i in range(9)]
        ks = _make_ks(k=k)
        labels, _ = ks.split_with_report(texts)
        valid_labels = set(labels[labels >= 0].tolist())
        assert valid_labels == set(range(k)), (
            f"Expected labels to cover {set(range(k))}, got {valid_labels}"
        )

    def test_labels_match_split_labels(self):
        """Labels from split_with_report must match split_labels for the same input."""
        texts = [f"text {i}" for i in range(8)]
        k = 3
        ks = _make_ks(k=k)
        labels_from_report, _ = ks.split_with_report(texts)
        labels_direct = ks.split_labels(texts)
        assert np.array_equal(labels_from_report, labels_direct), (
            "Labels from split_with_report differ from split_labels"
        )

    def test_report_is_clustering_report_instance(self):
        """The second element of the tuple must be a ClusteringReport instance."""
        from semantic_clusterer.report import ClusteringReport
        texts = [f"item {i}" for i in range(6)]
        ks = _make_ks(k=2)
        _, report = ks.split_with_report(texts)
        assert isinstance(report, ClusteringReport), (
            f"Expected ClusteringReport, got {type(report)}"
        )

    def test_n_clusters_plus_n_noise_leq_n_input(self):
        """n_clustered + n_noise must equal n_input_texts."""
        texts = ["alpha", None, "beta", float("nan"), "gamma", "delta"]
        k = 2
        ks = _make_ks(k=k)
        _, report = ks.split_with_report(texts)
        assert report.n_clustered + report.n_noise == report.n_input_texts, (
            f"n_clustered({report.n_clustered}) + n_noise({report.n_noise}) "
            f"!= n_input_texts({report.n_input_texts})"
        )

    def test_empty_texts_returns_empty_labels(self):
        """split_with_report([]) must return an empty labels array without raising."""
        ks = _make_ks(k=2)
        labels, report = ks.split_with_report([])
        assert labels.shape == (0,)
        assert labels.dtype == np.int32

    @pytest.mark.parametrize("k", [2, 3, 4, 5])
    def test_n_clusters_equals_k_parametrized(self, k):
        """n_clusters == k for various k values. (Req 11.5)"""
        n = k * 3  # ensure enough rows
        texts = [f"sentence {i}" for i in range(n)]
        ks = _make_ks(k=k)
        _, report = ks.split_with_report(texts)
        assert report.n_clusters == k


class TestSplitWithReportFullIntegrity:
    """Comprehensive integrity check for a single split_with_report call."""

    def test_full_report_integrity(self):
        """All required fields populate correctly in a single call."""
        import warnings as _warnings
        k = 3
        seed = 7
        texts = [f"unique document content about subject {i}" for i in range(9)]
        ks = _make_ks(k=k, random_state=seed)

        # suppress the dim-band warning for fake embedder (dim=64 is below lower bound)
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore", UserWarning)
            labels, report = ks.split_with_report(texts)

        # --- chosen_params ---
        cp = report.chosen_params
        assert cp["requested_k"] == k
        assert cp["algorithm_used"] in {
            "agglomerative-cut-k",
            "bisecting-kmeans",
            "spectral-cosine",
            "balanced-kmeans",
            "minibatch-kmeans-assign",
            "constrained-kmeans",
            "identical-embeddings-tiebreak",
        }
        assert cp["pipeline_tier"] in {"tiny", "small", "medium", "large"}
        assert isinstance(cp["embedding_dim"], int) and cp["embedding_dim"] > 0
        assert "dim_band" in cp

        # --- intrinsic_metrics ---
        im = report.intrinsic_metrics
        assert "silhouette" in im
        assert "davies_bouldin" in im
        assert isinstance(im["per_cluster_size"], list) and len(im["per_cluster_size"]) == k
        assert isinstance(im["per_cluster_cohesion"], list) and len(im["per_cluster_cohesion"]) == k
        assert sum(im["per_cluster_size"]) == len(texts)
        for sz in im["per_cluster_size"]:
            assert sz >= 1

        # --- report level ---
        assert report.n_clusters == k, f"n_clusters {report.n_clusters} != k {k}"
        assert report.n_noise == int((labels == -1).sum())
        assert report.pipeline_tier in {"tiny", "small", "medium", "large"}
        assert report.random_state == seed
        assert isinstance(report.warnings, list)
