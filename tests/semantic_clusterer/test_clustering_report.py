"""Unit tests for ClusteringReport field-shape and JSON round-trip.

Covers:
  1. Every required field is present and typed correctly on a successful run
  2. to_dict() produces JSON-native nested types
  3. json.loads(json.dumps(report.to_dict())) == report.to_dict()
  4. numpy scalars coerced (np.float32, np.int64, np.bool_)
  5. NaN/Inf coerced to None
  6. Non-string element to cluster_with_report raises TypeError

"""

import json
import math
from typing import List

import numpy as np
import pytest

from semantic_clusterer import SemanticClusterer
from semantic_clusterer.report import ClusteringReport, _coerce_json


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _make_stub_embedder(n_texts: int, dim: int = 32, seed: int = 42):
    """Return a callable embedder that produces deterministic synthetic embeddings."""
    rng = np.random.default_rng(seed)
    # Pre-generate a pool large enough for any call
    pool = rng.standard_normal((max(n_texts, 200), dim)).astype(np.float32)
    # Normalise rows
    norms = np.linalg.norm(pool, axis=1, keepdims=True)
    pool = pool / np.where(norms == 0, 1.0, norms)

    class _StubEmbedder:
        def embed(self, texts: List[str]) -> np.ndarray:
            n = len(texts)
            return pool[:n].copy()

    return _StubEmbedder()


# Synthetic texts with clear cluster structure (3 groups × 4 texts + 1 outlier)
_TEXTS = [
    # Group A – revenue
    "monthly revenue report",
    "revenue per month",
    "total sales monthly",
    "monthly income report",
    # Group B – users
    "list all users",
    "show user accounts",
    "display user list",
    "get all users",
    # Group C – weather
    "weather forecast today",
    "check weather conditions",
    "what is the weather",
    "weather report now",
    # Outlier
    "random unrelated text",
]


@pytest.fixture(scope="module")
def clusterer_and_report():
    """Run cluster_with_report once and cache the result for the whole module."""
    embedder = _make_stub_embedder(len(_TEXTS), dim=32)
    clusterer = SemanticClusterer(embedding_model=embedder, random_state=42)
    labels, report = clusterer.cluster_with_report(_TEXTS)
    return clusterer, labels, report


# ---------------------------------------------------------------------------
# 1. Every required field is present and typed correctly
# ---------------------------------------------------------------------------

class TestClusteringReportFieldShape:
    """Field presence and type constraints."""

    def test_n_input_texts_is_nonneg_int(self, clusterer_and_report):
        _, _, report = clusterer_and_report
        assert isinstance(report.n_input_texts, int)
        assert report.n_input_texts >= 0

    def test_n_clustered_is_nonneg_int(self, clusterer_and_report):
        _, _, report = clusterer_and_report
        assert isinstance(report.n_clustered, int)
        assert report.n_clustered >= 0

    def test_n_noise_is_nonneg_int(self, clusterer_and_report):
        _, _, report = clusterer_and_report
        assert isinstance(report.n_noise, int)
        assert report.n_noise >= 0

    def test_n_clusters_is_nonneg_int(self, clusterer_and_report):
        _, _, report = clusterer_and_report
        assert isinstance(report.n_clusters, int)
        assert report.n_clusters >= 0

    def test_counts_sum_to_n_input_texts(self, clusterer_and_report):
        """n_clustered + n_noise == n_input_texts."""
        _, _, report = clusterer_and_report
        assert report.n_clustered + report.n_noise == report.n_input_texts

    def test_n_input_texts_matches_input_length(self, clusterer_and_report):
        _, _, report = clusterer_and_report
        assert report.n_input_texts == len(_TEXTS)

    def test_pipeline_tier_is_valid_literal(self, clusterer_and_report):
        _, _, report = clusterer_and_report
        assert report.pipeline_tier in {"tiny", "small", "medium", "large"}

    def test_embedding_dim_is_positive_int(self, clusterer_and_report):
        _, _, report = clusterer_and_report
        assert isinstance(report.embedding_dim, int)
        assert report.embedding_dim >= 1

    def test_dim_band_is_valid_literal(self, clusterer_and_report):
        _, _, report = clusterer_and_report
        assert report.dim_band in {"low", "mid", "high", "xhigh"}

    def test_dataset_profile_is_dict(self, clusterer_and_report):
        _, _, report = clusterer_and_report
        assert isinstance(report.dataset_profile, dict)

    def test_chosen_params_is_dict(self, clusterer_and_report):
        _, _, report = clusterer_and_report
        assert isinstance(report.chosen_params, dict)

    def test_intrinsic_metrics_is_dict(self, clusterer_and_report):
        _, _, report = clusterer_and_report
        assert isinstance(report.intrinsic_metrics, dict)

    def test_phase_timings_is_dict_of_nonneg_floats(self, clusterer_and_report):
        _, _, report = clusterer_and_report
        assert isinstance(report.phase_timings, dict)
        for key, val in report.phase_timings.items():
            assert isinstance(key, str)
            assert isinstance(val, float)
            assert val >= 0.0

    def test_warnings_is_list_of_strings(self, clusterer_and_report):
        _, _, report = clusterer_and_report
        assert isinstance(report.warnings, list)
        for w in report.warnings:
            assert isinstance(w, str)

    def test_confidence_level_is_valid_literal(self, clusterer_and_report):
        _, _, report = clusterer_and_report
        assert report.confidence_level in {"high", "low"}

    def test_random_state_is_int(self, clusterer_and_report):
        _, _, report = clusterer_and_report
        assert isinstance(report.random_state, int)
        assert report.random_state == 42

    def test_library_version_is_str(self, clusterer_and_report):
        _, _, report = clusterer_and_report
        assert isinstance(report.library_version, str)
        assert len(report.library_version) > 0

    def test_labels_shape_matches_input(self, clusterer_and_report):
        """Returned label array must be int32 with length == len(texts)."""
        _, labels, _ = clusterer_and_report
        assert labels.shape == (len(_TEXTS),)
        assert labels.dtype == np.int32


# ---------------------------------------------------------------------------
# 2. to_dict() produces JSON-native nested types
# ---------------------------------------------------------------------------

_JSON_NATIVE = (str, int, float, bool, type(None))


def _assert_json_native(obj, path: str = "root"):
    """Recursively assert that obj contains only JSON-native types."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            assert isinstance(k, str), f"{path}: dict key {k!r} is not str"
            _assert_json_native(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            _assert_json_native(v, f"{path}[{i}]")
    else:
        assert isinstance(obj, _JSON_NATIVE), (
            f"{path}: expected JSON-native type, got {type(obj).__name__!r} ({obj!r})"
        )
        # float must be finite (NaN/Inf are not JSON-native)
        if isinstance(obj, float):
            assert math.isfinite(obj), (
                f"{path}: float value {obj!r} is not finite (should be coerced to None)"
            )


class TestToDict:
    """To_dict() produces JSON-native types."""

    def test_to_dict_returns_dict(self, clusterer_and_report):
        _, _, report = clusterer_and_report
        d = report.to_dict()
        assert isinstance(d, dict)

    def test_to_dict_all_values_json_native(self, clusterer_and_report):
        _, _, report = clusterer_and_report
        d = report.to_dict()
        _assert_json_native(d)

    def test_json_dumps_does_not_raise(self, clusterer_and_report):
        """json.dumps(report.to_dict()) must succeed without exception."""
        _, _, report = clusterer_and_report
        d = report.to_dict()
        # Should not raise
        json.dumps(d)

    def test_to_dict_contains_required_keys(self, clusterer_and_report):
        _, _, report = clusterer_and_report
        d = report.to_dict()
        required_keys = {
            "n_input_texts", "n_clustered", "n_noise", "n_clusters",
            "pipeline_tier", "embedding_dim", "dim_band",
            "dataset_profile", "chosen_params", "intrinsic_metrics",
            "phase_timings", "warnings", "confidence_level",
            "random_state", "library_version",
        }
        for key in required_keys:
            assert key in d, f"Missing key in to_dict(): {key!r}"


# ---------------------------------------------------------------------------
# 3. JSON round-trip: json.loads(json.dumps(report.to_dict())) == report.to_dict()
# ---------------------------------------------------------------------------

class TestJsonRoundTrip:
    """JSON round-trip identity."""

    def test_json_round_trip_equals_to_dict(self, clusterer_and_report):
        _, _, report = clusterer_and_report
        d = report.to_dict()
        round_tripped = json.loads(json.dumps(d))
        assert round_tripped == d

    def test_json_round_trip_on_minimal_report(self):
        """Round-trip on a hand-constructed minimal report."""
        report = ClusteringReport(
            n_input_texts=10,
            n_clustered=8,
            n_noise=2,
            n_clusters=3,
            pipeline_tier="small",
            embedding_dim=384,
            dim_band="low",
            dataset_profile={"cluster_tendency": 0.65},
            chosen_params={"pca_target": 32},
            intrinsic_metrics={"score": 0.72},
            phase_timings={"embedding": 0.1, "clustering": 0.5},
            warnings=[],
            confidence_level="high",
            random_state=42,
            library_version="0.1.0",
        )
        d = report.to_dict()
        round_tripped = json.loads(json.dumps(d))
        assert round_tripped == d


# ---------------------------------------------------------------------------
# 4. numpy scalars coerced (np.float32, np.int64, np.bool_)
# ---------------------------------------------------------------------------

class TestNumpyScalarCoercion:
    """Numpy scalars coerced to JSON-native equivalents."""

    def test_np_float32_coerced_to_float(self):
        report = ClusteringReport(
            n_input_texts=5,
            n_clustered=5,
            n_noise=0,
            n_clusters=2,
            pipeline_tier="tiny",
            embedding_dim=384,
            dim_band="low",
            dataset_profile={"score": np.float32(0.75)},
            chosen_params={"ratio": np.float32(0.5)},
            intrinsic_metrics={"score": np.float32(0.8)},
            phase_timings={"embedding": float(np.float32(0.1))},
            warnings=[],
            confidence_level="high",
            random_state=42,
            library_version="0.1.0",
        )
        d = report.to_dict()
        _assert_json_native(d)
        # Specifically check the coerced value is a plain float
        assert type(d["dataset_profile"]["score"]) is float
        assert type(d["chosen_params"]["ratio"]) is float
        assert type(d["intrinsic_metrics"]["score"]) is float

    def test_np_int64_coerced_to_int(self):
        report = ClusteringReport(
            n_input_texts=5,
            n_clustered=5,
            n_noise=0,
            n_clusters=2,
            pipeline_tier="tiny",
            embedding_dim=384,
            dim_band="low",
            dataset_profile={"count": np.int64(42)},
            chosen_params={"k": np.int64(3)},
            intrinsic_metrics={"n": np.int64(10)},
            phase_timings={},
            warnings=[],
            confidence_level="high",
            random_state=42,
            library_version="0.1.0",
        )
        d = report.to_dict()
        _assert_json_native(d)
        assert type(d["dataset_profile"]["count"]) is int
        assert type(d["chosen_params"]["k"]) is int
        assert type(d["intrinsic_metrics"]["n"]) is int

    def test_np_bool_coerced_to_bool(self):
        report = ClusteringReport(
            n_input_texts=5,
            n_clustered=5,
            n_noise=0,
            n_clusters=2,
            pipeline_tier="tiny",
            embedding_dim=384,
            dim_band="low",
            dataset_profile={"flag": np.bool_(True)},
            chosen_params={"enabled": np.bool_(False)},
            intrinsic_metrics={"ok": np.bool_(True)},
            phase_timings={},
            warnings=[],
            confidence_level="high",
            random_state=42,
            library_version="0.1.0",
        )
        d = report.to_dict()
        _assert_json_native(d)
        assert type(d["dataset_profile"]["flag"]) is bool
        assert d["dataset_profile"]["flag"] is True
        assert type(d["chosen_params"]["enabled"]) is bool
        assert d["chosen_params"]["enabled"] is False

    def test_mixed_numpy_scalars_in_nested_dict(self):
        """A dict with mixed numpy scalar types should all be coerced."""
        report = ClusteringReport(
            n_input_texts=5,
            n_clustered=5,
            n_noise=0,
            n_clusters=2,
            pipeline_tier="tiny",
            embedding_dim=384,
            dim_band="low",
            dataset_profile={
                "f32": np.float32(1.5),
                "i64": np.int64(7),
                "b_": np.bool_(True),
                "nested": {"deep": np.float32(0.25)},
            },
            chosen_params={},
            intrinsic_metrics={},
            phase_timings={},
            warnings=[],
            confidence_level="high",
            random_state=42,
            library_version="0.1.0",
        )
        d = report.to_dict()
        _assert_json_native(d)
        assert type(d["dataset_profile"]["f32"]) is float
        assert type(d["dataset_profile"]["i64"]) is int
        assert type(d["dataset_profile"]["b_"]) is bool
        assert type(d["dataset_profile"]["nested"]["deep"]) is float

    def test_numpy_array_in_dict_coerced_to_list(self):
        """numpy arrays stored in dicts should be coerced to plain lists."""
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        report = ClusteringReport(
            n_input_texts=5,
            n_clustered=5,
            n_noise=0,
            n_clusters=2,
            pipeline_tier="tiny",
            embedding_dim=384,
            dim_band="low",
            dataset_profile={"scores": arr},
            chosen_params={},
            intrinsic_metrics={},
            phase_timings={},
            warnings=[],
            confidence_level="high",
            random_state=42,
            library_version="0.1.0",
        )
        d = report.to_dict()
        _assert_json_native(d)
        assert isinstance(d["dataset_profile"]["scores"], list)
        assert d["dataset_profile"]["scores"] == [1.0, 2.0, 3.0]

    def test_coerce_json_directly_np_float32(self):
        assert type(_coerce_json(np.float32(0.5))) is float
        assert _coerce_json(np.float32(0.5)) == pytest.approx(0.5, abs=1e-5)

    def test_coerce_json_directly_np_int64(self):
        assert type(_coerce_json(np.int64(99))) is int
        assert _coerce_json(np.int64(99)) == 99

    def test_coerce_json_directly_np_bool(self):
        assert type(_coerce_json(np.bool_(True))) is bool
        assert _coerce_json(np.bool_(True)) is True
        assert _coerce_json(np.bool_(False)) is False


# ---------------------------------------------------------------------------
# 5. NaN/Inf coerced to None
# ---------------------------------------------------------------------------

class TestNanInfCoercion:
    """NaN and Inf must be coerced to None."""

    def _report_with_special(self, value):
        return ClusteringReport(
            n_input_texts=5,
            n_clustered=5,
            n_noise=0,
            n_clusters=2,
            pipeline_tier="tiny",
            embedding_dim=384,
            dim_band="low",
            dataset_profile={"special": value},
            chosen_params={},
            intrinsic_metrics={},
            phase_timings={},
            warnings=[],
            confidence_level="high",
            random_state=42,
            library_version="0.1.0",
        )

    def test_plain_float_nan_coerced_to_none(self):
        report = self._report_with_special(float("nan"))
        d = report.to_dict()
        assert d["dataset_profile"]["special"] is None

    def test_plain_float_inf_coerced_to_none(self):
        report = self._report_with_special(float("inf"))
        d = report.to_dict()
        assert d["dataset_profile"]["special"] is None

    def test_plain_float_neg_inf_coerced_to_none(self):
        report = self._report_with_special(float("-inf"))
        d = report.to_dict()
        assert d["dataset_profile"]["special"] is None

    def test_np_float32_nan_coerced_to_none(self):
        report = self._report_with_special(np.float32("nan"))
        d = report.to_dict()
        assert d["dataset_profile"]["special"] is None

    def test_np_float32_inf_coerced_to_none(self):
        report = self._report_with_special(np.float32("inf"))
        d = report.to_dict()
        assert d["dataset_profile"]["special"] is None

    def test_np_float64_nan_coerced_to_none(self):
        report = self._report_with_special(np.float64("nan"))
        d = report.to_dict()
        assert d["dataset_profile"]["special"] is None

    def test_nan_in_nested_list_coerced_to_none(self):
        report = self._report_with_special([1.0, float("nan"), float("inf"), 2.0])
        d = report.to_dict()
        assert d["dataset_profile"]["special"] == [1.0, None, None, 2.0]

    def test_nan_in_nested_dict_coerced_to_none(self):
        report = self._report_with_special({"a": float("nan"), "b": 1.0})
        d = report.to_dict()
        assert d["dataset_profile"]["special"]["a"] is None
        assert d["dataset_profile"]["special"]["b"] == 1.0

    def test_to_dict_with_nan_is_json_serialisable(self):
        """After coercion, json.dumps must not raise even with NaN inputs."""
        report = self._report_with_special(float("nan"))
        d = report.to_dict()
        # Must not raise
        json.dumps(d)

    def test_coerce_json_directly_nan(self):
        assert _coerce_json(float("nan")) is None

    def test_coerce_json_directly_inf(self):
        assert _coerce_json(float("inf")) is None

    def test_coerce_json_directly_neg_inf(self):
        assert _coerce_json(float("-inf")) is None

    def test_coerce_json_directly_np_float32_nan(self):
        assert _coerce_json(np.float32("nan")) is None

    def test_coerce_json_directly_np_float32_inf(self):
        assert _coerce_json(np.float32("inf")) is None


# ---------------------------------------------------------------------------
# 6. Non-string element to cluster_with_report raises TypeError
# ---------------------------------------------------------------------------

class TestClusterWithReportTypeError:
    """Non-string elements must raise TypeError."""

    @pytest.fixture
    def clusterer(self):
        embedder = _make_stub_embedder(20, dim=32)
        return SemanticClusterer(embedding_model=embedder, random_state=42)

    def test_integer_element_raises_type_error(self, clusterer):
        with pytest.raises(TypeError):
            clusterer.cluster_with_report(["valid text", 42, "another text"])

    def test_list_element_raises_type_error(self, clusterer):
        with pytest.raises(TypeError):
            clusterer.cluster_with_report(["valid text", ["nested", "list"]])

    def test_dict_element_raises_type_error(self, clusterer):
        with pytest.raises(TypeError):
            clusterer.cluster_with_report(["valid text", {"key": "value"}])

    def test_bytes_element_raises_type_error(self, clusterer):
        with pytest.raises(TypeError):
            clusterer.cluster_with_report(["valid text", b"bytes"])

    def test_bool_element_raises_type_error(self, clusterer):
        # bool is a subclass of int, not str — should raise
        with pytest.raises(TypeError):
            clusterer.cluster_with_report(["valid text", True])

    def test_none_element_does_not_raise(self, clusterer):
        """None is treated as a missing value (filtered), not a type error."""
        # Should not raise — None is allowed as a missing-value sentinel
        labels, report = clusterer.cluster_with_report(
            ["valid text one", None, "valid text two"] * 4
        )
        assert labels is not None
        assert report is not None

    def test_all_strings_does_not_raise(self, clusterer):
        """A list of pure strings must not raise TypeError."""
        labels, report = clusterer.cluster_with_report(
            ["hello world", "foo bar", "baz qux"] * 4
        )
        assert labels.shape == (12,)
        assert isinstance(report, ClusteringReport)

    def test_type_error_raised_before_partial_report(self, clusterer):
        """TypeError must be raised without producing a partial ClusteringReport."""
        # We verify this by checking the exception is raised cleanly
        with pytest.raises(TypeError) as exc_info:
            clusterer.cluster_with_report(["valid", 999])
        # The error message should mention the offending type
        assert "int" in str(exc_info.value).lower() or "non-string" in str(exc_info.value).lower()
