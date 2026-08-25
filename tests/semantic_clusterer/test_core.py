"""Integration tests for SemanticClusterer."""


import sys
import types
import numpy as np
import pytest

from semantic_clusterer import ClustererConfig, SemanticClusterer, SemanticClustererConfig
from semantic_clusterer.pipeline.small import (
    _recover_small_noise_labels,
    _score_small_clustering,
)
from semantic_clusterer.reduction.umap_utils import adaptive_small_umap_components as _adaptive_small_umap_components
from semantic_clusterer.pipeline import small as small_mod
from semantic_clusterer.utils.similarity import normalize_vectors


class TestSemanticClustererBasic:
    """Basic functionality tests."""

    def test_init_default(self):
        """Test default initialization."""
        clusterer = SemanticClusterer()
        assert clusterer.config is not None
        assert clusterer.verbose is False

    def test_init_with_config(self):
        """Test initialization with config."""
        config = ClustererConfig(batch_size=32)
        clusterer = SemanticClusterer(config=config)
        assert clusterer.config.batch_size == 32

    def test_init_with_dict_config(self):
        """Test initialization with dict config."""
        clusterer = SemanticClusterer(config={"batch_size": 64})
        assert clusterer.config.batch_size == 64

    def test_init_with_verbose(self):
        """Test verbose mode."""
        clusterer = SemanticClusterer(verbose=True)
        assert clusterer.verbose is True

    def test_empty_input(self):
        """Test with empty input."""
        clusterer = SemanticClusterer()
        result = clusterer.cluster([])
        assert result == []

    def test_single_input(self):
        """Test with single text."""
        clusterer = SemanticClusterer()
        result = clusterer.cluster(["hello world"])
        assert len(result) == 1
        assert result[0] == ["hello world"]

    def test_single_input_detailed(self):
        """Test single input with detailed output."""
        clusterer = SemanticClusterer()
        result = clusterer.cluster(["hello world"], return_format="detailed")
        assert len(result) == 1
        assert result[0]["representative"] == "hello world"
        assert result[0]["size"] == 1


class TestSemanticClustererWithMock:
    """Tests using mock embedder."""

    def test_cluster_with_mock_embedder(self, sample_texts, mock_embedder):
        """Test clustering with mock embedder."""
        clusterer = SemanticClusterer(embedding_model=mock_embedder)
        result = clusterer.cluster(sample_texts)

        # Should return list of lists
        assert isinstance(result, list)
        assert all(isinstance(cluster, list) for cluster in result)

        # All texts should be in some cluster
        all_texts = []
        for cluster in result:
            all_texts.extend(cluster)
        assert len(all_texts) == len(sample_texts)

    def test_cluster_detailed_format(self, sample_texts, mock_embedder):
        """Test detailed output format."""
        clusterer = SemanticClusterer(embedding_model=mock_embedder)
        result = clusterer.cluster(sample_texts, return_format="detailed")

        # Should return list of dicts
        assert isinstance(result, list)
        assert all(isinstance(cluster, dict) for cluster in result)

        # Check required fields
        for cluster in result:
            assert "cluster_id" in cluster
            assert "representative" in cluster
            assert "items" in cluster
            assert "size" in cluster
            assert "confidence" in cluster

            assert isinstance(cluster["items"], list)
            assert cluster["size"] == len(cluster["items"])
            assert 0 <= cluster["confidence"] <= 1

    def test_embedder_called(self, sample_texts, mock_embedder):
        """Test that embedder is called."""
        clusterer = SemanticClusterer(embedding_model=mock_embedder)
        clusterer.cluster(sample_texts)

        assert mock_embedder._call_count > 0

    def test_custom_config(self, sample_texts, mock_embedder):
        """Test with custom config options."""
        config = ClustererConfig(
            batch_size=32,
        )
        clusterer = SemanticClusterer(
            embedding_model=mock_embedder,
            config=config,
        )
        result = clusterer.cluster(sample_texts)

        assert isinstance(result, list)


class TestSemanticClustererConfig:
    """Tests for configuration behavior."""

    def test_strategy_auto_tiny(self):
        """Test auto strategy for tiny data (≤150)."""
        config = ClustererConfig()
        assert config.get_strategy_for_size(50) == "tiny"
        assert config.get_strategy_for_size(150) == "tiny"

    def test_strategy_auto_small(self):
        """Test auto strategy for small data (151-5000)."""
        config = ClustererConfig()
        assert config.get_strategy_for_size(151) == "small"
        assert config.get_strategy_for_size(1000) == "small"
        assert config.get_strategy_for_size(5000) == "small"

    def test_strategy_auto_medium(self):
        """Test auto strategy for medium data."""
        config = ClustererConfig()
        assert config.get_strategy_for_size(10000) == "medium"

    def test_strategy_auto_large(self):
        """Test auto strategy for large data."""
        config = ClustererConfig()
        assert config.get_strategy_for_size(100000) == "large"

    def test_reduction_auto_tiny(self):
        """Test auto reduction for tiny strategy (no reduction)."""
        config = ClustererConfig()
        assert config.get_reduction_for_strategy("tiny") is None

    def test_reduction_auto_small(self):
        """Test auto reduction for small strategy."""
        config = ClustererConfig()
        assert config.get_reduction_for_strategy("small") is None

    def test_reduction_auto_medium(self):
        """Test auto reduction for medium strategy."""
        config = ClustererConfig()
        assert config.get_reduction_for_strategy("medium") == "pca"


class TestAdaptiveSmallPipelineHelpers:
    """Tests for adaptive small-pipeline helper behavior."""

    def test_umap_components_scale_with_embedding_dim(self):
        """Higher-dimensional embedders should search higher UMAP dimensions."""
        assert _adaptive_small_umap_components(384) == [8, 10, 12]
        assert _adaptive_small_umap_components(768) == [6, 8, 9]
        assert _adaptive_small_umap_components(1536) == [8, 10, 12]

    def test_composite_score_penalizes_giant_cluster(self):
        """Balanced clusters should beat a single merged giant cluster."""
        rng = np.random.default_rng(42)

        centers = np.eye(3, 12, dtype=np.float32)
        embeddings = []
        labels_good = []
        for cluster_id, center in enumerate(centers):
            cluster_points = center + rng.normal(0.0, 0.02, size=(20, 12))
            embeddings.append(cluster_points)
            labels_good.extend([cluster_id] * len(cluster_points))

        embeddings = normalize_vectors(np.vstack(embeddings).astype(np.float32))
        labels_good = np.array(labels_good, dtype=np.int32)
        labels_bad = np.zeros(len(labels_good), dtype=np.int32)

        good_score = _score_small_clustering(embeddings, labels_good, density_score=0.35)
        bad_score = _score_small_clustering(embeddings, labels_bad, density_score=0.35)

        assert good_score["score"] > bad_score["score"]
        assert good_score["largest_ratio"] < bad_score["largest_ratio"]
        assert "stability" in good_score
        assert "fragmentation" in good_score

    def test_small_refinement_gate_skips_balanced_solution(self):
        """Balanced, well-separated results should not trigger extra sweeps."""
        should_refine, reason = small_mod._should_refine_small_solution(
            {
                "score": 0.78,
                "noise_ratio": 0.04,
                "largest_ratio": 0.12,
                "n_clusters": 10.0,
                "separation": 0.62,
                "stability": 0.81,
            }
        )

        assert should_refine is False
        assert reason == ""

    def test_small_refinement_gate_detects_blob_like_solution(self):
        """Large, weakly separated dominant clusters should trigger refinement."""
        should_refine, reason = small_mod._should_refine_small_solution(
            {
                "score": 0.66,
                "noise_ratio": 0.08,
                "largest_ratio": 0.42,
                "n_clusters": 5.0,
                "separation": 0.18,
                "stability": 0.34,
            }
        )

        assert should_refine is True
        assert "blob" in reason

    def test_small_pipeline_does_not_refit_duplicate_zero_min_dist_umap(self, monkeypatch):
        """The primary sweep should cache one UMAP fit per unique (nn, nc, min_dist)."""
        fit_calls = []

        class FakeUMAP:
            def __init__(
                self,
                n_neighbors,
                n_components,
                min_dist,
                metric,
                random_state=None,
                n_jobs=-1,
                **kwargs,
            ):
                self.params = (n_neighbors, n_components, float(min_dist))

            def fit_transform(self, embeddings):
                fit_calls.append(self.params)
                dims = min(max(1, self.params[1]), embeddings.shape[1])
                return embeddings[:, :dims]

        class FakeHDBSCAN:
            def __init__(self, *args, **kwargs):
                self.relative_validity_ = 0.35

            def fit_predict(self, embeddings):
                labels = (np.arange(len(embeddings)) % 3).astype(np.int32)
                self.relative_validity_ = 0.35
                return labels

            def fit(self, embeddings):
                self.relative_validity_ = 0.35
                return self

        monkeypatch.setitem(sys.modules, "umap", types.SimpleNamespace(UMAP=FakeUMAP))
        monkeypatch.setitem(sys.modules, "hdbscan", types.SimpleNamespace(HDBSCAN=FakeHDBSCAN))

        monkeypatch.setattr(small_mod, "compute_optimal_umap_neighbors", lambda n: 10)
        monkeypatch.setattr(small_mod, "compute_optimal_umap_components", lambda d, n: 3)
        monkeypatch.setattr(small_mod, "_adaptive_small_cluster_sizes", lambda n, d=None: [6])
        monkeypatch.setattr(small_mod, "_should_refine_small_solution", lambda metrics: (False, ""))
        monkeypatch.setattr(small_mod, "_recover_small_noise_labels", lambda embeddings, labels: labels)
        monkeypatch.setattr(
            small_mod,
            "_split_oversized_small_clusters",
            lambda embeddings, labels, *args, **kwargs: labels,
        )
        monkeypatch.setattr(small_mod, "merge_near_duplicate_clusters", lambda embeddings, labels, **kw: labels)

        rng = np.random.default_rng(123)
        embeddings = normalize_vectors(rng.normal(size=(24, 6)).astype(np.float32))
        labels = small_mod.cluster_small(embeddings)

        assert labels.shape == (24,)
        assert fit_calls.count((10, 3, 0.0)) == 1
        assert fit_calls.count((10, 3, 0.05)) == 1
        assert len(fit_calls) == 2

    def test_noise_recovery_only_assigns_confident_points(self):
        """Ambiguous points should remain noise after cautious recovery."""
        rng = np.random.default_rng(7)

        center_a = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        center_b = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)

        cluster_a = center_a + rng.normal(0.0, 0.02, size=(8, 4))
        cluster_b = center_b + rng.normal(0.0, 0.02, size=(8, 4))
        confident_noise = np.array([[0.97, 0.08, 0.0, 0.0]], dtype=np.float32)
        ambiguous_noise = np.array([[0.71, 0.70, 0.0, 0.0]], dtype=np.float32)

        embeddings = normalize_vectors(
            np.vstack([cluster_a, cluster_b, confident_noise, ambiguous_noise]).astype(np.float32)
        )
        labels = np.array([0] * 8 + [1] * 8 + [-1, -1], dtype=np.int32)

        recovered = _recover_small_noise_labels(embeddings, labels)

        assert recovered[-2] == 0
        assert recovered[-1] == -1


class TestSemanticClustererEmbed:
    """Tests for the embed method."""

    def test_embed_method(self, sample_texts, mock_embedder):
        """Test direct embedding generation."""
        clusterer = SemanticClusterer(embedding_model=mock_embedder)
        embeddings = clusterer.embed(sample_texts)

        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape[0] == len(sample_texts)

    def test_embed_empty(self, mock_embedder):
        """Test embedding empty list."""
        clusterer = SemanticClusterer(embedding_model=mock_embedder)
        embeddings = clusterer.embed([])

        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape[0] == 0


class TestSemanticClustererValidation:
    """Tests for input validation and edge cases."""

    def test_init_with_dict_config_and_verbose_does_not_crash(self):
        """Test that dict config with verbose=False and verbose=True doesn't crash."""
        clusterer = SemanticClusterer(config={"batch_size": 64, "verbose": False}, verbose=True)
        assert clusterer.verbose is True

    def test_invalid_return_format_raises(self, sample_texts, mock_embedder):
        """Test that invalid return_format raises ValueError."""
        clusterer = SemanticClusterer(embedding_model=mock_embedder)
        with pytest.raises(ValueError):
            clusterer.cluster(sample_texts, return_format="bogus")

    def test_filtered_texts_are_dropped(self, mock_embedder):
        """Test that filtered/empty texts are not emitted as fake clusters."""
        clusterer = SemanticClusterer(embedding_model=mock_embedder)
        result = clusterer.cluster(["", "   ", "valid text"])
        # Result should only contain valid text, not empty strings
        all_texts = [t for cluster in result for t in cluster]
        assert "" not in all_texts
        assert "   " not in all_texts

    def test_single_text_filtered_out_returns_empty(self, mock_embedder):
        """Test that if all texts are filtered out, empty result is returned."""
        clusterer = SemanticClusterer(embedding_model=mock_embedder)
        result = clusterer.cluster(["", " "])
        assert result == []

    def test_embed_empty_returns_empty_2d_array(self, mock_embedder):
        """Test that embed([]) returns a 2D array with shape (0, 0)."""
        clusterer = SemanticClusterer(embedding_model=mock_embedder)
        embeddings = clusterer.embed([])
        assert isinstance(embeddings, np.ndarray)
        assert len(embeddings.shape) == 2
        assert embeddings.shape[0] == 0

    def test_config_reduction_components_never_zero(self):
        """Test that get_reduction_components never returns 0."""
        config = ClustererConfig()
        assert config.get_reduction_components(1) == 1
        assert config.get_reduction_components(2) >= 1
        assert config.get_reduction_components(100) >= 1

    def test_invalid_config_strategy_raises(self):
        """Test that passing strategy as a config field raises ValueError."""
        with pytest.raises((ValueError, TypeError)):
            ClustererConfig(strategy="invalid")

    def test_invalid_config_reduction_raises(self):
        """Test that passing reduction as a config field raises ValueError."""
        with pytest.raises((ValueError, TypeError)):
            ClustererConfig(reduction="invalid")

    def test_invalid_batch_size_raises(self):
        """Test that batch_size <= 0 raises ValueError."""
        with pytest.raises(ValueError):
            ClustererConfig(batch_size=0)

    def test_invalid_config_dict_field_raises(self):
        """Test that invalid config dict fields raise ValueError."""
        with pytest.raises(ValueError, match="Invalid config field"):
            SemanticClusterer(config={"noise_threshold": 0.2})

    def test_invalid_config_dict_multiple_fields_raises(self):
        """Test that multiple invalid fields are rejected."""
        with pytest.raises(ValueError, match="Invalid config field"):
            SemanticClusterer(config={"_min_cluster_size": 10, "_min_samples": 5})

    def test_valid_config_dict_fields_work(self):
        """Test that valid config dict fields work."""
        clusterer = SemanticClusterer(config={
            "batch_size": 128,
            "normalize_embeddings": False,
        })
        assert clusterer.config.batch_size == 128
        assert clusterer.config.normalize_embeddings is False

    def test_strategy_and_reduction_are_rejected_in_config_dict(self):
        """strategy and reduction are internal — passing them via dict raises ValueError."""
        with pytest.raises(ValueError, match="Invalid config field"):
            SemanticClusterer(config={"strategy": "small"})
        with pytest.raises(ValueError, match="Invalid config field"):
            SemanticClusterer(config={"reduction": "pca"})
