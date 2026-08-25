"""End-to-end tests for the persistence / production API.

Covers:
  * ``fit`` populates ``_fitted_state`` with the right shapes.
  * ``predict`` is row-aligned and respects the OOD threshold modes.
  * ``fit_predict`` matches the labels stored on ``_fitted_state``.
  * ``save`` writes a complete, JSON-readable manifest.
  * ``load`` reconstructs an instance whose predictions match the original.
  * ``predict`` before ``fit`` raises ``RuntimeError``.
  * ``assign_to_centroids`` low-level helper (centroid lookup math).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List

import numpy as np
import pytest

from semantic_clusterer import SemanticClusterer
from semantic_clusterer.persistence import (
    ClusterStats,
    FittedState,
    assign_to_centroids,
    load_state,
    save_state,
)


# Synthetic 32-dim embeddings are below the supported "low" dim-band floor
# (256), so the package emits a UserWarning during fit/predict.  We
# acknowledge the warning at module level rather than silencing it
# globally.
pytestmark = pytest.mark.filterwarnings(
    "ignore:Embedding dim .* is below low band lower bound:UserWarning"
)



# ---------------------------------------------------------------------------
# Fixtures local to this module
# ---------------------------------------------------------------------------

_DIM = 32


def _normed(rng: np.random.Generator, n: int, dim: int = _DIM) -> np.ndarray:
    """Return ``(n, dim)`` L2-normalised float32 vectors."""
    v = rng.standard_normal((n, dim)).astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v


def _make_three_cluster_dataset(seed: int = 0) -> tuple[List[str], np.ndarray]:
    """3 well-separated tight clusters of 30 rows each (90 total) + matched embeddings."""
    rng = np.random.default_rng(seed)
    centers = _normed(rng, 3, _DIM)

    embeddings = np.empty((90, _DIM), dtype=np.float32)
    texts: List[str] = []
    for cluster_idx, center in enumerate(centers):
        for k in range(30):
            v = center + rng.standard_normal(_DIM).astype(np.float32) * 0.02
            embeddings[cluster_idx * 30 + k] = v / np.linalg.norm(v)
            texts.append(f"cluster {cluster_idx} item {k}")

    return texts, embeddings


class _ReplayEmbedder:
    """Deterministic embedder that returns rows from a fixed pool by text identity.

    The clusterer dedupes by content before embedding; we look the cleaned
    text up in a dict so train/predict on identical texts always returns
    the same vector.
    """

    def __init__(self, texts: List[str], embeddings: np.ndarray):
        self._lookup = {t: embeddings[i] for i, t in enumerate(texts)}
        self._dim = embeddings.shape[1]
        self._fallback_seed = 12345

    def embed(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        rng = np.random.default_rng(self._fallback_seed)
        out = np.empty((len(texts), self._dim), dtype=np.float32)
        for i, t in enumerate(texts):
            if t in self._lookup:
                out[i] = self._lookup[t]
            else:
                # Fresh deterministic vector — same text → same vector
                local = np.random.default_rng(abs(hash(t)) % (2**32))
                v = local.standard_normal(self._dim).astype(np.float32)
                out[i] = v / np.linalg.norm(v)
        return out


@pytest.fixture
def fitted_clusterer():
    """A SemanticClusterer fitted on 90 texts with 3 clean clusters."""
    texts, embeddings = _make_three_cluster_dataset(seed=0)
    embedder = _ReplayEmbedder(texts, embeddings)
    clusterer = SemanticClusterer(embedding_model=embedder, random_state=42)
    clusterer.fit(texts)
    return clusterer, texts, embedder


# ---------------------------------------------------------------------------
# fit()
# ---------------------------------------------------------------------------

class TestFit:
    """``fit()`` populates the fitted state with the expected shape."""

    def test_fit_returns_self(self, fitted_clusterer):
        clusterer, _, _ = fitted_clusterer
        # fit() in the fixture already ran; calling again should still return self
        result = clusterer.fit(["a", "b", "c"] * 10)
        assert result is clusterer

    def test_is_fitted_true_after_fit(self, fitted_clusterer):
        clusterer, _, _ = fitted_clusterer
        # is_fitted is a property
        assert clusterer.is_fitted is True

    def test_state_centroids_shape(self, fitted_clusterer):
        clusterer, _, _ = fitted_clusterer
        state = clusterer._fitted_state
        assert state is not None
        assert state.centroids.ndim == 2
        assert state.centroids.dtype == np.float32
        assert state.centroids.shape[0] == state.n_clusters

    def test_state_cluster_ids_contiguous(self, fitted_clusterer):
        clusterer, _, _ = fitted_clusterer
        state = clusterer._fitted_state
        ids = state.cluster_ids.tolist()
        # Public cluster ids must be contiguous from 0 (no -1, no gaps).
        assert ids == list(range(len(ids)))

    def test_centroids_l2_normalised(self, fitted_clusterer):
        clusterer, _, _ = fitted_clusterer
        state = clusterer._fitted_state
        if state.centroids.size:
            norms = np.linalg.norm(state.centroids, axis=1)
            np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_train_labels_aligned_with_input(self, fitted_clusterer):
        clusterer, texts, _ = fitted_clusterer
        state = clusterer._fitted_state
        assert state.train_labels.shape == (len(texts),)
        assert state.train_labels.dtype == np.int32

    def test_cluster_cohesion_size_matches_n_clusters(self, fitted_clusterer):
        clusterer, _, _ = fitted_clusterer
        state = clusterer._fitted_state
        # cluster_cohesion may be empty when keyword extraction is off, but
        # when populated it must match n_clusters.
        if state.cluster_cohesion:
            assert len(state.cluster_cohesion) == state.n_clusters
            for cs in state.cluster_cohesion:
                assert isinstance(cs, ClusterStats)
                assert cs.size >= 1
                assert -1.0 <= cs.min_sim <= 1.0
                assert -1.0 <= cs.mean_sim <= 1.0
                assert -1.0 <= cs.p10_sim <= 1.0


# ---------------------------------------------------------------------------
# predict()
# ---------------------------------------------------------------------------

class TestPredict:
    """``predict()`` is row-aligned and respects the OOD threshold."""

    def test_predict_row_count_matches_input(self, fitted_clusterer):
        clusterer, _, _ = fitted_clusterer
        out = clusterer.predict(["one", "two", "three"])
        assert isinstance(out, np.ndarray)
        assert out.dtype == np.int32
        assert out.shape == (3,)

    def test_predict_empty_returns_empty(self, fitted_clusterer):
        clusterer, _, _ = fitted_clusterer
        out = clusterer.predict([])
        assert out.shape == (0,)
        assert out.dtype == np.int32

    def test_predict_assigns_training_texts_to_known_cluster(self, fitted_clusterer):
        clusterer, texts, _ = fitted_clusterer
        out = clusterer.predict(texts[:30])  # rows 0..29 are cluster 0 in the synthetic data
        # All thirty should land in the same cluster id (whatever the public id is).
        valid = out[out >= 0]
        if valid.size:
            assert len(np.unique(valid)) == 1

    def test_predict_missing_values_get_minus_one(self, fitted_clusterer):
        clusterer, _, _ = fitted_clusterer
        out = clusterer.predict(["valid text", None, "another"])
        assert out[1] == -1

    def test_predict_empty_strings_get_minus_one(self, fitted_clusterer):
        clusterer, _, _ = fitted_clusterer
        out = clusterer.predict(["valid text", "", "..."])
        assert out[1] == -1
        assert out[2] == -1

    def test_predict_threshold_none_disables_ood(self, fitted_clusterer):
        clusterer, _, _ = fitted_clusterer
        # Garbage strings get garbage embeddings — with threshold=None they
        # MUST still land somewhere.
        out = clusterer.predict(
            ["totally unrelated thing"] * 5, outlier_threshold=None
        )
        assert np.all(out >= 0)

    def test_predict_threshold_inf_marks_everything_ood(self, fitted_clusterer):
        clusterer, texts, _ = fitted_clusterer
        # Threshold above 1.0 (max possible cosine) → every row OOD.
        out = clusterer.predict(texts[:5], outlier_threshold=1.5)
        assert np.all(out == -1)

    def test_predict_global_uses_calibrated_threshold(self, fitted_clusterer):
        clusterer, texts, _ = fitted_clusterer
        # outlier_threshold="global" — should match a fresh call with
        # the explicit global float.
        global_out = clusterer.predict(texts[:10], outlier_threshold="global")
        state = clusterer._fitted_state
        if state.auto_outlier_threshold is not None:
            explicit = clusterer.predict(
                texts[:10],
                outlier_threshold=float(state.auto_outlier_threshold),
            )
            np.testing.assert_array_equal(global_out, explicit)

    def test_predict_auto_delegates_to_adaptive(self, fitted_clusterer):
        clusterer, texts, _ = fitted_clusterer
        # default outlier_threshold="auto" should produce the exact same
        # predictions as explicitly passing outlier_threshold="adaptive"
        auto = clusterer.predict(texts[:10], outlier_threshold="auto")
        adaptive = clusterer.predict(texts[:10], outlier_threshold="adaptive")
        np.testing.assert_array_equal(auto, adaptive)

    def test_predict_before_fit_raises(self):
        clusterer = SemanticClusterer()
        with pytest.raises(RuntimeError, match="not been fitted"):
            clusterer.predict(["anything"])


# ---------------------------------------------------------------------------
# fit_predict()
# ---------------------------------------------------------------------------

class TestFitPredict:
    """``fit_predict`` returns the same labels stored on the fitted state."""

    def test_fit_predict_labels_shape_matches_input(self):
        texts, embeddings = _make_three_cluster_dataset(seed=1)
        clusterer = SemanticClusterer(
            embedding_model=_ReplayEmbedder(texts, embeddings),
            random_state=42,
        )
        labels = clusterer.fit_predict(texts)
        assert labels.shape == (len(texts),)
        assert labels.dtype == np.int32

    def test_fit_predict_matches_train_labels(self):
        texts, embeddings = _make_three_cluster_dataset(seed=2)
        clusterer = SemanticClusterer(
            embedding_model=_ReplayEmbedder(texts, embeddings),
            random_state=42,
        )
        labels = clusterer.fit_predict(texts)
        np.testing.assert_array_equal(labels, clusterer._fitted_state.train_labels)


# ---------------------------------------------------------------------------
# save() / load() round-trip
# ---------------------------------------------------------------------------

class TestSaveLoad:
    """``save`` + ``load`` produces an instance with identical predictions."""

    def test_save_creates_manifest(self, fitted_clusterer, tmp_path):
        clusterer, _, _ = fitted_clusterer
        out_dir = tmp_path / "model"
        clusterer.save(str(out_dir))

        assert out_dir.is_dir()
        manifest_path = out_dir / "manifest.json"
        assert manifest_path.exists(), "manifest.json must be written last"

        # Manifest is valid JSON with required fields
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for key in ("schema", "embedding_dim", "n_clusters", "cluster_ids", "config_snapshot"):
            assert key in manifest, f"Missing manifest key: {key!r}"

    def test_save_writes_centroids_and_labels(self, fitted_clusterer, tmp_path):
        clusterer, _, _ = fitted_clusterer
        out_dir = tmp_path / "model"
        clusterer.save(str(out_dir))

        files = {p.name for p in out_dir.iterdir()}
        assert "centroids.npy" in files
        assert "labels.npy" in files

    def test_save_before_fit_raises(self, tmp_path):
        clusterer = SemanticClusterer()
        with pytest.raises(RuntimeError, match="not been fitted"):
            clusterer.save(str(tmp_path / "model"))

    def test_load_round_trip_predicts_identically(self, fitted_clusterer, tmp_path):
        clusterer, texts, embedder = fitted_clusterer
        out_dir = tmp_path / "model"
        clusterer.save(str(out_dir))

        # Reload — note: we re-inject the same embedder
        loaded = SemanticClusterer.load(str(out_dir), embedding_model=embedder)
        assert loaded.is_fitted is True

        original_labels = clusterer.predict(texts[:10])
        loaded_labels = loaded.predict(texts[:10])
        np.testing.assert_array_equal(original_labels, loaded_labels)

    def test_load_centroids_match_original(self, fitted_clusterer, tmp_path):
        clusterer, _, embedder = fitted_clusterer
        out_dir = tmp_path / "model"
        clusterer.save(str(out_dir))

        loaded = SemanticClusterer.load(str(out_dir), embedding_model=embedder)

        original_state = clusterer._fitted_state
        loaded_state = loaded._fitted_state

        np.testing.assert_allclose(
            loaded_state.centroids,
            original_state.centroids,
            atol=1e-6,
        )
        np.testing.assert_array_equal(
            loaded_state.cluster_ids,
            original_state.cluster_ids,
        )
        assert loaded_state.n_clusters == original_state.n_clusters
        assert loaded_state.embedding_dim == original_state.embedding_dim
        assert loaded_state.dim_band == original_state.dim_band

    def test_load_missing_dir_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            SemanticClusterer.load(str(tmp_path / "does_not_exist"))

    def test_load_missing_manifest_raises(self, tmp_path):
        empty_dir = tmp_path / "no_manifest"
        empty_dir.mkdir()
        with pytest.raises(FileNotFoundError, match="manifest"):
            SemanticClusterer.load(str(empty_dir))

    def test_load_unsupported_schema_raises(self, fitted_clusterer, tmp_path):
        clusterer, _, _ = fitted_clusterer
        out_dir = tmp_path / "model"
        clusterer.save(str(out_dir))

        # Tamper with the manifest schema so load() should reject it.
        manifest_path = out_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["schema"] = 999
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        with pytest.raises(ValueError, match="schema"):
            SemanticClusterer.load(str(out_dir))

    def test_save_state_load_state_round_trip(self, fitted_clusterer, tmp_path):
        """The low-level ``save_state``/``load_state`` round-trip preserves geometry."""
        clusterer, _, _ = fitted_clusterer
        out_dir = tmp_path / "model"
        save_state(clusterer._fitted_state, str(out_dir))

        loaded = load_state(str(out_dir))
        np.testing.assert_allclose(
            loaded.centroids,
            clusterer._fitted_state.centroids,
            atol=1e-6,
        )


# ---------------------------------------------------------------------------
# assign_to_centroids — the low-level math used by predict
# ---------------------------------------------------------------------------

class TestAssignToCentroids:
    """Direct tests for the centroid-assignment helper."""

    def test_assign_picks_nearest_centroid(self):
        # Two centroids on opposite poles
        centroids = np.array(
            [[1.0, 0.0], [0.0, 1.0]],
            dtype=np.float32,
        )
        cluster_ids = np.array([7, 13], dtype=np.int32)

        # Three queries close to centroid 0, 1, 0 respectively
        queries = np.array(
            [[0.99, 0.10], [0.10, 0.99], [0.95, 0.30]],
            dtype=np.float32,
        )
        queries /= np.linalg.norm(queries, axis=1, keepdims=True)

        out = assign_to_centroids(queries, centroids, cluster_ids)
        assert out.tolist() == [7, 13, 7]
        assert out.dtype == np.int32

    def test_assign_returns_minus_one_below_threshold(self):
        centroids = np.array([[1.0, 0.0]], dtype=np.float32)
        cluster_ids = np.array([0], dtype=np.int32)

        # A query orthogonal to the centroid → cosine 0 → below 0.5 → OOD.
        queries = np.array([[0.0, 1.0]], dtype=np.float32)

        out = assign_to_centroids(
            queries, centroids, cluster_ids, outlier_threshold=0.5
        )
        assert out.tolist() == [-1]

    def test_assign_threshold_none_never_returns_minus_one(self):
        centroids = np.array([[1.0, 0.0]], dtype=np.float32)
        cluster_ids = np.array([0], dtype=np.int32)
        queries = np.array([[0.0, 1.0], [-1.0, 0.0]], dtype=np.float32)

        out = assign_to_centroids(queries, centroids, cluster_ids, outlier_threshold=None)
        assert (out >= 0).all()

    def test_assign_empty_queries_returns_empty(self):
        centroids = np.array([[1.0, 0.0]], dtype=np.float32)
        cluster_ids = np.array([0], dtype=np.int32)

        out = assign_to_centroids(
            np.empty((0, 2), dtype=np.float32),
            centroids,
            cluster_ids,
        )
        assert out.shape == (0,)
        assert out.dtype == np.int32

    def test_assign_empty_centroids_returns_all_minus_one(self):
        queries = np.array([[1.0, 0.0]], dtype=np.float32)
        out = assign_to_centroids(
            queries,
            np.empty((0, 2), dtype=np.float32),
            np.empty((0,), dtype=np.int32),
        )
        assert out.tolist() == [-1]

    def test_assign_adaptive_thresholds_filters_by_cluster(self):
        centroids = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        cluster_ids = np.array([0, 1], dtype=np.int32)
        
        # Query 1: similarity to centroid 0 is 0.9.
        # Query 2: similarity to centroid 1 is 0.9.
        queries = np.array([[0.9, np.sqrt(1.0 - 0.9**2)], [np.sqrt(1.0 - 0.9**2), 0.9]], dtype=np.float32)
        
        # Adaptive thresholds:
        # Cluster 0 threshold is 0.95 (so Query 1 will be OOD)
        # Cluster 1 threshold is 0.85 (so Query 2 will be kept as cluster 1)
        adaptive = {0: 0.95, 1: 0.85}
        
        out = assign_to_centroids(
            queries, centroids, cluster_ids, adaptive_thresholds=adaptive
        )
        assert out.tolist() == [-1, 1]
