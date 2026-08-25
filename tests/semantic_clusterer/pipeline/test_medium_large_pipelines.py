import sys
import types

import numpy as np

from semantic_clusterer.config import ClustererConfig
from semantic_clusterer.pipeline.large import _global_stitch_clusters, cluster_large
from semantic_clusterer.pipeline.medium import cluster_medium
from semantic_clusterer.pipeline.profile import DatasetProfile
from semantic_clusterer.pipeline.tuning import compute_large_coarse_partition_count
from semantic_clusterer.utils.similarity import normalize_vectors


def _make_clustered_embeddings(
    n_clusters: int,
    points_per_cluster: int,
    dim: int,
    noise_scale: float = 0.05,
) -> np.ndarray:
    rng = np.random.default_rng(42)
    centers = rng.normal(0.0, 1.0, size=(n_clusters, dim)).astype(np.float32)
    embeddings = []
    for center in centers:
        cluster = center + rng.normal(0.0, noise_scale, size=(points_per_cluster, dim)).astype(np.float32)
        embeddings.append(cluster)
    return normalize_vectors(np.vstack(embeddings).astype(np.float32))


def test_medium_respects_reduction_none(monkeypatch):
    embeddings = _make_clustered_embeddings(n_clusters=3, points_per_cluster=40, dim=64)

    def fail_get_reducer(*args, **kwargs):
        raise AssertionError("medium pipeline should not build a reducer when reduction=None")

    monkeypatch.setattr("semantic_clusterer.pipeline.medium.get_reducer", fail_get_reducer)

    # Patch _BaseConfig.get_reduction_for_strategy to always return None
    import semantic_clusterer.config as _cfg_mod
    monkeypatch.setattr(_cfg_mod._BaseConfig, "get_reduction_for_strategy", lambda self, s: None)

    labels = cluster_medium(embeddings, ClustererConfig(), log_fn=lambda _: None)

    assert labels.shape[0] == embeddings.shape[0]
    assert np.any(labels >= 0)


def test_large_respects_reduction_none(monkeypatch):
    embeddings = _make_clustered_embeddings(n_clusters=4, points_per_cluster=30, dim=64)

    def fail_get_reducer(*args, **kwargs):
        raise AssertionError("large pipeline should not build a reducer when reduction=None")

    monkeypatch.setattr("semantic_clusterer.pipeline.large.get_reducer", fail_get_reducer)

    import semantic_clusterer.config as _cfg_mod
    monkeypatch.setattr(_cfg_mod._BaseConfig, "get_reduction_for_strategy", lambda self, s: None)

    labels = cluster_large(embeddings, ClustererConfig(), log_fn=lambda _: None)

    assert labels.shape[0] == embeddings.shape[0]
    assert np.any(labels >= 0)


def test_large_partition_count_is_capped_by_sample_count():
    profile = DatasetProfile(
        n_samples=5,
        n_features=3072,
        effective_rank=512,
        variance_decay_ratio=0.8,
        local_density_mean=0.2,
        local_density_cv=0.9,
        distance_concentration=0.3,
        duplicate_ratio=0.0,
        near_duplicate_ratio=0.0,
        cluster_tendency=0.7,
        imbalance_tendency=0.3,
        memory_pressure=0.2,
    )

    n_coarse = compute_large_coarse_partition_count(profile)
    assert 2 <= n_coarse <= profile.n_samples


def test_medium_umap_cache_is_scoped_to_each_representation(monkeypatch):
    import semantic_clusterer.pipeline.medium as medium_mod

    fake_profile = DatasetProfile(
        n_samples=120,
        n_features=256,
        effective_rank=96,
        variance_decay_ratio=0.8,
        local_density_mean=0.2,
        local_density_cv=0.8,
        distance_concentration=0.4,
        duplicate_ratio=0.0,
        near_duplicate_ratio=0.0,
        cluster_tendency=0.8,
        imbalance_tendency=0.2,
        memory_pressure=0.1,
    )

    monkeypatch.setattr(medium_mod, "compute_dataset_profile", lambda embeddings: fake_profile)
    monkeypatch.setattr(medium_mod, "compute_medium_reduction_dimension", lambda profile, base_anchor_fn: 80)
    monkeypatch.setattr(
        medium_mod,
        "compute_reduction_candidates",
        lambda profile, target_dim, num_candidates=3: [64, 96],
    )
    monkeypatch.setattr(
        medium_mod,
        "compute_medium_hdbscan_candidates",
        lambda profile: {"min_cluster_sizes": [5], "min_samples": [1], "methods": ["eom"]},
    )
    monkeypatch.setattr(
        medium_mod,
        "compute_umap_parameters",
        lambda profile, reduction_dim: {
            "n_neighbors_candidates": [15],
            "n_components_candidates": [6],
        },
    )
    monkeypatch.setattr(medium_mod, "should_trigger_refinement", lambda metrics, thresholds: (False, ""))

    class DummyReducer:
        def __init__(self, n_components: int):
            self.n_components = n_components

        def fit_transform(self, embeddings: np.ndarray) -> np.ndarray:
            return np.full((len(embeddings), self.n_components), float(self.n_components), dtype=np.float32)

    monkeypatch.setattr(
        medium_mod,
        "get_reducer",
        lambda method, n_components, n_samples: DummyReducer(n_components),
    )

    umap_calls = []

    class FakeUMAP:
        def __init__(self, n_neighbors, n_components, min_dist, metric, random_state=None, n_jobs=-1, **kwargs):
            self.n_neighbors = n_neighbors
            self.n_components = n_components

        def fit_transform(self, embeddings):
            umap_calls.append((embeddings.shape[1], float(np.mean(embeddings))))
            return np.full((len(embeddings), self.n_components), float(np.mean(embeddings)), dtype=np.float32)

    class FakeHDBSCAN:
        def __init__(
            self,
            min_cluster_size,
            min_samples,
            metric,
            cluster_selection_method,
            cluster_selection_epsilon=0.0,
            gen_min_span_tree=False,
            approx_min_span_tree=True,
        ):
            self.relative_validity_ = 0.25

        def fit_predict(self, embeddings):
            midpoint = len(embeddings) // 2
            labels = np.zeros(len(embeddings), dtype=np.int32)
            labels[midpoint:] = 1
            return labels

    monkeypatch.setitem(sys.modules, "umap", types.SimpleNamespace(UMAP=FakeUMAP))
    monkeypatch.setitem(sys.modules, "hdbscan", types.SimpleNamespace(HDBSCAN=FakeHDBSCAN))

    embeddings = _make_clustered_embeddings(n_clusters=3, points_per_cluster=40, dim=256)
    labels = medium_mod.cluster_medium(embeddings, ClustererConfig(), log_fn=lambda _: None)

    assert labels.shape[0] == embeddings.shape[0]
    assert len(umap_calls) == 2
    assert {call[0] for call in umap_calls} == {64, 96}


def test_global_stitch_only_merges_cross_shard_duplicates():
    rng = np.random.default_rng(7)
    dim = 24

    center_a = normalize_vectors(rng.normal(0.0, 1.0, size=(1, dim)).astype(np.float32))[0]
    center_b = normalize_vectors(rng.normal(0.0, 1.0, size=(1, dim)).astype(np.float32))[0]

    cluster_0 = normalize_vectors(center_a + rng.normal(0.0, 0.01, size=(5, dim)).astype(np.float32))
    cluster_1 = normalize_vectors(center_a + rng.normal(0.0, 0.01, size=(5, dim)).astype(np.float32))
    cluster_2 = normalize_vectors(center_b + rng.normal(0.0, 0.01, size=(5, dim)).astype(np.float32))
    cluster_3 = normalize_vectors(center_b + rng.normal(0.0, 0.01, size=(5, dim)).astype(np.float32))

    embeddings = np.vstack([cluster_0, cluster_1, cluster_2, cluster_3]).astype(np.float32)
    labels = np.array([0] * 5 + [1] * 5 + [2] * 5 + [3] * 5, dtype=np.int32)
    label_to_shard = {0: 0, 1: 1, 2: 2, 3: 2}

    stitched = _global_stitch_clusters(embeddings, labels, label_to_shard, similarity_threshold=0.95)

    first_pair = set(stitched[:10])
    second_pair = set(stitched[10:])

    assert len(first_pair) == 1
    assert len(second_pair) == 2
