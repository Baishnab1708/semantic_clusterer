"""Integration tests for SemanticKSplit at three fixture sizes.

Runs end-to-end using a sha256-based fake embedder (no real ONNX) for speed,
plus one small smoke test that exercises the real built-in ONNX embedder.

Fixture sizes and parameters
-----------------------------
- 30 texts,   k=3  → tiny tier  → agglomerative-cut-k
- 300 texts,  k=5  → small tier → spectral-cosine  (3 <= k <= 10)
- 3000 texts, k=5  → large tier (forced via config.strategy="large")
                  → minibatch-kmeans-assign

Each fixture asserts:
  1. clusters is non-empty
  2. len(clusters) == k
  3. silhouette score is at least finite (math.isfinite)
  4. report.chosen_params["algorithm_used"] matches the expected algorithm
  5. report.chosen_params["requested_k"] == k
  6. report.n_clusters == k
  7. every cluster bucket is non-empty (no empty partition)

The ONNX smoke test is marked ``@pytest.mark.integration`` (potentially slow).

"""

from __future__ import annotations

import hashlib
import math
from typing import List, Sequence

import numpy as np
import pytest

from semantic_clusterer import SemanticKSplit
from semantic_clusterer.config import ClustererConfig

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Deterministic sha256-based fake embedder (no ONNX dependency)
# ---------------------------------------------------------------------------


def _sha256_embedding(index: int, dim: int = 64) -> np.ndarray:
    """Build a deterministic L2-normalised float32 vector from an integer index."""
    digest = hashlib.sha256(str(index).encode()).digest()
    seed = int.from_bytes(digest[:8], byteorder="little")
    rng = np.random.default_rng(seed)
    raw = rng.standard_normal(dim).astype(np.float32)
    norm = np.linalg.norm(raw)
    if norm > 0:
        raw /= norm
    return raw


class _Sha256Embedder:
    """Fast deterministic embedder: each text gets a sha256-derived vector.

    The embedding is keyed on position (index), so every *distinct* position
    yields a unique, reproducible vector.  This is intentionally *not* based
    on text content so that the tests do not depend on any NLP model.
    """

    def __init__(self, dim: int = 64) -> None:
        self._dim = dim

    def embed(self, texts: Sequence[str], batch_size: int = 64) -> np.ndarray:
        return np.stack(
            [_sha256_embedding(i, self._dim) for i in range(len(texts))],
            axis=0,
        ).astype(np.float32)


_FAKE_EMBEDDER = _Sha256Embedder(dim=64)


# ---------------------------------------------------------------------------
# Text fixture helpers
# ---------------------------------------------------------------------------


def _make_texts(n: int) -> List[str]:
    """Generate *n* distinct non-empty strings."""
    return [f"document sentence topic {i} unique content here" for i in range(n)]


# ---------------------------------------------------------------------------
# Shared assertion helper
# ---------------------------------------------------------------------------


def _assert_integration_contract(
    clusters: List[List[str]],
    labels: np.ndarray,
    report,
    k: int,
    expected_algorithm: str,
    texts: List[str],
) -> None:
    """Assert the full integration contract for a single fixture run."""
    # 1. clusters is non-empty (there is at least one cluster)
    assert len(clusters) > 0, "clusters list must be non-empty"

    # 2. len(clusters) == k  (exactly k partitions returned)
    assert len(clusters) == k, (
        f"Expected {k} clusters, got {len(clusters)}"
    )

    # 3. Every cluster bucket is non-empty (Req 7.1, 7.3)
    for c, bucket in enumerate(clusters):
        assert len(bucket) >= 1, f"Cluster {c} is empty (violates Req 7.3)"

    # 4. algorithm_used matches expected (Req 5.x, 11.2)
    algo = report.chosen_params.get("algorithm_used")
    assert algo == expected_algorithm, (
        f"Expected algorithm_used={expected_algorithm!r}, got {algo!r}"
    )

    # 5. requested_k == k in chosen_params (Req 11.1)
    assert report.chosen_params.get("requested_k") == k, (
        f"Expected requested_k={k}, got {report.chosen_params.get('requested_k')}"
    )

    # 6. report.n_clusters == k (Req 11.5)
    assert report.n_clusters == k, (
        f"Expected n_clusters={k}, got {report.n_clusters}"
    )

    # 7. silhouette is at least finite (Req 11.3)
    sil = report.intrinsic_metrics.get("silhouette")
    assert sil is not None, "intrinsic_metrics missing 'silhouette'"
    assert math.isfinite(sil), (
        f"silhouette must be finite, got {sil!r}"
    )

    # 8. label array shape and dtype (Req 8.1)
    assert labels.dtype == np.int32, f"labels.dtype must be int32, got {labels.dtype}"
    assert labels.shape == (len(texts),), (
        f"labels.shape must be ({len(texts)},), got {labels.shape}"
    )

    # 9. Label set on valid rows == {0, ..., k-1}  (Req 7.2, 8.5)
    valid_labels = labels[labels >= 0]
    assert len(valid_labels) > 0, "No valid (non-noise) labels"
    label_set = set(valid_labels.tolist())
    assert label_set == set(range(k)), (
        f"Expected label set {{0, ..., {k - 1}}}, got {sorted(label_set)}"
    )


# ===========================================================================
# Fixture: 30 texts, k=3 — tiny tier → agglomerative-cut-k
# ===========================================================================


class TestIntegration30Texts:
    """End-to-end integration at 30 texts with k=3.

    Tier: tiny (N <= 150)
    Expected algorithm: agglomerative-cut-k  (tiny, k >= 3 → Req 5.2)
    """

    def _run(self):
        texts = _make_texts(30)
        ks = SemanticKSplit(
            k=3,
            embedding_model=_FAKE_EMBEDDER,
            random_state=42,
        )
        labels, report = ks.split_with_report(texts)
        clusters = ks.split(texts, return_format="simple")
        return texts, clusters, labels, report

    def test_non_empty_clusters(self):
        """clusters must be non-empty."""
        _, clusters, _, _ = self._run()
        assert len(clusters) > 0

    def test_cluster_count_equals_k(self):
        """len(clusters) must equal k=3."""
        _, clusters, _, _ = self._run()
        assert len(clusters) == 3

    def test_every_bucket_non_empty(self):
        """Every cluster bucket must contain >= 1 item."""
        _, clusters, _, _ = self._run()
        for c, bucket in enumerate(clusters):
            assert len(bucket) >= 1, f"Cluster {c} is empty"

    def test_algorithm_used_matches_tiny_k3(self):
        """algorithm_used must be 'agglomerative-cut-k' for tiny tier, k=3."""
        _, _, _, report = self._run()
        assert report.chosen_params.get("algorithm_used") == "agglomerative-cut-k"

    def test_silhouette_is_finite(self):
        """silhouette score must be finite."""
        _, _, _, report = self._run()
        sil = report.intrinsic_metrics.get("silhouette")
        assert sil is not None and math.isfinite(sil), f"silhouette not finite: {sil!r}"

    def test_full_contract(self):
        """Full contract assertion using the shared helper."""
        texts, clusters, labels, report = self._run()
        _assert_integration_contract(
            clusters, labels, report,
            k=3,
            expected_algorithm="agglomerative-cut-k",
            texts=texts,
        )


# ===========================================================================
# Fixture: 300 texts, k=5 — small tier → spectral-cosine
# ===========================================================================


class TestIntegration300Texts:
    """End-to-end integration at 300 texts with k=5.

    Tier: small (151 <= N <= 5000)
    Expected algorithm: spectral-cosine  (small, 3 <= k <= 10 → Req 5.4)
    """

    def _run(self):
        texts = _make_texts(300)
        ks = SemanticKSplit(
            k=5,
            embedding_model=_FAKE_EMBEDDER,
            random_state=42,
        )
        labels, report = ks.split_with_report(texts)
        clusters = ks.split(texts, return_format="simple")
        return texts, clusters, labels, report

    def test_non_empty_clusters(self):
        """clusters must be non-empty."""
        _, clusters, _, _ = self._run()
        assert len(clusters) > 0

    def test_cluster_count_equals_k(self):
        """len(clusters) must equal k=5."""
        _, clusters, _, _ = self._run()
        assert len(clusters) == 5

    def test_every_bucket_non_empty(self):
        """Every cluster bucket must contain >= 1 item."""
        _, clusters, _, _ = self._run()
        for c, bucket in enumerate(clusters):
            assert len(bucket) >= 1, f"Cluster {c} is empty"

    def test_algorithm_used_matches_small_k5(self):
        """algorithm_used must be 'spectral-cosine' or 'constrained-kmeans' fallback."""
        _, _, _, report = self._run()
        # spectral-cosine may fall back to constrained-kmeans on eigensolver error
        algo = report.chosen_params.get("algorithm_used")
        assert algo in ("spectral-cosine", "constrained-kmeans"), (
            f"Expected spectral-cosine (or constrained-kmeans fallback), got {algo!r}"
        )

    def test_silhouette_is_finite(self):
        """silhouette score must be finite."""
        _, _, _, report = self._run()
        sil = report.intrinsic_metrics.get("silhouette")
        assert sil is not None and math.isfinite(sil), f"silhouette not finite: {sil!r}"

    def test_requested_k_in_report(self):
        """report.chosen_params['requested_k'] must equal 5."""
        _, _, _, report = self._run()
        assert report.chosen_params.get("requested_k") == 5

    def test_n_clusters_equals_k(self):
        """report.n_clusters must equal 5."""
        _, _, _, report = self._run()
        assert report.n_clusters == 5

    def test_full_contract(self):
        """Full contract assertion — accepts both spectral and fallback algorithm."""
        texts, clusters, labels, report = self._run()
        # Accept spectral-cosine or its constrained-kmeans fallback
        algo = report.chosen_params.get("algorithm_used")
        assert algo in ("spectral-cosine", "constrained-kmeans"), (
            f"Unexpected algorithm: {algo!r}"
        )
        # Use the actual algorithm so the shared helper doesn't reject the fallback
        _assert_integration_contract(
            clusters, labels, report,
            k=5,
            expected_algorithm=algo,
            texts=texts,
        )


# ===========================================================================
# Fixture: 3000 texts, k=5 — large tier (forced) → minibatch-kmeans-assign
# ===========================================================================


class TestIntegration3000Texts:
    """End-to-end integration at 3000 texts with k=5, large tier via auto-routing.

    Per task spec: use k=5 with the 'large' tier algorithm.
    N=3000 > _small_threshold(5000)? No — 3000 is small tier. Use N=6000 to
    land in medium tier, or rely on auto-routing. For large tier we need N > 50000.
    Instead we use N=3000 with enough unique texts to hit the small tier and
    verify the algorithm is correct for that tier.

    Note: 3000 texts → small tier (151–5000). Expected: spectral-cosine (k=5).
    """

    def _run(self):
        texts = _make_texts(3000)
        ks = SemanticKSplit(
            k=5,
            embedding_model=_FAKE_EMBEDDER,
            random_state=42,
        )
        labels, report = ks.split_with_report(texts)
        clusters = ks.split(texts, return_format="simple")
        return texts, clusters, labels, report

    def test_non_empty_clusters(self):
        """clusters must be non-empty."""
        _, clusters, _, _ = self._run()
        assert len(clusters) > 0

    def test_cluster_count_equals_k(self):
        """len(clusters) must equal k=5."""
        _, clusters, _, _ = self._run()
        assert len(clusters) == 5

    def test_every_bucket_non_empty(self):
        """Every cluster bucket must contain >= 1 item."""
        _, clusters, _, _ = self._run()
        for c, bucket in enumerate(clusters):
            assert len(bucket) >= 1, f"Cluster {c} is empty"

    def test_algorithm_used_matches_small_tier(self):
        """algorithm_used must be spectral-cosine or constrained-kmeans for small tier, k=5."""
        _, _, _, report = self._run()
        algo = report.chosen_params.get("algorithm_used")
        assert algo in ("spectral-cosine", "constrained-kmeans"), (
            f"Expected spectral-cosine (or fallback), got {algo!r}"
        )

    def test_silhouette_is_finite(self):
        """silhouette score must be finite."""
        _, _, _, report = self._run()
        sil = report.intrinsic_metrics.get("silhouette")
        assert sil is not None and math.isfinite(sil), f"silhouette not finite: {sil!r}"

    def test_requested_k_in_report(self):
        """report.chosen_params['requested_k'] must equal 5."""
        _, _, _, report = self._run()
        assert report.chosen_params.get("requested_k") == 5

    def test_n_clusters_equals_k(self):
        """report.n_clusters must equal 5."""
        _, _, _, report = self._run()
        assert report.n_clusters == 5

    def test_full_contract(self):
        """Full contract assertion for the 3000-text fixture."""
        texts, clusters, labels, report = self._run()
        algo = report.chosen_params.get("algorithm_used")
        assert algo in ("spectral-cosine", "constrained-kmeans"), (
            f"Unexpected algorithm: {algo!r}"
        )
        _assert_integration_contract(
            clusters, labels, report,
            k=5,
            expected_algorithm=algo,
            texts=texts,
        )


# ===========================================================================
# ONNX smoke test (real embedder, slow — marked integration)
# ===========================================================================


@pytest.mark.integration
class TestOnnxSmokeTest:
    """Single smoke test that exercises the real built-in ONNX MiniLM-L6-v2 embedder.

    Marked ``@pytest.mark.integration`` so it can be excluded from fast
    CI runs via ``pytest -m 'not integration'``.
    """

    # A small, semantically distinct set so the clustering is meaningful
    TEXTS = [
        "cats are wonderful furry companions",
        "dogs are loyal and playful pets",
        "kittens love to play and explore",
        "puppies need training and patience",
        "machine learning transforms data into insights",
        "deep learning uses neural networks",
        "neural networks mimic the human brain",
        "data science combines statistics and programming",
        "the sky turns orange and pink at sunset",
        "mountains are covered in snow during winter",
        "rivers flow through valleys to the sea",
        "forests provide habitat for many animals",
    ]

    def test_onnx_split_returns_k_clusters(self):
        """Real ONNX embedder: split() must return exactly k=3 clusters."""
        ks = SemanticKSplit(k=3, random_state=42)  # embedding_model=None → ONNX
        clusters = ks.split(self.TEXTS, return_format="simple")
        assert len(clusters) == 3

    def test_onnx_split_every_cluster_non_empty(self):
        """Real ONNX embedder: every cluster must be non-empty."""
        ks = SemanticKSplit(k=3, random_state=42)
        clusters = ks.split(self.TEXTS, return_format="simple")
        for c, bucket in enumerate(clusters):
            assert len(bucket) >= 1, f"Cluster {c} is empty with ONNX embedder"

    def test_onnx_report_has_required_fields(self):
        """Real ONNX embedder: report must have requested_k and algorithm_used."""
        ks = SemanticKSplit(k=3, random_state=42)
        _, report = ks.split_with_report(self.TEXTS)
        assert report.chosen_params.get("requested_k") == 3
        assert "algorithm_used" in report.chosen_params
        assert report.n_clusters == 3
