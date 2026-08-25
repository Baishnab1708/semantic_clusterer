"""Unit tests for k_algorithms selection matrix and restart helpers.

Covers:
- _select_k_algorithm: exhaustive (tier, k) -> algorithm string assertions
- _pick_better: silhouette > Davies-Bouldin > restart_index ordering
- _run_restarts: identical labels for identical seed sequences

"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from semantic_clusterer.k_algorithms.selection import _select_k_algorithm
from semantic_clusterer.k_algorithms.restart import (
    _RestartCandidate,
    _pick_better,
    _run_restarts,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_config() -> MagicMock:
    """Return a minimal mock ClustererConfig sufficient for _select_k_algorithm."""
    return MagicMock()


def _simple_config() -> SimpleNamespace:
    """Return a SimpleNamespace config as an alternative lightweight mock."""
    return SimpleNamespace()


def _make_candidate(
    n: int = 10,
    k: int = 2,
    silhouette: float = 0.5,
    davies_bouldin: float = 1.0,
    restart_index: int = 0,
) -> _RestartCandidate:
    """Build a _RestartCandidate with synthetic labels."""
    rng = np.random.default_rng(restart_index)
    labels = (rng.integers(0, k, size=n)).astype(np.int32)
    return _RestartCandidate(
        labels=labels,
        silhouette=silhouette,
        davies_bouldin=davies_bouldin,
        restart_index=restart_index,
    )


# ---------------------------------------------------------------------------
# Tests: _select_k_algorithm — exhaustive (tier, k) matrix
# ---------------------------------------------------------------------------

class TestSelectKAlgorithmMatrix:
    """Exhaustive (tier, k) -> algorithm string assertions."""

    # --- tiny tier -----------------------------------------------------------

    def test_tiny_k2_returns_bisecting_kmeans(self):
        """Tiny tier, k==2 -> bisecting-kmeans."""
        result = _select_k_algorithm("tiny", 2, _mock_config())
        assert result == "bisecting-kmeans"

    def test_tiny_k3_returns_agglomerative_cut_k(self):
        """Tiny tier, k==3 -> agglomerative-cut-k."""
        result = _select_k_algorithm("tiny", 3, _mock_config())
        assert result == "agglomerative-cut-k"

    def test_tiny_k10_returns_balanced_kmeans(self):
        """Tiny tier, k==10 -> balanced-kmeans."""
        result = _select_k_algorithm("tiny", 10, _mock_config())
        assert result == "balanced-kmeans"

    # --- small tier ----------------------------------------------------------

    def test_small_k2_returns_bisecting_kmeans(self):
        """Small tier, k==2 -> bisecting-kmeans."""
        result = _select_k_algorithm("small", 2, _mock_config())
        assert result == "bisecting-kmeans"

    def test_small_k3_returns_spectral_cosine(self):
        """Small tier, 3<=k<=10 -> spectral-cosine."""
        result = _select_k_algorithm("small", 3, _mock_config())
        assert result == "spectral-cosine"

    def test_small_k10_returns_spectral_cosine(self):
        """Small tier, k==10 (boundary) -> spectral-cosine."""
        result = _select_k_algorithm("small", 10, _mock_config())
        assert result == "spectral-cosine"

    def test_small_k11_returns_balanced_kmeans(self):
        """Small tier, k==11 (just over boundary) -> balanced-kmeans."""
        result = _select_k_algorithm("small", 11, _mock_config())
        assert result == "balanced-kmeans"

    def test_small_k20_returns_balanced_kmeans(self):
        """Small tier, k==20 -> balanced-kmeans."""
        result = _select_k_algorithm("small", 20, _mock_config())
        assert result == "balanced-kmeans"

    # --- medium tier ---------------------------------------------------------

    def test_medium_k2_returns_balanced_kmeans(self):
        """Medium tier, k==2 -> balanced-kmeans."""
        result = _select_k_algorithm("medium", 2, _mock_config())
        assert result == "balanced-kmeans"

    def test_medium_k5_returns_balanced_kmeans(self):
        """Medium tier, k==5 -> balanced-kmeans."""
        result = _select_k_algorithm("medium", 5, _mock_config())
        assert result == "balanced-kmeans"

    # --- large tier ----------------------------------------------------------

    def test_large_k2_returns_minibatch_kmeans_assign(self):
        """Large tier, k==2 -> minibatch-kmeans-assign."""
        result = _select_k_algorithm("large", 2, _mock_config())
        assert result == "minibatch-kmeans-assign"

    def test_large_k100_returns_minibatch_kmeans_assign(self):
        """Large tier, k==100 -> minibatch-kmeans-assign."""
        result = _select_k_algorithm("large", 100, _mock_config())
        assert result == "minibatch-kmeans-assign"

    # --- config object variants ----------------------------

    def test_accepts_simplenamespace_config(self):
        """Function must accept any config object (SimpleNamespace)."""
        result = _select_k_algorithm("medium", 3, _simple_config())
        assert result == "balanced-kmeans"

    def test_accepts_none_config(self):
        """Config parameter is reserved for future use; None is safe."""
        result = _select_k_algorithm("large", 5, None)
        assert result == "minibatch-kmeans-assign"

    # --- unknown tier raises ValueError ---------------------------------------

    def test_unknown_tier_raises_value_error(self):
        """Unknown tier should raise ValueError with informative message."""
        with pytest.raises(ValueError, match="unknown|Unknown"):
            _select_k_algorithm("xlarge", 5, _mock_config())


# ---------------------------------------------------------------------------
# Tests: _pick_better — silhouette > DB > restart_index ordering
# ---------------------------------------------------------------------------

class TestPickBetter:
    """_pick_better honors silhouette > -DB > restart_index."""

    def test_higher_silhouette_wins(self):
        """Higher silhouette is strictly preferred (primary criterion)."""
        a = _make_candidate(silhouette=0.8, davies_bouldin=2.0, restart_index=1)
        b = _make_candidate(silhouette=0.3, davies_bouldin=0.5, restart_index=0)
        # a has higher silhouette so should win, even if b has better DB and lower index
        result = _pick_better(a, b)
        assert result is a

    def test_lower_db_wins_when_silhouette_tied(self):
        """Lower Davies-Bouldin wins when silhouette scores are equal (secondary criterion)."""
        a = _make_candidate(silhouette=0.5, davies_bouldin=0.8, restart_index=1)
        b = _make_candidate(silhouette=0.5, davies_bouldin=1.5, restart_index=0)
        # a has lower DB so should win, even if b has lower restart_index
        result = _pick_better(a, b)
        assert result is a

    def test_lower_restart_index_wins_when_both_scores_tied(self):
        """Lower restart_index wins when silhouette and DB are both equal (tertiary criterion)."""
        a = _make_candidate(silhouette=0.5, davies_bouldin=1.0, restart_index=0)
        b = _make_candidate(silhouette=0.5, davies_bouldin=1.0, restart_index=1)
        # a has lower restart_index so should win
        result = _pick_better(a, b)
        assert result is a

    def test_lower_restart_index_wins_ties_reversed_args(self):
        """Order of arguments should not change the winner."""
        a = _make_candidate(silhouette=0.5, davies_bouldin=1.0, restart_index=0)
        b = _make_candidate(silhouette=0.5, davies_bouldin=1.0, restart_index=1)
        # Even when order is reversed, a (restart_index=0) wins
        result = _pick_better(b, a)
        assert result is a

    def test_nan_silhouette_loses_to_finite_silhouette(self):
        """NaN silhouette is treated as worst case (mapped to +inf in key)."""
        a = _make_candidate(silhouette=float("nan"), davies_bouldin=0.1, restart_index=0)
        b = _make_candidate(silhouette=-0.9, davies_bouldin=100.0, restart_index=99)
        # b has a finite (even negative) silhouette, which is better than NaN
        result = _pick_better(a, b)
        assert result is b

    def test_inf_db_loses_to_finite_db_when_silhouette_tied(self):
        """Infinite Davies-Bouldin is treated as worst case."""
        a = _make_candidate(silhouette=0.5, davies_bouldin=float("inf"), restart_index=0)
        b = _make_candidate(silhouette=0.5, davies_bouldin=5.0, restart_index=1)
        # b has finite DB so should win despite higher restart_index
        result = _pick_better(a, b)
        assert result is b

    def test_symmetry_equal_candidates(self):
        """When both candidates are identical in scores, the first arg wins (<=)."""
        a = _make_candidate(silhouette=0.5, davies_bouldin=1.0, restart_index=0)
        b = _make_candidate(silhouette=0.5, davies_bouldin=1.0, restart_index=0)
        # a <= b when equal, so a should be returned
        result = _pick_better(a, b)
        assert result is a

    def test_selection_key_order_silhouette_dominates_db_and_index(self):
        """selection_key correctly encodes the three-level ordering."""
        high_sil = _make_candidate(silhouette=0.9, davies_bouldin=10.0, restart_index=99)
        low_sil = _make_candidate(silhouette=0.1, davies_bouldin=0.1, restart_index=0)
        # high_sil wins despite terrible DB and high restart index
        assert _pick_better(high_sil, low_sil) is high_sil
        assert _pick_better(low_sil, high_sil) is high_sil


# ---------------------------------------------------------------------------
# Tests: _RestartCandidate.selection_key
# ---------------------------------------------------------------------------

class TestRestartCandidateSelectionKey:
    """Unit tests for _RestartCandidate.selection_key()."""

    def test_key_structure(self):
        """selection_key returns a 3-tuple."""
        cand = _make_candidate(silhouette=0.5, davies_bouldin=1.0, restart_index=2)
        key = cand.selection_key()
        assert len(key) == 3

    def test_key_negates_silhouette(self):
        """First element of key is -silhouette for finite silhouette."""
        cand = _make_candidate(silhouette=0.7, davies_bouldin=1.0, restart_index=0)
        key = cand.selection_key()
        assert key[0] == pytest.approx(-0.7)

    def test_key_preserves_db(self):
        """Second element of key is the raw davies_bouldin value."""
        cand = _make_candidate(silhouette=0.5, davies_bouldin=2.3, restart_index=0)
        key = cand.selection_key()
        assert key[1] == pytest.approx(2.3)

    def test_key_preserves_restart_index(self):
        """Third element of key is the restart_index."""
        cand = _make_candidate(silhouette=0.5, davies_bouldin=1.0, restart_index=7)
        key = cand.selection_key()
        assert key[2] == 7

    def test_nan_silhouette_mapped_to_inf(self):
        """NaN silhouette maps to +inf so this candidate sorts last."""
        cand = _make_candidate(silhouette=float("nan"), davies_bouldin=1.0, restart_index=0)
        key = cand.selection_key()
        assert key[0] == float("inf")

    def test_inf_db_preserved(self):
        """Infinite Davies-Bouldin maps to +inf (already worst)."""
        cand = _make_candidate(silhouette=0.5, davies_bouldin=float("inf"), restart_index=0)
        key = cand.selection_key()
        assert key[1] == float("inf")


# ---------------------------------------------------------------------------
# Tests: _run_restarts — determinism under fixed seeds
# ---------------------------------------------------------------------------

class TestRunRestarts:
    """_run_restarts produces identical labels for identical seed sequences."""

    def _make_embeddings(self, n: int = 20, d: int = 16, seed: int = 0) -> np.ndarray:
        """Create reproducible L2-normalised embeddings."""
        rng = np.random.default_rng(seed)
        emb = rng.standard_normal((n, d)).astype(np.float32)
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        return emb / np.where(norms == 0, 1.0, norms)

    def _deterministic_algorithm(
        self, emb: np.ndarray, k: int, seed: int
    ) -> np.ndarray:
        """Deterministic fake algorithm: round-robin assignment seeded by seed.

        Deliberately varies by seed so that different restarts produce
        different (but reproducible) partitions.
        """
        n = emb.shape[0]
        rng = np.random.default_rng(seed)
        # Shuffle and assign round-robin so the result depends on seed
        perm = rng.permutation(n)
        labels = np.empty(n, dtype=np.int32)
        for rank, orig in enumerate(perm):
            labels[orig] = rank % k
        return labels

    def test_same_seed_produces_identical_labels(self):
        """Two calls with the same (emb, k, seed, n_restarts) return identical labels."""
        emb = self._make_embeddings(n=30, d=16)
        labels1 = _run_restarts(self._deterministic_algorithm, emb, k=3, seed=42, n_restarts=5)
        labels2 = _run_restarts(self._deterministic_algorithm, emb, k=3, seed=42, n_restarts=5)
        assert np.array_equal(labels1, labels2)

    def test_different_seed_may_produce_different_labels(self):
        """Different seeds may produce different labels (non-trivial algorithm)."""
        emb = self._make_embeddings(n=30, d=16)
        labels1 = _run_restarts(self._deterministic_algorithm, emb, k=3, seed=0, n_restarts=5)
        labels2 = _run_restarts(self._deterministic_algorithm, emb, k=3, seed=999, n_restarts=5)
        # Not guaranteed to differ, but for this deterministic algorithm they should
        # (at minimum the test is valid — we just check shape and dtype)
        assert labels1.shape == labels2.shape
        assert labels1.dtype == np.int32

    def test_output_shape_and_dtype(self):
        """Output labels have the expected shape and dtype."""
        n = 25
        k = 4
        emb = self._make_embeddings(n=n, d=16)
        labels = _run_restarts(self._deterministic_algorithm, emb, k=k, seed=7, n_restarts=3)
        assert labels.shape == (n,)
        assert labels.dtype == np.int32

    def test_output_values_in_valid_range(self):
        """Output values are all in [0, k-1]."""
        n = 20
        k = 3
        emb = self._make_embeddings(n=n, d=16)
        labels = _run_restarts(self._deterministic_algorithm, emb, k=k, seed=1, n_restarts=4)
        assert np.all(labels >= 0)
        assert np.all(labels < k)

    def test_seed_sequence_is_correct(self):
        """Each restart i receives seed_i = (seed + i) % 2**32."""
        received_seeds = []

        def recording_algorithm(emb: np.ndarray, k: int, seed: int) -> np.ndarray:
            received_seeds.append(seed)
            n = emb.shape[0]
            return np.zeros(n, dtype=np.int32)

        emb = self._make_embeddings(n=10, d=8)
        base_seed = 100
        n_restarts = 5
        _run_restarts(recording_algorithm, emb, k=2, seed=base_seed, n_restarts=n_restarts)

        expected_seeds = [(base_seed + i) % (2**32) for i in range(n_restarts)]
        assert received_seeds == expected_seeds

    def test_seed_wraparound_at_2pow32(self):
        """Seed wraps correctly at 2**32 - 1."""
        received_seeds = []

        def recording_algorithm(emb: np.ndarray, k: int, seed: int) -> np.ndarray:
            received_seeds.append(seed)
            return np.zeros(emb.shape[0], dtype=np.int32)

        emb = self._make_embeddings(n=5, d=8)
        base_seed = 2**32 - 2  # near overflow
        _run_restarts(recording_algorithm, emb, k=2, seed=base_seed, n_restarts=4)

        expected = [
            (base_seed + i) % (2**32) for i in range(4)
        ]
        assert received_seeds == expected

    def test_n_restarts_1_returns_only_result(self):
        """With n_restarts=1 the single restart result is returned."""
        emb = self._make_embeddings(n=12, d=8)
        labels = _run_restarts(self._deterministic_algorithm, emb, k=2, seed=0, n_restarts=1)
        assert labels.shape == (12,)

    def test_best_candidate_selected_by_silhouette(self):
        """_run_restarts selects the restart with the best silhouette score.

        We construct two restarts with known silhouette outcomes: one that
        produces a tight 2-cluster partition (high silhouette) and one that
        lumps everything into one cluster (undefined silhouette -> NaN -> worst).
        """
        n = 30
        d = 4
        # Two tight, well-separated clusters
        rng = np.random.default_rng(0)
        cluster_a = rng.standard_normal((n // 2, d)) + np.array([5.0] * d)
        cluster_b = rng.standard_normal((n // 2, d)) + np.array([-5.0] * d)
        emb = np.vstack([cluster_a, cluster_b]).astype(np.float32)
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        emb = emb / np.where(norms == 0, 1.0, norms)

        good_labels = np.array([0] * (n // 2) + [1] * (n // 2), dtype=np.int32)
        bad_labels = np.zeros(n, dtype=np.int32)  # all same cluster → undefined silhouette

        call_count = [0]

        def alternating_algorithm(emb_arg: np.ndarray, k: int, seed: int) -> np.ndarray:
            # Restart 0 → bad, Restart 1 → good
            idx = call_count[0]
            call_count[0] += 1
            return bad_labels.copy() if idx == 0 else good_labels.copy()

        result = _run_restarts(alternating_algorithm, emb, k=2, seed=0, n_restarts=2)
        # The good partition (restart 1) should win
        assert np.array_equal(result, good_labels)
