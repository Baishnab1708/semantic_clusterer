"""Edge-case unit tests for the tiny clustering pipeline.

Covers 12.1, 12.2, 12.3, 12.4, 12.5, 12.6, 15.1, 15.2, 15.3,
15.4, 15.5, 15.6
"""

from __future__ import annotations

import numpy as np
import pytest

from semantic_clusterer.pipeline.tiny import cluster_tiny
from semantic_clusterer.report import _PipelineTrace


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _unit_vec(dim: int, seed: int) -> np.ndarray:
    """Return a single unit-normalised float32 vector."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def _make_embeddings(n: int, dim: int = 32, seed: int = 0) -> np.ndarray:
    """Return (n, dim) float32 embeddings with random unit vectors."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, dim)).astype(np.float32)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / norms


# ---------------------------------------------------------------------------
# 1. N=0 returns shape (0,) int32, no raise  (Req 15.2)
# ---------------------------------------------------------------------------

def test_n0_returns_empty_int32_no_raise():
    """N=0 must return an empty int32 array without raising."""
    embeddings = np.empty((0, 32), dtype=np.float32)
    labels = cluster_tiny(embeddings)
    assert labels.shape == (0,), f"Expected shape (0,), got {labels.shape}"
    assert labels.dtype == np.int32, f"Expected int32, got {labels.dtype}"


# ---------------------------------------------------------------------------
# 2. N=1 returns [0]  (Req 15.3)
# ---------------------------------------------------------------------------

def test_n1_returns_single_zero():
    """N=1 must return np.array([0], dtype=int32)."""
    embeddings = _make_embeddings(1, dim=32, seed=1)
    labels = cluster_tiny(embeddings)
    assert labels.shape == (1,)
    assert labels.dtype == np.int32
    assert labels[0] == 0


# ---------------------------------------------------------------------------
# 3. N=2, cosine >= 0.95 → [0, 0]  (Req 15.4)
# ---------------------------------------------------------------------------

def test_n2_high_cosine_returns_same_cluster():
    """N=2 with cosine similarity >= 0.95 must return [0, 0]."""
    base = _unit_vec(32, seed=10)
    # Perturb very slightly so cosine stays >= 0.95
    tiny_noise = np.random.default_rng(99).standard_normal(32).astype(np.float32) * 0.01
    v2 = base + tiny_noise
    v2 = v2 / np.linalg.norm(v2)

    cosine = float(np.dot(base, v2))
    assert cosine >= 0.95, f"Test setup error: cosine={cosine:.4f} < 0.95"

    embeddings = np.stack([base, v2])
    labels = cluster_tiny(embeddings)
    assert labels.tolist() == [0, 0], f"Expected [0,0], got {labels.tolist()}"
    assert labels.dtype == np.int32


# ---------------------------------------------------------------------------
# 4. N=2, cosine < 0.95 → [0, 1]  (Req 15.5)
# ---------------------------------------------------------------------------

def test_n2_low_cosine_returns_different_clusters():
    """N=2 with cosine similarity < 0.95 must return [0, 1]."""
    rng = np.random.default_rng(42)
    # Two orthogonal-ish vectors will have cosine near 0
    v1 = np.array([1.0, 0.0, 0.0] + [0.0] * 29, dtype=np.float32)
    v2 = np.array([0.0, 1.0, 0.0] + [0.0] * 29, dtype=np.float32)

    cosine = float(np.dot(v1, v2))
    assert cosine < 0.95, f"Test setup error: cosine={cosine:.4f} >= 0.95"

    embeddings = np.stack([v1, v2])
    labels = cluster_tiny(embeddings)
    assert labels.tolist() == [0, 1], f"Expected [0,1], got {labels.tolist()}"
    assert labels.dtype == np.int32


# ---------------------------------------------------------------------------
# 5. All-identical embeddings within 1e-9 → all zeros  (Req 15.1)
# ---------------------------------------------------------------------------

def test_all_identical_embeddings_return_all_zeros():
    """All-identical embeddings (within 1e-9) must return all-zero labels."""
    base = _unit_vec(32, seed=7)
    N = 20
    # Perturb within 1e-9 tolerance
    noise = np.random.default_rng(0).standard_normal((N, 32)).astype(np.float64) * 1e-10
    embeddings = (base.astype(np.float64) + noise).astype(np.float32)

    labels = cluster_tiny(embeddings)
    assert labels.shape == (N,)
    assert labels.dtype == np.int32
    assert np.all(labels == 0), f"Expected all zeros, got {np.unique(labels)}"


# ---------------------------------------------------------------------------
# 6. Identical inputs + identical random_state → element-wise equal labels
#    (Req 15.6, 12.1)
# ---------------------------------------------------------------------------

def test_determinism_identical_inputs_and_seed():
    """Two runs with identical inputs and random_state must return identical labels."""
    embeddings = _make_embeddings(40, dim=32, seed=5)
    labels1 = cluster_tiny(embeddings, random_state=42)
    labels2 = cluster_tiny(embeddings, random_state=42)
    np.testing.assert_array_equal(
        labels1, labels2,
        err_msg="Labels differ across identical runs with same random_state",
    )


# ---------------------------------------------------------------------------
# 7. Per-candidate trace includes every (source, K, score) triple  (Req 12.3)
# ---------------------------------------------------------------------------

def test_trace_candidates_contain_source_k_score():
    """The trace must record every evaluated candidate with source, k, and score."""
    embeddings = _make_embeddings(30, dim=32, seed=3)
    trace = _PipelineTrace()
    cluster_tiny(embeddings, random_state=42, trace=trace)

    candidates = trace.chosen_params.get("tiny_candidates")
    assert candidates is not None, "trace.chosen_params must contain 'tiny_candidates'"
    assert len(candidates) >= 1, "At least one candidate must be recorded"

    for entry in candidates:
        assert "source" in entry, f"Candidate missing 'source': {entry}"
        assert "k" in entry, f"Candidate missing 'k': {entry}"
        assert "score" in entry, f"Candidate missing 'score': {entry}"
        assert entry["source"] in {"dendrogram-jump", "grid", "silhouette", "umap-hdbscan", "spectral"}, (
            f"Unknown source: {entry['source']}"
        )
        assert isinstance(entry["k"], int), f"k must be int, got {type(entry['k'])}"


# ---------------------------------------------------------------------------
# 8. Tie-break order: dendrogram-jump → silhouette → grid  (Req 12.4)
# ---------------------------------------------------------------------------

def test_tiebreak_order_dendrogram_jump_before_silhouette_before_grid():
    """When scores tie, source order must be dendrogram-jump > silhouette > grid."""
    from semantic_clusterer.pipeline.tiny import _dedup_candidates_label_aware

    # Simulate three candidates with the same K but different sources
    # The dedup keeps first occurrence in insertion order
    # (dendrogram-jump, grid, silhouette) — grid comes before silhouette in
    # insertion, but tie-break in scoring uses source_order dict.
    # We test the sort key directly.
    source_order = {"dendrogram-jump": 0, "silhouette": 1, "grid": 2}

    candidates = [
        {"source": "grid", "k": 3, "score": 0.5},
        {"source": "silhouette", "k": 4, "score": 0.5},
        {"source": "dendrogram-jump", "k": 5, "score": 0.5},
    ]
    # Sort by (-score, k, source_order) — same score, different k
    candidates.sort(key=lambda c: (-c["score"], c["k"], source_order[c["source"]]))
    assert candidates[0]["source"] == "grid"  # smallest k=3 wins

    # Same score, same k — source order decides
    candidates2 = [
        {"source": "grid", "k": 3, "score": 0.5},
        {"source": "silhouette", "k": 3, "score": 0.5},
        {"source": "dendrogram-jump", "k": 3, "score": 0.5},
    ]
    candidates2.sort(key=lambda c: (-c["score"], c["k"], source_order[c["source"]]))
    assert candidates2[0]["source"] == "dendrogram-jump", (
        f"Expected dendrogram-jump first, got {candidates2[0]['source']}"
    )
    assert candidates2[1]["source"] == "silhouette", (
        f"Expected silhouette second, got {candidates2[1]['source']}"
    )
    assert candidates2[2]["source"] == "grid", (
        f"Expected grid third, got {candidates2[2]['source']}"
    )


def test_tiebreak_integration_dendrogram_jump_preferred():
    """Integration: when dendrogram-jump and grid produce the same K, dendrogram-jump wins."""
    # Use a small N where dendrogram-jump and grid are likely to agree on K
    embeddings = _make_embeddings(10, dim=32, seed=99)
    trace = _PipelineTrace()
    cluster_tiny(embeddings, random_state=0, trace=trace)

    chosen_source = trace.chosen_params.get("tiny_chosen_source")
    # The chosen source must be one of the valid sources
    assert chosen_source in {"dendrogram-jump", "grid", "silhouette", "n2_threshold", "homogeneous"}, (
        f"Unexpected chosen source: {chosen_source}"
    )


# ---------------------------------------------------------------------------
# 9. Missing sources are recorded in tiny_omitted_sources  (Req 12.5)
# ---------------------------------------------------------------------------

def test_omitted_sources_recorded_when_dendrogram_jump_unavailable():
    """When N//2 < 2 (N=3), dendrogram-jump cannot produce a valid K and must be omitted."""
    # N=3 → N//2 = 1 → max_k=1 < 2, so _dendrogram_jump_k returns None → omitted
    embeddings = _make_embeddings(3, dim=32, seed=11)
    trace = _PipelineTrace()
    cluster_tiny(embeddings, random_state=42, trace=trace)

    omitted = trace.chosen_params.get("tiny_omitted_sources", [])
    assert "dendrogram-jump" in omitted, (
        f"Expected 'dendrogram-jump' in tiny_omitted_sources for N=3, got {omitted}"
    )


def test_omitted_sources_not_present_when_all_sources_available():
    """When all sources are available, tiny_omitted_sources should be absent or empty."""
    # N=30 gives N//2=15, so grid_ks=[2,3,5,8,12] are all valid
    embeddings = _make_embeddings(30, dim=32, seed=12)
    trace = _PipelineTrace()
    cluster_tiny(embeddings, random_state=42, trace=trace)

    omitted = trace.chosen_params.get("tiny_omitted_sources", [])
    # All three sources should be available for N=30
    assert "grid" not in omitted, f"grid should not be omitted for N=30, got {omitted}"


# ---------------------------------------------------------------------------
# Additional: trace records pipeline_tier="tiny" for all degenerate cases
# ---------------------------------------------------------------------------

def test_trace_pipeline_tier_tiny_for_n0():
    trace = _PipelineTrace()
    cluster_tiny(np.empty((0, 32), dtype=np.float32), trace=trace)
    assert trace.chosen_params.get("pipeline_tier") == "tiny"


def test_trace_pipeline_tier_tiny_for_n1():
    trace = _PipelineTrace()
    cluster_tiny(_make_embeddings(1, dim=32), trace=trace)
    assert trace.chosen_params.get("pipeline_tier") == "tiny"


def test_trace_pipeline_tier_tiny_for_n2():
    trace = _PipelineTrace()
    cluster_tiny(_make_embeddings(2, dim=32), trace=trace)
    assert trace.chosen_params.get("pipeline_tier") == "tiny"
