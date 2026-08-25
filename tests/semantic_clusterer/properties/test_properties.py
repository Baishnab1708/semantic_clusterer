"""Property-based tests for semantic-clusterer v0.1.0 (P1–P8).

Each test is decorated with @given and validates a formal correctness property.
Tests are gated on small synthetic inputs so they run quickly without fixtures.

"""

from __future__ import annotations

import json
import math
from typing import List

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from semantic_clusterer import ClustererConfig, SemanticClusterer
from semantic_clusterer.report import ClusteringReport

# ---------------------------------------------------------------------------
# Shared strategies and helpers
# ---------------------------------------------------------------------------

# Small N range so tests run fast; covers tiny (≤150) and small (151–5000) tiers
_N_TINY = st.integers(min_value=4, max_value=60)
_N_SMALL = st.integers(min_value=151, max_value=300)
_DIM = 32  # small dim for speed; still in "low" band after resolve_dim_band fallback

_SEEDS = st.integers(min_value=0, max_value=2**16 - 1)


def _make_clustered_embeddings(
    n: int,
    n_clusters: int,
    dim: int,
    seed: int,
    noise_frac: float = 0.10,
) -> np.ndarray:
    """Generate L2-normalised float32 embeddings with mild cluster structure."""
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((n_clusters, dim)).astype(np.float32)
    norms = np.linalg.norm(centers, axis=1, keepdims=True)
    centers /= np.where(norms == 0, 1.0, norms)

    out = np.empty((n, dim), dtype=np.float32)
    for i in range(n):
        if rng.random() < noise_frac:
            v = rng.standard_normal(dim).astype(np.float32)
        else:
            c = centers[i % n_clusters]
            v = c + rng.standard_normal(dim).astype(np.float32) * 0.15
        norm = float(np.linalg.norm(v))
        out[i] = v / max(norm, 1e-8)
    return out


def _make_stub_embedder(embeddings: np.ndarray):
    """Return a callable embedder that returns slices of a fixed embedding matrix."""
    class _Stub:
        def embed(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
            n = len(texts)
            if n <= len(embeddings):
                return embeddings[:n].copy()
            # Tile if more texts than embeddings
            reps = math.ceil(n / len(embeddings))
            tiled = np.tile(embeddings, (reps, 1))[:n]
            return tiled.copy()
    return _Stub()


def _texts(n: int) -> List[str]:
    return [f"text sample number {i}" for i in range(n)]


def _is_permutation_equivalent(L1: np.ndarray, L2: np.ndarray) -> bool:
    """Check if two label arrays are permutation-equivalent (ignoring noise)."""
    if len(L1) != len(L2):
        return False
    mask = (L1 >= 0) & (L2 >= 0)
    if not np.any(mask):
        return True
    # Build bijection
    mapping: dict[int, int] = {}
    for a, b in zip(L1[mask], L2[mask]):
        a, b = int(a), int(b)
        if a in mapping:
            if mapping[a] != b:
                return False
        else:
            mapping[a] = b
    # Check noise alignment
    noise_mask = L1 == -1
    if not np.all(L2[noise_mask] == -1):
        return False
    return True


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

# Property 1
@given(n=_N_TINY, seed=_SEEDS)
@settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_p1_determinism(n: int, seed: int) -> None:
    """Two runs with identical (embeddings, random_state) produce permutation-equivalent labels."""
    embeddings = _make_clustered_embeddings(n, n_clusters=min(4, n // 2), dim=_DIM, seed=seed)
    texts = _texts(n)
    embedder = _make_stub_embedder(embeddings)

    c1 = SemanticClusterer(embedding_model=embedder, random_state=42)
    c2 = SemanticClusterer(embedding_model=embedder, random_state=42)

    L1 = c1.cluster_labels(texts)
    L2 = c2.cluster_labels(texts)

    assert L1.shape == L2.shape == (n,)
    assert L1.dtype == L2.dtype == np.int32
    assert _is_permutation_equivalent(L1, L2), (
        f"Labels not permutation-equivalent for n={n}, seed={seed}"
    )


# ---------------------------------------------------------------------------
# Embedding quality monotonicity
# ---------------------------------------------------------------------------

# Property 2
@given(n=_N_TINY, seed=_SEEDS)
@settings(max_examples=30, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_p2_embedding_quality_monotonicity(n: int, seed: int) -> None:
    """Stronger embedder (tighter clusters) yields score >= default - 0.06."""
    from tests.conftest import StubDeterministicEmbedder, StubStrongerEmbedder

    texts = _texts(n)

    default_embedder = StubDeterministicEmbedder(seed=seed)
    stronger_embedder = StubStrongerEmbedder(seed=seed + 1, n_clusters=min(4, n // 2))

    c_default = SemanticClusterer(embedding_model=default_embedder, random_state=42)
    c_stronger = SemanticClusterer(embedding_model=stronger_embedder, random_state=42)

    _, r_default = c_default.cluster_with_report(texts)
    _, r_stronger = c_stronger.cluster_with_report(texts)

    score_default = float(r_default.intrinsic_metrics.get("score", 0.0))
    score_stronger = float(r_stronger.intrinsic_metrics.get("score", 0.0))

    # Stronger model score must be >= default - 0.06
    assert score_stronger >= score_default - 0.06, (
        f"Stronger embedder score {score_stronger:.4f} < default {score_default:.4f} - 0.06"
    )


# ---------------------------------------------------------------------------
# Label range invariant
# ---------------------------------------------------------------------------

# Property 3
@given(n=_N_TINY, seed=_SEEDS)
@settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_p3_label_range_invariant(n: int, seed: int) -> None:
    """Labels are int32, shape (n,), values in {-1} ∪ {0..K-1} with no gaps."""
    embeddings = _make_clustered_embeddings(n, n_clusters=min(4, n // 2), dim=_DIM, seed=seed)
    texts = _texts(n)
    embedder = _make_stub_embedder(embeddings)

    c = SemanticClusterer(embedding_model=embedder, random_state=42)
    labels = c.cluster_labels(texts)

    assert labels.shape == (n,), f"Expected shape ({n},), got {labels.shape}"
    assert labels.dtype == np.int32, f"Expected int32, got {labels.dtype}"

    valid = labels[labels >= 0]
    if valid.size > 0:
        unique = np.unique(valid)
        expected = np.arange(len(unique))
        assert np.array_equal(unique, expected), (
            f"Non-contiguous labels: {unique.tolist()}"
        )
    # All values must be in {-1} ∪ {0..K-1}
    assert np.all(labels >= -1), "Labels contain values < -1"


# ---------------------------------------------------------------------------
# No all-noise floor
# ---------------------------------------------------------------------------

# Property 4
@given(n=_N_TINY, seed=_SEEDS)
@settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_p4_no_all_noise(n: int, seed: int) -> None:
    """When N >= 2, noise ratio must be <= 0.85 (no all-noise result)."""
    embeddings = _make_clustered_embeddings(n, n_clusters=min(4, n // 2), dim=_DIM, seed=seed)
    texts = _texts(n)
    embedder = _make_stub_embedder(embeddings)

    c = SemanticClusterer(embedding_model=embedder, random_state=42)
    labels = c.cluster_labels(texts)

    noise_ratio = float(np.mean(labels == -1))
    assert noise_ratio <= 0.85, (
        f"Noise ratio {noise_ratio:.3f} > 0.85 for n={n}, seed={seed}"
    )


# ---------------------------------------------------------------------------
# Tier boundary continuity
# ---------------------------------------------------------------------------

# Property 5
@given(seed=_SEEDS)
@settings(max_examples=20, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_p5_tier_boundary_continuity(seed: int) -> None:
    """Score difference across tier boundary B=150 with delta=1 must be <= 0.10."""
    B = 150
    delta = 1
    n1, n2 = B - delta, B + delta

    embeddings = _make_clustered_embeddings(
        max(n1, n2), n_clusters=5, dim=_DIM, seed=seed
    )
    texts_n1 = _texts(n1)
    texts_n2 = _texts(n2)

    embedder1 = _make_stub_embedder(embeddings[:n1])
    embedder2 = _make_stub_embedder(embeddings[:n2])

    c1 = SemanticClusterer(embedding_model=embedder1, random_state=42)
    c2 = SemanticClusterer(embedding_model=embedder2, random_state=42)

    _, r1 = c1.cluster_with_report(texts_n1)
    _, r2 = c2.cluster_with_report(texts_n2)

    s1 = float(r1.intrinsic_metrics.get("score", 0.0))
    s2 = float(r2.intrinsic_metrics.get("score", 0.0))

    assert abs(s1 - s2) <= 0.10, (
        f"Score cliff at B={B}, delta={delta}: s({n1})={s1:.4f}, s({n2})={s2:.4f}, "
        f"diff={abs(s1-s2):.4f} > 0.10"
    )


# ---------------------------------------------------------------------------
# Report JSON round-trip
# ---------------------------------------------------------------------------

# Property 6
@given(n=_N_TINY, seed=_SEEDS)
@settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_p6_report_json_round_trip(n: int, seed: int) -> None:
    """report.to_dict() must be JSON-serialisable and round-trip stable."""
    embeddings = _make_clustered_embeddings(n, n_clusters=min(4, n // 2), dim=_DIM, seed=seed)
    texts = _texts(n)
    embedder = _make_stub_embedder(embeddings)

    c = SemanticClusterer(embedding_model=embedder, random_state=42)
    _, report = c.cluster_with_report(texts)

    d = report.to_dict()
    serialised = json.dumps(d)  # must not raise
    round_tripped = json.loads(serialised)
    assert round_tripped == d, "JSON round-trip changed the dict"


# ---------------------------------------------------------------------------
# Strict mode hard limit
# ---------------------------------------------------------------------------

# Property 7
@given(extra=st.integers(min_value=1, max_value=100))
@settings(max_examples=20, deadline=None)
def test_p7_strict_mode_hard_limit(extra: int) -> None:
    """N > 200_000 with allow_oversized_datasets=False must raise ValueError."""
    n = 200_000 + extra
    c = SemanticClusterer(
        config=ClustererConfig(allow_oversized_datasets=False),
        random_state=42,
    )
    texts = [f"t{i}" for i in range(n)]
    with pytest.raises(ValueError) as exc_info:
        c.cluster_labels(texts)
    msg = str(exc_info.value)
    assert "200_000" in msg or "200000" in msg, (
        f"ValueError message does not mention 200_000: {msg!r}"
    )


# ---------------------------------------------------------------------------
# Idempotent compaction
# ---------------------------------------------------------------------------

# Property 8
@given(n=_N_TINY, seed=_SEEDS)
@settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_p8_idempotent_compaction(n: int, seed: int) -> None:
    """Labels must form the contiguous set {0..K-1} with no gaps (idempotent compaction)."""
    embeddings = _make_clustered_embeddings(n, n_clusters=min(4, n // 2), dim=_DIM, seed=seed)
    texts = _texts(n)
    embedder = _make_stub_embedder(embeddings)

    c = SemanticClusterer(embedding_model=embedder, random_state=42)
    labels = c.cluster_labels(texts)

    valid = labels[labels >= 0]
    if valid.size > 0:
        unique_sorted = np.sort(np.unique(valid))
        expected = np.arange(len(unique_sorted))
        assert np.array_equal(unique_sorted, expected), (
            f"Labels not compacted: {unique_sorted.tolist()}"
        )
