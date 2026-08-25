"""Property-based tests for SemanticKSplit report field population.

Report Field Population

For every generated input, assert all required report fields are correctly
populated by split_with_report:
  - chosen_params: requested_k, algorithm_used, pipeline_tier, dim_band
  - intrinsic_metrics: silhouette, davies_bouldin, per_cluster_size (length k),
                       per_cluster_cohesion (length k)
  - report-level: n_clusters == k, n_noise == (labels == -1).sum(),
                  pipeline_tier is valid, random_state in report

"""

from __future__ import annotations

import hashlib
from typing import List, Sequence

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from semantic_clusterer.k_split import SemanticKSplit

# ---------------------------------------------------------------------------
# Sha256-derived deterministic fake embedder (no ONNX)
# ---------------------------------------------------------------------------

_VALID_ALGORITHMS = frozenset({
    "agglomerative-cut-k",
    "bisecting-kmeans",
    "spherical-kmeans",
    "spectral-cosine",
    "constrained-kmeans",
    "balanced-kmeans",
    "minibatch-kmeans-assign",
    "identical-embeddings-tiebreak",
})

_VALID_TIERS = frozenset({"tiny", "small", "medium", "large"})
_VALID_DIM_BANDS = frozenset({"low", "mid", "high", "xhigh"})


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
    """Fake embedder returning sha256-derived position-indexed vectors."""

    def __init__(self, dim: int = 64) -> None:
        self._dim = dim

    def embed(self, texts: Sequence[str], batch_size: int = 64) -> np.ndarray:
        return np.stack(
            [_sha256_embedding(i, self._dim) for i in range(len(texts))],
            axis=0,
        ).astype(np.float32)


_FAKE_EMBEDDER = _Sha256Embedder(dim=64)

# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

# N in [k, k+50] where k is in [2, 10]; keep small for speed
_K_ST = st.integers(min_value=2, max_value=6)
_SEED_ST = st.integers(min_value=0, max_value=2**16 - 1)


def _texts_strategy(n: int) -> List[str]:
    """Generate n distinct non-empty text strings."""
    return [f"unique document sentence number {i}" for i in range(n)]


# ---------------------------------------------------------------------------
# Report Field Population
# ---------------------------------------------------------------------------


@given(
    k=_K_ST,
    extra=st.integers(min_value=0, max_value=20),
    seed=_SEED_ST,
)
@settings(max_examples=30, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_property_06_report_field_population(k: int, extra: int, seed: int) -> None:
    """All required report fields are correctly populated.


    For every generated (texts, k, seed), after calling split_with_report:
    - chosen_params contains 'requested_k' == k            (Req 11.1)
    - chosen_params contains 'algorithm_used' ∈ valid set   (Req 11.2)
    - intrinsic_metrics contains 'silhouette'               (Req 11.3)
    - intrinsic_metrics contains 'davies_bouldin'           (Req 11.3)
    - intrinsic_metrics contains 'per_cluster_size' of len k (Req 11.4)
    - intrinsic_metrics contains 'per_cluster_cohesion' of len k (Req 11.4)
    - report.n_clusters == k                                (Req 11.5)
    - report.n_noise == (labels == -1).sum()                (Req 11.6)
    - report.pipeline_tier ∈ {'tiny','small','medium','large'} (Req 11.7)
    - chosen_params contains 'dim_band'                     (Req 15.6)
    - report.random_state == seed                           (Req 10.3)
    """
    n = k + extra  # n >= k always
    texts = _texts_strategy(n)

    ks = SemanticKSplit(
        embedding_model=_FAKE_EMBEDDER,
        k=k,
        random_state=seed,
    )
    labels, report = ks.split_with_report(texts)

    chosen = report.chosen_params
    metrics = report.intrinsic_metrics

    # --- Requested_k in chosen_params ---
    assert "requested_k" in chosen, (
        f"chosen_params missing 'requested_k' for k={k}, n={n}, seed={seed}"
    )
    assert chosen["requested_k"] == k, (
        f"requested_k={chosen['requested_k']!r} != k={k} for n={n}, seed={seed}"
    )

    # --- Algorithm_used in chosen_params, valid literal ---
    assert "algorithm_used" in chosen, (
        f"chosen_params missing 'algorithm_used' for k={k}, n={n}, seed={seed}"
    )
    assert chosen["algorithm_used"] in _VALID_ALGORITHMS, (
        f"algorithm_used={chosen['algorithm_used']!r} not in valid set "
        f"for k={k}, n={n}, seed={seed}"
    )

    # --- Silhouette and davies_bouldin in intrinsic_metrics ---
    assert "silhouette" in metrics, (
        f"intrinsic_metrics missing 'silhouette' for k={k}, n={n}, seed={seed}"
    )
    assert "davies_bouldin" in metrics, (
        f"intrinsic_metrics missing 'davies_bouldin' for k={k}, n={n}, seed={seed}"
    )

    # Silhouette must be a finite float (or NaN in degenerate cases, but not missing)
    sil = metrics["silhouette"]
    assert isinstance(sil, (int, float)), (
        f"silhouette must be numeric, got {type(sil).__name__} for k={k}, n={n}"
    )

    db = metrics["davies_bouldin"]
    assert isinstance(db, (int, float)), (
        f"davies_bouldin must be numeric, got {type(db).__name__} for k={k}, n={n}"
    )

    # --- Per_cluster_size list of length k ---
    assert "per_cluster_size" in metrics, (
        f"intrinsic_metrics missing 'per_cluster_size' for k={k}, n={n}, seed={seed}"
    )
    pcs = metrics["per_cluster_size"]
    assert isinstance(pcs, list), (
        f"per_cluster_size must be a list, got {type(pcs).__name__} for k={k}"
    )
    assert len(pcs) == k, (
        f"per_cluster_size length={len(pcs)} != k={k} for n={n}, seed={seed}"
    )
    # All sizes must be non-negative integers
    for c, sz in enumerate(pcs):
        assert isinstance(sz, int) and sz >= 0, (
            f"per_cluster_size[{c}]={sz!r} is not a non-negative int for k={k}"
        )

    # --- Per_cluster_cohesion list of length k ---
    assert "per_cluster_cohesion" in metrics, (
        f"intrinsic_metrics missing 'per_cluster_cohesion' for k={k}, n={n}, seed={seed}"
    )
    pcc = metrics["per_cluster_cohesion"]
    assert isinstance(pcc, list), (
        f"per_cluster_cohesion must be a list, got {type(pcc).__name__} for k={k}"
    )
    assert len(pcc) == k, (
        f"per_cluster_cohesion length={len(pcc)} != k={k} for n={n}, seed={seed}"
    )
    # All cohesion values must be floats
    for c, coh in enumerate(pcc):
        assert isinstance(coh, (int, float)), (
            f"per_cluster_cohesion[{c}]={coh!r} is not numeric for k={k}"
        )

    # --- N_clusters == k ---
    assert report.n_clusters == k, (
        f"report.n_clusters={report.n_clusters} != k={k} for n={n}, seed={seed}"
    )

    # --- N_noise == (labels == -1).sum() ---
    expected_noise = int((labels == -1).sum())
    assert report.n_noise == expected_noise, (
        f"report.n_noise={report.n_noise} != (labels==-1).sum()={expected_noise} "
        f"for k={k}, n={n}, seed={seed}"
    )

    # --- Pipeline_tier is a valid tier string ---
    assert report.pipeline_tier in _VALID_TIERS, (
        f"report.pipeline_tier={report.pipeline_tier!r} not in valid set "
        f"for k={k}, n={n}, seed={seed}"
    )

    # --- Dim_band is present in chosen_params ---
    assert "dim_band" in chosen, (
        f"chosen_params missing 'dim_band' for k={k}, n={n}, seed={seed}"
    )
    assert chosen["dim_band"] in _VALID_DIM_BANDS, (
        f"dim_band={chosen['dim_band']!r} not in valid set for k={k}, n={n}"
    )

    # --- Random_state in report equals the seed used ---
    assert report.random_state == seed, (
        f"report.random_state={report.random_state} != seed={seed} "
        f"for k={k}, n={n}"
    )


@given(
    k=_K_ST,
    extra=st.integers(min_value=0, max_value=20),
    seed=_SEED_ST,
)
@settings(max_examples=30, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_property_06_per_cluster_size_sums_to_clustered_count(
    k: int, extra: int, seed: int
) -> None:
    """Per-cluster sizes must sum to the total number of clustered (non-noise) rows.

    """
    n = k + extra
    texts = _texts_strategy(n)

    ks = SemanticKSplit(
        embedding_model=_FAKE_EMBEDDER,
        k=k,
        random_state=seed,
    )
    labels, report = ks.split_with_report(texts)

    pcs = report.intrinsic_metrics["per_cluster_size"]
    total_from_sizes = sum(pcs)
    n_clustered = report.n_clustered

    assert total_from_sizes == n_clustered, (
        f"sum(per_cluster_size)={total_from_sizes} != n_clustered={n_clustered} "
        f"for k={k}, n={n}, seed={seed}"
    )


@given(
    k=_K_ST,
    extra=st.integers(min_value=0, max_value=20),
    seed=_SEED_ST,
)
@settings(max_examples=30, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_property_06_pipeline_tier_consistent_with_chosen_params(
    k: int, extra: int, seed: int
) -> None:
    """report.pipeline_tier must equal chosen_params['pipeline_tier'].

    """
    n = k + extra
    texts = _texts_strategy(n)

    ks = SemanticKSplit(
        embedding_model=_FAKE_EMBEDDER,
        k=k,
        random_state=seed,
    )
    _, report = ks.split_with_report(texts)

    tier_top_level = report.pipeline_tier
    tier_chosen = report.chosen_params.get("pipeline_tier")

    assert tier_top_level == tier_chosen, (
        f"report.pipeline_tier={tier_top_level!r} != "
        f"chosen_params['pipeline_tier']={tier_chosen!r} "
        f"for k={k}, n={n}, seed={seed}"
    )
