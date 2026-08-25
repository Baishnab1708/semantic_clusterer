"""Property-based test for SemanticKSplit: Split / Labels Round-Trip.

**Split / Labels Round-Trip**

Asserts that for every cluster ``c in range(k)``:
    ``split(texts)[c] == [texts[i] for i in range(len(texts)) if labels[i] == c]``

where ``labels = split_labels(texts)`` and both calls share the same
``random_state``.

"""

from __future__ import annotations

import hashlib
from typing import List, Sequence

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from semantic_clusterer.k_split import SemanticKSplit

# ---------------------------------------------------------------------------
# Fast deterministic fake embedder (sha256-derived, no ONNX)
# ---------------------------------------------------------------------------


def _sha256_embedding(index: int, dim: int = 64) -> np.ndarray:
    """Build a deterministic L2-normalised float32 vector from an index.

    Algorithm:
    1. Hash the index as bytes with sha256 to get a 32-byte digest.
    2. Interpret the first 8 bytes of the digest as a uint64 seed.
    3. Use that seed to initialise a numpy RNG and draw ``dim``
       standard-normal floats, then L2-normalise.
    """
    digest = hashlib.sha256(str(index).encode()).digest()
    seed = int.from_bytes(digest[:8], byteorder="little")
    rng = np.random.default_rng(seed)
    raw = rng.standard_normal(dim).astype(np.float32)
    norm = float(np.linalg.norm(raw))
    if norm > 0:
        raw /= norm
    return raw


class _Sha256Embedder:
    """Fake embedder returning sha256-derived vectors indexed by position.

    Each call to ``embed`` maps the i-th text to ``_sha256_embedding(i)``.
    This gives deterministic, distinct vectors without any real model.
    """

    def __init__(self, dim: int = 64) -> None:
        self._dim = dim

    def embed(self, texts: Sequence[str], batch_size: int = 64) -> np.ndarray:
        out = np.stack(
            [_sha256_embedding(i, self._dim) for i in range(len(texts))],
            axis=0,
        )
        return out.astype(np.float32)


_FAKE_EMBEDDER = _Sha256Embedder(dim=64)

# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

# Texts: non-empty strings of at least 3 chars to survive preprocessing.
_TEXT_ELEM = st.text(
    alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd"), min_codepoint=32),
    min_size=3,
    max_size=40,
).filter(lambda s: s.strip())

# Generate a list of N distinct non-empty strings.
_TEXTS_STRATEGY = st.integers(min_value=4, max_value=80).flatmap(
    lambda n: st.lists(
        _TEXT_ELEM,
        min_size=n,
        max_size=n,
        unique=True,
    )
)

# Seed for random_state
_SEEDS = st.integers(min_value=0, max_value=2**16 - 1)


def _k_for(texts: List[str]) -> st.SearchStrategy:
    """Return a strategy for k given a texts list."""
    n = len(texts)
    return st.integers(min_value=2, max_value=min(n, 10))


# ---------------------------------------------------------------------------
# Split / Labels Round-Trip
# ---------------------------------------------------------------------------


@given(
    texts=_TEXTS_STRATEGY,
    seed=_SEEDS,
)
@settings(max_examples=50, deadline=None)
def test_property_03_round_trip(texts: List[str], seed: int) -> None:
    """**Split / Labels Round-Trip**

    For every ``c in range(k)``:
        ``split(texts)[c] == [texts[i] for i in range(len(texts)) if labels[i] == c]``

    when both ``split`` and ``split_labels`` share the same ``random_state``.

    """
    # Determine k based on texts length (keep small for speed)
    n = len(texts)
    k = max(2, min(n // 2, 5))

    ks = SemanticKSplit(
        embedding_model=_FAKE_EMBEDDER,
        k=k,
        random_state=seed,
    )

    # Call split_labels and split on the same instance with the same random_state.
    labels = ks.split_labels(texts)
    clusters = ks.split(texts, return_format="simple")

    # Basic shape invariants
    assert len(clusters) == k, (
        f"Expected {k} clusters, got {len(clusters)} for n={n}, seed={seed}"
    )
    assert len(labels) == n, (
        f"Expected labels length {n}, got {len(labels)} for n={n}, seed={seed}"
    )

    # core assertion: round-trip membership
    # clusters[c] must equal the sub-list of texts with labels[i] == c,
    # in original input order.
    for c in range(k):
        expected = [texts[i] for i in range(n) if labels[i] == c]
        actual = clusters[c]
        assert actual == expected, (
            f"Round-trip failed for c={c}, k={k}, n={n}, seed={seed}:\n"
            f"  expected: {expected}\n"
            f"  actual:   {actual}"
        )

    # Concatenation of clusters is a permutation of
    # all valid-row texts.
    all_clustered = [t for cluster in clusters for t in cluster]
    all_valid = [texts[i] for i in range(n) if labels[i] >= 0]
    assert sorted(all_clustered) == sorted(all_valid), (
        f"Concatenation of clusters is not a permutation of valid texts "
        f"for k={k}, n={n}, seed={seed}"
    )

    # Every cluster must be non-empty
    for c in range(k):
        assert len(clusters[c]) >= 1, (
            f"Cluster {c} is empty for k={k}, n={n}, seed={seed}"
        )


@given(
    texts=_TEXTS_STRATEGY,
    seed=_SEEDS,
)
@settings(max_examples=30, deadline=None)
def test_property_03_round_trip_varying_k(texts: List[str], seed: int) -> None:
    """holds for multiple values of k drawn from the same texts list.

    Tests that the round-trip property is not sensitive to the particular k
    value chosen.

    """
    n = len(texts)
    # Try two different k values
    for k in [2, min(n, 4)]:
        if k > n:
            continue

        ks = SemanticKSplit(
            embedding_model=_FAKE_EMBEDDER,
            k=k,
            random_state=seed,
        )

        labels = ks.split_labels(texts)
        clusters = ks.split(texts, return_format="simple")

        assert len(clusters) == k
        assert len(labels) == n

        # Core round-trip check
        for c in range(k):
            expected = [texts[i] for i in range(n) if labels[i] == c]
            assert clusters[c] == expected, (
                f"Round-trip failed for c={c}, k={k}, n={n}, seed={seed}"
            )

        # Permutation check
        all_clustered = [t for cluster in clusters for t in cluster]
        all_valid = [texts[i] for i in range(n) if labels[i] >= 0]
        assert sorted(all_clustered) == sorted(all_valid), (
            f"Concatenation check failed for k={k}, n={n}, seed={seed}"
        )
