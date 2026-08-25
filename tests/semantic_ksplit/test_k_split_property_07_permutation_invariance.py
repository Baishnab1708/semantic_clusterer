"""Permutation Invariance of Cluster Membership


For any (texts, k, seed, sigma) where sigma is a permutation of
range(len(texts)), the multiset of frozensets of cluster members must be
identical whether we split the original texts or the permuted texts:

    Counter(frozenset(c) for c in A) == Counter(frozenset(c) for c in B)

where:
    A = SemanticKSplit(k=k, random_state=seed).split(texts)
    B = SemanticKSplit(k=k, random_state=seed).split([texts[i] for i in sigma])

A sha256-based fake embedder is used so that tests run cheaply without the
real ONNX model. The embedder is *content-based* (maps text -> deterministic
vector), which is what makes permutation invariance meaningful: the same
text always gets the same embedding regardless of its position in the input.

Note: Permutation invariance is guaranteed only when the clustering algorithm
is purely distance-based (all algorithms except the identical-embeddings
round-robin and rare degenerate paths). We generate distinct non-empty texts
to sidestep those paths.

"""

from __future__ import annotations

import hashlib
from collections import Counter
from typing import List

import numpy as np
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from semantic_clusterer import SemanticKSplit


# ---------------------------------------------------------------------------
# SHA256-based content-aware fake embedder
# ---------------------------------------------------------------------------

def _sha256_embedding(text: str, dim: int = 64) -> np.ndarray:
    """Map a text string to a deterministic L2-normalised float32 vector.

    The embedding is a function of the *content* of the text (not its position),
    so that permuting the input list does not change individual embeddings.

    Algorithm:
    1. SHA-256 hash the UTF-8 encoded text.
    2. Use the first 8 bytes as a uint64 seed for a numpy RNG.
    3. Draw `dim` standard-normal float32 values, then L2-normalise.
    """
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], byteorder="little")
    rng = np.random.default_rng(seed)
    raw = rng.standard_normal(dim).astype(np.float32)
    norm = float(np.linalg.norm(raw))
    if norm > 0:
        raw /= norm
    return raw


class _Sha256Embedder:
    """Content-based fake embedder using sha256-derived vectors."""

    def __init__(self, dim: int = 64) -> None:
        self._dim = dim

    def embed(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        return np.stack(
            [_sha256_embedding(t, self._dim) for t in texts], axis=0
        )


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Generate a fixed-format distinct text using an integer index.
# "word_{i}_topic_{j}" is guaranteed distinct for distinct (i, j) pairs,
# avoiding any filtering overhead.
@st.composite
def _permutation_invariance_inputs(draw):
    """Draw (texts, k, seed, sigma).

    - texts: list of N distinct non-empty strings (4 <= N <= 20)
    - k: integer in [2, N]
    - seed: integer in [0, 2**16 - 1] (smaller range for speed)
    - sigma: a permutation of range(N)
    """
    n = draw(st.integers(min_value=4, max_value=20))
    # Generate N distinct strings using a set of N distinct integers as seeds.
    # This avoids expensive filter rejections.
    indices = draw(
        st.lists(
            st.integers(min_value=0, max_value=10_000),
            min_size=n,
            max_size=n,
            unique=True,
        )
    )
    # Produce distinct human-readable strings from the indices
    texts = [f"document word{idx} topic{idx % 7} phrase{idx % 13}" for idx in indices]

    k = draw(st.integers(min_value=2, max_value=n))
    seed = draw(st.integers(min_value=0, max_value=2**16 - 1))
    # Draw a permutation of range(n)
    sigma = draw(st.permutations(list(range(n))))
    return texts, k, seed, sigma


# ---------------------------------------------------------------------------
# Property test
# ---------------------------------------------------------------------------

@given(inputs=_permutation_invariance_inputs())
@settings(max_examples=30, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_permutation_invariance_of_cluster_membership(inputs):
    """

    Permutation Invariance of Cluster Membership.

    For a fixed (texts, k, seed) and any permutation sigma of range(len(texts)):
        Counter(frozenset(c) for c in A) == Counter(frozenset(c) for c in B)
    where:
        A = split(texts)
        B = split([texts[i] for i in sigma])

    Both A and B are obtained from SemanticKSplit(k=k, random_state=seed)
    using the sha256 content-based fake embedder, so the embeddings depend
    only on text content and not on input order.

    Failure report includes (texts, k, seed, sigma) to aid reproduction.
    """
    texts, k, seed, sigma = inputs

    embedder = _Sha256Embedder(dim=64)

    ks = SemanticKSplit(embedding_model=embedder, k=k, random_state=seed)

    # Compute A: split the original texts
    A = ks.split(texts)

    # Compute B: split the permuted texts
    permuted_texts = [texts[i] for i in sigma]
    B = ks.split(permuted_texts)

    # Both calls must return exactly k clusters
    assert len(A) == k, (
        f"A has {len(A)} clusters, expected {k}. "
        f"texts={texts!r}, k={k}, seed={seed}, sigma={sigma}"
    )
    assert len(B) == k, (
        f"B has {len(B)} clusters, expected {k}. "
        f"texts={texts!r}, k={k}, seed={seed}, sigma={sigma}"
    )

    # Permutation invariance: the multiset of frozensets must match
    multiset_A = Counter(frozenset(c) for c in A)
    multiset_B = Counter(frozenset(c) for c in B)

    assert multiset_A == multiset_B, (
        f"Permutation invariance violated.\n"
        f"  texts  = {texts!r}\n"
        f"  k      = {k}\n"
        f"  seed   = {seed}\n"
        f"  sigma  = {sigma}\n"
        f"  A      = {A!r}\n"
        f"  B      = {B!r}\n"
        f"  Counter(A) = {multiset_A}\n"
        f"  Counter(B) = {multiset_B}"
    )
