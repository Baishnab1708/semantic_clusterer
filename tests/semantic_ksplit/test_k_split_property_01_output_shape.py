"""Property-based test for SemanticKSplit — Output Shape and Coverage.

For all (texts, k, seed) where texts is a list of 2–30 distinct non-empty strings
and 2 <= k <= len(texts), the following must hold:

1. dtype == np.int32
2. shape == (len(texts),)
3. set(labels[labels >= 0]) == set(range(k))
4. Every cluster has >= 1 member
"""

from __future__ import annotations

import hashlib
import string
from typing import List, Tuple

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from semantic_clusterer import SemanticKSplit

# ---------------------------------------------------------------------------
# sha256-based fake embedder
# ---------------------------------------------------------------------------

# Embedding dimension — large enough for stable clustering but small enough
# for fast test runs.
_FAKE_DIM = 64

# Alphabet for generated texts: lowercase letters and digits only.
# Using no spaces or punctuation ensures texts survive TextPreprocessor
# (lowercase=True, remove_punctuation=True) completely unchanged, so
# every generated text maps to a distinct preprocessed string and
# N_Unique == len(texts) is guaranteed.
_SAFE_ALPHABET = string.ascii_lowercase + string.digits


class _Sha256FakeEmbedder:
    """Deterministic fake embedder using sha256 hash of text content.

    Each text maps to an L2-normalised float32 vector derived from the
    sha256 digest of its UTF-8 encoding.  Two different texts produce
    different digests (with overwhelming probability) and therefore
    different embedding vectors, giving the clustering algorithm non-trivial
    structure to partition.
    """

    def __init__(self, dim: int = _FAKE_DIM) -> None:
        self._dim = dim

    def embed(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        """Return an (N, dim) float32 embedding matrix."""
        return np.stack([self._embed_one(t) for t in texts], axis=0)

    def _embed_one(self, text: str) -> np.ndarray:
        """Hash text -> uint64 seed -> deterministic unit vector."""
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        # Use first 8 bytes of the digest as a uint64 RNG seed.
        seed = int.from_bytes(digest[:8], byteorder="little")
        rng = np.random.default_rng(seed)
        raw = rng.standard_normal(self._dim).astype(np.float32)
        norm = float(np.linalg.norm(raw))
        if norm > 0:
            raw /= norm
        return raw


# ---------------------------------------------------------------------------
# Hypothesis composite strategy
# ---------------------------------------------------------------------------


@st.composite
def _text_list_k_seed(
    draw: st.DrawFn,
) -> Tuple[List[str], int, int]:
    """Generate a valid (texts, k, seed) triple.

    - texts: list of 2–30 distinct non-empty strings over [a-z0-9]
    - k:     integer in [2, len(texts)]
    - seed:  integer in [0, 2**32 - 1]
    """
    texts: List[str] = draw(
        st.lists(
            st.text(alphabet=_SAFE_ALPHABET, min_size=1),
            min_size=2,
            max_size=30,
            unique=True,
        )
    )
    k: int = draw(st.integers(min_value=2, max_value=len(texts)))
    seed: int = draw(st.integers(min_value=0, max_value=2**32 - 1))
    return texts, k, seed


# ---------------------------------------------------------------------------
# Property test
# ---------------------------------------------------------------------------


@given(_text_list_k_seed())
@settings(max_examples=100, deadline=None)
def test_output_shape_and_coverage(args: Tuple[List[str], int, int]) -> None:
    """Output Shape and Coverage.

    11.5, 17.1, 17.2, 17.3, 19.2, 21.1, 21.2**

    Asserts for every generated (texts, k, seed):
    1. labels.dtype == np.int32
    2. labels.shape == (len(texts),)
    3. set(labels[labels >= 0]) == set(range(k))
    4. Every cluster c in range(k) has at least one member
    """
    texts, k, seed = args

    ks = SemanticKSplit(
        embedding_model=_Sha256FakeEmbedder(),
        k=k,
        random_state=seed,
    )

    try:
        labels = ks.split_labels(texts)
    except Exception as exc:
        raise AssertionError(
            f"split_labels raised unexpectedly.\n"
            f"  Counterexample: texts={texts!r}, k={k!r}, seed={seed!r}\n"
            f"  Exception: {type(exc).__name__}: {exc}"
        ) from exc

    _ce = f"Counterexample: texts={texts!r}, k={k!r}, seed={seed!r}"

    # ---- Assertion 1: dtype == np.int32 ------------------------------------
    assert labels.dtype == np.int32, (
        f"Expected dtype=np.int32, got {labels.dtype} | {_ce}"
    )

    # ---- Assertion 2: shape == (len(texts),) --------------------------------
    assert labels.shape == (len(texts),), (
        f"Expected shape=({len(texts)},), got {labels.shape} | {_ce}"
    )

    # ---- Assertion 3: set of valid labels == set(range(k)) -----------------
    # All inputs are valid strings (non-None, non-empty) so all labels must
    # be >= 0 and exactly cover [0, k-1].
    valid_labels = labels[labels >= 0]
    label_set = set(int(x) for x in valid_labels)
    assert label_set == set(range(k)), (
        f"Expected label set == set(range({k})), got {label_set} | {_ce}"
    )

    # ---- Assertion 4: every cluster has >= 1 member -------------------------
    for c in range(k):
        count = int((labels == c).sum())
        assert count >= 1, (
            f"Cluster {c} is empty (0 members) | {_ce}"
        )
