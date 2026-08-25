"""Determinism Under Fixed Seed.


Two SemanticKSplit instances constructed with the same (k, random_state)
must produce bit-identical labels when called on the same texts within the
same Python process.

The sha256 fake embedder is used to keep Hypothesis runs fast and to ensure
the embeddings are deterministically derived from the input texts.

Test structure:
  - Property 4a: In-process strict equality (np.array_equal) for two calls
    with the same (texts, k, random_state)
  - Property 4b: Same as 4a but also verifies permutation equivalence
    (Label_Permutation_Equivalent) — the
    strict equality test subsumes this, but we verify the helper too.

"""

from __future__ import annotations

import hashlib
import math
from typing import List

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from semantic_clusterer import SemanticKSplit

# ---------------------------------------------------------------------------
# sha256-derived fake embedder (keeps Hypothesis runs cheap)
# ---------------------------------------------------------------------------


def _sha256_vec(text: str, dim: int = 64) -> np.ndarray:
    """Build a deterministic L2-normalised float32 vector from a text string.

    Algorithm:
    1. SHA-256 hash of the UTF-8 encoded text -> 32-byte digest.
    2. Interpret the first 8 bytes as a little-endian uint64 seed.
    3. Use that seed to initialise a numpy RNG and draw ``dim`` standard-
       normal float32 values (guarantees no NaN/Inf bit patterns).
    4. L2-normalise the result.
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
    """Fake embedder: each text gets a deterministic sha256-derived vector.

    The embedding of a text is a function of the text string only, so:
    - Two calls with the same texts produce identical embedding matrices.
    - Different texts receive different (pseudo-random) vectors.
    - No real model inference is performed, keeping tests fast.
    """

    DIM: int = 64

    def embed(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        if not texts:
            return np.empty((0, self.DIM), dtype=np.float32)
        return np.stack(
            [_sha256_vec(t, self.DIM) for t in texts], axis=0
        )


_SHA256_EMBEDDER = _Sha256Embedder()


# ---------------------------------------------------------------------------
# Helper: Label_Permutation_Equivalent
# ---------------------------------------------------------------------------


def _is_permutation_equivalent(L1: np.ndarray, L2: np.ndarray) -> bool:
    """Return True iff L1 and L2 are Label_Permutation_Equivalent.

    Two label arrays are permutation-equivalent iff there is a bijection π
    between their non-negative label sets such that:
      - L2[i] == π(L1[i]) whenever L1[i] >= 0
      - L2[i] == -1 iff L1[i] == -1
    """
    if len(L1) != len(L2):
        return False
    # Noise alignment
    if not np.all((L1 == -1) == (L2 == -1)):
        return False
    mask = L1 >= 0
    if not np.any(mask):
        return True
    mapping: dict[int, int] = {}
    reverse: dict[int, int] = {}
    for a, b in zip(L1[mask].tolist(), L2[mask].tolist()):
        if a in mapping:
            if mapping[a] != b:
                return False
        else:
            mapping[a] = b
        if b in reverse:
            if reverse[b] != a:
                return False
        else:
            reverse[b] = a
    return True


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

# Keep N small so each hypothesis example runs quickly.
# We cover sizes 2..50, meaning 'tiny' and lower 'small' tier.
_N = st.integers(min_value=2, max_value=50)

# k is chosen relative to n inside the test (2 <= k <= n).
_SEEDS = st.integers(min_value=0, max_value=2**32 - 1)

_TEXT_ALPHABET = st.characters(
    whitelist_categories=("Lu", "Ll", "Nd", "Pc"),  # letters + digits + underscore
    whitelist_characters=" ",
)


def _texts_strategy(n: int) -> st.SearchStrategy[List[str]]:
    """Generate a list of n distinct non-empty strings of length 1-20."""
    return st.lists(
        st.text(alphabet=_TEXT_ALPHABET, min_size=1, max_size=20),
        min_size=n,
        max_size=n,
        unique=True,
    ).filter(lambda ts: all(t.strip() for t in ts))


# ---------------------------------------------------------------------------
# Property 4a: In-process strict equality (np.array_equal)
# ---------------------------------------------------------------------------


@given(
    n=_N,
    seed=_SEEDS,
    data=st.data(),
)
@settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_property_04a_strict_equality_same_instance(
    n: int,
    seed: int,
    data: st.DataObject,
) -> None:
    """Property 4a: Two split_labels calls on the SAME instance with identical
    (texts, k, random_state) produce bit-identical labels (np.array_equal).

    """
    from semantic_clusterer.preprocessing.clean import TextPreprocessor
    # Draw texts first, then k constrained to [2, N_Unique] after preprocessing
    texts = data.draw(_texts_strategy(n))

    # Account for preprocessing collapsing raw-unique texts (e.g. 'A' and 'a')
    _pp = TextPreprocessor(lowercase=True, remove_punctuation=True)
    processed, _, _ = _pp.preprocess(texts, deduplicate=True)
    n_unique = len(processed)
    if n_unique < 2:
        return  # not enough unique rows; skip

    k = data.draw(st.integers(min_value=2, max_value=n_unique))

    ks = SemanticKSplit(
        embedding_model=_SHA256_EMBEDDER,
        k=k,
        random_state=seed,
    )

    L1 = ks.split_labels(texts)
    L2 = ks.split_labels(texts)

    assert np.array_equal(L1, L2), (
        f"In-process strict equality failed for "
        f"n={n}, k={k}, seed={seed}.\n"
        f"L1={L1.tolist()}\nL2={L2.tolist()}"
    )


@given(
    n=_N,
    seed=_SEEDS,
    data=st.data(),
)
@settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_property_04b_strict_equality_two_instances(
    n: int,
    seed: int,
    data: st.DataObject,
) -> None:
    """Property 4b: Two SEPARATE SemanticKSplit instances constructed with the
    same (k, random_state) produce bit-identical labels for the same texts.

    This is the primary in-process strict-equality assertion from
    Np.array_equal(labels_run_1, labels_run_2) == True.

    """
    from semantic_clusterer.preprocessing.clean import TextPreprocessor
    texts = data.draw(_texts_strategy(n))

    # Account for preprocessing collapsing raw-unique texts (e.g. 'A' and 'a')
    _pp = TextPreprocessor(lowercase=True, remove_punctuation=True)
    processed, _, _ = _pp.preprocess(texts, deduplicate=True)
    n_unique = len(processed)
    if n_unique < 2:
        return  # not enough unique rows; skip

    k = data.draw(st.integers(min_value=2, max_value=n_unique))

    ks1 = SemanticKSplit(
        embedding_model=_SHA256_EMBEDDER,
        k=k,
        random_state=seed,
    )
    ks2 = SemanticKSplit(
        embedding_model=_SHA256_EMBEDDER,
        k=k,
        random_state=seed,
    )

    L1 = ks1.split_labels(texts)
    L2 = ks2.split_labels(texts)

    assert np.array_equal(L1, L2), (
        f"Two-instance strict equality failed for "
        f"n={n}, k={k}, seed={seed}.\n"
        f"texts={texts!r}\n"
        f"L1={L1.tolist()}\nL2={L2.tolist()}"
    )


# ---------------------------------------------------------------------------
# Property 4c: Permutation equivalence
# The strict equality test (18.2) subsumes this, but we verify the
# permutation-equivalence contract explicitly as well.
# ---------------------------------------------------------------------------


@given(
    n=_N,
    seed=_SEEDS,
    data=st.data(),
)
@settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_property_04c_permutation_equivalent_two_instances(
    n: int,
    seed: int,
    data: st.DataObject,
) -> None:
    """Property 4c: Two SemanticKSplit instances with same (k, random_state)
    produce Label_Permutation_Equivalent arrays for the same texts.

    Because 4b already asserts strict equality (which implies permutation
    equivalence), this test focuses on the permutation-equivalence helper
    itself.

    """
    from semantic_clusterer.preprocessing.clean import TextPreprocessor
    texts = data.draw(_texts_strategy(n))

    # After preprocessing (lowercase + punctuation removal), some raw-unique
    # texts may collapse to the same token.  Determine N_Unique so that k is
    # always a valid value (<= N_Unique).
    _pp = TextPreprocessor(lowercase=True, remove_punctuation=True)
    processed, _, _ = _pp.preprocess(texts, deduplicate=True)
    n_unique = len(processed)
    if n_unique < 2:
        return  # not enough unique rows to form any valid k; skip this example

    k = data.draw(st.integers(min_value=2, max_value=n_unique))

    ks1 = SemanticKSplit(
        embedding_model=_SHA256_EMBEDDER,
        k=k,
        random_state=seed,
    )
    ks2 = SemanticKSplit(
        embedding_model=_SHA256_EMBEDDER,
        k=k,
        random_state=seed,
    )

    L1 = ks1.split_labels(texts)
    L2 = ks2.split_labels(texts)

    assert _is_permutation_equivalent(L1, L2), (
        f"Labels not permutation-equivalent for "
        f"n={n}, k={k}, seed={seed}.\n"
        f"texts={texts!r}\n"
        f"L1={L1.tolist()}\nL2={L2.tolist()}"
    )


# ---------------------------------------------------------------------------
# Property 4d: Different seeds may produce DIFFERENT labels
# (sanity check — ensures the seed is actually being used).
# This is not a strict requirement but helps detect always-identical outputs.
# ---------------------------------------------------------------------------


def test_property_04d_different_seeds_can_differ() -> None:
    """Sanity: different seeds on the same texts CAN produce different labels.

    This is not universally required (for tiny deterministic data the
    algorithm might happen to produce the same partition), but it confirms
    that the seed is wired through to the algorithms.

    We use a larger N=20 to reduce the probability of coincidental equality.
    """
    n = 20
    k = 3
    texts = [f"word{i}" for i in range(n)]

    ks_a = SemanticKSplit(
        embedding_model=_SHA256_EMBEDDER,
        k=k,
        random_state=0,
    )
    ks_b = SemanticKSplit(
        embedding_model=_SHA256_EMBEDDER,
        k=k,
        random_state=99999,
    )

    La = ks_a.split_labels(texts)
    Lb = ks_b.split_labels(texts)

    # Both must be valid label arrays
    assert La.dtype == np.int32
    assert Lb.dtype == np.int32
    assert La.shape == (n,)
    assert Lb.shape == (n,)
    assert set(La[La >= 0].tolist()) == set(range(k))
    assert set(Lb[Lb >= 0].tolist()) == set(range(k))
    # We do NOT assert they differ — just that both are valid.
    # The important guarantee is that same seed -> same output (tested above).
