"""Property-based test for filtered-row label mapping (Property 2).


Property: FOR ALL inputs mixing valid strings with None, NaN, and
empty/whitespace strings, the returned Split_Labels array satisfies:

  1. ``labels[i] == -1``  iff  input row ``i`` is filtered
     (None, NaN, or empty/whitespace after preprocessing)
  2. ``0 <= labels[i] < k``  for every non-filtered row ``i``

Uses ``@given`` with ``@settings(max_examples=50, deadline=None)``.
Uses sha256 fake embedder.
"""

from __future__ import annotations

import hashlib
import math
from typing import List

import numpy as np
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from semantic_clusterer import SemanticKSplit
from semantic_clusterer.preprocessing.clean import TextPreprocessor


# ---------------------------------------------------------------------------
# sha256 fake embedder (same pattern as test_k_split_algorithms.py)
# ---------------------------------------------------------------------------


def _sha256_embedding(index: int, dim: int = 64) -> np.ndarray:
    """Build a deterministic L2-normalised float32 vector from an index.

    Algorithm:
    1. Hash the index as bytes with sha256 to get a 32-byte digest.
    2. Extract the first 8 bytes as a uint64 seed.
    3. Use that seed to initialise a numpy RNG and draw ``dim`` standard-normal
       float32 values (guarantees no NaN/Inf bit patterns).
    4. L2-normalise the result.
    """
    digest = hashlib.sha256(str(index).encode()).digest()
    seed = int.from_bytes(digest[:8], byteorder="little")
    rng = np.random.default_rng(seed)
    raw = rng.standard_normal(dim).astype(np.float32)
    norm = np.linalg.norm(raw)
    if norm > 0:
        raw /= norm
    return raw


class _Sha256Embedder:
    """Embedder returning sha256-derived L2-normalised vectors keyed by position."""

    DIM: int = 64

    def embed(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        return np.stack(
            [_sha256_embedding(i, self.DIM) for i in range(len(texts))],
            axis=0,
        )


# ---------------------------------------------------------------------------
# Helper: determine if an item is filtered by the preprocessor
# ---------------------------------------------------------------------------

_PREPROCESSOR = TextPreprocessor(lowercase=True, remove_punctuation=True)


def _is_filtered(item) -> bool:
    """Return ``True`` if *item* will receive label ``-1`` after preprocessing.

    Mirrors the logic in :class:`~semantic_clusterer.preprocessing.clean.TextPreprocessor`:

    - ``None``          → filtered
    - ``float("nan")``  → filtered
    - ``str`` that becomes empty after cleaning → filtered
    """
    if item is None:
        return True
    if isinstance(item, float) and math.isnan(item):
        return True
    if isinstance(item, str):
        cleaned = _PREPROCESSOR._clean_text(item)
        return cleaned is None or len(cleaned) == 0
    return False


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

# Valid texts: purely lowercase alphanumeric (min 3 chars).
# Already in their final preprocessed form — no punctuation to strip,
# already lowercase — so preprocessing leaves them unchanged.
_VALID_TEXT = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyz0123456789",
    min_size=3,
    max_size=20,
)

# Invalid items covering all three filtered categories from
_INVALID_ITEM = st.one_of(
    st.none(),                  # None
    st.just(float("nan")),      # NaN
    st.just(""),                # empty string
    st.just("   "),             # whitespace-only
    st.just("  \t  "),          # tabs + spaces
)


@st.composite
def _mixed_texts_and_k(draw):
    """Strategy producing ``(texts, k)`` with a mix of valid and invalid items.

    Guarantees:
    - ``k`` in ``[2, 5]`` (tiny tier keeps tests fast)
    - ``n_valid >= k`` unique valid strings survive preprocessing
    - ``n_invalid >= 1`` (at least one filtered row exercises the -1 contract)
    """
    k = draw(st.integers(min_value=2, max_value=5))
    n_valid = draw(st.integers(min_value=k, max_value=k + 10))
    n_invalid = draw(st.integers(min_value=1, max_value=8))

    # Unique valid texts — already lowercase alphanumeric, so unique before
    # preprocessing implies unique after preprocessing.
    valid_texts = draw(
        st.lists(
            _VALID_TEXT,
            min_size=n_valid,
            max_size=n_valid,
            unique=True,
        )
    )

    # Invalid items (duplicates allowed; each still maps to -1)
    invalid_items = draw(
        st.lists(
            _INVALID_ITEM,
            min_size=n_invalid,
            max_size=n_invalid,
        )
    )

    # Combine and shuffle via a drawn permutation
    all_items = list(valid_texts) + list(invalid_items)
    perm = draw(st.permutations(list(range(len(all_items)))))
    texts = [all_items[i] for i in perm]

    return texts, k


# ---------------------------------------------------------------------------
# Property test
# ---------------------------------------------------------------------------


@given(data=_mixed_texts_and_k())
@settings(
    max_examples=50,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
def test_filtered_row_label_mapping(data) -> None:
    """Filtered-Row Label Mapping.


    For every generated input mixing valid and invalid rows:

    1. ``labels[i] == -1``  iff  input row ``i`` is filtered
       (None, NaN, or empty/whitespace — Req 8.3, 19.1)
    2. ``0 <= labels[i] < k``  for every non-filtered row ``i`` (Req 8.4)
    """
    texts, k = data

    # Determine expected filtered/valid status for each position
    filtered_indices = [i for i, t in enumerate(texts) if _is_filtered(t)]
    valid_indices = [i for i, t in enumerate(texts) if not _is_filtered(t)]

    # Count unique valid texts post-preprocessing.  If k > N_Unique, split_labels
    # would raise ValueError (covered by separate tests) — skip with assume().
    unique_valid: set = set()
    for i in valid_indices:
        cleaned = _PREPROCESSOR._clean_text(texts[i])
        if cleaned is not None and len(cleaned) > 0:
            unique_valid.add(cleaned)

    assume(len(unique_valid) >= k)

    embedder = _Sha256Embedder()
    ks = SemanticKSplit(embedding_model=embedder, k=k, random_state=42)
    labels = ks.split_labels(texts)  # type: ignore[arg-type]

    # Shape and dtype sanity
    assert labels.shape == (len(texts),), (
        f"Expected shape ({len(texts)},), got {labels.shape}. k={k}"
    )
    assert labels.dtype == np.int32, (
        f"Expected dtype int32, got {labels.dtype}. k={k}"
    )

    # --- Assertion 1 (Req 8.3, 19.1): filtered rows must carry label -1 ---
    for i in filtered_indices:
        assert labels[i] == -1, (
            f"labels[{i}]={labels[i]} != -1 for filtered item {texts[i]!r}. "
            f"k={k}, texts={texts!r}"
        )

    # --- Assertion 2 (Req 8.4): valid rows must have label in [0, k-1] ---
    for i in valid_indices:
        assert 0 <= labels[i] < k, (
            f"labels[{i}]={labels[i]} not in [0, {k - 1}] for valid "
            f"item {texts[i]!r}. k={k}, texts={texts!r}"
        )
