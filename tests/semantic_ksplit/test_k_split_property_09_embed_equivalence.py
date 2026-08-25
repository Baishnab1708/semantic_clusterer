"""Embed Equivalence with SemanticClusterer.


With ``embedding_model=None`` (built-in ONNX), ``SemanticKSplit`` and
``SemanticClusterer`` must produce identical embedding arrays for the same
input texts, since both delegate to the same OnnxEmbedder under the hood.
"""

from __future__ import annotations

import pytest

pytest.importorskip("hdbscan", reason="SemanticClusterer requires hdbscan")

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from semantic_clusterer import SemanticClusterer, SemanticKSplit

# ---------------------------------------------------------------------------
# Try to import SemanticClusterer; skip the entire module if hdbscan is not
# available (SemanticClusterer hard-requires hdbscan at construction time).
# SemanticKSplit does NOT require hdbscan, but we need both
# classes to run this comparison test.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Short, printable ASCII words — realistic but fast to embed
_WORD = st.text(
    alphabet=st.characters(whitelist_categories=("Lu", "Ll"), min_codepoint=65),
    min_size=3,
    max_size=20,
)

_SENTENCE = st.lists(_WORD, min_size=3, max_size=12).map(" ".join)

# Small batches: 10–50 distinct sentences, as specified in the task
_TEXTS = st.lists(_SENTENCE, min_size=10, max_size=50).filter(
    lambda ts: len(set(ts)) >= 2  # at least 2 distinct items for k=2
)


# ---------------------------------------------------------------------------
# Property 9
# ---------------------------------------------------------------------------


@settings(max_examples=10, deadline=None)
@given(texts=_TEXTS)
def test_embed_equivalence_with_semantic_clusterer(texts: list[str]) -> None:
    """Embed Equivalence with SemanticClusterer.

    ``SemanticKSplit(k=2).embed(texts)`` must be exactly equal (via
    ``np.array_equal``) to ``SemanticClusterer().embed(texts)`` when both
    use the default ``embedding_model=None`` (built-in ONNX MiniLM-L6-v2).

    Both classes:
    - resolve ``embedding_model=None`` to the same ``OnnxEmbedder`` instance
      type with the same normalisation settings.
    - preprocess texts the same way (lowercase + strip punctuation, no
      deduplication in ``embed``).
    - call ``embedder.embed(valid_texts, batch_size=config.batch_size)``.

    """
    ks_embeddings = SemanticKSplit(k=2).embed(texts)
    sc_embeddings = SemanticClusterer().embed(texts)

    assert np.array_equal(ks_embeddings, sc_embeddings), (
        f"Embed outputs differ for {len(texts)} texts.\n"
        f"SemanticKSplit shape: {ks_embeddings.shape}, "
        f"SemanticClusterer shape: {sc_embeddings.shape}\n"
        f"Max absolute diff: "
        f"{np.max(np.abs(ks_embeddings.astype(np.float64) - sc_embeddings.astype(np.float64)))}"
        if ks_embeddings.shape == sc_embeddings.shape
        else ""
    )
