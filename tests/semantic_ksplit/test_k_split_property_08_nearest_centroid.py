"""Out-of-Sample Nearest-Centroid Assignment.

Validates that ``_assign_to_nearest_centroid`` selects the centroid with the
highest cosine similarity and breaks exact ties by the lowest centroid index,
which matches ``np.argmax`` semantics on L2-normalized vectors.

"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from semantic_clusterer.k_algorithms.minibatch_assign import _assign_to_nearest_centroid

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Dimensions and cluster/row counts: small to keep tests fast
_DIM = st.integers(min_value=2, max_value=64)
_N_CENTROIDS = st.integers(min_value=2, max_value=20)
_N_QUERIES = st.integers(min_value=1, max_value=50)


def _l2_normalize(arr: np.ndarray) -> np.ndarray:
    """L2-normalize each row; fall back to the zero vector if norm is zero."""
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)
    return arr / norms


@st.composite
def _centroids_and_queries(draw):
    """Generate L2-normalised (centroids, query_rows) arrays."""
    dim = draw(_DIM)
    k = draw(_N_CENTROIDS)
    m = draw(_N_QUERIES)
    seed = draw(st.integers(min_value=0, max_value=2**31 - 1))

    rng = np.random.default_rng(seed)
    raw_centroids = rng.standard_normal((k, dim)).astype(np.float64)
    raw_queries = rng.standard_normal((m, dim)).astype(np.float64)

    centroids = _l2_normalize(raw_centroids)
    queries = _l2_normalize(raw_queries)

    return centroids, queries


# ---------------------------------------------------------------------------
# Property 8
# ---------------------------------------------------------------------------


@given(_centroids_and_queries())
@settings(max_examples=100, deadline=None)
def test_property_08_nearest_centroid_assignment(data):
    """

    For all L2-normalised ``(centroids, query_rows)``:
    * ``_assign_to_nearest_centroid(query_rows, centroids)`` must equal
      ``np.argmax(query_rows @ centroids.T, axis=1)`` (argmax of cosine
      similarity, which equals the dot product on unit vectors).
    * Ties are broken by the lowest centroid index because ``np.argmax``
      returns the *first* occurrence of the maximum value.
    """
    centroids, query_rows = data

    # Expected: argmax dot product over L2-normalised vectors == argmax cosine
    # Because both centroids and query_rows are already L2-normalised, the dot
    # product equals the cosine similarity directly.
    expected = np.argmax(query_rows @ centroids.T, axis=1).astype(np.int32)

    actual = _assign_to_nearest_centroid(query_rows, centroids)

    assert actual.dtype == np.int32, (
        f"Expected dtype int32, got {actual.dtype}"
    )
    assert actual.shape == (query_rows.shape[0],), (
        f"Expected shape ({query_rows.shape[0]},), got {actual.shape}"
    )
    np.testing.assert_array_equal(
        actual,
        expected,
        err_msg=(
            f"Assignment mismatch.\n"
            f"centroids shape: {centroids.shape}\n"
            f"query_rows shape: {query_rows.shape}\n"
            f"expected: {expected}\n"
            f"actual:   {actual}"
        ),
    )
