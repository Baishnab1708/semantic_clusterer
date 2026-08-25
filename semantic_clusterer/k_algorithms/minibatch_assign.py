"""minibatch_assign.py — MiniBatchKMeans-based assignment for the large tier.

Implements ``_minibatch_kmeans_assign`` which runs a
single ``MiniBatchKMeans`` fit and then performs a final hard cosine
assignment.  Also exposes ``_assign_to_nearest_centroid``
used by the oversized out-of-sample step.

"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import MiniBatchKMeans

from semantic_clusterer.utils.similarity import normalize_vectors


# ---------------------------------------------------------------------------
# _minibatch_kmeans_assign
# ---------------------------------------------------------------------------


def _minibatch_kmeans_assign(
    emb: np.ndarray,
    k: int,
    seed: int,
) -> np.ndarray:
    """Cluster ``emb`` into ``k`` groups using MiniBatchKMeans with a final
    hard cosine assignment.

    The algorithm runs a single ``MiniBatchKMeans`` fit (no multi-restart for
    cost reasons) and then overrides the Euclidean centroid assignment with
    a proper cosine (normalized dot-product) argmax pass, which provides a
    final hard assignment of every input row to its nearest centroid in
    normalized embedding space.

    Parameters
    ----------
    emb:
        Float array of shape ``(N, D)`` — embedding matrix.  Need not be
        pre-normalised; normalisation is applied inside this function before
        the final assignment step.
    k:
        Number of clusters (``Requested_K``).  Must satisfy ``k >= 1`` and
        ``k <= N``.
    seed:
        Integer random state forwarded to ``MiniBatchKMeans.random_state``.
        Ensures deterministic output within the ``Determinism_Scope``
.

    Returns
    -------
    np.ndarray
        Int32 label array of shape ``(N,)`` with values in ``[0, k-1]``.

    """
    model = MiniBatchKMeans(
        n_clusters=k,
        random_state=seed,
        batch_size=2048,
        n_init=3,
        max_iter=100,
        reassignment_ratio=0.01,
    )
    model.fit(emb)

    centroids: np.ndarray = model.cluster_centers_

    # L2-normalise both centroids and embeddings so that dot product equals
    # cosine similarity.  normalize_vectors clips norms to 1e-10 to avoid
    # division by zero for zero vectors.
    centroids_norm = normalize_vectors(centroids.astype(np.float64))
    emb_norm = normalize_vectors(emb.astype(np.float64))

    # Final hard assignment: argmax(emb_norm @ centroids_norm.T) over axis 1.
    # np.argmax returns the *first* maximum index, so ties are broken by the
    # lowest centroid id — matching
    sims = emb_norm @ centroids_norm.T  # shape (N, k)
    labels = np.argmax(sims, axis=1).astype(np.int32)
    return labels


# ---------------------------------------------------------------------------
# _assign_to_nearest_centroid
# ---------------------------------------------------------------------------


def _assign_to_nearest_centroid(
    query_rows: np.ndarray,
    centroids: np.ndarray,
) -> np.ndarray:
    """Assign each row in ``query_rows`` to its nearest centroid by cosine
    similarity.

    Both inputs are L2-normalised internally so the dot product equals the
    cosine similarity.  Ties are broken by the lowest centroid index because
    ``np.argmax`` returns the first occurrence of the maximum value
.

    This helper is used by:
    * The oversized out-of-sample assignment step
      (``_oversized_subsample_and_assign_k`` in ``oversized.py``).
    * tests that verify the nearest-centroid contract.

    Parameters
    ----------
    query_rows:
        Float array of shape ``(M, D)`` — the rows to assign.
    centroids:
        Float array of shape ``(k, D)`` — one centroid per cluster.

    Returns
    -------
    np.ndarray
        Int32 array of shape ``(M,)`` with values in ``[0, k-1]``.

    """
    # L2-normalise both to get cosine similarity via dot product
    q_norm = normalize_vectors(query_rows.astype(np.float64))
    c_norm = normalize_vectors(centroids.astype(np.float64))

    # sims[i, j] = cosine_similarity(query_rows[i], centroids[j])
    sims = q_norm @ c_norm.T  # shape (M, k)

    # argmax along axis=1; first max occurrence -> lowest cluster id on tie
    return np.argmax(sims, axis=1).astype(np.int32)
