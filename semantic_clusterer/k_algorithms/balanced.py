"""balanced.py — Balanced KMeans with multi-restart for SemanticKSplit.

Implements ``_balanced_kmeans``, the ``balanced-kmeans`` algorithm used for:

* ``Pipeline_Tier == "small"`` and ``K > 10``
* ``Pipeline_Tier == "medium"``

Multiple restarts with deterministic seeding are run via ``_run_restarts``
and the best partition is selected by cosine-silhouette / Davies–Bouldin
score.

"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans

from semantic_clusterer.k_algorithms.restart import _run_restarts


# ---------------------------------------------------------------------------
# _balanced_kmeans
# ---------------------------------------------------------------------------


def _balanced_kmeans(
    emb: np.ndarray,
    k: int,
    seed: int,
    n_restarts: int = 3,
) -> np.ndarray:
    """Cluster *emb* into *k* groups using KMeans with multi-restart selection.

    Runs ``n_restarts`` independent KMeans trials seeded deterministically
    from *seed* via ``_run_restarts`` (``seed_i = (seed + i) % 2**32``)
    and returns the label array from the restart with the best
    cosine-silhouette score (tie-broken by Davies–Bouldin index, then
    restart index).

    Parameters
    ----------
    emb:
        Float array of shape ``(N, D)`` — the (optionally PCA-reduced)
        embedding matrix.  Must have at least *k* rows.
    k:
        Number of clusters (>= 2).
    seed:
        Base random state.  The i-th restart receives
        ``seed_i = (seed + i) % 2**32``.
    n_restarts:
        Number of independent restarts to run; defaults to 3 per the task
        specification.

    Returns
    -------
    np.ndarray
        Int32 label array of shape ``(N,)`` with values in ``[0, k-1]``.

    """

    def _single_run(emb_: np.ndarray, k_: int, seed_i: int) -> np.ndarray:
        """Run a single seeded KMeans trial with cosine reassignment."""
        model = KMeans(
            n_clusters=k_,
            init="k-means++",
            n_init=1,
            random_state=seed_i,
            algorithm="lloyd",
        )
        labels = model.fit_predict(emb_).astype(np.int32)

        # Spherical KMeans / Cosine Reassignment Pass
        # Eliminates "dead centroids" that plague Euclidean KMeans on normalized data.
        emb_norms = np.linalg.norm(emb_, axis=1, keepdims=True)
        emb_norms[emb_norms == 0] = 1.0
        emb_normed = emb_ / emb_norms

        best_labels = labels.copy()
        for _ in range(3):
            centroids = np.zeros((k_, emb_.shape[1]), dtype=np.float32)
            valid_k = 0
            for c in range(k_):
                mask = best_labels == c
                if np.any(mask):
                    mean = np.mean(emb_normed[mask], axis=0)
                    norm = np.linalg.norm(mean)
                    if norm > 0:
                        centroids[c] = mean / norm
                        valid_k += 1
            
            if valid_k == 0:
                break
                
            sims = np.dot(emb_normed, centroids.T)
            for c in range(k_):
                if np.sum(best_labels == c) == 0:
                    sims[:, c] = -np.inf

            new_labels = np.argmax(sims, axis=1).astype(np.int32)
            if np.array_equal(best_labels, new_labels):
                break
            best_labels = new_labels

        return best_labels

    return _run_restarts(
        algorithm_fn=_single_run,
        emb=emb,
        k=k,
        seed=seed,
        n_restarts=n_restarts,
    )
