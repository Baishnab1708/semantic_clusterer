"""bisecting.py — Bisecting-KMeans wrapper with multi-restart for SemanticKSplit.

Implements ``_bisecting_kmeans(emb, k, seed) -> np.ndarray[int32]`` using
``sklearn.cluster.BisectingKMeans`` with a multi-restart loop driven by the
restart helpers in ``restart.py``.

Restart count:
    - ``n_restarts = 5`` when ``k == 2``
    - ``n_restarts = 3`` otherwise

Seed schedule:
    ``seed_i = (seed + i) % (2**32)`` for the i-th restart

The best restart is selected by ``_pick_better`` / ``_run_restarts`` from
``restart.py`` which ranks candidates by cosine-silhouette (higher is better),
then Davies–Bouldin index (lower is better), then restart index (lower wins on
ties).

"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import BisectingKMeans

from semantic_clusterer.k_algorithms.restart import _run_restarts


def _bisecting_kmeans(
    emb: np.ndarray,
    k: int,
    seed: int,
) -> np.ndarray:
    """Cluster *emb* into *k* groups using BisectingKMeans with multi-restart.

    Uses ``sklearn.cluster.BisectingKMeans`` with
    ``bisecting_strategy="largest_cluster"`` and ``n_init=1`` (one internal
    init per restart; the outer restart loop supplies diversity).

    Number of restarts:
        - 5 when ``k == 2``
        - 3 otherwise

    The i-th restart uses ``seed_i = (seed + i) % (2**32)``.
    The best result is chosen by cosine-silhouette → Davies–Bouldin →
    restart index.

    Parameters
    ----------
    emb:
        2-D float array of shape ``(N, D)`` — the (typically L2-normalised)
        embedding matrix.
    k:
        Number of clusters (``>= 2``).
    seed:
        Base random state.  The i-th restart receives
        ``(seed + i) % (2**32)``.

    Returns
    -------
    np.ndarray
        1-D int32 label array of shape ``(N,)`` with values in ``[0, k-1]``.

    """
    n_restarts = 5 if k == 2 else 3

    def _single_run(emb: np.ndarray, k: int, seed_i: int) -> np.ndarray:
        model = BisectingKMeans(
            n_clusters=k,
            random_state=seed_i,
            n_init=1,
            bisecting_strategy="largest_cluster",
        )
        return model.fit_predict(emb).astype(np.int32)

    return _run_restarts(
        algorithm_fn=_single_run,
        emb=emb,
        k=k,
        seed=seed,
        n_restarts=n_restarts,
    )
