"""agglomerative.py — Agglomerative clustering wrapper for SemanticKSplit.

Implements the ``agglomerative-cut-k`` algorithm used when
``Pipeline_Tier == "tiny"`` and ``K >= 3``.

The algorithm performs a single, fully deterministic call to
``sklearn.cluster.AgglomerativeClustering`` with average linkage on cosine
distance, cut at exactly ``k`` clusters.  No multi-restart is needed because
agglomerative hierarchical clustering with a fixed linkage criterion and a
fixed cut height is itself deterministic given the same input data.
"""

import numpy as np
from sklearn.cluster import AgglomerativeClustering


def _agglomerative_cut_k(emb: np.ndarray, k: int) -> np.ndarray:
    """Cluster *emb* into exactly *k* groups via agglomerative clustering.

    Uses ``AgglomerativeClustering(n_clusters=k, metric="cosine",
    linkage="average")`` — a single deterministic call that cuts the
    dendrogram at exactly ``k`` clusters.

    Because average-linkage agglomerative clustering on a fixed distance
    matrix is fully deterministic (no random state), multi-restart
    selection (as applied to KMeans-family algorithms) is unnecessary
    and not performed here.

    Parameters
    ----------
    emb:
        2-D float array of shape ``(N_Unique, D)`` containing the
        (optionally L2-normalised) embedding vectors.  Must have at
        least ``k`` rows.
    k:
        Number of clusters to produce.  Must satisfy ``2 <= k <= N_Unique``.

    Returns
    -------
    np.ndarray
        1-D ``int32`` array of shape ``(N_Unique,)`` with cluster label
        values in the range ``[0, k-1]`` (no gaps, no ``-1`` values).

    """
    model = AgglomerativeClustering(
        n_clusters=k,
        metric="cosine",
        linkage="average",
    )
    labels = model.fit_predict(emb)
    return labels.astype(np.int32)
