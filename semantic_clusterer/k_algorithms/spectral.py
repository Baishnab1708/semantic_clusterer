"""spectral.py — Spectral-cosine clustering with constrained-kmeans fallback.

Implements the ``spectral-cosine`` algorithm used by SemanticKSplit for the
``small`` tier when ``3 <= k <= 10``.  If the SpectralClustering eigensolver
fails to converge, the implementation transparently falls back to
``constrained-kmeans`` (a plain KMeans wrapper) and records the substitution
in the pipeline trace.

Public functions
----------------
- ``_cosine_affinity_matrix(emb)``     — build the symmetric affinity matrix
- ``_spectral_cosine(emb, k, seed, trace)`` — main spectral clustering call
- ``_constrained_kmeans(emb, k, seed)``     — KMeans fallback wrapper

"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.cluster import KMeans, SpectralClustering


# ---------------------------------------------------------------------------
# _cosine_affinity_matrix
# ---------------------------------------------------------------------------


def _cosine_affinity_matrix(emb: np.ndarray) -> np.ndarray:
    """Build a symmetric cosine-affinity matrix from an embedding array.

    The affinity is defined as::

        A = clip((emb_norm @ emb_norm.T + 1) / 2, 0, 1)

    where ``emb_norm`` is ``emb`` with every row L2-normalised to unit length.
    The ``(x + 1) / 2`` transform maps cosine similarities from ``[-1, 1]``
    to ``[0, 1]``, making the matrix non-negative as required by
    ``SpectralClustering(affinity="precomputed")``.  ``np.clip`` guards
    against tiny floating-point excursions outside ``[0, 1]``.

    Parameters
    ----------
    emb:
        Float array of shape ``(N, D)``.  Need not be pre-normalised;
        L2-normalisation is applied internally.

    Returns
    -------
    np.ndarray
        Symmetric float64 array of shape ``(N, N)`` with values in
        ``[0, 1]``.

    """
    # L2-normalise rows so that the dot product equals cosine similarity.
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    # Avoid division by zero for zero-norm rows (treat as unit vector).
    norms = np.where(norms == 0, 1.0, norms)
    emb_norm = emb / norms

    # Cosine similarity matrix, mapped from [-1, 1] to [0, 1].
    affinity = np.clip((emb_norm @ emb_norm.T + 1.0) / 2.0, 0.0, 1.0)
    return affinity


# ---------------------------------------------------------------------------
# _constrained_kmeans
# ---------------------------------------------------------------------------


def _constrained_kmeans(
    emb: np.ndarray,
    k: int,
    seed: int,
) -> np.ndarray:
    """Run KMeans with k-means++ init and return int32 labels.

    This is the fallback algorithm used when ``spectral-cosine``'s
    eigensolver fails to converge.  A single KMeans fit with ``n_init=3``
    is used to amortise sensitivity to the random initialisation without
    the full multi-restart loop (the caller may wrap this in
    ``_run_restarts`` when needed).

    Parameters
    ----------
    emb:
        Float array of shape ``(N, D)`` — the embedding matrix.
    k:
        Number of clusters requested.
    seed:
        Integer random state forwarded to ``KMeans(random_state=seed)``.

    Returns
    -------
    np.ndarray
        Int32 array of shape ``(N,)`` with values in ``[0, k-1]``.

    """
    model = KMeans(
        n_clusters=k,
        init="k-means++",
        n_init=3,
        random_state=seed,
    )
    labels = model.fit_predict(emb).astype(np.int32)
    return labels


# ---------------------------------------------------------------------------
# _spectral_cosine
# ---------------------------------------------------------------------------


def _spectral_cosine(
    emb: np.ndarray,
    k: int,
    seed: int,
    trace: Any,
) -> np.ndarray:
    """Cluster ``emb`` into ``k`` groups using spectral clustering on cosine affinity.

    Builds a precomputed cosine-affinity matrix via
    ``_cosine_affinity_matrix`` and passes it to
    ``SpectralClustering(affinity="precomputed", assign_labels="kmeans")``.

    Fallback behaviour
    ------------------
    If *any* exception is raised during spectral clustering (most commonly
    an eigensolver non-convergence ``ArpackNoConvergence`` error), the
    method transparently falls back to ``_constrained_kmeans`` and
    records the substitution::

        trace.chosen_params["algorithm_used"] = "constrained-kmeans"

    This ensures the pipeline always produces a valid partition even when
    the graph Laplacian is ill-conditioned.

    Parameters
    ----------
    emb:
        Float array of shape ``(N, D)`` — the embedding matrix.
    k:
        Number of clusters requested.
    seed:
        Integer random state forwarded to ``SpectralClustering`` (and to
        the KMeans fallback when triggered).
    trace:
        Pipeline trace object with a ``chosen_params`` dict attribute.
        Used to record ``"algorithm_used"`` when the fallback is invoked.

    Returns
    -------
    np.ndarray
        Int32 array of shape ``(N,)`` with values in ``[0, k-1]``.

    """
    affinity = _cosine_affinity_matrix(emb)

    try:
        model = SpectralClustering(
            n_clusters=k,
            affinity="precomputed",
            assign_labels="kmeans",
            random_state=seed,
        )
        labels = model.fit_predict(affinity).astype(np.int32)
    except Exception:
        # Eigensolver failed to converge (or any other spectral error).
        # Fall back to constrained-kmeans and update the trace.
        labels = _constrained_kmeans(emb, k, seed)
        trace.chosen_params["algorithm_used"] = "constrained-kmeans"

    return labels
