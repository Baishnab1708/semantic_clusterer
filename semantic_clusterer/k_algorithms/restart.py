"""restart.py — Multi-restart selection helpers for SemanticKSplit.

Implements the restart infrastructure used by k-aware algorithms that
benefit from multiple seeded trials (bisecting-kmeans, balanced-kmeans,
constrained-kmeans).  The core data structures and helper functions are:

- ``_RestartCandidate``  — frozen dataclass holding one restart's result
- ``_selection_score``   — compute (silhouette_cosine, -davies_bouldin)
- ``_pick_better``       — return the candidate with the better key
- ``_run_restarts``      — iterate restarts and return the best labels

"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Tuple

import numpy as np
from sklearn.metrics import davies_bouldin_score, silhouette_score


# ---------------------------------------------------------------------------
# _RestartCandidate
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _RestartCandidate:
    """Immutable record of one restart's clustering result and quality scores.

    Fields
    ------
    labels:
        Shape ``(N_Unique,)`` int32 label array produced by one restart.
        Excluded from equality and hash comparisons (numpy arrays are not
        hashable and we compare candidates via ``selection_key()`` only).
    silhouette:
        Cosine-silhouette score.  ``float('nan')`` when undefined
        (e.g. single cluster).
    davies_bouldin:
        Davies–Bouldin index (lower is better).  ``float('inf')`` when
        undefined.
    restart_index:
        Zero-based restart index; used as the final tie-break so that
        the *first* restart wins when all quality scores are equal
.
    """

    labels: np.ndarray = field(compare=False, hash=False)
    silhouette: float
    davies_bouldin: float
    restart_index: int

    def selection_key(self) -> Tuple[float, float, int]:
        """Return a sortable tuple for ascending comparison.

        Tuple layout ``(a, b, c)`` — lower is better:

        * ``a`` = ``-silhouette`` if finite, else ``+inf``
          (maximise silhouette by minimising its negation).
        * ``b`` = ``davies_bouldin`` if finite, else ``+inf``
          (minimise Davies–Bouldin directly).
        * ``c`` = ``restart_index``
          (prefer the first restart on exact ties

        """
        sil_key = (
            -self.silhouette if np.isfinite(self.silhouette) else np.inf
        )
        dbi_key = (
            self.davies_bouldin if np.isfinite(self.davies_bouldin) else np.inf
        )
        return (sil_key, dbi_key, self.restart_index)


# ---------------------------------------------------------------------------
# _pick_better
# ---------------------------------------------------------------------------


def _pick_better(
    a: _RestartCandidate,
    b: _RestartCandidate,
) -> _RestartCandidate:
    """Return the candidate with the lower (better) ``selection_key()``.

    When ``a.selection_key() <= b.selection_key()``, ``a`` is returned;
    otherwise ``b``.  The ``<=`` comparison represents the tie-break by
    silhouette → Davies–Bouldin → restart index.

    Parameters
    ----------
    a, b:
        Two ``_RestartCandidate`` instances to compare.

    Returns
    -------
    _RestartCandidate
        The better of the two candidates.
    """
    return a if a.selection_key() <= b.selection_key() else b


# ---------------------------------------------------------------------------
# _selection_score
# ---------------------------------------------------------------------------


def _selection_score(
    emb: np.ndarray,
    labels: np.ndarray,
) -> Tuple[float, float]:
    """Compute ``(silhouette_cosine, -davies_bouldin)`` for restart ranking.

    Both returned values are oriented so that *higher is better*, making
    them directly comparable in a single tuple.

    Edge cases:
    * Fewer than two distinct labels → ``(nan, -inf)`` so the candidate
      is ranked last (``selection_key()`` maps ``nan`` silhouette to
      ``+inf`` and ``inf`` Davies–Bouldin to ``+inf``).
    * Any sklearn exception (e.g. all points identical, degenerate
      affinity matrix) → the affected metric is clamped to its worst
      fallback value.

    Parameters
    ----------
    emb:
        Float array of shape ``(N, D)`` — the embedding matrix passed to
        the algorithm (already L2-normalised if normalization is enabled).
    labels:
        Int array of shape ``(N,)`` — cluster assignments, values in
        ``[0, k-1]``.  May contain ``-1`` for noise rows but those are
        uncommon at this stage.

    Returns
    -------
    (silhouette_cosine, neg_davies_bouldin):
        ``silhouette_cosine`` in ``[-1, 1]`` (nan when undefined).
        ``neg_davies_bouldin`` = ``-DB`` where ``DB >= 0``
        (``-inf`` when undefined).

    """
    # Determine unique valid labels (exclude -1 noise if present)
    valid_mask = labels >= 0
    unique_labels = np.unique(labels[valid_mask]) if np.any(valid_mask) else np.array([])

    if len(unique_labels) < 2:
        # Single-cluster or empty partition: both scores undefined
        return float("nan"), float("-inf")

    # --- Silhouette score (cosine distance) --------------------------------
    try:
        sil = float(silhouette_score(emb, labels, metric="cosine"))
    except Exception:
        sil = float("nan")

    # --- Davies–Bouldin index (raw, lower is better) -----------------------
    try:
        dbi = float(davies_bouldin_score(emb, labels))
        neg_dbi = -dbi
    except Exception:
        neg_dbi = float("-inf")

    return sil, neg_dbi


# ---------------------------------------------------------------------------
# _run_restarts
# ---------------------------------------------------------------------------


def _run_restarts(
    algorithm_fn: Callable[[np.ndarray, int, int], np.ndarray],
    emb: np.ndarray,
    k: int,
    seed: int,
    n_restarts: int,
) -> np.ndarray:
    """Run ``n_restarts`` seeded trials and return the best label array.

    Iterates ``i in range(n_restarts)``, computes
    ``seed_i = (seed + i) % (2**32)`` for each restart, calls
    ``algorithm_fn(emb, k, seed_i)``, scores the result, and keeps
    the ``_RestartCandidate`` with the lowest ``selection_key()``.

    Parameters
    ----------
    algorithm_fn:
        Callable with signature ``(emb, k, seed) -> np.ndarray[int32]``.
        Must return a label array of shape ``(N,)`` with values in
        ``[0, k-1]``.
    emb:
        Float array of shape ``(N, D)`` — embedding matrix.
    k:
        Number of clusters requested.
    seed:
        Base random state (``Random_State``).  The i-th restart receives
        ``seed_i = (seed + i) % (2**32)``.
    n_restarts:
        Total number of restarts to run (>= 1).

    Returns
    -------
    np.ndarray
        The int32 label array from the best restart, shape ``(N,)``.

    """
    best: _RestartCandidate | None = None

    for i in range(n_restarts):
        seed_i = (seed + i) % (2**32)
        labels_i = algorithm_fn(emb, k, seed_i)

        sil, neg_dbi = _selection_score(emb, labels_i)
        # Store raw Davies–Bouldin (positive); selection_key() handles sign
        candidate = _RestartCandidate(
            labels=labels_i,
            silhouette=sil,
            davies_bouldin=-neg_dbi,  # negate back from -DB to raw DB
            restart_index=i,
        )

        if best is None:
            best = candidate
        else:
            best = _pick_better(best, candidate)

    # best is never None because n_restarts >= 1
    assert best is not None, "n_restarts must be >= 1"
    return best.labels
