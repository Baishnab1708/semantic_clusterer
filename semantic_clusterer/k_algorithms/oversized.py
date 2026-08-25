"""oversized.py — Subsample-then-assign path for SemanticKSplit when N > 200_000.

Implements ``_oversized_subsample_and_assign_k(emb, k, seed, trace) ->
np.ndarray[int32]``, the k-aware counterpart to
``SemanticClusterer._oversized_subsample_and_assign``.

Algorithm:

1. Draw 200_000 distinct row indices deterministically using
   ``np.random.default_rng(seed)``.
2. Sort indices for cache-friendly access.
3. Fit ``_minibatch_kmeans_assign(subsample, k, seed)`` on the subsample.
4. Repair any empty clusters via ``_repair_empty_clusters``.
5. Build per-cluster centroids from L2-normalised subsample embeddings.
6. Assign every out-of-sample row to its nearest centroid via
   ``_assign_to_nearest_centroid`` (ties broken by lowest centroid index).
7. Assemble the full ``(N,)`` label array.
8. Append ``"oversized-subsampled"`` to ``trace.warnings`` exactly once.

The gating ``ValueError`` and ``UserWarning`` messages are formatted from
module-level constants imported from ``semantic_clusterer.core``; this
guarantees byte-for-byte identical wording between ``SemanticClusterer``
and ``SemanticKSplit``.

"""

from __future__ import annotations

import numpy as np

# Re-export the shared message-format constants so callers (e.g. k_split.py)
# can import them from a single location and produce wording identical to
# SemanticClusterer.
from semantic_clusterer.core import (  # noqa: F401
    _OVERSIZED_ERROR_MSG_FMT,
    _OVERSIZED_WARN_MSG_FMT,
)
from semantic_clusterer.k_algorithms.minibatch_assign import (
    _assign_to_nearest_centroid,
    _minibatch_kmeans_assign,
)
from semantic_clusterer.k_algorithms.repair import _repair_empty_clusters
from semantic_clusterer.utils.similarity import normalize_vectors

# The hard limit used by both SemanticClusterer and SemanticKSplit.
_OVERSIZED_LIMIT: int = 200_000


def _oversized_subsample_and_assign_k(
    emb: np.ndarray,
    k: int,
    seed: int,
    trace,
) -> np.ndarray:
    """Subsample-then-assign clustering for ``N > 200_000`` inputs.

    The function performs the following steps:

    1. **Subsample** — draw exactly ``200_000`` distinct row indices from
       ``[0, N)`` without replacement using a seeded
       ``np.random.default_rng(seed)``.  The index array is sorted for
       cache-friendly access.
    2. **Fit** — run ``_minibatch_kmeans_assign(subsample, k, seed)`` on the
       200_000-row subsample to obtain initial cluster labels.
    3. **Repair** — pass the subsample labels through
       ``_repair_empty_clusters`` so every label in ``[0, k-1]`` is
       present.
    4. **Centroids** — L2-normalise the subsample embeddings, compute the
       mean embedding per cluster, then re-normalise the resulting
       centroid vectors.
    5. **Out-of-sample assignment** — for every row *not* in the subsample,
       call ``_assign_to_nearest_centroid(out_rows, centroids)`` which
       returns the lowest-index centroid on cosine-similarity ties
.
    6. **Assemble** — place subsample labels and out-of-sample labels into
       a fresh ``(N,)`` int32 array at their original positions
.
    7. **Trace** — append ``"oversized-subsampled"`` to ``trace.warnings``
       exactly once via ``trace.warn(...)``.

    Parameters
    ----------
    emb:
        Float array of shape ``(N, D)`` where ``N > 200_000``.  Need not be
        pre-normalised; normalisation is applied internally for centroid
        construction.
    k:
        Number of clusters (``Requested_K``).  Must satisfy
        ``2 <= k <= 200_000`` so that the subsample is large enough to
        contain at least one point per cluster.
    seed:
        Integer random state derived from ``ClustererConfig.random_state``.
        Forwarded unchanged to ``_minibatch_kmeans_assign``,
        ``_repair_empty_clusters``, and ``np.random.default_rng``.
    trace:
        An object with a ``warnings`` list and a ``warn(str)`` method (e.g.
        ``_PipelineTrace``).  Receives ``"oversized-subsampled"`` and,
        potentially, ``"empty-cluster-repaired"`` from the repair pass.

    Returns
    -------
    np.ndarray
        Int32 label array of shape ``(N,)`` with values in ``[0, k-1]``
        for every row (no ``-1`` values are introduced by this function;
        filtered/missing rows are handled by the caller).

    """
    N, D = emb.shape

    # ------------------------------------------------------------------
    # Step 1 — deterministic subsample of LIMIT distinct indices
    # ------------------------------------------------------------------
    rng = np.random.default_rng(seed)
    subsample_idx: np.ndarray = rng.choice(N, size=_OVERSIZED_LIMIT, replace=False)
    subsample_idx.sort()  # cache-friendly; doesn't affect correctness
    subsample: np.ndarray = emb[subsample_idx]

    # ------------------------------------------------------------------
    # Step 2 — fit MiniBatchKMeans on the subsample
    # ------------------------------------------------------------------
    sub_labels: np.ndarray = _minibatch_kmeans_assign(subsample, k, seed=seed)

    # ------------------------------------------------------------------
    # Step 3 — repair empty clusters on the subsample
    # ``_repair_empty_clusters`` modifies sub_labels in-place and also
    # returns it; it may append "empty-cluster-repaired" to trace.warnings.
    # ------------------------------------------------------------------
    sub_labels = _repair_empty_clusters(subsample, sub_labels, k, seed, trace)

    # ------------------------------------------------------------------
    # Step 4 — build per-cluster centroids in normalized space
    # L2-normalise subsample -> compute mean per cluster -> re-normalise
    # ------------------------------------------------------------------
    norm_subsample = normalize_vectors(subsample.astype(np.float64))  # (LIMIT, D)

    centroids_raw = np.zeros((k, D), dtype=np.float64)
    for c in range(k):
        mask = sub_labels == c
        if mask.any():
            centroids_raw[c] = np.mean(norm_subsample[mask], axis=0)
        # If a cluster is still empty after repair (shouldn't happen but
        # guarded here), the zero centroid will receive no further
        # assignment — _assign_to_nearest_centroid will still produce a
        # valid argmax.

    # Re-normalise the mean vectors to obtain proper unit centroids
    centroids = normalize_vectors(centroids_raw)  # (k, D)

    # ------------------------------------------------------------------
    # Step 5 + 6 — assemble full label array
    # Subsample rows get their labels from sub_labels; out-of-sample rows
    # get labels from nearest-centroid cosine assignment.
    # ------------------------------------------------------------------
    full_labels = np.full(N, -1, dtype=np.int32)

    # Place subsample labels at their original positions
    full_labels[subsample_idx] = sub_labels

    # Identify out-of-sample rows
    out_mask = np.ones(N, dtype=bool)
    out_mask[subsample_idx] = False
    out_idx: np.ndarray = np.where(out_mask)[0]

    if out_idx.size > 0:
        out_emb = emb[out_idx]  # (M, D)
        # _assign_to_nearest_centroid normalises internally
        out_labels = _assign_to_nearest_centroid(out_emb, centroids)
        full_labels[out_idx] = out_labels

    # ------------------------------------------------------------------
    # Step 7 — append trace warning exactly once
    # ------------------------------------------------------------------
    trace.warn("oversized-subsampled")

    return full_labels
