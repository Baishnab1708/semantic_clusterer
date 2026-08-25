"""repair.py — Empty-cluster repair pass for SemanticKSplit.

Implements ``_repair_empty_clusters(emb, labels, k, seed, trace) -> np.ndarray[int32]``.

The repair loop bisects the largest cluster (ties broken by lowest cluster id
via ``np.argmax``) using ``_bisecting_kmeans(sub, 2, rng_seed)`` from
``bisecting.py`` until every label in ``[0, k-1]`` is represented.

After at least one repair iteration, ``"empty-cluster-repaired"`` is appended
to ``trace.warnings`` exactly once via ``trace.warn()``.

"""

from __future__ import annotations

import numpy as np

from semantic_clusterer.k_algorithms.bisecting import _bisecting_kmeans


def _repair_empty_clusters(
    emb: np.ndarray,
    labels: np.ndarray,
    k: int,
    seed: int,
    trace,
) -> np.ndarray:
    """Repair empty clusters until every label in ``[0, k-1]`` is present.

    Algorithm (per iteration):

    1. Find present labels via ``np.unique(labels)``.
    2. Compute ``missing = sorted(set(range(k)) - set(int(x) for x in present))``.
    3. If not missing, break the loop.
    4. ``sizes = np.bincount(labels, minlength=k)``;
       ``target = int(np.argmax(sizes))`` — lowest cluster id on ties because
       ``np.argmax`` returns the first occurrence of the maximum.
    5. ``target_idx = np.where(labels == target)[0]``, ``sub = emb[target_idx]``.
    6. ``sub_labels = _bisecting_kmeans(sub, 2, rng_seed)``.
    7. ``smaller_subhalf = int(np.argmin(np.bincount(sub_labels, minlength=2)))``.
    8. ``smaller_idx = target_idx[sub_labels == smaller_subhalf]``.
    9. ``labels[smaller_idx] = missing[0]``.
    10. ``rng_seed = (rng_seed + 1) % (2**32)``.

    When at least one repair iteration ran, ``"empty-cluster-repaired"`` is
    appended to ``trace.warnings`` exactly once (``trace.warn()`` already
    deduplicates, so calling it once after the loop is sufficient).

    Parameters
    ----------
    emb:
        2-D float array of shape ``(N_Unique, D)`` — the (typically
        L2-normalised) embedding matrix.
    labels:
        1-D int32 label array of shape ``(N_Unique,)``.  Modified **in-place**
        and also returned for convenience.
    k:
        Total number of clusters requested.  Labels must end up covering
        exactly ``{0, 1, ..., k-1}``.
    seed:
        Base random state for the repair bisection steps.  The first repair
        iteration uses ``seed`` directly; subsequent iterations use
        ``(seed + 1) % (2**32)``, ``(seed + 2) % (2**32)``, etc.
    trace:
        An object with a ``warnings`` list attribute and a ``warn(str)``
        method (e.g. ``_PipelineTrace``).  The string
        ``"empty-cluster-repaired"`` is appended via ``trace.warn()`` at most
        once per call.

    Returns
    -------
    np.ndarray
        The (possibly modified) ``labels`` array, dtype ``np.int32``,
        shape ``(N_Unique,)``, with values in ``[0, k-1]``.

    """
    repaired = False
    rng_seed = seed

    while True:
        present = np.unique(labels)
        missing = sorted(set(range(k)) - set(int(x) for x in present))
        if not missing:
            break

        repaired = True

        # Largest cluster; np.argmax returns the first (lowest id) on ties
        sizes = np.bincount(labels, minlength=k)
        target = int(np.argmax(sizes))

        target_idx = np.where(labels == target)[0]
        sub = emb[target_idx]

        sub_labels = _bisecting_kmeans(sub, 2, rng_seed)

        smaller_subhalf = int(np.argmin(np.bincount(sub_labels, minlength=2)))
        smaller_idx = target_idx[sub_labels == smaller_subhalf]
        labels[smaller_idx] = missing[0]

        rng_seed = (rng_seed + 1) % (2**32)

    if repaired:
        trace.warn("empty-cluster-repaired")

    return labels
