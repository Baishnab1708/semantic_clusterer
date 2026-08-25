"""External clustering metrics (require ground-truth labels).

These answer the question buyers actually care about: "did the library recover
the real topics?" They are computed only over rows the library actually
assigned to a cluster (label >= 0); rows marked noise (-1) are excluded from
the matched comparison and reported separately as ``noise_ratio``.

Metric glossary
---------------
ari            Adjusted Rand Index in [-1, 1]. 1.0 = perfect, 0.0 = random.
nmi            Normalised Mutual Information in [0, 1]. 1.0 = perfect.
v_measure      Harmonic mean of homogeneity and completeness, [0, 1].
homogeneity    Each cluster contains only members of one true class, [0, 1].
completeness   All members of a true class land in one cluster, [0, 1].
"""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
from sklearn.metrics import (
    adjusted_rand_score,
    completeness_score,
    homogeneity_score,
    normalized_mutual_info_score,
    v_measure_score,
)


def external_metrics(
    true_labels: Sequence[int],
    pred_labels: Sequence[int],
) -> Dict[str, float]:
    """Compute label-based agreement metrics over clustered rows only.

    Rows predicted as noise (-1) are dropped before scoring so a model is not
    rewarded for refusing to cluster, nor punished for honest noise detection.
    ``coverage`` and ``noise_ratio`` report that trade-off separately.
    """
    true = np.asarray(true_labels)
    pred = np.asarray(pred_labels)

    n_total = len(true)
    clustered = pred >= 0
    n_clustered = int(clustered.sum())
    noise_ratio = float(1.0 - n_clustered / n_total) if n_total else 0.0

    if n_clustered < 2 or len(np.unique(pred[clustered])) < 1:
        return {
            "ari": 0.0,
            "nmi": 0.0,
            "v_measure": 0.0,
            "homogeneity": 0.0,
            "completeness": 0.0,
            "coverage": float(n_clustered / n_total) if n_total else 0.0,
            "noise_ratio": noise_ratio,
            "n_pred_clusters": int(len(np.unique(pred[clustered]))) if n_clustered else 0,
        }

    t = true[clustered]
    p = pred[clustered]

    return {
        "ari": float(adjusted_rand_score(t, p)),
        "nmi": float(normalized_mutual_info_score(t, p)),
        "v_measure": float(v_measure_score(t, p)),
        "homogeneity": float(homogeneity_score(t, p)),
        "completeness": float(completeness_score(t, p)),
        "coverage": float(n_clustered / n_total),
        "noise_ratio": noise_ratio,
        "n_pred_clusters": int(len(np.unique(p))),
    }
