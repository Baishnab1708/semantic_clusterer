"""
degenerate.py — helpers for handling degenerate embedding inputs.

When all preprocessed embeddings are exactly identical
(within tolerance), SemanticKSplit skips normal clustering and uses a
deterministic round-robin assignment instead, recording the algorithm as
"identical-embeddings-tiebreak".
"""

import numpy as np


def _all_identical(emb: np.ndarray, tol: float = 1e-9) -> bool:
    """Return True when every row of *emb* is within *tol* of the first row.

    Uses ``np.all(np.abs(emb - emb[0:1]) <= tol)`` so the check is
    vectorised and exact with the contract in the design document.

    Parameters
    ----------
    emb:
        2-D float array of shape ``(N, D)``.  Must have at least one row.
    tol:
        Absolute element-wise tolerance.  Defaults to ``1e-9``.

    Returns
    -------
    bool
        ``True`` if every element of every row is within *tol* of the
        corresponding element of ``emb[0]``; ``False`` otherwise.
    """
    return bool(np.all(np.abs(emb - emb[0:1]) <= tol))


def _round_robin_labels(N_Unique: int, k: int) -> np.ndarray:
    """Return a deterministic round-robin label array of dtype ``int32``.

    Assigns label ``i % k`` to the ``i``-th unique row (in original input
    order).  This guarantees:
    - every label in ``[0, k-1]`` appears at least once when
      ``N_Unique >= k``.
    - the assignment is fully deterministic and requires no random seed.

    Parameters
    ----------
    N_Unique:
        Number of unique rows to label (``>= k``).
    k:
        Number of target clusters (``>= 2``).

    Returns
    -------
    np.ndarray
        1-D int32 array of shape ``(N_Unique,)`` where element ``i``
        equals ``i % k``.
    """
    return np.array([i % k for i in range(N_Unique)], dtype=np.int32)
