"""Optional dependency helpers.

Provides graceful import wrappers for optional heavy dependencies so that
callers can check availability without catching ImportError themselves.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Generator, Optional, Type

import numpy as np


@contextmanager
def _seeded_global_numpy(seed: int) -> Generator[None, None, None]:
    """Context manager that temporarily sets numpy's global RNG seed.

    Saves the current global numpy random state, seeds the global RNG with
    *seed*, yields, then unconditionally restores the prior state on exit
    (including on exception).

    .. warning::
        The **only** third-party code permitted inside the ``with``-block is
        HDBSCAN.  Do not call any other library that may consume or mutate
        ``numpy``'s global RNG state while this context manager is active.
        Violating this constraint undermines the determinism guarantee
        described in the README's "Determinism" section.

    Parameters
    ----------
    seed:
        Integer seed forwarded to ``np.random.seed()``.  Must be in the
        range ``[0, 2**32 - 1]`` (the same range accepted by
        ``ClustererConfig.random_state``).

    Examples
    --------
    >>> import numpy as np
    >>> from semantic_clusterer.optional_deps import _seeded_global_numpy
    >>> with _seeded_global_numpy(42):
    ...     labels = hdbscan_clusterer.fit_predict(reduced_embeddings)
    """
    state = np.random.get_state()
    try:
        np.random.seed(seed)
        yield
    finally:
        np.random.set_state(state)


def try_import_umap() -> Optional[Type]:
    """Attempt to import UMAP and return the class, or *None* on failure.

    Used by the PCA-only fallback path (task 9.1) to detect whether
    ``umap-learn`` is available at runtime.

    Returns
    -------
    type or None
        The ``umap.UMAP`` class if the package is importable, else ``None``.
    """
    try:
        import warnings
        # Ignore UMAP's warning about overriding n_jobs to 1 for determinism when random_state is set.
        warnings.filterwarnings("ignore", category=UserWarning, message=".*n_jobs value.*")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ImportWarning, module="umap")
            from umap import UMAP
        return UMAP
    except ImportError:
        return None
