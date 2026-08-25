"""Tests for optional-dependency degradation paths.

Test 1: When umap-learn is unavailable, the small pipeline falls back to
        PCA-only clustering and records the warning in the report.

Test 2: When hdbscan is unavailable at constructor time, SemanticClusterer
        raises ImportError with the expected message.
"""

from __future__ import annotations

import sys
import warnings
from typing import List
from unittest.mock import patch

import numpy as np
import pytest

from semantic_clusterer import SemanticClusterer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_UMAP_WARNING = "umap-unavailable, used PCA-only fallback"
_N_TEXTS = 300
_DIM = 384


def _make_stub_embedder(n: int = _N_TEXTS, dim: int = _DIM, seed: int = 0) -> object:
    """Return a callable embedder that produces deterministic L2-normalised embeddings.

    Generates *n* embeddings with mild cluster structure (5 groups) so that
    HDBSCAN can find at least one cluster even without UMAP.
    """
    rng = np.random.default_rng(seed)
    n_clusters = 5
    centers = rng.standard_normal((n_clusters, dim)).astype(np.float32)
    # Normalise centers
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)

    # Build a fixed embedding matrix
    embeddings = np.empty((n, dim), dtype=np.float32)
    for i in range(n):
        c = centers[i % n_clusters]
        noise = rng.standard_normal(dim).astype(np.float32) * 0.05
        v = c + noise
        embeddings[i] = v / np.linalg.norm(v)

    class _StubEmbedder:
        def embed(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
            k = len(texts)
            return embeddings[:k].copy()

    return _StubEmbedder()


# ---------------------------------------------------------------------------
# Test 1 — PCA-only fallback when umap-learn is unavailable
# ---------------------------------------------------------------------------

def test_umap_unavailable_pca_only_fallback():
    """When try_import_umap returns None, the small pipeline:
    - completes successfully
    - records 'umap-unavailable, used PCA-only fallback' in report.warnings
    - emits exactly one UserWarning with the same message

    """
    texts = [f"sample text number {i}" for i in range(_N_TEXTS)]
    embedder = _make_stub_embedder(n=_N_TEXTS)

    with patch(
        "semantic_clusterer.optional_deps.try_import_umap",
        return_value=None,
    ):
        clusterer = SemanticClusterer(embedding_model=embedder)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            labels, report = clusterer.cluster_with_report(texts)

    # Run must complete and return a valid label array
    assert labels.shape == (len(texts),), "label array shape mismatch"
    assert labels.dtype == np.int32, "label array dtype must be int32"

    # The warning message must appear in report.warnings
    assert any(
        _UMAP_WARNING in w for w in report.warnings
    ), f"Expected '{_UMAP_WARNING}' in report.warnings, got: {report.warnings}"

    # Exactly one UserWarning with the expected message must have been emitted
    umap_user_warnings = [
        w for w in caught
        if issubclass(w.category, UserWarning) and _UMAP_WARNING in str(w.message)
    ]
    assert len(umap_user_warnings) >= 1, (
        f"Expected at least one UserWarning containing '{_UMAP_WARNING}', "
        f"got: {[str(w.message) for w in caught if issubclass(w.category, UserWarning)]}"
    )


# ---------------------------------------------------------------------------
# Test 2 — ImportError when hdbscan is missing at constructor time
# ---------------------------------------------------------------------------

def test_hdbscan_missing_raises_import_error_with_message():
    """When hdbscan is not importable, SemanticClusterer.__init__ raises
    ImportError whose message contains 'hdbscan' and 'pip install hdbscan'.

    """
    # Remove hdbscan from sys.modules so the eager import check fails
    hdbscan_module = sys.modules.pop("hdbscan", None)
    try:
        with patch.dict(sys.modules, {"hdbscan": None}):
            with pytest.raises(ImportError) as exc_info:
                SemanticClusterer()

        error_msg = str(exc_info.value)
        assert "hdbscan" in error_msg, (
            f"Expected 'hdbscan' in ImportError message, got: {error_msg!r}"
        )
        assert "pip install hdbscan" in error_msg, (
            f"Expected 'pip install hdbscan' in ImportError message, got: {error_msg!r}"
        )
    finally:
        # Restore hdbscan in sys.modules
        if hdbscan_module is not None:
            sys.modules["hdbscan"] = hdbscan_module
