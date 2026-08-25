"""Public-API surface tests for semantic_clusterer.

Validates

Tests:
  1. __all__ equals the locked seven-symbol set.
  2. SemanticClusterer exposes only the four expected non-underscore callable attrs.
  3. SemanticClusterer does NOT define sklearn-style fit/predict methods.
  4. inspect.signature of cluster, cluster_labels, and embed matches frozen baselines.
"""

import inspect
from typing import List, Literal, Union

import numpy as np
import pytest

import semantic_clusterer


# ---------------------------------------------------------------------------
# 1. __all__ surface
# ---------------------------------------------------------------------------

LOCKED_ALL = {
    "SemanticClusterer",
    "SemanticKSplit",
    "SemanticClustererConfig",
    "SemanticKSplitConfig",
    "ClusteringReport",
    "SUPPORTED_DIM_BANDS",
    "normalize_embedding_model",
    "validate_embeddings",
    "__version__",
    # persistence
    "FittedState",
    "ClusterStats",
    # Backward-compat alias
    "ClustererConfig",
}


def test_all_equals_locked_set():
    """__all__ must be exactly the locked set — no more, no less."""
    assert set(semantic_clusterer.__all__) == LOCKED_ALL, (
        f"__all__ mismatch.\n"
        f"  Extra:   {set(semantic_clusterer.__all__) - LOCKED_ALL}\n"
        f"  Missing: {LOCKED_ALL - set(semantic_clusterer.__all__)}"
    )


# ---------------------------------------------------------------------------
# 2. SemanticClusterer callable attributes
# ---------------------------------------------------------------------------

# v0.1.0 contract: density-mode (cluster, cluster_labels, cluster_with_report,
# embed) plus the persistence/production API (fit, predict, fit_predict,
# save, load) and the topic accessors.
EXPECTED_PUBLIC_CALLABLES = {
    "cluster",
    "cluster_labels",
    "cluster_with_report",
    "embed",
    # Production API
    "fit",
    "predict",
    "fit_predict",
    "save",
    "load",
    "get_topic_keywords",
    "get_topic_labels",
}


def test_semantic_clusterer_public_callables():
    """SemanticClusterer must expose exactly the expected set of public callables."""
    from semantic_clusterer import SemanticClusterer

    # Collect all non-underscore-prefixed callable attributes defined on the class
    # (not inherited from object, to avoid noise from __init_subclass__ etc.)
    public_callables = {
        name
        for name in dir(SemanticClusterer)
        if not name.startswith("_") and callable(getattr(SemanticClusterer, name))
    }

    assert public_callables == EXPECTED_PUBLIC_CALLABLES, (
        f"Public callable attribute mismatch.\n"
        f"  Extra:   {public_callables - EXPECTED_PUBLIC_CALLABLES}\n"
        f"  Missing: {EXPECTED_PUBLIC_CALLABLES - public_callables}"
    )


# ---------------------------------------------------------------------------
# 3. Sklearn-style methods are now part of the public contract (v0.3.0)
# ---------------------------------------------------------------------------

REQUIRED_SKLEARN_METHODS = {"fit", "predict", "fit_predict"}


def test_sklearn_style_methods_present():
    """SemanticClusterer MUST define fit, predict, fit_predict (v0.3.0 contract)."""
    from semantic_clusterer import SemanticClusterer

    missing = [m for m in REQUIRED_SKLEARN_METHODS if not hasattr(SemanticClusterer, m)]
    assert not missing, (
        f"SemanticClusterer is missing required sklearn-style methods: {missing}"
    )


# ---------------------------------------------------------------------------
# 4. Frozen signature baselines
# ---------------------------------------------------------------------------
# Baselines are derived from the design document's locked signatures.
# Each entry is: (param_name, kind, default, annotation_check_fn)
# annotation_check_fn receives the annotation object and returns True if valid.

def _is_list_str(ann) -> bool:
    """Accept List[str] or list[str] or typing.List[str]."""
    origin = getattr(ann, "__origin__", None)
    args = getattr(ann, "__args__", ())
    # typing.List[str] -> origin is list, args is (str,)
    return origin is list and args == (str,)


def _is_ndarray(ann) -> bool:
    return ann is np.ndarray


def _is_empty(ann) -> bool:
    return ann is inspect.Parameter.empty


# Frozen baseline for SemanticClusterer.cluster
# def cluster(self, texts: List[str], return_format: Literal["simple", "detailed"] = "simple")
CLUSTER_BASELINE = [
    # (name, kind, has_default, default_value_or_sentinel, annotation_check)
    ("texts",         inspect.Parameter.POSITIONAL_OR_KEYWORD, False, inspect.Parameter.empty, _is_list_str),
    ("return_format", inspect.Parameter.POSITIONAL_OR_KEYWORD, True,  "simple",                lambda a: True),
]

# Frozen baseline for SemanticClusterer.cluster_labels
# def cluster_labels(self, texts: List[str]) -> np.ndarray
CLUSTER_LABELS_BASELINE = [
    ("texts", inspect.Parameter.POSITIONAL_OR_KEYWORD, False, inspect.Parameter.empty, _is_list_str),
]

# Frozen baseline for SemanticClusterer.embed
# def embed(self, texts: List[str]) -> np.ndarray
EMBED_BASELINE = [
    ("texts", inspect.Parameter.POSITIONAL_OR_KEYWORD, False, inspect.Parameter.empty, _is_list_str),
]


def _check_signature(method, baseline, method_name):
    """Assert that a method's signature matches the frozen baseline."""
    sig = inspect.signature(method)
    params = [
        (name, p)
        for name, p in sig.parameters.items()
        if name != "self"
    ]

    assert len(params) == len(baseline), (
        f"{method_name}: expected {len(baseline)} non-self parameter(s), "
        f"got {len(params)}: {[n for n, _ in params]}"
    )

    for (actual_name, param), (exp_name, exp_kind, exp_has_default, exp_default, ann_check) in zip(params, baseline):
        # Parameter name
        assert actual_name == exp_name, (
            f"{method_name}: parameter name mismatch — expected '{exp_name}', got '{actual_name}'"
        )

        # Parameter kind (positional-or-keyword, keyword-only, etc.)
        assert param.kind == exp_kind, (
            f"{method_name}.{actual_name}: kind mismatch — "
            f"expected {exp_kind.name}, got {param.kind.name}"
        )

        # Default presence and value
        has_default = param.default is not inspect.Parameter.empty
        assert has_default == exp_has_default, (
            f"{method_name}.{actual_name}: "
            f"{'expected a default' if exp_has_default else 'expected no default'}, "
            f"got default={param.default!r}"
        )
        if exp_has_default:
            assert param.default == exp_default, (
                f"{method_name}.{actual_name}: default mismatch — "
                f"expected {exp_default!r}, got {param.default!r}"
            )

        # Annotation (only checked when annotation is present)
        if param.annotation is not inspect.Parameter.empty:
            assert ann_check(param.annotation), (
                f"{method_name}.{actual_name}: annotation {param.annotation!r} "
                "failed the baseline check"
            )


def test_cluster_signature():
    """cluster() signature must match the frozen baseline."""
    from semantic_clusterer import SemanticClusterer
    _check_signature(SemanticClusterer.cluster, CLUSTER_BASELINE, "cluster")


def test_cluster_labels_signature():
    """cluster_labels() signature must match the frozen baseline."""
    from semantic_clusterer import SemanticClusterer
    _check_signature(SemanticClusterer.cluster_labels, CLUSTER_LABELS_BASELINE, "cluster_labels")


def test_embed_signature():
    """embed() signature must match the frozen baseline."""
    from semantic_clusterer import SemanticClusterer
    _check_signature(SemanticClusterer.embed, EMBED_BASELINE, "embed")


# ---------------------------------------------------------------------------
# 5. Return-type annotations on cluster_labels and embed
# ---------------------------------------------------------------------------

def test_cluster_labels_return_annotation():
    """cluster_labels() return annotation must be np.ndarray."""
    from semantic_clusterer import SemanticClusterer
    sig = inspect.signature(SemanticClusterer.cluster_labels)
    ret = sig.return_annotation
    if ret is not inspect.Parameter.empty:
        assert ret is np.ndarray, (
            f"cluster_labels return annotation should be np.ndarray, got {ret!r}"
        )


def test_embed_return_annotation():
    """embed() return annotation must be np.ndarray."""
    from semantic_clusterer import SemanticClusterer
    sig = inspect.signature(SemanticClusterer.embed)
    ret = sig.return_annotation
    if ret is not inspect.Parameter.empty:
        assert ret is np.ndarray, (
            f"embed return annotation should be np.ndarray, got {ret!r}"
        )


# ---------------------------------------------------------------------------
# 6. All seven __all__ symbols are importable (smoke test)
# ---------------------------------------------------------------------------

def test_all_symbols_importable():
    """Every symbol in __all__ must be accessible via getattr on the module."""
    missing = [s for s in LOCKED_ALL if getattr(semantic_clusterer, s, None) is None]
    assert not missing, f"Symbols in __all__ not accessible on module: {missing}"
