"""Smoke tests for the SemanticKSplit public API surface.

- Package-level export and __all__
- Exactly four non-underscore methods exposed
- Constructor signature matches spec
- Existing exports remain intact

These tests are intentionally import-only / introspection-only; no
embedding model is invoked.
"""

import inspect
import importlib

import pytest


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _public_methods(cls):
    """Return the set of non-underscore-prefixed callable method names on a class."""
    return {
        name
        for name, member in inspect.getmembers(cls, predicate=inspect.isfunction)
        if not name.startswith("_")
    }


# ---------------------------------------------------------------------------
# Import resolves without error
# ---------------------------------------------------------------------------

class TestImport:
    def test_import_resolves_without_error(self):
        """from semantic_clusterer import SemanticKSplit must not raise."""
        from semantic_clusterer import SemanticKSplit  # noqa: F401
        assert SemanticKSplit is not None

    def test_imported_object_is_a_class(self):
        from semantic_clusterer import SemanticKSplit
        assert inspect.isclass(SemanticKSplit)


# ---------------------------------------------------------------------------
# SemanticKSplit in __all__
# ---------------------------------------------------------------------------

class TestAllExport:
    def test_semantic_k_split_in_dunder_all(self):
        import semantic_clusterer
        assert "SemanticKSplit" in semantic_clusterer.__all__


# ---------------------------------------------------------------------------
# 2.7 – exactly four non-underscore methods
# ---------------------------------------------------------------------------

class TestPublicMethodSurface:
    EXPECTED_PUBLIC_METHODS = {
        "split", "split_labels", "split_with_report", "embed",
        # v0.1.0 additions (non-classmethods)
        "cluster", "fit", "predict", "fit_predict", "save",
        "get_topic_keywords", "get_topic_labels",
    }
    # load is a classmethod — not captured by inspect.isfunction
    EXPECTED_PUBLIC_CLASSMETHODS = {"load"}

    def test_exactly_four_public_methods(self):
        from semantic_clusterer import SemanticKSplit
        public = _public_methods(SemanticKSplit)
        # Allow a superset — new methods may be present
        missing = self.EXPECTED_PUBLIC_METHODS - public
        assert not missing, (
            f"SemanticKSplit is missing expected public methods: {missing}"
        )

    def test_load_classmethod_present(self):
        from semantic_clusterer import SemanticKSplit
        assert hasattr(SemanticKSplit, "load")
        assert callable(SemanticKSplit.load)

    def test_split_is_callable(self):
        from semantic_clusterer import SemanticKSplit
        assert callable(SemanticKSplit.split)

    def test_split_labels_is_callable(self):
        from semantic_clusterer import SemanticKSplit
        assert callable(SemanticKSplit.split_labels)

    def test_split_with_report_is_callable(self):
        from semantic_clusterer import SemanticKSplit
        assert callable(SemanticKSplit.split_with_report)

    def test_embed_is_callable(self):
        from semantic_clusterer import SemanticKSplit
        assert callable(SemanticKSplit.embed)


# ---------------------------------------------------------------------------
# Constructor signature
# Expected: __init__(self, embedding_model=None, *, k, config=None,
#                    verbose=False, random_state=42)
# ---------------------------------------------------------------------------

class TestConstructorSignature:
    def _get_init_params(self):
        from semantic_clusterer import SemanticKSplit
        sig = inspect.signature(SemanticKSplit.__init__)
        # Remove 'self'
        params = dict(sig.parameters)
        params.pop("self", None)
        return params

    def test_embedding_model_param_exists_with_none_default(self):
        params = self._get_init_params()
        assert "embedding_model" in params
        p = params["embedding_model"]
        assert p.default is None

    def test_k_param_exists_and_is_required(self):
        params = self._get_init_params()
        assert "k" in params
        p = params["k"]
        # k is keyword-only with no default → required
        assert p.kind == inspect.Parameter.KEYWORD_ONLY
        assert p.default is inspect.Parameter.empty

    def test_config_param_exists_with_none_default(self):
        params = self._get_init_params()
        assert "config" in params
        p = params["config"]
        assert p.default is None

    def test_verbose_param_exists_with_false_default(self):
        params = self._get_init_params()
        assert "verbose" in params
        p = params["verbose"]
        assert p.default is False

    def test_random_state_param_exists_with_42_default(self):
        params = self._get_init_params()
        assert "random_state" in params
        p = params["random_state"]
        assert p.default == 42

    def test_embedding_model_is_positional_or_keyword(self):
        params = self._get_init_params()
        p = params["embedding_model"]
        assert p.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.POSITIONAL_ONLY,
        )

    def test_k_config_verbose_random_state_are_keyword_only(self):
        params = self._get_init_params()
        keyword_only = {
            name
            for name, p in params.items()
            if p.kind == inspect.Parameter.KEYWORD_ONLY
        }
        assert {"k", "config", "verbose", "random_state"}.issubset(keyword_only)

    def test_no_extra_params(self):
        """Constructor must not expose unexpected parameters beyond the specified set."""
        params = self._get_init_params()
        expected_core = {"embedding_model", "k", "config", "verbose", "random_state"}
        # quality is a v0.1.0 addition — allowed
        allowed_extra = {"quality"}
        allowed = expected_core | allowed_extra
        unexpected = set(params.keys()) - allowed
        assert not unexpected, (
            f"Unexpected params: {unexpected}"
        )


# ---------------------------------------------------------------------------
# Existing exports remain available and unchanged
# ---------------------------------------------------------------------------

class TestExistingExports:
    EXPECTED_LEGACY_EXPORTS = [
        "SemanticClusterer",
        "ClustererConfig",
        "ClusteringReport",
        "SUPPORTED_DIM_BANDS",
        "normalize_embedding_model",
        "validate_embeddings",
        "__version__",
    ]

    def test_all_legacy_names_in_dunder_all(self):
        import semantic_clusterer
        for name in self.EXPECTED_LEGACY_EXPORTS:
            assert name in semantic_clusterer.__all__, (
                f"Expected {name!r} in semantic_clusterer.__all__"
            )

    def test_semantic_clusterer_importable(self):
        from semantic_clusterer import SemanticClusterer  # noqa: F401
        assert SemanticClusterer is not None

    def test_clusterer_config_importable(self):
        from semantic_clusterer import ClustererConfig  # noqa: F401
        assert ClustererConfig is not None

    def test_clustering_report_importable(self):
        from semantic_clusterer import ClusteringReport  # noqa: F401
        assert ClusteringReport is not None

    def test_supported_dim_bands_importable(self):
        from semantic_clusterer import SUPPORTED_DIM_BANDS  # noqa: F401
        assert SUPPORTED_DIM_BANDS is not None

    def test_normalize_embedding_model_importable(self):
        from semantic_clusterer import normalize_embedding_model  # noqa: F401
        assert normalize_embedding_model is not None

    def test_validate_embeddings_importable(self):
        from semantic_clusterer import validate_embeddings  # noqa: F401
        assert validate_embeddings is not None

    def test_version_importable(self):
        from semantic_clusterer import __version__  # noqa: F401
        assert isinstance(__version__, str)
        assert len(__version__) > 0
