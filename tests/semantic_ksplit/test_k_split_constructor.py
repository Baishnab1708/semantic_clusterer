"""Unit tests for SemanticKSplit constructor validation and config reconciliation.

Covers:
- Bad ``k`` values: str, bool, < 2
- Bad ``random_state`` values: bool, out of range
- Config reconciliation: dict path and ClustererConfig path
- ``embedding_model=None`` resolves to built-in ONNX OnnxEmbedder
- ``embedding_model`` with a callable / encode-style model accepted

"""

from __future__ import annotations

from typing import List, Sequence
from unittest.mock import patch

import numpy as np
import pytest

from semantic_clusterer.config import ClustererConfig
from semantic_clusterer.k_split import SemanticKSplit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeEmbedder:
    """Minimal embedding-model stub with a .embed() method.

    Returns deterministic float32 vectors derived from text length so
    no real model is needed.
    """

    def embed(self, texts: Sequence[str], batch_size: int = 64) -> np.ndarray:
        dim = 8
        out = np.zeros((len(texts), dim), dtype=np.float32)
        for i, t in enumerate(texts):
            rng = np.random.default_rng(len(t) % 1000 + i)
            row = rng.standard_normal(dim).astype(np.float32)
            norm = np.linalg.norm(row)
            out[i] = row / norm if norm > 0 else row
        return out


class _FakeEncodeModel:
    """Stub with .encode() method (SentenceTransformers-style)."""

    def encode(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        dim = 8
        out = np.zeros((len(texts), dim), dtype=np.float32)
        for i, t in enumerate(texts):
            rng = np.random.default_rng(len(t) % 1000 + i + 100)
            row = rng.standard_normal(dim).astype(np.float32)
            norm = np.linalg.norm(row)
            out[i] = row / norm if norm > 0 else row
        return out


def _callable_embedder(texts: List[str]) -> np.ndarray:
    """Callable embedding function."""
    dim = 8
    out = np.zeros((len(texts), dim), dtype=np.float32)
    for i, t in enumerate(texts):
        rng = np.random.default_rng(len(t) % 1000 + i + 200)
        row = rng.standard_normal(dim).astype(np.float32)
        norm = np.linalg.norm(row)
        out[i] = row / norm if norm > 0 else row
    return out


# ===========================================================================
# Bad k type: str and bool raise TypeError
# ===========================================================================


class TestKTypeValidation:
    """k must be a plain int; str and bool raise TypeError. (Req 4.1)"""

    def test_k_str_raises_type_error(self):
        """k='3' must raise TypeError with message containing 'k must be an int'."""
        with pytest.raises(TypeError, match="k must be an int"):
            SemanticKSplit(k="3")

    def test_k_float_raises_type_error(self):
        """k=3.0 must raise TypeError with message containing 'k must be an int'."""
        with pytest.raises(TypeError, match="k must be an int"):
            SemanticKSplit(k=3.0)

    def test_k_none_raises_type_error(self):
        """k=None must raise TypeError with message containing 'k must be an int'."""
        with pytest.raises(TypeError, match="k must be an int"):
            SemanticKSplit(k=None)

    def test_k_bool_true_raises_type_error(self):
        """k=True must raise TypeError; bool is a subclass of int but is rejected (Req 4.1)."""
        with pytest.raises(TypeError, match="k must be an int"):
            SemanticKSplit(k=True)

    def test_k_bool_false_raises_type_error(self):
        """k=False must raise TypeError."""
        with pytest.raises(TypeError, match="k must be an int"):
            SemanticKSplit(k=False)

    def test_k_list_raises_type_error(self):
        """k=[2] must raise TypeError."""
        with pytest.raises(TypeError, match="k must be an int"):
            SemanticKSplit(k=[2])


# ===========================================================================
# Bad k value: k < 2 raises ValueError
# ===========================================================================


class TestKValueValidation:
    """k must be >= 2; smaller values raise ValueError. (Req 4.2)"""

    def test_k_zero_raises_value_error(self):
        """k=0 must raise ValueError with message 'k must be >= 2'."""
        with pytest.raises(ValueError, match="k must be >= 2"):
            SemanticKSplit(k=0)

    def test_k_one_raises_value_error(self):
        """k=1 must raise ValueError with message 'k must be >= 2'."""
        with pytest.raises(ValueError, match="k must be >= 2"):
            SemanticKSplit(k=1)

    def test_k_negative_raises_value_error(self):
        """k=-5 must raise ValueError with message 'k must be >= 2'."""
        with pytest.raises(ValueError, match="k must be >= 2"):
            SemanticKSplit(k=-5)

    def test_k_two_is_accepted(self):
        """k=2 is the minimum valid value — must not raise."""
        ks = SemanticKSplit(k=2)
        assert ks._k == 2

    def test_k_large_is_accepted(self):
        """k=1000 must not raise at construction time (N not yet known)."""
        ks = SemanticKSplit(k=1000)
        assert ks._k == 1000


# ===========================================================================
# Bad random_state raises ValueError
# ===========================================================================


class TestRandomStateValidation:
    """random_state must be a plain int in [0, 2**32 - 1]. (Req 4.3)"""

    def test_random_state_bool_raises(self):
        """random_state=True must raise ValueError (bool subclasses int but is rejected)."""
        with pytest.raises(ValueError):
            SemanticKSplit(k=2, random_state=True)

    def test_random_state_false_raises(self):
        """random_state=False must raise ValueError."""
        with pytest.raises(ValueError):
            SemanticKSplit(k=2, random_state=False)

    def test_random_state_negative_raises(self):
        """random_state=-1 is out of range; must raise ValueError."""
        with pytest.raises(ValueError, match="random_state must be in"):
            SemanticKSplit(k=2, random_state=-1)

    def test_random_state_too_large_raises(self):
        """random_state=2**32 is out of range; must raise ValueError."""
        with pytest.raises(ValueError, match="random_state must be in"):
            SemanticKSplit(k=2, random_state=2**32)

    def test_random_state_str_raises(self):
        """random_state='42' must raise ValueError (wrong type)."""
        with pytest.raises(ValueError):
            SemanticKSplit(k=2, random_state="42")  # type: ignore[arg-type]

    def test_random_state_zero_is_accepted(self):
        """random_state=0 is the minimum valid seed."""
        ks = SemanticKSplit(k=2, random_state=0)
        assert ks._random_state == 0

    def test_random_state_max_is_accepted(self):
        """random_state=2**32 - 1 is the maximum valid seed."""
        ks = SemanticKSplit(k=2, random_state=2**32 - 1)
        assert ks._random_state == 2**32 - 1

    def test_random_state_default_42(self):
        """Default random_state must be 42."""
        ks = SemanticKSplit(k=2)
        assert ks._random_state == 42


# ===========================================================================
# Dict config reconciliation
# ===========================================================================


class TestDictConfigReconciliation:
    """dict config is validated and reconciled with kwarg precedence. (Req 3.5)"""

    def test_dict_config_valid_fields_accepted(self):
        """A valid dict config must be accepted without raising."""
        ks = SemanticKSplit(k=2, config={"batch_size": 32})
        assert ks.config.batch_size == 32

    def test_dict_config_random_state_kwarg_wins(self):
        """kwarg random_state wins over config dict value (Req 3.6 precedence)."""
        ks = SemanticKSplit(k=2, config={"random_state": 10}, random_state=99)
        assert ks._random_state == 99
        assert ks.config.random_state == 99

    def test_dict_config_invalid_field_raises(self):
        """A dict config with an unknown field must raise ValueError."""
        with pytest.raises(ValueError):
            SemanticKSplit(k=2, config={"nonexistent_field": True})

    def test_dict_config_verbose_merged(self):
        """verbose from dict and kwarg are merged with OR logic."""
        ks = SemanticKSplit(k=2, config={"verbose": False}, verbose=True)
        assert ks.verbose is True

    def test_dict_config_strategy_rejected(self):
        """strategy is internal — passing it via dict must raise ValueError."""
        with pytest.raises(ValueError):
            SemanticKSplit(k=2, config={"strategy": "medium"})

    def test_dict_config_batch_size_preserved(self):
        """batch_size from dict config is stored on self.config."""
        ks = SemanticKSplit(k=2, config={"batch_size": 32})
        assert ks.config.batch_size == 32

    def test_dict_config_none_uses_defaults(self):
        """config=None must produce a default ClustererConfig."""
        ks = SemanticKSplit(k=2, config=None)
        assert isinstance(ks.config, ClustererConfig)


# ===========================================================================
# ClustererConfig reconciliation
# ===========================================================================


class TestClustererConfigReconciliation:
    """ClustererConfig is accepted and kwarg random_state wins on conflict. (Req 3.6)"""

    def test_clusterer_config_accepted(self):
        """A ClustererConfig instance must be accepted without modification."""
        cfg = ClustererConfig(random_state=7)
        ks = SemanticKSplit(k=2, config=cfg, random_state=7)
        assert ks.config.random_state == 7

    def test_clusterer_config_kwarg_random_state_wins(self):
        """kwarg random_state overrides config.random_state when they differ."""
        cfg = ClustererConfig(random_state=10)
        ks = SemanticKSplit(k=2, config=cfg, random_state=55)
        assert ks._random_state == 55
        assert ks.config.random_state == 55

    def test_clusterer_config_verbose_merged(self):
        """verbose is merged: if either config.verbose or kwarg verbose is True, result is True."""
        cfg = ClustererConfig(verbose=False)
        ks = SemanticKSplit(k=2, config=cfg, verbose=True)
        assert ks.verbose is True

    def test_clusterer_config_verbose_false_both(self):
        """verbose is False when both config and kwarg are False."""
        cfg = ClustererConfig(verbose=False)
        ks = SemanticKSplit(k=2, config=cfg, verbose=False)
        assert ks.verbose is False

    def test_clusterer_config_default_random_state_unchanged(self):
        """When random_state kwarg equals default (42), config value is used if same."""
        cfg = ClustererConfig(random_state=42)
        ks = SemanticKSplit(k=2, config=cfg, random_state=42)
        assert ks._random_state == 42

    def test_clusterer_config_non_default_random_state_from_config(self):
        """When kwarg is default 42 and config has different value, kwarg wins."""
        cfg = ClustererConfig(random_state=99)
        # kwarg default=42 wins by design (kwarg always wins on conflict)
        ks = SemanticKSplit(k=2, config=cfg)
        # random_state default is 42; it replaces cfg.random_state=99
        assert ks._random_state == 42


# ===========================================================================
# Embedding_model=None resolves to built-in OnnxEmbedder
# ===========================================================================


class TestEmbeddingModelNoneResolvesToOnnx:
    """embedding_model=None must lazily resolve to built-in ONNX. (Req 3.3)"""

    def test_default_is_none_raw_model(self):
        """The raw embedding model attribute is None before first embed call."""
        ks = SemanticKSplit(k=2)
        assert ks._raw_embedding_model is None

    def test_embedder_not_initialized_at_construction(self):
        """_embedder is None at construction time (lazy init)."""
        ks = SemanticKSplit(k=2)
        assert ks._embedder is None

    def test_get_embedder_returns_onnx_model(self):
        """_get_embedder() must return an OnnxEmbedder when embedding_model=None."""
        from semantic_clusterer.embedding.onnx_model import OnnxEmbedder

        ks = SemanticKSplit(k=2)
        embedder = ks._get_embedder()
        assert isinstance(embedder, OnnxEmbedder)

    def test_second_call_returns_same_instance(self):
        """_get_embedder() must cache and return the same instance on repeated calls."""
        ks = SemanticKSplit(k=2)
        e1 = ks._get_embedder()
        e2 = ks._get_embedder()
        assert e1 is e2


# ===========================================================================
# Custom embedding_model variants are accepted
# ===========================================================================


class TestCustomEmbeddingModelAccepted:
    """Various embedding_model types accepted by normalize_embedding_model. (Req 3.4)"""

    def test_embed_method_object_accepted(self):
        """Object with .embed() must be accepted without raising."""
        ks = SemanticKSplit(k=2, embedding_model=_FakeEmbedder())
        assert ks._raw_embedding_model is not None

    def test_encode_method_object_accepted(self):
        """Object with .encode() (SentenceTransformers-style) must be accepted."""
        ks = SemanticKSplit(k=2, embedding_model=_FakeEncodeModel())
        assert ks._raw_embedding_model is not None

    def test_callable_embedding_accepted(self):
        """A callable function must be accepted as embedding_model."""
        ks = SemanticKSplit(k=2, embedding_model=_callable_embedder)
        assert ks._raw_embedding_model is not None

    def test_custom_model_custom_embedder_flag(self):
        """When a custom model is provided, _custom_embedder must be True."""
        ks = SemanticKSplit(k=2, embedding_model=_FakeEmbedder())
        assert ks._custom_embedder is True

    def test_no_model_custom_embedder_flag_false(self):
        """When embedding_model=None, _custom_embedder must be False."""
        ks = SemanticKSplit(k=2)
        assert ks._custom_embedder is False
