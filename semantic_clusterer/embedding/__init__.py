"""Embedding layer for semantic text representation."""

from semantic_clusterer.embedding.adapters import (
    CallableAdapter,
    EmbeddingModel,
    EncodeAdapter,
    LangchainAdapter,
    normalize_embedding_model,
    validate_embeddings,
)
from semantic_clusterer.embedding.base import BaseEmbedder
from semantic_clusterer.embedding.onnx_model import OnnxEmbedder

__all__ = [
    # Base protocol (legacy)
    "BaseEmbedder",
    # Built-in embedder
    "OnnxEmbedder",
    # Model-agnostic protocol
    "EmbeddingModel",
    # Adapters
    "EncodeAdapter",
    "LangchainAdapter",
    "CallableAdapter",
    # Normalization functions
    "normalize_embedding_model",
    "validate_embeddings",
]
