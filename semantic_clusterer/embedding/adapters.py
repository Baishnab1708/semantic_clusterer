"""Model-agnostic embedding adapters.

Normalizes different embedding model interfaces into a consistent interface.

Supported model types:
    - Objects with `.embed(texts)` method
    - Objects with `.encode(texts)` method (SentenceTransformers/HuggingFace)
    - Objects with `.embed_documents(texts)` method (LangChain)
    - Callable functions `fn(texts) -> embeddings`

Each adapter handles batch_size appropriately:
    - EncodeAdapter: Passes to HF/SentenceTransformers natively
    - LangchainAdapter: Ignores (LangChain handles batching internally)
    - CallableAdapter: Manual chunking to protect custom APIs
    - NativeEmbedAdapter: Auto-detects support, falls back gracefully
"""

from collections.abc import Sequence
from typing import Any, Callable, Optional, Protocol

import numpy as np


class EmbeddingModel(Protocol):
    """Protocol for the internal embedding interface."""

    def embed(self, texts: Sequence[str], batch_size: int = 64) -> np.ndarray:
        """Generate embeddings for texts.
        
        Args:
            texts: Text strings to embed.
            batch_size: Batch size hint for processing.
            
        Returns:
            Numpy array of shape (n_texts, embedding_dim).
        """
        ...


class EncodeAdapter:
    """Adapter for models with `.encode()` method (SentenceTransformers/HuggingFace).
    
    Passes batch_size natively to the model for optimal GPU performance.
    """

    __slots__ = ("model",)

    def __init__(self, model: Any) -> None:
        self.model = model

    def embed(self, texts: Sequence[str], batch_size: int = 64) -> np.ndarray:
        """Generate embeddings using .encode() with native batching."""
        texts_list = list(texts) if not isinstance(texts, list) else texts
        return np.asarray(self.model.encode(texts_list, batch_size=batch_size))


class LangchainAdapter:
    """Adapter for LangChain embedding models.
    
    Ignores batch_size - LangChain handles batching internally.
    """

    __slots__ = ("model",)

    def __init__(self, model: Any) -> None:
        self.model = model

    def embed(self, texts: Sequence[str], batch_size: int = 64) -> np.ndarray:
        """Generate embeddings using .embed_documents()."""
        texts_list = list(texts) if not isinstance(texts, list) else texts
        return np.asarray(self.model.embed_documents(texts_list))


class CallableAdapter:
    """Adapter for callable embedding functions.
    
    Uses manual chunking to protect custom APIs from payload limits.
    """

    __slots__ = ("fn",)

    def __init__(self, fn: Callable[[Sequence[str]], np.ndarray]) -> None:
        self.fn = fn

    def embed(self, texts: Sequence[str], batch_size: int = 64) -> np.ndarray:
        """Generate embeddings with manual chunking for safety."""
        texts_list = list(texts) if not isinstance(texts, list) else texts

        # Single batch - no chunking needed
        if len(texts_list) <= batch_size:
            return np.asarray(self.fn(texts_list))

        # Manual chunking to protect custom APIs
        all_embeddings = []
        for i in range(0, len(texts_list), batch_size):
            chunk = texts_list[i : i + batch_size]
            all_embeddings.append(np.asarray(self.fn(chunk)))

        return np.vstack(all_embeddings)


class NativeEmbedAdapter:
    """Adapter for models with existing .embed() method.
    
    Auto-detects if model accepts batch_size, falls back gracefully if not.
    """

    __slots__ = ("model", "_accepts_batch_size")

    def __init__(self, model: Any) -> None:
        self.model = model
        self._accepts_batch_size: Optional[bool] = None

    def embed(self, texts: Sequence[str], batch_size: int = 64) -> np.ndarray:
        """Generate embeddings, passing batch_size if supported."""
        texts_list = list(texts) if not isinstance(texts, list) else texts

        # Detect batch_size support on first call
        if self._accepts_batch_size is None:
            try:
                result = self.model.embed(texts_list, batch_size=batch_size)
                self._accepts_batch_size = True
                return np.asarray(result)
            except TypeError:
                self._accepts_batch_size = False
                return np.asarray(self.model.embed(texts_list))

        # Use cached result
        if self._accepts_batch_size:
            return np.asarray(self.model.embed(texts_list, batch_size=batch_size))
        return np.asarray(self.model.embed(texts_list))


def normalize_embedding_model(model: Any) -> EmbeddingModel:
    """Normalize any supported embedding model to the standard interface.
    
    Args:
        model: The embedding model. Supports:
            - Object with .embed() method
            - Object with .encode() method (SentenceTransformers)
            - Object with .embed_documents() method (LangChain)
            - Callable function
    
    Returns:
        Adapter implementing EmbeddingModel protocol.
    
    Raises:
        TypeError: If model is None or unsupported type.
    """
    # None is not handled here - caller owns default model creation
    if model is None:
        raise TypeError(
            "embedding_model cannot be None in normalize_embedding_model(). "
            "Use SemanticClusterer() for default ONNX embedder."
        )

    # Reject basic Python types that might have conflicting methods
    # (e.g., str has .encode() but it's not an embedding model)
    if isinstance(model, (str, bytes, dict, list, tuple, set, frozenset)):
        raise TypeError(
            f"Unsupported embedding_model type: {type(model).__name__}.\n"
            "Expected an embedding model object or callable, not a basic Python type."
        )

    # Already has .embed() method - wrap to handle batch_size gracefully
    if hasattr(model, "embed") and callable(model.embed):
        return NativeEmbedAdapter(model)

    # SentenceTransformers / HuggingFace style (.encode method)
    if hasattr(model, "encode") and callable(model.encode):
        return EncodeAdapter(model)

    # LangChain style (.embed_documents method)
    if hasattr(model, "embed_documents") and callable(model.embed_documents):
        return LangchainAdapter(model)

    # Callable function
    if callable(model):
        return CallableAdapter(model)

    raise TypeError(
        "Unsupported embedding_model type.\n"
        "Expected one of:\n"
        "  - object with .embed(texts) method\n"
        "  - object with .encode(texts) method (SentenceTransformers)\n"
        "  - object with .embed_documents(texts) method (LangChain)\n"
        "  - callable function: fn(texts) -> embeddings\n"
        f"Got: {type(model).__name__}"
    )


def validate_embeddings(
    embeddings: Any,
    texts: Sequence[str],
    *,
    allow_empty: bool = False,
) -> np.ndarray:
    """Validate embeddings and convert to float32 numpy array.
    
    Ensures embeddings are 2D, match text count, and contain no NaN/Inf.
    
    Args:
        embeddings: Embeddings to validate.
        texts: Original texts (for length validation).
        allow_empty: Allow empty inputs.
    
    Returns:
        Validated float32 numpy array.
    
    Raises:
        ValueError: Invalid shape, values, or empty input.
        TypeError: Cannot convert to numpy array.
    """
    if len(texts) == 0:
        if allow_empty:
            return np.empty((0, 0), dtype=np.float32)
        raise ValueError("Input texts cannot be empty")

    # Convert to numpy array if needed
    if not isinstance(embeddings, np.ndarray):
        try:
            embeddings = np.asarray(embeddings)
        except (ValueError, TypeError) as e:
            raise TypeError(
                f"Embeddings must be a numpy array or array-like. "
                f"Got {type(embeddings).__name__}: {e}"
            )

    # Handle single text case (1D array -> 2D)
    if embeddings.ndim == 1:
        if len(texts) == 1:
            embeddings = embeddings.reshape(1, -1)
        else:
            raise ValueError(
                f"Embeddings must be 2D. Got 1D array of shape {embeddings.shape}. "
                f"For single text, return shape (1, dim) not (dim,)"
            )

    # Must be 2D
    if embeddings.ndim != 2:
        raise ValueError(
            f"Embeddings must be 2D array of shape (n_texts, embedding_dim). "
            f"Got {embeddings.ndim}D array with shape {embeddings.shape}"
        )

    # Check count matches
    if embeddings.shape[0] != len(texts):
        raise ValueError(
            f"Embedding count mismatch: got {embeddings.shape[0]} embeddings "
            f"for {len(texts)} texts"
        )

    # Validate dtype before other numeric checks - reject non-numeric types
    if not np.issubdtype(embeddings.dtype, np.number):
        raise TypeError(
            f"Embeddings must be numeric, got dtype {embeddings.dtype}. "
            "Ensure your embedding model returns numeric arrays, not strings or objects."
        )

    # Check for NaN/Inf (only valid for numeric types)
    if not np.isfinite(embeddings).all():
        nan_count = np.sum(np.isnan(embeddings))
        inf_count = np.sum(np.isinf(embeddings))
        raise ValueError(
            f"Embeddings contain invalid values: {nan_count} NaN, {inf_count} Inf. "
            "Check your embedding model output."
        )

    # Convert to float32 for consistent memory usage
    return embeddings.astype(np.float32)


__all__ = [
    "EmbeddingModel",
    "EncodeAdapter",
    "LangchainAdapter",
    "CallableAdapter",
    "normalize_embedding_model",
    "validate_embeddings",
]
