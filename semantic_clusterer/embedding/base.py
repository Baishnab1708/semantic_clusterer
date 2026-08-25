"""Base protocol for embedding models."""

from typing import List, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class BaseEmbedder(Protocol):
    """Protocol for embedding models.
    
    Implement this to create custom embedders for SemanticClusterer.
    The batch_size parameter is optional for backward compatibility.
    """

    def embed(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        """Generate embeddings for texts.
        
        Args:
            texts: Text strings to embed.
            batch_size: Batch size hint (optional).
            
        Returns:
            Numpy array of shape (n_texts, embedding_dim).
        """
        ...
