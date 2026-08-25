"""Base protocol and factory for dimensionality reduction."""

from typing import Literal, Optional, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class BaseReducer(Protocol):
    """Protocol for dimensionality reduction methods."""

    def fit(self, embeddings: np.ndarray) -> "BaseReducer":
        """Fit the reducer to the data.
        
        Args:
            embeddings: Array of shape (n_samples, n_features).
            
        Returns:
            Self for method chaining.
        """
        ...

    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        """Transform the embeddings to lower dimensions.
        
        Args:
            embeddings: Array of shape (n_samples, n_features).
            
        Returns:
            Reduced array of shape (n_samples, n_components).
        """
        ...

    def fit_transform(self, embeddings: np.ndarray) -> np.ndarray:
        """Fit and transform in one step.
        
        Args:
            embeddings: Array of shape (n_samples, n_features).
            
        Returns:
            Reduced array of shape (n_samples, n_components).
        """
        ...


def get_reducer(
    method: Optional[Literal["pca"]],
    n_components: int,
    n_samples: int,
) -> Optional[BaseReducer]:
    """Factory function to get the appropriate reducer.
    
    Args:
        method: Reduction method ("pca" or None).
        n_components: Number of dimensions to reduce to.
        n_samples: Number of samples.
        
    Returns:
        A reducer instance or None if method is None.
        
    Raises:
        ValueError: If an unknown method is specified.
    """
    if method is None:
        return None

    if n_components < 1:
        raise ValueError("n_components must be >= 1")

    if n_samples < 2:
        return None

    if method == "pca":
        from semantic_clusterer.reduction.pca import PCAReducer
        return PCAReducer(n_components=n_components)

    else:
        raise ValueError(f"Unknown reduction method: {method}")
