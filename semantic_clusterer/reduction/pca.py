"""PCA-based dimensionality reduction."""

import numpy as np
from sklearn.decomposition import PCA


class PCAReducer:
    """PCA-based dimensionality reduction.
    
    Wrapper around sklearn PCA for consistent interface.
    
    Advantages:
    - Fast and deterministic
    - Good for dense embeddings
    - Linear transformation preserves global structure
    
    Attributes:
        n_components: Number of dimensions to reduce to.
        random_state: Random seed for reproducibility.
    """

    def __init__(
        self,
        n_components: int = 50,
        random_state: int = 42,
    ):
        """Initialize PCA reducer.
        
        Args:
            n_components: Target dimensionality.
            random_state: Random seed.
        """
        self.n_components = n_components
        self.random_state = random_state
        self._pca = PCA(
            n_components=n_components,
            random_state=random_state,
            svd_solver="auto",
        )
        self._is_fitted = False

    def fit(self, embeddings: np.ndarray) -> "PCAReducer":
        """Fit PCA to the embeddings.
        
        Args:
            embeddings: Array of shape (n_samples, n_features).
            
        Returns:
            Self for method chaining.
        """
        if embeddings.shape[0] == 0:
            raise ValueError("embeddings must contain at least one sample")

        # Adjust n_components if necessary
        n_components = max(1, min(
            self.n_components,
            embeddings.shape[0],
            embeddings.shape[1]
        ))

        if n_components != self._pca.n_components:
            self._pca = PCA(
                n_components=n_components,
                random_state=self.random_state,
                svd_solver="auto",
            )

        self._pca.fit(embeddings)
        self._is_fitted = True
        return self

    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        """Transform embeddings to lower dimensions.
        
        Args:
            embeddings: Array of shape (n_samples, n_features).
            
        Returns:
            Reduced array of shape (n_samples, n_components).
        """
        if not self._is_fitted:
            raise RuntimeError("PCAReducer must be fitted before transform")

        return self._pca.transform(embeddings).astype(np.float32)

    def fit_transform(self, embeddings: np.ndarray) -> np.ndarray:
        """Fit and transform in one step.
        
        Args:
            embeddings: Array of shape (n_samples, n_features).
            
        Returns:
            Reduced array of shape (n_samples, n_components).
        """
        if embeddings.shape[0] == 0:
            raise ValueError("embeddings must contain at least one sample")

        # Adjust n_components if necessary
        n_components = max(1, min(
            self.n_components,
            embeddings.shape[0],
            embeddings.shape[1]
        ))

        if n_components != self._pca.n_components:
            self._pca = PCA(
                n_components=n_components,
                random_state=self.random_state,
                svd_solver="auto",
            )

        result = self._pca.fit_transform(embeddings)
        self._is_fitted = True
        return result.astype(np.float32)

    @property
    def explained_variance_ratio(self) -> np.ndarray:
        """Get the explained variance ratio of each component."""
        if not self._is_fitted:
            raise RuntimeError("PCAReducer must be fitted first")
        return self._pca.explained_variance_ratio_
