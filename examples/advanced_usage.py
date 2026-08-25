"""
Advanced Usage Example
======================

This example demonstrates advanced features of semantic_clusterer:
- Custom configuration
- Custom embedding models
- 4-Tier Scale-aware Strategies (Tiny/Small/Medium/Large)
"""

import hashlib
from typing import List

import numpy as np

from semantic_clusterer import ClustererConfig, SemanticClusterer


# Example: Custom embedding model
class CustomEmbedder:
    """Example custom embedding model.
    
    In practice, you might use:
    - sentence-transformers
    - OpenAI embeddings
    - Any model with an embed() method
    """

    def __init__(self, dim: int = 384):
        self.dim = dim
        # For demo: use random but consistent embeddings
        self._cache = {}

    def embed(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for texts."""
        embeddings = []
        for text in texts:
            if text not in self._cache:
                # Use deterministic seeding for reproducibility
                seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)
                rng = np.random.default_rng(seed)
                self._cache[text] = rng.standard_normal(self.dim).astype(np.float32)
            embeddings.append(self._cache[text])
        return np.array(embeddings)


def example_custom_config():
    """Example with custom configuration."""
    print("=" * 50)
    print("Custom Configuration")
    print("=" * 50)

    # Configure for specific use case
    # valid strategies: 'auto' (4-tier routing), 'small', 'medium', 'large' ('tiny' is internal only)
    config = ClustererConfig(
        strategy="small",          # Force small data strategy (HDBSCAN only)
        reduction=None,            # PCA is supported, None means no reduction
        batch_size=32,             # Smaller batches for limited memory
        normalize_embeddings=True, # Normalize for cosine similarity
    )

    clusterer = SemanticClusterer(config=config, verbose=True)

    texts = [
        "quarterly revenue analysis",
        "revenue breakdown by quarter",
        "user account settings",
        "manage user preferences",
        "weather alert system",
    ]

    results = clusterer.cluster(texts, return_format="detailed")

    for r in results:
        print(f"Cluster {r['cluster_id']}: {r['items']}")


def example_custom_embedder():
    """Example with custom embedding model."""
    print("\n" + "=" * 50)
    print("Custom Embedding Model")
    print("=" * 50)

    # Use your own embedding model
    custom_model = CustomEmbedder(dim=256)

    clusterer = SemanticClusterer(
        embedding_model=custom_model,
        verbose=True,
    )

    texts = [
        "machine learning tutorial",
        "deep learning guide",
        "cooking recipes",
        "baking instructions",
    ]

    clusters = clusterer.cluster(texts)

    for i, cluster in enumerate(clusters):
        print(f"Cluster {i}: {cluster}")


def example_dict_config():
    """Example with dict-based configuration."""
    print("\n" + "=" * 50)
    print("Dict-based Configuration")
    print("=" * 50)

    # Can also pass config as a dictionary
    clusterer = SemanticClusterer(
        config={
            "strategy": "auto",
            "reduction": "auto",
            "batch_size": 64,
            "normalize_embeddings": True,
        }
    )

    texts = ["hello", "hi", "goodbye", "bye"]
    clusters = clusterer.cluster(texts)
    print(f"Clusters: {clusters}")


def example_embedding_only():
    """Example of using just the embedding functionality."""
    print("\n" + "=" * 50)
    print("Embedding Only (no clustering)")
    print("=" * 50)

    custom_model = CustomEmbedder(dim=128)
    clusterer = SemanticClusterer(embedding_model=custom_model)

    texts = ["hello world", "hi there"]
    embeddings = clusterer.embed(texts)

    print(f"Embedding shape: {embeddings.shape}")
    print(f"First embedding (truncated): {embeddings[0][:5]}...")


def example_medium_strategy():
    """Example forcing medium data strategy."""
    print("\n" + "=" * 50)
    print("Medium Strategy (with reduction)")
    print("=" * 50)

    config = ClustererConfig(
        strategy="medium",  # Force reduction + HDBSCAN
        reduction="pca",    # PCA is the default reduction method
    )

    custom_model = CustomEmbedder(dim=384)
    clusterer = SemanticClusterer(
        embedding_model=custom_model,
        config=config,
        verbose=True,
    )

    # Generate more texts
    texts = []
    topics = ["revenue", "users", "weather", "marketing", "support"]
    for topic in topics:
        for i in range(10):
            texts.append(f"{topic} related query {i}")

    results = clusterer.cluster(texts, return_format="detailed")

    print(f"\nFound {len(results)} clusters:")
    for r in results:
        print(f"  Cluster {r['cluster_id']}: {r['size']} items, "
              f"confidence={r['confidence']:.2%}")


def main():
    """Run all advanced examples."""
    example_custom_config()
    example_custom_embedder()
    example_dict_config()
    example_embedding_only()
    example_medium_strategy()


if __name__ == "__main__":
    main()
