"""Pytest fixtures for semantic_clusterer tests."""

from typing import List

import numpy as np
import pytest


@pytest.fixture
def sample_texts() -> List[str]:
    """Sample texts for testing clustering."""
    return [
        # Cluster 1: Revenue related
        "monthly revenue",
        "revenue per month",
        "total sales monthly",
        "monthly income report",

        # Cluster 2: User management
        "list users",
        "show all users",
        "display user list",
        "get user accounts",

        # Cluster 3: Weather
        "what is the weather",
        "weather forecast today",
        "check weather conditions",

        # Outlier
        "random unrelated text",
    ]


@pytest.fixture
def sample_embeddings() -> np.ndarray:
    """Mock embeddings for testing (skips actual model inference)."""
    np.random.seed(42)

    # Create clustered embeddings
    n_clusters = 3
    n_per_cluster = 4
    n_noise = 1
    dim = 384

    embeddings = []

    for i in range(n_clusters):
        # Cluster center
        center = np.random.randn(dim)
        center = center / np.linalg.norm(center)

        # Points near center
        for _ in range(n_per_cluster):
            noise = np.random.randn(dim) * 0.1
            point = center + noise
            point = point / np.linalg.norm(point)
            embeddings.append(point)

    # Add noise point far from clusters
    noise_point = np.random.randn(dim)
    noise_point = noise_point / np.linalg.norm(noise_point)
    embeddings.append(noise_point)

    return np.array(embeddings, dtype=np.float32)


@pytest.fixture
def mock_embedder(sample_embeddings):
    """Mock embedder that returns pre-computed embeddings."""
    class MockEmbedder:
        def __init__(self, embeddings):
            self._embeddings = embeddings
            self._call_count = 0

        def embed(self, texts: List[str]) -> np.ndarray:
            self._call_count += 1
            n = len(texts)
            # Return the first n embeddings
            return self._embeddings[:n].copy()

    return MockEmbedder(sample_embeddings)


@pytest.fixture
def large_sample_texts() -> List[str]:
    """Large sample for scalability testing."""
    base_texts = [
        "monthly revenue report",
        "quarterly sales analysis",
        "user account management",
        "weather forecast service",
        "data processing pipeline",
    ]

    # Generate variations
    texts = []
    for i in range(100):
        for base in base_texts:
            texts.append(f"{base} {i}")
    return texts


@pytest.fixture
def filtered_texts() -> List[str]:
    """Texts that should be filtered out during preprocessing."""
    return ["", "a", "  "]


@pytest.fixture
def custom_dim_embedder():
    """Custom embedder with non-384 dimension for testing."""
    class CustomDimEmbedder:
        def __init__(self, dim: int = 128):
            self.dim = dim
            self._call_count = 0

        def embed(self, texts: List[str]) -> np.ndarray:
            self._call_count += 1
            np.random.seed(42)
            return np.random.randn(len(texts), self.dim).astype(np.float32)

    return CustomDimEmbedder(dim=128)


@pytest.fixture
def clustered_labels() -> np.ndarray:
    """Pre-computed cluster labels for testing."""
    # 4 items per cluster + 1 noise
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, -1], dtype=np.int32)
    return labels


# ---------------------------------------------------------------------------
# Task 8.5 — Stub deterministic embedders for property-based tests
# ---------------------------------------------------------------------------

class StubDeterministicEmbedder:
    """Deterministic stub embedder producing L2-normalised float32 vectors.

    Uses a fixed seed so every call with the same texts returns the same
    embeddings.  The embedding is derived from a hash of the text index so
    different texts get different (but reproducible) vectors.
    """

    DIM: int = 384

    def __init__(self, seed: int = 0):
        self._seed = seed

    def embed(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        n = len(texts)
        rng = np.random.default_rng(self._seed)
        # Generate a fixed pool large enough for any call
        pool = rng.standard_normal((max(n, 1), self.DIM)).astype(np.float32)
        norms = np.linalg.norm(pool, axis=1, keepdims=True)
        pool = pool / np.where(norms == 0, 1.0, norms)
        return pool[:n].copy()


class StubStrongerEmbedder:
    """Stub embedder that produces tighter clusters than StubDeterministicEmbedder.

    Generates embeddings with lower intra-cluster variance so the intrinsic
    score is provably >= the default embedder's score on the same fixture.
    """

    DIM: int = 384

    def __init__(self, seed: int = 1, n_clusters: int = 5):
        self._seed = seed
        self._n_clusters = n_clusters

    def embed(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        n = len(texts)
        rng = np.random.default_rng(self._seed)
        # Create tight cluster centres
        centers = rng.standard_normal((self._n_clusters, self.DIM)).astype(np.float32)
        norms = np.linalg.norm(centers, axis=1, keepdims=True)
        centers = centers / np.where(norms == 0, 1.0, norms)

        out = np.empty((n, self.DIM), dtype=np.float32)
        for i in range(n):
            c = centers[i % self._n_clusters]
            # Very small noise → tight clusters → higher intrinsic score
            noise = rng.standard_normal(self.DIM).astype(np.float32) * 0.02
            v = c + noise
            out[i] = v / max(float(np.linalg.norm(v)), 1e-8)
        return out


@pytest.fixture
def stub_embedder() -> StubDeterministicEmbedder:
    """Pytest fixture: deterministic stub embedder (seed=0)."""
    return StubDeterministicEmbedder(seed=0)


@pytest.fixture
def stub_stronger_embedder() -> StubStrongerEmbedder:
    """Pytest fixture: stronger stub embedder with tighter clusters (seed=1)."""
    return StubStrongerEmbedder(seed=1)
