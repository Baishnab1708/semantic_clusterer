"""SemanticKSplit with a custom embedder + detailed output.

Any object with ``embed(texts) -> np.ndarray`` (or ``encode``,
``embed_documents``) plugs in as the embedding model.

Run:
    python examples/06_ksplit_custom_embedder.py
"""

import numpy as np

from semantic_clusterer import SemanticKSplit


class MockEmbedder:
    """Toy embedder for demo purposes; replace with a real model."""

    def embed(self, texts):
        rng = np.random.default_rng(0)
        return rng.random((len(texts), 384), dtype=np.float32)


def main() -> None:
    texts = [
        "Apple releases new iPhone model",
        "Tech giant announces latest smartphone",
        "The new iOS update is available",
        "Healthy apple pie recipe",
        "How to bake an apple pie from scratch",
        "Best dessert recipes for fall",
    ]

    ks = SemanticKSplit(embedding_model=MockEmbedder(), k=2, random_state=0)
    detailed = ks.split(texts, return_format="detailed")

    print(f"Got {len(detailed)} clusters (expected exactly 2)\n")
    for c in detailed:
        print(f"Cluster {c['cluster_id']} (size={c['size']}, "
              f"confidence={c['confidence']:.2f}):")
        print(f"  representative: {c['representative']}")
        for item in c["items"]:
            print(f"  - {item}")
        print()


if __name__ == "__main__":
    main()
