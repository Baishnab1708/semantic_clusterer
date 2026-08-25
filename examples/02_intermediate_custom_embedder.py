"""Intermediate: custom embedder + detailed output with topic keywords.

Bring your own embedding model. Anything with an
``embed(texts) -> np.ndarray`` (or ``encode``, or ``embed_documents``) method
works.  This example also shows the v0.3.0 ``keywords`` and ``topic_label``
fields that are automatically populated in the detailed return format.

Run:
    python examples/02_intermediate_custom_embedder.py
"""

import numpy as np

from semantic_clusterer import SemanticClusterer


class MockEmbedder:
    """Toy embedder that returns deterministic random vectors.

    Replace this with a real model — SentenceTransformers, OpenAI,
    or any object with ``.embed()`` / ``.encode()`` / ``.embed_documents()``.
    """

    def embed(self, texts):
        # Deterministic seed so output is reproducible in demos.
        rng = np.random.default_rng(42)
        return rng.random((len(texts), 384), dtype=np.float32)


def main() -> None:
    texts = [
        # Tech news cluster
        "Apple releases new iPhone model",
        "Tech giant announces latest smartphone",
        "The new iOS update is now available",
        # Cooking / food cluster
        "Healthy apple pie recipe",
        "How to bake an apple pie from scratch",
        "Best dessert recipes for fall",
    ]

    clusterer = SemanticClusterer(embedding_model=MockEmbedder())
    clusters = clusterer.cluster(texts, return_format="detailed")

    print(f"Found {len(clusters)} cluster(s)\n")
    for c in clusters:
        print(f"Cluster {c['cluster_id']}  —  {c.get('topic_label', '(no label)')}")
        print(f"  size:           {c['size']}")
        print(f"  confidence:     {c.get('confidence', 0.0):.2f}")
        print(f"  representative: {c['representative']}")
        # c-TF-IDF keywords (v0.3.0) — top 5 shown
        kw = ", ".join(c.get("keywords", [])[:5]) or "(none)"
        print(f"  keywords:       {kw}")
        print("  items:")
        for item in c["items"]:
            print(f"    - {item}")
        print()


if __name__ == "__main__":
    main()
