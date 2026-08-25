"""Beginner: zero-config clustering with `SemanticClusterer`.

Run:
    python examples/01_beginner_zero_config.py
"""

from semantic_clusterer import SemanticClusterer


def main() -> None:
    texts = [
        "How do I reset my password?",
        "I forgot my password, please help",
        "Where is the password reset page?",
        "What are your business hours?",
        "When do you open?",
        "Are you open on weekends?",
        "My order hasn't arrived yet.",
        "Where is my package?",
        "Delivery is delayed.",
    ]

    clusterer = SemanticClusterer()
    clusters = clusterer.cluster(texts)

    print(f"Found {len(clusters)} clusters\n")
    for i, group in enumerate(clusters):
        print(f"Cluster {i + 1} ({len(group)} items):")
        for item in group:
            print(f"  - {item}")
        print()


if __name__ == "__main__":
    main()
