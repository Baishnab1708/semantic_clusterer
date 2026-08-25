"""
Basic Usage Example
===================

This example demonstrates the simplest way to use semantic_clusterer.
"""

from semantic_clusterer import SemanticClusterer


def main():
    # Sample texts to cluster
    texts = [
        # Revenue related
        "monthly revenue report",
        "revenue per month analysis",
        "total sales this month",
        "monthly income summary",

        # User management
        "list all users",
        "show user accounts",
        "display active users",
        "get user list",

        # Weather queries
        "what is the weather today",
        "weather forecast for tomorrow",
        "check current weather",

        # Random
        "hello world",
    ]

    # Create clusterer with default settings (zero-config)
    # Note: First run may download model files (~90MB)
    try:
        clusterer = SemanticClusterer()
    except Exception as exc:
        print(f"Failed to initialize SemanticClusterer: {exc}")
        return

    # Cluster the texts (simple output)
    print("=" * 50)
    print("Simple Output (List of Lists)")
    print("=" * 50)

    clusters = clusterer.cluster(texts)

    for i, cluster in enumerate(clusters):
        print(f"\nCluster {i}:")
        for text in cluster:
            print(f"  - {text}")

    # Detailed output with metadata
    print("\n" + "=" * 50)
    print("Detailed Output (with metadata)")
    print("=" * 50)

    detailed_results = clusterer.cluster(texts, return_format="detailed")

    for result in detailed_results:
        print(f"\nCluster {result['cluster_id']}:")
        print(f"  Representative: {result['representative']}")
        print(f"  Size: {result['size']}")
        print(f"  Confidence: {result['confidence']:.2%}")
        print("  Items:")
        for item in result['items']:
            print(f"    - {item}")


if __name__ == "__main__":
    main()
