"""SemanticKSplit basics: partition a corpus into exactly k groups.

SemanticKSplit is a backward-compatibility shim over
``SemanticClusterer(n_clusters=k)``. Both APIs are demonstrated here.
Use the unified class for new code.

Run:
    python examples/04_ksplit_basic.py
"""

from semantic_clusterer import SemanticClusterer, SemanticKSplit


def main() -> None:
    texts = [
        # finance
        "monthly revenue report",
        "quarterly revenue summary",
        "annual revenue forecast",
        # users
        "list all users",
        "show user accounts",
        "fetch user details",
        # deployments
        "deploy to production",
        "rollback the release",
        "promote staging to prod",
    ]

    # --- v0.3.0 unified API (preferred) ---
    print("=== SemanticClusterer(n_clusters=3) ===\n")
    sc = SemanticClusterer(n_clusters=3, random_state=0)
    groups_unified = sc.cluster(texts)
    print(f"Got {len(groups_unified)} groups (expected exactly 3)")
    for i, group in enumerate(groups_unified):
        print(f"\n  Group {i} ({len(group)} items):")
        for item in group:
            print(f"    - {item}")

    # --- Backward-compat shim ---
    print("\n\n=== SemanticKSplit(k=3) [backward compat] ===\n")
    ks = SemanticKSplit(k=3, random_state=0)
    groups_ks = ks.split(texts)
    print(f"Got {len(groups_ks)} groups (expected exactly 3)")
    for i, group in enumerate(groups_ks):
        print(f"\n  Group {i} ({len(group)} items):")
        for item in group:
            print(f"    - {item}")


if __name__ == "__main__":
    main()
