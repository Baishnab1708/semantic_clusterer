"""SemanticKSplit: row-aligned labels and full run report.

Demonstrates:
- ``split_labels`` for an int32 label array aligned to the input list
- ``split_with_report`` for the full ``ClusteringReport``
- handling of ``None`` / empty inputs (label `-1`)

Run:
    python examples/05_ksplit_labels_and_report.py
"""

from semantic_clusterer import SemanticKSplit


def main() -> None:
    texts = [
        "monthly revenue report",
        "revenue per month",
        None,                       # filtered -> label -1
        "list all users",
        "show user accounts",
        "",                         # filtered -> label -1
        "deploy to production",
        "rollback the release",
    ]

    ks = SemanticKSplit(k=3, random_state=42)

    labels = ks.split_labels(texts)
    print("split_labels:")
    for text, lbl in zip(texts, labels):
        print(f"  label={lbl}  text={text!r}")

    print("\nsplit_with_report:")
    labels, report = ks.split_with_report(texts)
    print(f"  requested_k:    {report.chosen_params.get('requested_k')}")
    print(f"  algorithm_used: {report.chosen_params.get('algorithm_used')}")
    print(f"  pipeline_tier:  {report.pipeline_tier}")
    print(f"  dim_band:       {report.dim_band}  (D={report.embedding_dim})")
    print(f"  n_clusters:     {report.n_clusters}")
    print(f"  n_noise (-1):   {report.n_noise}")
    print(f"  confidence:     {report.confidence_level}")

    sizes = report.intrinsic_metrics.get("per_cluster_size")
    if sizes is not None:
        print(f"  per_cluster_size: {sizes}")


if __name__ == "__main__":
    main()
