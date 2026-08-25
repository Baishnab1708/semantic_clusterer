"""Advanced: full pipeline control with SemanticClustererConfig + run report.

Demonstrates:
  - Strategy and reduction overrides via SemanticClustererConfig
  - cluster_granularity preset to control output coarseness
  - Pinning HDBSCAN via min_cluster_size and min_samples
  - Deterministic seeding with random_state
  - Inspecting a ClusteringReport (timings, metrics, chosen params)

Run:
    python examples/03_advanced_full_control.py
"""

import json

from semantic_clusterer import SemanticClustererConfig, SemanticClusterer


def main() -> None:
    base_texts = [
        "Fixing login issues on the web portal",
        "Cannot access my account online",
        "Web app login is down",
        "Setting up a new database cluster",
        "Database connection timeout error",
        "Optimizing SQL queries for the DB",
    ]
    # Repeat to build a corpus large enough for the small pipeline tier.
    texts = base_texts * 50

    config = SemanticClustererConfig(
        cluster_granularity="balanced",   # "fine" | "balanced" | "coarse"
        batch_size=32,
        normalize_embeddings=True,
        # v0.1.0: pin HDBSCAN's most-tuned parameter explicitly
        min_cluster_size=3,
        min_samples=2,
        verbose=True,
        random_state=42,
    )

    clusterer = SemanticClusterer(config=config)
    labels, report = clusterer.cluster_with_report(texts)

    print(f"\nReport summary")
    print(f"  inputs:            {report.n_input_texts}")
    print(f"  clustered:         {report.n_clustered}")
    print(f"  noise:             {report.n_noise}")
    print(f"  clusters:          {report.n_clusters}")
    print(f"  pipeline:          {report.pipeline_tier}")
    print(f"  dim_band:          {report.dim_band}  (D={report.embedding_dim})")
    print(f"  confidence_level:  {report.confidence_level}")
    print(f"  random_state:      {report.random_state}")

    if report.phase_timings:
        print("\n  Phase timings (seconds):")
        for phase, secs in sorted(report.phase_timings.items()):
            print(f"    {phase:<20} {secs:.3f}s")

    chosen = report.chosen_params
    print("\n  Chosen params (subset):")
    for key in ("pipeline_tier", "hdbscan_min_cluster_size", "hdbscan_min_samples"):
        if key in chosen:
            print(f"    {key}: {chosen[key]}")

    if report.warnings:
        print("\n  Warnings:")
        for w in report.warnings:
            print(f"    - {w}")

    # Full report is JSON-coercible (NaN / Inf → None, numpy → Python types)
    report_dict = report.to_dict()
    _ = json.dumps(report_dict)   # would raise if any non-serialisable value slipped through
    print("\n  report.to_dict() is JSON-serialisable ✓")


if __name__ == "__main__":
    main()
