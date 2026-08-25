"""Benchmark SemanticClusterer (variable-K) across pipeline tiers.

One ``--tiers`` flag drives the whole run. Each tier maps to a concrete
(dataset, size) via TIER_PLAN in bench_common (20NG for tiny/small/medium,
AG News for large). Every run emits an extremely detailed JSON record.

Examples
--------
All tiers, built-in MiniLM::

    python benchmarks/run_clusterer.py --embedder minilm --tiers tiny small medium large

All tiers, mpnet (downloads model once)::

    python benchmarks/run_clusterer.py --embedder mpnet  --tiers tiny small medium large

Paid OpenAI, skip the expensive large tier::

    python benchmarks/run_clusterer.py --embedder openai3small --tiers tiny small medium
"""

from __future__ import annotations

import argparse
import warnings
from typing import Any, Dict, List

from bench_common import (
    ALL_TIERS,
    TIER_PLAN,
    Timer,
    build_embedder,
    environment,
    fmt,
    print_table,
    save_results,
    wrap_with_cache,
)
from bench_data import load_benchmark_dataset
from bench_report import build_detailed_record, headline

from semantic_clusterer import SemanticClusterer


def run_tier(
    tier: str,
    embedder: Any,
    embedder_info: Dict[str, Any],
    granularity: str,
    seed: int,
) -> Dict[str, Any]:
    """Run SemanticClusterer on the dataset/size mapped to ``tier``."""
    dataset, n_docs = TIER_PLAN[tier]
    print(f"\n=== [{tier}] {dataset} n_docs={n_docs} | {embedder_info['name']} ===")
    print("Loading dataset...")
    texts, labels, target_names = load_benchmark_dataset(dataset, n_docs=n_docs, seed=seed)
    print(f"Loaded {len(texts)} docs / {len(set(labels))} true classes. Clustering...")

    # Wrap with disk cache for ALL embedders.
    # Cache key = (alias, dataset, actual_n_docs, seed) so it's always exact.
    # First run: embeds and saves .npy. Every future run: loads from disk, free.
    active_embedder = wrap_with_cache(
        embedder, embedder_info["alias"], dataset, len(texts), seed
    )

    sc = SemanticClusterer(
        embedding_model=active_embedder,
        cluster_granularity=granularity,
        random_state=seed,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("default")
        with Timer() as t:
            pred, report = sc.cluster_with_report(texts)

    record = build_detailed_record(
        kind="SemanticClusterer",
        dataset=dataset,
        embedder_info=embedder_info,
        tier_requested=tier,
        texts=texts,
        true_labels=labels,
        pred_labels=pred,
        target_names=target_names,
        report=report,
        seconds=t.seconds,
    )
    record["granularity"] = granularity
    print(headline(record))
    return record


def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark SemanticClusterer across tiers.")
    ap.add_argument("--embedder", default="minilm",
                    help="minilm | mpnet | openai3small | any ST model name")
    ap.add_argument("--tiers", nargs="+", default=ALL_TIERS,
                    choices=ALL_TIERS, help="Which pipeline tiers to run.")
    ap.add_argument("--granularity", default="balanced",
                    choices=["fine", "balanced", "coarse"])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    embedder, info = build_embedder(args.embedder)
    print(f"Embedder: {info['name']}  | cost: {info['cost']}")
    print(f"Tiers:    {', '.join(args.tiers)}")

    records: List[Dict[str, Any]] = []
    for tier in args.tiers:
        records.append(run_tier(tier, embedder, info, args.granularity, args.seed))

    print("\n================ SemanticClusterer summary ================")
    print_table(
        [
            {
                "tier": r["tier_actual"],
                "n_docs": r["n_docs"],
                "true_cls": r["n_true_classes"],
                "pred_k": r["n_pred_clusters"],
                "ari": fmt(r["external_metrics"]["ari"]),
                "nmi": fmt(r["external_metrics"]["nmi"]),
                "v_measure": fmt(r["external_metrics"]["v_measure"]),
                "score": fmt(r["intrinsic_metrics"].get("score", 0.0)),
                "coverage": fmt(r["external_metrics"]["coverage"]),
                "conf": r["confidence_level"],
                "secs": r["seconds"],
            }
            for r in records
        ],
        columns=["tier", "n_docs", "true_cls", "pred_k", "ari", "nmi",
                 "v_measure", "score", "coverage", "conf", "secs"],
    )

    payload = {
        "kind": "SemanticClusterer",
        "embedder_alias": info.get("alias"),
        "embedder_name": info.get("name"),
        "granularity": args.granularity,
        "seed": args.seed,
        "environment": environment(),
        "runs": records,
    }
    out = save_results(f"clusterer_{info.get('alias', args.embedder)}.json", payload)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
