"""Benchmark SemanticKSplit (fixed-K) across pipeline tiers.

Mirrors run_clusterer.py. Because SemanticKSplit needs k up front, k is set to
the number of true topic classes in the dataset (the fairest fixed-K setting),
unless overridden with --k. Every run emits an extremely detailed JSON record.

Examples
--------
    python benchmarks/run_ksplit.py --embedder minilm --tiers tiny small medium large
    python benchmarks/run_ksplit.py --embedder mpnet  --tiers tiny small medium large
    python benchmarks/run_ksplit.py --embedder openai3small --tiers tiny small medium
"""

from __future__ import annotations

import argparse
import warnings
from typing import Any, Dict, List, Optional

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

from semantic_clusterer import SemanticKSplit


def run_tier(
    tier: str,
    embedder: Any,
    embedder_info: Dict[str, Any],
    quality: str,
    seed: int,
    k_override: Optional[int],
) -> Dict[str, Any]:
    """Run SemanticKSplit on the dataset/size mapped to ``tier``."""
    dataset, n_docs = TIER_PLAN[tier]
    texts, labels, target_names = load_benchmark_dataset(dataset, n_docs=n_docs, seed=seed)
    k = k_override if k_override else len(set(labels))
    print(f"\n=== [{tier}] {dataset} n_docs={len(texts)} k={k} | {embedder_info['name']} ===")
    print("Splitting...")

    # Wrap with disk cache for ALL embedders.
    # Reuses cache files already created by run_clusterer.py for the same
    # (alias, dataset, n_docs, seed) — ksplit pays zero embedding cost if
    # clusterer already ran first.
    active_embedder = wrap_with_cache(
        embedder, embedder_info["alias"], dataset, len(texts), seed
    )

    ks = SemanticKSplit(
        embedding_model=active_embedder,
        k=k,
        quality=quality,
        random_state=seed,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("default")
        with Timer() as t:
            pred, report = ks.split_with_report(texts)

    record = build_detailed_record(
        kind="SemanticKSplit",
        dataset=dataset,
        embedder_info=embedder_info,
        tier_requested=tier,
        texts=texts,
        true_labels=labels,
        pred_labels=pred,
        target_names=target_names,
        report=report,
        seconds=t.seconds,
        k=k,
    )
    record["quality"] = quality
    record["algorithm_used"] = report.chosen_params.get("algorithm_used", "?")
    print(headline(record))
    return record


def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark SemanticKSplit across tiers.")
    ap.add_argument("--embedder", default="minilm",
                    help="minilm | mpnet | openai3small | any ST model name")
    ap.add_argument("--tiers", nargs="+", default=ALL_TIERS,
                    choices=ALL_TIERS, help="Which pipeline tiers to run.")
    ap.add_argument("--quality", default="balanced",
                    choices=["fast", "balanced", "best"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--k", type=int, default=None,
                    help="Override k. Default: number of true classes.")
    args = ap.parse_args()

    embedder, info = build_embedder(args.embedder)
    print(f"Embedder: {info['name']}  | cost: {info['cost']}")
    print(f"Tiers:    {', '.join(args.tiers)}")

    records: List[Dict[str, Any]] = []
    for tier in args.tiers:
        records.append(run_tier(tier, embedder, info, args.quality, args.seed, args.k))

    print("\n================ SemanticKSplit summary ================")
    print_table(
        [
            {
                "tier": r["tier_actual"],
                "n_docs": r["n_docs"],
                "k": r["k"],
                "algorithm": r.get("algorithm_used", "?"),
                "ari": fmt(r["external_metrics"]["ari"]),
                "nmi": fmt(r["external_metrics"]["nmi"]),
                "v_measure": fmt(r["external_metrics"]["v_measure"]),
                "homog": fmt(r["external_metrics"]["homogeneity"]),
                "compl": fmt(r["external_metrics"]["completeness"]),
                "score": fmt(r["intrinsic_metrics"].get("score", 0.0)),
                "secs": r["seconds"],
            }
            for r in records
        ],
        columns=["tier", "n_docs", "k", "algorithm", "ari", "nmi", "v_measure",
                 "homog", "compl", "score", "secs"],
    )

    payload = {
        "kind": "SemanticKSplit",
        "embedder_alias": info.get("alias"),
        "embedder_name": info.get("name"),
        "quality": args.quality,
        "seed": args.seed,
        "environment": environment(),
        "runs": records,
    }
    out = save_results(f"ksplit_{info.get('alias', args.embedder)}.json", payload)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
