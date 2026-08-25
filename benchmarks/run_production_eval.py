"""Benchmark the production API (.fit and .predict) for SemanticClusterer and SemanticKSplit.

Evaluates how well the clustering generalizes from a training split to a test split.
Reuses the cached full-dataset embeddings to ensure fast and cheap (zero API cost) runs.

Usage:
    python benchmarks/run_production_eval.py --embedder minilm --tier tiny
    python benchmarks/run_production_eval.py --embedder mpnet --tier tiny
    python benchmarks/run_production_eval.py --embedder openai3small --tier tiny
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.model_selection import train_test_split

# Import helpers from benchmark suite
from bench_common import (
    ALL_TIERS,
    TIER_PLAN,
    Timer,
    build_embedder,
    cache_exists,
    environment,
    fmt,
    load_cache,
    print_table,
    save_cache,
    save_results,
)
from bench_data import load_benchmark_dataset
from bench_metrics import external_metrics
from semantic_clusterer import SemanticClusterer, SemanticKSplit
from semantic_clusterer.preprocessing.clean import TextPreprocessor


class LookupOrComputeEmbedder:
    """An embedder proxy that reuses pre-computed full-dataset embeddings from disk cache.

    If a text query is not found in the pre-loaded cache (or if the cache file
    doesn't exist), it falls back to embedding it via the actual underlying model.
    """

    def __init__(
        self,
        inner_embedder: Any,
        alias: str,
        dataset: str,
        full_texts: List[str],
        seed: int,
        preprocessor: TextPreprocessor,
    ):
        self.inner_embedder = inner_embedder
        self.alias = alias
        self.dataset = dataset
        self.full_texts = full_texts
        self.seed = seed
        self.preprocessor = preprocessor
        self.resolved_inner = None
        self.text_to_emb: Dict[str, np.ndarray] = {}

        # 1. Preprocess full_texts using the exact same logic
        processed_full, raw_to_proc, _ = self.preprocessor.preprocess(full_texts, deduplicate=True)

        # 2. Try loading full dataset cache
        n_docs = len(full_texts)
        if cache_exists(alias, dataset, n_docs, seed):
            full_embeddings = load_cache(alias, dataset, n_docs, seed)
            if len(processed_full) == len(full_embeddings):
                print(f"  [LookupEmbedder] Successfully loaded cached {full_embeddings.shape} for '{alias}'")
                # Map BOTH raw and cleaned texts to the corresponding embedding
                for raw_idx, proc_idx in raw_to_proc.items():
                    if proc_idx >= 0:
                        raw_txt = full_texts[raw_idx]
                        clean_txt = processed_full[proc_idx]
                        emb = full_embeddings[proc_idx]
                        self.text_to_emb[raw_txt] = emb
                        self.text_to_emb[clean_txt] = emb
            else:
                print(
                    f"  [LookupEmbedder] Warning: Cache size ({len(full_embeddings)}) "
                    f"mismatched preprocessed size ({len(processed_full)}). Computing embeddings instead."
                )

        if not self.text_to_emb:
            print(f"  [LookupEmbedder] Cache miss/invalid. Computing full embeddings on-the-fly...")
            self._resolve_inner()
            embeddings_list = []
            
            # Embed the unique preprocessed texts
            total = len(processed_full)
            batch_size = 64
            for start in range(0, total, batch_size):
                batch = processed_full[start : start + batch_size]
                if hasattr(self.resolved_inner, "encode"):
                    raw = self.resolved_inner.encode(batch, batch_size=batch_size, show_progress_bar=False)
                elif hasattr(self.resolved_inner, "embed_documents"):
                    raw = self.resolved_inner.embed_documents(batch)
                elif hasattr(self.resolved_inner, "embed"):
                    raw = self.resolved_inner.embed(batch, batch_size=batch_size)
                elif callable(self.resolved_inner):
                    raw = self.resolved_inner(batch)
                else:
                    raise RuntimeError(f"Cannot call embedder interface on: {type(self.resolved_inner)}")
                
                embeddings_list.extend(raw)
                
            full_embeddings = np.asarray(embeddings_list, dtype=np.float32)
            save_cache(alias, dataset, n_docs, seed, full_embeddings)
            
            # Map BOTH raw and cleaned texts to the computed embeddings
            for raw_idx, proc_idx in raw_to_proc.items():
                if proc_idx >= 0:
                    raw_txt = full_texts[raw_idx]
                    clean_txt = processed_full[proc_idx]
                    emb = full_embeddings[proc_idx]
                    self.text_to_emb[raw_txt] = emb
                    self.text_to_emb[clean_txt] = emb

    def _resolve_inner(self) -> None:
        if self.resolved_inner is None:
            if self.inner_embedder is not None:
                self.resolved_inner = self.inner_embedder
            else:
                # miniLM ONNX fallback
                from semantic_clusterer.embedding.onnx_model import OnnxEmbedder
                self.resolved_inner = OnnxEmbedder(batch_size=64, normalize=True, verbose=False)

    def embed(self, texts: List[str], batch_size: int = 64, **kwargs: Any) -> np.ndarray:
        # Find missing texts
        missing_texts = [txt for txt in texts if txt not in self.text_to_emb]
        
        if missing_texts:
            print(f"  [LookupEmbedder] Cache miss for {len(missing_texts)} texts. Embedding in batch...")
            self._resolve_inner()
            embeddings_list = []
            
            total = len(missing_texts)
            for start in range(0, total, batch_size):
                batch = missing_texts[start : start + batch_size]
                if hasattr(self.resolved_inner, "encode"):
                    raw = self.resolved_inner.encode(batch, batch_size=batch_size, show_progress_bar=False)
                elif hasattr(self.resolved_inner, "embed_documents"):
                    raw = self.resolved_inner.embed_documents(batch)
                elif hasattr(self.resolved_inner, "embed"):
                    raw = self.resolved_inner.embed(batch, batch_size=batch_size)
                elif callable(self.resolved_inner):
                    raw = self.resolved_inner(batch)
                else:
                    raise RuntimeError(f"Cannot call embedder interface on: {type(self.resolved_inner)}")
                
                embeddings_list.extend(raw)
                
            for txt, emb in zip(missing_texts, embeddings_list):
                self.text_to_emb[txt] = np.asarray(emb, dtype=np.float32)
                
        return np.array([self.text_to_emb[txt] for txt in texts], dtype=np.float32)


def print_keywords_summary(model: Any, name: str) -> None:
    """Print the c-TF-IDF keywords for the fitted clusters to show cluster analysis capability."""
    print(f"\n--- {name} Cluster Analysis (c-TF-IDF Keywords) ---")
    try:
        topic_labels = model.get_topic_labels()
        topic_keywords = model.get_topic_keywords()
        # Sort by cluster id (excluding noise -1)
        for cid in sorted(int(k) for k in topic_labels.keys() if k >= 0):
            keywords = topic_keywords.get(cid, [])
            keywords_str = ", ".join(f"{w} ({s:.2f})" for w, s in keywords[:5])
            print(f"  Cluster {cid:2d} | Label: {topic_labels[cid]:<30} | Keywords: {keywords_str}")
    except Exception as exc:
        print(f"  Failed to retrieve topic info: {exc}")


def run_embedder_eval(
    embedder_alias: str,
    tiers: List[str],
    test_size: float = 0.2,
    seed: int = 42,
) -> None:
    raw_embedder, info = build_embedder(embedder_alias)
    print(f"\n==============================================================")
    print(f"Evaluating {info['name']} across tiers: {', '.join(tiers)}")
    print(f"==============================================================")

    clusterer_records: List[Dict[str, Any]] = []
    ksplit_records: List[Dict[str, Any]] = []

    for tier in tiers:
        # Load dataset
        dataset, n_docs = TIER_PLAN[tier]
        print(f"\n--- [{tier}] {dataset} n_docs={n_docs} ---")
        texts, labels, target_names = load_benchmark_dataset(dataset, n_docs=n_docs, seed=seed)
        
        # Split train/test
        train_texts, test_texts, train_labels, test_labels = train_test_split(
            texts,
            labels,
            test_size=test_size,
            random_state=seed,
            stratify=labels,
        )
        
        preprocessor = TextPreprocessor(lowercase=True, remove_punctuation=True)
        lookup_embedder = LookupOrComputeEmbedder(
            inner_embedder=raw_embedder,
            alias=info["alias"],
            dataset=dataset,
            full_texts=texts,
            seed=seed,
            preprocessor=preprocessor,
        )

        # 1. SemanticClusterer
        print(f"\n[SemanticClusterer] Running balanced granularity on '{tier}'...")
        sc = SemanticClusterer(
            embedding_model=lookup_embedder,
            cluster_granularity="balanced",
            random_state=seed,
        )
        with Timer() as t_fit:
            train_preds = sc.fit_predict(train_texts)
        fit_metrics = external_metrics(train_labels, train_preds)
        clusterer_records.append({
            "tier": tier,
            "Model": "SemanticClusterer",
            "Phase": "Fit/Train",
            "Threshold": "N/A",
            "ARI": fit_metrics["ari"],
            "NMI": fit_metrics["nmi"],
            "V-Measure": fit_metrics["v_measure"],
            "Coverage": fit_metrics["coverage"],
            "Noise Ratio": fit_metrics["noise_ratio"],
            "Pred K": fit_metrics["n_pred_clusters"],
            "Secs": t_fit.seconds,
        })
        
        print_keywords_summary(sc, "SemanticClusterer")

        with Timer() as t_pred_auto:
            test_preds_auto = sc.predict(test_texts, outlier_threshold="auto")
        auto_metrics = external_metrics(test_labels, test_preds_auto)
        clusterer_records.append({
            "tier": tier,
            "Model": "SemanticClusterer",
            "Phase": "Predict/Test",
            "Threshold": "auto",
            "ARI": auto_metrics["ari"],
            "NMI": auto_metrics["nmi"],
            "V-Measure": auto_metrics["v_measure"],
            "Coverage": auto_metrics["coverage"],
            "Noise Ratio": auto_metrics["noise_ratio"],
            "Pred K": auto_metrics["n_pred_clusters"],
            "Secs": t_pred_auto.seconds,
        })

        with Timer() as t_pred_none:
            test_preds_none = sc.predict(test_texts, outlier_threshold=None)
        none_metrics = external_metrics(test_labels, test_preds_none)
        clusterer_records.append({
            "tier": tier,
            "Model": "SemanticClusterer",
            "Phase": "Predict/Test",
            "Threshold": "None",
            "ARI": none_metrics["ari"],
            "NMI": none_metrics["nmi"],
            "V-Measure": none_metrics["v_measure"],
            "Coverage": none_metrics["coverage"],
            "Noise Ratio": none_metrics["noise_ratio"],
            "Pred K": none_metrics["n_pred_clusters"],
            "Secs": t_pred_none.seconds,
        })

        # 2. SemanticKSplit
        k = len(set(train_labels))
        print(f"\n[SemanticKSplit] Running balanced quality with k={k} on '{tier}'...")
        ks = SemanticKSplit(
            embedding_model=lookup_embedder,
            k=k,
            quality="balanced",
            random_state=seed,
        )
        with Timer() as t_ks_fit:
            train_preds_ks = ks.fit_predict(train_texts)
        ks_fit_metrics = external_metrics(train_labels, train_preds_ks)
        ksplit_records.append({
            "tier": tier,
            "Model": "SemanticKSplit",
            "Phase": "Fit/Train",
            "Threshold": "N/A",
            "ARI": ks_fit_metrics["ari"],
            "NMI": ks_fit_metrics["nmi"],
            "V-Measure": ks_fit_metrics["v_measure"],
            "Coverage": ks_fit_metrics["coverage"],
            "Noise Ratio": ks_fit_metrics["noise_ratio"],
            "Pred K": ks_fit_metrics["n_pred_clusters"],
            "Secs": t_ks_fit.seconds,
        })

        print_keywords_summary(ks, "SemanticKSplit")

        with Timer() as t_ks_pred_none:
            test_preds_ks_none = ks.predict(test_texts, outlier_threshold=None)
        ks_none_metrics = external_metrics(test_labels, test_preds_ks_none)
        ksplit_records.append({
            "tier": tier,
            "Model": "SemanticKSplit",
            "Phase": "Predict/Test",
            "Threshold": "None",
            "ARI": ks_none_metrics["ari"],
            "NMI": ks_none_metrics["nmi"],
            "V-Measure": ks_none_metrics["v_measure"],
            "Coverage": ks_none_metrics["coverage"],
            "Noise Ratio": ks_none_metrics["noise_ratio"],
            "Pred K": ks_none_metrics["n_pred_clusters"],
            "Secs": t_ks_pred_none.seconds,
        })

        with Timer() as t_ks_pred_auto:
            test_preds_ks_auto = ks.predict(test_texts, outlier_threshold="auto")
        ks_auto_metrics = external_metrics(test_labels, test_preds_ks_auto)
        ksplit_records.append({
            "tier": tier,
            "Model": "SemanticKSplit",
            "Phase": "Predict/Test",
            "Threshold": "auto",
            "ARI": ks_auto_metrics["ari"],
            "NMI": ks_auto_metrics["nmi"],
            "V-Measure": ks_auto_metrics["v_measure"],
            "Coverage": ks_auto_metrics["coverage"],
            "Noise Ratio": ks_auto_metrics["noise_ratio"],
            "Pred K": ks_auto_metrics["n_pred_clusters"],
            "Secs": t_ks_pred_auto.seconds,
        })

    # Print summary tables
    print(f"\n================ Production API Evaluation: SemanticClusterer ({info['name']}) ================")
    print_table(
        [
            {
                "Tier": r["tier"],
                "Phase": r["Phase"],
                "Thresh": r["Threshold"],
                "ARI": fmt(r["ARI"]),
                "NMI": fmt(r["NMI"]),
                "V-Meas": fmt(r["V-Measure"]),
                "Cov": fmt(r["Coverage"]),
                "Noise": fmt(r["Noise Ratio"]),
                "K": str(r["Pred K"]),
                "Secs": fmt(r["Secs"], nd=2),
            }
            for r in clusterer_records
        ],
        columns=["Tier", "Phase", "Thresh", "ARI", "NMI", "V-Meas", "Cov", "Noise", "K", "Secs"],
    )

    print(f"\n================ Production API Evaluation: SemanticKSplit ({info['name']}) ================")
    print_table(
        [
            {
                "Tier": r["tier"],
                "Phase": r["Phase"],
                "Thresh": r["Threshold"],
                "ARI": fmt(r["ARI"]),
                "NMI": fmt(r["NMI"]),
                "V-Meas": fmt(r["V-Measure"]),
                "Cov": fmt(r["Coverage"]),
                "Noise": fmt(r["Noise Ratio"]),
                "K": str(r["Pred K"]),
                "Secs": fmt(r["Secs"], nd=2),
            }
            for r in ksplit_records
        ],
        columns=["Tier", "Phase", "Thresh", "ARI", "NMI", "V-Meas", "Cov", "Noise", "K", "Secs"],
    )

    # Save outputs (produces exactly 6 files when running for 3 models)
    clusterer_payload = {
        "kind": "SemanticClusterer_Production",
        "embedder_alias": info.get("alias"),
        "embedder_name": info.get("name"),
        "seed": seed,
        "environment": environment(),
        "runs": clusterer_records,
    }
    save_path_sc = save_results(f"production_clusterer_{info.get('alias')}.json", clusterer_payload)
    print(f"\nSaved clusterer report -> {save_path_sc}")

    ksplit_payload = {
        "kind": "SemanticKSplit_Production",
        "embedder_alias": info.get("alias"),
        "embedder_name": info.get("name"),
        "seed": seed,
        "environment": environment(),
        "runs": ksplit_records,
    }
    save_path_ks = save_results(f"production_ksplit_{info.get('alias')}.json", ksplit_payload)
    print(f"Saved ksplit report    -> {save_path_ks}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate .fit and .predict performance of clustering models.")
    ap.add_argument(
        "--embedder",
        default="minilm",
        choices=["minilm", "mpnet", "openai3small", "all"],
        help="Embedder alias: minilm | mpnet | openai3small | all",
    )
    ap.add_argument(
        "--tiers",
        nargs="+",
        default=ALL_TIERS,
        choices=ALL_TIERS,
        help="Which tiers to run (tiny, small, medium)",
    )
    ap.add_argument("--test_size", type=float, default=0.2, help="Proportion of dataset for testing")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    args = ap.parse_args()

    # Disable warning clutter for clean presentation
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if args.embedder == "all":
            for emb in ["minilm", "mpnet", "openai3small"]:
                run_embedder_eval(emb, args.tiers, args.test_size, args.seed)
        else:
            run_embedder_eval(args.embedder, args.tiers, args.test_size, args.seed)


if __name__ == "__main__":
    main()
