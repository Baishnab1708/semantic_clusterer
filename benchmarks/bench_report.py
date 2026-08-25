"""Build a clean, benchmark-focused per-run record.

Only fields that are directly useful for:
  (a) evaluating clustering quality against ground truth, or
  (b) calibrating the runtime quality floor (baseline_scores.json)

are included. Internal pipeline traces, phase timings, dataset profiles,
and decorative fields are excluded.

Record structure
----------------
identity        : kind, dataset, embedder, tier, k, n_docs, n_true_classes, seed, seconds
routing         : tier_actual, embedding_dim, dim_band, n_pred_clusters, n_noise,
                  confidence_level, warnings
external_metrics: ari, nmi, v_measure, homogeneity, completeness, coverage, noise_ratio
intrinsic_metrics: score, cohesion, separation, stability, coverage, fragmentation,
                   largest_ratio, noise_ratio, n_clusters  (+ silhouette/davies_bouldin
                   for KSplit)
per_cluster     : cluster_id, size, purity, dominant_true_class/name  (one row per cluster)
true_class_recovery: true_name, n_docs, best_cluster_fraction, n_noise  (one row per class)
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from bench_metrics import external_metrics


# ---------------------------------------------------------------------------
# Intrinsic metrics — only the fields that matter for calibration / quality
# ---------------------------------------------------------------------------

_INTRINSIC_KEEP = {
    "score", "cohesion", "separation", "stability",
    "coverage", "fragmentation", "largest_ratio",
    "noise_ratio", "n_clusters",
    # KSplit extras
    "silhouette", "davies_bouldin",
}


def _clean_intrinsic(raw: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in raw.items() if k in _INTRINSIC_KEEP}


# ---------------------------------------------------------------------------
# Per-cluster breakdown
# ---------------------------------------------------------------------------

def _per_cluster_breakdown(
    true_labels: Sequence[int],
    pred_labels: Sequence[int],
    target_names: Sequence[str],
) -> List[Dict[str, Any]]:
    """One row per predicted cluster: size, purity, dominant true class."""
    true = np.asarray(true_labels)
    pred = np.asarray(pred_labels)
    rows: List[Dict[str, Any]] = []
    for cid in sorted(int(c) for c in np.unique(pred) if c >= 0):
        mask = pred == cid
        members_true = true[mask]
        counts = Counter(members_true.tolist())
        dominant_cls, dominant_n = counts.most_common(1)[0]
        size = int(mask.sum())
        rows.append({
            "cluster_id": cid,
            "size": size,
            "purity": round(dominant_n / size, 4) if size else 0.0,
            "dominant_true_class": int(dominant_cls),
            "dominant_true_name": (
                target_names[dominant_cls]
                if 0 <= dominant_cls < len(target_names)
                else str(dominant_cls)
            ),
        })
    return rows


# ---------------------------------------------------------------------------
# Per-true-class recovery
# ---------------------------------------------------------------------------

def _true_class_recovery(
    true_labels: Sequence[int],
    pred_labels: Sequence[int],
    target_names: Sequence[str],
) -> List[Dict[str, Any]]:
    """One row per true class: how well it was recovered as one cluster.

    best_cluster_fraction near 1.0 = class landed in one coherent cluster.
    Low value = class was scattered across many clusters.
    """
    true = np.asarray(true_labels)
    pred = np.asarray(pred_labels)
    rows: List[Dict[str, Any]] = []
    for cls in sorted(set(true.tolist())):
        mask = true == cls
        total = int(mask.sum())
        preds_here = pred[mask]
        clustered = preds_here[preds_here >= 0]
        best_n = int(Counter(clustered.tolist()).most_common(1)[0][1]) if clustered.size else 0
        rows.append({
            "true_name": (
                target_names[cls] if 0 <= cls < len(target_names) else str(cls)
            ),
            "n_docs": total,
            "best_cluster_fraction": round(best_n / total, 4) if total else 0.0,
            "n_noise": int((preds_here < 0).sum()),
        })
    return rows


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build_detailed_record(
    *,
    kind: str,
    dataset: str,
    embedder_info: Dict[str, Any],
    tier_requested: str,
    texts: Sequence[str],
    true_labels: Sequence[int],
    pred_labels: Sequence[int],
    target_names: Sequence[str],
    report: Any,
    seconds: float,
    k: Optional[int] = None,
) -> Dict[str, Any]:
    """Assemble a clean, benchmark-focused record for one run."""
    ext = external_metrics(true_labels, pred_labels)
    intr = _clean_intrinsic(report.to_dict()["intrinsic_metrics"])

    return {
        # --- identity ---
        "kind": kind,
        "dataset": dataset,
        "embedder": embedder_info.get("alias"),
        "tier_requested": tier_requested,
        "tier_actual": report.pipeline_tier,
        "k": k,
        "n_docs": len(texts),
        "n_true_classes": len(set(true_labels)),
        "seed": report.random_state,
        "seconds": round(seconds, 2),
        # --- routing ---
        "embedding_dim": report.embedding_dim,
        "dim_band": report.dim_band,
        "n_pred_clusters": report.n_clusters,
        "n_noise": report.n_noise,
        "confidence_level": report.confidence_level,
        "warnings": list(report.warnings),
        # --- quality vs ground truth ---
        "external_metrics": ext,
        # --- quality without ground truth (used for floor calibration) ---
        "intrinsic_metrics": intr,
        # --- per-cluster and per-class breakdowns ---
        "per_cluster": _per_cluster_breakdown(true_labels, pred_labels, target_names),
        "true_class_recovery": _true_class_recovery(true_labels, pred_labels, target_names),
    }


# ---------------------------------------------------------------------------
# Console headline
# ---------------------------------------------------------------------------

def headline(record: Dict[str, Any]) -> str:
    ext = record["external_metrics"]
    return (
        f"  tier={record['tier_actual']:>6}  "
        f"dim={record['embedding_dim']:>4}({record['dim_band']})  "
        f"pred_k={record['n_pred_clusters']:>3}  "
        f"ARI={ext['ari']:.4f}  NMI={ext['nmi']:.4f}  "
        f"V={ext['v_measure']:.4f}  "
        f"score={record['intrinsic_metrics'].get('score', 0.0):.4f}  "
        f"{record['seconds']}s"
    )
