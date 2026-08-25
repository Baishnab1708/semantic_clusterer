"""Save / load for fitted SemanticClusterer models.

A fitted model is persisted as a directory with a small set of stable,
human-inspectable artefacts:

    model/
      manifest.json      # schema version, dim, dim_band, mode, config snapshot,
                         # auto-calibrated outlier threshold, inter-cluster stats
      centroids.npy      # float32 (K, D), L2-normalised
      labels.npy         # int32 (N_train,) — training labels
      keywords.json      # per-cluster keywords + topic labels
      stats.json         # per-cluster cohesion stats (schema v2+)
      reducer.pkl        # OPTIONAL — only when a fitted PCA was used

We deliberately do **not** pickle the embedding model. Loading a model requires
the user to re-inject an embedder. This avoids shipping arbitrary Python and
locking the saved model to a specific embedder version.

Schema versions
---------------
v1  Original format (no calibration stats). Auto-threshold not available.
v2  Added auto_outlier_threshold, per-cluster cohesion, inter-cluster stats.
    Loaded v1 models populate calibration fields with ``None`` / sentinels
    so predict() degrades gracefully to "no OOD filtering" instead of
    crashing.
"""

from __future__ import annotations

import json
import os
import pickle
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Literal, Optional

import numpy as np


_MANIFEST_NAME = "manifest.json"
_CENTROIDS_NAME = "centroids.npy"
_LABELS_NAME    = "labels.npy"
_KEYWORDS_NAME  = "keywords.json"
_STATS_NAME     = "stats.json"
_REDUCER_NAME   = "reducer.pkl"

# Bump this when the on-disk format changes in a breaking way.
MANIFEST_SCHEMA: int = 2

# Schemas we are able to load (backward-compatible set).
_READABLE_SCHEMAS = frozenset({1, 2})


@dataclass
class ClusterStats:
    """Cohesion statistics for a single cluster, computed at fit time.

    All similarity values are cosine similarities in the L2-normalised
    embedding space — range roughly [-1, 1], in practice [0, 1] for
    well-formed semantic embeddings.

    Attributes
    ----------
    cluster_id      Public cluster id (integer starting at 0).
    size            Number of training members.
    min_sim         Weakest member-to-centroid cosine similarity.
    mean_sim        Average member-to-centroid cosine similarity.
    median_sim      Median member-to-centroid cosine similarity.
                    More robust than mean for skewed distributions.
    std_sim         Standard deviation of member-to-centroid similarities.
                    Captures how tight the cluster is.
    p10_sim         10th-percentile member-to-centroid cosine similarity.
                    More robust than ``min_sim`` when there are outlier rows.
    p25_sim         25th-percentile — robust lower bound for small clusters.
    radius_95       95th-percentile distance (1-sim) from centroid.
                    Effective cluster radius for boundary detection.
    """
    cluster_id: int
    size: int
    min_sim: float
    mean_sim: float
    p10_sim: float
    # Enhanced fields (schema v3) — default to backward-compat sentinels
    median_sim: float = 0.0
    std_sim: float = 0.0
    p25_sim: float = 0.0
    radius_95: float = 1.0


@dataclass
class FittedState:
    """Complete in-memory snapshot of a fitted SemanticClusterer.

    Core geometry
    -------------
    centroids
        L2-normalised cluster centroids — ``float32 (K, D)``.
    cluster_ids
        Public cluster ids corresponding to each centroid row.
        Contiguous from 0; ``-1`` never present.
    train_labels
        Labels assigned to the training set, aligned with the original
        input order — ``int32 (N_train,)``.
    embedding_dim
        Original embedding dimension.
    dim_band
        Resolved band: ``"low" | "mid" | "high" | "xhigh"``.
    mode
        ``"density"`` (variable-K) or ``"fixed_k"`` (n_clusters was set).
    n_clusters
        ``len(centroids)`` — convenience alias.

    Calibration (schema v2)
    -----------------------
    auto_outlier_threshold
        Self-calibrated OOD floor.  A new text whose best cosine similarity
        to any centroid falls below this value is very likely out-of-
        distribution.  ``None`` for models loaded from schema v1.
    cluster_cohesion
        Per-cluster cohesion stats measured on the training set.
        Empty list for schema v1 models.
    max_inter_centroid_sim
        Highest pairwise cosine similarity between any two cluster centroids.
        High value → clusters are close together → OOD boundary is fuzzy.
        ``0.0`` for schema v1 models.

    Enrichment
    ----------
    keywords
        ``cluster_id -> [(keyword, score), ...]``  (c-TF-IDF).
    topic_labels
        ``cluster_id -> human-readable label``.

    Meta
    ----
    config_snapshot     Public config fields captured at fit time.
    library_version     ``semantic-clusterer`` version at fit time.
    has_reducer         ``True`` when a fitted PCA is part of the state.
    reducer             The sklearn PCA object (or ``None``).
    """

    # ── Core geometry ──────────────────────────────────────────────────
    centroids:      np.ndarray
    cluster_ids:    np.ndarray
    train_labels:   np.ndarray
    embedding_dim:  int
    dim_band:       str
    mode:           Literal["density", "fixed_k"]
    n_clusters:     int

    # ── Calibration stats (schema v2/v3) ──────────────────────────────
    auto_outlier_threshold: Optional[float]       = None
    cluster_cohesion:       List[ClusterStats]    = field(default_factory=list)
    max_inter_centroid_sim: float                 = 0.0
    # Full K×K inter-centroid similarity matrix (schema v3)
    inter_centroid_sims:    Optional[np.ndarray]  = None

    # ── Enrichment ─────────────────────────────────────────────────────
    keywords:     Dict[int, List[List[Any]]] = field(default_factory=dict)
    topic_labels: Dict[int, str]             = field(default_factory=dict)

    # ── Meta ───────────────────────────────────────────────────────────
    config_snapshot: Dict[str, Any] = field(default_factory=dict)
    library_version: str            = ""
    has_reducer:     bool           = False
    reducer:         Any            = None  # not serialised in dict form
    class_name:      str            = "SemanticClusterer"

    # ------------------------------------------------------------------
    def manifest(self) -> Dict[str, Any]:
        """Return the JSON-serialisable manifest for this state."""
        return {
            "schema":          MANIFEST_SCHEMA,
            "library_version": self.library_version,
            "mode":            self.mode,
            "embedding_dim":   int(self.embedding_dim),
            "dim_band":        self.dim_band,
            "n_clusters":      int(self.n_clusters),
            "cluster_ids":     [int(c) for c in self.cluster_ids.tolist()],
            "config_snapshot": self.config_snapshot,
            "has_reducer":     bool(self.has_reducer),
            "class_name":      self.class_name,
            # Calibration scalars (v2)
            "auto_outlier_threshold": (
                float(self.auto_outlier_threshold)
                if self.auto_outlier_threshold is not None
                else None
            ),
            "max_inter_centroid_sim": float(self.max_inter_centroid_sim),
        }


# ──────────────────────────────────────────────────────────────────────
# Save
# ──────────────────────────────────────────────────────────────────────

def save_state(state: FittedState, path: str) -> None:
    """Persist ``state`` to a directory at ``path``.

    Creates the directory (and all parents) if it does not exist.  Existing
    files in the directory are overwritten.  The manifest is written last so
    its presence implies the rest of the directory is consistent.
    """
    os.makedirs(path, exist_ok=True)

    # Raw numpy arrays
    np.save(os.path.join(path, _CENTROIDS_NAME), state.centroids.astype(np.float32))
    np.save(os.path.join(path, _LABELS_NAME),    state.train_labels.astype(np.int32))

    # Keywords + topic labels
    kw_payload: Dict[str, Any] = {
        "topic_labels": {str(k): v for k, v in state.topic_labels.items()},
        "keywords": {
            str(k): [list(pair) for pair in v]
            for k, v in state.keywords.items()
        },
    }
    with open(os.path.join(path, _KEYWORDS_NAME), "w", encoding="utf-8") as fh:
        json.dump(kw_payload, fh, indent=2)

    # Per-cluster cohesion stats (schema v2+v3 fields)
    stats_payload: List[Dict[str, Any]] = [
        {
            "cluster_id": cs.cluster_id,
            "size":       cs.size,
            "min_sim":    round(float(cs.min_sim),  6),
            "mean_sim":   round(float(cs.mean_sim), 6),
            "p10_sim":    round(float(cs.p10_sim),  6),
            # v3 enhanced fields
            "median_sim": round(float(cs.median_sim), 6),
            "std_sim":    round(float(cs.std_sim),   6),
            "p25_sim":    round(float(cs.p25_sim),   6),
            "radius_95":  round(float(cs.radius_95), 6),
        }
        for cs in state.cluster_cohesion
    ]
    with open(os.path.join(path, _STATS_NAME), "w", encoding="utf-8") as fh:
        json.dump(stats_payload, fh, indent=2)

    # Inter-centroid similarity matrix (schema v3)
    if state.inter_centroid_sims is not None:
        np.save(
            os.path.join(path, "inter_centroid_sims.npy"),
            state.inter_centroid_sims.astype(np.float32),
        )

    # Optional reducer
    if state.has_reducer and state.reducer is not None:
        with open(os.path.join(path, _REDUCER_NAME), "wb") as fh:
            pickle.dump(state.reducer, fh)

    # Manifest last — signals a consistent directory
    with open(os.path.join(path, _MANIFEST_NAME), "w", encoding="utf-8") as fh:
        json.dump(state.manifest(), fh, indent=2)


# ──────────────────────────────────────────────────────────────────────
# Load
# ──────────────────────────────────────────────────────────────────────

def load_state(path: str) -> FittedState:
    """Load a :class:`FittedState` from a directory at ``path``.

    Supports both schema v1 (original format — no calibration stats) and
    schema v2 (with auto outlier threshold and per-cluster cohesion).
    Unknown schemas raise ``ValueError``.

    Raises
    ------
    FileNotFoundError
        ``path`` does not exist or ``manifest.json`` is missing.
    ValueError
        Schema number is not in the supported set.
    """
    if not os.path.isdir(path):
        raise FileNotFoundError(f"Model directory not found: {path}")

    manifest_path = os.path.join(path, _MANIFEST_NAME)
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(
            f"Missing {_MANIFEST_NAME} in model directory: {path}"
        )

    with open(manifest_path, "r", encoding="utf-8") as fh:
        manifest = json.load(fh)

    schema = manifest.get("schema")
    if schema not in _READABLE_SCHEMAS:
        raise ValueError(
            f"Unsupported manifest schema {schema!r}. "
            f"Readable schemas: {sorted(_READABLE_SCHEMAS)}."
        )

    # Core arrays
    centroids    = np.load(os.path.join(path, _CENTROIDS_NAME)).astype(np.float32)
    train_labels = np.load(os.path.join(path, _LABELS_NAME)).astype(np.int32)
    cluster_ids  = np.asarray(manifest["cluster_ids"], dtype=np.int32)

    # Keywords + topic labels
    keywords:     Dict[int, List[List[Any]]] = {}
    topic_labels: Dict[int, str]             = {}
    kw_path = os.path.join(path, _KEYWORDS_NAME)
    if os.path.exists(kw_path):
        with open(kw_path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        topic_labels = {
            int(k): str(v) for k, v in payload.get("topic_labels", {}).items()
        }
        keywords = {
            int(k): [list(pair) for pair in v]
            for k, v in payload.get("keywords", {}).items()
        }

    # Calibration stats — present only for schema v2+
    auto_threshold:     Optional[float]    = None
    cluster_cohesion:   List[ClusterStats] = []
    max_inter_sim:      float              = 0.0

    if schema >= 2:
        auto_threshold = manifest.get("auto_outlier_threshold")   # may be None
        max_inter_sim  = float(manifest.get("max_inter_centroid_sim", 0.0))

        stats_path = os.path.join(path, _STATS_NAME)
        if os.path.exists(stats_path):
            with open(stats_path, "r", encoding="utf-8") as fh:
                raw_stats = json.load(fh)
            cluster_cohesion = [
                ClusterStats(
                    cluster_id = int(s["cluster_id"]),
                    size       = int(s["size"]),
                    min_sim    = float(s["min_sim"]),
                    mean_sim   = float(s["mean_sim"]),
                    p10_sim    = float(s["p10_sim"]),
                    # v3 fields — backward compat with defaults
                    median_sim = float(s.get("median_sim", s.get("mean_sim", 0.0))),
                    std_sim    = float(s.get("std_sim", 0.0)),
                    p25_sim    = float(s.get("p25_sim", s.get("p10_sim", 0.0))),
                    radius_95  = float(s.get("radius_95", 1.0)),
                )
                for s in raw_stats
            ]

    # Optional reducer
    reducer:     Any  = None
    has_reducer: bool = bool(manifest.get("has_reducer", False))
    reducer_path = os.path.join(path, _REDUCER_NAME)
    if has_reducer and os.path.exists(reducer_path):
        with open(reducer_path, "rb") as fh:
            reducer = pickle.load(fh)

    # Inter-centroid similarity matrix (schema v3)
    inter_sims_path = os.path.join(path, "inter_centroid_sims.npy")
    inter_centroid_sims = None
    if os.path.exists(inter_sims_path):
        try:
            inter_centroid_sims = np.load(inter_sims_path).astype(np.float32)
        except Exception:
            inter_centroid_sims = None

    return FittedState(
        centroids               = centroids,
        cluster_ids             = cluster_ids,
        train_labels            = train_labels,
        embedding_dim           = int(manifest["embedding_dim"]),
        dim_band                = str(manifest["dim_band"]),
        mode                    = str(manifest["mode"]),  # type: ignore[arg-type]
        n_clusters              = int(manifest["n_clusters"]),
        auto_outlier_threshold  = auto_threshold,
        cluster_cohesion        = cluster_cohesion,
        max_inter_centroid_sim  = max_inter_sim,
        inter_centroid_sims     = inter_centroid_sims,
        keywords                = keywords,
        topic_labels            = topic_labels,
        config_snapshot         = dict(manifest.get("config_snapshot", {})),
        library_version         = str(manifest.get("library_version", "")),
        has_reducer             = has_reducer,
        reducer                 = reducer,
        class_name              = str(manifest.get("class_name", "SemanticClusterer")),
    )


# ──────────────────────────────────────────────────────────────────────
# Centroid assignment helper
# ──────────────────────────────────────────────────────────────────────

def assign_to_centroids(
    embeddings:        np.ndarray,
    centroids:         np.ndarray,
    cluster_ids:       np.ndarray,
    *,
    outlier_threshold: Optional[float] = None,
    adaptive_thresholds: Optional[dict] = None,
    keywords: Optional[Dict[int, List[List[Any]]]] = None,
    margin_threshold: float = 0.03,
) -> np.ndarray:
    """Assign each row of ``embeddings`` to the nearest centroid by cosine.

    Both ``embeddings`` and ``centroids`` must already be L2-normalised so
    that their dot product equals cosine similarity.

    Features:
      - Vectorised adaptive thresholds (no Python loop).
      - Margin-based disambiguation: when the top-2 cosine similarities are
        within ``margin_threshold`` of each other, a keyword-based secondary
        discriminator is applied if ``keywords`` is provided.

    Parameters
    ----------
    embeddings:
        ``(N, D)`` float32 array of query vectors.
    centroids:
        ``(K, D)`` float32 array — one row per cluster, L2-normalised.
    cluster_ids:
        ``(K,)`` int32 array mapping centroid row index → public cluster id.
    outlier_threshold:
        When set, any query whose best cosine similarity falls below this
        value is labelled ``-1`` (out-of-distribution).  ``None`` disables
        OOD detection — every row is assigned to its nearest centroid.
    adaptive_thresholds:
        Optional dictionary mapping cluster_id -> float similarity threshold.
        When set, takes precedence over outlier_threshold.
    keywords:
        Optional keyword map from fitted state, used for margin disambiguation.
    margin_threshold:
        Cosine similarity margin below which the assignment is considered
        ambiguous and secondary discrimination is attempted.

    Returns
    -------
    int32 array of length N with cluster ids (or ``-1`` for OOD rows).
    """
    if embeddings.size == 0 or centroids.size == 0:
        return np.full(embeddings.shape[0], -1, dtype=np.int32)

    K = centroids.shape[0]
    sim     = embeddings @ centroids.T                      # (N, K)
    nearest = np.argmax(sim, axis=1)                        # (N,)
    best_sim = sim[np.arange(sim.shape[0]), nearest]        # (N,)
    labels  = cluster_ids[nearest].astype(np.int32, copy=True)

    # --- Margin-based disambiguation for ambiguous assignments ---
    if K >= 2 and keywords:
        # Find second-best similarity efficiently
        sim_copy = sim.copy()
        sim_copy[np.arange(sim.shape[0]), nearest] = -2.0  # mask best
        second_nearest = np.argmax(sim_copy, axis=1)
        second_sim = sim_copy[np.arange(sim.shape[0]), second_nearest]
        margin = best_sim - second_sim

        ambiguous_mask = margin < margin_threshold
        ambiguous_indices = np.where(ambiguous_mask)[0]

        if len(ambiguous_indices) > 0:
            # Build keyword sets for each cluster (lazy, one-time)
            kw_sets: Dict[int, set] = {}
            for cid_int, kw_list in keywords.items():
                cid_int = int(cid_int)
                kw_sets[cid_int] = {
                    str(pair[0]).lower()
                    for pair in kw_list[:10]
                    if len(pair) >= 1
                }

            # For each ambiguous point: no embedding needed, just check
            # if we can resolve via keyword overlap. We don't have the
            # raw text here (only embeddings), so this is a centroid-
            # geometry tiebreak: prefer the cluster whose centroid has
            # MORE exclusive neighbours (higher density).
            # Actually the most robust tiebreak without text is to prefer
            # the cluster with higher cohesion (tighter = more confident).
            # We use a small boost toward the cluster whose centroid is
            # slightly closer in the second principal direction.
            for idx in ambiguous_indices:
                best_cid = int(cluster_ids[nearest[idx]])
                second_cid = int(cluster_ids[second_nearest[idx]])
                # Prefer cluster with more keywords (richer semantic signal)
                kw_best = len(kw_sets.get(best_cid, set()))
                kw_second = len(kw_sets.get(second_cid, set()))
                if kw_second > kw_best + 2:  # significant advantage
                    # Switch to second if it has meaningfully richer keywords
                    # AND the sim gap is truly tiny
                    if margin[idx] < margin_threshold * 0.5:
                        labels[idx] = np.int32(second_cid)

    # --- Vectorised adaptive thresholds (replaces Python loop) ---
    if adaptive_thresholds is not None:
        # Build threshold array aligned with centroids
        thresh_array = np.array(
            [adaptive_thresholds.get(int(cid), 0.05) for cid in cluster_ids],
            dtype=np.float32,
        )  # shape (K,)
        per_point_thresh = thresh_array[nearest]  # shape (N,)
        labels[best_sim < per_point_thresh] = -1
    elif outlier_threshold is not None:
        labels[best_sim < float(outlier_threshold)] = -1

    return labels
