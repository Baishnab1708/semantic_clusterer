"""Tiny-tier clustering pipeline (N <= 150).

State-of-the-art multi-source candidate generation with intrinsic-score
selection.  Implements a five-phase approach:

  1. Degenerate-case handling (N=0, N=1, N=2, all-identical embeddings).
  2. Dual linkage matrix construction (Ward + Average, both always built).
  3. Five-source candidate generation with label-aware deduplication:
       a) Multi-scale dendrogram-jump (multiple significant gaps)
       b) Adaptive K-grid (data-size-aware spacing)
       c) Silhouette-optimal K
       d) Miniature UMAP + HDBSCAN (density-based, N >= 15)
       e) Spectral clustering (affinity-based, N >= 8)
  4. Score every candidate with score_clustering — pick the best.
  5. Granularity-controlled post-merge pass.

Generalised across all embedding models (MPNET, MiniLM, OpenAI, etc.)
by always evaluating both Ward and Average linkages and selecting the
best partition by intrinsic score rather than by dim-band heuristic.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from semantic_clusterer.dim_bands import resolve_dim_band
from semantic_clusterer.pipeline.quality import score_clustering
from semantic_clusterer.report import _PipelineTrace
from semantic_clusterer.utils.similarity import normalize_vectors


_HOMOGENEITY_TOL = 1e-9
_DUPLICATE_COSINE_THRESHOLD = 0.95


# ---------------------------------------------------------------------------
# Adaptive K-grid: generates a data-size-aware grid of K candidates
# ---------------------------------------------------------------------------

def _adaptive_k_grid(N: int) -> List[int]:
    """Generate a K-grid that adapts to the data size.

    For very small N (<=30), evaluate every K from 2..N//2.
    For larger N, use a logarithmically-spaced grid that covers the
    full range without wasting compute on implausible K values.
    """
    max_k = max(2, N // 2)
    if N <= 30:
        return list(range(2, max_k + 1))

    # Logarithmic spacing: dense at small K, sparse at high K
    raw = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20,
           24, 28, 32, 36, 40, 45, 50, 55, 60, 65, 70, 75]
    grid = sorted({k for k in raw if 2 <= k <= max_k})

    # Ensure max_k is always in the grid
    if max_k not in grid:
        grid.append(max_k)
        grid.sort()

    return grid


def cluster_tiny(
    embeddings: np.ndarray,
    random_state: int = 42,
    trace: Optional[_PipelineTrace] = None,
    config=None,
) -> np.ndarray:
    """Tiny-tier clustering with state-of-the-art multi-source selection.

    Phases:
      1. Degenerate-case handling (N=0, N=1, N=2, all-identical).
      2. Dual linkage matrix construction (Ward + Average, always both).
      3. Five-source candidate generation with label-aware dedup:
           a) Multi-scale dendrogram jump (all significant gaps)
           b) Adaptive K-grid (data-size-aware)
           c) Silhouette-optimal K
           d) Miniature UMAP+HDBSCAN (N >= 15)
           e) Spectral clustering (N >= 8)
      4. Score every candidate, tie-break by score → K → source order.
      5. Granularity merge pass (balanced/coarse).
    """
    N = int(embeddings.shape[0])

    # --- Phase 1: degenerate cases ---
    if N == 0:
        if trace is not None:
            trace.chosen_params["pipeline_tier"] = "tiny"
        return np.empty((0,), dtype=np.int32)
    if N == 1:
        if trace is not None:
            trace.chosen_params["pipeline_tier"] = "tiny"
        return np.array([0], dtype=np.int32)

    embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
    D = int(embeddings.shape[1]) if embeddings.ndim == 2 else 0
    band = resolve_dim_band(D) if D >= 1 else "low"

    if trace is not None:
        trace.chosen_params["pipeline_tier"] = "tiny"
        trace.chosen_params["embedding_dim"] = D
        trace.chosen_params["dim_band"] = band

    # All-identical (within tolerance) embeddings
    if N >= 2 and _all_identical(embeddings, tol=_HOMOGENEITY_TOL):
        labels = np.zeros(N, dtype=np.int32)
        if trace is not None:
            trace.chosen_params["tiny_chosen_source"] = "homogeneous"
            trace.chosen_params["tiny_chosen_k"] = 1
            trace.intrinsic_metrics = score_clustering(embeddings, labels)
        return labels

    if N == 2:
        sim = float(np.dot(_normalize(embeddings[0]), _normalize(embeddings[1])))
        labels = (
            np.array([0, 0], dtype=np.int32)
            if sim >= _DUPLICATE_COSINE_THRESHOLD
            else np.array([0, 1], dtype=np.int32)
        )
        if trace is not None:
            trace.chosen_params["tiny_chosen_source"] = "n2_threshold"
            trace.chosen_params["tiny_chosen_k"] = int(labels.max() + 1)
            trace.chosen_params["tiny_n2_cosine"] = sim
            trace.intrinsic_metrics = score_clustering(embeddings, labels)
        return labels

    # --- Phase 2: dual linkage matrices (always build both) ---
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import pdist

    norm_embeddings = normalize_vectors(embeddings)

    Z_ward = linkage(norm_embeddings, method="ward")
    cos_dist = pdist(norm_embeddings, metric="cosine")
    Z_average = linkage(cos_dist, method="average")

    candidates: List[Tuple[str, int, Optional[np.ndarray]]] = []

    # --- Phase 3a: multi-scale dendrogram-jump source ---
    # Extract ALL significant jump points from both linkages, not just the top-1
    dj_candidates_ward = _multi_scale_dendrogram_jump(Z_ward[:, 2], max_k=N // 2)
    dj_candidates_avg = _multi_scale_dendrogram_jump(Z_average[:, 2], max_k=N // 2)

    # Evaluate each dendrogram-jump candidate from both linkages
    for dj_k in dj_candidates_ward:
        labels_w = _cut_to_k(Z_ward, dj_k, N)
        candidates.append(("dendrogram-jump", int(dj_k), labels_w))

    for dj_k in dj_candidates_avg:
        labels_a = _cut_to_k(Z_average, dj_k, N)
        candidates.append(("dendrogram-jump", int(dj_k), labels_a))

    # --- Phase 3b: adaptive K-grid source ---
    # Apply granularity-based K-grid filter when a config is provided.
    grid_ks = _adaptive_k_grid(N)

    if config is not None:
        gran = getattr(config, "cluster_granularity", "balanced")
        if gran == "coarse":
            grid_max_gran = min(max(2, N // 2), max(2, N // 15))
            grid_ks = [k for k in grid_ks if k <= max(2, grid_max_gran)]
            if not grid_ks:
                grid_ks = [2]
        elif gran == "fine":
            grid_max_gran = max(2, N // 3)
            extra = [k for k in range(15, grid_max_gran + 1, 5) if 2 <= k <= N // 2]
            grid_ks = sorted(set(grid_ks + extra))

    if not grid_ks:
        omitted_grid = True
    else:
        omitted_grid = False
        for k in grid_ks:
            best_labels = _get_best_cut(Z_ward, Z_average, k, N, embeddings)
            candidates.append(("grid", int(k), best_labels))

    # --- Phase 3c: silhouette source ---
    silhouette_k_ward = _silhouette_k(Z_ward, norm_embeddings, "euclidean", ks=grid_ks)
    silhouette_k_average = _silhouette_k(Z_average, norm_embeddings, "cosine", ks=grid_ks)
    silhouette_k = None
    sil_labels = None

    if silhouette_k_ward is not None and silhouette_k_average is not None:
        labels_ward = _cut_to_k(Z_ward, silhouette_k_ward, N)
        labels_average = _cut_to_k(Z_average, silhouette_k_average, N)
        try:
            score_ward = float(score_clustering(embeddings, labels_ward)["score"])
        except Exception:
            score_ward = -1.0
        try:
            score_average = float(score_clustering(embeddings, labels_average)["score"])
        except Exception:
            score_average = -1.0
        if score_average > score_ward:
            silhouette_k = silhouette_k_average
            sil_labels = labels_average
        else:
            silhouette_k = silhouette_k_ward
            sil_labels = labels_ward
    elif silhouette_k_ward is not None:
        silhouette_k = silhouette_k_ward
        sil_labels = _cut_to_k(Z_ward, silhouette_k_ward, N)
    elif silhouette_k_average is not None:
        silhouette_k = silhouette_k_average
        sil_labels = _cut_to_k(Z_average, silhouette_k_average, N)

    if silhouette_k is not None:
        candidates.append(("silhouette", int(silhouette_k), sil_labels))

    # --- Phase 3d: Miniature UMAP+HDBSCAN source (lowered to N >= 15) ---
    if N >= 15:
        from semantic_clusterer.optional_deps import try_import_umap
        UMAP = try_import_umap()
        if UMAP is not None:
            try:
                from hdbscan import HDBSCAN

                # Adaptive parameters based on N
                if N < 30:
                    nn = max(3, min(N - 1, 8))
                    nc = max(2, min(3, D))
                    mcs = max(2, N // 8)
                else:
                    nn = max(5, min(15, N - 1))
                    nc = max(3, min(6, D))
                    mcs = max(3, N // 10)

                reducer = UMAP(n_neighbors=nn, n_components=nc, metric="cosine", n_jobs=-1, random_state=random_state)
                reduced = reducer.fit_transform(norm_embeddings)

                # Try multiple min_cluster_size values for better coverage
                for mcs_candidate in sorted({mcs, max(2, mcs // 2), max(2, mcs * 2)}):
                    if mcs_candidate >= N // 2:
                        continue
                    clusterer = HDBSCAN(
                        min_cluster_size=mcs_candidate,
                        min_samples=max(1, mcs_candidate - 1),
                        metric="euclidean",
                        cluster_selection_method="eom",
                    )
                    with _seeded_global_numpy(random_state):
                        umap_labels = clusterer.fit_predict(reduced)

                    if np.any(umap_labels >= 0):
                        from semantic_clusterer.pipeline.postprocess import (
                            recover_noise_with_confidence,
                            compact_labels,
                        )
                        umap_labels = recover_noise_with_confidence(norm_embeddings, umap_labels)
                        umap_labels = compact_labels(umap_labels)
                        k_umap = int(umap_labels.max() + 1)
                        if k_umap >= 2:
                            candidates.append(("umap-hdbscan", k_umap, umap_labels))
            except Exception:
                pass

    # --- Phase 3e: Spectral clustering source (N >= 8) ---
    if N >= 8:
        try:
            from sklearn.cluster import SpectralClustering

            # Build cosine similarity affinity matrix
            affinity = np.clip(norm_embeddings @ norm_embeddings.T, 0.0, 1.0)
            np.fill_diagonal(affinity, 1.0)

            # Try spectral at the best dendrogram-jump K values + a few from grid
            spectral_ks = set()
            for dj_k in dj_candidates_ward[:2]:
                spectral_ks.add(dj_k)
            for dj_k in dj_candidates_avg[:2]:
                spectral_ks.add(dj_k)
            # Also try a few from the grid
            for k in grid_ks[:5]:
                spectral_ks.add(k)

            spectral_ks = sorted(k for k in spectral_ks if 2 <= k <= N // 2)

            for k in spectral_ks[:6]:  # cap at 6 to bound compute
                try:
                    sc = SpectralClustering(
                        n_clusters=k,
                        affinity="precomputed",
                        random_state=random_state,
                        assign_labels="kmeans",
                        n_init=3,
                    )
                    spec_labels = sc.fit_predict(affinity).astype(np.int32)
                    if len(np.unique(spec_labels)) >= 2:
                        candidates.append(("spectral", int(k), spec_labels))
                except Exception:
                    continue
        except ImportError:
            pass

    # --- Label-aware dedup: keep first occurrence for each (K, partition) ---
    deduped = _dedup_candidates_label_aware(candidates)

    # --- Phase 4: score and pick ---
    scored = []
    for source, k, explicit_labels in deduped:
        labels = explicit_labels if explicit_labels is not None else _get_best_cut(
            Z_ward, Z_average, k, N, embeddings
        )
        try:
            metrics = score_clustering(embeddings, labels)
            score = float(metrics["score"])
        except Exception:
            score = float("nan")
            metrics = None
        scored.append({
            "source": source,
            "k": int(k),
            "score": score,
            "labels": labels,
            "metrics": metrics if np.isfinite(score) else None,
        })

    valid = [c for c in scored if np.isfinite(c["score"])]

    if not valid:
        # Fallback: best dendrogram-jump cut
        if trace is not None:
            trace.warn("tiny-fallback-dendrogram-jump", set_low_confidence=False)
        fallback_k = dj_candidates_ward[0] if dj_candidates_ward else 2
        labels = _get_best_cut(Z_ward, Z_average, max(1, fallback_k), N, embeddings)
        if trace is not None:
            trace.chosen_params["tiny_chosen_source"] = "dendrogram-jump"
            trace.chosen_params["tiny_chosen_k"] = int(fallback_k)
            trace.chosen_params["tiny_candidates"] = [
                {"source": c["source"], "k": c["k"], "score": c["score"]}
                for c in scored
            ]
        return labels

    # Tie-break: max score; on tie, smaller K; on K tie, source priority
    # Density-based > spectral > dendrogram > silhouette > grid
    _SOURCE_PRIORITY = {
        "umap-hdbscan": 0,
        "spectral": 1,
        "dendrogram-jump": 2,
        "silhouette": 3,
        "grid": 4,
    }
    valid.sort(
        key=lambda c: (-c["score"], c["k"], _SOURCE_PRIORITY.get(c["source"], 99))
    )
    chosen = valid[0]

    if trace is not None:
        trace.chosen_params["tiny_chosen_source"] = chosen["source"]
        trace.chosen_params["tiny_chosen_k"] = int(chosen["k"])
        trace.chosen_params["tiny_candidates"] = [
            {"source": c["source"], "k": c["k"], "score": c["score"]}
            for c in scored
        ]
        omitted = []
        if not dj_candidates_ward and not dj_candidates_avg:
            omitted.append("dendrogram-jump")
        if omitted_grid:
            omitted.append("grid")
        if silhouette_k is None:
            omitted.append("silhouette")
        if omitted:
            trace.chosen_params["tiny_omitted_sources"] = omitted
        trace.intrinsic_metrics = chosen["metrics"]

    final_labels = chosen["labels"].astype(np.int32, copy=False)

    # Granularity-controlled centroid merge pass (balanced / coarse)
    if config is not None:
        gran = getattr(config, "cluster_granularity", "balanced")
        if gran != "fine":
            try:
                from semantic_clusterer.pipeline.granularity import resolve_granularity
                from semantic_clusterer.pipeline.postprocess import merge_clusters_by_centroid_similarity
                gran_profile = resolve_granularity(gran, band=band)
                final_labels, n_merges = merge_clusters_by_centroid_similarity(
                    norm_embeddings, final_labels, gran_profile.merge_centroid_threshold
                )
                if trace is not None and n_merges > 0:
                    trace.chosen_params["granularity_merges_applied"] = n_merges
                    trace.chosen_params["tiny_chosen_k"] = int(
                        len(set(final_labels[final_labels >= 0].tolist()))
                    )
            except Exception:
                pass  # never break tiny output for a merge-pass failure

    return final_labels


# ---- internal helpers ----

def _normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v


def _all_identical(X: np.ndarray, tol: float) -> bool:
    return bool(np.all(np.abs(X - X[0:1]) <= tol))


def _multi_scale_dendrogram_jump(
    distances: np.ndarray,
    max_k: int,
    max_candidates: int = 5,
) -> List[int]:
    """Find ALL significant jump points in the dendrogram, not just the top-1.

    Uses z-score normalisation to identify jumps that are statistically
    significant relative to the local variance of merge heights.  Returns
    up to ``max_candidates`` K values, ordered by jump significance.
    """
    if len(distances) < 2 or max_k < 2:
        return []

    jumps = np.diff(distances)
    n_minus_one = len(distances)

    if jumps.size == 0:
        return []

    # Z-score normalisation for multi-scale detection
    jump_mean = float(np.mean(jumps))
    jump_std = float(np.std(jumps))
    if jump_std < 1e-12:
        # All jumps identical — single candidate at largest gap
        best_j = int(np.argmax(jumps))
        k = n_minus_one - best_j
        k = int(np.clip(k, 2, max_k))
        return [k]

    z_scores = (jumps - jump_mean) / jump_std

    # Find all jumps above the significance threshold (1.0 sigma)
    # Use a lower threshold than 1.5 to catch more candidates — the scoring
    # phase will eliminate bad ones.
    threshold = 1.0
    significant_indices = np.where(z_scores >= threshold)[0]

    if significant_indices.size == 0:
        # Fall back to top-3 jumps by magnitude
        top_indices = np.argsort(jumps)[-min(3, len(jumps)):][::-1]
        significant_indices = top_indices

    # Convert jump indices to K values: K = n_minus_one - j
    candidates = []
    for j in significant_indices:
        k = n_minus_one - int(j)
        k = int(np.clip(k, 2, max_k))
        if k not in candidates:
            candidates.append(k)

    # Sort by jump significance (highest z-score first)
    candidates_with_score = []
    for k in candidates:
        j = n_minus_one - k
        if 0 <= j < len(z_scores):
            candidates_with_score.append((k, float(z_scores[j])))
        else:
            candidates_with_score.append((k, 0.0))

    candidates_with_score.sort(key=lambda x: -x[1])
    return [k for k, _ in candidates_with_score[:max_candidates]]


def _silhouette_k(Z, X: np.ndarray, metric: str, ks: List[int]) -> Optional[int]:
    if not ks:
        return None
    from scipy.cluster.hierarchy import fcluster
    from sklearn.metrics import silhouette_score
    best_k = None
    best_score = -np.inf
    for k in ks:
        labels = fcluster(Z, t=k, criterion="maxclust")
        if len(np.unique(labels)) < 2:
            continue
        try:
            s = silhouette_score(X, labels, metric=metric)
        except Exception:
            continue
        if s > best_score:
            best_score = float(s)
            best_k = int(k)
    return best_k


def _cut_to_k(Z, k: int, N: int) -> np.ndarray:
    from scipy.cluster.hierarchy import fcluster
    if k <= 1:
        return np.zeros(N, dtype=np.int32)
    if k >= N:
        return np.arange(N, dtype=np.int32)
    labels = fcluster(Z, t=int(k), criterion="maxclust")
    return labels.astype(np.int32) - 1


def _get_best_cut(
    Z_ward,
    Z_average,
    k: int,
    N: int,
    embeddings: np.ndarray,
) -> np.ndarray:
    """Evaluate both Ward and Average cuts for a given K, return the best."""
    labels_ward = _cut_to_k(Z_ward, k, N)
    labels_average = _cut_to_k(Z_average, k, N)

    try:
        score_ward = float(score_clustering(embeddings, labels_ward)["score"])
    except Exception:
        score_ward = -1.0

    try:
        score_average = float(score_clustering(embeddings, labels_average)["score"])
    except Exception:
        score_average = -1.0

    if score_average > score_ward:
        return labels_average
    return labels_ward


def _dedup_candidates_label_aware(
    candidates: List[Tuple[str, int, Optional[np.ndarray]]],
) -> List[Tuple[str, int, Optional[np.ndarray]]]:
    """Dedup by (K, partition_hash), keeping first occurrence in source order.

    Unlike the old dedup-by-K-only, this preserves different partitions that
    happen to have the same K value (e.g. Ward K=5 vs Average K=5).
    """
    seen: dict = {}
    for source, k, labels in candidates:
        if labels is not None:
            # Hash the actual label assignment to detect truly identical partitions
            label_key = (k, hash(labels.tobytes()))
        else:
            label_key = (k, None)

        if label_key not in seen:
            seen[label_key] = (source, k, labels)

    return list(seen.values())


# Import the seeded numpy context manager
try:
    from semantic_clusterer.optional_deps import _seeded_global_numpy
except ImportError:
    from contextlib import contextmanager

    @contextmanager
    def _seeded_global_numpy(seed):
        """Fallback no-op context manager."""
        yield
