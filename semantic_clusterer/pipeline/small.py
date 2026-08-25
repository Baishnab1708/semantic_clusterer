from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from semantic_clusterer.pipeline.postprocess import (
    compact_labels as _compact_labels,
    merge_near_duplicate_clusters,
    recover_noise_with_confidence,
)
from semantic_clusterer.pipeline.quality import compute_cluster_stats, score_clustering
from semantic_clusterer.pipeline.utils import _unique_int_candidates, _select_diverse_candidates
from semantic_clusterer.utils.similarity import normalize_vectors
from semantic_clusterer.reduction.umap_utils import (
    compute_optimal_umap_neighbors,
    compute_optimal_umap_components,
)
from semantic_clusterer.dim_bands import resolve_dim_band
from semantic_clusterer.pipeline.tuning import get_band_grid
from semantic_clusterer.optional_deps import _seeded_global_numpy

SMALL_REFINEMENT_SCORE_THRESHOLD = 0.38
SMALL_REFINEMENT_NOISE_THRESHOLD = 0.35
SMALL_BLOB_RATIO_FLOOR = 0.25
SMALL_BLOB_RATIO_MULTIPLIER = 1.50
SMALL_BLOB_RATIO_CEILING = 0.55
SMALL_BLOB_MAX_SEPARATION = 0.35
SMALL_BLOB_MIN_STABILITY = 0.45


def _adaptive_small_cluster_sizes(n_samples: int, embedding_dim: int = 768) -> List[int]:
    """Return HDBSCAN min_cluster_size candidates for the small pipeline.

    Delegates to get_band_grid for band-aware grids.
    Kept for backward compatibility with internal callers.
    """
    band = resolve_dim_band(embedding_dim)
    grid = get_band_grid(band, "small")
    max_cluster_size = max(3, n_samples - 1)
    candidates = [
        max(5, int(ratio * n_samples))
        for ratio in grid.hdbscan_min_cluster_size_ratios
    ]
    return _unique_int_candidates(candidates, max_cluster_size)


def _score_small_clustering(
    embeddings: np.ndarray,
    labels: np.ndarray,
    density_score: float | None,
    fragmentation_weight_boost: float = 0.0,
) -> Dict[str, float]:
    """Score the small pipeline with the shared intrinsic clustering objective.

    ``fragmentation_weight_boost`` scales up the default fragmentation penalty
    weight so ``balanced`` and ``coarse`` granularity presets actively
    discourage micro-fragmentation even when the user pins a small
    ``min_cluster_size``. ``0.0`` reproduces the historical behaviour
    (``fine``).
    """
    weights = None
    if fragmentation_weight_boost > 0.0:
        from semantic_clusterer.pipeline.quality import DEFAULT_CLUSTER_SCORE_WEIGHTS
        base = float(DEFAULT_CLUSTER_SCORE_WEIGHTS["fragmentation_penalty"])
        weights = {"fragmentation_penalty": base * (1.0 + fragmentation_weight_boost)}
    metrics = score_clustering(
        embeddings, labels, density_score=density_score, weights=weights
    )
    if metrics["n_clusters"] <= 0 and metrics["coverage"] <= 0.0:
        metrics["score"] = -1.0
    return metrics


def _compute_small_blob_threshold(n_clusters: float) -> float:
    """Allow balanced low-K solutions while still catching true giant blobs."""
    approx_clusters = max(1.0, float(n_clusters))
    balanced_ratio = 1.0 / approx_clusters
    return float(
        np.clip(
            max(SMALL_BLOB_RATIO_FLOOR, balanced_ratio * SMALL_BLOB_RATIO_MULTIPLIER),
            SMALL_BLOB_RATIO_FLOOR,
            SMALL_BLOB_RATIO_CEILING,
        )
    )


def _should_refine_small_solution(metrics: Dict[str, float]) -> Tuple[bool, str]:
    """Trigger extra search only for clearly weak or blob-like solutions."""
    reasons: List[str] = []

    if metrics["score"] < SMALL_REFINEMENT_SCORE_THRESHOLD:
        reasons.append("low_score")

    if metrics["noise_ratio"] > SMALL_REFINEMENT_NOISE_THRESHOLD:
        reasons.append("high_noise")

    blob_threshold = _compute_small_blob_threshold(metrics.get("n_clusters", 0.0))
    blob_like = (
        metrics["largest_ratio"] > blob_threshold
        and (
            metrics.get("n_clusters", 0.0) <= 1.0
            or metrics.get("separation", 0.0) < SMALL_BLOB_MAX_SEPARATION
            or metrics.get("stability", 0.5) < SMALL_BLOB_MIN_STABILITY
        )
    )
    if blob_like:
        reasons.append("blob")

    return bool(reasons), ",".join(reasons)


def _recover_small_noise_labels(
    embeddings: np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    """Recover confidently assignable noise points using adaptive thresholds."""
    labels = labels.copy()
    noise_indices = np.where(labels == -1)[0]

    if len(noise_indices) == 0:
        return labels

    stats = compute_cluster_stats(embeddings, labels)
    valid_labels = stats["labels"]
    if len(valid_labels) == 0:
        return labels

    noise_embeddings = embeddings[noise_indices]
    distances = 1.0 - np.clip(np.dot(noise_embeddings, stats["centroids"].T), -1.0, 1.0)
    best_indices = np.argmin(distances, axis=1)
    best_distances = distances[np.arange(len(noise_indices)), best_indices]

    core_max_distances = np.zeros(len(valid_labels))
    for i, cluster_id in enumerate(valid_labels):
        cluster_distances = 1.0 - np.clip(
            np.dot(embeddings[labels == cluster_id], stats["centroids"][i]), -1.0, 1.0
        )
        core_max_distances[i] = np.max(cluster_distances) if len(cluster_distances) > 0 else 0.0

    assignable_thresholds = np.maximum(core_max_distances * 1.5, 0.40)
    best_thresholds = assignable_thresholds[best_indices]

    if len(valid_labels) >= 2:
        top_two = np.partition(distances, kth=1, axis=1)[:, :2]
        second_best = np.max(top_two, axis=1)
        margin = second_best - best_distances
        confident = (best_distances <= best_thresholds) & (margin > 0.01)
    else:
        confident = best_distances <= best_thresholds

    for row_idx, noise_idx in enumerate(noise_indices):
        if confident[row_idx]:
            labels[noise_idx] = int(valid_labels[best_indices[row_idx]])

    return labels



def _split_oversized_small_clusters(
    embeddings: np.ndarray,
    labels: np.ndarray,
    umap_params: Dict[str, float],
    hdbscan_params: Dict[str, float],
    umap_cls: Any,
    hdbscan_cls: Any,
    max_split_passes: int = 1,
    max_clusters_to_split: int = 2,
    random_state: int = 42,
) -> np.ndarray:
    """Split weak oversized clusters only when split clearly improves score."""
    refined = labels.copy()
    valid_labels, counts = np.unique(refined[refined >= 0], return_counts=True)
    if len(valid_labels) == 0:
        return refined

    baseline_metrics = _score_small_clustering(embeddings, refined, None)
    baseline_score = baseline_metrics["score"]

    next_label = int(np.max(valid_labels)) + 1

    for split_pass in range(max_split_passes):
        clusters_split_this_pass = 0

        for cluster_id, cluster_size in sorted(
            zip(valid_labels, counts),
            key=lambda item: item[1],
            reverse=True,
        ):
            if clusters_split_this_pass >= max_clusters_to_split:
                break

            cluster_mask = refined == cluster_id
            cluster_points = embeddings[cluster_mask]
            centroid = normalize_vectors(np.mean(cluster_points, axis=0, keepdims=True))[0]
            cohesion = float(np.mean(np.clip(np.dot(cluster_points, centroid), -1.0, 1.0)))

            if cluster_size < max(12, int(hdbscan_params["min_cluster_size"]) * 2) or cohesion >= 0.88:
                continue

            sub_neighbors = int(min(cluster_size - 1, max(10, int(umap_params["n_neighbors"]) - 5)))
            if sub_neighbors < 5:
                continue

            sub_components = int(max(3, min(cluster_points.shape[1], int(umap_params["n_components"]) - 1)))
            sub_min_cluster_size = int(
                max(3, min(int(hdbscan_params["min_cluster_size"]), max(3, cluster_size // 6)))
            )
            sub_min_samples = min(2, max(1, sub_min_cluster_size - 1))

            reducer = umap_cls(
                n_neighbors=sub_neighbors,
                n_components=sub_components,
                min_dist=0.0,
                metric="cosine",
                n_jobs=-1,
            )
            reduced = reducer.fit_transform(cluster_points)
            clusterer = hdbscan_cls(
                min_cluster_size=sub_min_cluster_size,
                min_samples=sub_min_samples,
                metric="euclidean",
                cluster_selection_method="leaf",
                cluster_selection_epsilon=0.0,
                gen_min_span_tree=True,
            )
            with _seeded_global_numpy(random_state):
                sub_labels = clusterer.fit_predict(reduced)
            valid_sub_labels = np.unique(sub_labels[sub_labels >= 0])

            if len(valid_sub_labels) < 2:
                continue

            sub_counts = np.array(
                [np.sum(sub_labels == label) for label in valid_sub_labels], dtype=np.int32
            )
            if len(sub_counts) == 0 or np.max(sub_counts) >= int(cluster_size * 0.9):
                continue

            cluster_indices = np.where(cluster_mask)[0]
            largest_sub_label = int(valid_sub_labels[np.argmax(sub_counts)])

            candidate = refined.copy()
            remap: Dict[int, int] = {largest_sub_label: int(cluster_id)}
            for sub_label in valid_sub_labels:
                sub_label = int(sub_label)
                if sub_label == largest_sub_label:
                    continue
                remap[sub_label] = next_label
                next_label += 1

            for local_idx, global_idx in enumerate(cluster_indices):
                sub_label = int(sub_labels[local_idx])
                if sub_label >= 0:
                    candidate[global_idx] = remap[sub_label]
                else:
                    candidate[global_idx] = int(cluster_id)

            candidate_metrics = _score_small_clustering(embeddings, candidate, None)
            if candidate_metrics["score"] > baseline_score:
                refined = candidate
                baseline_score = candidate_metrics["score"]
                clusters_split_this_pass += 1

        if clusters_split_this_pass == 0:
            break

        valid_labels, counts = np.unique(refined[refined >= 0], return_counts=True)

    return refined



def cluster_small(
    embeddings: np.ndarray,
    *,
    config=None,
    random_state: int = 42,
    trace=None,
    log_fn=None,
) -> np.ndarray:
    """Clustering strategy for small datasets (150-5K).

    Args:
        embeddings: Float32 array of shape (N, D).
        config: Optional ClustererConfig (keyword-only).
        random_state: Integer RNG seed (keyword-only, default 42).
        trace: Optional _PipelineTrace accumulator (keyword-only).
        log_fn: Optional logging callable (keyword-only).

    Returns:
        int32 numpy array of cluster labels, shape (N,).
    """
    def _log(msg):
        if log_fn:
            log_fn(msg)

    N = len(embeddings)
    _log(f"Using speed-optimized small strategy (N={N})")

    from semantic_clusterer.optional_deps import try_import_umap
    try:
        from hdbscan import HDBSCAN
    except ImportError as exc:
        raise ImportError(
            "hdbscan is required for the small pipeline. "
            "Install it with: pip install hdbscan"
        ) from exc
    UMAP = try_import_umap()  # None if umap-learn not installed

    norm_embeddings = normalize_vectors(embeddings)
    embedding_dim = embeddings.shape[1]

    # Resolve dim band and get parameter grid
    band = resolve_dim_band(embedding_dim)
    grid = get_band_grid(band, "small")

    # Resolve granularity profile early so the score function can use the
    # ``fragmentation_penalty_weight`` boost during the parameter sweep.
    from semantic_clusterer.pipeline.granularity import resolve_granularity
    gran_preset = getattr(config, "cluster_granularity", "balanced") if config else "balanced"
    gran_profile = resolve_granularity(gran_preset, band=band)
    _frag_boost = float(gran_profile.fragmentation_penalty_weight)
    if band == "low":
        # Reduce fragmentation penalty boost for low band to allow more clusters
        _frag_boost = 0.0

    # Populate trace routing info
    if trace is not None:
        trace.chosen_params["pipeline_tier"] = "small"
        trace.chosen_params["embedding_dim"] = embedding_dim
        trace.chosen_params["dim_band"] = band

    # -----------------------------------------------------------------------
    # Phase 1: Pre-reduction (PCA)
    # -----------------------------------------------------------------------
    t_pca_start = time.perf_counter()
    search_embeddings = norm_embeddings
    reduction_info = "none"
    pca_target_used: Optional[int] = None

    if embedding_dim >= 256:
        # For the low band, the grid has [D] as a placeholder — no PCA needed
        # since D is already small. For other bands, pick the largest pca_target
        # that is still < embedding_dim.
        if band == "low":
            # No PCA for low band (D is already small)
            pca_target_used = embedding_dim
        else:
            valid_pca_targets = [t for t in grid.pca_targets if t < embedding_dim]
            if valid_pca_targets:
                pca_target = max(valid_pca_targets)
            else:
                pca_target = grid.pca_targets[-1]
            pca_target_used = pca_target

            pca_dim = min(
                pca_target,
                embedding_dim,
                max(20, int(N * 0.60)),
                max(2, N - 1),
            )
            if pca_dim < embedding_dim:
                _log(f"Applying PCA pre-reduction: {embedding_dim} -> {pca_dim}")
                from sklearn.decomposition import PCA
                pca = PCA(
                    n_components=pca_dim,
                    svd_solver="randomized",
                    random_state=random_state,
                )
                search_embeddings = normalize_vectors(pca.fit_transform(norm_embeddings))
                reduction_info = f"pca-{pca_dim}"
    else:
        pca_target_used = embedding_dim

    if trace is not None:
        trace.time("pca", time.perf_counter() - t_pca_start)

    # -----------------------------------------------------------------------
    # Phase 2: Compute adaptive parameter anchors
    # -----------------------------------------------------------------------
    opt_neighbors = compute_optimal_umap_neighbors(N)

    if embedding_dim > 512:
        # High dim: slightly smaller neighborhood to allow more local structure separation
        opt_neighbors = max(opt_neighbors, int(np.clip(np.log2(N) * 1.8, 10, min(30, N - 1))))
    else:
        # Low dim: standard neighborhood
        opt_neighbors = max(opt_neighbors, int(np.clip(np.log2(N) * 1.5, 10, min(25, N - 1))))

    opt_components = compute_optimal_umap_components(embedding_dim, N)
    min_cluster_candidates = _adaptive_small_cluster_sizes(N, embedding_dim)

    # User override: when config.min_cluster_size is set, pin the grid to
    # exactly that value so HDBSCAN behaves predictably across runs.
    if config is not None and getattr(config, "min_cluster_size", None) is not None:
        min_cluster_candidates = [int(config.min_cluster_size)]

    # Apply granularity floor to the candidate list
    from semantic_clusterer.pipeline.granularity import apply_mcs_floor, mcs_candidate_spread, compute_mcs_floor
    if config is not None:
        user_mcs = getattr(config, "min_cluster_size", None)
        if user_mcs is not None:
            min_cluster_candidates = [int(user_mcs)]
        else:
            # Floor-anchored geometric spread (3 values) so HDBSCAN can search
            # a real range and cluster count can grow with the corpus instead
            # of collapsing.
            min_cluster_candidates = mcs_candidate_spread(gran_profile, N)

            # Cap the maximum min_cluster_size candidate relative to the granularity floor.
            # This prevents selecting excessively coarse solutions (e.g. mcs=56) on low-dim models
            # while scaling dynamically with the corpus size N.
            if band == "low" and gran_preset == "balanced":
                floor = compute_mcs_floor(gran_profile, N)
                min_cluster_candidates = [c for c in min_cluster_candidates if c <= int(floor * 1.8)]

    # Read sweep parameters from the band grid
    search_methods = grid.hdbscan_methods
    search_min_dists = grid.umap_min_dists
    search_min_samples = grid.hdbscan_min_samples

    # User override: pin min_samples when set
    if config is not None and getattr(config, "min_samples", None) is not None:
        search_min_samples = [int(config.min_samples)]

    umap_cache: Dict[tuple, np.ndarray] = {}
    results: List[Dict] = []

    # Emit umap-unavailable warning once if UMAP is not available
    _umap_warning_msg = "umap-unavailable, used PCA-only fallback"
    if UMAP is None:
        import warnings as _warnings
        _warnings.warn(_umap_warning_msg, UserWarning, stacklevel=2)
        if trace is not None:
            trace.warn(_umap_warning_msg)

    def run_sweep(nn: int, nc: int):
        nn = min(nn, N - 1)
        nc = min(nc, search_embeddings.shape[1])
        if UMAP is None:
            # PCA-only fallback: run HDBSCAN directly on search_embeddings
            for mcs in min_cluster_candidates:
                for ms in search_min_samples:
                    for method in search_methods:
                        clusterer = HDBSCAN(
                            min_cluster_size=mcs,
                            min_samples=ms,
                            metric="euclidean",
                            cluster_selection_method=method,
                            cluster_selection_epsilon=0.0,
                            approx_min_span_tree=True,
                            gen_min_span_tree=False,
                        )
                        with _seeded_global_numpy(random_state):
                            labels = clusterer.fit_predict(search_embeddings)
                        density = getattr(clusterer, "relative_validity_", None)

                        metrics = _score_small_clustering(
                            norm_embeddings, labels, density,
                            fragmentation_weight_boost=_frag_boost,
                        )

                        results.append({
                            "score": metrics["score"],
                            "metrics": metrics,
                            "labels": labels.astype(np.int32),
                            "params": {
                                "n_neighbors": nn,
                                "n_components": nc,
                                "min_dist": 0.0,
                                "min_cluster_size": mcs,
                                "min_samples": ms,
                                "method": method,
                            },
                        })
            return
        for mcs in min_cluster_candidates:
            for ms in search_min_samples:
                for method in search_methods:
                    for md in search_min_dists:
                        reducer_key = (nn, nc, md)
                        if reducer_key not in umap_cache:
                            reducer = UMAP(
                                n_neighbors=nn,
                                n_components=nc,
                                min_dist=md,
                                metric="cosine",
                                random_state=random_state,
                                n_jobs=1,
                            )
                            umap_cache[reducer_key] = reducer.fit_transform(search_embeddings)

                        reduced = umap_cache[reducer_key]
                        clusterer = HDBSCAN(
                            min_cluster_size=mcs,
                            min_samples=ms,
                            metric="euclidean",
                            cluster_selection_method=method,
                            cluster_selection_epsilon=0.0,
                            approx_min_span_tree=True,
                            gen_min_span_tree=False,
                        )
                        with _seeded_global_numpy(random_state):
                            labels = clusterer.fit_predict(reduced)
                        density = getattr(clusterer, "relative_validity_", None)

                        metrics = _score_small_clustering(
                            norm_embeddings, labels, density,
                            fragmentation_weight_boost=_frag_boost,
                        )

                        results.append({
                            "score": metrics["score"],
                            "metrics": metrics,
                            "labels": labels.astype(np.int32),
                            "params": {
                                "n_neighbors": nn,
                                "n_components": nc,
                                "min_dist": md,
                                "min_cluster_size": mcs,
                                "min_samples": ms,
                                "method": method,
                            },
                        })

    # Phase 2: Primary Sweep
    t_sweep_start = time.perf_counter()
    _log(f"Phase 2: Primary sweep (UMAP nn={opt_neighbors}, comps={opt_components})")
    run_sweep(opt_neighbors, opt_components)

    if trace is not None:
        trace.time("umap_sweep", time.perf_counter() - t_sweep_start)

    # Phase 3: Optional Refinement
    results.sort(key=lambda x: x["score"], reverse=True)
    best_initial = results[0]

    should_refine, refine_reason = _should_refine_small_solution(best_initial["metrics"])
    if should_refine:
        _log(f"Phase 3: Quality refinement (reason: {refine_reason})...")
        t_refine_start = time.perf_counter()
        for multiplier in [0.75, 1.35]:
            refine_nn = int(np.clip(opt_neighbors * multiplier, 10, N - 1))
            if refine_nn != opt_neighbors:
                run_sweep(refine_nn, opt_components)
        if trace is not None:
            trace.time("refinement", time.perf_counter() - t_refine_start)

    # Final selection: evaluate top 3 candidates on post-processed score
    results.sort(key=lambda x: x["score"], reverse=True)
    top_candidates = _select_diverse_candidates(results, N)

    best_post_score = -2.0
    best_candidate = top_candidates[0]
    best_post_labels = best_candidate["labels"]
    best_post_metrics = {}

    from semantic_clusterer.pipeline.granularity import resolve_granularity
    from semantic_clusterer.pipeline.postprocess import merge_clusters_by_centroid_similarity

    gran_preset = getattr(config, "cluster_granularity", "balanced") if config else "balanced"
    gran_profile = resolve_granularity(gran_preset, band=band)

    t_postprocess_start = time.perf_counter()

    # For small datasets, low-dimensional centroids have higher variance.
    # We use the mid-band merge thresholds to prevent premature merging of distinct topics,
    # ensuring the solution is generalized across all granularity presets.
    merge_threshold = gran_profile.merge_centroid_threshold
    if band == "low":
        mid_profile = resolve_granularity(gran_preset, band="mid")
        merge_threshold = mid_profile.merge_centroid_threshold

    for cand in top_candidates:
        cand_labels = cand["labels"].copy()
        cand_p = cand["params"]

        cand_labels = _recover_small_noise_labels(norm_embeddings, cand_labels)
        if UMAP is not None:
            cand_labels = _split_oversized_small_clusters(
                norm_embeddings,
                cand_labels,
                {"n_neighbors": cand_p["n_neighbors"], "n_components": cand_p["n_components"]},
                {"min_cluster_size": cand_p["min_cluster_size"]},
                UMAP,
                HDBSCAN,
                max_split_passes=1,
                max_clusters_to_split=2,
                random_state=random_state,
            )
        cand_labels = merge_near_duplicate_clusters(norm_embeddings, cand_labels)
        cand_labels = _recover_small_noise_labels(norm_embeddings, cand_labels)
        cand_labels, n_merges = merge_clusters_by_centroid_similarity(
            norm_embeddings, cand_labels, merge_threshold
        )

        post_metrics = _score_small_clustering(
            norm_embeddings, cand_labels, None,
            fragmentation_weight_boost=_frag_boost,
        )
        post_score = post_metrics.get("score", -1.0)

        if post_score > best_post_score:
            best_post_score = post_score
            best_post_labels = cand_labels
            best_candidate = cand
            best_post_metrics = post_metrics
            cand["n_merges"] = n_merges

    best_labels = best_post_labels
    best_p = best_candidate["params"]
    best_metrics = best_post_metrics

    if trace is not None:
        trace.time("postprocess", time.perf_counter() - t_postprocess_start)

    _log(
        f"Best small configuration: reduction={reduction_info}, "
        f"UMAP(nn={best_p['n_neighbors']}, nc={best_p['n_components']}), "
        f"HDBSCAN(mcs={best_p['min_cluster_size']}, method='{best_p['method']}'), "
        f"score={best_metrics.get('score', -1.0):.4f}, "
        f"n_clusters={int(best_metrics.get('n_clusters', 0))}"
    )

    # Populate trace chosen_params
    if trace is not None:
        trace.chosen_params.update({
            "pipeline_tier": "small",
            "embedding_dim": embedding_dim,
            "dim_band": band,
            "pca_target": pca_target_used,
            "umap_n_neighbors": best_p["n_neighbors"],
            "umap_n_components": best_p["n_components"],
            "umap_min_dist": best_p.get("min_dist", 0.0),
            "hdbscan_min_cluster_size": best_p["min_cluster_size"],
            "hdbscan_min_samples": best_p.get("min_samples", 1),
            "hdbscan_method": best_p["method"],
            # Fields not used by small pipeline
            "coarse_k": None,
            "shard_size": None,
        })
        n_merges = best_candidate.get("n_merges", 0)
        if n_merges > 0:
            trace.chosen_params["granularity_merges_applied"] = n_merges
        trace.intrinsic_metrics = dict(best_metrics)

    return best_labels.astype(np.int32)
