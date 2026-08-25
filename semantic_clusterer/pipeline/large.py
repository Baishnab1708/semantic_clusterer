from __future__ import annotations

import time
import numpy as np
from typing import Any, Dict, List, Optional

from semantic_clusterer.clustering.centroid_fallback import CentroidFallback
from semantic_clusterer.dim_bands import resolve_dim_band
from semantic_clusterer.optional_deps import _seeded_global_numpy
from semantic_clusterer.pipeline.postprocess import (
    compact_labels,
    merge_near_duplicate_clusters,
    recover_noise_with_confidence,
    split_oversized_clusters,
    merge_clusters_by_centroid_similarity,
)
from semantic_clusterer.pipeline.profile import compute_dataset_profile
from semantic_clusterer.pipeline.quality import compute_cluster_stats, score_clustering
from semantic_clusterer.pipeline.tuning import (
    compute_large_coarse_partition_count,
    compute_large_min_samples_candidates,
    compute_large_reduction_dimension,
    compute_large_target_shard_size,
    compute_reduction_candidates,
    get_band_grid,
)
from semantic_clusterer.pipeline.granularity import (
    resolve_granularity,
    mcs_candidate_spread,
)
from semantic_clusterer.pipeline.utils import _select_diverse_candidates
from semantic_clusterer.reduction.base import get_reducer
from semantic_clusterer.utils.similarity import normalize_vectors


def cluster_large(
    embeddings: np.ndarray,
    config=None,
    *,
    random_state: int = 42,
    trace=None,
    log_fn=None,
    verbose: bool = False,
) -> np.ndarray:
    """Clustering strategy for large datasets (50K-200K).

    Uses a two-stage pipeline: coarse partitioning via MiniBatchKMeans
    followed by per-shard density clustering via HDBSCAN, global
    stitching, and a full post-processing pipeline (split oversized,
    granularity merge, noise recovery).

    Args:
        embeddings: Float32 array of shape (N, D).
        config: Optional ClustererConfig (positional, kept for backward compat).
        random_state: Integer RNG seed (keyword-only, default 42).
        trace: Optional _PipelineTrace accumulator (keyword-only).
        log_fn: Optional logging callable (keyword-only).
        verbose: Enable verbose printing (keyword-only).

    Returns:
        int32 numpy array of cluster labels, shape (N,).
    """

    def _log(msg: str) -> None:
        if log_fn:
            log_fn(msg)
        if verbose:
            print(msg)

    N, D = embeddings.shape
    _log(f"Using adaptive large strategy (N={N}, D={D})")

    try:
        from hdbscan import HDBSCAN
        from sklearn.cluster import MiniBatchKMeans
    except ImportError as exc:
        raise ImportError(
            "hdbscan and scikit-learn are required for the large pipeline. "
            "Install them with: pip install hdbscan scikit-learn"
        ) from exc

    from semantic_clusterer.optional_deps import try_import_umap
    UMAP = try_import_umap()  # None if umap-learn not installed

    # Resolve dim band and get parameter grid
    band = resolve_dim_band(D)
    grid = get_band_grid(band, "large")

    # Resolve granularity profile
    gran_preset = getattr(config, "cluster_granularity", "balanced") if config is not None else "balanced"
    gran_profile = resolve_granularity(gran_preset, band=band)

    # Populate trace routing info
    if trace is not None:
        trace.chosen_params["pipeline_tier"] = "large"
        trace.chosen_params["embedding_dim"] = D
        trace.chosen_params["dim_band"] = band

    norm_embeddings = normalize_vectors(embeddings)

    # -------------------------------------------------------------------
    # Phase 1: Profiling
    # -------------------------------------------------------------------
    _log("Phase 1: Profiling dataset...")
    t_profile_start = time.perf_counter()
    profile = compute_dataset_profile(norm_embeddings, sample_size=min(N, 4096))
    if trace is not None:
        trace.time("profile", time.perf_counter() - t_profile_start)
    _log(
        f"  Effective rank: {profile.effective_rank}/{D}, "
        f"Density CV: {profile.local_density_cv:.2f}, "
        f"Cluster tendency: {profile.cluster_tendency:.2%}"
    )

    # -------------------------------------------------------------------
    # Phase 2: Multi-reduction candidates
    # -------------------------------------------------------------------
    reduction_method = config.get_reduction_for_strategy("large") if config is not None else "pca"
    reduced_reps: Dict[int, np.ndarray] = {}
    reduction_candidates: List[int]

    _log("Phase 2: Computing reduction strategy...")
    t_reduction_start = time.perf_counter()

    if reduction_method is None:
        reduction_candidates = [D]
        reduced_reps[D] = norm_embeddings
        _log("  Reduction disabled by config; using normalized embedding space")
    else:
        base_dim = config.get_reduction_components(D, N) if config is not None else None
        if base_dim is not None:
            target_dim = compute_large_reduction_dimension(profile, lambda d, n: base_dim)
        else:
            target_dim = compute_large_reduction_dimension(profile)

        # Generate 2 reduction candidates (speed-conscious for large datasets)
        reduction_candidates = compute_reduction_candidates(
            profile, target_dim, num_candidates=2,
        )
        _log(f"  Reduction candidates: {reduction_candidates}")

        for target_components in reduction_candidates:
            if target_components >= D:
                reduced_reps[target_components] = norm_embeddings
                continue
            _log(f"  PCA: {D} -> {target_components}")
            reducer = get_reducer(reduction_method, target_components, N)
            reduced_reps[target_components] = normalize_vectors(
                reducer.fit_transform(norm_embeddings)
            )

    if trace is not None:
        trace.time("reduction", time.perf_counter() - t_reduction_start)

    # -------------------------------------------------------------------
    # Phase 3: Coarse partitioning parameters
    # -------------------------------------------------------------------
    _log("Phase 3: Computing coarse partitions...")
    target_shard_size = compute_large_target_shard_size(profile)
    n_coarse = compute_large_coarse_partition_count(profile)
    n_coarse = min(max(1, n_coarse), N)
    _log(f"  Target shard size: {target_shard_size}, coarse clusters: {n_coarse}")

    if N <= 1:
        return np.arange(N, dtype=np.int32)

    # Use the first (smallest) reduction candidate for coarse partitioning
    coarse_rep_key = reduction_candidates[0]
    coarse_embeddings = reduced_reps[coarse_rep_key]

    # -------------------------------------------------------------------
    # Phase 4: MiniBatchKMeans coarse partitioning
    # -------------------------------------------------------------------
    _log("Phase 4: Running initial MiniBatchKMeans...")
    t_coarse_start = time.perf_counter()
    kmeans = MiniBatchKMeans(
        n_clusters=n_coarse,
        batch_size=max(1024, min(8192, target_shard_size * 2)),
        random_state=random_state,
        n_init=3,
    )
    coarse_labels = kmeans.fit_predict(coarse_embeddings)

    if trace is not None:
        trace.time("coarse_kmeans", time.perf_counter() - t_coarse_start)

    # -------------------------------------------------------------------
    # Phase 5: Shard balancing
    # -------------------------------------------------------------------
    _log("Phase 5: Balancing shard sizes...")
    shard_assignments = _balance_shards(
        coarse_labels,
        coarse_embeddings,
        target_size=target_shard_size,
        max_depth=2,
        random_state=random_state,
    )
    shard_ids = [int(shard_id) for shard_id in np.unique(shard_assignments) if shard_id >= 0]
    _log(f"  Final shard count: {len(shard_ids)}")

    # -------------------------------------------------------------------
    # Phase 6: Per-shard density clustering with multi-reduction search
    # -------------------------------------------------------------------
    _log("Phase 6: Local clustering within shards...")
    t_shard_start = time.perf_counter()
    final_labels = np.full(N, -1, dtype=np.int32)
    label_to_shard: Dict[int, int] = {}
    next_global_label = 0

    # HDBSCAN methods from band grid
    methods = list(grid.hdbscan_methods)

    # Collect per-shard scores aligned to shard_ids order
    shard_scores: List[Optional[float]] = []

    for shard_id in shard_ids:
        shard_mask = shard_assignments == shard_id
        shard_indices = np.where(shard_mask)[0]
        shard_size = len(shard_indices)
        if shard_size == 0:
            shard_scores.append(None)
            continue

        shard_orig = norm_embeddings[shard_mask]

        if shard_size == 1:
            final_labels[shard_indices[0]] = next_global_label
            label_to_shard[next_global_label] = shard_id
            next_global_label += 1
            shard_scores.append(None)
            continue

        # Run multi-reduction shard clustering
        local_result = _cluster_single_shard(
            shard_mask,
            reduced_reps,
            reduction_candidates,
            shard_orig,
            profile,
            gran_profile,
            HDBSCAN,
            methods=methods,
            random_state=random_state,
        )

        local_labels = local_result["labels"]
        shard_score = local_result["score"]

        # Record per-shard score: None if no valid clusters or non-finite
        if not np.any(local_labels >= 0) or not np.isfinite(shard_score) or shard_score < 0:
            shard_scores.append(None)
        else:
            shard_scores.append(float(shard_score))

        if not np.any(local_labels >= 0):
            local_labels = _fallback_cluster_shard(shard_orig)

        if not np.any(local_labels >= 0):
            continue

        local_labels = recover_noise_with_confidence(shard_orig, local_labels)
        unique_local = [int(label) for label in np.unique(local_labels) if label >= 0]
        local_to_global = {label: next_global_label + idx for idx, label in enumerate(unique_local)}

        for local_idx, global_idx in enumerate(shard_indices):
            label = int(local_labels[local_idx])
            if label < 0:
                continue
            global_label = local_to_global[label]
            final_labels[global_idx] = global_label
            label_to_shard[global_label] = shard_id

        next_global_label += len(unique_local)

    if trace is not None:
        trace.time("shard_clustering", time.perf_counter() - t_shard_start)

    # Store per-shard scores in trace
    if trace is not None:
        trace.chosen_params["shard_scores"] = shard_scores

    # Check quality floor: >30% of shards below 0.30 (None counts as below-threshold)
    if shard_scores:
        n_below = sum(1 for s in shard_scores if s is None or s < 0.30)
        if n_below / len(shard_scores) > 0.30:
            if trace is not None:
                trace.warn("large-low-shard-quality", set_low_confidence=True)

    if not np.any(final_labels >= 0):
        _log("All shard-local clustering failed; running conservative global fallback")
        from sklearn.cluster import MiniBatchKMeans
        global_fallback = MiniBatchKMeans(
            n_clusters=max(2, int(N / 1500)),
            batch_size=max(1024, min(4096, N)),
            n_init=3,
            random_state=random_state,
        )
        fallback_labels = global_fallback.fit_predict(coarse_embeddings)
        if np.any(fallback_labels >= 0):
            final_labels = fallback_labels.astype(np.int32)

    # -------------------------------------------------------------------
    # Phase 7: Global centroid stitching
    # -------------------------------------------------------------------
    _log("Phase 7: Global centroid stitching...")
    t_stitch_start = time.perf_counter()

    # Profile-based threshold: slightly more aggressive since granularity
    # merge downstream handles fine-grained merging
    stitch_threshold = 0.93 if profile.near_duplicate_ratio < 0.05 else 0.96
    final_labels = _global_stitch_clusters(
        norm_embeddings,
        final_labels,
        label_to_shard,
        similarity_threshold=stitch_threshold,
    )

    if trace is not None:
        trace.time("stitch", time.perf_counter() - t_stitch_start)

    # -------------------------------------------------------------------
    # Phase 8: Full post-processing pipeline (matching medium quality)
    # -------------------------------------------------------------------
    _log("Phase 8: Post-processing...")
    t_postprocess_start = time.perf_counter()

    # Step 1: Noise recovery
    final_labels = recover_noise_with_confidence(norm_embeddings, final_labels)

    # Step 2: Split oversized clusters (if UMAP available)
    if UMAP is not None:
        final_labels = split_oversized_clusters(
            norm_embeddings,
            final_labels,
            lambda e, l: score_clustering(e, l),
            umap_cls=UMAP,
            hdbscan_cls=HDBSCAN,
            max_split_passes=2,
            max_clusters_to_split=5,
            size_threshold_ratio=0.08,
            min_quality_gain=0.005,
        )

    # Step 3: Merge near-duplicates
    final_labels = merge_near_duplicate_clusters(norm_embeddings, final_labels)

    # Step 4: Noise recovery again after merges
    final_labels = recover_noise_with_confidence(norm_embeddings, final_labels)

    # Step 5: Granularity-driven centroid merge
    final_labels, n_merges = merge_clusters_by_centroid_similarity(
        norm_embeddings, final_labels, gran_profile.merge_centroid_threshold,
    )

    # Step 6: Final noise recovery
    final_labels = recover_noise_with_confidence(norm_embeddings, final_labels)

    # Step 7: CentroidFallback for residual noise
    if float(np.mean(final_labels == -1)) > 0.30 and np.any(final_labels >= 0):
        _log("Applying centroid fallback for residual noise")
        fallback = CentroidFallback(noise_threshold=0.30, metric="cosine")
        final_labels = fallback.recluster_noise(norm_embeddings, final_labels)

    result_labels = compact_labels(final_labels).astype(np.int32)

    if trace is not None:
        trace.time("postprocess", time.perf_counter() - t_postprocess_start)

    # Populate trace with chosen params and metrics
    if trace is not None:
        try:
            final_metrics = score_clustering(norm_embeddings, result_labels)
            trace.intrinsic_metrics = dict(final_metrics)
        except Exception:
            pass
        if n_merges > 0:
            trace.chosen_params["granularity_merges_applied"] = n_merges

    return result_labels


def _cluster_single_shard(
    shard_mask: np.ndarray,
    reduced_reps: Dict[int, np.ndarray],
    reduction_candidates: List[int],
    shard_orig: np.ndarray,
    profile,
    gran_profile,
    hdbscan_cls,
    methods: List[str],
    random_state: int = 42,
) -> Dict[str, object]:
    """Cluster a single shard with multi-reduction and granularity-aware search."""
    shard_size = len(shard_orig)

    # Granularity-aware min_cluster_size candidates (sqrt-scaled, not linear)
    mcs_candidates = mcs_candidate_spread(gran_profile, shard_size)
    # Cap at 3 candidates to keep shard search bounded
    mcs_candidates = mcs_candidates[:3]

    # Adaptive min_samples from profile
    min_samples_candidates = compute_large_min_samples_candidates(
        profile, mcs_candidates[0] if mcs_candidates else 10,
    )[:2]

    candidates: List[Dict[str, Any]] = []

    # Search across reduction candidates
    for rep_key in reduction_candidates:
        rep_data = reduced_reps[rep_key]
        shard_search = rep_data[shard_mask]

        for min_cluster_size in mcs_candidates:
            for min_samples in min_samples_candidates:
                for method in methods:
                    try:
                        clusterer = hdbscan_cls(
                            min_cluster_size=min_cluster_size,
                            min_samples=min_samples,
                            metric="euclidean",
                            cluster_selection_method=method,
                            gen_min_span_tree=False,
                            approx_min_span_tree=True,
                        )
                        with _seeded_global_numpy(random_state):
                            labels = clusterer.fit_predict(shard_search)
                        if not np.any(labels >= 0):
                            continue

                        density = getattr(clusterer, "relative_validity_", None)
                        metrics = score_clustering(shard_orig, labels, density)
                        candidates.append({
                            "labels": labels.astype(np.int32),
                            "score": metrics["score"],
                            "params": {
                                "min_cluster_size": int(min_cluster_size),
                                "min_samples": int(min_samples),
                                "method": method,
                                "rep_key": int(rep_key),
                                "metrics": metrics,
                            }
                        })
                    except Exception:
                        continue

    if not candidates:
        return {"labels": np.full(shard_size, -1, dtype=np.int32), "params": {}, "score": -1.0}

    candidates.sort(key=lambda x: x["score"], reverse=True)
    top_candidates = candidates[:3]

    best_post_score = -2.0
    best_candidate = top_candidates[0]
    best_post_labels = best_candidate["labels"]

    for cand in top_candidates:
        cand_labels = cand["labels"].copy()
        cand_labels = recover_noise_with_confidence(shard_orig, cand_labels)
        post_metrics = score_clustering(shard_orig, cand_labels, None)
        post_score = post_metrics.get("score", -1.0)

        if post_score > best_post_score:
            best_post_score = post_score
            best_post_labels = cand_labels
            best_candidate = cand

    return {"labels": best_post_labels, "params": best_candidate["params"], "score": best_post_score}


def _fallback_cluster_shard(shard_orig: np.ndarray) -> np.ndarray:
    """Conservative fallback when local density clustering fails."""
    shard_size = len(shard_orig)
    if shard_size == 0:
        return np.array([], dtype=np.int32)

    if shard_size <= 3:
        return np.arange(shard_size, dtype=np.int32)

    centroid = normalize_vectors(np.mean(shard_orig, axis=0, keepdims=True))[0]
    cohesion = float(np.mean(np.clip(shard_orig @ centroid, -1.0, 1.0)))

    if cohesion >= 0.92 and shard_size <= 16:
        return np.zeros(shard_size, dtype=np.int32)

    return np.full(shard_size, -1, dtype=np.int32)


def _balance_shards(
    coarse_labels: np.ndarray,
    embeddings: np.ndarray,
    target_size: int = 1500,
    max_depth: int = 2,
    random_state: int = 42,
) -> np.ndarray:
    """Recursively split oversized shards to balance partition sizes."""
    from sklearn.cluster import MiniBatchKMeans

    balanced = coarse_labels.copy()
    next_label = int(np.max(coarse_labels)) + 1

    for _ in range(max_depth):
        changed = False
        for label in [int(value) for value in np.unique(balanced) if value >= 0]:
            mask = balanced == label
            size = int(np.sum(mask))
            if size <= max(target_size, int(target_size * 1.35)):
                continue

            shard_data = embeddings[mask]
            n_splits = int(np.ceil(size / max(1, target_size)))
            n_splits = min(size, max(2, n_splits))
            if n_splits >= size:
                continue

            try:
                splitter = MiniBatchKMeans(
                    n_clusters=n_splits,
                    batch_size=max(256, min(4096, target_size)),
                    random_state=random_state,
                    n_init=2,
                )
                sub_labels = splitter.fit_predict(shard_data)
            except Exception:
                continue

            shard_indices = np.where(mask)[0]
            for local_idx, global_idx in enumerate(shard_indices):
                sub_label = int(sub_labels[local_idx])
                if sub_label == 0:
                    balanced[global_idx] = label
                else:
                    balanced[global_idx] = next_label + sub_label - 1

            next_label += n_splits - 1
            changed = True

        if not changed:
            break

    return balanced


def _global_stitch_clusters(
    embeddings: np.ndarray,
    labels: np.ndarray,
    label_to_shard: Dict[int, int],
    similarity_threshold: float = 0.95,
) -> np.ndarray:
    """Merge clearly duplicated clusters across shard boundaries."""
    stitched = labels.copy()
    stats = compute_cluster_stats(embeddings, stitched)
    valid_labels = [int(label) for label in stats["labels"]]
    if len(valid_labels) < 2:
        return stitched

    if len(valid_labels) > 5000:
        import warnings as _warnings
        _warnings.warn(
            f"{len(valid_labels)} clusters in stitch; skipping full stitch (approximate search)",
            UserWarning,
            stacklevel=2,
        )
        return stitched

    centroids = stats["centroids"]
    similarities = np.clip(centroids @ centroids.T, -1.0, 1.0)
    distances = 1.0 - similarities
    radii = np.array(
        [
            float(np.percentile(1.0 - np.clip(embeddings[stitched == label] @ centroids[idx], -1.0, 1.0), 90))
            if np.any(stitched == label)
            else 0.0
            for idx, label in enumerate(valid_labels)
        ],
        dtype=np.float32,
    )

    top_k = min(3, len(valid_labels) - 1)
    if top_k < 1:
        return stitched

    neighbor_order = np.argsort(-similarities, axis=1)

    parent = list(range(len(valid_labels)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        root_a = find(a)
        root_b = find(b)
        if root_a != root_b:
            parent[root_b] = root_a

    for i in range(len(valid_labels)):
        shard_i = label_to_shard.get(valid_labels[i], -1)
        i_neighbors = [int(idx) for idx in neighbor_order[i, 1 : top_k + 1]]
        for j in i_neighbors:
            if j <= i:
                continue

            shard_j = label_to_shard.get(valid_labels[j], -1)
            if shard_i == shard_j:
                continue

            j_neighbors = [int(idx) for idx in neighbor_order[j, 1 : top_k + 1]]
            if i not in j_neighbors:
                continue

            similarity = float(similarities[i, j])
            distance = float(distances[i, j])
            radius_limit = max(float(radii[i]), float(radii[j])) + 0.03

            if similarity < similarity_threshold:
                continue
            if distance > max(0.08, radius_limit):
                continue

            union(i, j)

    label_roots: Dict[int, int] = {}
    for idx, label in enumerate(valid_labels):
        root = find(idx)
        root_label = valid_labels[root]
        label_roots[label] = root_label

    for old_label, root_label in label_roots.items():
        if old_label != root_label:
            stitched[stitched == old_label] = root_label

    return stitched
