from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from semantic_clusterer.dim_bands import resolve_dim_band
from semantic_clusterer.optional_deps import _seeded_global_numpy
from semantic_clusterer.pipeline.postprocess import (
    compact_labels,
    merge_near_duplicate_clusters,
    recover_noise_with_confidence,
    split_oversized_clusters,
)
from semantic_clusterer.pipeline.profile import compute_dataset_profile
from semantic_clusterer.pipeline.quality import score_clustering, should_trigger_refinement
from semantic_clusterer.pipeline.tuning import (
    compute_medium_hdbscan_candidates,
    compute_medium_reduction_dimension,
    compute_reduction_candidates,
    compute_refinement_trigger_thresholds,
    compute_umap_parameters,
    get_band_grid,
)
from semantic_clusterer.reduction.base import get_reducer
from semantic_clusterer.utils.similarity import normalize_vectors
from semantic_clusterer.pipeline.utils import _select_diverse_candidates


def cluster_medium(
    embeddings: np.ndarray,
    config=None,
    *,
    random_state: int = 42,
    trace=None,
    log_fn=None,
) -> np.ndarray:
    """Clustering strategy for medium datasets (5K-50K).

    Args:
        embeddings: Float32 array of shape (N, D).
        config: Optional ClustererConfig (positional, kept for backward compat).
        random_state: Integer RNG seed (keyword-only, default 42).
        trace: Optional _PipelineTrace accumulator (keyword-only).
        log_fn: Optional logging callable (keyword-only).

    Returns:
        int32 numpy array of cluster labels, shape (N,).
    """

    def _log(msg: str) -> None:
        if log_fn:
            log_fn(msg)

    N, D = embeddings.shape
    _log(f"Using adaptive medium strategy (N={N}, D={D})")

    try:
        from hdbscan import HDBSCAN
    except ImportError as exc:
        raise ImportError(
            "hdbscan is required for the medium pipeline. Install it with: pip install hdbscan"
        ) from exc

    from semantic_clusterer.optional_deps import try_import_umap
    UMAP = try_import_umap()  # None if umap-learn not installed

    # Emit umap-unavailable warning once if UMAP is not available
    _umap_warning_msg = "umap-unavailable, used PCA-only fallback"
    if UMAP is None:
        import warnings as _warnings
        _warnings.warn(_umap_warning_msg, UserWarning, stacklevel=2)
        if trace is not None:
            trace.warn(_umap_warning_msg)

    norm_embeddings = normalize_vectors(embeddings)

    # Resolve dim band and get parameter grid
    band = resolve_dim_band(D)
    grid = get_band_grid(band, "medium")

    # Populate trace routing info
    if trace is not None:
        trace.chosen_params["pipeline_tier"] = "medium"
        trace.chosen_params["embedding_dim"] = D
        trace.chosen_params["dim_band"] = band

    # -----------------------------------------------------------------------
    # Phase 0: xhigh PCA pre-reduction
    # -----------------------------------------------------------------------
    # When band == "xhigh", apply a PCA pre-reduction step before any UMAP
    # step, using grid.pca_targets[0] as the target dimensionality.
    t_pca_start = time.perf_counter()
    pre_reduced_embeddings = norm_embeddings
    pca_pre_dim: Optional[int] = None

    if band == "xhigh":
        pca_pre_target = grid.pca_targets[0]
        if pca_pre_target < D:
            _log(f"Phase 0: xhigh PCA pre-reduction: {D} -> {pca_pre_target}")
            from sklearn.decomposition import PCA
            pca_pre = PCA(
                n_components=pca_pre_target,
                svd_solver="randomized",
                random_state=random_state,
            )
            pre_reduced_embeddings = normalize_vectors(pca_pre.fit_transform(norm_embeddings))
            pca_pre_dim = pca_pre_target
        else:
            pca_pre_dim = D

    if trace is not None:
        trace.time("pca_pre", time.perf_counter() - t_pca_start)

    _log("Phase 1: Profiling dataset...")
    t_profile_start = time.perf_counter()
    profile = compute_dataset_profile(norm_embeddings)
    if trace is not None:
        trace.time("profile", time.perf_counter() - t_profile_start)
    _log(
        f"  Effective rank: {profile.effective_rank}/{D} "
        f"({profile.effective_rank_ratio():.2%}), "
        f"Duplicates: {profile.duplicate_ratio:.1%}, "
        f"Cluster tendency: {profile.cluster_tendency:.2%}"
    )

    reduction_method = config.get_reduction_for_strategy("medium") if config is not None else "pca"
    reduction_candidates: List[int]
    reduced_reps: Dict[int, np.ndarray] = {}

    _log("Phase 2: Computing reduction strategy...")
    t_reduction_start = time.perf_counter()
    if reduction_method is None:
        reduction_candidates = [D]
        reduced_reps[D] = pre_reduced_embeddings
        _log("  Reduction disabled by config; clustering in normalized embedding space")
    else:
        # Compute reduction candidates using the profile-aware helper
        # (which internally can be informed by the band grid)
        base_dim = config.get_reduction_components(D, N) if config is not None else None
        if base_dim is not None:
            target_dim = compute_medium_reduction_dimension(profile, lambda d, n: base_dim)
        else:
            target_dim = compute_medium_reduction_dimension(profile)
        reduction_candidates = compute_reduction_candidates(profile, target_dim, num_candidates=3)
        _log(f"  Reduction candidates: {reduction_candidates}")

        # The effective input dim is the pre-reduced dim (for xhigh) or D
        input_dim = pca_pre_dim if pca_pre_dim is not None else D

        for target_components in reduction_candidates:
            if target_components >= input_dim:
                # No further reduction needed; use the pre-reduced embeddings
                if input_dim not in reduced_reps:
                    reduced_reps[input_dim] = pre_reduced_embeddings
                # Remap this candidate to input_dim
                reduced_reps[target_components] = pre_reduced_embeddings
                continue

            _log(f"  PCA: {input_dim} -> {target_components}")
            reducer = get_reducer(reduction_method, target_components, N)
            reduced = reducer.fit_transform(pre_reduced_embeddings)
            reduced_reps[target_components] = normalize_vectors(reduced)

    if trace is not None:
        trace.time("reduction", time.perf_counter() - t_reduction_start)

    # -----------------------------------------------------------------------
    # Phase 3: HDBSCAN candidate generation
    # -----------------------------------------------------------------------
    # Use grid-based ratios for min_cluster_size candidates
    hdbscan_min_cluster_sizes = [
        max(5, int(ratio * N))
        for ratio in grid.hdbscan_min_cluster_size_ratios
    ]
    # Deduplicate and sort
    hdbscan_min_cluster_sizes = sorted(list(set(hdbscan_min_cluster_sizes)))

    # Also pull min_samples and methods from the grid
    min_samples_values = list(grid.hdbscan_min_samples)
    methods = list(grid.hdbscan_methods)

    # User override: pin to a single mcs/ms when explicitly set on config
    if config is not None and getattr(config, "min_cluster_size", None) is not None:
        hdbscan_min_cluster_sizes = [int(config.min_cluster_size)]
    if config is not None and getattr(config, "min_samples", None) is not None:
        min_samples_values = [int(config.min_samples)]

    # Apply granularity floor
    from semantic_clusterer.pipeline.granularity import (
        resolve_granularity, apply_mcs_floor, mcs_candidate_spread,
    )
    if config is not None:
        gran_preset = getattr(config, "cluster_granularity", "balanced")
        band_for_gran = band  # already resolved earlier in the function
        gran_profile = resolve_granularity(gran_preset, band=band_for_gran)
        user_mcs = getattr(config, "min_cluster_size", None)
        if user_mcs is not None:
            hdbscan_min_cluster_sizes = [int(user_mcs)]
        else:
            # Floor-anchored geometric spread gives HDBSCAN a real search
            # range around the floor (3 values, kept compact so the medium
            # sweep stays affordable).
            hdbscan_min_cluster_sizes = mcs_candidate_spread(gran_profile, N)

    _log(
        f"Phase 3: HDBSCAN candidates mcs={hdbscan_min_cluster_sizes}, "
        f"min_samples={min_samples_values}, methods={methods}"
    )

    candidates: List[Dict[str, Any]] = []
    umap_cache: Dict[Tuple[int, int, int, float], np.ndarray] = {}

    def evaluate_hdbscan(
        cluster_input: np.ndarray,
        params: Dict[str, Any],
    ) -> None:
        try:
            clusterer = HDBSCAN(
                min_cluster_size=params["min_cluster_size"],
                min_samples=params["min_samples"],
                metric="euclidean",
                cluster_selection_method=params["method"],
                cluster_selection_epsilon=0.0,
                gen_min_span_tree=False,
                approx_min_span_tree=True,
            )
            with _seeded_global_numpy(random_state):
                labels = clusterer.fit_predict(cluster_input)
            density = getattr(clusterer, "relative_validity_", None)
            metrics = score_clustering(norm_embeddings, labels, density)
            if metrics["n_clusters"] <= 0 and metrics["coverage"] <= 0.0:
                return

            candidates.append(
                {
                    "score": metrics["score"],
                    "metrics": metrics,
                    "labels": labels.astype(np.int32),
                    "params": dict(params),
                }
            )
        except Exception:
            return

    def evaluate_direct_rep(rep_key: int, rep_embeddings: np.ndarray) -> None:
        direct_mcs = hdbscan_min_cluster_sizes[: 4 if N <= 20_000 else 3]
        direct_ms = min_samples_values[: len(min_samples_values) if N <= 25_000 else 2]
        direct_methods = methods if profile.local_density_cv > 0.45 else methods[:1]

        for mcs in direct_mcs:
            for ms in direct_ms:
                for method in direct_methods:
                    evaluate_hdbscan(
                        rep_embeddings,
                        {
                            "dim": rep_key,
                            "path": "direct",
                            "umap": None,
                            "min_cluster_size": int(mcs),
                            "min_samples": int(ms),
                            "method": method,
                        },
                    )

    def evaluate_umap_rep(rep_key: int, rep_embeddings: np.ndarray, refine: bool = False) -> None:
        if UMAP is None:
            return

        umap_params = compute_umap_parameters(profile, rep_key)
        nn_candidates = umap_params["n_neighbors_candidates"]
        nc_candidates = umap_params["n_components_candidates"]

        # Clip to valid range for this dataset
        nn_candidates = [max(5, min(N - 1, nn)) for nn in nn_candidates]
        nn_candidates = sorted(list(set(nn_candidates)))

        if refine and nn_candidates:
            nn_candidates = sorted(
                list(
                    {
                        max(5, min(N - 1, int(nn_candidates[0] * 0.80))),
                        nn_candidates[0],
                        max(5, min(N - 1, int(nn_candidates[-1] * 1.25))),
                    }
                )
            )

        if N > 25_000:
            nn_candidates = nn_candidates[:2]
            nc_candidates = nc_candidates[:1]

        umap_methods = methods if profile.local_density_cv > 0.55 else methods[:1]
        umap_mcs = hdbscan_min_cluster_sizes[: 3 if N > 20_000 else 4]
        umap_ms = min_samples_values[:2]

        for nn in nn_candidates:
            for nc in nc_candidates:
                cache_key = (rep_key, nn, nc, 0.0)
                if cache_key not in umap_cache:
                    try:
                        reducer = UMAP(
                            n_neighbors=nn,
                            n_components=nc,
                            min_dist=0.0,
                            metric="cosine",
                            n_jobs=-1,
                        )
                        umap_cache[cache_key] = reducer.fit_transform(rep_embeddings)
                    except Exception:
                        continue

                reduced = umap_cache.get(cache_key)
                if reduced is None:
                    continue

                for mcs in umap_mcs:
                    for ms in umap_ms:
                        for method in umap_methods:
                            evaluate_hdbscan(
                                reduced,
                                {
                                    "dim": rep_key,
                                    "path": "umap",
                                    "umap": (nn, nc),
                                    "min_cluster_size": int(mcs),
                                    "min_samples": int(ms),
                                    "method": method,
                                },
                            )

    _log("Phase 4: Running primary candidate sweep...")
    t_sweep_start = time.perf_counter()
    for rep_key in reduction_candidates:
        rep_embeddings = reduced_reps[rep_key]
        evaluate_direct_rep(rep_key, rep_embeddings)

    if reduction_method is not None and UMAP is not None:
        center_idx = len(reduction_candidates) // 2
        umap_rep_keys = [reduction_candidates[center_idx]]
        if profile.effective_rank_ratio() > 0.30 and len(reduction_candidates) > 1:
            extra_key = reduction_candidates[-1]
            if extra_key == umap_rep_keys[0]:
                extra_key = reduction_candidates[0]
            umap_rep_keys.append(extra_key)
        umap_rep_keys = sorted(list(set(int(key) for key in umap_rep_keys)))

        for rep_key in umap_rep_keys:
            if rep_key <= 64 and profile.local_density_cv < 0.40:
                continue
            evaluate_umap_rep(rep_key, reduced_reps[rep_key], refine=False)

    if trace is not None:
        trace.time("sweep", time.perf_counter() - t_sweep_start)

    if not candidates:
        _log("Warning: no adaptive candidates succeeded; using conservative HDBSCAN fallback")
        fallback_input = reduced_reps[reduction_candidates[0]]
        fallback_mcs = max(5, int(max(1, N) * 0.001))
        evaluate_hdbscan(
            fallback_input,
            {
                "dim": reduction_candidates[0],
                "path": "direct",
                "umap": None,
                "min_cluster_size": fallback_mcs,
                "min_samples": 2,
                "method": "eom",
            },
        )

    if not candidates:
        if trace is not None:
            trace.chosen_params.update({
                "pipeline_tier": "medium",
                "embedding_dim": D,
                "dim_band": band,
            })
        return compact_labels(np.full(N, -1, dtype=np.int32))

    candidates.sort(key=lambda item: item["score"], reverse=True)
    best = candidates[0]
    best_labels = best["labels"]
    best_metrics = best["metrics"]
    _log(
        f"Best primary candidate: score={best_metrics['score']:.4f}, "
        f"coverage={best_metrics['coverage']:.2%}, "
        f"noise={best_metrics['noise_ratio']:.2%}"
    )

    thresholds = compute_refinement_trigger_thresholds(profile)
    should_refine, reason = should_trigger_refinement(best_metrics, thresholds)
    original_candidates = candidates
    refinement_candidates: List[Dict[str, Any]] = []
    if should_refine:
        _log(f"Phase 5: Refinement pass (reason: {reason})...")
        t_refine_start = time.perf_counter()
        candidates = refinement_candidates

        best_dim = int(best["params"]["dim"])
        best_rep = reduced_reps[best_dim]
        best_mcs = int(best["params"]["min_cluster_size"])
        best_ms = int(best["params"].get("min_samples", 1))
        best_method = str(best["params"]["method"])

        mcs_variants = sorted(
            list(
                {
                    max(5, int(best_mcs * 0.70)),
                    best_mcs,
                    max(5, int(best_mcs * 1.30)),
                }
            )
        )
        ms_variants = sorted(list({max(1, best_ms - 1), best_ms, best_ms + 2}))
        method_variants = sorted(list(set([best_method] + methods)))

        for mcs in mcs_variants:
            for ms in ms_variants:
                for method in method_variants:
                    evaluate_hdbscan(
                        best_rep,
                        {
                            "dim": best_dim,
                            "path": "direct",
                            "umap": None,
                            "min_cluster_size": int(mcs),
                            "min_samples": int(ms),
                            "method": method,
                        },
                    )

        if reduction_method is not None and UMAP is not None and N <= 35_000:
            evaluate_umap_rep(best_dim, best_rep, refine=True)

    candidates = original_candidates + refinement_candidates
    candidates.sort(key=lambda item: item["score"], reverse=True)
    top_candidates = _select_diverse_candidates(candidates, N)

    best_post_score = -2.0
    best_candidate = top_candidates[0]
    best_post_labels = best_candidate["labels"]
    best_post_metrics = {}

    from semantic_clusterer.pipeline.granularity import resolve_granularity
    from semantic_clusterer.pipeline.postprocess import merge_clusters_by_centroid_similarity

    gran_preset = getattr(config, "cluster_granularity", "balanced") if config else "balanced"
    gran_profile = resolve_granularity(gran_preset, band=band)

    _log("Phase 6: Post-Selection Rescoring & Post-processing...")
    t_postprocess_start = time.perf_counter()

    for cand in top_candidates:
        cand_labels = cand["labels"].copy()

        cand_labels = recover_noise_with_confidence(norm_embeddings, cand_labels)
        cand_labels = split_oversized_clusters(
            norm_embeddings,
            cand_labels,
            lambda e, l: score_clustering(e, l),
            umap_cls=UMAP,
            hdbscan_cls=HDBSCAN,
            max_split_passes=2,
            max_clusters_to_split=3 if N > 15_000 else 2,
            size_threshold_ratio=0.10 if profile.imbalance_tendency > 0.60 else 0.12,
            min_quality_gain=0.005,
        )
        cand_labels = merge_near_duplicate_clusters(norm_embeddings, cand_labels)
        cand_labels = recover_noise_with_confidence(norm_embeddings, cand_labels)
        cand_labels = compact_labels(cand_labels).astype(np.int32)
        
        cand_labels, n_merges = merge_clusters_by_centroid_similarity(
            norm_embeddings, cand_labels, gran_profile.merge_centroid_threshold
        )

        post_metrics = score_clustering(norm_embeddings, cand_labels, None)
        post_score = post_metrics.get("score", -1.0)

        if post_score > best_post_score:
            best_post_score = post_score
            best_post_labels = cand_labels
            best_candidate = cand
            best_post_metrics = post_metrics
            cand["n_merges"] = n_merges

    if trace is not None:
        trace.time("postprocess", time.perf_counter() - t_postprocess_start)

    best_labels = best_post_labels
    best_metrics = best_post_metrics
    best_params = best_candidate["params"]
    result_labels = compact_labels(best_labels).astype(np.int32)

    # Populate trace with chosen params and metrics
    if trace is not None:
        umap_info = best_params.get("umap")
        trace.chosen_params.update({
            "pipeline_tier": "medium",
            "embedding_dim": D,
            "dim_band": band,
            "pca_pre_dim": pca_pre_dim,
            "reduction_dim": best_params.get("dim"),
            "umap_n_neighbors": umap_info[0] if umap_info else None,
            "umap_n_components": umap_info[1] if umap_info else None,
            "hdbscan_min_cluster_size": best_params.get("min_cluster_size"),
            "hdbscan_min_samples": best_params.get("min_samples"),
            "hdbscan_method": best_params.get("method"),
            "path": best_params.get("path"),
        })
        n_merges = best_candidate.get("n_merges", 0)
        if n_merges > 0:
            trace.chosen_params["granularity_merges_applied"] = n_merges
        trace.intrinsic_metrics = dict(best_metrics)

    return result_labels
