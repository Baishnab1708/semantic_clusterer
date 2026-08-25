"""k_split.py — SemanticKSplit: partition a corpus into exactly k clusters.

``SemanticKSplit`` complements ``SemanticClusterer``.  Where
``SemanticClusterer`` discovers a *variable* number of density-based clusters,
``SemanticKSplit`` partitions a corpus into **exactly `k` non-empty clusters**
supplied by the caller (``2 <= k <= N_Unique``).

Public API
----------
::

    from semantic_clusterer import SemanticKSplit

    ks = SemanticKSplit(k=3)
    labels   = ks.split_labels(texts)           # int32 ndarray, shape (N,)
    clusters = ks.split(texts)                  # List[List[str]], len == k
    labels, report = ks.split_with_report(texts)
    embeddings = ks.embed(texts)

Notes
-----
- This module does **not** import ``hdbscan``.
- Label ``-1`` is reserved exclusively for filtered/missing rows
  (None, NaN, or empty after preprocessing).  Every preprocessed row
  receives a label in ``[0, k-1]``.
- Determinism is guaranteed within the ``Determinism_Scope`` defined in the
  same library version, same Python minor, same OS family, same
  numpy/scikit-learn major.minor.

"""

from __future__ import annotations

import math
import sys
import time
import warnings
from dataclasses import replace
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np

from semantic_clusterer.config import ClustererConfig, SemanticKSplitConfig, _validate_config_dict, _PUBLIC_KSPLIT_FIELDS
from semantic_clusterer.core import _OVERSIZED_ERROR_MSG_FMT, _OVERSIZED_WARN_MSG_FMT
from semantic_clusterer.dim_bands import resolve_dim_band
from semantic_clusterer.embedding.adapters import (
    EmbeddingModel,
    normalize_embedding_model,
    validate_embeddings,
)
from semantic_clusterer.k_algorithms.agglomerative import _agglomerative_cut_k
from semantic_clusterer.k_algorithms.balanced import _balanced_kmeans
from semantic_clusterer.k_algorithms.bisecting import _bisecting_kmeans
from semantic_clusterer.k_algorithms.degenerate import _all_identical, _round_robin_labels
from semantic_clusterer.k_algorithms.minibatch_assign import _minibatch_kmeans_assign
from semantic_clusterer.k_algorithms.oversized import _oversized_subsample_and_assign_k
from semantic_clusterer.k_algorithms.repair import _repair_empty_clusters
from semantic_clusterer.k_algorithms.selection import _select_k_algorithm
from semantic_clusterer.k_algorithms.spectral import _spectral_cosine
from semantic_clusterer.output.formatter import ClusterResult, OutputFormatter
from semantic_clusterer.pipeline.quality import score_clustering
from semantic_clusterer.k_algorithms.restart import _selection_score
from semantic_clusterer.preprocessing.clean import TextPreprocessor
from semantic_clusterer.report import ClusteringReport, _PipelineTrace
from semantic_clusterer.utils.similarity import normalize_vectors

try:
    from semantic_clusterer import __version__ as _LIB_VERSION
except Exception:
    _LIB_VERSION = "0.1.0"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_OVERSIZED_LIMIT: int = 200_000


class SemanticKSplit:
    """Partition a text corpus into exactly *k* non-empty clusters.

    Unlike :class:`SemanticClusterer`, which discovers a variable number of
    density-based clusters, ``SemanticKSplit`` guarantees **exactly** ``k``
    non-empty partitions regardless of the natural structure of the data.

    The ``-1`` label is reserved exclusively for rows that were ``None``,
    ``NaN``, or empty after preprocessing — it never appears for a row that
    survived preprocessing.

    Parameters
    ----------
    embedding_model:
        Embedding model for text vectorisation.  Accepts the same types as
        :func:`~semantic_clusterer.embedding.adapters.normalize_embedding_model`
        (object with ``.embed`` / ``.encode`` / ``.embed_documents``, or a
        callable).  Pass ``None`` to use the built-in ONNX MiniLM-L6-v2
        model.
    k:
        **Required keyword argument.**  Number of clusters to produce.
        Must be an ``int`` (not ``bool``) in ``[2, N_Unique]``.
    config:
        Clustering configuration.  Accepts a :class:`ClustererConfig`
        instance, a dict of the same fields, or ``None`` for defaults.
    verbose:
        Enable verbose progress output.
    random_state:
        Integer seed in ``[0, 2**32 - 1]`` for all random components.
        When both this kwarg and ``config.random_state`` are supplied,
        this kwarg wins.

    Raises
    ------
    TypeError
        If ``k`` is not an ``int`` or is a ``bool``.
    ValueError
        If ``k < 2``.
    ValueError
        If ``random_state`` is not a valid integer seed.

    Examples
    --------
    Basic usage::

        from semantic_clusterer import SemanticKSplit

        texts = ["cats are furry", "dogs are loyal", "sky is blue",
                 "ocean is vast", "pizza is delicious", "pasta is great"]
        ks = SemanticKSplit(k=2)
        labels = ks.split_labels(texts)
        clusters = ks.split(texts)

    With custom embedder::

        ks = SemanticKSplit(embedding_model=my_model, k=5, random_state=0)
        labels, report = ks.split_with_report(texts)
    """

    def __init__(
        self,
        embedding_model: Optional[Any] = None,
        *,
        k: int,
        config: Optional[Union[ClustererConfig, SemanticKSplitConfig, Dict]] = None,
        verbose: bool = False,
        random_state: int = 42,
        quality: str = "balanced",
    ) -> None:
        # ------------------------------------------------------------------
        # 1. Validate k
        # ------------------------------------------------------------------
        if not isinstance(k, int) or isinstance(k, bool):
            raise TypeError("k must be an int")
        if k < 2:
            raise ValueError("k must be >= 2")

        # ------------------------------------------------------------------
        # 2. Validate random_state — same wording as SemanticClusterer
        # ------------------------------------------------------------------
        if not isinstance(random_state, int) or isinstance(random_state, bool):
            raise ValueError(
                f"random_state must be an int, got {type(random_state).__name__!r}"
            )
        if not (0 <= random_state <= 2**32 - 1):
            raise ValueError(
                f"random_state must be in [0, 2**32 - 1], got {random_state}"
            )

        # ------------------------------------------------------------------
        # 3. Reconcile config with kwarg precedence
        # ------------------------------------------------------------------
        if config is None:
            self.config = ClustererConfig(verbose=verbose, random_state=random_state)
        elif isinstance(config, dict):
            cfg = dict(config)
            cfg_verbose = cfg.pop("verbose", False) or verbose
            # kwarg wins: always set random_state from kwarg
            cfg["random_state"] = random_state
            _validate_config_dict(cfg, _PUBLIC_KSPLIT_FIELDS)
            self.config = ClustererConfig(**cfg, verbose=cfg_verbose)
        else:
            # ClustererConfig/SemanticKSplitConfig supplied: kwarg wins on conflict
            if random_state != 42 or config.random_state != random_state:
                self.config = replace(
                    config,
                    random_state=random_state,
                    verbose=config.verbose or verbose,
                )
            else:
                self.config = replace(config, verbose=config.verbose or verbose)

        # Store quality on config if not already set (quality kwarg wins)
        if quality != "balanced" or not hasattr(self.config, "quality"):
            # Store quality as an attribute on self for dispatch
            self._quality = quality
        else:
            self._quality = getattr(self.config, "quality", quality)

        # ------------------------------------------------------------------
        # 4. Store instance attributes
        # ------------------------------------------------------------------
        self._k: int = k
        self._random_state: int = int(self.config.random_state)
        self._raw_embedding_model = embedding_model
        self._embedder: Optional[EmbeddingModel] = None
        self._custom_embedder: bool = embedding_model is not None
        self.verbose: bool = self.config.verbose

        # ------------------------------------------------------------------
        # 5. Construct shared helpers (mirrors SemanticClusterer exactly)
        # ------------------------------------------------------------------
        self._preprocessor = TextPreprocessor(
            lowercase=True,
            remove_punctuation=True,
        )
        self._formatter = OutputFormatter(
            include_centroid=False,
            include_confidence=True,
            exclude_noise=False,
        )

        # Persistence state
        self._fitted_state = None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_embedder(self) -> EmbeddingModel:
        """Lazily initialise and return the embedding model."""
        if self._embedder is None:
            if self._raw_embedding_model is None:
                from semantic_clusterer.embedding.onnx_model import OnnxEmbedder
                self._embedder = OnnxEmbedder(
                    batch_size=self.config.batch_size,
                    normalize=self.config.normalize_embeddings,
                    verbose=self.verbose,
                )
            else:
                self._embedder = normalize_embedding_model(self._raw_embedding_model)
        return self._embedder

    def _log(self, message: str) -> None:
        """Print a verbose progress message if verbose mode is enabled."""
        if self.verbose:
            print(f"  [verbose] {message}")

    def _status(self, message: str, end: str = "\n") -> None:
        """Print a minimal pipeline status message (always visible)."""
        sys.stderr.write(f"  {message}{end}")
        sys.stderr.flush()

    def _embed_texts(
        self,
        texts: List[str],
        *,
        progress_callback=None,
    ) -> np.ndarray:
        """Embed a non-empty list of preprocessed texts and validate the result.

        ``progress_callback`` is an optional per-batch callback wired to the
        active pipeline progress bar.
        """
        if not texts:
            raise ValueError("Input texts cannot be empty")

        embedder = self._get_embedder()

        self._log(f"Embedding {len(texts)} texts...")
        try:
            embeddings = embedder.embed(
                texts,
                batch_size=self.config.batch_size,
                progress_callback=progress_callback,
            )
        except TypeError:
            embeddings = embedder.embed(texts, batch_size=self.config.batch_size)
            if progress_callback is not None:
                try:
                    progress_callback(len(texts))
                except Exception:
                    pass

        embeddings = validate_embeddings(embeddings, texts)

        # Normalise if configured (OnnxEmbedder already normalises internally)
        if self.config.normalize_embeddings and self._custom_embedder:
            embeddings = normalize_vectors(embeddings)

        return embeddings

    def _apply_reduction(
        self,
        emb: np.ndarray,
        tier: str,
        trace: _PipelineTrace,
    ) -> np.ndarray:
        """Apply PCA reduction when indicated by tier and config.

        Mirrors SemanticClusterer's reduction pass for medium and
        small-k>10 tiers.  For all other tiers the
        original embedding is returned unchanged.
        """
        reduction = self.config.get_reduction_for_strategy(tier)  # type: ignore[arg-type]
        if reduction is None:
            return emb

        N, D = emb.shape

        if reduction == "pca":
            n_components = self.config.get_reduction_components(D, N)
            n_components = min(n_components, N, D)
            if n_components >= D:
                return emb  # no benefit from reduction
            try:
                from sklearn.decomposition import PCA
                pca = PCA(
                    n_components=n_components,
                    random_state=self._random_state,
                )
                reduced = pca.fit_transform(emb).astype(np.float32)
                self._log(f"PCA: {D} -> {n_components} dims")
                return reduced
            except Exception as exc:
                trace.warn(f"pca-reduction-failed: {exc}")
                return emb

        # Potential UMAP path — mirror SemanticClusterer's posture
        #: if try_import_umap() returns None, fall back
        # to PCA-only and warn.
        if reduction == "umap":
            try:
                from semantic_clusterer.optional_deps import try_import_umap
                umap_cls = try_import_umap()
            except ImportError:
                umap_cls = None

            if umap_cls is None:
                trace.warn("umap-unavailable, used PCA-only fallback")
                warnings.warn(
                    "umap-learn is not installed; falling back to PCA reduction.",
                    UserWarning,
                    stacklevel=4,
                )
                return self._apply_reduction_pca(emb, tier, trace)

        return emb

    def _apply_reduction_pca(
        self,
        emb: np.ndarray,
        tier: str,
        trace: _PipelineTrace,
    ) -> np.ndarray:
        """Apply PCA reduction unconditionally (used as UMAP fallback)."""
        N, D = emb.shape
        n_components = self.config.get_reduction_components(D, N)
        n_components = min(n_components, N, D)
        if n_components >= D:
            return emb
        try:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=n_components, random_state=self._random_state)
            return pca.fit_transform(emb).astype(np.float32)
        except Exception as exc:
            trace.warn(f"pca-reduction-failed: {exc}")
            return emb

    def _dispatch_algorithm(
        self,
        algorithm_name: str,
        emb: np.ndarray,
        emb_reduced: np.ndarray,
        k: int,
        seed: int,
        trace: _PipelineTrace,
        n_restarts: int = 1,
    ) -> np.ndarray:
        """Dispatch to the appropriate k-algorithm and return int32 labels."""
        from semantic_clusterer.k_algorithms.restart import _run_restarts

        if algorithm_name == "agglomerative-cut-k":
            # Deterministic — skip restart
            return _agglomerative_cut_k(emb, k)

        if algorithm_name == "bisecting-kmeans":
            if n_restarts <= 1:
                return _bisecting_kmeans(emb, k, seed)
            return _run_restarts(_bisecting_kmeans, emb, k, seed, n_restarts)

        if algorithm_name == "spectral-cosine":
            # Spectral clustering updates trace.chosen_params["algorithm_used"]
            # to "constrained-kmeans" when it falls back.
            if n_restarts <= 1:
                return _spectral_cosine(emb, k, seed, trace)
            # For restarts, wrap in a lambda that ignores trace for inner calls
            def _spectral_fn(e, kk, s):
                return _spectral_cosine(e, kk, s, trace)
            return _run_restarts(_spectral_fn, emb, k, seed, n_restarts)

        if algorithm_name == "balanced-kmeans":
            # Medium/small-k>10: run on the (optionally PCA-reduced) embedding
            if n_restarts <= 1:
                return _balanced_kmeans(emb_reduced, k, seed)
            return _run_restarts(_balanced_kmeans, emb_reduced, k, seed, n_restarts)

        if algorithm_name == "minibatch-kmeans-assign":
            if n_restarts <= 1:
                return _minibatch_kmeans_assign(emb, k, seed)
            return _run_restarts(_minibatch_kmeans_assign, emb, k, seed, n_restarts)

        raise ValueError(f"Unknown algorithm: {algorithm_name!r}")

    # ------------------------------------------------------------------
    # Private core engine
    # ------------------------------------------------------------------

    def _run_split(
        self,
        texts: List[Optional[str]],
        trace: _PipelineTrace,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[int, int]]:
        """Core pipeline: preprocess → embed → split → project labels.

        Parameters
        ----------
        texts:
            Raw input list (may contain None/NaN/empty strings).
        trace:
            Active ``_PipelineTrace`` accumulator.

        Returns
        -------
        labels_full:
            ``int32`` array of shape ``(N_Input,)`` with values in
            ``{-1} ∪ [0, k-1]``.
        emb_unique:
            Float32 array of shape ``(N_Unique, D)`` — embeddings for the
            deduplicated rows (needed by ``split`` for the detailed formatter).
            Returns an empty ``(0, 0)`` array when no valid rows remain.
        orig_to_proc:
            Mapping from original input index to processed index (``-1`` for
            filtered rows).
        """
        k = self._k
        seed = self._random_state
        n_input = len(texts)

        # ----------------------------------------------------------------
        # Step 1: Empty list → empty array
        # ----------------------------------------------------------------
        if n_input == 0:
            return (
                np.array([], dtype=np.int32),
                np.empty((0, 0), dtype=np.float32),
                {},
            )

        # ----------------------------------------------------------------
        # Step 2: Validate element types
        # ----------------------------------------------------------------
        for t in texts:
            if not isinstance(t, str) and not (
                t is None or (isinstance(t, float) and math.isnan(t))
            ):
                raise TypeError(
                    f"texts contains non-string element: {type(t).__name__!r}"
                )

        # ----------------------------------------------------------------
        # Step 3: Record requested_k in trace
        # ----------------------------------------------------------------
        trace.chosen_params["requested_k"] = k

        # ----------------------------------------------------------------
        # Step 4: Preprocess with deduplication
        # ----------------------------------------------------------------
        processed, orig_to_proc, _ = self._preprocessor.preprocess(
            texts, deduplicate=True
        )

        labels_full = np.full(n_input, -1, dtype=np.int32)

        # ----------------------------------------------------------------
        # Step 5: No valid rows → return all -1
        # ----------------------------------------------------------------
        if len(processed) == 0:
            return labels_full, np.empty((0, 0), dtype=np.float32), orig_to_proc

        N_Unique = len(processed)

        # ----------------------------------------------------------------
        # Step 6: Embed
        # ----------------------------------------------------------------
        from semantic_clusterer.utils.progress import PipelineProgress

        with PipelineProgress(n_texts=n_input, verbose=self.verbose) as prog:

            # ── Phase 1: Embedding (live progress bar) ────────────────
            prog.start_embedding(n_unique=N_Unique)
            t_embed = time.perf_counter()
            emb = self._embed_texts(
                processed,
                progress_callback=prog.embedding_callback,
            )
            trace.time("embedding", time.perf_counter() - t_embed)
            prog.end_embedding()

            D = int(emb.shape[1])

            # ----------------------------------------------------------------
            # Step 7: k upper-bound check
            # ----------------------------------------------------------------
            if k > N_Unique:
                raise ValueError(
                    f"k cannot exceed the number of unique inputs: "
                    f"requested_k={k}, n_unique={N_Unique}"
                )

            # ----------------------------------------------------------------
            # Step 8: k == N_Unique warning
            # ----------------------------------------------------------------
            if k == N_Unique:
                warnings.warn(
                    "k equals the number of unique inputs; each cluster will "
                    "contain a single point",
                    UserWarning,
                    stacklevel=3,
                )

            # ----------------------------------------------------------------
            # Step 9: Dim-band and tier resolution
            # ----------------------------------------------------------------
            dim_band = resolve_dim_band(D)
            tier = self.config.get_strategy_for_size(N_Unique)

            trace.chosen_params["pipeline_tier"] = tier
            trace.chosen_params["dim_band"] = dim_band
            trace.chosen_params["embedding_dim"] = D

            self._log(
                f"N_Unique={N_Unique}, tier={tier}, dim_band={dim_band}, k={k}"
            )

            # ── Phase 2: Clustering (live progress bar) ────────────────
            prog.start_clustering()

            # ----------------------------------------------------------------
            # Step 10: Oversized gating
            # ----------------------------------------------------------------
            cap = self.config.max_samples
            if cap is not None and N_Unique > cap:
                allow_oversized = getattr(self.config, "allow_oversized_datasets", False)
                if not allow_oversized and cap == _OVERSIZED_LIMIT:
                    raise ValueError(
                        _OVERSIZED_ERROR_MSG_FMT.format(N=N_Unique)
                    )
                if cap == _OVERSIZED_LIMIT:
                    warn_msg = _OVERSIZED_WARN_MSG_FMT.format(N=N_Unique)
                else:
                    warn_msg = (
                        f"Dataset size {N_Unique} exceeds max_samples={cap}; "
                        f"subsampling to {cap} points."
                    )
                warnings.warn(warn_msg, UserWarning, stacklevel=3)
                trace.warn("oversized-subsampled")
                prog.clustering_phase("clustering", 0.85)
                t_cluster = time.perf_counter()
                labels_unique = _oversized_subsample_and_assign_k(
                    emb, k, seed, trace
                )
                trace.time("clustering", time.perf_counter() - t_cluster)
                prog.clustering_phase("scoring", 0.10)
                t_score = time.perf_counter()
                metrics = score_clustering(emb, labels_unique)
                trace.time("scoring", time.perf_counter() - t_score)
                trace.intrinsic_metrics.update(metrics)
                self._add_per_cluster_metrics(emb, labels_unique, k, trace)
                for orig_idx, proc_idx in orig_to_proc.items():
                    if proc_idx >= 0:
                        labels_full[orig_idx] = labels_unique[proc_idx]
                prog.end_clustering(n_clusters=k)
                return labels_full, emb, orig_to_proc

            # ----------------------------------------------------------------
            # Step 11: Identical-embeddings short-circuit
            # ----------------------------------------------------------------
            if _all_identical(emb):
                labels_unique = _round_robin_labels(N_Unique, k)
                trace.chosen_params["algorithm_used"] = "identical-embeddings-tiebreak"
                trace.warn("identical-embeddings-tiebreak")
                self._log("All embeddings identical; using round-robin assignment.")
                metrics = score_clustering(emb, labels_unique)
                trace.intrinsic_metrics.update(metrics)
                self._add_per_cluster_metrics(emb, labels_unique, k, trace)
                for orig_idx, proc_idx in orig_to_proc.items():
                    if proc_idx >= 0:
                        labels_full[orig_idx] = labels_unique[proc_idx]
                prog.end_clustering(n_clusters=k)
                return labels_full, emb, orig_to_proc

            # ----------------------------------------------------------------
            # Step 12: Algorithm selection and dispatch
            # ----------------------------------------------------------------
            algorithm_name = _select_k_algorithm(tier, k, self.config)
            trace.chosen_params["algorithm_used"] = algorithm_name
            self._log(f"Algorithm selected: {algorithm_name}")

            from semantic_clusterer.k_algorithms.quality_profile import resolve_quality
            quality_preset = getattr(self, "_quality", None) or getattr(self.config, "quality", "balanced")
            q_profile = resolve_quality(quality_preset, tier=tier)
            trace.chosen_params["quality"] = quality_preset
            trace.chosen_params["quality_n_restarts"] = q_profile.n_restarts

            prog.clustering_phase("reduction", 0.15)
            emb_reduced = self._apply_reduction(emb, tier, trace)

            # Canonical row sort for permutation invariance
            sort_perm = np.lexsort(emb.astype(np.float64).T[::-1])
            emb_sorted = emb[sort_perm]
            emb_reduced_sorted = (
                emb_sorted if emb_reduced is emb else emb_reduced[sort_perm]
            )

            t_cluster = time.perf_counter()
            labels_sorted = self._dispatch_algorithm(
                algorithm_name, emb_sorted, emb_reduced_sorted, k, seed, trace,
                n_restarts=q_profile.n_restarts,
            )
            trace.time("clustering", time.perf_counter() - t_cluster)
            prog.clustering_phase("clustering", 0.70)

            # ----------------------------------------------------------------
            # Step 13: Empty-cluster repair
            # ----------------------------------------------------------------
            labels_sorted = _repair_empty_clusters(
                emb_sorted, labels_sorted, k, seed, trace
            )

            labels_unique = np.empty(N_Unique, dtype=np.int32)
            labels_unique[sort_perm] = labels_sorted

            # ----------------------------------------------------------------
            # Step 14: Intrinsic scoring
            # ----------------------------------------------------------------
            prog.clustering_phase("scoring", 0.10)
            t_score = time.perf_counter()
            metrics = score_clustering(emb, labels_unique)
            trace.time("scoring", time.perf_counter() - t_score)
            trace.intrinsic_metrics.update(metrics)
            self._add_per_cluster_metrics(emb, labels_unique, k, trace)

            # ----------------------------------------------------------------
            # Step 15: Project labels through orig_to_proc
            # ----------------------------------------------------------------
            for orig_idx, proc_idx in orig_to_proc.items():
                if proc_idx >= 0:
                    labels_full[orig_idx] = labels_unique[proc_idx]

            prog.end_clustering(n_clusters=k)

        return labels_full, emb, orig_to_proc

    def _add_per_cluster_metrics(
        self,
        emb: np.ndarray,
        labels_unique: np.ndarray,
        k: int,
        trace: _PipelineTrace,
    ) -> None:
        """Append per_cluster_size, per_cluster_cohesion, silhouette, and
        davies_bouldin to trace metrics.

        """
        # per_cluster_size: list of length k, c-th entry = |cluster c|
        sizes = [int(np.sum(labels_unique == c)) for c in range(k)]
        trace.intrinsic_metrics["per_cluster_size"] = sizes

        # per_cluster_cohesion: mean cosine similarity within each cluster
        emb_norm = normalize_vectors(emb.astype(np.float64))
        cohesions = []
        for c in range(k):
            mask = labels_unique == c
            if mask.sum() == 0:
                cohesions.append(0.0)
                continue
            cluster_emb = emb_norm[mask]
            centroid = cluster_emb.mean(axis=0)
            norm_c = np.linalg.norm(centroid)
            if norm_c > 0:
                centroid = centroid / norm_c
            sims = np.clip(cluster_emb @ centroid, -1.0, 1.0)
            cohesions.append(float(np.mean(sims)))
        trace.intrinsic_metrics["per_cluster_cohesion"] = cohesions

        # silhouette and davies_bouldin
        # Computed via _selection_score which handles edge cases gracefully.
        sil, neg_dbi = _selection_score(emb, labels_unique)
        trace.intrinsic_metrics["silhouette"] = sil
        # _selection_score returns neg_dbi = -davies_bouldin; negate back.
        trace.intrinsic_metrics["davies_bouldin"] = (
            -neg_dbi if np.isfinite(neg_dbi) else float("inf")
        )

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def split_labels(self, texts: List[Optional[str]]) -> np.ndarray:
        """Partition ``texts`` and return a row-aligned int32 label array.

        Parameters
        ----------
        texts:
            List of text strings.  May contain ``None``, ``float("nan")``,
            or empty strings; those receive label ``-1``.

        Returns
        -------
        np.ndarray
            ``int32`` array of shape ``(len(texts),)``.  Values in
            ``[0, k-1]`` for valid rows; ``-1`` for filtered rows.

        Raises
        ------
        TypeError
            If any element is not a ``str``, ``None``, or ``NaN``.
        ValueError
            If ``k > N_Unique``.

        """
        if not texts:
            return np.array([], dtype=np.int32)
        trace = _PipelineTrace()
        labels_full, _, _ = self._run_split(texts, trace)
        return labels_full

    def split(
        self,
        texts: List[Optional[str]],
        return_format: Literal["simple", "detailed"] = "simple",
    ) -> Union[List[List[str]], List[ClusterResult]]:
        """Partition ``texts`` into ``k`` groups.

        Parameters
        ----------
        texts:
            List of text strings (may contain ``None`` / ``NaN``).
        return_format:
            - ``"simple"`` (default): returns ``List[List[str]]`` of length ``k``.
            - ``"detailed"``: returns a list of dicts with ``cluster_id``,
              ``representative``, ``items``, ``size``, and ``confidence``.

        Returns
        -------
        List[List[str]] or List[ClusterResult]
            Length ``k``.  Empty list ``[]`` when no valid rows remain.

        Raises
        ------
        ValueError
            If ``return_format`` is not ``"simple"`` or ``"detailed"``.
        TypeError
            If any element of ``texts`` is not a ``str``, ``None``, or ``NaN``.

        """
        if return_format not in ("simple", "detailed"):
            raise ValueError(
                "return_format must be either 'simple' or 'detailed'"
            )

        if not texts:
            return []

        trace = _PipelineTrace()
        labels_full, emb_unique, orig_to_proc = self._run_split(texts, trace)

        k = self._k

        # Build list of valid original indices (those that survived preprocessing)
        valid_original_indices = [
            i for i, proc_idx in orig_to_proc.items() if proc_idx >= 0
        ]

        if not valid_original_indices:
            return []  #

        if return_format == "simple":
            # Bucket original texts by label, preserving original input order
            buckets: List[List[str]] = [[] for _ in range(k)]
            for i in valid_original_indices:
                label = int(labels_full[i])
                if 0 <= label < k:
                    buckets[label].append(texts[i])  # type: ignore[arg-type]
            return buckets

        # --- detailed ---
        # Build (text, label, embedding) triples preserving original input order
        output_texts = [texts[i] for i in valid_original_indices]  # type: ignore[index]
        output_labels = labels_full[valid_original_indices]

        # Build per-valid-index embedding array for the formatter
        if emb_unique.shape[0] > 0:
            output_embeddings = np.zeros(
                (len(valid_original_indices), emb_unique.shape[1]),
                dtype=np.float32,
            )
            for out_idx, orig_idx in enumerate(valid_original_indices):
                proc_idx = orig_to_proc[orig_idx]
                if 0 <= proc_idx < emb_unique.shape[0]:
                    output_embeddings[out_idx] = emb_unique[proc_idx]
        else:
            output_embeddings = np.zeros((len(valid_original_indices), 1), dtype=np.float32)

        detailed = self._formatter.format_detailed(
            output_texts, output_embeddings, output_labels
        )
        return detailed

    def split_with_report(
        self,
        texts: List[Optional[str]],
    ) -> Tuple[np.ndarray, ClusteringReport]:
        """Partition ``texts`` and return both labels and a structured run report.

        Parameters
        ----------
        texts:
            List of text strings (may contain ``None`` / ``NaN``).

        Returns
        -------
        (labels, report):
            ``labels`` — ``int32`` ndarray of shape ``(len(texts),)``.
            ``report`` — :class:`ClusteringReport` with all fields populated.

        """
        trace = _PipelineTrace()

        if not texts:
            labels_full = np.array([], dtype=np.int32)
        else:
            labels_full, _, _ = self._run_split(texts, trace)

        k = self._k
        n_total = len(texts)
        n_noise = int(np.sum(labels_full == -1))
        n_clustered = n_total - n_noise

        report = ClusteringReport(
            n_input_texts=n_total,
            n_clustered=n_clustered,
            n_noise=n_noise,
            n_clusters=k,  #
            pipeline_tier=trace.chosen_params.get("pipeline_tier", "small"),
            embedding_dim=trace.chosen_params.get("embedding_dim", 0),
            dim_band=trace.chosen_params.get("dim_band", "low"),
            dataset_profile={},  # No dataset profile needed for fixed-k routing
            chosen_params=trace.chosen_params,
            intrinsic_metrics=trace.intrinsic_metrics,
            phase_timings=trace.phase_timings,
            warnings=list(trace.warnings),
            confidence_level=trace.confidence_level,
            random_state=self._random_state,
            library_version=_LIB_VERSION,
        )
        return labels_full, report

    def embed(self, texts: List[Optional[str]]) -> np.ndarray:
        """Generate embeddings for ``texts`` without clustering.

        Mirrors :meth:`SemanticClusterer.embed` behaviour exactly
.

        Parameters
        ----------
        texts:
            List of text strings (may contain ``None`` / ``NaN``).

        Returns
        -------
        np.ndarray
            Float32 array of shape ``(N_Valid, D)``, or ``(0, 0)`` when
            no valid rows remain.

        """
        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        processed = self._preprocessor.preprocess_simple(texts)
        valid_texts = [t for t in processed if t is not None]
        if not valid_texts:
            return np.empty((0, 0), dtype=np.float32)
        return self._embed_texts(valid_texts)

    def cluster(
        self,
        texts: List[Optional[str]],
        return_format: Literal["simple", "detailed"] = "simple",
    ) -> Union[List[List[str]], List]:
        """Alias for :meth:`split` — provides a consistent API with SemanticClusterer.

        Parameters
        ----------
        texts:
            List of text strings.
        return_format:
            ``"simple"`` (default) or ``"detailed"``.

        Returns
        -------
        Same as :meth:`split`.
        """
        return self.split(texts, return_format=return_format)

    # ------------------------------------------------------------------
    # Production API: fit / predict / save / load
    # ------------------------------------------------------------------

    def fit(self, texts: List[str]) -> "SemanticKSplit":
        """Fit on a corpus and store centroids for predict().

        Args:
            texts: Training corpus.

        Returns:
            ``self`` to support fluent chaining.
        """
        from semantic_clusterer.persistence import (
            FittedState,
            ClusterStats,
            save_state,
            load_state,
            assign_to_centroids,
        )
        import math as _math
        from semantic_clusterer.dim_bands import resolve_dim_band as _resolve_dim_band

        labels = self.split_labels(texts)
        self._fitted_state = self._build_fitted_state(texts, labels)
        return self

    def predict(
        self,
        texts: List[str],
        *,
        outlier_threshold=None,
    ) -> np.ndarray:
        """Assign new texts to existing clusters by nearest centroid.

        Parameters
        ----------
        texts:
            New texts to classify.
        outlier_threshold:
            Float threshold, ``None`` (no OOD detection), or ``"auto"``
            to use the calibrated training threshold.

        Returns
        -------
        ``int32`` array aligned with ``texts``.
        """
        from semantic_clusterer.persistence import assign_to_centroids

        self._require_fitted()
        state = self._fitted_state

        # Resolve to a concrete float, None, or adaptive dict before doing any work.
        threshold = None
        adaptive_thresholds = None

        # Resolve to a concrete float, None, or adaptive dict before doing any work.
        threshold = None
        adaptive_thresholds = None

        if outlier_threshold in ("auto", "adaptive"):
            if state.cluster_cohesion:
                # Per-cluster adaptive thresholds — size-aware + tightness + confusion
                max_inter = getattr(state, "max_inter_centroid_sim", 0.0)

                inter_sims = getattr(state, "inter_centroid_sims", None)
                confused_counts = {}
                if inter_sims is not None and inter_sims.shape[0] > 1:
                    for i, stat in enumerate(state.cluster_cohesion):
                        if i < inter_sims.shape[0]:
                            row = inter_sims[i].copy()
                            row[i] = -1.0
                            confused_counts[stat.cluster_id] = int(np.sum(row > 0.7))

                adaptive_thresholds = {}

                # Detect small-data regime: centroids from very few
                # training members overfit and produce artificially
                # tight thresholds that reject valid test data.
                total_train = sum(s.size for s in state.cluster_cohesion)
                max_csize = max((s.size for s in state.cluster_cohesion), default=0)
                small_data = total_train < 200 or max_csize < 30

                for stat in state.cluster_cohesion:
                    if stat.size >= 50:
                        base = stat.p10_sim
                    elif stat.size >= 10:
                        blend = (stat.size - 10) / 40.0
                        base = getattr(stat, 'p25_sim', stat.p10_sim) * (1 - blend) + stat.p10_sim * blend
                    else:
                        base = getattr(stat, 'p25_sim', stat.p10_sim)

                    tightness_bonus = max(0.0, (stat.mean_sim - 0.5)) * 0.1
                    n_confused = confused_counts.get(stat.cluster_id, 0)
                    confusion_relax = 0.03 * n_confused

                    # Dynamic pullback: reduce for small data where centroids
                    # overfit to few training points and produce artificially
                    # tight similarity distributions.
                    if small_data:
                        pullback = 0.45 + 0.10 * min(1.0, stat.size / 50.0)
                        tightness_bonus = min(tightness_bonus, 0.02)
                    else:
                        pullback = 0.70 + 0.15 * min(1.0, stat.size / 50.0)

                    threshold_c = max(
                        0.05,
                        base * pullback - confusion_relax + tightness_bonus
                        - 0.02 * max(0.0, max_inter)
                    )

                    # Safety ceiling for small data: cap at the global
                    # auto_outlier_threshold to prevent centroid overfitting
                    # from producing per-cluster thresholds higher than what
                    # the global training distribution justifies.
                    if small_data and state.auto_outlier_threshold is not None:
                        threshold_c = min(threshold_c, state.auto_outlier_threshold)

                    adaptive_thresholds[stat.cluster_id] = threshold_c
            else:
                threshold = state.auto_outlier_threshold
        elif outlier_threshold == "global":
            threshold = state.auto_outlier_threshold
        elif outlier_threshold is None:
            threshold = None
        else:
            threshold = float(outlier_threshold)

        if not texts:
            return np.array([], dtype=np.int32)

        processed, orig_to_proc, _ = self._preprocessor.preprocess(
            texts, deduplicate=True
        )

        labels = np.full(len(texts), -1, dtype=np.int32)
        if not processed:
            return labels

        emb = self._embed_texts(processed)

        if state.has_reducer and state.reducer is not None:
            try:
                emb = state.reducer.transform(emb).astype(np.float32)
            except Exception as exc:
                raise RuntimeError(
                    f"Saved reducer failed to transform new embeddings: {exc}"
                ) from exc

        emb_norm = normalize_vectors(emb)

        proc_labels = assign_to_centroids(
            emb_norm,
            state.centroids,
            state.cluster_ids,
            outlier_threshold=threshold,
            adaptive_thresholds=adaptive_thresholds,
            keywords=state.keywords if state.keywords else None,
        )

        for orig_idx, proc_idx in orig_to_proc.items():
            if proc_idx >= 0:
                labels[orig_idx] = proc_labels[proc_idx]

        return labels

    def fit_predict(self, texts: List[str]) -> np.ndarray:
        """Fit on ``texts`` and return the training labels."""
        labels = self.split_labels(texts)
        self._fitted_state = self._build_fitted_state(texts, labels)
        return labels

    def save(self, path: str) -> None:
        """Persist the fitted model to a directory at ``path``.

        Raises:
            RuntimeError: If the model has not been fitted.
        """
        from semantic_clusterer.persistence import save_state
        self._require_fitted()
        save_state(self._fitted_state, path)

    @classmethod
    def load(
        cls,
        path: str,
        *,
        embedding_model=None,
        verbose: bool = False,
    ) -> "SemanticKSplit":
        """Load a previously-saved model from ``path``.

        Args:
            path: Directory written by ``save()``.
            embedding_model: Embedder to use for ``predict()`` calls.
            verbose: Verbose logging.

        Returns:
            A reconstructed ``SemanticKSplit`` instance.
        """
        from semantic_clusterer.persistence import load_state
        state = load_state(path)

        # Determine k from the loaded state
        k = int(state.n_clusters)
        if k < 2:
            k = 2

        # Rebuild config from snapshot
        from semantic_clusterer.config import ClustererConfig
        config_snapshot = dict(state.config_snapshot)
        from semantic_clusterer.config import _PUBLIC_CONFIG_FIELDS
        config_snapshot = {
            k_: v for k_, v in config_snapshot.items() if k_ in _PUBLIC_CONFIG_FIELDS
        }
        try:
            cfg = ClustererConfig(**config_snapshot)
        except (TypeError, ValueError):
            cfg = ClustererConfig()

        instance = cls(
            embedding_model=embedding_model,
            k=k,
            config=cfg,
            verbose=verbose,
            random_state=cfg.random_state,
        )
        instance._fitted_state = state
        return instance

    # ------------------------------------------------------------------
    # Topic accessors (require a fitted model)
    # ------------------------------------------------------------------

    def get_topic_keywords(self, cluster_id=None):
        """Return c-TF-IDF keywords for one or all clusters."""
        self._require_fitted()
        all_kw = {
            int(k): [(str(w), float(s)) for w, s in v]
            for k, v in (self._fitted_state.keywords or {}).items()
        }
        if cluster_id is None:
            return all_kw
        if cluster_id not in all_kw:
            raise KeyError(f"Unknown cluster_id: {cluster_id}")
        return all_kw[cluster_id]

    def get_topic_labels(self):
        """Return ``cluster_id -> human-readable label`` mapping."""
        self._require_fitted()
        return {int(k): str(v) for k, v in (self._fitted_state.topic_labels or {}).items()}

    @property
    def outlier_threshold(self):
        """The auto-calibrated OOD threshold computed during ``fit()``."""
        if self._fitted_state is None:
            return None
        return self._fitted_state.auto_outlier_threshold

    @property
    def cluster_stats(self):
        """Per-cluster cohesion statistics measured on the training set."""
        if self._fitted_state is None or not self._fitted_state.cluster_cohesion:
            return None
        return [
            {
                "cluster_id": cs.cluster_id,
                "size":       cs.size,
                "min_sim":    round(cs.min_sim,    4),
                "mean_sim":   round(cs.mean_sim,   4),
                "median_sim": round(cs.median_sim,  4),
                "std_sim":    round(cs.std_sim,     4),
                "p10_sim":    round(cs.p10_sim,    4),
                "p25_sim":    round(cs.p25_sim,    4),
                "radius_95":  round(cs.radius_95,  4),
            }
            for cs in self._fitted_state.cluster_cohesion
        ]

    @property
    def is_fitted(self) -> bool:
        """``True`` once ``fit()`` or ``load()`` has populated the state."""
        return self._fitted_state is not None

    # ------------------------------------------------------------------
    # Private helpers for fit/predict
    # ------------------------------------------------------------------

    def _require_fitted(self) -> None:
        if self._fitted_state is None:
            raise RuntimeError(
                "SemanticKSplit has not been fitted. Call .fit(texts) "
                "or .load(path) first."
            )

    def _build_fitted_state(self, texts, labels):
        """Compute centroids, calibration stats, keywords, and topic labels."""
        import math as _math
        from semantic_clusterer.dim_bands import resolve_dim_band as _resolve_dim_band
        from semantic_clusterer.persistence import FittedState, ClusterStats

        def _is_real(t):
            if t is None:
                return False
            if isinstance(t, float) and _math.isnan(t):
                return False
            if not isinstance(t, str):
                return False
            return bool(t.strip())

        valid_mask = np.array([_is_real(t) for t in texts], dtype=bool)
        valid_texts = [t for t, ok in zip(texts, valid_mask) if ok]
        valid_labels = labels[valid_mask] if valid_mask.any() else labels

        if valid_texts:
            embeddings = self._embed_texts(valid_texts)
        else:
            embeddings = np.empty((0, 0), dtype=np.float32)

        embedding_dim = int(embeddings.shape[1]) if embeddings.size else 0
        dim_band = _resolve_dim_band(embedding_dim) if embedding_dim else "low"

        if embeddings.size and valid_labels.size:
            emb_norm = normalize_vectors(embeddings)
            cluster_ids_int = sorted(int(c) for c in np.unique(valid_labels) if c >= 0)
        else:
            emb_norm = embeddings
            cluster_ids_int = []

        K = len(cluster_ids_int)

        if K and emb_norm.size:
            centroids = np.zeros((K, emb_norm.shape[1]), dtype=np.float32)
            for i, cid in enumerate(cluster_ids_int):
                mask = valid_labels == cid
                raw = emb_norm[mask].mean(axis=0, keepdims=True)
                centroids[i] = normalize_vectors(raw)[0]
        else:
            centroids = np.empty((0, embedding_dim), dtype=np.float32)

        # Per-cluster cohesion stats
        cluster_cohesion = []
        all_member_sims = []

        if K and emb_norm.size:
            for i, cid in enumerate(cluster_ids_int):
                mask = valid_labels == cid
                members = emb_norm[mask]
                sims = (members @ centroids[i]).clip(-1.0, 1.0)

                min_s    = float(sims.min())
                mean_s   = float(sims.mean())
                median_s = float(np.median(sims))
                std_s    = float(np.std(sims))
                p10_s    = float(np.percentile(sims, 10))
                p25_s    = float(np.percentile(sims, 25))
                radius_95_s = float(np.percentile(1.0 - sims, 95))

                cluster_cohesion.append(
                    ClusterStats(
                        cluster_id=cid,
                        size=int(mask.sum()),
                        min_sim=min_s,
                        mean_sim=mean_s,
                        p10_sim=p10_s,
                        median_sim=median_s,
                        std_sim=std_s,
                        p25_sim=p25_s,
                        radius_95=radius_95_s,
                    )
                )
                all_member_sims.extend(sims.tolist())

        # Auto-calibrate outlier threshold
        if len(all_member_sims) >= 2:
            global_p5 = float(np.percentile(all_member_sims, 5))
            if K > 1:
                inter = centroids @ centroids.T
                np.fill_diagonal(inter, -1.0)
                max_inter = float(inter.max())
                inter_centroid_sims = centroids @ centroids.T
            else:
                max_inter = 0.0
                inter_centroid_sims = None
            auto_threshold = max(
                0.05,
                global_p5 * 0.90 - 0.05 * max(0.0, max_inter),
            )
        else:
            max_inter = 0.0
            auto_threshold = None
            inter_centroid_sims = None

        # Keywords + topic labels
        keywords = {}
        topic_labels = {}
        try:
            from semantic_clusterer.representation.keywords import (
                extract_cluster_keywords,
                generate_topic_label,
            )
            if self.config.extract_keywords and valid_texts and valid_labels.size:
                kw_map = extract_cluster_keywords(
                    valid_texts,
                    valid_labels,
                    top_n=self.config.keywords_top_n,
                )
                for cid, pairs in kw_map.items():
                    keywords[int(cid)] = [[str(w), float(s)] for w, s in pairs]
                    topic_labels[int(cid)] = generate_topic_label(pairs)
        except Exception:
            pass

        # Config snapshot
        from semantic_clusterer.config import _PUBLIC_CONFIG_FIELDS
        cfg_snapshot = {}
        for name in _PUBLIC_CONFIG_FIELDS:
            if hasattr(self.config, name):
                value = getattr(self.config, name)
                if isinstance(value, np.generic):
                    value = value.item()
                cfg_snapshot[name] = value

        return FittedState(
            centroids=centroids,
            cluster_ids=np.asarray(cluster_ids_int, dtype=np.int32),
            train_labels=labels.astype(np.int32),
            embedding_dim=embedding_dim,
            dim_band=dim_band,
            mode="fixed_k",
            n_clusters=K,
            auto_outlier_threshold=auto_threshold,
            cluster_cohesion=cluster_cohesion,
            max_inter_centroid_sim=max_inter if K > 1 else 0.0,
            inter_centroid_sims=inter_centroid_sims,
            keywords=keywords,
            topic_labels=topic_labels,
            config_snapshot=cfg_snapshot,
            library_version=_LIB_VERSION,
            has_reducer=False,
            reducer=None,
        )
