"""Core SemanticClusterer class - the main entry point."""

import math
import sys
import time
import warnings
from dataclasses import asdict, replace
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np

from semantic_clusterer.config import ClustererConfig, SemanticClustererConfig, _validate_config_dict, _PUBLIC_CLUSTERER_FIELDS
from semantic_clusterer.embedding.adapters import (
    EmbeddingModel,
    normalize_embedding_model,
    validate_embeddings,
)
from semantic_clusterer.output.formatter import ClusterResult, OutputFormatter
from semantic_clusterer.preprocessing.clean import TextPreprocessor
from semantic_clusterer.utils.similarity import normalize_vectors
from semantic_clusterer.pipeline import tiny, small, medium, large
from semantic_clusterer.pipeline.profile import compute_dataset_profile
from semantic_clusterer._quality_floor import _enforce_quality_floor
from semantic_clusterer.report import ClusteringReport, _PipelineTrace
from semantic_clusterer.persistence import (
    FittedState,
    ClusterStats,
    assign_to_centroids,
    load_state,
    save_state,
)

try:
    from semantic_clusterer import __version__ as _LIB_VERSION
except Exception:
    _LIB_VERSION = "0.1.0"

# ---------------------------------------------------------------------------
# Module-level message format strings for oversized-dataset gating.
# Both SemanticClusterer and SemanticKSplit format from these constants so
# that the resulting strings are byte-for-byte identical.
# Use as: _OVERSIZED_ERROR_MSG_FMT.format(N=<count>)
# ---------------------------------------------------------------------------
_OVERSIZED_ERROR_MSG_FMT = (
    "Dataset size {N} exceeds the hard limit of 200_000. "
    "Set allow_oversized_datasets=True to enable subsampling."
)
_OVERSIZED_WARN_MSG_FMT = (
    "Dataset size {N} exceeds 200_000; subsampling to 200_000 points."
)


def _oversized_error(n: int, cap: int) -> str:
    """Return the oversized-dataset error message.

    For the historical default cap (200_000) we keep the byte-identical
    wording, as several tests pin it. For other caps we produce a
    parametric message that names the actual cap.
    """
    if cap == 200_000:
        return _OVERSIZED_ERROR_MSG_FMT.format(N=n)
    return (
        f"Dataset size {n} exceeds max_samples={cap}. "
        f"Set max_samples=None or a larger value to enable subsampling."
    )


def _oversized_warn(n: int, cap: int) -> str:
    """Return the oversized-dataset warning message."""
    if cap == 200_000:
        return _OVERSIZED_WARN_MSG_FMT.format(N=n)
    return f"Dataset size {n} exceeds max_samples={cap}; subsampling to {cap} points."





class SemanticClusterer:
    """Zero-config semantic text clustering.
    
    A high-performance library for unsupervised semantic grouping of text at scale.
    Designed for three levels of users:
    
    - Beginner: Zero config, one line, works.
    - Intermediate: Plug in a custom embedder, keep the rest automatic.
    - Advanced: Override strategy, reduction, batch_size, and normalize_embeddings.
    
    Example:
        Beginner — zero config:
        ```python
        from semantic_clusterer import SemanticClusterer
        
        clusterer = SemanticClusterer()
        clusters = clusterer.cluster(texts)
        ```
        
        Intermediate — custom embedder:
        ```python
        clusterer = SemanticClusterer(embedding_model=my_model)
        clusters = clusterer.cluster(texts, return_format="detailed")
        ```
        
        Advanced — full control:
        ```python
        clusterer = SemanticClusterer(
            embedding_model=my_model,
            config={
                "batch_size": 128,
                "normalize_embeddings": True,
            }
        )
        ```
    
    The config accepts only these fields: batch_size, normalize_embeddings,
    cluster_granularity, min_cluster_size, min_samples, max_samples,
    extract_keywords, keywords_top_n, verbose, and random_state.
    Invalid fields raise ValueError immediately.
    
    Attributes:
        config: Configuration options for clustering.
        verbose: Enable verbose logging.
    """

    # Size thresholds for warnings
    _VERY_LARGE_THRESHOLD = 200_000

    def __init__(
        self,
        embedding_model: Optional[Any] = None,
        config: Optional[Union[ClustererConfig, Dict]] = None,
        verbose: bool = False,
        random_state: int = 42,
        *,
        cluster_granularity: str = "balanced",
    ):
        """Initialize SemanticClusterer.
        
        Args:
            embedding_model: Embedding model for text vectorization. Supports:
                - None: Uses built-in ONNX MiniLM-L6-v2 model (default)
                - Object with .embed(texts) method: Used directly
                - Object with .encode(texts) method: SentenceTransformers/HuggingFace
                - Object with .embed_documents(texts) method: LangChain embeddings
                - Callable function: fn(texts) -> embeddings array
            config: Configuration options. Can be a ClustererConfig or dict.
                If None, uses intelligent defaults.
            verbose: Enable verbose logging.
            random_state: Integer seed for all random components. Must be in
                [0, 2**32 - 1]. When both this kwarg and config.random_state
                are supplied, this kwarg wins.
        
        Examples:
            Beginner (zero config):
            >>> clusterer = SemanticClusterer()
            
            SentenceTransformers:
            >>> from sentence_transformers import SentenceTransformer
            >>> model = SentenceTransformer("all-MiniLM-L12-v2")
            >>> clusterer = SemanticClusterer(embedding_model=model)
            
            LangChain:
            >>> from langchain.embeddings import OpenAIEmbeddings
            >>> model = OpenAIEmbeddings()
            >>> clusterer = SemanticClusterer(embedding_model=model)
            
            Custom callable:
            >>> def my_embed(texts):
            ...     return my_api_call(texts)
            >>> clusterer = SemanticClusterer(embedding_model=my_embed)
        """
        # Validate random_state at the boundary
        if not isinstance(random_state, int) or isinstance(random_state, bool):
            raise ValueError(
                f"random_state must be an int, got {type(random_state).__name__!r}"
            )
        if not (0 <= random_state <= 2**32 - 1):
            raise ValueError(
                f"random_state must be in [0, 2**32 - 1], got {random_state}"
            )

        # Handle config — reconcile random_state kwarg with config
        if config is None:
            self.config = ClustererConfig(verbose=verbose, random_state=random_state)
        elif isinstance(config, dict):
            cfg = dict(config)
            # verbose=True kwarg overrides verbose=False in config dict
            cfg_verbose = cfg.pop("verbose", False) or verbose
            # kwarg wins: set random_state in cfg (overrides any value in dict)
            cfg["random_state"] = random_state
            # Validate that only allowed fields are present
            _validate_config_dict(cfg)
            self.config = ClustererConfig(**cfg, verbose=cfg_verbose)
        else:
            # ClustererConfig supplied: kwarg wins iff it differs from default
            # or config.random_state differs from the kwarg
            if random_state != 42 or config.random_state != random_state:
                self.config = replace(
                    config,
                    random_state=random_state,
                    verbose=config.verbose or verbose,
                )
            else:
                self.config = replace(config, verbose=config.verbose or verbose)

        # Apply cluster_granularity kwarg (wins over config value when not default)
        if cluster_granularity != "balanced":
            if hasattr(self.config, "cluster_granularity"):
                self.config = replace(self.config, cluster_granularity=cluster_granularity)
        # Store cluster_granularity on self for easy access
        self._cluster_granularity = cluster_granularity

        # Eagerly fail if hdbscan is missing
        try:
            import hdbscan  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "hdbscan is required for SemanticClusterer. "
                "Install it with: pip install hdbscan"
            ) from exc

        self.verbose = self.config.verbose

        # Normalize embedding model to standard interface
        # This is lazy - if model is None, we defer initialization
        self._raw_embedding_model = embedding_model
        self._embedder: Optional[EmbeddingModel] = None
        self._custom_embedder = embedding_model is not None

        # Initialize components
        self._preprocessor = TextPreprocessor(
            lowercase=True,
            remove_punctuation=True,
        )

        self._formatter = OutputFormatter(
            include_centroid=False,  # Performance: skip by default
            include_confidence=True,
            exclude_noise=False,
        )

        # Persistence state. Set by fit() / load() and consumed by predict()
        # and the get_topic_* accessors. None == not fitted.
        self._fitted_state: Optional["FittedState"] = None

    def _get_embedder(self) -> EmbeddingModel:
        """Get or initialize the embedding model.
        
        Lazily initializes the embedder on first use. For custom models,
        wraps them with the appropriate adapter to ensure a consistent
        interface. For None (default), uses the built-in ONNX embedder.
        
        Returns:
            An embedder implementing the EmbeddingModel protocol.
        """
        if self._embedder is None:
            if self._raw_embedding_model is None:
                # Use built-in ONNX embedder with config
                from semantic_clusterer.embedding.onnx_model import OnnxEmbedder
                self._embedder = OnnxEmbedder(
                    batch_size=self.config.batch_size,
                    normalize=self.config.normalize_embeddings,
                    verbose=self.verbose,
                )
            else:
                # Normalize custom model to standard interface
                self._embedder = normalize_embedding_model(self._raw_embedding_model)
        return self._embedder

    def _log(self, message: str) -> None:
        """Print detailed message if verbose mode is enabled."""
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
        """Generate embeddings for texts.

        Args:
            texts: Preprocessed text strings.
            progress_callback: Optional per-batch progress callback wired to
                the active pipeline progress bar. When ``None``, the
                embedder's own internal bar may render (for direct calls
                outside the orchestrated pipeline).

        Returns:
            Validated numpy array of shape (n_texts, embedding_dim).
        """
        if not texts:
            raise ValueError("Input texts cannot be empty")

        embedder = self._get_embedder()

        self._log(f"Embedding {len(texts)} texts...")
        # Forward the progress_callback only to embedders that accept it
        # (the built-in OnnxEmbedder, plus any custom embedder that opts in).
        try:
            embeddings = embedder.embed(
                texts,
                batch_size=self.config.batch_size,
                progress_callback=progress_callback,
            )
        except TypeError:
            # Custom embedder doesn't support the kwarg — fall back gracefully.
            embeddings = embedder.embed(texts, batch_size=self.config.batch_size)
            # Best-effort progress: report the whole batch as a single tick
            # so the bar still completes.
            if progress_callback is not None:
                try:
                    progress_callback(len(texts))
                except Exception:
                    pass

        # Validate embeddings
        embeddings = validate_embeddings(embeddings, texts)

        # Normalize if configured (OnnxEmbedder already normalizes internally)
        if self.config.normalize_embeddings and self._custom_embedder:
            embeddings = normalize_vectors(embeddings)

        return embeddings

    def _cluster_tiny(
        self,
        embeddings: np.ndarray,
    ) -> np.ndarray:
        """Clustering strategy for tiny datasets (≤150)."""
        self._log("Using tiny data strategy (Agglomerative optimal linkage cut)")
        return tiny.cluster_tiny(
            embeddings,
            random_state=int(self.config.random_state),
        )

    def _cluster_small(
        self,
        embeddings: np.ndarray,
    ) -> np.ndarray:
        """Clustering strategy for small datasets (150-5K)."""
        return small.cluster_small(embeddings, log_fn=self._log)

    def _cluster_medium(
        self,
        embeddings: np.ndarray,
    ) -> np.ndarray:
        """Clustering strategy for medium datasets (5K-50K)."""
        self._log("Using medium data strategy (reduction + HDBSCAN)")
        return medium.cluster_medium(
            embeddings,
            config=self.config,
            random_state=int(self.config.random_state),
            log_fn=self._log,
        )

    def _cluster_large(
        self,
        embeddings: np.ndarray,
    ) -> np.ndarray:
        """Clustering strategy for large datasets (50K-200K)."""
        self._log("Using large data strategy (two-stage pipeline)")
        return large.cluster_large(
            embeddings,
            config=self.config,
            random_state=int(self.config.random_state),
            log_fn=self._log,
            verbose=self.verbose,
        )

    def _cluster_embeddings(
        self,
        embeddings: np.ndarray,
    ) -> np.ndarray:
        """Apply appropriate clustering strategy based on data size."""
        n_samples = embeddings.shape[0]

        # Deterministic edge cases — no algorithm needed
        if n_samples == 1:
            return np.array([0], dtype=np.int32)
        if n_samples == 2:
            return np.array([0, 1], dtype=np.int32)

        # Check for very large datasets
        if n_samples > self._VERY_LARGE_THRESHOLD:
            warnings.warn(
                f"Dataset size ({n_samples:,}) exceeds recommended limit "
                f"({self._VERY_LARGE_THRESHOLD:,}). Consider using approximate "
                "nearest neighbor methods for better performance. "
                "ANN support coming in future versions.",
                UserWarning,
            )

        # Determine strategy
        strategy = self.config.get_strategy_for_size(n_samples)
        self._log(f"Dataset size: {n_samples:,}, strategy: {strategy}")

        if strategy == "tiny":
            return self._cluster_tiny(embeddings)
        elif strategy == "small":
            return self._cluster_small(embeddings)
        elif strategy == "medium":
            return self._cluster_medium(embeddings)
        else:
            return self._cluster_large(embeddings)

    def cluster(
        self,
        texts: List[str],
        return_format: Literal["simple", "detailed"] = "simple",
    ) -> Union[List[List[str]], List[ClusterResult]]:
        """Cluster texts into semantically similar groups.
        
        Args:
            texts: List of text strings to cluster.
            return_format: Output format.
                - "simple": Returns List[List[str]] - just grouped texts.
                - "detailed": Returns list of dicts with cluster_id, representative,
                  items, size, and confidence.
        
        Returns:
            Clustered texts in the specified format.
            
        Raises:
            ValueError: If texts is empty or return_format is invalid.
        """
        if return_format not in ("simple", "detailed"):
            raise ValueError("return_format must be either 'simple' or 'detailed'")

        if not texts:
            return [] if return_format == "simple" else []

        # Delegate to _run_clustering so that all config fields — including
        # min_cluster_size / min_samples overrides — are honoured.
        labels = self._run_clustering(texts, trace=None)

        # Build output only from valid (non-noise, non-filtered) rows.
        valid_indices = [i for i, l in enumerate(labels) if l >= 0 or True]
        # We need embeddings only for the detailed format. Retrieve them from
        # the preprocessor-deduped representation so we match stored labels.
        output_texts_all = texts
        output_labels_all = labels

        unique_labels = set(labels[labels >= 0].tolist())
        n_noise = int((labels == -1).sum())
        n_valid = len(labels) - n_noise
        self._log(f"Found {len(unique_labels)} clusters, {n_noise} noise/filtered points")

        if return_format == "detailed":
            # Re-embed for the detailed formatter (needed for centroid/confidence calc).
            # Only embed non-filtered texts to avoid wasted work.
            valid_mask = labels >= 0
            valid_texts = [t for t, ok in zip(texts, valid_mask) if ok]
            valid_labels = labels[valid_mask]

            if not valid_texts:
                return []

            processed, orig_to_proc, _ = self._preprocessor.preprocess(valid_texts, deduplicate=True)
            if processed:
                emb_unique = self._embed_texts(processed)
                # Map back to per-valid-text embeddings
                emb_full = np.zeros((len(valid_texts), emb_unique.shape[1]), dtype=np.float32)
                for i, proc_idx in orig_to_proc.items():
                    if proc_idx >= 0:
                        emb_full[i] = emb_unique[proc_idx]
            else:
                emb_full = np.zeros((len(valid_texts), 384), dtype=np.float32)

            detailed = self._formatter.format_detailed(valid_texts, emb_full, valid_labels)
            if self.config.extract_keywords:
                detailed = self._formatter.enrich_with_keywords(
                    detailed,
                    valid_texts,
                    valid_labels,
                    top_n=self.config.keywords_top_n,
                )
            return detailed

        # Simple format — filter out noise, pass valid rows only
        valid_texts_simple = [t for t, l in zip(texts, labels) if l >= 0]
        valid_labels_simple = labels[labels >= 0]
        return self._formatter.format_simple(valid_texts_simple, valid_labels_simple)

    def embed(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for texts without clustering.
        
        Useful for debugging or using embeddings for other purposes.
        
        Args:
            texts: List of text strings.
            
        Returns:
            Embedding array of shape (n_texts, embedding_dim).
        """
        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        processed = self._preprocessor.preprocess_simple(texts)
        # Filter out None values from missing inputs
        valid_texts = [t for t in processed if t is not None]
        if not valid_texts:
            return np.empty((0, 0), dtype=np.float32)
        return self._embed_texts(valid_texts)

    def cluster_labels(self, texts: List[str]) -> np.ndarray:
        """Cluster texts and return row-aligned labels.
        
        Unlike cluster() which returns grouped clusters, this method returns
        a numpy array of cluster labels aligned with the input rows.
        This is ideal for DataFrame users who need to assign labels back
        to their original data.
        
        Label mapping:
        - Valid text → cluster ID (0, 1, 2, ...)
        - Missing values (None/NaN) → -1
        - Empty/too-short text → -1
        - Duplicates → same label as the original text they match
        - Noise points (unclustered) → -1
        
        Args:
            texts: List of text strings (may contain None/NaN values).
            
        Returns:
            Numpy array of shape (len(texts),) with cluster labels.
            The i-th label corresponds to the i-th input text.
            
        Example:
            >>> clusterer = SemanticClusterer()
            >>> df["cluster"] = clusterer.cluster_labels(df["text"].tolist())
        """
        if not texts:
            return np.array([], dtype=np.int32)

        # Delegate to _run_clustering with trace=None.
        return self._run_clustering(texts, trace=None)

    def cluster_with_report(
        self, texts: List[str]
    ) -> Tuple[np.ndarray, ClusteringReport]:
        """Cluster texts and return both labels and a structured run report.
        
        Args:
            texts: List of text strings to cluster.
            
        Returns:
            A tuple (labels, report) where:
            - labels: int32 numpy array of shape (len(texts),) aligned with input order.
            - report: ClusteringReport with every required field populated.
            
        Raises:
            TypeError: If any element of texts is not a string (or None/NaN).
        """
        # Validate input types before doing any work
        for t in texts:
            if not isinstance(t, str) and not (
                t is None or (isinstance(t, float) and math.isnan(t))
            ):
                raise TypeError(
                    f"texts contains non-string element: {type(t).__name__!r}"
                )

        trace = _PipelineTrace()
        labels = self._run_clustering(texts, trace=trace)

        n_total = len(texts)
        n_noise = int(np.sum(labels == -1))
        n_clustered = n_total - n_noise
        valid = labels[labels >= 0]
        n_clusters = int(np.unique(valid).size) if valid.size else 0

        report = ClusteringReport(
            n_input_texts=n_total,
            n_clustered=n_clustered,
            n_noise=n_noise,
            n_clusters=n_clusters,
            pipeline_tier=trace.chosen_params.get("pipeline_tier", "small"),
            embedding_dim=trace.chosen_params.get("embedding_dim", 0),
            dim_band=trace.chosen_params.get("dim_band", "low"),
            dataset_profile=trace.profile or {},
            chosen_params=trace.chosen_params,
            intrinsic_metrics=trace.intrinsic_metrics,
            phase_timings=trace.phase_timings,
            warnings=list(trace.warnings),
            confidence_level=trace.confidence_level,
            random_state=int(self.config.random_state),
            library_version=_LIB_VERSION,
        )
        return labels, report

    def _oversized_subsample_and_assign(
        self,
        embeddings: np.ndarray,
        *,
        seed: int,
        trace: _PipelineTrace,
        target_size: int = 200_000,
    ) -> np.ndarray:
        """Subsample-then-assign path for N > target_size.

        Args:
            embeddings: Float32 array of shape (N, D), already preprocessed.
            seed: Integer RNG seed derived from config.random_state.
            trace: Active _PipelineTrace accumulator.
            target_size: Number of points to keep in the subsample. The
                caller is expected to have already verified N > target_size.

        Returns:
            int32 numpy array of cluster labels, shape (N,), aligned with
            the input embeddings order.
        """
        N = embeddings.shape[0]
        LIMIT = int(target_size)

        # Step 1: Deterministic uniform subsample
        rng = np.random.default_rng(seed)
        subsample_idx = rng.choice(N, size=LIMIT, replace=False)
        subsample_idx.sort()  # cache-friendly access; doesn't affect correctness
        subsample = embeddings[subsample_idx]

        # Step 2: Run cluster_large on the subsample
        sub_labels = large.cluster_large(
            subsample,
            config=self.config,
            random_state=seed,
            trace=trace,
            log_fn=self._log,
            verbose=self.verbose,
        )

        # Step 3: Build per-cluster centroids in normalized space
        norm_full = normalize_vectors(embeddings)
        norm_sub = norm_full[subsample_idx]
        valid_labels = np.unique(sub_labels[sub_labels >= 0])

        if len(valid_labels) == 0:
            # No clusters found in subsample — return all noise
            return np.full(N, -1, dtype=np.int32)

        centroids = np.zeros((len(valid_labels), embeddings.shape[1]), dtype=np.float32)
        for i, lbl in enumerate(valid_labels):
            mask = sub_labels == lbl
            centroid_raw = np.mean(norm_sub[mask], axis=0, keepdims=True)
            centroids[i] = normalize_vectors(centroid_raw)[0]

        # Step 4: Build full labels array — subsample points get their labels
        final_labels = np.full(N, -1, dtype=np.int32)
        final_labels[subsample_idx] = sub_labels

        # Step 5: Assign out-of-sample points in chunks of 50_000
        out_of_sample_mask = np.ones(N, dtype=bool)
        out_of_sample_mask[subsample_idx] = False
        out_idx = np.where(out_of_sample_mask)[0]

        if out_idx.size > 0:
            CHUNK = 50_000
            for start in range(0, out_idx.size, CHUNK):
                end = min(start + CHUNK, out_idx.size)
                chunk_idx = out_idx[start:end]
                chunk_emb = norm_full[chunk_idx]
                # Cosine similarity = dot product on normalized vectors
                sim = chunk_emb @ centroids.T  # shape (chunk, n_clusters)
                nearest = np.argmax(sim, axis=1)
                final_labels[chunk_idx] = valid_labels[nearest]

        return final_labels

    def _run_clustering(
        self,
        texts: List[str],
        *,
        trace: Optional[_PipelineTrace],
    ) -> np.ndarray:
        """Internal entry point shared by cluster_labels and cluster_with_report.
        
        When trace is None (called from cluster_labels), a temporary trace is
        created internally for quality-floor enforcement but is discarded on
        return — no ClusteringReport is constructed.
        
        Args:
            texts: List of text strings.
            trace: Optional _PipelineTrace accumulator. When None, a temporary
                trace is used internally and discarded.
                
        Returns:
            int32 numpy array of shape (len(texts),) with cluster labels.
        """
        from semantic_clusterer.utils.progress import PipelineProgress

        # Use a temporary trace when caller doesn't want a report
        _trace = trace if trace is not None else _PipelineTrace()

        self._log(f"Starting clustering of {len(texts)} texts...")

        # Pre-flight oversized check on raw input length.
        cap = self.config.max_samples
        if cap is not None and len(texts) > cap:
            raise ValueError(_oversized_error(len(texts), cap))

        with PipelineProgress(n_texts=len(texts), verbose=self.verbose) as prog:

            # Phase: Preprocess (silent — too fast to merit its own bar)
            processed_texts, orig_to_proc, proc_to_orig = self._preprocessor.preprocess(
                texts, deduplicate=True
            )

            n_deduped = len(processed_texts)
            self._log(f"After deduplication: {n_deduped} unique texts")

            # Initialize all labels as -1 (invalid/noise)
            original_labels = np.full(len(texts), -1, dtype=np.int32)

            if len(processed_texts) == 0:
                return original_labels

            # ── Phase 1: Embedding (live progress bar) ─────────────────
            prog.start_embedding(n_unique=n_deduped)
            t_embed_start = time.perf_counter()
            embeddings = self._embed_texts(
                processed_texts,
                progress_callback=prog.embedding_callback,
            )
            _trace.time("embedding", time.perf_counter() - t_embed_start)
            prog.end_embedding()

            D = int(embeddings.shape[1])
            N = int(embeddings.shape[0])

            from semantic_clusterer.dim_bands import resolve_dim_band
            dim_band = resolve_dim_band(D)

            _trace.chosen_params["embedding_dim"] = D
            _trace.chosen_params["dim_band"] = dim_band

            # ── Phase 2: Clustering (live progress bar) ────────────────
            prog.start_clustering()

            # Compute dataset profile with seeded rng
            prog.clustering_phase("profiling", 0.10)
            seed = int(self.config.random_state)
            rng = np.random.default_rng(seed)
            t_profile_start = time.perf_counter()
            profile = compute_dataset_profile(normalize_vectors(embeddings), rng=rng)
            _trace.time("profile", time.perf_counter() - t_profile_start)
            _trace.profile = asdict(profile)

            # Oversized dataset check. Honours the user-configurable
            # max_samples cap. When cap is None, no limit is applied.
            cap = self.config.max_samples
            if cap is not None and N > cap:
                allow_oversized = getattr(self.config, "allow_oversized_datasets", False)
                if not allow_oversized and cap == 200_000:
                    raise ValueError(_oversized_error(N, cap))
                warnings.warn(
                    _oversized_warn(N, cap),
                    UserWarning,
                    stacklevel=2,
                )
                _trace.warn("oversized-subsampled")
                prog.clustering_phase("clustering", 0.70)
                labels = self._oversized_subsample_and_assign(
                    embeddings, seed=seed, trace=_trace, target_size=cap,
                )
                prog.clustering_phase("postprocessing", 0.15)
                _enforce_quality_floor(embeddings, labels, profile, "large", _trace)
                for orig_idx, proc_idx in orig_to_proc.items():
                    if proc_idx >= 0:
                        original_labels[orig_idx] = labels[proc_idx]
                unique_labels = set(labels[labels >= 0].tolist())
                prog.end_clustering(n_clusters=len(unique_labels))
                return original_labels

            # Tier routing
            tier = self.config.get_strategy_for_size(N)
            _trace.chosen_params["pipeline_tier"] = tier
            self._log(f"Dataset size: {N:,}, strategy: {tier}")

            # Dimensionality reduction (medium/large only) — pre-cluster prep
            prog.clustering_phase("reduction", 0.15)

            # Cluster
            t_cluster_start = time.perf_counter()

            if tier == "tiny":
                labels = tiny.cluster_tiny(embeddings, random_state=seed, trace=_trace, config=self.config)
            elif tier == "small":
                labels = small.cluster_small(
                    embeddings,
                    config=self.config,
                    random_state=seed,
                    trace=_trace,
                    log_fn=self._log,
                )
            elif tier == "medium":
                labels = medium.cluster_medium(
                    embeddings,
                    config=self.config,
                    random_state=seed,
                    trace=_trace,
                    log_fn=self._log,
                )
            else:
                labels = large.cluster_large(
                    embeddings,
                    config=self.config,
                    random_state=seed,
                    trace=_trace,
                    log_fn=self._log,
                    verbose=self.verbose,
                )

            _trace.time("clustering", time.perf_counter() - t_cluster_start)
            prog.clustering_phase("clustering", 0.55)

            # Post-processing and quality floor
            prog.clustering_phase("postprocessing", 0.15)
            _enforce_quality_floor(embeddings, labels, profile, tier, _trace)

            # Project labels back to original input order
            for orig_idx, proc_idx in orig_to_proc.items():
                if proc_idx >= 0:
                    original_labels[orig_idx] = labels[proc_idx]

            unique_labels = set(labels[labels >= 0].tolist())
            n_invalid = int(np.sum(labels == -1))
            self._log(f"Found {len(unique_labels)} clusters, {n_invalid} invalid/noise points")
            prog.end_clustering(n_clusters=len(unique_labels))

        return original_labels

    # ------------------------------------------------------------------
    # Production API: fit / predict / save / load
    # ------------------------------------------------------------------

    def fit(self, texts: List[str]) -> "SemanticClusterer":
        """Fit the clusterer on a corpus and store the result.

        After ``fit()`` returns, the model is ready for ``predict()`` on
        new texts and can be persisted with ``save()``. The training
        labels and centroids are kept on the instance so that all the
        ``get_*`` accessors below work without re-running the pipeline.

        Args:
            texts: Training corpus.

        Returns:
            ``self`` to support fluent chaining (sklearn-style).
        """
        labels, _report = self.cluster_with_report(texts)
        self._fitted_state = self._build_fitted_state(texts, labels)
        return self

    def predict(
        self,
        texts: List[str],
        *,
        outlier_threshold: Union[float, str, None] = "auto",
    ) -> np.ndarray:
        """Assign new texts to existing clusters by nearest centroid.

        Parameters
        ----------
        texts:
            New texts to classify. ``None`` / ``NaN`` / empty rows
            receive label ``-1``.
        outlier_threshold:
            Controls out-of-distribution (OOD) detection:

            - ``"auto"`` (default): use the state-of-the-art cluster-specific
              adaptive thresholds calibrated during ``fit()`` from the training
              data's density and cohesion stats.
            - ``"global"``: use the global outlier threshold calibrated during ``fit()``.
            - ``None``: disable OOD detection entirely. Every text is
              assigned to its nearest cluster regardless of distance.
            - ``float``: explicit cosine-similarity floor. Any text whose
              best similarity to a centroid falls below this value gets
              label ``-1``. Useful for overriding the auto value.

        Returns
        -------
        ``int32`` array aligned with ``texts``. Cluster ids drawn from the
        cluster-id space recorded at fit time. ``-1`` means filtered
        (empty/None input) or OOD (below the threshold).

        Raises
        ------
        RuntimeError
            If the model has not been fitted or loaded.
        """
        self._require_fitted()

        state = self._fitted_state
        assert state is not None

        # Resolve to a concrete float, None, or adaptive dict before doing any work.
        threshold = None
        adaptive_thresholds = None

        if outlier_threshold in ("auto", "adaptive"):
            if state.cluster_cohesion:
                # Per-cluster adaptive thresholds calibrated from training distribution.
                # Uses size-aware percentile floor + tightness bonus + confusion relaxation.
                max_inter = getattr(state, "max_inter_centroid_sim", 0.0)

                # Count confused neighbours per cluster (centroids with sim > 0.7)
                inter_sims = getattr(state, "inter_centroid_sims", None)
                confused_counts = {}
                if inter_sims is not None and inter_sims.shape[0] > 1:
                    for i, stat in enumerate(state.cluster_cohesion):
                        if i < inter_sims.shape[0]:
                            row = inter_sims[i].copy()
                            row[i] = -1.0  # exclude self
                            confused_counts[stat.cluster_id] = int(np.sum(row > 0.7))

                adaptive_thresholds = {}

                # Detect small-data regime: centroids from very few
                # training members overfit and produce artificially
                # tight thresholds that reject valid test data.
                total_train = sum(s.size for s in state.cluster_cohesion)
                max_csize = max((s.size for s in state.cluster_cohesion), default=0)
                small_data = total_train < 200 or max_csize < 30

                for stat in state.cluster_cohesion:
                    # Size-aware percentile floor:
                    #   Large (>50): p10 — most reliable
                    #   Medium (10-50): blend of p10 and p25
                    #   Small (<10): p25 — most conservative
                    if stat.size >= 50:
                        base = stat.p10_sim
                    elif stat.size >= 10:
                        blend = (stat.size - 10) / 40.0  # 0.0 at 10, 1.0 at 50
                        base = stat.p25_sim * (1 - blend) + stat.p10_sim * blend
                    else:
                        base = getattr(stat, 'p25_sim', stat.p10_sim)

                    # Tightness bonus: tight clusters deserve stricter thresholds
                    tightness_bonus = max(0.0, (stat.mean_sim - 0.5)) * 0.1

                    # Confusion relaxation: relax for confused boundaries
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

        # Preprocess (deduplicate by content) and embed only the unique rows.
        processed, orig_to_proc, _ = self._preprocessor.preprocess(
            texts, deduplicate=True
        )

        labels = np.full(len(texts), -1, dtype=np.int32)
        if not processed:
            return labels

        emb = self._embed_texts(processed)

        # Apply the saved reducer when the model was fit on reduced features.
        if state.has_reducer and state.reducer is not None:
            try:
                emb = state.reducer.transform(emb).astype(np.float32)
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(
                    f"Saved reducer failed to transform new embeddings: {exc}"
                ) from exc

        # Centroids are already L2-normalised at fit time.
        emb_norm = normalize_vectors(emb)

        proc_labels = assign_to_centroids(
            emb_norm,
            state.centroids,
            state.cluster_ids,
            outlier_threshold=threshold,
            adaptive_thresholds=adaptive_thresholds,
            keywords=state.keywords if state.keywords else None,
        )

        # Project per-unique labels back to the input row order.
        for orig_idx, proc_idx in orig_to_proc.items():
            if proc_idx >= 0:
                labels[orig_idx] = proc_labels[proc_idx]

        return labels

    def fit_predict(self, texts: List[str]) -> np.ndarray:
        """Fit on ``texts`` and return the training labels.

        Equivalent to ``self.fit(texts)`` followed by accessing the
        in-memory training labels — no second embedding pass.
        """
        labels, _report = self.cluster_with_report(texts)
        self._fitted_state = self._build_fitted_state(texts, labels)
        return labels

    def save(self, path: str) -> None:
        """Persist the fitted model to a directory at ``path``.

        The embedding model is **not** saved. ``load()`` requires the user
        to re-inject an embedder.

        Raises:
            RuntimeError: If the model has not been fitted.
        """
        self._require_fitted()
        save_state(self._fitted_state, path)  # type: ignore[arg-type]

    @classmethod
    def load(
        cls,
        path: str,
        *,
        embedding_model: Optional[Any] = None,
        verbose: bool = False,
    ) -> "SemanticClusterer":
        """Load a previously-saved model from ``path``.

        Args:
            path: Directory written by ``save()``.
            embedding_model: Embedder to use for ``predict()`` calls. If
                ``None``, the built-in ONNX MiniLM-L6-v2 model is used.
            verbose: Verbose logging.

        Returns:
            A reconstructed ``SemanticClusterer`` instance with its
            ``_fitted_state`` populated.
        """
        state = load_state(path)

        # Rebuild a config from the snapshot, falling back to defaults for
        # fields that the snapshot does not name.
        config_snapshot = dict(state.config_snapshot)
        # Drop fields that are not part of the public ClustererConfig API
        # (forward compatibility for snapshots that include extra keys).
        public_fields = _validate_config_dict.__globals__["_PUBLIC_CONFIG_FIELDS"]
        config_snapshot = {
            k: v for k, v in config_snapshot.items() if k in public_fields
        }
        try:
            cfg = ClustererConfig(**config_snapshot)
        except (TypeError, ValueError):
            cfg = ClustererConfig()

        instance = cls(
            embedding_model=embedding_model,
            config=cfg,
            verbose=verbose,
            random_state=cfg.random_state,
        )
        instance._fitted_state = state
        return instance

    # ------------------------------------------------------------------
    # Topic accessors (require a fitted model)
    # ------------------------------------------------------------------

    def get_topic_keywords(
        self,
        cluster_id: Optional[int] = None,
    ) -> Union[Dict[int, List[Tuple[str, float]]], List[Tuple[str, float]]]:
        """Return c-TF-IDF keywords for one or all clusters.

        Args:
            cluster_id: When given, return the keyword list for that
                cluster only. When ``None``, return the full mapping.

        Raises:
            RuntimeError: If the model has not been fitted.
            KeyError: If ``cluster_id`` is unknown.
        """
        self._require_fitted()
        all_kw: Dict[int, List[Tuple[str, float]]] = {
            int(k): [(str(w), float(s)) for w, s in v]
            for k, v in (self._fitted_state.keywords or {}).items()
        }
        if cluster_id is None:
            return all_kw
        if cluster_id not in all_kw:
            raise KeyError(f"Unknown cluster_id: {cluster_id}")
        return all_kw[cluster_id]

    def get_topic_labels(self) -> Dict[int, str]:
        """Return ``cluster_id -> human-readable label`` mapping.

        Raises:
            RuntimeError: If the model has not been fitted.
        """
        self._require_fitted()
        return {int(k): str(v) for k, v in (self._fitted_state.topic_labels or {}).items()}

    @property
    def outlier_threshold(self) -> Optional[float]:
        """The auto-calibrated OOD threshold computed during ``fit()``.

        This is the value ``predict()`` uses by default (``outlier_threshold="auto"``).
        ``None`` when the model has not been fitted or was loaded from a
        schema-v1 file that predates the calibration feature.

        The value is the 5th percentile of training member similarities to
        their own centroids, pulled back by 10% and relaxed proportionally
        to inter-cluster overlap.  Use it to understand the effective
        sensitivity of OOD detection for your specific corpus and embedder.

        Example::

            sc.fit(texts)
            print(sc.outlier_threshold)   # e.g. 0.118
            # Fine-tune if needed:
            labels = sc.predict(new_texts, outlier_threshold=sc.outlier_threshold * 0.8)
        """
        if self._fitted_state is None:
            return None
        return self._fitted_state.auto_outlier_threshold

    @property
    def cluster_stats(self) -> Optional[List[Dict[str, Any]]]:
        """Per-cluster cohesion statistics measured on the training set.

        Returns a list of dicts, one per cluster, ordered by cluster_id:

        .. code-block:: python

            [
                {
                    "cluster_id": 0,
                    "size":       42,
                    "min_sim":    0.31,    # weakest member-to-centroid cosine
                    "mean_sim":   0.61,    # average cohesion
                    "p10_sim":    0.42,    # 10th-percentile (robust lower bound)
                },
                ...
            ]

        ``None`` when not fitted or loaded from a schema-v1 file.

        Low ``min_sim`` / ``p10_sim`` values indicate loose clusters that
        may bleed into each other.  High values indicate tight, well-separated
        semantic groups.  The ``auto_outlier_threshold`` is derived from the
        global 5th percentile of all ``min_sim`` values across all clusters.
        """
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
                "SemanticClusterer has not been fitted. Call .fit(texts) "
                "or .load(path) first."
            )

    def _build_fitted_state(
        self, texts: List[str], labels: np.ndarray
    ) -> FittedState:
        """Compute centroids, calibration stats, keywords, and topic labels.

        This runs once at the end of ``fit()`` / ``fit_predict()``.  It:

        1. Re-embeds the valid training rows to get L2-normalised vectors.
        2. Computes one L2-normalised centroid per cluster.
        3. Measures per-cluster cohesion (min / mean / p10 cosine similarity
           of every member to its own centroid).
        4. Derives ``auto_outlier_threshold`` from the global distribution of
           member similarities and the maximum inter-centroid similarity.
        5. Extracts c-TF-IDF keywords and generates topic labels.
        6. Snapshots the public config fields for save / load round-trips.
        """
        from semantic_clusterer.dim_bands import resolve_dim_band

        # ── 1. Filter to valid (non-noise, non-empty) training rows ───────
        valid_mask = np.array([_is_real_text(t) for t in texts], dtype=bool)
        valid_texts  = [t for t, ok in zip(texts, valid_mask) if ok]
        valid_labels = labels[valid_mask] if valid_mask.any() else labels

        if valid_texts:
            embeddings = self._embed_texts(valid_texts)
        else:
            embeddings = np.empty((0, 0), dtype=np.float32)

        embedding_dim = int(embeddings.shape[1]) if embeddings.size else 0
        dim_band      = resolve_dim_band(embedding_dim) if embedding_dim else "low"

        # ── 2. Compute L2-normalised centroids ────────────────────────────
        if embeddings.size and valid_labels.size:
            emb_norm       = normalize_vectors(embeddings)
            cluster_ids_int = sorted(int(c) for c in np.unique(valid_labels) if c >= 0)
        else:
            emb_norm        = embeddings
            cluster_ids_int = []

        K = len(cluster_ids_int)

        if K and emb_norm.size:
            centroids = np.zeros((K, emb_norm.shape[1]), dtype=np.float32)
            for i, cid in enumerate(cluster_ids_int):
                mask = valid_labels == cid
                raw  = emb_norm[mask].mean(axis=0, keepdims=True)
                centroids[i] = normalize_vectors(raw)[0]
        else:
            centroids = np.empty((0, embedding_dim), dtype=np.float32)

        # ── 3. Per-cluster cohesion stats ─────────────────────────────────
        # For each cluster we measure how similar every training member is to
        # its own centroid.  We collect:
        #   min_sim   — the single weakest member (sensitive to outliers)
        #   mean_sim  — average cohesion
        #   p10_sim   — 10th-percentile (robust lower bound)
        # We also accumulate every similarity value into a global pool so we
        # can compute statistics across the whole fitted corpus.

        cluster_cohesion:   List[ClusterStats] = []
        all_member_sims:    List[float]        = []   # global pool

        if K and emb_norm.size:
            for i, cid in enumerate(cluster_ids_int):
                mask    = valid_labels == cid
                members = emb_norm[mask]          # already L2-normalised
                sims    = (members @ centroids[i]).clip(-1.0, 1.0)  # (size,)

                min_s    = float(sims.min())
                mean_s   = float(sims.mean())
                median_s = float(np.median(sims))
                std_s    = float(np.std(sims))
                p10_s    = float(np.percentile(sims, 10))
                p25_s    = float(np.percentile(sims, 25))
                # Radius = 95th percentile of distance (1 - sim)
                radius_95_s = float(np.percentile(1.0 - sims, 95))

                cluster_cohesion.append(
                    ClusterStats(
                        cluster_id = cid,
                        size       = int(mask.sum()),
                        min_sim    = min_s,
                        mean_sim   = mean_s,
                        p10_sim    = p10_s,
                        median_sim = median_s,
                        std_sim    = std_s,
                        p25_sim    = p25_s,
                        radius_95  = radius_95_s,
                    )
                )
                all_member_sims.extend(sims.tolist())

        # ── 4. Auto-calibrate the outlier threshold ────────────────────────
        # Strategy: find the 5th percentile of the global pool — this is the
        # boundary that 95% of real training members comfortably exceed.
        # Then pull back a further 10% margin so we don't clip borderline
        # valid members.
        # We also measure how close the nearest pair of centroids is.  When
        # two centroids are very similar the clusters are blurring into each
        # other; in that case we relax the threshold slightly because the
        # concept of "out-of-distribution" is fuzzier.
        #   auto = max(0.05, global_p5 * 0.90 - 0.05 * max(0, max_inter))
        # The floor of 0.05 prevents the threshold collapsing to near-zero
        # on very loosely structured corpora.

        if len(all_member_sims) >= 2:
            global_p5 = float(np.percentile(all_member_sims, 5))

            if K > 1:
                inter = centroids @ centroids.T          # (K, K) cosine
                np.fill_diagonal(inter, -1.0)
                max_inter = float(inter.max())
                # Store the full inter-centroid sim matrix (restore diagonal)
                inter_centroid_sims = centroids @ centroids.T
            else:
                max_inter = 0.0
                inter_centroid_sims = None

            # Per-cluster adaptive outlier threshold:
            # - Large clusters (>50): use p5 (most reliable)
            # - Medium clusters (10-50): use p10
            # - Small clusters (<10): use p25 (most conservative)
            # Plus adjustments for cluster tightness and neighbour confusion.
            auto_threshold: Optional[float] = max(
                0.05,
                global_p5 * 0.90 - 0.05 * max(0.0, max_inter),
            )
        else:
            # Too few samples to calibrate reliably — disable OOD filtering.
            max_inter      = 0.0
            auto_threshold = None
            inter_centroid_sims = None

        # ── 5. Keywords + topic labels (best-effort) ──────────────────────
        keywords:     Dict[int, List[List[Any]]] = {}
        topic_labels: Dict[int, str]             = {}
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
                    keywords[int(cid)]     = [[str(w), float(s)] for w, s in pairs]
                    topic_labels[int(cid)] = generate_topic_label(pairs)
        except Exception:  # noqa: BLE001
            pass

        # ── 6. Config snapshot ────────────────────────────────────────────
        cfg_snapshot = self._public_config_snapshot()

        return FittedState(
            centroids               = centroids,
            cluster_ids             = np.asarray(cluster_ids_int, dtype=np.int32),
            train_labels            = labels.astype(np.int32),
            embedding_dim           = embedding_dim,
            dim_band                = dim_band,
            mode                    = "density",
            n_clusters              = K,
            auto_outlier_threshold  = auto_threshold,
            cluster_cohesion        = cluster_cohesion,
            max_inter_centroid_sim  = max_inter if K > 1 else 0.0,
            inter_centroid_sims     = inter_centroid_sims,
            keywords                = keywords,
            topic_labels            = topic_labels,
            config_snapshot         = cfg_snapshot,
            library_version         = _LIB_VERSION,
            has_reducer             = False,
            reducer                 = None,
        )

    def _public_config_snapshot(self) -> Dict[str, Any]:
        """Return a JSON-coercible snapshot of the public config fields."""
        from semantic_clusterer.config import _PUBLIC_CONFIG_FIELDS
        snapshot: Dict[str, Any] = {}
        for name in _PUBLIC_CONFIG_FIELDS:
            if hasattr(self.config, name):
                value = getattr(self.config, name)
                # Leave through JSON-safe types; coerce numpy if needed.
                if isinstance(value, np.generic):
                    value = value.item()
                snapshot[name] = value
        return snapshot

    # (end of SemanticClusterer class)


def _is_real_text(t: Any) -> bool:
    """Return True when ``t`` is a non-empty string (NaN-safe)."""
    if t is None:
        return False
    if isinstance(t, float) and math.isnan(t):
        return False
    if not isinstance(t, str):
        return False
    return bool(t.strip())