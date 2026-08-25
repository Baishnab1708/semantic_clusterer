# Changelog

All notable changes to `semantic_clusterer` are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-06-30

First public release of `semantic_clusterer`, unifying the variable-K density clustering and fixed-K partitioning pipelines under a clean, serializable, and high-performance API.

### Added

* **Production lifecycle serialization**: Added complete `fit()`, `predict()`, `save()`, and `load()` support. Fitted coordinate states are stored as compact NumPy (`.npy`) and JSON files for fast millisecond-scale prediction serving without loading PyTorch or GPU resources.
* **Scale-adaptive multi-tier routing**: Automatically selects execution tiers (Tiny, Small, Medium, Large) dynamically based on dataset scale $N$:
  * **Tiny ($N \le 150$)**: Deterministic Ward and Average linkage agglomerative cuts with dendrogram-jump height analysis.
  * **Small ($151 \le N \le 5000$)**: Multi-restart PCA + UMAP + HDBSCAN sweeping.
  * **Medium ($5001 \le N \le 50000$)**: Profile-guided PCA reductions and fast sweeps.
  * **Large ($N > 50000$)**: coarse MiniBatchKMeans sharding, recursive shard balancing, per-shard clustering, and centroid-stitching.
* **Dimension bands**: Four embedding bands (`low`, `mid`, `high`, `xhigh`) from 256 to 16,384 dimensions to map custom parameter sweeps automatically without manual adjustments.
* **Adaptive outlier thresholds**: Calibrates per-cluster tightness boundaries and global noise fallbacks during training to filter out unaligned prompts as noise (`-1`) during prediction.
* **Deterministic execution & permutation invariance**: Integrates robust seed propagation through HDBSCAN sweeps, and lexicographical row sorting to ensure identical cluster labels regardless of row order.
* **c-TF-IDF keyword labels**: Automatically computes term relevance stats and selects two non-overlapping representative terms as topic labels using Character Trigram MMR.
* **Universal embedder adapters**: Built-in support for sentence-transformers, LangChain, Azure OpenAI, and local ONNX MiniLM models.
