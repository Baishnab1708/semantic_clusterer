# semantic-clusterer - User Guide

A practical, task-oriented guide to `semantic-clusterer`. It starts with the
absolute basics and builds up to production deployment and power-user tuning.
Every snippet is runnable as-is.

If you want to understand *how the engine works internally* (pipeline tiers,
scoring, determinism), read [`WORKING.md`](WORKING.md). This guide is about
*what to do*; `WORKING.md` is about *why it does it*.

---

## Contents

1. [Mental model - pick your class](#1-mental-model-pick-your-class)
2. [Install and first run](#2-install-and-first-run)
3. [Level 1 - Beginner: one line](#3-level-1-beginner-one-line)
4. [Level 2 - Detailed output, labels, keywords](#4-level-2-detailed-output-labels-keywords)
5. [Level 3 - Fixed-K with SemanticKSplit](#5-level-3-fixed-k-with-semanticksplit)
6. [Level 4 - Custom embedders](#6-level-4-custom-embedders)
7. [Level 5 - The two tuning knobs](#7-level-5-the-two-tuning-knobs)
8. [Level 6 - Production lifecycle: fit / predict / save / load](#8-level-6-production-lifecycle-fit-predict-save-load)
9. [Level 7 - Out-of-distribution detection](#9-level-7-out-of-distribution-detection)
10. [Level 8 - Run reports and introspection](#10-level-8-run-reports-and-introspection)
11. [Level 9 - Configuration reference](#11-level-9-configuration-reference)
12. [Level 10 - Power-user controls](#12-level-10-power-user-controls)
13. [Working with DataFrames](#13-working-with-dataframes)
14. [Determinism and reproducibility](#14-determinism-and-reproducibility)
15. [Performance and scaling](#15-performance-and-scaling)
16. [Error handling and edge cases](#16-error-handling-and-edge-cases)
17. [Troubleshooting / FAQ](#17-troubleshooting-faq)
18. [Migration notes](#18-migration-notes)

---

## 1. Mental model - pick your class

The library exposes exactly two entry-point classes. Picking the right one is
the only decision you have to make up front.

| You want… | Use | Returns |
|-----------|-----|---------|
| The data to decide how many groups exist | `SemanticClusterer` | A variable number of clusters; some items may be flagged as noise (`-1`) |
| Exactly `k` groups, no matter what | `SemanticKSplit` | Exactly `k` non-empty groups; every valid item gets a group |

Rules of thumb:

- **Don't know the number of topics?** Use `SemanticClusterer`. It finds the
  natural structure and can mark outliers as noise.
- **Need to fill `k` buckets** (e.g. "split these 10,000 tickets across 8
  reviewers")? Use `SemanticKSplit`. It guarantees `k` non-empty partitions.

Both classes:

- Accept the same kinds of embedding models (or use the built-in one).
- Share the same `fit / predict / save / load` production lifecycle.
- Produce the same output formats (`simple`, `detailed`, row-aligned labels).
- Are deterministic for a fixed `random_state`.

---

## 2. Install and first run

```bash
pip install semantic-clusterer
```

Requirements:

- Python 3.9 or newer.
- `hdbscan` and `umap-learn` are installed automatically as hard dependencies.

The first time you cluster *without* supplying your own embedder, the library
downloads the built-in ONNX MiniLM-L6-v2 model (~90 MB) into
`~/.cache/semantic_clusterer/`. Every later run uses that local cache, so it is
a one-time cost. The download is checksum-verified and retried once on
corruption.

Verify your install:

```python
import semantic_clusterer
print(semantic_clusterer.__version__)
```

---

## 3. Level 1 - Beginner: one line

The zero-config path. No model, no parameters — just text in, groups out.

```python
from semantic_clusterer import SemanticClusterer

texts = [
    "How do I reset my password?",
    "I forgot my login password",
    "What are your business hours?",
    "When do you open on weekdays?",
    "My package hasn't arrived",
    "Where is my order?",
]

clusterer = SemanticClusterer()
groups = clusterer.cluster(texts)

for i, group in enumerate(groups):
    print(f"Cluster {i}:")
    for t in group:
        print(f"   - {t}")
```

`groups` is a `List[List[str]]` — each inner list is one cluster, holding the
original (unmodified) input strings. Items the algorithm considers noise are
simply omitted from the `simple` output.

That's the entire beginner experience. Everything below is opt-in.

---

## 4. Level 2 - Detailed output, labels, keywords

### Detailed clusters

Pass `return_format="detailed"` to get rich metadata per cluster instead of
plain lists.

```python
clusters = SemanticClusterer().cluster(texts, return_format="detailed")

for c in clusters:
    print(c["cluster_id"])      # 0, 1, 2, ...
    print(c["topic_label"])     # e.g. "Reset Password & Login Account"
    print(c["keywords"][:5])    # top c-TF-IDF keywords
    print(c["representative"])  # the item closest to the cluster centroid
    print(c["size"])            # number of items
    print(c["confidence"])      # mean cosine similarity to centroid (0..1)
    print(c["items"])           # the original texts in this cluster
```

Each detailed cluster is a dictionary with these keys:

| Key | Type | Meaning |
|-----|------|---------|
| `cluster_id` | `int` | Stable cluster identifier (0-based) |
| `representative` | `str` | The single item nearest the cluster centroid |
| `items` | `List[str]` | All original texts assigned to this cluster |
| `size` | `int` | `len(items)` |
| `confidence` | `float` | Mean cosine similarity of members to the centroid. A **cohesion** measure, not a probability — values near 1.0 mean a tight, coherent cluster |
| `keywords` | `List[str]` | Top keywords from class-based TF-IDF (c-TF-IDF) |
| `topic_label` | `str` | A short human-readable label derived from the keywords |

### Row-aligned labels

When you need one label per input row (for joining back to a table), use
`cluster_labels()`:

```python
labels = SemanticClusterer().cluster_labels(texts)
# numpy int32 array, shape (len(texts),)
#   0, 1, 2, ... → cluster id
#  -1            → filtered (empty / None / NaN) or noise
```

The i-th label corresponds to the i-th input text. Duplicates get the same
label as the first occurrence; missing/empty inputs get `-1`.

### Turning keywords off

Keyword and topic-label generation is a pure post-processing step that never
affects cluster assignments. Disable it to save a little time:

```python
from semantic_clusterer import SemanticClustererConfig

sc = SemanticClusterer(config=SemanticClustererConfig(extract_keywords=False))
```

---

## 5. Level 3 - Fixed-K with SemanticKSplit

When you need a guaranteed number of groups:

```python
from semantic_clusterer import SemanticKSplit

ks = SemanticKSplit(k=3)          # k is a required keyword argument, >= 2
groups = ks.split(texts)          # always exactly 3 non-empty groups
```

`k` is **required** and must be an integer `>= 2`. The constraint
`2 <= k <= number of unique texts` is checked at run time. If `k` equals the
number of unique inputs, you get a warning (each cluster will have a single
member).

`SemanticKSplit` mirrors `SemanticClusterer`'s methods:

```python
groups   = ks.split(texts)                          # List[List[str]], length k
groups   = ks.split(texts, return_format="detailed")# rich dicts
labels   = ks.split_labels(texts)                   # row-aligned int32 array
labels, report = ks.split_with_report(texts)        # labels + ClusteringReport
vectors  = ks.embed(texts)                           # embeddings only
clusters = ks.cluster(texts)                         # alias for split()
```

Key differences from `SemanticClusterer`:

- It never returns fewer than `k` groups. If the algorithm leaves a cluster
  empty, an automatic repair step bisects the largest cluster until all `k`
  labels are populated.
- Valid inputs never receive `-1` from `split()`/`split_labels()` — only
  filtered rows (empty/None/NaN) do.
- It does not import `hdbscan` at all; it uses partition algorithms (k-means
  family, spectral, agglomerative) chosen by tier and `k`.

---

## 6. Level 4 - Custom embedders

The built-in ONNX MiniLM-L6-v2 is convenient but small. To use a stronger or
domain-specific model, pass any object that exposes one of these methods — the
library auto-detects the interface:

| Method | Typical source |
|--------|----------------|
| `.embed(texts) -> np.ndarray` | Custom embedder (preferred) |
| `.encode(texts, **kwargs) -> np.ndarray` | SentenceTransformers / HuggingFace |
| `.embed_documents(texts) -> list` | LangChain embeddings |
| `__call__(texts) -> np.ndarray` | A plain callable / lambda |

### SentenceTransformers

```python
from sentence_transformers import SentenceTransformer
from semantic_clusterer import SemanticClusterer

model = SentenceTransformer("all-mpnet-base-v2")   # 768-dim
sc = SemanticClusterer(embedding_model=model)
groups = sc.cluster(texts)
```

### A plain callable

```python
def my_embedder(texts):
    # must return a 2D array of shape (len(texts), dim)
    return my_api.encode(texts)

sc = SemanticClusterer(embedding_model=my_embedder)
```

Callables are automatically chunked in `batch_size` blocks to protect remote
APIs from oversized payloads.

### LangChain / Azure OpenAI

```python
import os
from langchain_openai import AzureOpenAIEmbeddings
from semantic_clusterer import SemanticClusterer

embedder = AzureOpenAIEmbeddings(
    azure_deployment=os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT"],
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    api_version=os.environ["AZURE_OPENAI_API_VERSION"],
)
sc = SemanticClusterer(embedding_model=embedder)
```

You do **not** need to retune anything when you change models. The library
detects the embedding dimension and selects a matching internal parameter grid
(see "dimension bands" in [`WORKING.md`](WORKING.md)). A 384-dim model and a
3072-dim model both work out of the box.

### What the library expects back

- A 2D numeric array shaped `(n_texts, embedding_dim)`.
- No `NaN` or `Inf` values (these raise a `ValueError`).
- For a single text, return shape `(1, dim)`, not `(dim,)`.

Embeddings are validated and cast to `float32` internally. If
`normalize_embeddings=True` (the default), custom-model outputs are
L2-normalised before clustering. The built-in ONNX embedder already normalises
its own output.

---

## 7. Level 5 - The two tuning knobs

You almost never need to tune this library. There is exactly **one** primary
knob per class, and both default to `"balanced"`.

### `cluster_granularity` — for `SemanticClusterer`

Controls how many clusters you get and how aggressively near-identical clusters
are merged.

| Value | Result | Use when |
|-------|--------|----------|
| `"fine"` | Most clusters, many sub-topics | Topic discovery, exploration |
| `"balanced"` (default) | Moderate, clean clusters | General use |
| `"coarse"` | Fewest, broadest clusters | High-level grouping |

```python
SemanticClusterer(cluster_granularity="balanced")   # default
SemanticClusterer(cluster_granularity="coarse")     # fewer, bigger
SemanticClusterer(cluster_granularity="fine")       # more, smaller
```

Under the hood, `granularity`:

- Raises the minimum cluster size floor (coarser → higher floor).
- Runs a centroid-merge pass after clustering (coarser → lower merge
  threshold, so more clusters merge).
- Adds a fragmentation penalty to the internal parameter search.

### `quality` — for `SemanticKSplit`

Controls how many independent restarts run before the best partition is
selected.

| Value | Restarts (small tier) | Use when |
|-------|----------------------|----------|
| `"fast"` | 1 | Quick iteration |
| `"balanced"` (default) | 5 | Production |
| `"best"` | 10 | Quality is critical |

```python
SemanticKSplit(k=8, quality="balanced")   # default
SemanticKSplit(k=8, quality="best")       # most thorough
SemanticKSplit(k=8, quality="fast")       # single pass
```

Restart counts scale down for larger tiers (medium/large) because each restart
is more expensive there. Deterministic algorithms (e.g. agglomerative in the
tiny tier) ignore the restart count — running them repeatedly gives the same
answer, so it would be wasted work.

> **Pipeline tier and dimensionality reduction are not knobs.** They are chosen
> automatically from dataset size and embedding dimension. This is deliberate:
> it keeps the API tiny and the results consistent. See
> [`WORKING.md`](WORKING.md) for the routing details.

---

## 8. Level 6 - Production lifecycle: fit / predict / save / load

The one-shot `cluster()` / `split()` methods re-run the whole pipeline every
call. For production you usually want to **train once** and then **serve
predictions** cheaply — possibly in a different process or on a different
machine. That is what `fit / predict / save / load` is for.

### The four-step lifecycle

```python
from semantic_clusterer import SemanticClusterer

# 1. Train once (offline) -----------------------------------------------
sc = SemanticClusterer(cluster_granularity="balanced")
sc.fit(corpus)                       # discovers clusters, computes centroids

# 2. Inspect what it learned --------------------------------------------
print(sc.is_fitted)                  # True
print(sc.outlier_threshold)          # auto-calibrated OOD floor, e.g. 0.34
print(sc.get_topic_labels())         # {0: "Reset Password", 1: "Shipping", ...}

# 3. Persist to disk ----------------------------------------------------
sc.save("./model")                   # a directory of small, inspectable files

# 4. Serve (separate process / deployment) ------------------------------
loaded = SemanticClusterer.load("./model")
labels = loaded.predict(new_texts)   # assigns each text to a trained cluster
```

### What `fit()` computes

`fit()` runs the full clustering pipeline and then builds a compact "fitted
state":

- **L2-normalised centroids** — one per discovered cluster.
- **Per-cluster cohesion stats** — min / mean / 10th-percentile member
  similarity to the centroid.
- **An auto-calibrated `outlier_threshold`** (see [section 9](#9-level-7--out-of-distribution-detection)).
- **Keywords and topic labels** per cluster (unless disabled).
- **A snapshot of the public config** for faithful save/load round-trips.

`fit()` returns `self`, so you can chain: `sc = SemanticClusterer().fit(corpus)`.

`fit_predict(texts)` does the same work and returns the training labels in one
call, without a second embedding pass.

### `predict()` — assigning new text

`predict()` embeds new texts, normalises them, and assigns each to the nearest
trained centroid by cosine similarity:

```python
labels = loaded.predict(new_texts)                       # OOD → -1 (auto)
labels = loaded.predict(new_texts, outlier_threshold=None)  # never -1 for OOD
labels = loaded.predict(new_texts, outlier_threshold=0.5)   # explicit floor
```

`predict()` does **not** re-cluster. It is a fast nearest-centroid assignment,
so it scales to large prediction batches.

### What gets saved

`save(path)` writes a directory:

```
model/
  manifest.json    schema version, embedding dim, dim_band, mode,
                   config snapshot, auto outlier threshold, class name
  centroids.npy    float32 (K, D), L2-normalised cluster centroids
  labels.npy       int32 (N_train,), training labels
  keywords.json    c-TF-IDF keywords + topic labels per cluster
  stats.json       per-cluster cohesion statistics
  reducer.pkl      OPTIONAL — only when a fitted PCA was part of the model
```

The files are deliberately small and human-inspectable. The manifest is
written last, so its presence signals a complete, consistent directory.

### The embedding model is never saved

This is intentional — it avoids pickling arbitrary Python and locking the saved
model to one embedder version. When you `load()`, re-inject the same embedder
you trained with:

```python
loaded = SemanticClusterer.load("./model", embedding_model=my_embedder)
```

If you trained with the built-in ONNX embedder, you can omit
`embedding_model` and it will be used again automatically.

> **Important:** Always predict with the *same* embedding model you trained
> with. Centroids live in that model's vector space; a different model produces
> incompatible vectors and meaningless assignments.

### Same lifecycle for SemanticKSplit

```python
ks = SemanticKSplit(k=8, quality="balanced")
ks.fit(corpus)
ks.save("./ksplit_model")

loaded_ks = SemanticKSplit.load("./ksplit_model")   # k is restored from manifest
labels = loaded_ks.predict(new_texts)
```

`SemanticKSplit.load()` reads `k` back from the saved manifest, so you don't
re-specify it.

---

## 9. Level 7 - Out-of-distribution detection

A common production need: "what happens when I predict on text that doesn't
belong to any trained cluster?" The library handles this with an
**auto-calibrated outlier threshold** and **per-cluster adaptive boundaries**.

### How it is calibrated

During `fit()`, the library measures how similar every training member is to
its own cluster centroid. It records detailed statistics (`min_sim`, `mean_sim`, 
`median_sim`, `std_sim`, `p10_sim`, `p25_sim`, and `radius_95`) and pools all member similarities.

1. **Global Threshold**: The **5th percentile** of the global pool, pulled back slightly and relaxed when centroids overlap. This is used by `"global"` mode.
2. **Adaptive Thresholds**: Per-cluster thresholds computed dynamically using:
   - **Size-aware percentile floor**: Large clusters ($>50$) use `p10`, small clusters ($<10$) use `p25` to prevent false assignments.
   - **Tightness bonus**: Highly cohesive clusters (high `mean_sim`) receive a tighter threshold.
   - **Confusion relaxation**: Centroids that are close to neighbors (similarity $>0.7$) have their thresholds relaxed.
   - **Small-data safety valve**: When trained on small corpora ($N < 200$ or max cluster $<30$), the pullback factor is dynamically relaxed and thresholds are capped at the global 5th-percentile baseline to prevent centroid overfitting on few training samples from rejecting valid test queries.

This per-cluster adaptive calculation runs automatically during `.fit()` and is stored in the model manifest (Schema v3).

### Using it in predict()

`predict()` accepts an `outlier_threshold` argument with the following modes:

```python
labels = sc.predict(new_texts)                           # "auto" (default, adaptive per-cluster)
labels = sc.predict(new_texts, outlier_threshold="global") # unified global threshold
labels = sc.predict(new_texts, outlier_threshold=None)    # disable OOD entirely (forced assignment)
labels = sc.predict(new_texts, outlier_threshold=0.5)     # explicit float floor
```

* **`"auto"` / `"adaptive"`** *(default)*: Uses size-aware, tightness-aware, and overlap-aware per-cluster thresholds. 
* **`"global"`**: Uses a single unified global outlier threshold.
* **`None`**: every text is assigned to its nearest cluster, no matter how far.
* **a `float`**: your own cosine-similarity floor; overrides the auto values.

### Margin-Based Disambiguation
When predicting, if a text falls near the decision boundary between two clusters (i.e. the difference between the top-2 cosine similarities is $<0.03$), the system resolves the tiebreaker using cluster density and cohesion stats (or token matching if keywords are present).

### Vectorized Performance
outlier assignment and adaptive filtering are fully vectorized in NumPy, making `.predict()` extremely fast even with thousands of clusters and batch predictions.

### Inspecting cohesion

`cluster_stats` exposes the per-cluster cohesion measured at fit time:

```python
for s in sc.cluster_stats:
    print(
        f"Topic {s['cluster_id']}: size={s['size']}, "
        f"mean={s['mean_sim']:.4f}, median={s['median_sim']:.4f}, "
        f"std={s['std_sim']:.4f}, radius_95={s['radius_95']:.4f}"
    )
```

- Low `mean_sim` / high `std_sim` / high `radius_95` $\rightarrow$ a loose cluster.
- High `mean_sim` / low `std_sim` / low `radius_95` $\rightarrow$ a tight, well-separated group.

### Fine-tuning

If `"auto"` is too aggressive or too lenient for your domain, set the threshold manually or scale the global value:

```python
# more permissive (fewer items flagged as OOD)
labels = sc.predict(new_texts, outlier_threshold=sc.outlier_threshold * 0.8)
```

The auto value works best when the texts you predict on come from the same
domain you trained on. If you train on support tickets and predict on news
headlines, set the threshold manually.

> Models loaded from an older schema (v1 or v2) degrade gracefully. They will load v3 fields with safe default fallbacks.

---

## 10. Level 8 - Run reports and introspection

For observability — logging, dashboards, debugging — use the `*_with_report`
methods. They return both the labels and a structured `ClusteringReport`.

```python
labels, report = SemanticClusterer().cluster_with_report(texts)

print(report.n_input_texts)     # 6
print(report.n_clusters)        # 3
print(report.n_noise)           # items labelled -1
print(report.pipeline_tier)     # "tiny" | "small" | "medium" | "large"
print(report.dim_band)          # "low" | "mid" | "high" | "xhigh"
print(report.embedding_dim)     # e.g. 384
print(report.chosen_params)     # the actual parameters the pipeline picked
print(report.intrinsic_metrics) # score, coverage, cohesion, separation, ...
print(report.phase_timings)     # seconds spent in each phase
print(report.warnings)          # any soft warnings raised during the run
print(report.confidence_level)  # "high" or "low"
```

### The report fields

| Field | Meaning |
|-------|---------|
| `n_input_texts` | Number of input rows |
| `n_clustered` | Rows assigned to a real cluster |
| `n_noise` | Rows labelled `-1` (noise or filtered) |
| `n_clusters` | Number of clusters found (for KSplit, equals `k`) |
| `pipeline_tier` | Which tier ran: tiny / small / medium / large |
| `embedding_dim` | Embedding dimensionality |
| `dim_band` | Resolved dimension band |
| `dataset_profile` | Statistical profile of the data (medium/large) |
| `chosen_params` | Concrete parameters selected by the search |
| `intrinsic_metrics` | Quality metrics (see below) |
| `phase_timings` | Wall-clock seconds per pipeline phase |
| `warnings` | Soft warnings (e.g. `"high-noise-ratio"`) |
| `confidence_level` | `"high"` normally, `"low"` when a quality check fired |
| `random_state` | The seed used |
| `library_version` | The `semantic-clusterer` version |

### JSON serialisation

The report is JSON-safe via `to_dict()` — `NaN`/`Inf` become `null`, numpy
types become native Python types:

```python
import json
labels, report = SemanticClusterer().cluster_with_report(texts)
json.dumps(report.to_dict())     # never raises
```

### Intrinsic metrics glossary

`intrinsic_metrics` (and `score_clustering`) report:

| Metric | Range | Higher is… |
|--------|-------|------------|
| `score` | 0..1 | Better — the composite objective |
| `coverage` | 0..1 | Better — fraction of points clustered (`1 - noise_ratio`) |
| `cohesion` | 0..1 | Better — average within-cluster similarity |
| `separation` | 0..1 | Better — average distance between cluster centroids |
| `stability` | 0..1 | Better — size balance across clusters |
| `density` | 0..1 | Better — HDBSCAN validity (when available) |
| `largest_ratio` | 0..1 | Lower — size of the biggest cluster as a fraction of N |
| `fragmentation` | 0..1 | Lower — fraction of micro-clusters |
| `noise_ratio` | 0..1 | Lower — fraction of unclustered points |
| `n_clusters` | int | — |

`SemanticKSplit` reports also include `silhouette`, `davies_bouldin`,
`per_cluster_size`, and `per_cluster_cohesion`, plus a `chosen_params`
`algorithm_used` field naming the exact algorithm that ran.

---

## 11. Level 9 - Configuration reference

You can configure either class with a typed config object or a plain dict.

### Passing config

```python
from semantic_clusterer import SemanticClusterer, SemanticClustererConfig

# Typed config object
sc = SemanticClusterer(config=SemanticClustererConfig(cluster_granularity="coarse"))

# Plain dict (validated — unknown keys raise ValueError)
sc = SemanticClusterer(config={"cluster_granularity": "coarse", "batch_size": 128})

# Some knobs are also direct constructor kwargs
sc = SemanticClusterer(cluster_granularity="coarse", random_state=7, verbose=True)
```

### `SemanticClustererConfig`

| Field | Type | Default | Meaning |
|-------|------|---------|---------|
| `cluster_granularity` | `"fine"\|"balanced"\|"coarse"` | `"balanced"` | Primary knob — cluster count vs. merging |
| `min_cluster_size` | `int >= 2` or `None` | `None` | Power-user HDBSCAN override; `None` = auto |
| `min_samples` | `int >= 1` or `None` | `None` | Power-user HDBSCAN override; `None` = auto |
| `extract_keywords` | `bool` | `True` | Generate c-TF-IDF keywords + topic labels |
| `keywords_top_n` | `int >= 1` | `10` | Keywords per cluster |
| `batch_size` | `int > 0` | `64` | Embedding batch size |
| `normalize_embeddings` | `bool` | `True` | L2-normalise custom-model output |
| `random_state` | `int` in `[0, 2³²-1]` | `42` | Seed for all randomness |
| `max_samples` | `int >= 1` or `None` | `200_000` | Hard cap; `None` disables it |
| `verbose` | `bool` | `False` | Print phase-by-phase progress |

### `SemanticKSplitConfig`

Same infrastructure fields as above, with one different primary knob and no
`min_cluster_size` / `min_samples`:

| Field | Type | Default | Meaning |
|-------|------|---------|---------|
| `quality` | `"fast"\|"balanced"\|"best"` | `"balanced"` | Primary knob — restart count |
| `extract_keywords` | `bool` | `True` | Keywords + topic labels |
| `keywords_top_n` | `int >= 1` | `10` | Keywords per cluster |
| `batch_size` | `int > 0` | `64` | Embedding batch size |
| `normalize_embeddings` | `bool` | `True` | L2-normalise custom-model output |
| `random_state` | `int` in `[0, 2³²-1]` | `42` | Seed |
| `max_samples` | `int >= 1` or `None` | `200_000` | Hard cap; `None` disables |
| `verbose` | `bool` | `False` | Progress output |

### Validation

- Out-of-range or wrong-type values raise `ValueError` at construction.
- Unknown dict keys raise `ValueError` naming the allowed fields.
- Tier routing and dimensionality reduction are **internal** — passing
  `strategy=` or `reduction=` is rejected as an unknown field.

### The `ClustererConfig` alias

`ClustererConfig` is a backward-compatible alias kept for older code. New code
should prefer `SemanticClustererConfig`. It carries the same public fields plus
a deprecated `allow_oversized_datasets` flag (use `max_samples` instead).

---

## 12. Level 10 - Power-user controls

These are escape hatches. The defaults are good; reach for these only when you
have a specific reason.

### Pinning HDBSCAN parameters (SemanticClusterer)

By default the small/medium pipelines search a grid of `min_cluster_size` and
`min_samples` values. To force exact values and make the run fully
predictable:

```python
SemanticClustererConfig(min_cluster_size=15, min_samples=5)
```

When set, these override the entire search grid (the granularity floor is still
respected for `min_cluster_size`).

- `min_cluster_size` — the smallest group HDBSCAN will call a cluster. Larger
  → fewer, bigger clusters. Must be `>= 2`.
- `min_samples` — how conservative the density estimate is. Larger → more
  points treated as noise. Must be `>= 1`.

### Removing the dataset cap

By default `N > 200_000` raises `ValueError` to protect you from accidental
huge runs. To allow it, set `max_samples`:

```python
SemanticClustererConfig(max_samples=None)       # no cap — uses subsample-then-assign
SemanticClustererConfig(max_samples=500_000)    # custom cap
```

When the cap is exceeded with `max_samples=None`, the library subsamples to the
limit, clusters the subsample, then assigns the rest by nearest centroid, and
records an `"oversized-subsampled"` warning in the report.

### Tuning OOD sensitivity

Covered in [section 9](#9-level-7--out-of-distribution-detection): pass a float
to `predict(outlier_threshold=...)`, or scale `sc.outlier_threshold`.

### Inspecting the chosen parameters

`chosen_params` in the report tells you exactly what ran, which is the best
starting point before manual tuning:

```python
_, report = sc.cluster_with_report(texts)
print(report.chosen_params)
# {'pipeline_tier': 'small', 'dim_band': 'low', 'umap_n_neighbors': 25,
#  'hdbscan_min_cluster_size': 18, 'hdbscan_method': 'eom', ...}
```

---

## 13. Working with DataFrames

`cluster_labels()` (and `split_labels()`) return a row-aligned array, which
makes pandas integration a one-liner.

```python
import pandas as pd
from semantic_clusterer import SemanticClusterer

df = pd.DataFrame({"text": ["...", "...", None, "..."]})

sc = SemanticClusterer()
df["cluster"] = sc.cluster_labels(df["text"].tolist())

# -1 marks rows that were empty/None/NaN or classified as noise
clustered = df[df["cluster"] >= 0]
noise     = df[df["cluster"] == -1]
```

Notes:

- Missing values (`None`/`NaN`) are safe — they map to `-1`, never crash.
- Duplicate texts get the same label as their first occurrence.
- The output length always equals the input length, so the column aligns.

For a production table you'd typically `fit()` once, `save()`, then `predict()`
on new rows:

```python
df_new["cluster"] = loaded.predict(df_new["text"].tolist())
```

---

## 14. Determinism and reproducibility

Two runs produce permutation-equivalent labels when they share all of:

- The same `semantic-clusterer` version.
- The same Python minor version.
- The same OS family.
- The same major.minor of `numpy`, `scikit-learn`, `hdbscan`, `umap-learn`.
- The same `random_state`.

Pin the seed on the constructor:

```python
sc = SemanticClusterer(random_state=42)
ks = SemanticKSplit(k=8, random_state=42)
```

"Permutation-equivalent" means the grouping is identical, though the integer
cluster *ids* may be a relabelling (cluster 0 in one run might be cluster 2 in
another). If you need stable ids across runs, `fit()` once and `save()`/`load()`
the model rather than re-clustering.

`random_state` must be an integer in `[0, 2³² − 1]`. Booleans are rejected. When
both a constructor `random_state` kwarg and a `config.random_state` are
supplied, the constructor kwarg wins.

---

## 15. Performance and scaling

### Tier routing by size

The library automatically routes by the number of *unique* texts:

| Tier | Unique N | Approach |
|------|----------|----------|
| tiny | 1 – 150 | Exact agglomerative search |
| small | 151 – 5 000 | UMAP + HDBSCAN parameter sweep |
| medium | 5 001 – 50 000 | Profiled, bounded sweep |
| large | 50 001 – 200 000 | Coarse k-means shards → per-shard HDBSCAN |

You don't choose the tier; it is derived from your data.

### Practical tips

- **Deduplicate cost is free.** Texts are deduplicated before embedding, so
  repeated strings only get embedded once.
- **`batch_size`** controls the embedding batch. Raise it for throughput on a
  GPU-backed model; lower it if a remote API rejects large payloads.
- **`fit` once, `predict` many.** `predict()` is a cheap nearest-centroid pass;
  prefer it over re-running `cluster()` for streaming/online workloads.
- **`extract_keywords=False`** shaves the c-TF-IDF post-processing if you only
  need labels.
- **`verbose=True`** prints per-phase timings so you can see where time goes;
  the same data is in `report.phase_timings`.

### Memory

The pipeline avoids building full `N×N` distance matrices on the larger tiers.
The large tier shards the data and clusters each shard independently, then
stitches near-duplicate clusters across shard boundaries, keeping peak memory
bounded.

> **⚠️ Large pipeline status:** The `large` tier (N > 50 000) has been upgraded
> to use multi-reduction PCA search and granularity post-processing, but **has not
> yet been benchmarked or tested** in v0.1.0 releases. Tiny, small, and medium
> tiers are fully verified and benchmarked.

---

## 16. Error handling and edge cases

The library is designed to degrade gracefully rather than crash.

| Situation | Behaviour |
|-----------|-----------|
| Empty input list | Returns `[]` (or empty array) — no error |
| Single text | Returns one cluster with that text |
| All texts identical | One cluster (clusterer) / round-robin split (KSplit) |
| `None` / `NaN` / empty rows | Labelled `-1`, never crash |
| Duplicate texts | Embedded once; share the first occurrence's label |
| Non-string element (e.g. a dict) | `TypeError` with a clear message |
| `NaN`/`Inf` in embeddings | `ValueError` from validation |
| `k < 2` (KSplit) | `ValueError` |
| `k > unique texts` (KSplit) | `ValueError` |
| `N > max_samples` | `ValueError` (default), or subsample with `max_samples=None` |
| `umap-learn` import fails | Small/medium fall back to PCA-only + `UserWarning` |
| `hdbscan` missing | `SemanticClusterer.__init__` raises `ImportError` |
| `predict()` before `fit()`/`load()` | `RuntimeError` |

When a soft quality issue is detected (very high noise, score below the tier
floor), the run still succeeds but `report.confidence_level` becomes `"low"`
and a tag is added to `report.warnings`.

---

## 17. Troubleshooting / FAQ

**Too many tiny clusters.**
Use `cluster_granularity="coarse"`, or raise `min_cluster_size`. Coarse also
runs a stronger merge pass.

**Everything collapsed into one giant cluster.**
Try `cluster_granularity="fine"`, lower `min_cluster_size`, or check whether
your embedder is producing meaningful vectors (clustering is only as good as
the embeddings).

**Lots of items end up as noise (`-1`).**
That's `SemanticClusterer` saying "these don't fit any dense region." If you
need every item placed, use `SemanticKSplit` instead, or lower
`min_cluster_size`. For `predict()`, pass `outlier_threshold=None`.

**`predict()` puts everything in `-1`.**
The OOD threshold is too strict for your prediction domain, or you're predicting
with a different embedder than you trained with. Try
`predict(outlier_threshold=None)` to confirm, then pick a manual float.

**Results differ between two machines.**
Check that the library, Python minor, and numpy/sklearn/hdbscan/umap versions
match. See [section 14](#14-determinism-and-reproducibility).

**First run is slow / downloads a file.**
That's the one-time ONNX model download (~90 MB) to
`~/.cache/semantic_clusterer/`. Subsequent runs use the cache. Supply your own
`embedding_model` to skip it entirely.

**A `UserWarning` about UMAP.**
`umap-learn` failed to import; small/medium pipelines fell back to PCA-only.
Reinstall `umap-learn` to restore full behaviour.

**`ValueError: Dataset size ... exceeds the hard limit of 200_000`.**
Set `max_samples=None` to enable the subsample-then-assign path, or a custom
integer cap.

**Can I force a specific pipeline tier?**
No. Tier and reduction are internal and chosen from your data. This is by
design; see [`WORKING.md`](WORKING.md).

---

## 18. Migration notes

### `strategy` and `reduction` are no longer public

Earlier configs accepted `strategy="auto|small|medium|large"` and
`reduction="auto|pca|None"`. These are now **internal**: tier routing and
dimensionality reduction are decided automatically from dataset size and
embedding dimension.

- Passing `strategy=` or `reduction=` to a config (object or dict) now raises
  `ValueError` (unknown field).
- There is no replacement knob — the behaviour you previously got from
  `strategy="auto"` / `reduction="auto"` is exactly the default now.
- If you previously forced a tier for testing, size your input to land in the
  desired tier instead (see the size ranges in [section 15](#15-performance-and-scaling)).

Everything else — `cluster_granularity`, `quality`, `min_cluster_size`,
`min_samples`, `max_samples`, the `fit/predict/save/load` lifecycle — is
unchanged.

---

## See also

- [`README.md`](README.md) — quick overview and API surface.
- [`WORKING.md`](WORKING.md) — internal architecture and algorithms.
- [`CHANGELOG.md`](CHANGELOG.md) — version history.
- `examples/` — eight runnable scripts covering every feature.
