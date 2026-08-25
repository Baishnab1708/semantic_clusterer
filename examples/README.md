# Examples

Runnable, self-contained scripts that demonstrate every major feature.

| File                                  | What it shows                                                             |
|---------------------------------------|---------------------------------------------------------------------------|
| `01_beginner_zero_config.py`          | Zero-config `SemanticClusterer.cluster()` — one line, works              |
| `02_intermediate_custom_embedder.py`  | Custom embedder + `return_format="detailed"` with keywords               |
| `03_advanced_full_control.py`         | `ClustererConfig` + `cluster_with_report` + JSON report                  |
| `04_ksplit_basic.py`                  | Fixed-K with `SemanticKSplit.split()` (backward-compat shim)             |
| `05_ksplit_labels_and_report.py`      | `split_labels` + `split_with_report` + intrinsic metrics                 |
| `06_ksplit_custom_embedder.py`        | `SemanticKSplit` with a custom embedder + detailed output                |
| `07_advanced_azure_openai.py`         | Azure OpenAI `text-embedding-3-small` + full config + report             |
| [`08_fit_predict_save_load.py`](https://github.com/Baishnab1708/semantic_clusterer/blob/main/examples/08_fit_predict_save_load.py)         | **v0.1.0** production workflow: `fit / predict / save / load`            |

## Quick reference

```python
# Beginner — one line
from semantic_clusterer import SemanticClusterer
clusters = SemanticClusterer().cluster(texts)

# With topic labels (v0.1.0)
clusters = SemanticClusterer().cluster(texts, return_format="detailed")
for c in clusters:
    print(c["topic_label"], c["keywords"][:5])

# Fixed-K — exact number of groups (v0.1.0 unified API)
sc = SemanticClusterer(n_clusters=4)
groups = sc.cluster(texts)            # always exactly 4

# Production workflow (v0.1.0)
sc = SemanticClusterer().fit(texts)   # train
sc.save("./model")                    # persist (no embedder stored)
loaded = SemanticClusterer.load("./model")
labels = loaded.predict(new_texts)   # serve
labels = loaded.predict(new_texts, outlier_threshold=0.7)  # OOD → -1
```

## Running

```bash
# No credentials needed (built-in ONNX embedder)
python examples/01_beginner_zero_config.py
python examples/02_intermediate_custom_embedder.py
python examples/03_advanced_full_control.py
python examples/04_ksplit_basic.py
python examples/05_ksplit_labels_and_report.py
python examples/06_ksplit_custom_embedder.py
python examples/08_fit_predict_save_load.py

# Requires: pip install langchain-openai python-dotenv
# Requires: .env with AZURE_OPENAI_* credentials
python examples/07_advanced_azure_openai.py
```

The first run with the built-in embedder downloads the ONNX MiniLM-L6-v2
model (~90 MB) into `~/.cache/semantic_clusterer/`. All subsequent runs use
the local cache.

> **Note on Scale:** The examples above run on small-to-medium corpora. `SemanticClusterer`
> automatically routes across 4 tiers (`tiny`, `small`, `medium`, `large`). The `tiny`,
> `small`, and `medium` tiers are fully tested and benchmarked in v0.1.0; the `large` tier
> (50K+ docs) has been architecturally upgraded but is untested in this release.

## What the saved model directory contains

`SemanticClusterer.save(path)` writes:

| File | Contents |
|------|----------|
| `manifest.json` | Schema version, dim, dim_band, mode, config snapshot |
| `centroids.npy` | L2-normalised cluster centroids (`float32`, shape `(K, D)`) |
| `labels.npy` | Training labels (`int32`, shape `(N_train,)`) |
| `keywords.json` | c-TF-IDF keywords and topic labels per cluster |
| `reducer.pkl` | Fitted sklearn PCA (only when reduction was applied) |

The embedding model is **not** saved. Re-inject it on load:

```python
loaded = SemanticClusterer.load("./model", embedding_model=my_embedder)
```
