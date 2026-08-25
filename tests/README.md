# `semantic_clusterer` test suite

The tests mirror the package layout under `semantic_clusterer/` so finding the
test for any module is a path-for-path lookup.

```
tests/
├── conftest.py                       Shared fixtures (sample texts, embedders, stubs)
├── fixtures/
│   └── release_ready/                Frozen baselines used by release-gate tests
├── api/
│   └── test_public_api.py            __all__ surface, signatures, return-type contracts
├── semantic_clusterer/               Mirrors semantic_clusterer/
│   ├── test_core.py                  SemanticClusterer integration tests
│   ├── test_clustering_report.py     ClusteringReport field-shape & JSON round-trip
│   ├── test_dim_bands.py             dim_bands.resolve_dim_band, BandGrid invariants
│   ├── test_optional_deps.py         Degradation when umap-learn / hdbscan missing
│   ├── test_release_v010.py          Row-aligned labels, NaN/None handling, adaptive PCA
│   ├── embedding/
│   │   ├── test_adapters.py          EncodeAdapter / LangchainAdapter / CallableAdapter
│   │   └── test_download_sha256.py   Model download SHA-256 verification
│   ├── persistence/
│   │   └── test_fit_predict_save_load.py
│   │                                 fit / predict / save / load / assign_to_centroids
│   ├── pipeline/
│   │   ├── test_tiny_pipeline.py     N≤150 degenerate cases, tie-break order, traces
│   │   └── test_medium_large_pipelines.py
│   │                                 Reduction respect, partition counts, stitching
│   ├── preprocessing/
│   │   └── test_text_preprocessor.py NFKC, dedup, missing values, type errors
│   └── properties/
│       └── test_properties.py        Hypothesis-driven release contracts P1–P8
└── semantic_ksplit/                  Mirrors semantic_clusterer/k_split.py
    └── test_k_split_*.py             Constructor, algorithms, edge cases, repair, etc.
```

## Running the suite

Run everything:

```sh
pytest
```

Run a single sub-tree (e.g. just persistence):

```sh
pytest tests/semantic_clusterer/persistence
```

Skip the slow / property-based tests:

```sh
pytest -m "not benchmark" --ignore=tests/semantic_clusterer/properties
```

Markers (`benchmark`, `integration`) are declared in `pyproject.toml`.

## Conventions

* Every fixture lives in `tests/conftest.py` so any test module can use it
  without an import.
* Property-based tests live under `tests/semantic_clusterer/properties` and
  use Hypothesis with bounded `max_examples` so the suite stays under a
  few minutes.
* Tests that drive the pipeline with synthetic embeddings smaller than the
  supported low-band dim (256) declare a module-level
  `pytest.mark.filterwarnings` for the dim-band fallback warning.
* New modules added to `semantic_clusterer/` should ship with a matching
  test file under the equivalent `tests/semantic_clusterer/...` path.
