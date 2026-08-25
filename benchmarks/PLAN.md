# Benchmarks Plan

The goal of this folder is to **earn developer trust** before release. Anyone
who lands here should be able to:

1. See, in 30 seconds, how `semantic-clusterer` performs on standard public
   benchmarks against well-known baselines.
2. Reproduce every number on their own machine with one command.
3. Inspect the raw artifacts (per-run JSON) to verify we are not cherry-picking.
4. Trust that the numbers are deterministic, version-pinned, and honest about
   our weaknesses as well as our strengths.

This is not a marketing exercise. It is a verifiable evidence pack.

---

## Guiding Principles

1. **Reproducibility first.** Every number in the published table is tied to
   a JSON record containing seed, package versions, hardware, and config. If
   it can't be reproduced, it doesn't ship.
2. **Honest comparisons.** Same embedder, same data split, same metric
   definitions across all baselines. We report wins and losses.
3. **Multiple datasets, multiple embedders.** Single-dataset benchmarks are
   suspicious; we run a matrix.
4. **Both clustering modes.** Variable-K (`SemanticClusterer`) and fixed-K
   (`SemanticKSplit` with k = number of ground-truth classes).
5. **Cheap to rerun.** A "smoke" subset runs in CI in under 5 minutes.
6. **Public datasets only.** No private corpora, no synthetic-only results.

---

## Folder Layout

```
benchmarks/
├── README.md                  # landing page with the headline results table
├── PLAN.md                    # this file
├── METHODOLOGY.md             # how we measure, why these metrics, fairness rules
│
├── datasets/                  # public dataset loaders, all deterministic
│   ├── __init__.py
│   ├── loader.py              # unified interface, caches to ~/.cache/sc_benchmarks
│   ├── newsgroups20.py        # 20 Newsgroups — 18 846 docs, 20 classes
│   ├── ag_news.py             # AG News — 127 600 train, 4 classes, short headlines
│   ├── trec.py                # TREC — 5 952 short questions, 6 coarse classes
│   └── dbpedia.py             # DBpedia 14, 50K subsample (optional, medium tier)
│
├── embedders/                 # interchangeable embedder backends
│   ├── __init__.py
│   ├── builtin_onnx.py        # our default ONNX MiniLM-L6-v2 (384-dim)
│   ├── st_mpnet.py            # sentence-transformers/all-mpnet-base-v2 (768)
│   └── openai_small.py        # text-embedding-3-small (1536), skipped if no key
│
├── baselines/                 # all baselines share one ABC
│   ├── __init__.py
│   ├── base.py                # Baseline ABC: name, fit(emb, k=None), predict(emb)
│   ├── ours_clusterer.py      # SemanticClusterer (variable K)
│   ├── ours_ksplit.py         # SemanticKSplit(k=n_classes)
│   ├── sklearn_kmeans.py      # sklearn KMeans(n_clusters=k)
│   ├── sklearn_agglo.py       # sklearn AgglomerativeClustering(n_clusters=k)
│   ├── hdbscan_raw.py         # HDBSCAN on same embeddings, no post-processing
│   ├── bertopic.py            # BERTopic (skipped if not installed)
│   └── top2vec.py             # Top2Vec (skipped if not installed)
│
├── metrics/
│   ├── __init__.py
│   ├── supervised.py          # NMI, ARI, V-measure, homogeneity, completeness, purity
│   ├── intrinsic.py           # silhouette and Davies–Bouldin on embeddings
│   └── runtime.py             # wall-clock, peak RSS via psutil
│
├── results/
│   ├── .gitkeep
│   ├── runs/                  # one JSON per (dataset, embedder, baseline, seed)
│   ├── summary.json           # aggregated table
│   ├── summary.md             # human-readable, copied into benchmarks/README.md
│   ├── machine_info.json      # CPU, RAM, OS captured by run_all
│   └── plots/                 # optional bar charts per metric
│
├── scripts/
│   ├── run_all.py             # full matrix: dataset × embedder × baseline × seed
│   ├── run_one.py             # one cell of the matrix, useful for debugging
│   ├── aggregate.py           # results/runs/*.json → summary.{json,md}
│   ├── render_plots.py        # matplotlib bar charts of NMI/ARI per dataset
│   └── verify_determinism.py  # runs each cell 3× with same seed, asserts identical labels
│
├── config.yaml                # which datasets / embedders / baselines / seeds to run
└── tests/
    └── test_smoke.py          # CI-friendly: 1 000-row subsample of 20news, asserts NMI ≥ floor
```

---

## What the Headline Table Will Look Like

The benchmarks/README.md will open with a table like this (numbers populated
after the first full run). Both modes are reported so readers see exactly how
the variable-K and fixed-K APIs perform.

### 20 Newsgroups (k=20, all-MiniLM-L6-v2 embeddings)

| Method                    | NMI ↑ | ARI ↑ | V-measure ↑ | n_clusters | Noise % | Fit time (s) |
|---------------------------|-------|-------|-------------|------------|---------|--------------|
| sklearn KMeans            | …     | …     | …           | 20         | 0.0%    | …            |
| sklearn Agglomerative     | …     | …     | …           | 20         | 0.0%    | …            |
| HDBSCAN (raw)             | …     | …     | …           | …          | …       | …            |
| BERTopic                  | …     | …     | …           | …          | …       | …            |
| Top2Vec                   | …     | …     | …           | …          | …       | …            |
| **SemanticClusterer**     | **…** | **…** | **…**       | …          | …       | …            |
| **SemanticKSplit (k=20)** | **…** | **…** | **…**       | 20         | 0.0%    | …            |

Same table for AG News (k=4) and TREC (k=6).

A second table reports the **multi-embedder matrix** — same baseline
(ours), three embedders (MiniLM, MPNet, OpenAI). This shows the dim-band
system actually pays off.

A third section is a **runtime/cost table** showing fit time, predict time
per 1 000 items (after `fit/predict` lands), and peak memory.

---

## Datasets — The Why

### 20 Newsgroups (primary)
- **Why it's the standard.** Used by every clustering paper since 2010.
  Comparable numbers exist everywhere.
- **Size.** ~18 846 documents → exercises the *medium* pipeline tier.
- **Difficulty.** 20 fine-grained classes with semantic overlap (e.g.
  `talk.religion.misc` vs `soc.religion.christian`) — discriminates good
  embeddings from poor ones.
- **Preprocessing.** Use sklearn's `remove=('headers', 'footers', 'quotes')`
  to prevent metadata leakage, the standard fairness setting.

### AG News (primary)
- **Why.** Industrial-scale (127 600 train), short headlines, only 4 classes.
  Very different shape from 20news.
- **Size.** Exercises the *large* pipeline tier and the
  `MiniBatchKMeans → per-shard HDBSCAN` path.
- **Difficulty.** Easy (4 well-separated topics) but tests **scalability**, not
  classification accuracy. We expect high NMI here; the interesting numbers
  are runtime and memory.

### TREC question classification (secondary)
- **Why.** ~5 952 short questions with 6 classes. Exercises the *small* tier
  and tests behaviour on **short inputs** where embeddings are noisy.
- **Size.** Small enough to run the full matrix in seconds.

### DBpedia 14, 50K subsample (optional)
- **Why.** A different domain (Wikipedia entities), 14 classes, gives a third
  data point for the medium tier.

For each dataset we record `n_classes` (= ground-truth k), source URL, and a
SHA256 of the cached file in `runs/*.json` so the dataset version is pinned.

---

## Embedders — The Why

Three is the minimum for a credible matrix:

1. **Built-in ONNX MiniLM-L6-v2 (384-dim, low band).** Default. Tests the
   out-of-the-box experience.
2. **`sentence-transformers/all-mpnet-base-v2` (768-dim, mid band).** A
   stronger but heavier model; tests that the dim-band system handles the
   step up without retuning.
3. **OpenAI `text-embedding-3-small` (1536-dim, high band).** Production-grade.
   Skipped automatically when `OPENAI_API_KEY` is unset.

We deliberately do **not** include the largest models (text-embedding-3-large,
BGE-large) in the headline run to keep cost bounded. They can be enabled via
`config.yaml`.

---

## Baselines — The Why

Each baseline answers a specific question.

| Baseline                  | Question it answers                                      |
|---------------------------|----------------------------------------------------------|
| sklearn KMeans            | Are we better than the cheapest fixed-K baseline?        |
| sklearn Agglomerative     | Are we better than a cheap hierarchical fixed-K?         |
| HDBSCAN (raw embeddings)  | Does our pipeline add value over plain HDBSCAN?          |
| BERTopic                  | Are we competitive with the dominant text-clustering lib?|
| Top2Vec                   | Same question, second framework                          |
| **SemanticClusterer**     | Our variable-K story                                     |
| **SemanticKSplit (k=k*)** | Our fixed-K story                                        |

All baselines run on **the same embeddings**, so we measure clustering
quality, not embedding quality. BERTopic is configured to use the same
embedder we use (we override its default).

---

## Metrics — The Why

### Supervised (primary — we have ground-truth labels)

| Metric         | Range         | Why we report it                                 |
|----------------|---------------|--------------------------------------------------|
| **NMI**        | `[0, 1]`      | Symmetric, permutation-invariant. Headline number. |
| **ARI**        | `[-1, 1]`     | Chance-corrected. Penalises trivial / degenerate clusterings. |
| **V-measure**  | `[0, 1]`      | Decomposes into homogeneity + completeness; useful for diagnosing failure mode. |
| **Purity**     | `[0, 1]`      | Intuitive. Acknowledged to be biased toward many clusters; reported for transparency, not as a primary metric. |

### Variable-K specific

- **`n_clusters_found`** — how many clusters the method discovered.
- **`abs(n_clusters_found - n_classes)`** — distance from the truth.
- **`noise_ratio`** — fraction with label `-1`. Methods that dump 90% to noise
  and get high NMI on the remaining 10% are flagged.

### Intrinsic (secondary — sanity check, no labels involved)

- **silhouette** on raw embeddings.
- **Davies–Bouldin** on raw embeddings.

These are reported but never used to rank methods because they are sensitive
to cluster count.

### Runtime

- **fit_seconds** — wall clock for the clustering call.
- **predict_seconds_per_1k** — once `fit/predict` lands, time to assign
  1 000 new texts.
- **peak_rss_mb** — peak resident set size from `psutil`.
- **embedding_seconds** — separated from clustering so different embedders
  can be compared fairly.

---

## Reproducibility Contract

Every run JSON contains:

```json
{
  "timestamp_utc": "2025-...",
  "dataset": {"name": "20news", "n_docs": 18846, "n_classes": 20, "sha256": "..."},
  "embedder": {"name": "builtin-onnx", "model": "all-MiniLM-L6-v2", "dim": 384},
  "baseline": {"name": "SemanticClusterer", "version": "0.2.0", "params": {...}},
  "seed": 42,
  "metrics": {
    "nmi": 0.512, "ari": 0.387, "v_measure": 0.519,
    "homogeneity": 0.534, "completeness": 0.504, "purity": 0.611,
    "n_clusters_found": 22, "noise_ratio": 0.04,
    "silhouette": 0.082, "davies_bouldin": 2.31,
    "fit_seconds": 18.4, "embedding_seconds": 12.1, "peak_rss_mb": 1240
  },
  "chosen_params": {...},        // copied from ClusteringReport when applicable
  "machine": {"os": "Linux", "cpu": "...", "python": "3.11.5"},
  "library_versions": {
    "semantic-clusterer": "0.2.0",
    "numpy": "1.26.0", "scikit-learn": "1.3.2",
    "hdbscan": "0.8.33", "umap-learn": "0.5.5",
    "onnxruntime": "1.16.3"
  }
}
```

To reproduce, a user runs:

```bash
pip install -e ".[bench]"
python benchmarks/scripts/run_all.py --seed 42
```

…and gets byte-identical NMI / ARI / V-measure values within the determinism
scope already documented in `WORKINGS.md`.

`scripts/verify_determinism.py` runs each cell three times with the same
seed and asserts the labels are identical. CI runs it on the smoke subset.

---

## Honesty Mechanisms

These are deliberate to avoid the appearance of cherry-picking.

1. **`results/runs/` is gitignored, but `summary.json` is committed.** Anyone
   can rerun and diff their results against ours.
2. **A `LOSSES.md` file** in `results/` enumerates every cell where a
   competitor beat us, with the gap. We update it after every run. A
   library that pretends it never loses is not credible.
3. **No selective seeds.** We report mean ± std over **3 seeds** (`42, 0, 7`)
   in the headline table. No "best of N."
4. **Pinned versions.** `pyproject.toml` adds a `[bench]` extra with exact
   versions of every baseline. A version drift in BERTopic should not silently
   change our numbers.
5. **Statistical significance note.** Where two methods are within `0.005`
   NMI, we say "tied" rather than ranking them.

---

## What Each Script Does (One-Line Summary)

| Script                          | Purpose                                                              |
|---------------------------------|----------------------------------------------------------------------|
| `scripts/run_all.py`            | Run the full matrix from `config.yaml`, write per-run JSON.          |
| `scripts/run_one.py`            | Run a single (dataset, embedder, baseline, seed) cell — for debug.   |
| `scripts/aggregate.py`          | Read `results/runs/*.json`, write `summary.json` and `summary.md`.   |
| `scripts/render_plots.py`       | Bar charts per dataset for NMI/ARI/runtime → `results/plots/*.png`.  |
| `scripts/verify_determinism.py` | Run each cell 3× and assert identical labels.                        |
| `tests/test_smoke.py`           | CI-friendly subset, asserts headline NMI does not regress.           |

---

## CI Integration

A new GitHub Actions job, `bench-smoke.yml`, runs on every PR:

- Loads a 1 000-doc subsample of 20 Newsgroups.
- Runs only `SemanticClusterer` + `SemanticKSplit(k=20)` + `sklearn KMeans` (skip BERTopic to keep CI fast).
- Asserts `NMI(SemanticClusterer) >= 0.40` and `NMI(SemanticKSplit) >= 0.45`
  (floors calibrated from the first full run, with a 0.05 buffer).
- Saves `summary.md` as a workflow artifact.

Floors prevent silent regressions. They are not aspirational targets — they
are tripwires.

The full benchmark (all datasets, all embedders, all baselines, 3 seeds)
runs nightly on a self-hosted runner or weekly on a manual `workflow_dispatch`
to keep CI minutes bounded.

---

## Documentation Touch Points

1. **Top-level `README.md`** gets a new "Benchmarks" section linking here,
   with the headline table copied from `benchmarks/results/summary.md`.
2. **`benchmarks/README.md`** is the landing page — the headline table plus a
   "How to reproduce" block.
3. **`benchmarks/METHODOLOGY.md`** explains every metric formally. Anyone
   skeptical of a number can verify the definition there.
4. The benchmarks/README.md ends with a "When NOT to use semantic-clusterer"
   section pointing to the `LOSSES.md` evidence. Honest engineers trust libraries
   that document their failure modes.

---

## Phased Rollout

Doing all of this at once is too much. Suggested phases:

### Phase 1 — Foundation (1 day)
- Folder structure, `METHODOLOGY.md`, `config.yaml`.
- `datasets/loader.py`, `datasets/newsgroups20.py`, `datasets/ag_news.py`.
- `metrics/supervised.py`, `metrics/runtime.py`.
- `baselines/base.py`, `baselines/ours_clusterer.py`, `baselines/ours_ksplit.py`,
  `baselines/sklearn_kmeans.py`.
- `scripts/run_one.py` working end-to-end on 20news + MiniLM + ours.

### Phase 2 — Matrix expansion (1 day)
- Remaining baselines: `sklearn_agglo`, `hdbscan_raw`, `bertopic`, `top2vec`.
- Remaining datasets: `trec`, `dbpedia`.
- Remaining embedders: `st_mpnet`, `openai_small`.
- `scripts/run_all.py`, `scripts/aggregate.py`.

### Phase 3 — Trust signals (half day)
- `scripts/verify_determinism.py`.
- `LOSSES.md` template.
- `tests/test_smoke.py`.
- CI workflow `bench-smoke.yml`.
- `scripts/render_plots.py`.

### Phase 4 — Publish (half day)
- Run Phase 1+2+3 with three seeds.
- Generate `summary.md`, copy headline tables into top-level `README.md`.
- Push tagged release.

Total: 3 working days end-to-end.

---

## Acceptance Criteria for "Trust Pack Complete"

Before tagging the release:

- [ ] Headline table populated for 20news, AG News, TREC.
- [ ] Three embedders covered (built-in + MPNet at minimum; OpenAI when key available).
- [ ] At least three external baselines covered (KMeans, HDBSCAN raw, one of BERTopic/Top2Vec).
- [ ] Three seeds, mean ± std reported.
- [ ] `verify_determinism.py` passes for every cell.
- [ ] `LOSSES.md` exists and lists every cell where a competitor wins.
- [ ] `tests/test_smoke.py` runs in CI in under 5 minutes.
- [ ] `summary.md` regenerates cleanly from `results/runs/*.json`.
- [ ] Reproduction instructions in `benchmarks/README.md` work on a fresh clone.

When every box is ticked, the release is shippable on the trust dimension.
