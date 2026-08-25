# Benchmarks

Honest, reproducible benchmarks for `semantic-clusterer` on **gold-standard,
labelled datasets** — the corpora topic-modelling tools like BERTopic are
evaluated on. Every number is reproducible with one command, across three
embedding models and all four pipeline tiers.

Two things are measured per run:

1. **External quality** (needs ground-truth labels) — *did we recover the real
   topics?* Reported as ARI, NMI, V-measure, homogeneity, completeness.
2. **Intrinsic score** — the library's own internal composite (cohesion,
   coverage, separation, …) read straight from the run report. This is what the
   runtime quality floor gates against, so we calibrate it from these runs.

> External metrics are computed **only over rows the library actually
> clustered** (label `>= 0`). Noise rows (`-1`) are excluded from the matched
> comparison and reported separately as `coverage` / `noise_ratio`, so a model
> is neither rewarded for refusing to cluster nor punished for honest noise
> detection.

---

## The three embedders

| Alias | Model | Dim | Band | Cost |
|-------|-------|-----|------|------|
| `minilm` | all-MiniLM-L6-v2 (built-in ONNX) | 384 | low | free, local CPU |
| `mpnet` | all-mpnet-base-v2 (sentence-transformers) | 768 | mid | free; downloads ~420 MB once |
| `openai3small` | text-embedding-3-small (Azure OpenAI) | 1536 | high | **PAID API** |

## Tier → dataset routing

A single `--tiers` flag drives each run. Each tier maps to a concrete
(dataset, size). 20 Newsgroups has only ~18k docs so it cannot reach the large
tier; AG News (120k) covers it.

| Tier | Dataset | Docs | Why |
|------|---------|------|-----|
| tiny | 20 Newsgroups | 120 | exact agglomerative path |
| small | 20 Newsgroups | 1 500 | UMAP + HDBSCAN sweep |
| medium | 20 Newsgroups | 15 000 | profiled sweep |
| large | AG News | 60 000 | shard → cluster → stitch |

20NG is loaded with headers/footers/quotes stripped (standard practice, no
label leakage) and stratified-subsampled deterministically.

---

## Install

```bash
pip install -r benchmarks/requirements.txt
```

`minilm` runs need nothing extra. `mpnet` needs `sentence-transformers`,
`openai3small` needs `langchain-openai` + `python-dotenv` and these `.env`
vars: `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`,
`AZURE_OPENAI_API_VERSION`, `AZURE_OPENAI_EMBEDDING_DEPLOYMENT`. The large tier
needs `datasets` (AG News).

---

## The 6 commands

Run from the **repository root**, one at a time. `openai3small` skips `large`
because it is expensive (billed per token). Each command overwrites its own
results file, so they are safe to re-run.

```bash
# ---- SemanticClusterer (variable-K) ----
# 1) MiniLM — all four tiers
python benchmarks/run_clusterer.py --embedder minilm       --tiers tiny small medium large

# 2) mpnet — all four tiers
python benchmarks/run_clusterer.py --embedder mpnet        --tiers tiny small medium large

# 3) text-embedding-3-small — tiny/small/medium only (skip large, costly)
python benchmarks/run_clusterer.py --embedder openai3small --tiers tiny small medium

# ---- SemanticKSplit (fixed-K = number of true classes) ----
# 4) MiniLM — all four tiers
python benchmarks/run_ksplit.py    --embedder minilm       --tiers tiny small medium large

# 5) mpnet — all four tiers
python benchmarks/run_ksplit.py    --embedder mpnet        --tiers tiny small medium large

# 6) text-embedding-3-small — tiny/small/medium only (skip large, costly)
python benchmarks/run_ksplit.py    --embedder openai3small --tiers tiny small medium

# ---- Production API Evaluation (Fit 80% -> Predict 20% with Auto Outlier Threshold) ----
python benchmarks/run_production_eval.py --embedder minilm       --tiers tiny small medium
python benchmarks/run_production_eval.py --embedder mpnet        --tiers tiny small medium
python benchmarks/run_production_eval.py --embedder openai3small --tiers tiny small medium
```

Outputs:

```
results/clusterer_minilm.json            results/ksplit_minilm.json
results/clusterer_mpnet.json             results/ksplit_mpnet.json
results/clusterer_openai3small.json      results/ksplit_openai3small.json
results/production_clusterer_minilm.json results/production_ksplit_minilm.json
results/production_clusterer_mpnet.json  results/production_ksplit_mpnet.json
results/production_clusterer_openai3small.json results/production_ksplit_openai3small.json
```

---

## Benchmark Results & BERTopic Comparison

Evaluated on 20 Newsgroups ($K=20$ ground-truth classes):

| Tier | Dataset Scale | BERTopic Baseline ARI | `SemanticClusterer` Best ARI | `SemanticKSplit` Best ARI | Winner |
|---|:---:|:---:|:---:|:---:|:---:|
| **Tiny** | $N=116$ | 0.1671 | **0.2010** *(OpenAI)* | **0.2918** *(OpenAI)* | 🏆 `semantic_clusterer` (+20.3%) |
| **Small** | $N=1,500$ | 0.4435 | **0.4922** *(OpenAI)* | 0.4186 *(OpenAI)* | 🏆 `semantic_clusterer` (+11.0%) |
| **Medium** | $N=15,000$ | 0.4246 | **0.4561** *(OpenAI)* | **0.4759** *(OpenAI)* | 🏆 `semantic_clusterer` (+7.4%) |

---

## After the runs — calibrate the quality floor

```bash
python benchmarks/calibrate_baseline.py \
    --results clusterer_minilm.json clusterer_mpnet.json clusterer_openai3small.json
```

This pools the SemanticClusterer runs by **actual tier** across all three
embedders and proposes per-tier floors:

```
floor = min(intrinsic_score in tier across all embedders) * 0.75
```

- **`min`, not mean** — the floor must hold for the hardest embedder/corpus.
- **0.75 safety margin** — real user data is messier than any benchmark.
- **propose, don't overwrite** — it writes
  `results/baseline_scores.proposed.json` and prints the one-line copy command
  to promote it. The live file is never touched automatically.

The `score` / `ari` fields written into the baseline are informational; only
`floor` is read at runtime.

---

## File map

```
benchmarks/
  README.md               this file
  requirements.txt        deps for mpnet / openai3small / agnews
  bench_data.py           20NG + AG News loaders, stratified subsampling
  bench_metrics.py        ARI / NMI / V-measure / homogeneity / completeness
  bench_common.py         embedder registry, TIER_PLAN, results IO, tables
  bench_report.py         builds the detailed per-run record
  run_clusterer.py        benchmark SemanticClusterer across tiers
  run_ksplit.py           benchmark SemanticKSplit across tiers
  run_production_eval.py  benchmark train/test fit -> predict generalization
  calibrate_baseline.py   propose calibrated baseline_scores.json (no overwrite)
  results/                JSON outputs + the baseline proposal
```
