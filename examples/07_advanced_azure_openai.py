"""Advanced: Azure OpenAI text-embedding-3-small + full pipeline control.

Demonstrates using a production-grade embedding model via LangChain's
AzureOpenAIEmbeddings with both SemanticClusterer and SemanticKSplit,
plus inspecting the full ClusteringReport.

Prerequisites:
    pip install langchain-openai python-dotenv

Environment variables (set in .env or your shell):
    AZURE_OPENAI_ENDPOINT
    AZURE_OPENAI_API_KEY
    AZURE_OPENAI_API_VERSION
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT   (e.g. "text-embedding-3-small")

Run:
    python examples/07_advanced_azure_openai.py
"""

import os
import sys

from dotenv import load_dotenv

load_dotenv()

# Validate environment before doing any work.
_REQUIRED = [
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_API_VERSION",
    "AZURE_OPENAI_EMBEDDING_DEPLOYMENT",
]
_missing = [k for k in _REQUIRED if not os.environ.get(k)]
if _missing:
    print(f"Missing environment variables: {', '.join(_missing)}")
    print("Set them in a .env file or your shell and re-run.")
    sys.exit(1)

from langchain_openai import AzureOpenAIEmbeddings

from semantic_clusterer import ClustererConfig, SemanticClusterer, SemanticKSplit

# ---------------------------------------------------------------------------
# Build the embedder.
# The LangChain AzureOpenAIEmbeddings class exposes embed_documents()
# which the library auto-detects and wraps. No adapter needed.
#
# text-embedding-3-small produces 1536-dim vectors (dim band: "high").
# The pipeline automatically selects band-appropriate PCA / UMAP / HDBSCAN
# grids — no manual tuning required.
# ---------------------------------------------------------------------------
embedder = AzureOpenAIEmbeddings(
    azure_deployment=os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT"],
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    api_version=os.environ["AZURE_OPENAI_API_VERSION"],
)

texts = [
    # Password / auth
    "How do I reset my password?",
    "I forgot my login credentials",
    "My account is locked after too many attempts",
    "Where is the password reset link?",
    # Billing
    "I was charged twice for the same order",
    "When will my refund appear on my card?",
    "How do I update my payment method?",
    "My invoice shows the wrong amount",
    # Delivery
    "My package has not arrived",
    "Where is my order right now?",
    "The delivery was marked as complete but nothing arrived",
    "Can I change my shipping address after ordering?",
    # Technical issues
    "The app crashes when I try to open it",
    "I cannot log in on the mobile app",
    "The website gives me a 500 error",
    "Video playback is buffering constantly",
]

# ---------------------------------------------------------------------------
# Part 1: Variable-K clustering with SemanticClusterer.
# Force the small strategy since N=16, but use verbose to see every phase.
# ---------------------------------------------------------------------------
print("=" * 60)
print("Part 1: SemanticClusterer (variable K)")
print("=" * 60)

config = ClustererConfig(
    normalize_embeddings=True,
    verbose=True,
    random_state=42,
)

clusterer = SemanticClusterer(embedding_model=embedder, config=config)
labels, report = clusterer.cluster_with_report(texts)

print(f"\nRun summary")
print(f"  inputs:         {report.n_input_texts}")
print(f"  clusters found: {report.n_clusters}")
print(f"  noise (-1):     {report.n_noise}")
print(f"  pipeline_tier:  {report.pipeline_tier}")
print(f"  dim_band:       {report.dim_band}  (D={report.embedding_dim})")
print(f"  confidence:     {report.confidence_level}")
print(f"  random_state:   {report.random_state}")

if report.warnings:
    print(f"\n  warnings: {report.warnings}")

clusters = clusterer.cluster(texts, return_format="detailed")
print(f"\nDetailed clusters:")
for c in clusters:
    print(f"\n  [{c['cluster_id']}] {c['representative']}  "
          f"(size={c['size']}, confidence={c['confidence']:.3f})")
    for item in c["items"]:
        print(f"      - {item}")

# ---------------------------------------------------------------------------
# Part 2: Fixed-K partitioning with SemanticKSplit.
# Force exactly 4 groups matching the 4 support topics above.
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Part 2: SemanticKSplit (fixed K=4)")
print("=" * 60)

ks = SemanticKSplit(
    embedding_model=embedder,
    k=4,
    config={"normalize_embeddings": True},
    verbose=True,
    random_state=42,
)

labels, report = ks.split_with_report(texts)

print(f"\nRun summary")
print(f"  requested_k:    {report.chosen_params.get('requested_k')}")
print(f"  algorithm_used: {report.chosen_params.get('algorithm_used')}")
print(f"  pipeline_tier:  {report.pipeline_tier}")
print(f"  dim_band:       {report.dim_band}  (D={report.embedding_dim})")
print(f"  confidence:     {report.confidence_level}")
print(f"  per_cluster_size:     {report.intrinsic_metrics.get('per_cluster_size')}")
print(f"  per_cluster_cohesion: "
      f"{[round(v, 3) for v in report.intrinsic_metrics.get('per_cluster_cohesion', [])]}")
print(f"  silhouette:     {report.intrinsic_metrics.get('silhouette', 'n/a'):.4f}")

groups = ks.split(texts, return_format="detailed")
print(f"\nPartitions:")
for c in groups:
    print(f"\n  [{c['cluster_id']}] {c['representative']}  "
          f"(size={c['size']}, confidence={c['confidence']:.3f})")
    for item in c["items"]:
        print(f"      - {item}")

# ---------------------------------------------------------------------------
# Part 3: Row-aligned labels (useful for DataFrame joins).
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Part 3: split_labels() for DataFrame integration")
print("=" * 60)

labels = ks.split_labels(texts)
print("\ntext → label")
for text, lbl in zip(texts, labels):
    print(f"  [{lbl}]  {text}")
