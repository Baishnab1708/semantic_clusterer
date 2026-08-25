"""Example 08 — fit / predict / save / load (v0.1.0)

Production workflow: train once on a corpus, persist the model, then serve
predictions on new texts from a separate process — for both SemanticClusterer
and SemanticKSplit.

Run:
    python examples/08_fit_predict_save_load.py
"""

import os
import shutil
import tempfile

from semantic_clusterer import SemanticClusterer, SemanticKSplit

TRAIN_TEXTS = [
    "How do I reset my password?",
    "I forgot my password and cannot login",
    "My account is locked out",
    "When are you open?",
    "What are your business hours today?",
    "Are you open on weekends?",
    "My package has not arrived",
    "Where is my order?",
    "The shipment is delayed",
    "I want a refund for my order",
    "How do I return this product?",
    "What is your refund policy?",
]

NEW_TEXTS = [
    "I cannot remember my password",        # → password cluster
    "What time do you close on Sunday?",    # → hours cluster
    "Track my missing delivery",            # → shipping cluster
    "Random unrelated nonsense input xyz",  # → OOD (below threshold)
]


def section(title: str) -> None:
    print(f"\n{'='*55}")
    print(f"  {title}")
    print("=" * 55)


def demo_semantic_clusterer() -> None:
    section("SemanticClusterer — fit / predict / save / load")

    # ── Train ─────────────────────────────────────────────────────────
    sc = SemanticClusterer(cluster_granularity="balanced", random_state=42)
    sc.fit(TRAIN_TEXTS)

    print(f"\nTrained: {sc._fitted_state.n_clusters} clusters")
    print(f"Auto outlier threshold: {sc.outlier_threshold:.4f}  (calibrated from training data)")

    print("\nTopic labels:")
    for cid, label in sorted(sc.get_topic_labels().items()):
        print(f"  cluster {cid}: {label}")

    print("\nCluster cohesion stats:")
    for s in sc.cluster_stats:
        print(f"  cluster {s['cluster_id']:>2}: size={s['size']:>3}  "
              f"min={s['min_sim']:.3f}  mean={s['mean_sim']:.3f}  p10={s['p10_sim']:.3f}")

    # ── Predict (auto threshold) ───────────────────────────────────────
    print(f"\npredict() — auto threshold={sc.outlier_threshold:.3f}  (default):")
    labels = sc.predict(NEW_TEXTS)
    for text, label in zip(NEW_TEXTS, labels):
        tag = "OOD" if label == -1 else f"cluster {label}"
        print(f"  {tag:>12}: {text}")

    # ── Predict (no OOD filtering) ─────────────────────────────────────
    print("\npredict() — outlier_threshold=None  (always assign):")
    labels_no_ood = sc.predict(NEW_TEXTS, outlier_threshold=None)
    for text, label in zip(NEW_TEXTS, labels_no_ood):
        print(f"  cluster {label:>2}: {text}")

    # ── Save / load ────────────────────────────────────────────────────
    tmpdir = tempfile.mkdtemp(prefix="semclust_demo_")
    try:
        path = os.path.join(tmpdir, "model")
        sc.save(path)
        print(f"\nSaved: {sorted(os.listdir(path))}")

        loaded = SemanticClusterer.load(path)
        print(f"Loaded: {loaded._fitted_state.n_clusters} clusters, "
              f"threshold={loaded.outlier_threshold:.4f}")

        reload_labels = loaded.predict(NEW_TEXTS)
        print(f"predict() identical after save/load: {(reload_labels == labels).all()}")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def demo_semantic_ksplit() -> None:
    section("SemanticKSplit — fit / predict / save / load")

    # ── Train ─────────────────────────────────────────────────────────
    ks = SemanticKSplit(k=3, quality="balanced", random_state=42)
    ks.fit(TRAIN_TEXTS)

    print(f"\nTrained: {ks._fitted_state.n_clusters} clusters (requested k=3)")
    print(f"Auto outlier threshold: {ks.outlier_threshold:.4f}")

    print("\nTopic labels:")
    for cid, label in sorted(ks.get_topic_labels().items()):
        print(f"  cluster {cid}: {label}")

    # ── Predict ────────────────────────────────────────────────────────
    print(f"\npredict() — auto threshold:")
    labels = ks.predict(NEW_TEXTS)
    for text, label in zip(NEW_TEXTS, labels):
        tag = "OOD" if label == -1 else f"cluster {label}"
        print(f"  {tag:>12}: {text}")

    # ── Save / load ────────────────────────────────────────────────────
    tmpdir = tempfile.mkdtemp(prefix="semksplit_demo_")
    try:
        path = os.path.join(tmpdir, "model")
        ks.save(path)
        print(f"\nSaved: {sorted(os.listdir(path))}")

        loaded = SemanticKSplit.load(path)
        reload_labels = loaded.predict(NEW_TEXTS)
        print(f"predict() identical after save/load: {(reload_labels == labels).all()}")

        # Cross-class load raises ValueError
        try:
            SemanticClusterer.load(path)
        except ValueError as e:
            print(f"Cross-class load correctly raises: {e}")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    demo_semantic_clusterer()
    demo_semantic_ksplit()
    print("\nAll examples completed.")
