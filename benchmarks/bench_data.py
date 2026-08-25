"""Gold-standard dataset loaders for the semantic-clusterer benchmark.

Two labelled datasets, both with ground-truth topic labels:

  - ``20ng``   : 20 Newsgroups (via scikit-learn, no extra dependency)
  - ``agnews`` : AG News (via HuggingFace ``datasets``, optional install)

Each loader returns ``(texts, labels, target_names)`` and supports stratified
subsampling to a target size, so you can deliberately land in a given pipeline
tier (tiny / small / medium / large) for calibration.

NOTE: this module is intentionally named ``bench_data`` (not ``datasets``) so
it never shadows the HuggingFace ``datasets`` package on the import path.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np


def _stratified_subsample(
    texts: Sequence[str],
    labels: np.ndarray,
    n_docs: Optional[int],
    seed: int,
) -> Tuple[List[str], List[int]]:
    """Take a class-proportional subsample of ``n_docs`` rows.

    Returns the full dataset unchanged when ``n_docs`` is None or larger than
    the corpus. Sampling is deterministic for a fixed seed.
    """
    labels = np.asarray(labels)
    n_total = len(texts)
    if n_docs is None or n_docs >= n_total:
        return list(texts), labels.tolist()

    rng = np.random.default_rng(seed)
    chosen: List[int] = []
    for cls in np.unique(labels):
        idx = np.where(labels == cls)[0]
        share = max(1, round(n_docs * len(idx) / n_total))
        share = min(share, len(idx))
        chosen.extend(rng.choice(idx, size=share, replace=False).tolist())

    rng.shuffle(chosen)
    chosen = chosen[:n_docs]
    sub_texts = [texts[i] for i in chosen]
    sub_labels = labels[chosen].tolist()
    return sub_texts, sub_labels


def load_20ng(
    n_docs: Optional[int] = None,
    seed: int = 42,
    categories: Optional[Sequence[str]] = None,
) -> Tuple[List[str], List[int], List[str]]:
    """Load 20 Newsgroups with headers/footers/quotes removed.

    Stripping headers/footers/quotes is standard research practice: it removes
    metadata that would otherwise leak the label, so the score reflects real
    semantic clustering rather than header matching.
    """
    from sklearn.datasets import fetch_20newsgroups

    bunch = fetch_20newsgroups(
        subset="all",
        categories=list(categories) if categories else None,
        remove=("headers", "footers", "quotes"),
        random_state=seed,
    )
    texts = [t.strip() for t in bunch.data]
    labels = np.asarray(bunch.target)

    # Some docs become empty once headers/footers/quotes are removed.
    keep = [i for i, t in enumerate(texts) if t]
    texts = [texts[i] for i in keep]
    labels = labels[keep]

    texts, labels = _stratified_subsample(texts, labels, n_docs, seed)
    return texts, labels, list(bunch.target_names)


def load_agnews(
    n_docs: Optional[int] = None,
    seed: int = 42,
) -> Tuple[List[str], List[int], List[str]]:
    """Load AG News (4 classes) via the HuggingFace ``datasets`` package.

    Optional dependency: ``pip install datasets``.
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:  # noqa: BLE001
        raise ImportError(
            "AG News requires the HuggingFace 'datasets' package. "
            "Install it with: pip install datasets"
        ) from exc

    ds = load_dataset("ag_news", split="train")
    texts = [str(t).strip() for t in ds["text"]]
    labels = np.asarray(ds["label"])

    keep = [i for i, t in enumerate(texts) if t]
    texts = [texts[i] for i in keep]
    labels = labels[keep]

    texts, labels = _stratified_subsample(texts, labels, n_docs, seed)
    return texts, labels, ["World", "Sports", "Business", "Sci/Tech"]


DATASETS = {
    "20ng": load_20ng,
    "agnews": load_agnews,
}


def load_benchmark_dataset(
    name: str,
    n_docs: Optional[int] = None,
    seed: int = 42,
) -> Tuple[List[str], List[int], List[str]]:
    """Dispatch to a named dataset loader."""
    if name not in DATASETS:
        raise ValueError(
            f"Unknown dataset {name!r}. Choices: {sorted(DATASETS)}"
        )
    return DATASETS[name](n_docs=n_docs, seed=seed)
