"""Shared helpers for the benchmark scripts.

  - EMBEDDER REGISTRY  : minilm / mpnet / openai3small
  - EMBEDDING CACHE    : openai3small embeddings saved to disk, reused on every
                         subsequent run — zero API calls after the first time.
  - TIER PLAN          : tier name -> (dataset, n_docs)
  - Results IO, timer, environment snapshot, table helpers.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

RESULTS_DIR = Path(__file__).parent / "results"
CACHE_DIR   = Path(__file__).parent / "cache"


# ---------------------------------------------------------------------------
# Tier plan
# ---------------------------------------------------------------------------

TIER_PLAN: Dict[str, Tuple[str, int]] = {
    "tiny":   ("20ng",   120),
    "small":  ("20ng",   1500),
    "medium": ("20ng",   15000),
    "large":  ("agnews", 60000),
}

ALL_TIERS = ["tiny", "small", "medium", "large"]


# ---------------------------------------------------------------------------
# Embedding cache (used by the paid OpenAI embedder)
# ---------------------------------------------------------------------------

def _cache_path(alias: str, dataset: str, n_docs: int, seed: int) -> Path:
    """Deterministic path for a cached embedding array.

    NOTE: ``n_docs`` should be the *actual* post-filter document count (i.e.
    ``len(texts)`` after dropping empty/blank rows), not the requested count
    passed to ``load_benchmark_dataset``.  The benchmark runner already does
    this correctly (``wrap_with_cache(..., len(texts), ...)``), but callers
    reusing this helper directly must pass the real count to ensure cache hits
    align with the data.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"emb__{alias}__{dataset}__n{n_docs}__seed{seed}.npy"


def cache_exists(alias: str, dataset: str, n_docs: int, seed: int) -> bool:
    return _cache_path(alias, dataset, n_docs, seed).exists()


def load_cache(alias: str, dataset: str, n_docs: int, seed: int) -> np.ndarray:
    return np.load(str(_cache_path(alias, dataset, n_docs, seed)))


def save_cache(
    alias: str, dataset: str, n_docs: int, seed: int, embeddings: np.ndarray
) -> None:
    np.save(str(_cache_path(alias, dataset, n_docs, seed)), embeddings)
    print(f"  [cache] saved {embeddings.shape} -> {_cache_path(alias, dataset, n_docs, seed).name}")


# ---------------------------------------------------------------------------
# OpenAI direct embedder (wraps openai Python SDK, batched)
# ---------------------------------------------------------------------------

class _OpenAIDirectEmbedder:
    """Calls OpenAI text-embedding-3-small directly via the openai SDK.

    Implements the .embed(texts, batch_size) interface the library expects.
    Batches requests to stay within the API's per-request token limit.
    """

    def __init__(self, api_key: str, model: str = "text-embedding-3-small"):
        import openai
        self._client = openai.OpenAI(api_key=api_key)
        self._model  = model

    def embed(self, texts: List[str], batch_size: int = 64, **_: Any) -> np.ndarray:
        # OpenAI text-embedding-3-small has an 8192-token limit per input.
        # ~6000 chars is a safe ceiling (avg ~1.3 chars/token for English).
        MAX_CHARS = 6000
        safe_texts = [t[:MAX_CHARS] if isinstance(t, str) else t for t in texts]

        all_vecs: List[List[float]] = []
        total = len(safe_texts)
        for start in range(0, total, batch_size):
            batch = safe_texts[start : start + batch_size]
            resp  = self._client.embeddings.create(input=batch, model=self._model)
            all_vecs.extend([item.embedding for item in resp.data])
            done = min(start + batch_size, total)
            print(f"  [openai] embedded {done}/{total}", end="\r", flush=True)
        print()
        return np.array(all_vecs, dtype=np.float32)


# ---------------------------------------------------------------------------
# Cached embedder wrapper
# ---------------------------------------------------------------------------

class _CachedEmbedder:
    """Wraps any embedder (including None = built-in ONNX) with a disk cache.

    Cache key: (alias, dataset, n_docs, seed) → one .npy file per combination.

    First run  : embeds via the inner model, saves to .npy, returns array.
    Future runs: loads from .npy, skips the inner model entirely — free.

    When inner is None (minilm built-in), the library's OnnxEmbedder is used
    transparently on the first run; the cache is still populated and reused.
    """

    def __init__(
        self,
        inner: Any,          # embedder object or None (built-in ONNX)
        alias: str,
        dataset: str,
        n_docs: int,
        seed: int,
    ):
        self._inner   = inner
        self._alias   = alias
        self._dataset = dataset
        self._n_docs  = n_docs
        self._seed    = seed
        self._resolved: Any = None   # lazily resolved built-in embedder

    def _get_inner(self) -> Any:
        """Return the real embedder, initialising the built-in one if needed."""
        if self._inner is not None:
            return self._inner
        if self._resolved is None:
            from semantic_clusterer.embedding.onnx_model import OnnxEmbedder
            self._resolved = OnnxEmbedder(batch_size=64, normalize=True, verbose=True)
        return self._resolved

    def embed(self, texts: List[str], batch_size: int = 64, **kwargs: Any) -> np.ndarray:
        if cache_exists(self._alias, self._dataset, self._n_docs, self._seed):
            arr = load_cache(self._alias, self._dataset, self._n_docs, self._seed)
            print(f"  [cache] loaded {arr.shape} — no model call needed")
            return arr
        print(f"  [cache] miss — embedding {len(texts)} texts and saving to disk...")
        inner = self._get_inner()

        # Call the inner model using whichever interface it exposes,
        # mirroring the library's own adapter detection order.
        if hasattr(inner, "encode"):
            # SentenceTransformers / HuggingFace
            raw = inner.encode(texts, batch_size=batch_size, show_progress_bar=False)
        elif hasattr(inner, "embed_documents"):
            # LangChain
            raw = inner.embed_documents(texts)
        elif hasattr(inner, "embed"):
            raw = inner.embed(texts, batch_size=batch_size)
        elif callable(inner):
            raw = inner(texts)
        else:
            raise RuntimeError(f"Cannot call inner embedder: {type(inner)}")

        arr = np.asarray(raw, dtype=np.float32)
        save_cache(self._alias, self._dataset, self._n_docs, self._seed, arr)
        return arr


def wrap_with_cache(
    embedder: Any,
    alias: str,
    dataset: str,
    n_docs: int,
    seed: int,
) -> _CachedEmbedder:
    """Wrap an embedder with the disk cache for a specific (dataset, n_docs, seed)."""
    return _CachedEmbedder(embedder, alias, dataset, n_docs, seed)


# ---------------------------------------------------------------------------
# Embedder builders
# ---------------------------------------------------------------------------

def _build_minilm():
    info = {
        "alias": "minilm",
        "name": "all-MiniLM-L6-v2 (built-in ONNX)",
        "expected_dim": 384,
        "dim_band": "low",
        "cost": "free (local CPU)",
        "cached": False,
    }
    return None, info   # None -> library uses its bundled ONNX model


def _build_mpnet():
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "mpnet needs sentence-transformers (downloads ~420 MB once). "
            "Install: pip install sentence-transformers"
        ) from exc
    model = SentenceTransformer("all-mpnet-base-v2")
    info = {
        "alias": "mpnet",
        "name": "all-mpnet-base-v2 (sentence-transformers)",
        "expected_dim": 768,
        "dim_band": "mid",
        "cost": "free (local; downloads model once)",
        "cached": False,
    }
    return model, info


def _build_openai3small():
    try:
        from dotenv import load_dotenv
        import openai  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "openai3small needs openai and python-dotenv. "
            "Install: pip install openai python-dotenv"
        ) from exc

    from dotenv import load_dotenv
    load_dotenv()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "openai3small requires OPENAI_API_KEY in your .env file.\n"
            "Add:  OPENAI_API_KEY=sk-..."
        )

    embedder = _OpenAIDirectEmbedder(api_key=api_key, model="text-embedding-3-small")
    info = {
        "alias": "openai3small",
        "name": "text-embedding-3-small (OpenAI direct)",
        "expected_dim": 1536,
        "dim_band": "high",
        "cost": "PAID API — billed per token (cached after first run)",
        "cached": True,   # signals runners to wrap with _CachedEmbedder
    }
    return embedder, info


_EMBEDDER_BUILDERS = {
    "minilm":                  _build_minilm,
    "builtin":                 _build_minilm,
    "onnx":                    _build_minilm,
    "mpnet":                   _build_mpnet,
    "all-mpnet-base-v2":       _build_mpnet,
    "openai3small":            _build_openai3small,
    "3small":                  _build_openai3small,
    "text-embedding-3-small":  _build_openai3small,
}


def build_embedder(alias: Optional[str]) -> Tuple[Any, Dict[str, Any]]:
    """Return (embedder_or_None, info_dict) for a registry alias."""
    key = (alias or "minilm").lower()
    if key in _EMBEDDER_BUILDERS:
        return _EMBEDDER_BUILDERS[key]()
    # Unknown -> try as a sentence-transformers model name
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(alias)
    info = {
        "alias": alias,
        "name": f"{alias} (sentence-transformers)",
        "expected_dim": None,
        "dim_band": "auto",
        "cost": "free (local; downloads model once)",
        "cached": False,
    }
    return model, info


# ---------------------------------------------------------------------------
# Timer / environment / IO
# ---------------------------------------------------------------------------

class Timer:
    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        self.seconds = time.perf_counter() - self._start


def environment() -> Dict[str, str]:
    info: Dict[str, str] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    for pkg in ("numpy", "scikit-learn", "hdbscan", "umap-learn", "semantic-clusterer"):
        try:
            from importlib.metadata import version
            info[pkg] = version(pkg)
        except Exception:
            info[pkg] = "unknown"
    return info


def save_results(filename: str, payload: Dict[str, Any]) -> Path:
    """Merge new runs into an existing results file (append, don't overwrite).

    Runs are keyed by tier_requested. Re-running the same tier replaces that
    tier's entry; all other tiers are preserved. Final order: tiny→small→medium→large.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / filename
    new_runs: List[Dict[str, Any]] = payload.get("runs", [])

    if out.exists():
        try:
            existing = json.loads(out.read_text(encoding="utf-8"))
            by_tier: Dict[str, Dict[str, Any]] = {
                r["tier_requested"]: r for r in existing.get("runs", [])
            }
            for r in new_runs:
                by_tier[r["tier_requested"]] = r
            tier_order = ["tiny", "small", "medium", "large"]
            payload = {**payload, "runs": [by_tier[t] for t in tier_order if t in by_tier]}
        except Exception:
            pass  # corrupt file -> overwrite

    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out


def load_results(filename: str) -> Optional[Dict[str, Any]]:
    path = RESULTS_DIR / filename
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def print_table(rows: List[Dict[str, Any]], columns: List[str]) -> None:
    if not rows:
        print("(no rows)")
        return
    widths = {c: max(len(c), *(len(f"{r.get(c, '')}") for r in rows)) for c in columns}
    header = "  ".join(c.ljust(widths[c]) for c in columns)
    print(header)
    print("  ".join("-" * widths[c] for c in columns))
    for r in rows:
        print("  ".join(f"{r.get(c, '')}".ljust(widths[c]) for c in columns))


def fmt(x: Any, nd: int = 4) -> str:
    if isinstance(x, (int, float)) and not isinstance(x, bool):
        return f"{x:.{nd}f}"
    return str(x)
