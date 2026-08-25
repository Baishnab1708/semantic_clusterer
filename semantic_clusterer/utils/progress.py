"""Progress reporting for the clustering pipeline.

Renders two clean, sequential bars — one for embedding, one for clustering —
each advancing smoothly from 0% to 100% with elapsed time and a summary line.

Usage
-----
::

    from semantic_clusterer.utils.progress import PipelineProgress

    with PipelineProgress(n_texts=5000, verbose=False) as prog:
        prog.preprocess()                        # (silent — fast housekeeping)

        prog.start_embedding(n_unique=4823)
        for batch_size in batches:
            embed_one_batch()
            prog.tick_embedding(batch_size)
        prog.end_embedding()

        prog.start_clustering()
        prog.clustering_phase("profiling", 0.10) # ~10% of clustering work
        prog.clustering_phase("clustering", 0.70)
        prog.clustering_phase("postprocessing", 0.20)
        prog.end_clustering(n_clusters=14)
"""

from __future__ import annotations

import sys
import time
from typing import Callable, Optional

# tqdm is a hard dependency (declared in pyproject.toml). Guarded import so
# the library still works in unusual environments where it's missing.
try:
    from tqdm import tqdm as _tqdm
    _TQDM_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TQDM_AVAILABLE = False


# ---------------------------------------------------------------------------
# Visual style
# ---------------------------------------------------------------------------

# Smooth bar made of solid + partial blocks. Renders identically on cmd,
# Powershell, bash, and macOS Terminal as long as the codepage is UTF-8
# (Python 3 default on every supported platform).
_BAR_CHARS = " ▏▎▍▌▋▊▉█"

# Bar width — small enough to fit in narrow terminals, large enough to look
# like a proper progress bar.
_BAR_WIDTH = 28

# Format string for both phase bars. tqdm fills in the variables.
#   {desc}       — phase name, e.g. "embedding"
#   {bar}        — animated bar
#   {percentage} — 0..100
#   {n_fmt}      — current units (e.g. "3,200")
#   {total_fmt}  — total units (e.g. "5,000")
#   {elapsed}    — wall-clock time so far (e.g. "0:00:12")
_PHASE_BAR_FORMAT = (
    "  INFO: {desc:<11} {bar} {percentage:3.0f}%  {n_fmt}/{total_fmt}  [{elapsed}]"
)

# Final summary lines written to stderr after each phase closes.
_DONE_PREFIX = "  INFO:"


def _fmt_count(n: int) -> str:
    """Pretty integer formatting with thousands separators."""
    return f"{n:,}"


def _fmt_seconds(s: float) -> str:
    """Compact duration formatting: 0.4s, 12.3s, 2m05s, 1h02m."""
    if s < 60:
        return f"{s:.1f}s"
    if s < 3600:
        m, sec = divmod(int(s), 60)
        return f"{m}m{sec:02d}s"
    h, rem = divmod(int(s), 3600)
    m = rem // 60
    return f"{h}h{m:02d}m"


# ---------------------------------------------------------------------------
# PipelineProgress
# ---------------------------------------------------------------------------

class PipelineProgress:
    """Two-phase progress display: embedding, then clustering.

    Each phase opens its own bar that animates 0% → 100% with a real
    work-proportional percentage. After the phase ends, the bar is replaced
    by a short summary line.

    The ``disable`` flag (or a missing ``tqdm`` install, or a non-tty stderr)
    forces a quiet, line-by-line fallback that's safe to redirect to a file.
    """

    def __init__(
        self,
        n_texts: int,
        verbose: bool = False,
        disable: bool = False,
    ) -> None:
        self._n_texts = int(n_texts)
        self._verbose = bool(verbose)

        # Quiet fallback when:
        #   - tqdm not installed, OR
        #   - caller forced disable, OR
        #   - stderr is redirected (e.g. captured by a CI runner, excluding Jupyter Notebooks)
        stderr_is_tty = bool(getattr(sys.stderr, "isatty", lambda: False)())
        is_jupyter = any(k in sys.modules for k in ("ipykernel", "google.colab"))
        self._disable = disable or not _TQDM_AVAILABLE or (not stderr_is_tty and not is_jupyter)

        self._t_pipeline_start = time.perf_counter()

        # Embedding phase state
        self._emb_bar: Optional[object] = None
        self._emb_total: int = 0
        self._emb_done: int = 0
        self._emb_t_start: float = 0.0

        # Clustering phase state
        self._clu_bar: Optional[object] = None
        # Clustering progress is tracked in 1000-unit "permille" steps so
        # fractional weights (e.g. 0.07) advance the bar by an integer
        # number of units instead of stalling.
        self._clu_total: int = 1000
        self._clu_done: int = 0
        self._clu_t_start: float = 0.0

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "PipelineProgress":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        # Defensively close any open bars (e.g. on exception)
        self._safe_close(self._emb_bar)
        self._emb_bar = None
        self._safe_close(self._clu_bar)
        self._clu_bar = None

    # ------------------------------------------------------------------
    # Embedding phase
    # ------------------------------------------------------------------

    def start_embedding(self, n_unique: Optional[int] = None) -> None:
        """Open the embedding bar.

        Parameters
        ----------
        n_unique:
            Number of *unique* texts that will be embedded. When omitted,
            falls back to the input count; the percentage is still
            meaningful because ``tick_embedding`` advances by the actual
            batch size each call.
        """
        total = int(n_unique if n_unique is not None else self._n_texts)
        self._emb_total = max(total, 1)
        self._emb_done = 0
        self._emb_t_start = time.perf_counter()

        if self._disable:
            # Quiet fallback: announce the phase, no live bar.
            self._write(f"  INFO: embedding   ... {_fmt_count(self._emb_total)} texts\n")
            return

        self._emb_bar = _tqdm(
            total=self._emb_total,
            desc="embedding",
            ncols=80,
            unit="",
            bar_format=_PHASE_BAR_FORMAT,
            ascii=_BAR_CHARS,
            file=sys.stderr,
            leave=True,         # leave the completed progress bar visible like TensorFlow
            dynamic_ncols=False,
            mininterval=0.05,   # smooth without overdrawing
            miniters=1,
        )

    def tick_embedding(self, n_done: int = 1) -> None:
        """Advance the embedding bar by ``n_done`` texts."""
        n = int(max(0, n_done))
        if n == 0:
            return
        self._emb_done = min(self._emb_done + n, self._emb_total)
        if self._emb_bar is not None:
            self._emb_bar.update(n)  # type: ignore[attr-defined]

    def end_embedding(self) -> None:
        """Close the embedding bar and print a one-line summary."""
        if self._emb_t_start == 0.0:
            return
        elapsed = time.perf_counter() - self._emb_t_start

        # Snap to 100% in case the embedder reported fewer ticks than expected
        if self._emb_bar is not None:
            remaining = self._emb_total - self._emb_done
            if remaining > 0:
                self._emb_bar.update(remaining)  # type: ignore[attr-defined]
            self._safe_close(self._emb_bar)
            self._emb_bar = None

        self._write(
            f"{_DONE_PREFIX} embedded   {_fmt_count(self._emb_total):>9} texts "
            f"in {_fmt_seconds(elapsed)}\n"
        )

    @property
    def embedding_callback(self) -> Callable[[int], None]:
        """Return a callable suitable for passing to ``OnnxEmbedder.embed``.

        The callback advances the embedding bar by the number of texts that
        completed in the most recent batch.
        """
        return self.tick_embedding

    # ------------------------------------------------------------------
    # Clustering phase (covers profiling + reduction + clustering +
    # postprocessing + scoring under a single user-facing label)
    # ------------------------------------------------------------------

    def start_clustering(self) -> None:
        """Open the clustering bar."""
        self._clu_done = 0
        self._clu_t_start = time.perf_counter()

        # Disable the live tqdm progress bar for the clustering phase
        # because the internal UMAP/HDBSCAN execution cannot tick dynamically
        # and gets stuck at 25%. Show a clean text status instead.
        self._write("  INFO: clustering  ...\n")
        return

    def clustering_phase(self, name: str, weight: float) -> None:
        """Advance the clustering bar by a fraction of its total.

        ``weight`` is a number in ``(0, 1]`` representing this sub-phase's
        share of the clustering work. The sub-phases inside one run should
        sum to roughly 1.0; the bar caps at 99% until ``end_clustering`` is
        called so the final snap to 100% always feels like completion.
        """
        if self._clu_t_start == 0.0:
            return

        delta = int(round(self._clu_total * float(max(0.0, min(1.0, weight)))))
        # Reserve at least 10 permille (1%) so the cap at 99% is respected
        cap = self._clu_total - 10
        new_done = min(self._clu_done + delta, cap)
        delta = new_done - self._clu_done
        self._clu_done = new_done

        if delta > 0 and self._clu_bar is not None:
            self._clu_bar.update(delta)  # type: ignore[attr-defined]

        if self._verbose:
            elapsed = time.perf_counter() - self._t_pipeline_start
            print(f"  [verbose] clustering · {name} (t={elapsed:.1f}s)")

    def end_clustering(self, n_clusters: Optional[int] = None) -> None:
        """Close the clustering bar and print a one-line summary."""
        if self._clu_t_start == 0.0:
            return
        elapsed = time.perf_counter() - self._clu_t_start

        if self._clu_bar is not None:
            remaining = self._clu_total - self._clu_done
            if remaining > 0:
                self._clu_bar.update(remaining)  # type: ignore[attr-defined]
            self._safe_close(self._clu_bar)
            self._clu_bar = None

        if n_clusters is None:
            tail = "complete"
        elif n_clusters == 1:
            tail = "found 1 cluster"
        else:
            tail = f"found {_fmt_count(n_clusters)} clusters"

        self._write(f"{_DONE_PREFIX} clustering {tail:>16}  in {_fmt_seconds(elapsed)}\n")

    # ------------------------------------------------------------------
    # Backwards-compatible shim
    # ------------------------------------------------------------------
    #
    # The legacy API was ``prog.step(name)`` for every internal sub-phase.
    # Pipeline modules still call it; we map the old names onto the new
    # two-phase model so we don't have to touch every caller.

    _CLUSTERING_PHASE_WEIGHTS = {
        "profiling":      0.10,
        "reduction":      0.20,
        "clustering":     0.50,
        "postprocessing": 0.15,
        "scoring":        0.05,
    }

    def step(self, name: str) -> None:
        """Legacy entrypoint: route an old-style step name to the new bars."""
        key = name.lower()

        if key == "preprocessing":
            # Silent — preprocessing is essentially instantaneous next to
            # embedding, and showing a separate bar for it just creates noise.
            return

        if key == "embedding":
            # Open the embedding bar lazily on the first embedding tick.
            # We don't know the unique count yet here, so default to n_texts.
            if self._emb_t_start == 0.0:
                self.start_embedding(self._n_texts)
            return

        # Anything else is a clustering sub-phase. Open the clustering bar
        # the first time we see one of these names, then weight-advance it.
        if self._clu_t_start == 0.0:
            # Auto-close embedding if it was open and never explicitly ended.
            if self._emb_t_start != 0.0 and self._emb_bar is not None:
                self.end_embedding()
            self.start_clustering()

        weight = self._CLUSTERING_PHASE_WEIGHTS.get(key, 0.05)
        self.clustering_phase(key, weight)

    def done(self, n_clusters: Optional[int] = None) -> None:
        """Legacy entrypoint: finish whichever bar is currently open."""
        if self._emb_bar is not None or (
            self._emb_t_start != 0.0 and self._clu_t_start == 0.0
        ):
            self.end_embedding()
        if self._clu_t_start != 0.0:
            self.end_clustering(n_clusters=n_clusters)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _safe_close(self, bar: Optional[object]) -> None:
        """Close a tqdm bar if it's open, ignoring any teardown errors."""
        if bar is None:
            return
        try:
            bar.close()  # type: ignore[attr-defined]
        except Exception:
            pass

    def _write(self, message: str) -> None:
        """Write to stderr without disturbing any open bar."""
        try:
            sys.stderr.write(message)
            sys.stderr.flush()
        except Exception:
            pass


def _plain_status(message: str, end: str = "\n") -> None:
    """Standalone stderr line used outside the PipelineProgress context."""
    sys.stderr.write(f"  {message}{end}")
    sys.stderr.flush()
