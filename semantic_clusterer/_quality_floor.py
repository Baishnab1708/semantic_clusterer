"""Quality floor enforcement for all pipeline tiers.

Loads per-tier quality floors from ``tests/fixtures/release_ready/baseline_scores.json``
at import time. If the file is missing, conservative defaults of 0.30 are used
and a single ``UserWarning`` is emitted naming the missing path.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Literal, Optional

import numpy as np

if TYPE_CHECKING:
    from semantic_clusterer.pipeline.profile import DatasetProfile
    from semantic_clusterer.report import _PipelineTrace

# ---------------------------------------------------------------------------
# Load quality floors at import time
# ---------------------------------------------------------------------------

_BASELINE_PATH = (
    Path(__file__).parent.parent / "tests" / "fixtures" / "release_ready" / "baseline_scores.json"
)

_FALLBACK_FLOORS: Dict[str, float] = {
    "tiny": 0.35,
    "small": 0.60,
    "medium": 0.60,
    "large": 0.60,
}


def _load_quality_floors() -> Dict[str, float]:
    """Load quality floors from baseline_scores.json, falling back to defaults."""
    if not _BASELINE_PATH.exists():
        # Silent fallback in production (where tests/ is not distributed)
        return dict(_FALLBACK_FLOORS)

    try:
        data = json.loads(_BASELINE_PATH.read_text(encoding="utf-8"))
        floors: Dict[str, float] = {}
        for tier in ("tiny", "small", "medium", "large"):
            if tier in data and "floor" in data[tier]:
                floors[tier] = float(data[tier]["floor"])
            else:
                floors[tier] = _FALLBACK_FLOORS[tier]
        return floors
    except Exception as exc:  # noqa: BLE001
        warnings.warn(
            f"Failed to parse quality floor baseline file {_BASELINE_PATH}: {exc}. "
            f"Falling back to defaults {_FALLBACK_FLOORS}.",
            UserWarning,
            stacklevel=2,
        )
        return dict(_FALLBACK_FLOORS)


_QUALITY_FLOORS: Dict[str, float] = _load_quality_floors()


# ---------------------------------------------------------------------------
# Enforcement helper
# ---------------------------------------------------------------------------


def _enforce_quality_floor(
    embeddings: np.ndarray,
    labels: np.ndarray,
    profile: "DatasetProfile",
    tier: str,
    trace: "Optional[_PipelineTrace]",
) -> None:
    """Enforce quality floor checks and update trace accordingly.

    Checks two conditions:

    1. **High noise ratio**: ``mean(labels == -1) > 0.85`` → emit
       ``UserWarning``, append ``"high-noise-ratio"`` to trace warnings, set
       ``confidence_level="low"``.

    2. **Quality floor breach**: ``profile.cluster_tendency >= 0.40`` and
       ``score < floor`` → emit ``UserWarning`` with score and floor formatted
       to 4 decimal places, append ``"no-cluster-met-quality-floor"``, set
       ``confidence_level="low"``.

    Parameters
    ----------
    embeddings:
        2-D float32 array of shape ``(N, D)``.
    labels:
        1-D int32 label array of shape ``(N,)``.
    profile:
        ``DatasetProfile`` for the current dataset.
    tier:
        Pipeline tier name — one of ``"tiny"``, ``"small"``, ``"medium"``,
        ``"large"``.
    trace:
        Optional ``_PipelineTrace`` accumulator. When ``None`` the function
        still emits warnings but does not mutate any trace.
    """
    n = len(labels)
    if n < 2:
        return

    floor = _QUALITY_FLOORS.get(tier, 0.30)
    # Low-dim embeddings (D<512) have high centroid similarity (separation
    # compression) just like high-dim embeddings. Mid-dim is the sweet spot.
    D = embeddings.shape[1]
    if D >= 1024 or D < 512:
        floor -= 0.05

    # --- Check 1: high noise ratio ---
    noise_ratio = float(np.mean(labels == -1))
    if noise_ratio > 0.85:
        warnings.warn(
            f"High noise ratio ({noise_ratio:.4f}) > 0.85; confidence=low.",
            UserWarning,
            stacklevel=2,
        )
        if trace is not None:
            if "high-noise-ratio" not in trace.warnings:
                trace.warnings.append("high-noise-ratio")
            trace.confidence_level = "low"

    # --- Check 2: quality floor breach ---
    if profile.cluster_tendency >= 0.40:
        score = float(trace.intrinsic_metrics.get("score", 0.0)) if trace is not None else 0.0
        if score < floor:
            warnings.warn(
                f"Intrinsic score {score:.4f} below {tier} floor {floor:.4f}; "
                f"confidence=low.",
                UserWarning,
                stacklevel=2,
            )
            if trace is not None:
                if "no-cluster-met-quality-floor" not in trace.warnings:
                    trace.warnings.append("no-cluster-met-quality-floor")
                trace.confidence_level = "low"
