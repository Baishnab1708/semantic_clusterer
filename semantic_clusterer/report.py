"""Clustering run report and pipeline trace accumulator."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Dict, List, Literal, Optional

import numpy as np


@dataclass
class ClusteringReport:
    """Structured run report for a single clustering invocation."""

    # Counts
    n_input_texts: int
    n_clustered: int
    n_noise: int
    n_clusters: int

    # Routing
    pipeline_tier: Literal["tiny", "small", "medium", "large"]
    embedding_dim: int
    dim_band: Literal["low", "mid", "high", "xhigh"]

    # Profile snapshot (DatasetProfile.__dict__)
    dataset_profile: Dict[str, Any]

    # Pipeline-specific chosen params
    chosen_params: Dict[str, Any]

    # Intrinsic metrics (full output of score_clustering)
    intrinsic_metrics: Dict[str, Any]

    # Phase timings in seconds
    phase_timings: Dict[str, float]

    # Warnings and confidence
    warnings: List[str]
    confidence_level: Literal["high", "low"]

    # Metadata
    random_state: int
    library_version: str

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-coercible dict copy of the report."""
        return _coerce_json(asdict(self))


@dataclass
class _PipelineTrace:
    """Internal accumulator for run report fields."""

    chosen_params: Dict[str, Any] = field(default_factory=dict)
    intrinsic_metrics: Dict[str, Any] = field(default_factory=dict)
    phase_timings: Dict[str, float] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    profile: Optional[Dict[str, Any]] = None
    confidence_level: Literal["high", "low"] = "high"

    def warn(self, message: str, *, set_low_confidence: bool = False) -> None:
        """Append a warning message (dedup-aware) and optionally set low confidence."""
        if message not in self.warnings:
            self.warnings.append(message)
        if set_low_confidence:
            self.confidence_level = "low"

    def time(self, phase: str, seconds: float) -> None:
        """Record a phase timing, clamped to >= 0.0."""
        self.phase_timings[phase] = float(max(0.0, seconds))


_NUMPY_INT = (np.integer,)
_NUMPY_FLOAT = (np.floating,)
_NUMPY_BOOL = (np.bool_,)


def _coerce_json(obj: Any) -> Any:
    """Recursively coerce an object to a JSON-safe representation.

    - numpy integer types  -> int
    - numpy float types    -> float (NaN/Inf -> None)
    - numpy bool types     -> bool
    - numpy arrays         -> list (elements recursively coerced)
    - plain float NaN/Inf  -> None
    - dict                 -> dict with str keys, values recursively coerced
    - list/tuple           -> list with elements recursively coerced
    - dataclass instances  -> dict via asdict, then recursively coerced
    - str/bool/None/int    -> returned as-is
    - anything else        -> str(obj) so json.dumps never raises
    """
    if obj is None or isinstance(obj, (str, bool)):
        return obj
    if isinstance(obj, _NUMPY_BOOL):
        return bool(obj)
    if isinstance(obj, _NUMPY_INT):
        return int(obj)
    if isinstance(obj, _NUMPY_FLOAT):
        f = float(obj)
        return f if np.isfinite(f) else None  # NaN / Inf -> None per JSON-safety
    if isinstance(obj, (int, float)):
        if isinstance(obj, float) and not np.isfinite(obj):
            return None
        return obj
    if isinstance(obj, np.ndarray):
        return [_coerce_json(x) for x in obj.tolist()]
    if isinstance(obj, dict):
        return {str(k): _coerce_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_coerce_json(x) for x in obj]
    if is_dataclass(obj):
        return _coerce_json(asdict(obj))
    # Fallback: stringify unknown types so json.dumps never raises
    return str(obj)
