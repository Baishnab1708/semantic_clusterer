"""Quality profiles for SemanticKSplit multi-restart selection.

Translates the user-facing quality preset ("fast", "balanced", "best")
into a concrete number of restarts for each pipeline tier.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class QualityProfile:
    """Restart parameters for one (preset, tier) combination."""
    n_restarts: int           # number of independent algorithm restarts
    silhouette_sample_cap: int  # max rows to pass to silhouette_score


# Table: (preset, tier) -> QualityProfile
# tiny/small are cheap so they get more restarts than medium/large.
_PROFILES: dict = {
    ("fast",     "tiny"):   QualityProfile(1,  2000),
    ("fast",     "small"):  QualityProfile(1,  5000),
    ("fast",     "medium"): QualityProfile(1,  5000),
    ("fast",     "large"):  QualityProfile(1,  5000),

    ("balanced", "tiny"):   QualityProfile(5,  2000),
    ("balanced", "small"):  QualityProfile(5,  5000),
    ("balanced", "medium"): QualityProfile(3,  5000),
    ("balanced", "large"):  QualityProfile(2,  5000),

    ("best",     "tiny"):   QualityProfile(15, 2000),
    ("best",     "small"):  QualityProfile(12, 5000),
    ("best",     "medium"): QualityProfile(10, 5000),
    ("best",     "large"):  QualityProfile(5,  5000),
}


def resolve_quality(
    preset: Literal["fast", "balanced", "best"],
    *,
    tier: str,
) -> QualityProfile:
    """Return the QualityProfile for a (preset, tier) pair."""
    key = (preset, tier)
    if key not in _PROFILES:
        return QualityProfile(1, 5000)  # safe default
    return _PROFILES[key]
