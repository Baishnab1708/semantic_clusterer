"""Granularity profiles for SemanticClusterer.

Translates the user-facing cluster_granularity preset ("fine", "balanced",
"coarse") into concrete pipeline parameters: min_cluster_size floor,
centroid merge threshold, and fragmentation penalty weight.

The only hardcoded constants in the library live here. They are derived
from two principles:
  1. Cosine concentration increases with embedding dimension, so merge
     thresholds decrease as the dim band rises.
  2. The natural number of topics in a corpus grows sub-linearly with the
     number of documents, so the min_cluster_size floor must grow like
     sqrt(N), never linearly. A linear floor (e.g. 5% of N) silently forbids
     more than ~20 clusters on large corpora and forces a collapse to 2-3
     clusters — the opposite of what more data should allow.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class GranularityProfile:
    """Concrete parameters for one (preset, band) combination."""
    mcs_ratio_floor: float            # legacy linear cap: max floor as fraction of N
    mcs_absolute_floor: int           # min_cluster_size never below this
    merge_centroid_threshold: float   # cosine sim above which to merge clusters
    fragmentation_penalty_weight: float  # added weight on fragmentation in score
    mcs_sqrt_coef: float = 1.0        # primary floor term: coef * sqrt(N)
    mcs_sub_floor_ratio: float = 0.0  # explore candidates below floor (0 = no sub-floor)


# Table: (preset, band) -> GranularityProfile
# Bands: low (256-511), mid (512-1023), high (1024-2047), xhigh (2048+)
#
# Floor model (see apply_mcs_floor):
#   floor = clip( mcs_sqrt_coef * sqrt(N),  mcs_absolute_floor,  mcs_ratio_floor * N )
# The sqrt term is the primary driver; mcs_ratio_floor is now an UPPER clamp
# that only binds on very small corpora (so the floor can't demand a cluster
# larger than X% of a tiny dataset). On large corpora the sqrt term wins,
# which is what lets cluster count grow with the corpus instead of collapsing.
#
# Sub-floor exploration (mcs_sub_floor_ratio):
#   fine    → 0.4  (explore 0.4× and 0.6× floor: most freedom)
#   balanced→ 0.6  (explore 0.6× floor: moderate freedom)
#   coarse  → 0.0  (no sub-floor: stays at or above floor)
# The scoring objective selects the best K from all candidates. Sub-floor
# candidates let HDBSCAN consider more clusters on rich datasets without
# forcing it — fragmentation penalty + stability guard against over-splitting.
_PROFILES: dict = {
    # ── fine ─────────────────────────────────────────────────────────
    # Smallest floor -> most clusters. Near-duplicate merge only.
    ("fine", "low"):   GranularityProfile(0.05, 5,  0.85, 0.0, mcs_sqrt_coef=0.2, mcs_sub_floor_ratio=0.4),
    ("fine", "mid"):   GranularityProfile(0.05, 5,  0.90, 0.0, mcs_sqrt_coef=0.3, mcs_sub_floor_ratio=0.4),
    ("fine", "high"):  GranularityProfile(0.05, 5,  0.96, 0.0, mcs_sqrt_coef=0.3, mcs_sub_floor_ratio=0.4),
    ("fine", "xhigh"): GranularityProfile(0.05, 5,  0.97, 0.0, mcs_sqrt_coef=0.3, mcs_sub_floor_ratio=0.4),

    # ── balanced (default) ────────────────────────────────────────────
    # Moderate floor (~sqrt(N/2)) + meaningful merge pass + light frag penalty.
    ("balanced", "low"):   GranularityProfile(0.10, 8,  0.72, 1.0, mcs_sqrt_coef=0.35, mcs_sub_floor_ratio=0.7),
    ("balanced", "mid"):   GranularityProfile(0.10, 8,  0.78, 1.0, mcs_sqrt_coef=0.5, mcs_sub_floor_ratio=0.7),
    ("balanced", "high"):  GranularityProfile(0.10, 8,  0.93, 1.0, mcs_sqrt_coef=0.35, mcs_sub_floor_ratio=0.7),
    ("balanced", "xhigh"): GranularityProfile(0.10, 8,  0.94, 1.0, mcs_sqrt_coef=0.35, mcs_sub_floor_ratio=0.7),

    # ── coarse ────────────────────────────────────────────────────────
    # Largest floor -> fewest, broadest clusters. Hard merge + strong penalty.
    # Lower threshold = more aggressive merging.
    ("coarse", "low"):   GranularityProfile(0.20, 15, 0.68, 3.0, mcs_sqrt_coef=1.0, mcs_sub_floor_ratio=0.0),
    ("coarse", "mid"):   GranularityProfile(0.20, 15, 0.75, 3.0, mcs_sqrt_coef=1.5, mcs_sub_floor_ratio=0.0),
    ("coarse", "high"):  GranularityProfile(0.20, 15, 0.88, 3.0, mcs_sqrt_coef=1.5, mcs_sub_floor_ratio=0.0),
    ("coarse", "xhigh"): GranularityProfile(0.20, 15, 0.90, 3.0, mcs_sqrt_coef=1.5, mcs_sub_floor_ratio=0.0),
}


def resolve_granularity(
    preset: Literal["fine", "balanced", "coarse"],
    *,
    band: str,
) -> GranularityProfile:
    """Return the GranularityProfile for a (preset, band) pair.

    Falls back to the "low" band entry if the band is unrecognised,
    so the function never raises on unexpected dim-band values.
    """
    key = (preset, band)
    if key not in _PROFILES:
        key = (preset, "low")
    return _PROFILES[key]


def apply_mcs_floor(
    candidates: list,
    profile: GranularityProfile,
    n_samples: int,
    user_override: int | None,
) -> list:
    """Filter a min_cluster_size candidate list using the granularity floor.

    The floor scales with sqrt(N) (the natural rate at which topic count grows
    with corpus size), clamped below by an absolute minimum and above by a
    fraction of N. The upper clamp only binds on small corpora; on large
    corpora the sqrt term dominates, so the achievable cluster count keeps
    growing with the data instead of collapsing to a handful of giant blobs.

    When the user has set an explicit min_cluster_size override, that value
    is used as-is and the granularity floor is ignored. This preserves the
    invariant that explicit user overrides always win.

    Returns a non-empty list — if all candidates fall below the floor,
    returns [floor] so the pipeline always has at least one candidate.
    """
    if user_override is not None:
        return [int(user_override)]

    floor = compute_mcs_floor(profile, n_samples)
    filtered = [c for c in candidates if c >= floor]
    return filtered if filtered else [floor]


def compute_mcs_floor(profile: GranularityProfile, n_samples: int) -> int:
    """Compute the min_cluster_size floor for a profile and corpus size.

    floor = clip( coef * sqrt(N),  absolute_floor,  ratio_floor * N )

    The sqrt term is primary. ``mcs_ratio_floor`` acts as an upper clamp so the
    floor never demands a cluster larger than a fixed fraction of a (small)
    corpus; on large corpora it does not bind.
    """
    n = max(1, int(n_samples))
    sqrt_term = profile.mcs_sqrt_coef * math.sqrt(n)
    upper_clamp = profile.mcs_ratio_floor * n
    floor = max(profile.mcs_absolute_floor, sqrt_term)
    floor = min(floor, max(profile.mcs_absolute_floor, upper_clamp))
    return int(max(2, round(floor)))


def mcs_candidate_spread(profile: GranularityProfile, n_samples: int) -> list:
    """Return a min_cluster_size search grid anchored on the floor.

    The spread is granularity-aware:

    - **fine** (``mcs_sub_floor_ratio=0.4``): explores below the floor
      aggressively ``[0.4×, 0.6×, floor, 2×, 3.5×]`` — up to 5 candidates,
      giving HDBSCAN maximum freedom to find many clusters.
    - **balanced** (``mcs_sub_floor_ratio=0.6``): moderate sub-floor
      exploration ``[0.6×, floor, 1.5×, 2.5×, 4×]`` — permits more clusters
      on rich datasets while the scoring objective guards against
      over-fragmentation.
    - **coarse** (``mcs_sub_floor_ratio=0.0``): no sub-floor exploration
      ``[floor, 2×, 3.5×]`` — stays at or above the floor for fewest clusters.

    All candidates are clamped to ``[absolute_floor, N/3]``, deduplicated,
    and sorted ascending. The scoring objective (not this function) decides
    which candidate wins.
    """
    floor = compute_mcs_floor(profile, n_samples)
    n = max(4, int(n_samples))
    abs_floor = max(2, profile.mcs_absolute_floor)
    cap = max(floor, n // 3)  # never demand a cluster larger than a third of the data

    # Build candidates above the floor (always present)
    raw = [floor]

    # Sub-floor candidates: controlled by mcs_sub_floor_ratio
    # 0.0 = no sub-floor (coarse), 0.6 = moderate (balanced), 0.4 = aggressive (fine)
    sfr = profile.mcs_sub_floor_ratio
    if sfr > 0.0 and floor > abs_floor:
        # Add one candidate at sfr × floor (e.g. 0.6× for balanced)
        raw.append(int(round(floor * sfr)))
        # For aggressive presets (sfr < 0.5), add a deeper candidate and an intermediate too
        if sfr < 0.5:
            raw.append(int(round(floor * (sfr * 0.5))))
            raw.append(int(round(floor * (sfr + 0.2))))  # e.g. 0.6× for fine
            raw.append(int(round(floor * (sfr + 0.2))))  # e.g. 0.6× for fine

    # Above-floor candidates: wider spread to cover the search space
    if sfr > 0.0:
        # With sub-floor, use a wider above-floor spread
        raw.extend([int(round(floor * 1.5)), int(round(floor * 2.5)), int(round(floor * 4.0))])
    else:
        # Without sub-floor (coarse), keep the original compact spread
        raw.extend([int(round(floor * 2.0)), int(round(floor * 3.5))])

    spread = sorted({int(max(abs_floor, min(c, cap))) for c in raw})
    return spread
