import numpy as np
from typing import List

def _clip01(value: float) -> float:
    """Clamp a floating-point value to [0, 1]."""
    return float(np.clip(value, 0.0, 1.0))

def _unique_int_candidates(values: List[int], upper: int) -> List[int]:
    """Deduplicate integer candidates while preserving order."""
    seen = set()
    result: List[int] = []

    for value in values:
        candidate = int(np.clip(value, 2, upper))
        if candidate not in seen:
            result.append(candidate)
            seen.add(candidate)

    return result


def _select_diverse_candidates(
    candidates: List[dict],
    n_samples: int,
    max_candidates: int = 3,
) -> List[dict]:
    """Select diverse candidates for post-selection rescoring.

    Picks:
      1. Best raw score (always)
      2. Candidate with K closest to sqrt(N) * 0.35 (natural topic estimate)
      3. Candidate with highest separation (if not already selected)

    Falls back to top-N by score if fewer than max_candidates unique options.
    """
    if len(candidates) <= max_candidates:
        return candidates

    selected = [candidates[0]]  # Best raw score
    selected_ids = {id(candidates[0])}

    # Target K ~ sqrt(N) * 0.35
    target_k = max(2, int(np.sqrt(n_samples) * 0.35))
    best_k_dist = float("inf")
    best_k_cand = None
    for c in candidates:
        k = c["metrics"]["n_clusters"]
        dist = abs(k - target_k)
        if dist < best_k_dist and id(c) not in selected_ids:
            best_k_dist = dist
            best_k_cand = c
    if best_k_cand is not None:
        selected.append(best_k_cand)
        selected_ids.add(id(best_k_cand))

    # Highest separation
    best_sep = -1.0
    best_sep_cand = None
    for c in candidates:
        sep = c["metrics"].get("separation", 0.0)
        if sep > best_sep and id(c) not in selected_ids:
            best_sep = sep
            best_sep_cand = c
    if best_sep_cand is not None:
        selected.append(best_sep_cand)
        selected_ids.add(id(best_sep_cand))

    # Fill remaining slots from top by score
    for c in candidates:
        if len(selected) >= max_candidates:
            break
        if id(c) not in selected_ids:
            selected.append(c)
            selected_ids.add(id(c))

    return selected[:max_candidates]

