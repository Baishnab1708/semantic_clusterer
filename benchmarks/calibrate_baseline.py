"""Propose a calibrated baseline_scores.json from real benchmark results.

Why this exists
---------------
The runtime quality floor in ``semantic_clusterer/_quality_floor.py`` loads
``tests/fixtures/release_ready/baseline_scores.json`` and uses the per-tier
``floor`` to softly downgrade confidence. Today those numbers come from
synthetic fixtures. This script replaces them with floors derived from REAL
gold-standard runs.

Calibration rule (deliberately conservative)
---------------------------------------------
For each tier we collect the library's own intrinsic ``score`` across every
benchmark run that landed in that tier, then set::

    floor = round(min(scores_in_tier) * SAFETY_MARGIN, 4)

We use ``min`` (not mean) and a safety margin < 1.0 so the floor is a genuine
lower bound. Setting the floor equal to an observed score would cause false
"low confidence" warnings on any harder corpus a user supplies. The recorded
``score`` / ``ari`` fields are informational baselines, not gates.

Safety
------
This script NEVER overwrites the live file. It writes a proposal to
``benchmarks/results/baseline_scores.proposed.json`` and prints the one-line
copy command for you to promote it after review.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from bench_common import RESULTS_DIR, load_results

SAFETY_MARGIN = 0.75
TIERS = ("tiny", "small", "medium", "large")
LIVE_BASELINE = (
    Path(__file__).parent.parent
    / "tests" / "fixtures" / "release_ready" / "baseline_scores.json"
)


def _run_score(run: Dict[str, Any]) -> float:
    """Read the intrinsic score from a detailed run record."""
    return float(run.get("intrinsic_metrics", {}).get("score", 0.0))


def _run_ari(run: Dict[str, Any]) -> float:
    """Read the external ARI from a detailed run record."""
    return float(run.get("external_metrics", {}).get("ari", 0.0))


def _collect_runs(result_files: List[str]) -> List[Dict[str, Any]]:
    """Gather all SemanticClusterer runs from the given results files.

    The quality floor gates SemanticClusterer (variable-K, density-based), so
    only those runs calibrate the floor. KSplit files are skipped.
    """
    runs: List[Dict[str, Any]] = []
    for fname in result_files:
        data = load_results(fname)
        if not data:
            print(f"  skip (not found): {fname}")
            continue
        if data.get("kind") != "SemanticClusterer":
            print(f"  skip (not SemanticClusterer): {fname}")
            continue
        for run in data.get("runs", []):
            runs.append(dict(run))
    return runs


def calibrate(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Build a baseline_scores.json proposal grouped by ACTUAL tier."""
    by_tier: Dict[str, List[Dict[str, Any]]] = {t: [] for t in TIERS}
    for run in runs:
        tier = run.get("tier_actual") or run.get("tier_requested")
        if tier in by_tier:
            by_tier[tier].append(run)

    # Start from the existing live file so untouched tiers are preserved.
    proposal: Dict[str, Any] = {}
    if LIVE_BASELINE.exists():
        try:
            proposal = json.loads(LIVE_BASELINE.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            proposal = {}

    for tier in TIERS:
        tier_runs = by_tier[tier]
        if not tier_runs:
            continue
        scores = [_run_score(r) for r in tier_runs]
        aris = [_run_ari(r) for r in tier_runs]
        best = min(tier_runs, key=_run_score)
        proposal[tier] = {
            "tier": tier,
            "fixture": f"gold:{best.get('dataset', '?')}:{best.get('n_docs', 0)}",
            "score": round(float(sum(scores) / len(scores)), 6),
            "ari": round(float(sum(aris) / len(aris)), 6),
            "floor": round(min(scores) * SAFETY_MARGIN, 4),
            "n_clusters": int(best.get("n_pred_clusters", 0)),
            "_calibration": {
                "n_runs": len(tier_runs),
                "embedders": sorted({r.get("embedder_alias", "?") for r in tier_runs}),
                "min_score": round(min(scores), 6),
                "max_score": round(max(scores), 6),
                "safety_margin": SAFETY_MARGIN,
            },
        }
    return proposal


def main() -> None:
    ap = argparse.ArgumentParser(description="Propose calibrated quality floors.")
    ap.add_argument(
        "--results",
        nargs="+",
        default=["clusterer_minilm.json", "clusterer_mpnet.json", "clusterer_openai3small.json"],
        help="Result files in benchmarks/results/ to calibrate from.",
    )
    args = ap.parse_args()

    print("Collecting runs from:", ", ".join(args.results))
    runs = _collect_runs(args.results)
    if not runs:
        print("No SemanticClusterer runs found. Run run_clusterer.py first.")
        return

    proposal = calibrate(runs)

    print("\nProposed per-tier floors (conservative lower bounds):")
    for tier in TIERS:
        if tier in proposal and "_calibration" in proposal[tier]:
            cal = proposal[tier]["_calibration"]
            print(
                f"  {tier:>6}: floor={proposal[tier]['floor']:.4f}  "
                f"(min_score={cal['min_score']:.4f} x {cal['safety_margin']}, "
                f"{cal['n_runs']} run(s), embedders={cal.get('embedders')})"
            )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "baseline_scores.proposed.json"
    out.write_text(json.dumps(proposal, indent=2), encoding="utf-8")

    print(f"\nProposal written -> {out}")
    print("\nReview it, then promote with:")
    print(f'  copy "{out}" "{LIVE_BASELINE}"')
    print("\n(Nothing was overwritten. The live baseline is unchanged.)")


if __name__ == "__main__":
    main()
