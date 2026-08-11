"""Estimate degradation-trend RUL from a health-score history.

HONEST SCOPE: this tool extrapolates a degradation trend to a failure
threshold. It does NOT predict calibrated time-to-failure — see
``aether_pdm.models.rul`` and ``docs/model-cards/rul-v1.md``.

Usage:
    python scripts/estimate_rul.py --synthetic
    python scripts/estimate_rul.py --scores-json scores.json --failure-threshold 0.9
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

from aether_pdm.data.synthetic import degradation_ramp
from aether_pdm.models.rul import estimate_rul_from_scores


def _normalize_history(entries: list[dict]) -> list[dict]:
    """Convert raw JSON entries to ``{"timestamp": hours, "health_score": x}``.

    Numeric timestamps are treated as elapsed hours. ISO-8601 timestamp
    strings are converted to hours relative to the first timestamp in the
    file. Malformed entries are skipped.
    """
    is_iso = any(isinstance(e.get("timestamp"), str) for e in entries)
    normalized: list[dict] = []
    for entry in entries:
        try:
            health_score = float(entry["health_score"])
        except (KeyError, TypeError, ValueError):
            continue
        ts = entry.get("timestamp")
        if isinstance(ts, (int, float)):
            hours = float(ts)
        elif isinstance(ts, str):
            try:
                hours = datetime.fromisoformat(ts).timestamp() / 3600.0
            except ValueError:
                continue
        else:
            continue
        normalized.append({"timestamp": hours, "health_score": health_score})
    if is_iso and normalized:
        t0 = min(float(e["timestamp"]) for e in normalized)
        for entry in normalized:
            entry["timestamp"] = float(entry["timestamp"]) - t0
    return normalized


def _print_result(asset_id: str, result: dict) -> None:
    """Human-readable RUL summary with the honesty caveats."""
    print(f"RUL estimate for asset '{asset_id}' (degradation-trend extrapolation)")
    print(f"  status      : {result['status']}")
    if result["rul_hours"] is None:
        print(f"  reason      : {result['reason']}")
    else:
        ci_low = result["ci_low_hours"]
        ci_high = result["ci_high_hours"]
        ci_str = "unbounded" if ci_high is None else f"{ci_low:.1f} - {ci_high:.1f}"
        print(f"  RUL         : {result['rul_hours']:.1f} hours (95% CI: {ci_str})")
    slope = result["slope"]
    slope_str = "n/a" if slope is None else f"{slope:.6f} index/hour"
    r2 = result["r_squared"]
    r2_str = "n/a" if r2 is None else f"{r2:.3f}"
    confidence = result["confidence"]
    confidence_str = (
        "n/a"
        if confidence is None
        else f"{confidence} (never 'high' - unvalidated extrapolation)"
    )
    print(f"  slope       : {slope_str}")
    print(f"  R^2         : {r2_str}")
    print(f"  confidence  : {confidence_str}")
    print(f"  n_points    : {result['n_points']}")
    print(f"  threshold   : {result['failure_threshold']}")
    print(f"  disclaimer  : {result['disclaimer']}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Degradation-trend RUL estimate. Honest scope: trend extrapolation, "
            "NOT calibrated time-to-failure."
        )
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--scores-json",
        type=Path,
        help=(
            "JSON file: list of {\"timestamp\": <hours | ISO-8601>, "
            "\"health_score\": 0-1}"
        ),
    )
    source.add_argument(
        "--synthetic",
        action="store_true",
        help="Generate a synthetic degradation ramp for the demo (no data needed)",
    )
    parser.add_argument(
        "--failure-threshold",
        type=float,
        default=0.9,
        help="Degradation index at which the asset is considered failed (default: 0.9)",
    )
    parser.add_argument(
        "--min-points",
        type=int,
        default=5,
        help="Minimum observations required for a trend fit (default: 5)",
    )
    parser.add_argument(
        "--asset-id",
        type=str,
        default="asset-001",
        help="Asset id reported in the output (default: asset-001)",
    )
    parser.add_argument(
        "--n-points",
        type=int,
        default=24,
        help="--synthetic only: number of inspection points (default: 24)",
    )
    parser.add_argument(
        "--span-hours",
        type=float,
        default=480.0,
        help="--synthetic only: ramp duration in hours (default: 480)",
    )
    args = parser.parse_args()

    if args.synthetic:
        ramp = degradation_ramp(n_points=args.n_points, span_hours=args.span_hours)
        history = [
            {"timestamp": float(row.hours), "health_score": float(row.health_score)}
            for row in ramp.itertuples()
        ]
        now_hours = float(ramp["hours"].iloc[-1])
    else:
        with open(args.scores_json, encoding="utf-8-sig") as fh:
            raw = json.load(fh)
        if not isinstance(raw, list):
            raise SystemExit(
                "--scores-json must contain a JSON list of "
                "{timestamp, health_score} objects"
            )
        history = _normalize_history(raw)
        if not history:
            raise SystemExit(
                "No valid {timestamp, health_score} entries found in the JSON file"
            )
        now_hours = float(max(e["timestamp"] for e in history))

    result = estimate_rul_from_scores(
        args.asset_id,
        history,
        now_hours=now_hours,
        failure_threshold=args.failure_threshold,
        min_points=args.min_points,
    )
    _print_result(args.asset_id, result)


if __name__ == "__main__":
    main()
