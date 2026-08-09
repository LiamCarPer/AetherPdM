"""CLI entrypoint for the CWRU->Paderborn domain shift study.

Usage:
    python scripts/run_domain_shift.py \
        --cwru-features data/interim/cwru/features_v2.parquet \
        --paderborn-features data/interim/paderborn/features_v2.parquet \
        --output reports/domain-shift-cwru-to-paderborn.md
"""

import argparse
from pathlib import Path

from aether_pdm.ops.domain_shift import run_domain_shift_study, write_domain_shift_report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train-on-CWRU / evaluate-on-Paderborn domain shift study"
    )
    parser.add_argument(
        "--cwru-features",
        type=Path,
        required=True,
        help="CWRU (source) feature Parquet, e.g. data/interim/cwru/features_v2.parquet",
    )
    parser.add_argument(
        "--paderborn-features",
        type=Path,
        required=True,
        help="Paderborn (target) feature Parquet, e.g. data/interim/paderborn/features_v2.parquet",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/domain-shift-cwru-to-paderborn.md"),
        help="Output Markdown report path",
    )
    parser.add_argument(
        "--mlflow-uri",
        type=str,
        default=None,
        help="MLflow tracking URI (default: sqlite:///mlflow.db)",
    )
    args = parser.parse_args()

    result = run_domain_shift_study(
        args.cwru_features,
        args.paderborn_features,
        mlflow_uri=args.mlflow_uri,
    )
    write_domain_shift_report(result, args.output)
    print(f"Report written to {args.output}")


if __name__ == "__main__":
    main()
