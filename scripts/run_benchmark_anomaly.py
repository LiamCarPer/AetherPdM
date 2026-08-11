"""CLI entrypoint for the anomaly detector benchmark (torch AE vs IsolationForest).

Usage:
    python scripts/run_benchmark_anomaly.py \
        --cwru-features data/interim/cwru_features/features_v2.parquet \
        --paderborn-features data/interim/paderborn/features_v1.parquet \
        --output reports/anomaly-benchmark.md --epochs 50
"""

import argparse
from pathlib import Path

from aether_pdm.ops.benchmark_anomaly import (
    benchmark_anomaly_detectors,
    write_benchmark_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark PyTorch autoencoder anomaly detector vs IsolationForest"
    )
    parser.add_argument(
        "--cwru-features",
        type=Path,
        required=True,
        help="CWRU feature Parquet with split/fault_type columns",
    )
    parser.add_argument(
        "--paderborn-features",
        type=Path,
        default=None,
        help="Optional Paderborn feature Parquet for the domain-shift probe",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/anomaly-benchmark.md"),
        help="Output Markdown report path",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Maximum torch training epochs (default: 50)",
    )
    parser.add_argument(
        "--mlflow-uri",
        type=str,
        default=None,
        help="MLflow tracking URI (default: sqlite:///mlflow.db)",
    )
    args = parser.parse_args()

    result = benchmark_anomaly_detectors(
        args.cwru_features,
        paderborn_features=args.paderborn_features,
        mlflow_uri=args.mlflow_uri,
        epochs=args.epochs,
    )
    report = write_benchmark_report(result, args.output)
    print(f"Report written to {report}")
    print(f"torch_wins = {result['torch_wins']}")
    print(
        "IF val: "
        f"DR={result['if_baseline']['detection_rate']:.4f} "
        f"FAR={result['if_baseline']['false_alarm_rate']:.4f}"
    )
    print(
        "Torch val: "
        f"DR={result['torch']['detection_rate']:.4f} "
        f"FAR={result['torch']['false_alarm_rate']:.4f}"
    )
    if result["torch"].get("paderborn"):
        print(
            "Paderborn (CWRU thresholds): "
            f"IF DR={result['if_baseline']['paderborn']['detection_rate']:.4f} "
            f"FAR={result['if_baseline']['paderborn']['false_alarm_rate']:.4f} | "
            f"torch DR={result['torch']['paderborn']['detection_rate']:.4f} "
            f"FAR={result['torch']['paderborn']['false_alarm_rate']:.4f}"
        )


if __name__ == "__main__":
    main()
