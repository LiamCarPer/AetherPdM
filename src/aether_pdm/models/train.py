"""
Unified training entrypoint.

Trains both anomaly and fault classifiers from a features Parquet file.
Logs models to MLflow.

Usage:
    python -m aether_pdm.models.train --features data/interim/features/features_v1.parquet
"""

import argparse
from pathlib import Path

import mlflow

from aether_pdm.models.anomaly import train_anomaly
from aether_pdm.models.fault import train_fault_classifier


def main() -> None:
    parser = argparse.ArgumentParser(description="Train AetherPdM models")
    parser.add_argument("--features", type=Path, required=True, help="Path to features Parquet")
    parser.add_argument("--mlflow-uri", type=str, default=None, help="MLflow tracking URI")
    parser.add_argument("--anomaly-contamination", type=float, default=0.05)
    parser.add_argument("--fault-estimators", type=int, default=300)
    args = parser.parse_args()

    mlflow.set_tracking_uri(args.mlflow_uri or "mlruns")

    print("=== Training Anomaly Detector ===")
    train_anomaly(
        args.features,
        contamination=args.anomaly_contamination,
        mlflow_uri=args.mlflow_uri,
    )

    print("\n=== Training Fault Classifier ===")
    train_fault_classifier(
        args.features,
        n_estimators=args.fault_estimators,
        mlflow_uri=args.mlflow_uri,
    )

    print("\nDone. Models registered in MLflow.")
    print(f"  Tracking URI: {mlflow.get_tracking_uri()}")


if __name__ == "__main__":
    main()
