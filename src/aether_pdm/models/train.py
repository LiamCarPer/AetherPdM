"""
Unified training entrypoint with config-driven training.

Trains both anomaly and fault classifiers from a features Parquet file.
Supports optional YAML config files for reproducible parameter sets.
Logs models to MLflow.

Usage:
    python -m aether_pdm.models.train --features data/interim/features/features_v1.parquet
    python -m aether_pdm.models.train --features .../features_v1.parquet
        --anomaly-config configs/train_anomaly.yaml
    python -m aether_pdm.models.train --features .../features_v1.parquet
        --fault-config configs/train_fault.yaml
"""

import argparse
from pathlib import Path

import mlflow
import yaml

from aether_pdm.models.anomaly import train_anomaly
from aether_pdm.models.fault import train_fault_classifier


def load_config(path: Path) -> dict:
    """Load a YAML config file."""
    with open(path) as f:
        return yaml.safe_load(f)


def _extract_model_params(cfg: dict) -> dict:
    """Extract model.params from a training config dict."""
    return cfg.get("model", {}).get("params", {})


def main() -> None:
    parser = argparse.ArgumentParser(description="Train AetherPdM models")
    parser.add_argument("--features", type=Path, required=True, help="Path to features Parquet")
    parser.add_argument("--mlflow-uri", type=str, default=None, help="MLflow tracking URI")
    parser.add_argument(
        "--anomaly-config", type=Path, default=None,
        help="Anomaly training config YAML (configs/train_anomaly.yaml)",
    )
    parser.add_argument(
        "--fault-config", type=Path, default=None,
        help="Fault training config YAML (configs/train_fault.yaml)",
    )
    parser.add_argument(
        "--calibrate", action="store_true", default=False,
        help="Run threshold calibration and probability scaling on val split after training",
    )
    # Legacy CLI args — used when no config file is provided
    parser.add_argument("--anomaly-contamination", type=float, default=0.05)
    parser.add_argument("--fault-estimators", type=int, default=300)
    args = parser.parse_args()

    mlflow.set_tracking_uri(args.mlflow_uri or "mlruns")

    # --- Anomaly ---
    anomaly_kwargs: dict = {"features_path": args.features, "mlflow_uri": args.mlflow_uri}
    if args.anomaly_config:
        cfg = load_config(args.anomaly_config)
        params = _extract_model_params(cfg)
        anomaly_kwargs["contamination"] = params.get("contamination", 0.05)
        anomaly_kwargs["n_estimators"] = params.get("n_estimators", 200)
        anomaly_kwargs["random_state"] = params.get("random_state", 42)
        # Optional feature column filter
        feature_list = cfg.get("features")
        if feature_list:
            anomaly_kwargs["feature_cols"] = feature_list
        # Optional split filter
        split = cfg.get("data", {}).get("split")
        if split:
            anomaly_kwargs["split"] = split
    else:
        anomaly_kwargs["contamination"] = args.anomaly_contamination

    print("=== Training Anomaly Detector ===")
    anomaly_model = train_anomaly(**anomaly_kwargs)

    # --- Fault ---
    fault_kwargs: dict = {"features_path": args.features, "mlflow_uri": args.mlflow_uri}
    if args.fault_config:
        cfg = load_config(args.fault_config)
        params = _extract_model_params(cfg)
        fault_kwargs["n_estimators"] = params.get("n_estimators", 300)
        fault_kwargs["max_depth"] = params.get("max_depth", 12)
        fault_kwargs["min_samples_leaf"] = params.get("min_samples_leaf", 4)
        fault_kwargs["class_weight"] = params.get("class_weight", "balanced")
        fault_kwargs["random_state"] = params.get("random_state", 42)
        # Optional feature column filter
        feature_list = cfg.get("features")
        if feature_list:
            fault_kwargs["feature_cols"] = feature_list
        # Optional split filter
        split = cfg.get("data", {}).get("split")
        if split:
            fault_kwargs["split"] = split
    else:
        fault_kwargs["n_estimators"] = args.fault_estimators

    print("\n=== Training Fault Classifier ===")
    fault_model, fault_le = train_fault_classifier(**fault_kwargs)

    # --- Calibration (optional) ---
    if args.calibrate:
        # Lazy imports to avoid potential circular-dependency issues
        from aether_pdm.models.calibrate import calibrate_anomaly_model, calibrate_fault_model

        print("\n=== Calibrating Anomaly Model ===")
        calibrate_anomaly_model(
            anomaly_model,
            features_path=args.features,
            mlflow_uri=args.mlflow_uri,
            target_recall=0.90,
            max_far=0.10,
            split="val",
        )

        print("\n=== Calibrating Fault Model ===")
        calibrate_fault_model(
            fault_model,
            fault_le,
            features_path=args.features,
            mlflow_uri=args.mlflow_uri,
            split="val",
        )

    print("\nDone. Models registered in MLflow.")
    print(f"  Tracking URI: {mlflow.get_tracking_uri()}")


if __name__ == "__main__":
    main()
