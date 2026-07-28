"""
Inference engine: loads models from MLflow and scores waveforms.

Usage:
    engine = InferenceEngine(mlflow_uri="sqlite:///mlflow.db")
    result = engine.score(waveform=np.array(...), sampling_rate=12000, rpm=1772)
"""

from pathlib import Path
from typing import Any

import mlflow
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

from aether_pdm.signal.features import compute_all_features
from aether_pdm.signal.window import sliding_windows

MODEL_ANOMALY = "aether-anomaly"
MODEL_FAULT = "aether-fault-clf"
FEATURE_VERSION = "v1"


_VERSION_ATTRS: dict[str, str] = {
    MODEL_ANOMALY: "anomaly_version",
    MODEL_FAULT: "fault_version",
}


class InferenceEngine:
    """Loads models from MLflow and runs inference on vibration waveforms."""

    def __init__(
        self,
        mlflow_uri: str = "sqlite:///mlflow.db",
        window_size: int = 2048,
        overlap: float = 0.5,
        anomaly_stage: str = "production",
        fault_stage: str = "production",
    ):
        mlflow.set_tracking_uri(mlflow_uri)
        self.client = mlflow.tracking.MlflowClient()
        self.window_size = window_size
        self.overlap = overlap
        self._load_models(anomaly_stage, fault_stage)

    def _load_models(self, anomaly_stage: str, fault_stage: str) -> None:
        """Load models from MLflow model registry."""
        self.anomaly_model = self._load_model(MODEL_ANOMALY, anomaly_stage)
        self.fault_model = self._load_model(MODEL_FAULT, fault_stage)

    def _load_model(self, name: str, stage: str):
        versions = self.client.get_latest_versions(name, stages=[stage])
        if not versions:
            versions = self.client.get_latest_versions(name, stages=["None"])
        if not versions:
            versions = self.client.search_model_versions(f"name='{name}'", max_results=1)
        if not versions:
            raise RuntimeError(f"No versions found for model '{name}'")
        model = mlflow.sklearn.load_model(versions[0].source)
        attr = _VERSION_ATTRS.get(name, f"{name}_version")
        setattr(self, attr, versions[0].version)
        # Read fault classes from run params if available
        if name == MODEL_FAULT:
            run = self.client.get_run(versions[0].run_id)
            classes_param = run.data.params.get("classes", "")
            if classes_param:
                self.fault_classes = classes_param.split(",")
        return model

        # Load label encoder from fault model classes
        self.fault_classes = self.fault_model.classes_.tolist()

    def score(
        self,
        waveform: np.ndarray,
        sampling_rate: float,
        rpm: float | None = None,
    ) -> dict[str, Any]:
        """
        Score a single vibration waveform.

        Returns a dict matching the ScoreResponse schema.
        """
        windows, _ = sliding_windows(waveform, self.window_size, self.overlap)
        if windows.shape[0] == 0:
            return {
                "health_score": 1.0,
                "anomaly_score": 0.0,
                "fault": {"class": "unknown", "confidence": 0.0},
                "alert": {"level": "healthy", "reason": "signal_too_short"},
                "top_features": [],
                "model_versions": {
                    "anomaly": getattr(self, "anomaly_version", "?"),
                    "fault": getattr(self, "fault_version", "?"),
                },
            }

        # Compute features for the first window
        features = compute_all_features(windows[0], sampling_rate, rpm)
        feature_values = np.array([[v for v in features.values()]])

        # Anomaly score
        # IsolationForest decision_function: positive = normal, negative = anomaly
        anomaly_raw = self.anomaly_model.decision_function(feature_values)[0]
        anomaly_score = float(1.0 / (1.0 + np.exp(anomaly_raw)))
        is_anomaly = int(anomaly_raw < 0)

        # Fault classification
        fault_probs = self.fault_model.predict_proba(feature_values)[0]
        fault_idx = int(np.argmax(fault_probs))
        fault_class = self.fault_classes[fault_idx]
        fault_confidence = float(fault_probs[fault_idx])

        # Map anomaly score to health (0 = bad, 1 = good)
        health_score = float(np.clip(1.0 - anomaly_score, 0, 1))

        # Determine alert level
        if is_anomaly and fault_class != "normal":
            alert_level = "critical"
            alert_reason = f"detected_{fault_class}_fault"
        elif is_anomaly:
            alert_level = "warning"
            alert_reason = "elevated_anomaly_score"
        else:
            alert_level = "healthy"
            alert_reason = None

        # Top features by contribution (absolute value)
        top_features = sorted(
            [{"name": k, "contribution": float(abs(v))} for k, v in features.items()],
            key=lambda x: x["contribution"],
            reverse=True,
        )[:5]

        return {
            "health_score": health_score,
            "anomaly_score": float(anomaly_score),
            "fault": {"class": fault_class, "confidence": fault_confidence},
            "alert": {"level": alert_level, "reason": alert_reason},
            "top_features": top_features,
            "model_versions": {
                "anomaly": getattr(self, "anomaly_version", "?"),
                "fault": getattr(self, "fault_version", "?"),
            },
        }
