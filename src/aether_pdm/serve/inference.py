"""
Inference engine: loads models from MLflow and scores waveforms.

Usage:
    engine = InferenceEngine(mlflow_uri="sqlite:///mlflow.db")
    result = engine.score(waveform=np.array(...), sampling_rate=12000, rpm=1772)
"""

import json
import logging
from typing import Any

import mlflow
import numpy as np

from aether_pdm.signal.features import compute_all_features
from aether_pdm.signal.window import sliding_windows

MODEL_ANOMALY = "aether-anomaly"
MODEL_FAULT = "aether-fault-clf"
FEATURE_VERSION = "v2"


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
        anomaly_alias: str = "production",
        fault_alias: str = "production",
    ):
        mlflow.set_tracking_uri(mlflow_uri)
        self.mlflow_uri = mlflow_uri
        self._manifests: dict[str, dict | None] = {}
        try:
            self.client = mlflow.tracking.MlflowClient()
            self._load_models(anomaly_alias, fault_alias)
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.warning("Failed to load models from MLflow (%s): %s", mlflow_uri, e)
            self.anomaly_model = None
            self.fault_model = None
            self.model_available = False
        else:
            self.model_available = True
        self.window_size = window_size
        self.overlap = overlap

    def _load_models(self, anomaly_alias: str, fault_alias: str) -> None:
        """Load models from the MLflow model registry by alias."""
        self.anomaly_model = self._load_model(MODEL_ANOMALY, anomaly_alias)
        self.fault_model = self._load_model(MODEL_FAULT, fault_alias)

    def _load_model(self, name: str, alias: str):
        """Load the version an alias points at, falling back to the latest."""
        try:
            version = self.client.get_model_version_by_alias(name, alias)
        except Exception:
            versions = self.client.search_model_versions(
                f"name='{name}'",
                order_by=["version_number DESC"],
                max_results=1,
            )
            if not versions:
                raise RuntimeError(f"No versions found for model '{name}'")
            version = versions[0]
        model = mlflow.sklearn.load_model(version.source)
        attr = _VERSION_ATTRS.get(name, f"{name}_version")
        setattr(self, attr, version.version)
        self._manifests[name] = self._read_manifest_tag(name, version.version)
        # Read fault classes from run params if available
        if name == MODEL_FAULT:
            run_id = version.run_id
            if run_id is None:
                raise RuntimeError(f"No tracking run for model '{name}'")
            run = self.client.get_run(run_id)
            classes_param = run.data.params.get("classes", "")
            if classes_param:
                self.fault_classes = classes_param.split(",")
        return model

    def _read_manifest_tag(self, name: str, version: Any) -> dict | None:
        """Read the GatedOps lineage manifest tag recorded at promotion time."""
        try:
            version_obj = self.client.get_model_version(name, str(version))
            raw = version_obj.tags.get("gatedops.manifest")
        except Exception:
            return None
        if not raw:
            return None
        try:
            return json.loads(raw)
        except ValueError:
            return None

    def _lineage(self, name: str) -> dict[str, str | None] | None:
        """Map a stored manifest to the lineage fields echoed on every score."""
        manifest = self._manifests.get(name)
        if not manifest:
            return None
        return {
            "model_name": manifest.get("model_name"),
            "model_version": manifest.get("model_version"),
            "artifact_hash": manifest.get("artifact_hash"),
            "git_sha": manifest.get("git_sha"),
            "run_id": manifest.get("run_id"),
            "data_hash": manifest.get("data_hash"),
        }

    def _lineage_block(self) -> dict[str, dict[str, str | None] | None]:
        return {
            "anomaly": self._lineage(MODEL_ANOMALY),
            "fault": self._lineage(MODEL_FAULT),
        }

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
        if not self.model_available:
            raise RuntimeError("Models not loaded. Train models first or check MLflow connection.")
        assert self.anomaly_model is not None
        assert self.fault_model is not None

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
                "lineage": self._lineage_block(),
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
        feat_list: list[dict[str, Any]] = []
        for k, v in features.items():
            feat_list.append({"name": k, "contribution": float(abs(v))})
        feat_list.sort(key=lambda x: x["contribution"], reverse=True)
        top_features = feat_list[:5]

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
            "lineage": self._lineage_block(),
        }
