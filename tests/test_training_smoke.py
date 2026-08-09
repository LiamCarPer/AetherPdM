"""End-to-end training smoke test: synthetic data → features → models → inference."""

from pathlib import Path

import pytest
from sklearn.ensemble import IsolationForest, RandomForestClassifier

from aether_pdm.data.synthetic import generate_dataset, synthetic_waveform
from aether_pdm.models.anomaly import train_anomaly
from aether_pdm.models.fault import FAULT_LABELS, train_fault_classifier
from aether_pdm.serve.inference import InferenceEngine
from aether_pdm.signal.pipeline import FEATURE_VERSION, process_dataset


@pytest.mark.slow
def test_training_smoke_full_pipeline(tmp_path: Path) -> None:
    """
    Full pipeline smoke test:
      1. Generate synthetic vibration data (10 normal + 10 faulty waveforms)
      2. Run the signal feature pipeline → Parquet with feature vectors
      3. Train anomaly detector (IsolationForest on healthy windows)
      4. Train fault classifier (RandomForest on all 4 classes)
      5. Calibrate anomaly threshold on val split (skipped — no val data)
      6. Load inference engine with trained models (bypassing MLflow registry)
      7. Score a known-faulty (inner_race) waveform → assert alert is not healthy
    """
    # ---- 1. Synthetic data ----
    mlflow_uri = "sqlite:///" + str(tmp_path / "mlruns.db").replace("\\", "/")
    data_path = generate_dataset(
        tmp_path / "synth",
        n_normal=10,
        n_faulty=10,
        seed=0,
    )

    # ---- 2. Feature pipeline ----
    feat_dir = tmp_path / "features"
    feat_dir.mkdir()
    df = process_dataset(data_path, output_dir=feat_dir, window_size=1024, overlap=0.5)
    features_path = feat_dir / f"features_{FEATURE_VERSION}.parquet"
    assert features_path.exists(), "Feature pipeline should produce output file"
    assert len(df) > 10, f"Expected >10 feature windows, got {len(df)}"
    assert "feature_version" in df.columns
    assert df["feature_version"].iloc[0] == FEATURE_VERSION

    # ---- 3. Train anomaly detector ----
    anomaly_model = train_anomaly(
        features_path,
        contamination=0.1,
        n_estimators=50,
        random_state=42,
        mlflow_uri=mlflow_uri,
    )
    assert isinstance(anomaly_model, IsolationForest)

    # ---- 4. Train fault classifier ----
    fault_model, le = train_fault_classifier(
        features_path,
        n_estimators=50,
        max_depth=8,
        random_state=42,
        mlflow_uri=mlflow_uri,
    )
    assert isinstance(fault_model, RandomForestClassifier)
    # LabelEncoder sorts alphabetically → ["ball", "inner_race", "normal", "outer_race"]
    assert set(le.classes_) == set(FAULT_LABELS), (
        f"Expected classes {sorted(FAULT_LABELS)}, got {sorted(le.classes_)}"
    )

    # ---- 5. Calibrate (anomaly) ----
    # generate_dataset now emits a "val" split (3 normal + 5 faulty with the
    # default n_normal=10, n_faulty=10), so calibration COULD run here. It is
    # intentionally skipped: the smoke test only validates training + inference,
    # and calibration is covered by test_calibrate.py / the bootstrap demo.

    # ---- 6. Inference engine (manual wiring, bypassing MLflow registry) ----
    # We wire the trained models directly into the engine without MLflow loading.
    engine = InferenceEngine(mlflow_uri=mlflow_uri, window_size=1024, overlap=0.5)
    # Directly set models to skip MLflow model loading (which would fail on empty DB)
    engine.anomaly_model = anomaly_model
    engine.fault_model = fault_model
    engine.fault_classes = list(le.classes_)
    engine.model_available = True

    # ---- 7. Score a faulty waveform ----
    faulty_waveform = synthetic_waveform(
        length=4096,
        rpm=1772,
        fault_type="inner_race",
        fault_diameter=0.021,
        seed=42,
    )
    result = engine.score(waveform=faulty_waveform, sampling_rate=12000, rpm=1772)

    # Assertions
    assert "health_score" in result
    assert "anomaly_score" in result
    assert result["fault"]["class"] is not None
    assert result["fault"]["confidence"] > 0.0
    assert result["alert"]["level"] in ("healthy", "warning", "critical")
    assert len(result["top_features"]) <= 5
    assert "anomaly" in result["model_versions"]
    assert "fault" in result["model_versions"]
    print(
        f"Smoke test passed. Alert: {result['alert']['level']}, "
        f"Fault: {result['fault']['class']} ({result['fault']['confidence']:.2f})"
    )
