"""Tests for MLflow promotion gate."""

import contextlib
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

from aether_pdm.ops.promote import evaluate_anomaly_candidate, evaluate_fault_candidate


def _write_features(tmp_path, split="val", n_normal=20, n_faulty=10):
    rows = []
    for _ in range(n_normal):
        rows.append({"rms": 0.1, "kurtosis": 3.0, "crest": 4.0, "skew": 0.0,
                     "fault_type": "normal", "split": split})
    for _ in range(n_faulty):
        rows.append({"rms": 0.5, "kurtosis": 8.0, "crest": 6.0, "skew": 0.5,
                     "fault_type": "inner_race", "split": split})
    df = pd.DataFrame(rows)
    path = tmp_path / "features.parquet"
    df.to_parquet(path, index=False)
    return path


def _mock_mlflow(monkeypatch):
    """Patch MLflow client/run APIs so promotion tests never hit a real server."""
    import aether_pdm.ops.promote as promote_mod

    transitions: list[tuple[str, str, str]] = []

    class FakeClient:
        def transition_model_version_stage(self, name, version, stage):
            transitions.append((name, str(version), stage))
            return None

    monkeypatch.setattr(promote_mod.mlflow, "set_tracking_uri", lambda uri: None)
    monkeypatch.setattr(promote_mod.mlflow.tracking, "MlflowClient", lambda: FakeClient())
    monkeypatch.setattr(
        promote_mod.mlflow, "start_run", lambda *args, **kwargs: contextlib.nullcontext()
    )
    monkeypatch.setattr(promote_mod.mlflow, "log_params", lambda *args, **kwargs: None)
    monkeypatch.setattr(promote_mod.mlflow, "log_metrics", lambda *args, **kwargs: None)
    return transitions


def _anomaly_metrics(**overrides):
    metrics = {
        "n_samples": 30,
        "n_faults": 10,
        "n_normal": 20,
        "false_alarm_rate": 0.05,
        "detection_rate": 0.95,
        "best_threshold": -0.1,
        "best_far": 0.05,
        "best_detection_rate": 0.95,
    }
    metrics.update(overrides)
    return metrics


def _perfect_fault_setup(tmp_path):
    """Return (model, le, features_path) for a perfect 2-class fault classifier.

    The DecisionTree is fit on the exact feature rows written by ``_write_features``
    with labels aligned to ``LabelEncoder`` alphabetical ordering
    (inner_race -> 0, normal -> 1), so evaluation is deterministic and perfect.
    """
    feat_path = _write_features(tmp_path)
    x = np.array([[0.1, 3.0, 4.0, 0.0]] * 20 + [[0.5, 8.0, 6.0, 0.5]] * 10)
    y = np.array([1] * 20 + [0] * 10)  # normal->1, inner_race->0
    model = DecisionTreeClassifier(random_state=42, max_depth=None).fit(x, y)
    le = LabelEncoder().fit(["normal", "inner_race"])
    return model, le, feat_path


def test_evaluate_anomaly_candidate(tmp_path):
    """Should return metrics dict with expected keys."""
    feat_path = _write_features(tmp_path)
    from sklearn.ensemble import IsolationForest
    x = np.random.randn(30, 4)  # fake features
    model = IsolationForest(contamination=0.1, random_state=42).fit(x)
    result = evaluate_anomaly_candidate(model, feat_path)
    assert "false_alarm_rate" in result
    assert "detection_rate" in result
    assert "best_threshold" in result
    assert "n_samples" in result
    assert result["n_samples"] == 30
    assert result["n_normal"] == 20
    assert result["n_faults"] == 10
    assert 0.0 <= result["false_alarm_rate"] <= 1.0
    assert 0.0 <= result["detection_rate"] <= 1.0


def test_evaluate_fault_candidate(tmp_path):
    """Should return f1_macro and balanced_accuracy."""
    feat_path = _write_features(tmp_path)
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder
    x = np.random.randn(30, 4)
    y = np.array([0] * 20 + [1] * 10)
    model = RandomForestClassifier(n_estimators=10, random_state=42).fit(x, y)
    le = LabelEncoder().fit(["normal", "inner_race"])
    result = evaluate_fault_candidate(model, le, feat_path)
    assert "f1_macro" in result
    assert "balanced_accuracy" in result
    assert result["n_samples"] == 30
    assert result["classes"] == ["inner_race", "normal"]


def test_evaluate_anomaly_empty_val_split(tmp_path):
    """Should raise ValueError if no val rows."""
    feat_path = _write_features(tmp_path, split="train")
    model = IsolationForest(contamination=0.1, random_state=42).fit(np.random.randn(10, 4))
    with pytest.raises(ValueError, match="split='val'"):
        evaluate_anomaly_candidate(model, feat_path)


def test_evaluate_anomaly_only_healthy_raises(tmp_path):
    """Should raise ValueError when val has no fault samples to measure recall."""
    rows = [
        {"rms": 0.1, "kurtosis": 3.0, "crest": 4.0, "skew": 0.0,
         "fault_type": "normal", "split": "val"}
        for _ in range(10)
    ]
    path = tmp_path / "healthy_only.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
    model = IsolationForest(contamination=0.1, random_state=42).fit(np.random.randn(10, 4))
    with pytest.raises(ValueError, match="fault sample"):
        evaluate_anomaly_candidate(model, path)


def test_evaluate_fault_empty_val_split(tmp_path):
    """Should raise ValueError if no val rows."""
    feat_path = _write_features(tmp_path, split="train")
    model = RandomForestClassifier(n_estimators=10, random_state=42).fit(
        np.random.randn(20, 4), np.array([0] * 10 + [1] * 10)
    )
    le = LabelEncoder().fit(["normal", "inner_race"])
    with pytest.raises(ValueError, match="split='val'"):
        evaluate_fault_candidate(model, le, feat_path)


def test_promote_anomaly_rejects_when_val_empty(monkeypatch, tmp_path):
    """promote_anomaly should raise ValueError when val split is empty."""
    import aether_pdm.ops.promote as promote_mod

    feat_path = _write_features(tmp_path, split="train")
    model = IsolationForest(contamination=0.1, random_state=42).fit(np.random.randn(10, 4))
    _mock_mlflow(monkeypatch)
    monkeypatch.setattr(promote_mod, "_load_candidate_model", lambda name, client: (model, 1))

    with pytest.raises(ValueError, match="split='val'"):
        promote_mod.promote_anomaly(features_path=feat_path)


def test_promote_anomaly_promotes_when_metrics_pass(monkeypatch, tmp_path):
    """Metrics above thresholds -> promoted, transition_model_version_stage called."""
    import aether_pdm.ops.promote as promote_mod

    feat_path = _write_features(tmp_path)
    model = IsolationForest(contamination=0.1, random_state=42).fit(np.random.randn(10, 4))
    transitions = _mock_mlflow(monkeypatch)
    monkeypatch.setattr(promote_mod, "_load_candidate_model", lambda name, client: (model, 1))
    monkeypatch.setattr(
        promote_mod, "evaluate_anomaly_candidate", lambda *a, **k: _anomaly_metrics()
    )

    result = promote_mod.promote_anomaly(features_path=feat_path)

    assert result["decision"] == "promoted"
    assert result["candidate_version"] == 1
    assert transitions == [("aether-anomaly", "1", "Production")]


def test_promote_anomaly_rejects_when_thresholds_not_met(monkeypatch, tmp_path):
    """Metrics below thresholds -> rejected, no transition call, reason logged."""
    import aether_pdm.ops.promote as promote_mod

    feat_path = _write_features(tmp_path)
    model = IsolationForest(contamination=0.1, random_state=42).fit(np.random.randn(10, 4))
    transitions = _mock_mlflow(monkeypatch)
    monkeypatch.setattr(promote_mod, "_load_candidate_model", lambda name, client: (model, 1))
    monkeypatch.setattr(
        promote_mod,
        "evaluate_anomaly_candidate",
        lambda *a, **k: _anomaly_metrics(false_alarm_rate=0.50, detection_rate=0.20),
    )

    result = promote_mod.promote_anomaly(features_path=feat_path)

    assert result["decision"] == "rejected"
    assert transitions == []
    assert "FAR" in result["reason"]
    assert "DR" in result["reason"]


def test_promote_fault_mocked(monkeypatch, tmp_path):
    """Perfect fault classifier on val split -> promoted, transition recorded."""
    import aether_pdm.ops.promote as promote_mod

    model, le, feat_path = _perfect_fault_setup(tmp_path)
    transitions = _mock_mlflow(monkeypatch)
    monkeypatch.setattr(promote_mod, "_load_candidate_model", lambda name, client: (model, 1))
    monkeypatch.setattr(
        promote_mod, "_load_fault_label_encoder", lambda client, name, version: le
    )

    result = promote_mod.promote_fault(features_path=feat_path)

    assert result["candidate_version"] == 1
    assert result["decision"] == "promoted"
    assert result["metrics"]["f1_macro"] == 1.0
    assert result["metrics"]["balanced_accuracy"] == 1.0
    assert transitions == [("aether-fault-clf", "1", "Production")]


def test_promote_fault_rejects_when_balanced_accuracy_below_gate(monkeypatch, tmp_path):
    """Should reject when f1_macro passes but balanced_accuracy fails the gate."""
    import aether_pdm.ops.promote as promote_mod

    feat_path = _write_features(tmp_path)
    model = RandomForestClassifier(n_estimators=10, random_state=42).fit(
        np.random.randn(30, 4), np.array([0] * 20 + [1] * 10)
    )
    le = LabelEncoder().fit(["normal", "inner_race"])
    transitions = _mock_mlflow(monkeypatch)
    monkeypatch.setattr(promote_mod, "_load_candidate_model", lambda name, client: (model, 1))
    monkeypatch.setattr(
        promote_mod, "_load_fault_label_encoder", lambda client, name, version: le
    )
    monkeypatch.setattr(
        promote_mod,
        "evaluate_fault_candidate",
        lambda *a, **k: {
            "n_samples": 30,
            "f1_macro": 0.99,
            "balanced_accuracy": 0.70,
            "classes": ["inner_race", "normal"],
        },
    )

    result = promote_mod.promote_fault(features_path=feat_path)

    assert result["decision"] == "rejected"
    assert transitions == []
    assert "balanced_accuracy" in result["reason"]


def test_promote_fault_rejects_when_f1_below_gate(monkeypatch, tmp_path):
    """Should reject when f1_macro fails the gate even if balanced_accuracy passes."""
    import aether_pdm.ops.promote as promote_mod

    feat_path = _write_features(tmp_path)
    model = RandomForestClassifier(n_estimators=10, random_state=42).fit(
        np.random.randn(30, 4), np.array([0] * 20 + [1] * 10)
    )
    le = LabelEncoder().fit(["normal", "inner_race"])
    transitions = _mock_mlflow(monkeypatch)
    monkeypatch.setattr(promote_mod, "_load_candidate_model", lambda name, client: (model, 1))
    monkeypatch.setattr(
        promote_mod, "_load_fault_label_encoder", lambda client, name, version: le
    )
    monkeypatch.setattr(
        promote_mod,
        "evaluate_fault_candidate",
        lambda *a, **k: {
            "n_samples": 30,
            "f1_macro": 0.50,
            "balanced_accuracy": 0.95,
            "classes": ["inner_race", "normal"],
        },
    )

    result = promote_mod.promote_fault(features_path=feat_path)

    assert result["decision"] == "rejected"
    assert transitions == []
    assert "f1_macro" in result["reason"]


def test_promote_fault_logs_numeric_metrics_only(monkeypatch, tmp_path):
    """Should drop the non-numeric 'classes' list before mlflow.log_metrics."""
    import aether_pdm.ops.promote as promote_mod

    model, le, feat_path = _perfect_fault_setup(tmp_path)
    _mock_mlflow(monkeypatch)
    logged_metrics: list[dict] = []
    monkeypatch.setattr(
        promote_mod.mlflow,
        "log_metrics",
        lambda *args, **kwargs: logged_metrics.append(args[0]),
    )
    monkeypatch.setattr(promote_mod, "_load_candidate_model", lambda name, client: (model, 1))
    monkeypatch.setattr(
        promote_mod, "_load_fault_label_encoder", lambda client, name, version: le
    )

    promote_mod.promote_fault(features_path=feat_path)

    assert logged_metrics
    logged = logged_metrics[0]
    assert "classes" not in logged
    assert "f1_macro" in logged
    assert "balanced_accuracy" in logged
    assert "n_samples" in logged


def test_numeric_metrics_drops_non_numeric():
    """Should drop lists and bools so mlflow.log_metrics never chokes."""
    from aether_pdm.ops.promote import _numeric_metrics

    result = _numeric_metrics({
        "f1_macro": 0.95,
        "balanced_accuracy": 0.92,
        "n_samples": 30,
        "classes": ["inner_race", "normal"],
        "flag": True,
    })

    assert result == {"f1_macro": 0.95, "balanced_accuracy": 0.92, "n_samples": 30.0}
    assert "classes" not in result
    assert "flag" not in result


def test_evaluate_anomaly_only_fault_raises(tmp_path):
    """Should raise ValueError when val has no normal samples to measure FAR."""
    rows = [
        {"rms": 0.5, "kurtosis": 8.0, "crest": 6.0, "skew": 0.5,
         "fault_type": "inner_race", "split": "val"}
        for _ in range(10)
    ]
    path = tmp_path / "fault_only.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
    model = IsolationForest(contamination=0.1, random_state=42).fit(np.random.randn(10, 4))

    with pytest.raises(ValueError, match="no normal samples"):
        evaluate_anomaly_candidate(model, path)


def test_evaluate_anomaly_missing_split_column_raises(tmp_path):
    """Should raise ValueError when the features file lacks a 'split' column."""
    path = tmp_path / "no_split.parquet"
    pd.DataFrame({
        "rms": [0.1, 0.5],
        "fault_type": ["normal", "inner_race"],
    }).to_parquet(path, index=False)
    model = IsolationForest(contamination=0.1, random_state=42).fit(np.random.randn(10, 4))

    with pytest.raises(ValueError, match="no 'split' column"):
        evaluate_anomaly_candidate(model, path)


def test_evaluate_fault_missing_split_column_raises(tmp_path):
    """Should raise ValueError when the features file lacks a 'split' column."""
    path = tmp_path / "no_split.parquet"
    pd.DataFrame({
        "rms": [0.1, 0.5],
        "fault_type": ["normal", "inner_race"],
    }).to_parquet(path, index=False)
    model = RandomForestClassifier(n_estimators=10, random_state=42).fit(
        np.random.randn(20, 4), np.array([0] * 10 + [1] * 10)
    )
    le = LabelEncoder().fit(["normal", "inner_race"])

    with pytest.raises(ValueError, match="no 'split' column"):
        evaluate_fault_candidate(model, le, path)


def test_evaluate_fault_no_known_classes_raises(tmp_path):
    """Should raise ValueError when the val split has no model classes."""
    rows = [
        {"rms": 0.1, "kurtosis": 3.0, "crest": 4.0, "skew": 0.0,
         "fault_type": "ball", "split": "val"}
        for _ in range(10)
    ]
    path = tmp_path / "unknown_class.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
    model = RandomForestClassifier(n_estimators=10, random_state=42).fit(
        np.random.randn(20, 4), np.array([0] * 10 + [1] * 10)
    )
    le = LabelEncoder().fit(["normal", "inner_race"])

    with pytest.raises(ValueError, match="No labeled fault samples"):
        evaluate_fault_candidate(model, le, path)


def test_evaluate_fault_single_class_raises(tmp_path):
    """Should raise ValueError when only one model class is present in val."""
    rows = [
        {"rms": 0.5, "kurtosis": 8.0, "crest": 6.0, "skew": 0.5,
         "fault_type": "inner_race", "split": "val"}
        for _ in range(10)
    ]
    path = tmp_path / "single_class.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
    model = RandomForestClassifier(n_estimators=10, random_state=42).fit(
        np.random.randn(20, 4), np.array([0] * 10 + [1] * 10)
    )
    le = LabelEncoder().fit(["normal", "inner_race"])

    with pytest.raises(ValueError, match="Only one class present"):
        evaluate_fault_candidate(model, le, path)


def test_load_candidate_model_prefers_staging(monkeypatch):
    """Should return the Staging version when one exists."""
    import aether_pdm.ops.promote as promote_mod

    class FakeClient:
        def get_latest_versions(self, name, stages=None):
            return [SimpleNamespace(version=2, source="staging-source")]

        def search_model_versions(self, query, order_by=None, max_results=None):
            return [SimpleNamespace(version=5, source="latest-source")]

    monkeypatch.setattr(
        promote_mod.mlflow,
        "sklearn",
        SimpleNamespace(load_model=lambda source: f"loaded:{source}"),
    )

    model, version = promote_mod._load_candidate_model("aether-fault-clf", FakeClient())

    assert version == 2
    assert model == "loaded:staging-source"


def test_load_candidate_model_falls_back_to_latest(monkeypatch):
    """Should fall back to the latest any-stage version when no Staging exists."""
    import aether_pdm.ops.promote as promote_mod

    class FakeClient:
        def get_latest_versions(self, name, stages=None):
            return []

        def search_model_versions(self, query, order_by=None, max_results=None):
            return [SimpleNamespace(version=7, source="latest-source")]

    monkeypatch.setattr(
        promote_mod.mlflow,
        "sklearn",
        SimpleNamespace(load_model=lambda source: f"loaded:{source}"),
    )

    model, version = promote_mod._load_candidate_model("aether-fault-clf", FakeClient())

    assert version == 7
    assert model == "loaded:latest-source"


def test_load_candidate_model_falls_back_when_registry_error(monkeypatch):
    """Should treat a missing registry as no Staging and fall back to latest."""
    from mlflow.exceptions import MlflowException

    import aether_pdm.ops.promote as promote_mod

    class FakeClient:
        def get_latest_versions(self, name, stages=None):
            raise MlflowException("registry unavailable")

        def search_model_versions(self, query, order_by=None, max_results=None):
            return [SimpleNamespace(version=3, source="latest-source")]

    monkeypatch.setattr(
        promote_mod.mlflow,
        "sklearn",
        SimpleNamespace(load_model=lambda source: f"loaded:{source}"),
    )

    model, version = promote_mod._load_candidate_model("aether-fault-clf", FakeClient())

    assert version == 3
    assert model == "loaded:latest-source"


def test_load_candidate_model_no_versions_raises():
    """Should raise ValueError when no model version exists for the name."""
    import aether_pdm.ops.promote as promote_mod

    class FakeClient:
        def get_latest_versions(self, name, stages=None):
            return []

        def search_model_versions(self, query, order_by=None, max_results=None):
            return []

    with pytest.raises(ValueError, match="No model versions found"):
        promote_mod._load_candidate_model("aether-fault-clf", FakeClient())


def test_load_fault_label_encoder_parses_classes():
    """Should rebuild a LabelEncoder from the 'classes' param logged at training."""
    import aether_pdm.ops.promote as promote_mod

    class FakeClient:
        def search_model_versions(self, query):
            return [SimpleNamespace(run_id="run-1")]

        def get_run(self, run_id):
            return SimpleNamespace(
                data=SimpleNamespace(params={"classes": "normal,inner_race,outer_race,ball"})
            )

    le = promote_mod._load_fault_label_encoder(FakeClient(), "aether-fault-clf", 1)

    assert list(le.classes_) == ["ball", "inner_race", "normal", "outer_race"]


def test_load_fault_label_encoder_version_not_found_raises():
    """Should raise ValueError when the model version cannot be found."""
    import aether_pdm.ops.promote as promote_mod

    class FakeClient:
        def search_model_versions(self, query):
            return []

    with pytest.raises(ValueError, match="Cannot find version"):
        promote_mod._load_fault_label_encoder(FakeClient(), "aether-fault-clf", 1)


def test_load_fault_label_encoder_missing_classes_raises():
    """Should raise ValueError when the 'classes' param is missing."""
    import aether_pdm.ops.promote as promote_mod

    class FakeClient:
        def search_model_versions(self, query):
            return [SimpleNamespace(run_id="run-1")]

        def get_run(self, run_id):
            return SimpleNamespace(data=SimpleNamespace(params={}))

    with pytest.raises(ValueError, match="No 'classes' param"):
        promote_mod._load_fault_label_encoder(FakeClient(), "aether-fault-clf", 1)
