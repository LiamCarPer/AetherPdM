"""Tests for streaming MQTT ingest (no real broker, fast)."""

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from aether_pdm.data.synthetic import synthetic_waveform
from aether_pdm.ingest.streaming import (
    NullSink,
    ParquetSink,
    StreamingConfig,
    StreamingConsumer,
    asset_id_from_topic,
)


def make_msg(topic: str, payload: bytes):
    return SimpleNamespace(topic=topic, payload=payload)


def valid_payload_bytes(length: int = 8192, seed: int = 7, **extra) -> bytes:
    w = synthetic_waveform(length=length, rpm=1772, seed=seed)
    body = {"waveform": w.tolist(), "sampling_rate": 12000, "rpm": 1772}
    body.update(extra)
    return json.dumps(body).encode()


# --- topic parsing ------------------------------------------------------------


def test_parse_topic_extracts_asset_id():
    config = StreamingConfig()
    pattern = config.topic_pattern
    assert asset_id_from_topic("aether/assets/motor-001/vibration", pattern) == "motor-001"
    assert asset_id_from_topic("aether/assets/pump-7x/vibration", pattern) == "pump-7x"
    assert asset_id_from_topic("aether/assets/motor-001/other", pattern) is None
    assert asset_id_from_topic("wrong/topic", pattern) is None
    assert asset_id_from_topic("aether/assets/only-two/vibration/extra", pattern) is None


# --- on_message: valid path ---------------------------------------------------


def test_on_message_valid_writes_features():
    config = StreamingConfig(window_size=2048, overlap=0.5)
    sink = NullSink()
    consumer = StreamingConsumer(config=config, sink=sink)

    consumer.on_message(
        None, None, make_msg("aether/assets/motor-001/vibration", valid_payload_bytes())
    )

    assert len(sink.writes) == 1
    asset_id, df = sink.writes[0]
    assert asset_id == "motor-001"
    assert len(df) > 0
    for col in [
        "rms",
        "kurtosis",
        "crest",
        "bpfo_amp",
        "window_id",
        "asset_id",
        "sampling_rate",
        "rpm",
        "received_at",
        "feature_version",
    ]:
        assert col in df.columns, f"missing column {col}"
    assert (df["asset_id"] == "motor-001").all()
    assert df["sampling_rate"].eq(12000).all()
    assert df["rpm"].eq(1772).all()


def test_on_message_valid_without_rpm():
    """rpm is optional; rows still carry fault-family features? No -- rpm=None skips them."""
    config = StreamingConfig(window_size=2048)
    sink = NullSink()
    consumer = StreamingConsumer(config=config, sink=sink)

    payload = valid_payload_bytes()
    body = json.loads(payload)
    body.pop("rpm")
    msg = make_msg("aether/assets/fan-1/vibration", json.dumps(body).encode())
    consumer.on_message(None, None, msg)

    assert len(sink.writes) == 1
    _, df = sink.writes[0]
    assert "rms" in df.columns
    assert df["rpm"].isna().all()


# --- on_message: invalid path -------------------------------------------------


@pytest.mark.parametrize(
    "payload",
    [
        b"not json at all",
        b"[]",  # not an object
        b"{}",  # missing required keys
        json.dumps({"sampling_rate": 12000}).encode(),  # missing waveform
        json.dumps({"waveform": [0.1, 0.2], "rpm": 1772}).encode(),  # missing sampling_rate
        json.dumps({"waveform": "oops", "sampling_rate": 12000}).encode(),  # bad waveform type
        json.dumps({"waveform": [0.1, "x"], "sampling_rate": 12000}).encode(),  # non-numeric
        json.dumps({"waveform": [float("nan"), 1.0], "sampling_rate": 12000}).encode(),  # NaN
        json.dumps({"waveform": [1.0, 2.0], "sampling_rate": -12000}).encode(),  # bad rate
        json.dumps({"waveform": [1.0] * 5000, "sampling_rate": 12000}).encode(),  # > max_buffer
    ],
)
def test_on_message_invalid_skips(payload):
    config = StreamingConfig(window_size=1024, max_buffer=4096)
    sink = NullSink()
    consumer = StreamingConsumer(config=config, sink=sink)

    msg = make_msg("aether/assets/motor-001/vibration", payload)
    consumer.on_message(None, None, msg)  # no crash
    assert sink.writes == []


def test_on_message_unmatched_topic_skips():
    config = StreamingConfig(window_size=1024)
    sink = NullSink()
    consumer = StreamingConsumer(config=config, sink=sink)

    consumer.on_message(None, None, make_msg("aether/assets/telemetry", valid_payload_bytes()))
    assert sink.writes == []


def test_on_message_too_short_skips():
    config = StreamingConfig(window_size=2048)
    sink = NullSink()
    consumer = StreamingConsumer(config=config, sink=sink)

    w = synthetic_waveform(length=1024, rpm=1772, seed=1)  # shorter than one window
    payload = json.dumps({"waveform": w.tolist(), "sampling_rate": 12000}).encode()
    consumer.on_message(None, None, make_msg("aether/assets/motor-001/vibration", payload))
    assert sink.writes == []


# --- parquet sink -------------------------------------------------------------


def test_parquet_sink_appends(tmp_path: Path):
    sink = ParquetSink(output_dir=tmp_path)
    df1 = pd.DataFrame({"rms": [1.0, 2.0], "asset_id": ["asset-a", "asset-a"]})
    df2 = pd.DataFrame({"rms": [3.0], "asset_id": ["asset-a"]})

    sink.write("asset-a", df1)
    sink.write("asset-a", df2)

    out = pd.read_parquet(tmp_path / "asset-a.parquet")
    assert len(out) == 3
    assert out["rms"].tolist() == [1.0, 2.0, 3.0]

    # Different asset -> separate file
    sink.write("asset-b", df1)
    assert (tmp_path / "asset-b.parquet").exists()
    assert len(pd.read_parquet(tmp_path / "asset-b.parquet")) == 2


def test_parquet_sink_ignores_empty_frames(tmp_path: Path):
    sink = ParquetSink(output_dir=tmp_path)
    sink.write("asset-a", pd.DataFrame())
    assert not (tmp_path / "asset-a.parquet").exists()


# --- consumer lifecycle -------------------------------------------------------


class FakeClient:
    """Stand-in for paho.mqtt.client.Client; records calls, never touches a broker."""

    def __init__(self):
        self.calls: list[tuple] = []
        self.on_connect = None
        self.on_message = None

    def connect(self, host, port, keepalive=60):
        self.calls.append(("connect", host, port, keepalive))
        return 0

    def subscribe(self, topic, qos=0):
        self.calls.append(("subscribe", topic, qos))

    def loop_start(self):
        self.calls.append(("loop_start",))

    def loop_stop(self):
        self.calls.append(("loop_stop",))

    def disconnect(self):
        self.calls.append(("disconnect",))


def test_consumer_start_stop(monkeypatch):
    import paho.mqtt.client as mqtt

    fake = FakeClient()
    monkeypatch.setattr(mqtt, "Client", lambda *a, **k: fake)

    config = StreamingConfig(
        broker_host="broker.test",
        broker_port=1884,
        topic_pattern="aether/assets/+/vibration",
        qos=1,
    )
    consumer = StreamingConsumer(config=config, sink=NullSink())

    consumer.start()
    # paho invokes on_connect on broker CONNACK; the fake calls it manually.
    assert fake.on_connect is not None
    fake.on_connect(fake, None, None, 0, None)
    consumer.stop()

    assert ("connect", "broker.test", 1884, 60) in fake.calls
    assert ("subscribe", "aether/assets/+/vibration", 1) in fake.calls
    assert ("loop_start",) in fake.calls
    assert ("loop_stop",) in fake.calls
    assert ("disconnect",) in fake.calls


def test_consumer_connect_failure_raises(monkeypatch):
    import paho.mqtt.client as mqtt

    class FailingClient(FakeClient):
        def connect(self, host, port, keepalive=60):
            self.calls.append(("connect", host, port, keepalive))
            return 5  # MQTT_ERR_REFUSED

    monkeypatch.setattr(mqtt, "Client", lambda *a, **k: FailingClient())
    consumer = StreamingConsumer(config=StreamingConfig(), sink=NullSink())
    with pytest.raises(ConnectionError):
        consumer.start()


# --- optional engine scoring --------------------------------------------------


class FakeEngine:
    model_available = True

    def __init__(self):
        self.scores = []

    def score(self, waveform, sampling_rate, rpm=None):
        self.scores.append((waveform, sampling_rate, rpm))
        return {
            "health_score": 0.2,
            "anomaly_score": 0.8,
            "fault": {"class": "inner_race", "confidence": 0.9},
            "alert": {"level": "critical", "reason": "detected_inner_race_fault"},
        }


def test_on_message_with_engine_scores_and_alerts():
    engine = FakeEngine()
    alerts = []
    consumer = StreamingConsumer(
        config=StreamingConfig(window_size=2048),
        sink=NullSink(),
        engine=engine,
        alert_callback=alerts.append,
    )

    consumer.on_message(
        None, None, make_msg("aether/assets/motor-001/vibration", valid_payload_bytes())
    )

    assert len(engine.scores) == 1
    assert len(alerts) == 1
    assert alerts[0]["alert"]["level"] == "critical"


def test_on_message_engine_failure_does_not_crash():
    class BrokenEngine:
        model_available = True

        def score(self, waveform, sampling_rate, rpm=None):
            raise RuntimeError("model exploded")

    consumer = StreamingConsumer(
        config=StreamingConfig(window_size=2048), sink=NullSink(), engine=BrokenEngine()
    )
    consumer.on_message(
        None, None, make_msg("aether/assets/motor-001/vibration", valid_payload_bytes())
    )
    assert len(consumer.sink.writes) == 1  # features still persisted
