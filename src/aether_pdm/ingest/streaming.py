"""
Streaming ingest: MQTT vibration payloads -> windowed features -> sink.

Consumes JSON messages on ``aether/assets/{asset_id}/vibration`` of the form::

    {"waveform": [...], "sampling_rate": 12000, "rpm": 1772}

runs the existing signal pipeline (``aether_pdm.signal.pipeline.process_waveform``)
to produce feature rows per window, and forwards them to a :class:`FeatureSink`
(e.g. Parquet) for persistence. An optional :class:`InferenceEngine` can score
each waveform as it arrives (only the first window, matching the engine's own
behaviour).

Design decisions (R4):
- **paho-mqtt 2.x** for transport; imports are lazy inside methods so the
  package import stays light and tests never need a real broker.
- **Topic -> asset_id**: the segment aligned with the ``+`` placeholder in the
  topic pattern (default ``aether/assets/+/vibration``) is the asset id, so
  custom patterns work without code changes.
- **max_buffer**: hard cap on accepted waveform length. Protects the consumer
  from a misconfigured/broken publisher sending multi-MB payloads.
- **Dropout resilience**: waveforms containing NaN/Inf, non-numeric values,
  zero/negative sampling rates, or shorter than one window are logged and
  skipped -- the consumer never crashes on bad data.
- **ParquetSink**: read-existing + concat + write under a lock. Simpler and
  more column-order tolerant than pyarrow's append mode, and correct under
  concurrent paho callbacks.
- **Timestamps**: one UTC timestamp per received message (``received_at``),
  shared by all windows of that message.

Usage:
    python scripts/run_streaming_consumer.py --config configs/streaming.yaml --sink parquet
"""

import json
import logging
import threading
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from aether_pdm.signal.pipeline import FEATURE_VERSION, process_waveform

if TYPE_CHECKING:
    from aether_pdm.serve.inference import InferenceEngine

logger = logging.getLogger(__name__)


class StreamingConfig(BaseModel):
    """Configuration for the MQTT streaming consumer."""

    broker_host: str = "localhost"
    broker_port: int = 1883
    topic_pattern: str = "aether/assets/+/vibration"
    window_size: int = Field(default=2048, ge=64)
    overlap: float = Field(default=0.5, ge=0.0, lt=1.0)
    max_buffer: int = Field(default=262_144, ge=1)  # ~22 s at 12 kHz
    qos: int = Field(default=1, ge=0, le=2)


class FeatureSink(Protocol):
    """Persists feature rows for one asset."""

    def write(self, asset_id: str, df: pd.DataFrame) -> None: ...


class ParquetSink:
    """Appends feature rows to ``{output_dir}/{asset_id}.parquet``."""

    def __init__(self, output_dir: Path | str) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def write(self, asset_id: str, df: pd.DataFrame) -> None:
        """Append ``df`` to the per-asset Parquet file (read + concat + write)."""
        if df.empty:
            return
        path = self.output_dir / f"{asset_id}.parquet"
        with self._lock:
            if path.exists():
                existing = pd.read_parquet(path)
                merged = pd.concat([existing, df], ignore_index=True)
            else:
                merged = df
            merged.to_parquet(path, index=False)


class NullSink:
    """In-memory no-op sink for tests: records every write."""

    def __init__(self) -> None:
        self.writes: list[tuple[str, pd.DataFrame]] = []

    def write(self, asset_id: str, df: pd.DataFrame) -> None:
        if not df.empty:
            self.writes.append((asset_id, df.copy()))


def asset_id_from_topic(topic: str, topic_pattern: str) -> str | None:
    """Extract the asset id from a topic using the ``+`` placeholder position.

    Returns ``None`` if the topic does not match the pattern's segment layout.
    """
    pattern_parts = topic_pattern.split("/")
    topic_parts = topic.split("/")
    if len(pattern_parts) != len(topic_parts):
        return None
    asset_id: str | None = None
    for pat, part in zip(pattern_parts, topic_parts, strict=True):
        if pat == "+":
            asset_id = part
        elif pat != part:
            return None
    return asset_id


def _valid_payload(
    raw: bytes, config: StreamingConfig
) -> tuple[np.ndarray, float, float | None] | None:
    """Validate and normalize a raw MQTT payload.

    Returns ``(waveform, sampling_rate, rpm)`` or ``None`` when the message is
    malformed (logged at warning level, caller skips).
    """
    try:
        payload = json.loads(raw)
    except (ValueError, TypeError):
        logger.warning("Skipping message: invalid JSON payload")
        return None
    if not isinstance(payload, dict):
        logger.warning("Skipping message: payload is not a JSON object")
        return None

    if "waveform" not in payload or "sampling_rate" not in payload:
        logger.warning(
            "Skipping message: missing required keys (need 'waveform' and 'sampling_rate')"
        )
        return None

    try:
        waveform = np.asarray(payload["waveform"], dtype=float)
    except (ValueError, TypeError):
        logger.warning("Skipping message: waveform is not numeric")
        return None
    if waveform.ndim != 1 or waveform.size == 0:
        logger.warning("Skipping message: waveform must be a non-empty 1D sequence")
        return None
    if waveform.size > config.max_buffer:
        logger.warning(
            "Skipping message: waveform length %d exceeds max_buffer %d",
            waveform.size,
            config.max_buffer,
        )
        return None
    if not np.isfinite(waveform).all():
        logger.warning("Skipping message: waveform contains NaN/Inf (sensor dropout)")
        return None
    if waveform.size < config.window_size:
        logger.warning(
            "Skipping message: waveform length %d < window_size %d",
            waveform.size,
            config.window_size,
        )
        return None

    try:
        sampling_rate = float(payload["sampling_rate"])
    except (ValueError, TypeError):
        logger.warning("Skipping message: sampling_rate is not numeric")
        return None
    if not np.isfinite(sampling_rate) or sampling_rate <= 0:
        logger.warning("Skipping message: sampling_rate must be positive")
        return None

    rpm: float | None = None
    if "rpm" in payload:
        try:
            rpm = float(payload["rpm"])
        except (ValueError, TypeError):
            logger.warning("Message has non-numeric rpm; treating as unknown")
            rpm = None
        if rpm is not None and (not np.isfinite(rpm) or rpm <= 0):
            rpm = None

    return waveform, sampling_rate, rpm


class StreamingConsumer:
    """MQTT consumer: waveform messages -> feature rows -> sink (+ optional scoring)."""

    def __init__(
        self,
        config: StreamingConfig,
        sink: FeatureSink,
        engine: "InferenceEngine | None" = None,
        alert_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.config = config
        self.sink = sink
        self.engine = engine
        self.alert_callback = alert_callback
        self._client: Any = None

    def on_message(self, client: Any, userdata: Any, msg: Any) -> None:
        """paho callback: decode, validate, window -> features, write + score."""
        asset_id = asset_id_from_topic(msg.topic, self.config.topic_pattern)
        if asset_id is None:
            logger.warning("Skipping message on unmatchable topic: %s", msg.topic)
            return

        parsed = _valid_payload(msg.payload, self.config)
        if parsed is None:
            return
        waveform, sampling_rate, rpm = parsed

        df = process_waveform(
            waveform,
            sampling_rate,
            rpm=rpm,
            window_size=self.config.window_size,
            overlap=self.config.overlap,
        )
        if df.empty:
            logger.warning("No windows generated for asset %s; skipping", asset_id)
            return

        df["asset_id"] = asset_id
        df["sampling_rate"] = sampling_rate
        df["rpm"] = rpm if rpm is not None else np.nan
        df["received_at"] = pd.Timestamp.now(tz="UTC")
        df["feature_version"] = FEATURE_VERSION

        self.sink.write(asset_id, df)

        if self.engine is not None:
            self._score_and_alert(asset_id, waveform, sampling_rate, rpm)

    def _score_and_alert(
        self, asset_id: str, waveform: np.ndarray, sampling_rate: float, rpm: float | None
    ) -> None:
        """Score the waveform with the inference engine; never raise into the loop."""
        engine = self.engine
        if engine is None:
            return
        try:
            if not getattr(engine, "model_available", False):
                logger.info("Engine has no models loaded; scoring skipped for %s", asset_id)
                return
            result = engine.score(waveform, sampling_rate, rpm)
        except Exception as exc:  # noqa: BLE001 - isolate engine failures from ingest
            logger.warning("Scoring failed for asset %s: %s", asset_id, exc)
            return

        alert = result.get("alert", {})
        level = alert.get("level", "healthy")
        log = logger.warning if level != "healthy" else logger.info
        log(
            "Asset %s | health=%.3f anomaly=%.3f fault=%s conf=%.3f alert=%s",
            asset_id,
            result.get("health_score", float("nan")),
            result.get("anomaly_score", float("nan")),
            result.get("fault", {}).get("class", "unknown"),
            result.get("fault", {}).get("confidence", 0.0),
            level,
        )
        if self.alert_callback is not None:
            try:
                self.alert_callback(result)
            except Exception as exc:  # noqa: BLE001 - alert hooks must not break ingest
                logger.warning("alert_callback failed for asset %s: %s", asset_id, exc)

    def start(self) -> None:
        """Connect to the broker and start the paho network loop."""
        # Lazy import keeps `import aether_pdm.ingest.streaming` free of paho.
        import paho.mqtt.client as mqtt

        client = mqtt.Client()
        client.on_connect = self._on_connect
        client.on_message = self.on_message
        result = client.connect(self.config.broker_host, self.config.broker_port, keepalive=60)
        if result != 0:
            raise ConnectionError(
                f"MQTT connect to {self.config.broker_host}:{self.config.broker_port} "
                f"failed with code {result}"
            )
        client.loop_start()
        self._client = client
        logger.info(
            "Streaming consumer connected to %s:%s, topic pattern '%s'",
            self.config.broker_host,
            self.config.broker_port,
            self.config.topic_pattern,
        )

    def _on_connect(
        self, client: Any, userdata: Any, flags: Any, reason_code: Any, properties: Any = None
    ) -> None:
        """Subscribe on connect (survives broker reconnects)."""
        client.subscribe(self.config.topic_pattern, qos=self.config.qos)
        logger.info("Subscribed to '%s' (qos=%d)", self.config.topic_pattern, self.config.qos)

    def stop(self) -> None:
        """Stop the paho network loop and disconnect."""
        if self._client is not None:
            self._client.loop_stop()
            self._client.disconnect()
            self._client = None
            logger.info("Streaming consumer stopped")


def load_config(path: Path | str) -> StreamingConfig:
    """Load a streaming config from YAML (nested ``broker``/``signal`` layout)."""
    import yaml

    with Path(path).open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    broker = raw.get("broker", {})
    signal = raw.get("signal", {})
    return StreamingConfig(
        broker_host=broker.get("host", "localhost"),
        broker_port=int(broker.get("port", 1883)),
        topic_pattern=raw.get("topic_pattern", "aether/assets/+/vibration"),
        qos=int(raw.get("qos", 1)),
        window_size=int(signal.get("window_size", 2048)),
        overlap=float(signal.get("overlap", 0.5)),
        max_buffer=int(raw.get("max_buffer", 262_144)),
    )


def main() -> None:
    """CLI entrypoint for the streaming consumer (see scripts/run_streaming_consumer.py)."""
    import argparse
    import time

    parser = argparse.ArgumentParser(description="AetherPdM MQTT streaming ingest")
    parser.add_argument("--config", type=Path, default=Path("configs/streaming.yaml"))
    parser.add_argument("--sink", choices=["parquet", "null"], default="parquet")
    parser.add_argument("--sink-dir", type=Path, default=Path("data/streaming/features"))
    parser.add_argument("--score", action="store_true", help="Attach InferenceEngine scoring")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    config = load_config(args.config)

    if args.sink == "null":
        sink: FeatureSink = NullSink()
        logger.info("Using NullSink (no persistence)")
    else:
        sink = ParquetSink(args.sink_dir)
        logger.info("Writing features to %s", args.sink_dir)

    engine: Any = None
    if args.score:
        from aether_pdm.serve.inference import InferenceEngine

        engine = InferenceEngine()
        if not engine.model_available:
            logger.warning("--score requested but no models available; running ingest-only")
            engine = None

    consumer = StreamingConsumer(config=config, sink=sink, engine=engine)
    try:
        consumer.start()
        logger.info("Running. Press Ctrl+C to stop.")
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        logger.info("Shutdown requested")
    finally:
        consumer.stop()


if __name__ == "__main__":
    main()
