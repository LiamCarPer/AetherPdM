"""
Prometheus metrics for the AetherPdM API.

Exposes:
- HTTP request counters + latency histograms (per endpoint/status)
- Business counters: predictions per fault class, alerts per level
- Model info gauge (loaded model versions)

Metrics are registered on a module-level CollectorRegistry so they
survive multiple app imports (FastAPI reload, tests).
"""

import time

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

# Single registry shared across app instances
REGISTRY = CollectorRegistry()

# --- HTTP middleware metrics ---
REQUEST_COUNT = Counter(
    "aetherpdm_http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status"],
    registry=REGISTRY,
)
REQUEST_DURATION = Histogram(
    "aetherpdm_http_request_duration_seconds",
    "HTTP request latency",
    ["method", "endpoint"],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
    registry=REGISTRY,
)

# --- Business metrics ---
PREDICTIONS_TOTAL = Counter(
    "aetherpdm_predictions_total",
    "Total predictions by fault class",
    ["class"],
    registry=REGISTRY,
)
ALERTS_TOTAL = Counter(
    "aetherpdm_alerts_total",
    "Total alerts by level",
    ["level"],
    registry=REGISTRY,
)
HEALTH_SCORE_GAUGE = Gauge(
    "aetherpdm_health_score",
    "Latest health score per asset",
    ["asset_id"],
    registry=REGISTRY,
)
MODEL_VERSION = Gauge(
    "aetherpdm_model_version",
    "Loaded model version by model name",
    ["model_name"],
    registry=REGISTRY,
)


class PrometheusMiddleware:
    """ASGI middleware recording request count + duration."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        method = scope["method"]
        endpoint = scope.get("path", "unknown")

        start = time.perf_counter()
        status = "500"

        async def _send_wrapper(message):
            nonlocal status
            if message["type"] == "http.response.start":
                status = str(message["status"])
            await send(message)

        try:
            await self.app(scope, receive, _send_wrapper)
        finally:
            duration = time.perf_counter() - start
            REQUEST_COUNT.labels(method=method, endpoint=endpoint, status=status).inc()
            REQUEST_DURATION.labels(method=method, endpoint=endpoint).observe(duration)


def metrics_response():
    """Return (body, content_type) for the /metrics endpoint."""
    return generate_latest(REGISTRY), CONTENT_TYPE_LATEST
