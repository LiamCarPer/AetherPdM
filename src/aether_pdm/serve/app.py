"""
FastAPI application for AetherPdM.

Endpoints:
  POST /v1/assets/{asset_id}/score  — Score a vibration waveform
  GET  /v1/alerts                   — List recent alerts
  GET  /health                      — Health check
"""

from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(
    title="AetherPdM API",
    version="0.1.0",
    description="Predictive maintenance scoring API for rotating equipment",
)


# --- Schemas ---


class ScoreRequest(BaseModel):
    waveform: list[float] = Field(..., description="Vibration signal samples")
    sampling_rate: float = Field(..., gt=0, description="Sampling rate in Hz")
    rpm: float | None = Field(None, gt=0, description="Shaft speed in RPM")


class FaultInfo(BaseModel):
    model_config = {"populate_by_name": True}
    class_name: str = Field(..., alias="class")
    confidence: float = Field(..., ge=0, le=1)


class AlertInfo(BaseModel):
    level: str = Field(..., pattern="^(healthy|warning|critical)$")
    reason: str | None = None


class ScoreResponse(BaseModel):
    asset_id: str
    model_versions: dict[str, str]
    health_score: float = Field(..., ge=0, le=1)
    anomaly_score: float = Field(..., ge=0, le=1)
    fault: FaultInfo | None = None
    alert: AlertInfo
    top_features: list[dict[str, Any]] = []


class AlertRecord(BaseModel):
    id: int
    asset_id: str
    level: str
    reason: str | None
    timestamp: datetime
    health_score: float


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "0.1.0"
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# --- In-memory store (placeholder, replace with DB) ---

_alerts: list[AlertRecord] = []
_alert_counter: int = 0


# --- Endpoints ---


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse()


@app.post("/v1/assets/{asset_id}/score", response_model=ScoreResponse)
async def score_asset(asset_id: str, request: ScoreRequest):
    # Placeholder: return a mock score with health status
    n = len(request.waveform)
    if n == 0:
        raise HTTPException(status_code=400, detail="Empty waveform")

    health_val = 0.92
    anomaly_val = 0.12
    level = "healthy"
    reason = None

    alert = AlertInfo(level=level, reason=reason)
    fault = FaultInfo(class_name="normal", confidence=0.95)

    return ScoreResponse(
        asset_id=asset_id,
        model_versions={"anomaly": "1", "fault": "1"},
        health_score=health_val,
        anomaly_score=anomaly_val,
        fault=fault,
        alert=alert,
        top_features=[{"name": "rms", "contribution": 0.15}],
    )


@app.get("/v1/alerts", response_model=list[AlertRecord])
async def list_alerts(limit: int = 20):
    return sorted(_alerts, key=lambda a: a.timestamp, reverse=True)[:limit]
