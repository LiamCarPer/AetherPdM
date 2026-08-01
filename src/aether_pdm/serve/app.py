"""
FastAPI application for AetherPdM.

Endpoints:
  POST /v1/assets/{asset_id}/score  — Score a vibration waveform, persist alert
  GET  /v1/alerts                   — List recent alerts from DB
  GET  /v1/assets                   — List registered assets
  GET  /health                      — Health check
"""

import os
from collections.abc import Generator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Annotated, Any

import numpy as np
from fastapi import Depends, FastAPI, HTTPException, Response
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from aether_pdm.db.database import get_session, init_db
from aether_pdm.db.repository import (
    get_asset,
    save_alert,
    save_score,
    upsert_asset,
)
from aether_pdm.db.repository import (
    list_alerts as db_list_alerts,
)
from aether_pdm.db.repository import (
    list_assets as db_list_assets,
)
from aether_pdm.serve.inference import InferenceEngine
from aether_pdm.serve.metrics import (
    ALERTS_TOTAL,
    HEALTH_SCORE_GAUGE,
    MODEL_VERSION,
    PREDICTIONS_TOTAL,
    PrometheusMiddleware,
    metrics_response,
)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    init_db()
    yield


app = FastAPI(
    title="AetherPdM API",
    version="0.1.0",
    description="Predictive maintenance scoring API for rotating equipment",
    lifespan=lifespan,
)

app.add_middleware(PrometheusMiddleware)

# --- Inference engine (lazy-loaded) ---

_engine: InferenceEngine | None = None


def get_engine() -> InferenceEngine:
    global _engine
    if _engine is None:
        mlflow_uri = os.environ.get("AETHER_MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
        _engine = InferenceEngine(mlflow_uri=mlflow_uri)
    return _engine


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
    score_id: int | None = None


class AlertRecord(BaseModel):
    id: int
    asset_id: str
    level: str
    reason: str | None
    health_score: float
    fault_class: str | None
    acknowledged: bool
    created_at: datetime

    model_config = {"from_attributes": True}


class AssetRecord(BaseModel):
    asset_id: str
    org: str
    plant: str
    asset_type: str | None
    rpm_nominal: float | None
    anomaly_threshold: float
    created_at: datetime

    model_config = {"from_attributes": True}


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "0.1.0"
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


# --- Dependencies ---


def get_db() -> Generator[Session, None, None]:
    with get_session() as session:
        yield session


DbDep = Annotated[Session, Depends(get_db)]


# --- Endpoints ---


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse()


@app.get("/metrics", include_in_schema=False)
async def metrics():
    body, content_type = metrics_response()
    return Response(content=body, media_type=content_type)


@app.get("/v1/assets", response_model=list[AssetRecord])
async def list_assets(db: DbDep):
    return db_list_assets(db)


@app.get("/v1/assets/{asset_id}")
async def get_asset_endpoint(asset_id: str, db: DbDep):
    asset = get_asset(db, asset_id)
    if not asset:
        raise HTTPException(status_code=404, detail="Asset not found")
    return AssetRecord.model_validate(asset)


@app.post("/v1/assets/{asset_id}/score", response_model=ScoreResponse)
async def score_asset(asset_id: str, request: ScoreRequest, db: DbDep):
    n = len(request.waveform)
    if n == 0:
        raise HTTPException(status_code=400, detail="Empty waveform")

    try:
        engine = get_engine()
        result = engine.score(
            waveform=np.array(request.waveform, dtype=float),
            sampling_rate=request.sampling_rate,
            rpm=request.rpm,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    # Prometheus business metrics (only on successful scoring)
    fault_class = result.get("fault", {}).get("class", "unknown")
    PREDICTIONS_TOTAL.labels(**{"class": fault_class}).inc()
    alert_level = result["alert"]["level"]
    ALERTS_TOTAL.labels(level=alert_level).inc()
    HEALTH_SCORE_GAUGE.labels(asset_id=asset_id).set(float(result["health_score"]))
    for model_name, version in result["model_versions"].items():
        try:
            numeric_version = float(version)
        except (ValueError, TypeError):
            # `labels()` registers the child series at 0.0 BEFORE set(); calling
            # it before conversion would export a misleading "model version 0"
            # sample for semver strings like "v1.2.3-beta". Skip entirely.
            continue
        MODEL_VERSION.labels(model_name=model_name).set(numeric_version)

    # Persist score
    record = save_score(db, asset_id, result)

    # Persist alert if non-healthy
    alert = result["alert"]
    save_alert(
        db,
        asset_id,
        level=alert["level"],
        reason=alert.get("reason"),
        health_score=result["health_score"],
        fault_class=result.get("fault", {}).get("class") if alert["level"] != "healthy" else None,
    )

    # Upsert asset metadata
    upsert_asset(db, asset_id)

    return ScoreResponse(
        asset_id=asset_id,
        model_versions=result["model_versions"],
        health_score=result["health_score"],
        anomaly_score=result["anomaly_score"],
        fault=FaultInfo(**result["fault"]) if result["fault"] else None,
        alert=AlertInfo(**alert),
        top_features=result["top_features"],
        score_id=int(record.id),
    )


@app.get("/v1/alerts", response_model=list[AlertRecord])
async def list_alerts(
    db: DbDep,
    asset_id: str | None = None,
    level: str | None = None,
    limit: int = 50,
):
    return db_list_alerts(db, asset_id=asset_id, level=level, limit=limit)
