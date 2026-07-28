"""SQLAlchemy ORM models for AetherPdM."""

from datetime import UTC, datetime

from sqlalchemy import Column, DateTime, Float, Integer, String, Text

from aether_pdm.db.database import Base


def _now() -> datetime:
    return datetime.now(UTC)


class Asset(Base):
    __tablename__ = "assets"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    asset_id = Column(String(100), unique=True, nullable=False, index=True)
    org = Column(String(100), default="default")
    plant = Column(String(100), default="default")
    asset_type = Column(String(50), nullable=True)
    rpm_nominal = Column(Float, nullable=True)
    load_nominal = Column(Float, nullable=True)
    sampling_rate = Column(Float, nullable=True)
    anomaly_threshold = Column(Float, default=0.8)
    created_at = Column(DateTime, default=_now)
    updated_at = Column(DateTime, default=_now, onupdate=_now)


class ScoreRecord(Base):
    __tablename__ = "score_records"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    asset_id = Column(String(100), nullable=False, index=True)
    health_score = Column(Float, nullable=False)
    anomaly_score = Column(Float, nullable=False)
    fault_class = Column(String(50), nullable=True)
    fault_confidence = Column(Float, nullable=True)
    model_version_anomaly = Column(String(20), nullable=True)
    model_version_fault = Column(String(20), nullable=True)
    window_size = Column(Integer, nullable=True)
    top_features = Column(Text, nullable=True)  # JSON string
    created_at = Column(DateTime, default=_now, index=True)


class Alert(Base):
    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    asset_id = Column(String(100), nullable=False, index=True)
    level = Column(String(20), nullable=False)  # healthy, warning, critical
    reason = Column(Text, nullable=True)
    health_score = Column(Float, nullable=False)
    fault_class = Column(String(50), nullable=True)
    acknowledged = Column(Integer, default=0)  # 0 = no, 1 = yes
    created_at = Column(DateTime, default=_now, index=True)
