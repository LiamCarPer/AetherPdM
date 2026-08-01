"""Tests for batch scoring + alert rules."""

from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from aether_pdm.db.database import Base
from aether_pdm.db.repository import (
    get_latest_score,
    list_alerts,
    save_alert,
    save_score,
    upsert_asset,
)
from aether_pdm.ops.batch_scorer import BatchScorer, main

# In-memory SQLite for tests
_test_engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
_test_session_local = sessionmaker(autocommit=False, autoflush=False, bind=_test_engine)


@pytest.fixture(autouse=True)
def setup_db():
    """Create tables before each test, drop after."""
    Base.metadata.create_all(bind=_test_engine)
    yield
    Base.metadata.drop_all(bind=_test_engine)


@pytest.fixture
def db_session():
    session = _test_session_local()
    try:
        yield session
    finally:
        session.close()


def _score_result(health_score=0.9, fault_class="normal", alert_level="healthy"):
    """Build a score result dict matching the InferenceEngine.score() schema."""
    return {
        "model_versions": {"anomaly": "1", "fault": "1"},
        "health_score": health_score,
        "anomaly_score": 1.0 - health_score,
        "fault": {"class": fault_class, "confidence": 0.9},
        "alert": {"level": alert_level, "reason": f"detected_{fault_class}"},
        "top_features": [],
    }


def _mock_engine(health_score=0.9, fault_class="normal", alert_level="healthy"):
    engine = MagicMock()
    engine.model_available = True
    engine.score.return_value = _score_result(health_score, fault_class, alert_level)
    return engine


def _seed_asset(db, asset_id, org="acme", rpm=1772):
    return upsert_asset(db, asset_id, org=org, plant="plant-1", rpm_nominal=rpm)


# ---------------------------------------------------------------------------
# score_asset
# ---------------------------------------------------------------------------


def test_score_asset_success(db_session):
    """Mock engine returns a valid result and the score is persisted."""
    asset = _seed_asset(db_session, "motor-1")
    db_session.commit()

    engine = _mock_engine()
    scorer = BatchScorer(engine=engine)

    result = scorer.score_asset(db_session, asset)
    assert result["health_score"] == 0.9

    # score_asset synthesized a diagnostic waveform when none was given
    _, kwargs = engine.score.call_args
    assert isinstance(kwargs["waveform"], np.ndarray)
    assert kwargs["waveform"].shape == (4096,)
    assert kwargs["rpm"] == 1772.0

    save_score(db_session, asset.asset_id, result)
    db_session.commit()

    latest = get_latest_score(db_session, "motor-1")
    assert latest is not None
    assert latest.health_score == 0.9


def test_score_asset_defaults_rpm_when_missing(db_session):
    """Assets without rpm_nominal fall back to the 1772 default."""
    asset = upsert_asset(db_session, "no-rpm", org="acme", plant="plant-1")
    db_session.commit()

    engine = _mock_engine()
    scorer = BatchScorer(engine=engine)

    scorer.score_asset(db_session, asset)
    _, kwargs = engine.score.call_args
    assert kwargs["rpm"] == 1772.0


def test_score_asset_uses_custom_rpm(db_session):
    """A custom rpm_nominal flows into the engine.score call."""
    asset = _seed_asset(db_session, "motor-1", rpm=1797)
    db_session.commit()

    engine = _mock_engine()
    scorer = BatchScorer(engine=engine)

    scorer.score_asset(db_session, asset)
    _, kwargs = engine.score.call_args
    assert kwargs["rpm"] == 1797.0


# ---------------------------------------------------------------------------
# run(): end-to-end
# ---------------------------------------------------------------------------


def test_run_scores_all_assets(db_session):
    """2 assets, mock engine, run() -> scored=2, both results present."""
    _seed_asset(db_session, "motor-1")
    _seed_asset(db_session, "motor-2")
    db_session.commit()

    scorer = BatchScorer(engine=_mock_engine())
    summary = scorer.run(db_session)

    assert summary["scored"] == 2
    assert {r["asset_id"] for r in summary["results"]} == {"motor-1", "motor-2"}
    assert all(r["alert_level"] == "healthy" for r in summary["results"])
    assert summary["alerts_raised"] == 0


@pytest.mark.parametrize("limit", [1, 2])
def test_run_respects_limit(db_session, limit):
    """run(limit=...) only scores the first N assets."""
    _seed_asset(db_session, "motor-1")
    _seed_asset(db_session, "motor-2")
    _seed_asset(db_session, "motor-3")
    db_session.commit()

    scorer = BatchScorer(engine=_mock_engine())
    summary = scorer.run(db_session, limit=limit)

    assert summary["scored"] == limit
    assert len(summary["results"]) == limit


def test_hysteresis_suppresses_first_alert(db_session):
    """hysteresis=3, score 1x non-healthy -> alert NOT raised."""
    _seed_asset(db_session, "motor-1")
    db_session.commit()

    scorer = BatchScorer(
        engine=_mock_engine(health_score=0.2, fault_class="inner_race", alert_level="critical"),
        hysteresis=3,
    )
    summary = scorer.run(db_session)

    assert summary["scored"] == 1
    assert summary["alerts_raised"] == 0
    assert summary["alerts_suppressed_by_hysteresis"] == 1
    assert list_alerts(db_session) == []
    assert summary["results"][0]["suppressed_by"] == "hysteresis"


def test_hysteresis_raises_after_n_scores(db_session):
    """hysteresis=3, score 3x non-healthy -> alert raised on the 3rd."""
    _seed_asset(db_session, "motor-1")
    db_session.commit()

    scorer = BatchScorer(
        engine=_mock_engine(health_score=0.2, fault_class="inner_race", alert_level="critical"),
        hysteresis=3,
    )
    s1 = scorer.run(db_session)
    s2 = scorer.run(db_session)
    s3 = scorer.run(db_session)

    assert s1["alerts_suppressed_by_hysteresis"] == 1
    assert s2["alerts_suppressed_by_hysteresis"] == 1
    assert s3["alerts_raised"] == 1
    assert s3["alerts_suppressed_by_hysteresis"] == 0

    alerts = list_alerts(db_session)
    assert len(alerts) == 1
    assert alerts[0].level == "critical"


def test_cooldown_suppresses_repeat_alert(db_session):
    """cooldown_min=30, prior alert created now -> suppressed_by_cooldown."""
    _seed_asset(db_session, "motor-1")
    now = datetime.now(UTC)
    seeded = save_alert(
        db_session,
        "motor-1",
        level="critical",
        reason="vibration",
        health_score=0.2,
    )
    seeded.created_at = now
    db_session.commit()

    scorer = BatchScorer(
        engine=_mock_engine(health_score=0.2, fault_class="inner_race", alert_level="critical"),
        hysteresis=1,
        cooldown_min=30,
    )
    summary = scorer.run(db_session, now=now)

    assert summary["alerts_raised"] == 0
    assert summary["alerts_suppressed_by_cooldown"] == 1
    assert summary["results"][0]["suppressed_by"] == "cooldown"
    assert len(list_alerts(db_session)) == 1


def test_cooldown_expired_allows_alert(db_session):
    """cooldown_min=30, prior alert created 60 min ago -> alert raised."""
    _seed_asset(db_session, "motor-1")
    now = datetime.now(UTC)
    seeded = save_alert(
        db_session,
        "motor-1",
        level="critical",
        reason="vibration",
        health_score=0.2,
    )
    seeded.created_at = now - timedelta(minutes=60)
    db_session.commit()

    scorer = BatchScorer(
        engine=_mock_engine(health_score=0.2, fault_class="inner_race", alert_level="critical"),
        hysteresis=1,
        cooldown_min=30,
    )
    summary = scorer.run(db_session, now=now)

    assert summary["alerts_raised"] == 1
    assert summary["alerts_suppressed_by_cooldown"] == 0
    assert len(list_alerts(db_session)) == 2


def test_healthy_score_no_alert(db_session):
    """Healthy score -> no alert of any kind."""
    _seed_asset(db_session, "motor-1")
    db_session.commit()

    scorer = BatchScorer(engine=_mock_engine())
    summary = scorer.run(db_session)

    assert summary["scored"] == 1
    assert summary["alerts_raised"] == 0
    assert summary["alerts_suppressed_by_hysteresis"] == 0
    assert summary["alerts_suppressed_by_cooldown"] == 0
    assert list_alerts(db_session) == []


def test_empty_assets_no_crash(db_session):
    """No assets -> scored=0, no crash."""
    scorer = BatchScorer(engine=_mock_engine())
    summary = scorer.run(db_session)

    assert summary["scored"] == 0
    assert summary["results"] == []
    assert summary["errors"] == []


def test_engine_error_recorded(db_session):
    """Mock engine raises RuntimeError -> errors list has entry, run completes."""
    _seed_asset(db_session, "motor-1")
    db_session.commit()

    engine = _mock_engine()
    engine.score.side_effect = RuntimeError("engine 503")
    scorer = BatchScorer(engine=engine)

    summary = scorer.run(db_session)

    assert summary["scored"] == 0
    assert len(summary["errors"]) == 1
    assert "motor-1" in summary["errors"][0]
    assert "engine 503" in summary["errors"][0]


def test_engine_no_models_error_recorded(db_session):
    """Engine with model_available=False -> each asset errors, run completes."""
    _seed_asset(db_session, "motor-1")
    db_session.commit()

    engine = _mock_engine()
    engine.model_available = False
    scorer = BatchScorer(engine=engine)

    summary = scorer.run(db_session)

    assert summary["scored"] == 0
    assert len(summary["errors"]) == 1
    assert "Models not loaded" in summary["errors"][0]


def test_org_scoping(db_session):
    """run(org="acme") only scores acme's assets."""
    _seed_asset(db_session, "acme-motor", org="acme")
    _seed_asset(db_session, "other-motor", org="other")
    db_session.commit()

    scorer = BatchScorer(engine=_mock_engine())
    summary = scorer.run(db_session, org="acme")

    assert summary["scored"] == 1
    assert [r["asset_id"] for r in summary["results"]] == ["acme-motor"]


# ---------------------------------------------------------------------------
# Alert rule internals
# ---------------------------------------------------------------------------


def test_hysteresis_satisfied_direct(db_session):
    """_hysteresis_satisfied needs N consecutive non-healthy scores."""
    _seed_asset(db_session, "motor-1")
    db_session.commit()

    scorer = BatchScorer(engine=_mock_engine(), hysteresis=2)

    # no scores yet -> not satisfied
    assert scorer._hysteresis_satisfied(db_session, "motor-1", 0.2) is False

    # 1 non-healthy score (hysteresis=2) -> still not satisfied
    save_score(db_session, "motor-1", _score_result(health_score=0.2, alert_level="warning"))
    db_session.commit()
    assert scorer._hysteresis_satisfied(db_session, "motor-1", 0.2) is False

    # 2 consecutive non-healthy scores -> satisfied
    save_score(db_session, "motor-1", _score_result(health_score=0.2, alert_level="warning"))
    db_session.commit()
    assert scorer._hysteresis_satisfied(db_session, "motor-1", 0.2) is True

    # healthy current score -> not satisfied even with recent non-healthy history
    save_score(db_session, "motor-1", _score_result(health_score=0.9))
    db_session.commit()
    assert scorer._hysteresis_satisfied(db_session, "motor-1", 0.9) is False


def test_cooldown_active_direct(db_session):
    """_cooldown_active suppresses only the same asset+level within the window."""
    _seed_asset(db_session, "motor-1")
    now = datetime.now(UTC)

    scorer = BatchScorer(engine=_mock_engine(), cooldown_min=30)

    # no prior alert -> inactive
    assert scorer._cooldown_active(db_session, "motor-1", "critical", now=now) is False

    seeded = save_alert(
        db_session,
        "motor-1",
        level="critical",
        reason="vibration",
        health_score=0.2,
    )
    seeded.created_at = now
    db_session.commit()

    assert scorer._cooldown_active(db_session, "motor-1", "critical", now=now) is True
    # different level for the same asset is NOT suppressed
    assert scorer._cooldown_active(db_session, "motor-1", "warning", now=now) is False


def test_cooldown_disabled_when_zero(db_session):
    """cooldown_min=0 disables the cooldown rule entirely."""
    _seed_asset(db_session, "motor-1")
    now = datetime.now(UTC)
    seeded = save_alert(
        db_session,
        "motor-1",
        level="critical",
        reason="vibration",
        health_score=0.2,
    )
    seeded.created_at = now
    db_session.commit()

    scorer = BatchScorer(engine=_mock_engine(), cooldown_min=0)
    assert scorer._cooldown_active(db_session, "motor-1", "critical", now=now) is False


def test_hysteresis_clamped_to_min_one():
    """hysteresis below 1 is clamped to 1."""
    scorer = BatchScorer(engine=_mock_engine(), hysteresis=0, cooldown_min=-5)
    assert scorer.hysteresis == 1
    assert scorer.cooldown_min == 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_main_runs():
    """Smoke test: main() parses args, runs the scorer, prints a summary."""
    with (
        patch("aether_pdm.ops.batch_scorer.init_db"),
        patch("aether_pdm.ops.batch_scorer.get_session") as mock_session_cm,
        patch("aether_pdm.ops.batch_scorer.BatchScorer") as mock_scorer_cls,
        patch(
            "sys.argv",
            ["run_batch_scorer.py", "--org", "acme", "--hysteresis", "3", "--cooldown-min", "30"],
        ),
    ):
        mock_session_cm.return_value.__enter__.return_value = MagicMock()
        mock_scorer_cls.return_value.run.return_value = {
            "scored": 2,
            "alerts_raised": 1,
            "alerts_suppressed_by_cooldown": 0,
            "alerts_suppressed_by_hysteresis": 1,
            "errors": ["motor-1: engine 503"],
            "results": [],
        }

        main()  # should not raise

    mock_scorer_cls.assert_called_once()
    mock_scorer_cls.return_value.run.assert_called_once()
