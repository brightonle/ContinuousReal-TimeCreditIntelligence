"""Tests for the SQLite audit log."""

import json
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from output.audit_log import AuditLog
from risk.models import RiskLevel, RiskOutput, Signal, SignalType


@pytest.fixture
def tmp_audit_log():
    """Create a temporary audit log for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = Path(f.name)
    log = AuditLog(db_path=db_path)
    yield log
    db_path.unlink(missing_ok=True)


def make_output(
    ticker: str = "TEST",
    risk_level: RiskLevel = RiskLevel.ELEVATED,
    confidence: float = 0.75,
    industry: str = "Banks",
    sector: str = "Financials",
    assessment_date: str | None = None,
) -> RiskOutput:
    return RiskOutput(
        company=f"{ticker} Corp",
        ticker=ticker,
        assessment_date=assessment_date or datetime.utcnow().isoformat(),
        risk_level=risk_level,
        confidence_score=confidence,
        industry=industry,
        sector=sector,
        market_cap_bucket="Large",
        signals=[
            Signal(
                signal_type=SignalType.FUNDING,
                description="Test signal",
                severity=risk_level,
            )
        ],
        rule_engine_flags=["test_flag"],
    )


def test_write_and_count(tmp_audit_log):
    assert tmp_audit_log.count() == 0
    tmp_audit_log.write(make_output())
    assert tmp_audit_log.count() == 1


def test_write_returns_rowid(tmp_audit_log):
    rowid = tmp_audit_log.write(make_output())
    assert isinstance(rowid, int)
    assert rowid >= 1


def test_get_latest_per_company_deduplicates(tmp_audit_log):
    tmp_audit_log.write(make_output("SIVB", RiskLevel.LOW, assessment_date="2022-01-01T00:00:00"))
    tmp_audit_log.write(make_output("SIVB", RiskLevel.HIGH, assessment_date="2022-11-05T00:00:00"))
    tmp_audit_log.write(make_output("JPM", RiskLevel.LOW))

    latest = tmp_audit_log.get_latest_per_company()
    assert len(latest) == 2

    sivb = next((o for o in latest if o.ticker == "SIVB"), None)
    assert sivb is not None
    # Should return the most recent record (HIGH, Nov 2022)
    assert sivb.risk_level == RiskLevel.HIGH


def test_get_history_ordered(tmp_audit_log):
    for i in range(3):
        tmp_audit_log.write(make_output("SIVB", assessment_date=f"2022-0{i+1}-01T00:00:00"))

    history = tmp_audit_log.get_history("SIVB")
    assert len(history) == 3
    # Most recent first
    assert history[0].assessment_date > history[1].assessment_date


def test_no_update_delete_methods(tmp_audit_log):
    assert not hasattr(tmp_audit_log, "update")
    assert not hasattr(tmp_audit_log, "delete")
    assert not hasattr(tmp_audit_log, "delete_record")


def test_mark_reviewed(tmp_audit_log):
    rowid = tmp_audit_log.write(make_output())
    tmp_audit_log.mark_reviewed(rowid, "REVIEWED")
    records = tmp_audit_log.get_all_records()
    record = next(r for r in records if r["id"] == rowid)
    assert record["human_review_status"] == "REVIEWED"


def test_mark_reviewed_invalid_status(tmp_audit_log):
    rowid = tmp_audit_log.write(make_output())
    with pytest.raises(ValueError):
        tmp_audit_log.mark_reviewed(rowid, "INVALID_STATUS")


def test_get_all_records(tmp_audit_log):
    for i in range(5):
        tmp_audit_log.write(make_output(f"CO{i}"))
    records = tmp_audit_log.get_all_records()
    assert len(records) == 5
    assert all("ticker" in r for r in records)


def test_roundtrip_preserves_signals(tmp_audit_log):
    output = make_output("SIVB", RiskLevel.HIGH)
    tmp_audit_log.write(output)

    latest = tmp_audit_log.get_latest_per_company()
    assert len(latest) == 1
    restored = latest[0]
    assert len(restored.signals) == 1
    assert restored.signals[0].signal_type == SignalType.FUNDING
