"""Tests for Pydantic risk models."""

import pytest
from datetime import datetime

from risk.models import (
    RiskLevel,
    RiskOutput,
    Signal,
    SignalType,
    SourceRef,
)


def test_risk_level_ordering():
    assert RiskLevel.HIGH > RiskLevel.ELEVATED
    assert RiskLevel.ELEVATED > RiskLevel.LOW
    assert not (RiskLevel.LOW > RiskLevel.HIGH)


def test_risk_level_max():
    assert RiskLevel.max(RiskLevel.HIGH, RiskLevel.LOW) == RiskLevel.HIGH
    assert RiskLevel.max(RiskLevel.LOW, RiskLevel.ELEVATED) == RiskLevel.ELEVATED
    assert RiskLevel.max(RiskLevel.LOW, RiskLevel.LOW) == RiskLevel.LOW


def test_signal_construction():
    sig = Signal(
        signal_type=SignalType.BALANCE_SHEET,
        description="HTM losses exceed equity",
        severity=RiskLevel.HIGH,
        triggered_by_rule=True,
        metric_name="htm_loss_equity_ratio",
        metric_value=0.97,
    )
    assert sig.severity == RiskLevel.HIGH
    assert sig.triggered_by_rule is True


def test_source_ref_optional_fields():
    ref = SourceRef(document_type="10-Q", document_date="2022-11-04")
    assert ref.section is None
    assert ref.excerpt is None


def test_risk_output_construction():
    output = RiskOutput(
        company="SVB Financial Group",
        ticker="SIVB",
        assessment_date=datetime(2022, 11, 5).isoformat(),
        risk_level=RiskLevel.ELEVATED,
        confidence_score=0.87,
    )
    assert output.human_review_status == "PENDING"
    assert output.is_elevated_or_high is True


def test_risk_output_low_is_not_elevated():
    output = RiskOutput(
        company="Test Corp",
        ticker="TEST",
        assessment_date="2024-01-01T00:00:00",
        risk_level=RiskLevel.LOW,
        confidence_score=0.9,
    )
    assert output.is_elevated_or_high is False


def test_risk_output_confidence_bounds():
    with pytest.raises(Exception):
        RiskOutput(
            company="Test",
            ticker="TEST",
            assessment_date="2024-01-01",
            risk_level=RiskLevel.LOW,
            confidence_score=1.5,  # out of bounds
        )


def test_risk_output_model_dump_audit():
    output = RiskOutput(
        company="SVB Financial Group",
        ticker="SIVB",
        assessment_date="2022-11-05T00:00:00",
        risk_level=RiskLevel.HIGH,
        confidence_score=0.85,
        rule_engine_flags=["htm_loss_equity_ratio"],
    )
    data = output.model_dump_audit()
    assert data["risk_level"] == "HIGH"
    assert isinstance(data["rule_engine_flags"], list)
