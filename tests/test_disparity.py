"""Tests for the disparity ratio monitor."""

import tempfile
from pathlib import Path

import pytest

from output.audit_log import AuditLog
from output.disparity_monitor import (
    calculate_disparity_ratios,
    disparity_summary_table,
)
from risk.models import RiskLevel, RiskOutput


def make_output(
    ticker: str,
    risk_level: RiskLevel,
    industry: str,
    sector: str = "Financials",
    market_cap_bucket: str = "Large",
) -> RiskOutput:
    return RiskOutput(
        company=f"{ticker} Corp",
        ticker=ticker,
        assessment_date="2022-11-05T00:00:00",
        risk_level=risk_level,
        confidence_score=0.8,
        industry=industry,
        sector=sector,
        market_cap_bucket=market_cap_bucket,
    )


@pytest.fixture
def tmp_log():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = Path(f.name)
    log = AuditLog(db_path=db_path)
    yield log
    db_path.unlink(missing_ok=True)


def test_empty_log_returns_no_results(tmp_log):
    results = calculate_disparity_ratios(tmp_log)
    assert results == []


def test_equal_flagging_rates_no_alert(tmp_log):
    # 2 Banks ELEVATED, 2 Insurers ELEVATED — equal rates → no alert
    for i in range(2):
        tmp_log.write(make_output(f"BANK{i}", RiskLevel.ELEVATED, "Banks"))
        tmp_log.write(make_output(f"INS{i}", RiskLevel.ELEVATED, "Insurance"))

    results = calculate_disparity_ratios(tmp_log)
    industry_results = [r for r in results if r.dimension == "industry"]
    # All ratios should be 1.0 — no alerts
    assert all(not r.is_flagged or r.low_sample_warning for r in industry_results)


def test_disproportionate_flagging_triggers_alert(tmp_log):
    # Banks: 3 out of 3 ELEVATED (100% rate)
    # Insurers: 0 out of 3 LOW (0% rate)
    # Disparity ratio = infinity → should trigger alert
    for i in range(3):
        tmp_log.write(make_output(f"BANK{i}", RiskLevel.ELEVATED, "Banks"))
        tmp_log.write(make_output(f"INS{i}", RiskLevel.LOW, "Insurance"))

    results = calculate_disparity_ratios(tmp_log)
    industry_results = [
        r for r in results
        if r.dimension == "industry" and not r.low_sample_warning
    ]
    flagged = [r for r in industry_results if r.is_flagged]
    assert len(flagged) >= 1


def test_low_sample_warning_with_small_groups(tmp_log):
    # Only 1 company per group — should have low_sample_warning
    tmp_log.write(make_output("BANK1", RiskLevel.HIGH, "Banks"))
    tmp_log.write(make_output("INS1", RiskLevel.LOW, "Insurance"))

    results = calculate_disparity_ratios(tmp_log)
    if results:
        assert all(r.low_sample_warning for r in results)


def test_disparity_summary_table_output(tmp_log):
    for i in range(3):
        tmp_log.write(make_output(f"BANK{i}", RiskLevel.ELEVATED, "Banks"))
        tmp_log.write(make_output(f"INS{i}", RiskLevel.LOW, "Insurance"))

    results = calculate_disparity_ratios(tmp_log)
    table = disparity_summary_table(results)
    assert isinstance(table, list)
    if table:
        assert "Dimension" in table[0]
        assert "Disparity Ratio" in table[0]
        assert "Alert" in table[0]


def test_market_cap_bucket_dimension(tmp_log):
    # Large cap: 3 ELEVATED; Small cap: 3 LOW
    for i in range(3):
        tmp_log.write(make_output(f"LG{i}", RiskLevel.ELEVATED, "Banks", market_cap_bucket="Large"))
        tmp_log.write(make_output(f"SM{i}", RiskLevel.LOW, "Banks", market_cap_bucket="Small"))

    results = calculate_disparity_ratios(tmp_log)
    cap_results = [r for r in results if r.dimension == "market_cap_bucket"]
    assert len(cap_results) > 0
