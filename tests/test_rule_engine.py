"""Tests for the hard-threshold rule engine."""

import pytest

from config import RuleThresholds
from risk.models import RiskLevel, SignalType
from risk.rule_engine import RuleEngine


@pytest.fixture
def engine():
    return RuleEngine()


@pytest.fixture
def svb_q3_2022_metrics():
    """Real metrics from SVB's Q3 2022 10-Q filing."""
    return {
        "ticker": "SIVB",
        "deposit_qoq_change": -0.088,       # -8.8%
        "htm_unrealized_loss": 15_900_000_000,
        "total_equity_latest": 16_300_000_000,
        "price_to_book": 0.85,
        "price_return_30d": -0.22,
        "data_as_of": "2022-11-04",
    }


def test_no_signals_for_healthy_company(engine):
    metrics = {
        "ticker": "HEALTHY",
        "deposit_qoq_change": 0.05,
        "revenue_qoq_change": 0.08,
        "price_to_book": 1.8,
        "price_return_30d": 0.03,
    }
    signals, floor = engine.check(metrics)
    assert signals == []
    assert floor == RiskLevel.LOW


def test_deposit_decline_triggers_signal(engine):
    metrics = {"ticker": "TEST", "deposit_qoq_change": -0.12}
    signals, floor = engine.check(metrics)
    assert any(s.signal_type == SignalType.FUNDING for s in signals)
    assert floor in (RiskLevel.ELEVATED, RiskLevel.HIGH)


def test_htm_ratio_triggers_signal(engine):
    metrics = {
        "ticker": "TEST",
        "htm_unrealized_loss": 9_000_000_000,
        "total_equity_latest": 10_000_000_000,  # ratio = 0.90 > 0.80 threshold
    }
    signals, floor = engine.check(metrics)
    htm_signals = [s for s in signals if s.signal_type == SignalType.BALANCE_SHEET]
    assert len(htm_signals) == 1
    assert htm_signals[0].metric_value == pytest.approx(0.9, abs=0.01)


def test_svb_q3_2022_fires_multiple_signals(engine, svb_q3_2022_metrics):
    signals, floor = engine.check(svb_q3_2022_metrics)

    signal_types = {s.signal_type for s in signals}
    assert SignalType.FUNDING in signal_types, "Deposit decline should fire"
    assert SignalType.BALANCE_SHEET in signal_types, "HTM ratio should fire"
    assert SignalType.MARKET in signal_types, "Price-to-book should fire"

    # With multiple HIGH signals, floor should be ELEVATED or HIGH
    assert floor in (RiskLevel.ELEVATED, RiskLevel.HIGH)


def test_svb_htm_ratio_is_high_severity(engine, svb_q3_2022_metrics):
    signals, _ = engine.check(svb_q3_2022_metrics)
    htm_sig = next(
        (s for s in signals if s.signal_type == SignalType.BALANCE_SHEET), None
    )
    assert htm_sig is not None
    # 97% > 90% threshold for HIGH
    assert htm_sig.severity == RiskLevel.HIGH


def test_credit_rating_downgrade_fires(engine):
    metrics = {"ticker": "TEST", "credit_rating_downgrade": True}
    signals, floor = engine.check(metrics)
    assert any(s.signal_type == SignalType.CREDIT_RATING for s in signals)
    assert floor == RiskLevel.HIGH


def test_flag_names_returns_triggered_rules(engine):
    metrics = {"ticker": "TEST", "deposit_qoq_change": -0.15}
    signals, _ = engine.check(metrics)
    flags = engine.flag_names(signals)
    assert "deposit_qoq_change" in flags


def test_custom_thresholds():
    custom = RuleThresholds(deposit_qoq_decline=-0.20)  # much more lenient
    engine = RuleEngine(thresholds=custom)
    metrics = {"ticker": "TEST", "deposit_qoq_change": -0.12}  # would fire with default
    signals, floor = engine.check(metrics)
    # With 20% threshold, -12% decline should NOT fire
    funding_signals = [s for s in signals if s.signal_type == SignalType.FUNDING]
    assert len(funding_signals) == 0
