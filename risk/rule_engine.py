"""
Hard-threshold rule engine.

Fires deterministic signals when financial metrics cross defined limits.
No LLM involved — this is pure Python logic that runs before the AI layer
so that hard stops cannot be overridden by model reasoning.

The rule engine's output signals are:
  - Passed into the Claude risk prompt as established facts
  - Used to set a minimum floor on the overall risk_level
    (Claude can raise the risk level but never lower it below what rules detect)

Metrics dict expected format (from market_fetcher.py and SEC parsing):
{
    "ticker": str,
    "deposit_qoq_change": float | None,       # e.g. -0.088 for -8.8%
    "revenue_qoq_change": float | None,
    "total_equity_latest": float | None,
    "htm_unrealized_loss": float | None,       # parsed from SEC filing metadata
    "price_to_book": float | None,
    "price_return_30d": float | None,
    "tier1_capital_ratio": float | None,
    "credit_rating_downgrade": bool,           # True if news contains downgrade signal
    ...
}
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from langchain_core.documents import Document

from config import THRESHOLDS, RuleThresholds
from risk.models import RiskLevel, Signal, SignalType, SourceRef

logger = logging.getLogger(__name__)


class RuleEngine:
    """
    Evaluates financial metrics against hard thresholds and returns Signals.
    """

    def __init__(self, thresholds: RuleThresholds = THRESHOLDS) -> None:
        self._t = thresholds

    def check(
        self,
        metrics: dict,
        market_docs: Optional[list[Document]] = None,
    ) -> tuple[list[Signal], RiskLevel]:
        """
        Run all threshold rules against the provided metrics.

        Args:
            metrics:     flat dict of financial metrics (from market_fetcher)
            market_docs: raw Document objects (used to extract HTM loss from SEC)

        Returns:
            (signals, minimum_risk_level) — the overall risk floor implied
            by the rule engine alone.
        """
        signals: list[Signal] = []

        # Extract HTM unrealized loss from market_docs if available
        if market_docs:
            htm_loss = self._extract_htm_loss(market_docs)
            if htm_loss is not None:
                metrics.setdefault("htm_unrealized_loss", htm_loss)

        # ── Rule: Deposit QoQ decline ─────────────────────────────────────────
        dep_change = metrics.get("deposit_qoq_change")
        if dep_change is not None and dep_change < self._t.deposit_qoq_decline:
            pct = dep_change * 100
            signals.append(Signal(
                signal_type=SignalType.FUNDING,
                description=(
                    f"Deposit balances declined {pct:.1f}% quarter-over-quarter, "
                    f"exceeding the -{abs(self._t.deposit_qoq_decline)*100:.0f}% threshold."
                ),
                severity=RiskLevel.ELEVATED if dep_change > -0.10 else RiskLevel.HIGH,
                metric_name="deposit_qoq_change",
                metric_value=dep_change,
                triggered_by_rule=True,
                source_refs=[SourceRef(
                    document_type="market_data",
                    document_date=metrics.get("data_as_of", ""),
                    section="Balance Sheet - Deposits",
                )],
            ))

        # ── Rule: Revenue QoQ decline ─────────────────────────────────────────
        rev_change = metrics.get("revenue_qoq_change")
        if rev_change is not None and rev_change < self._t.revenue_qoq_decline:
            pct = rev_change * 100
            signals.append(Signal(
                signal_type=SignalType.EARNINGS,
                description=(
                    f"Revenue declined {pct:.1f}% quarter-over-quarter, "
                    f"exceeding the -{abs(self._t.revenue_qoq_decline)*100:.0f}% threshold."
                ),
                severity=RiskLevel.ELEVATED,
                metric_name="revenue_qoq_change",
                metric_value=rev_change,
                triggered_by_rule=True,
                source_refs=[SourceRef(
                    document_type="market_data",
                    document_date=metrics.get("data_as_of", ""),
                    section="Income Statement",
                )],
            ))

        # ── Rule: HTM unrealized loss / Total equity ──────────────────────────
        htm_loss = metrics.get("htm_unrealized_loss")
        total_equity = metrics.get("total_equity_latest")
        if htm_loss is not None and total_equity and total_equity > 0:
            ratio = htm_loss / total_equity
            if ratio > self._t.htm_loss_equity_ratio:
                signals.append(Signal(
                    signal_type=SignalType.BALANCE_SHEET,
                    description=(
                        f"HTM unrealized losses represent {ratio*100:.1f}% of total equity "
                        f"(threshold: {self._t.htm_loss_equity_ratio*100:.0f}%). "
                        "If forced to realize these losses, the company could face insolvency."
                    ),
                    severity=RiskLevel.HIGH if ratio > 0.90 else RiskLevel.ELEVATED,
                    metric_name="htm_loss_equity_ratio",
                    metric_value=ratio,
                    triggered_by_rule=True,
                    source_refs=[SourceRef(
                        document_type="10-Q",
                        document_date=metrics.get("data_as_of", ""),
                        section="Note 4/5 - Investment Securities",
                    )],
                ))

        # ── Rule: Price-to-book ───────────────────────────────────────────────
        ptb = metrics.get("price_to_book")
        if ptb is not None and ptb < self._t.price_to_book_min:
            signals.append(Signal(
                signal_type=SignalType.MARKET,
                description=(
                    f"Price-to-book ratio of {ptb:.2f}x is below the {self._t.price_to_book_min:.2f}x "
                    "floor, suggesting the market is pricing in significant asset impairment."
                ),
                severity=RiskLevel.ELEVATED,
                metric_name="price_to_book",
                metric_value=ptb,
                triggered_by_rule=True,
                source_refs=[SourceRef(
                    document_type="market_data",
                    document_date=metrics.get("data_as_of", ""),
                )],
            ))

        # ── Rule: 30-day price return ─────────────────────────────────────────
        ret_30d = metrics.get("price_return_30d")
        if ret_30d is not None and ret_30d < self._t.price_return_30d:
            pct = ret_30d * 100
            signals.append(Signal(
                signal_type=SignalType.MARKET,
                description=(
                    f"30-day stock return of {pct:.1f}% signals significant market stress, "
                    f"exceeding the {self._t.price_return_30d*100:.0f}% threshold."
                ),
                severity=RiskLevel.HIGH if ret_30d < -0.40 else RiskLevel.ELEVATED,
                metric_name="price_return_30d",
                metric_value=ret_30d,
                triggered_by_rule=True,
                source_refs=[SourceRef(
                    document_type="price_history",
                    document_date=metrics.get("data_as_of", ""),
                )],
            ))

        # ── Rule: Credit rating downgrade detected in news ────────────────────
        if metrics.get("credit_rating_downgrade", False):
            signals.append(Signal(
                signal_type=SignalType.CREDIT_RATING,
                description=(
                    "A credit rating downgrade was detected in recent news coverage. "
                    "This is a high-urgency signal requiring immediate review."
                ),
                severity=RiskLevel.HIGH,
                triggered_by_rule=True,
                source_refs=[SourceRef(
                    document_type="news",
                    document_date=metrics.get("data_as_of", ""),
                    section="Credit Rating News",
                )],
            ))

        # ── Rule: Tier-1 capital ratio ────────────────────────────────────────
        tier1 = metrics.get("tier1_capital_ratio")
        if tier1 is not None and tier1 < self._t.tier1_capital_ratio_min:
            signals.append(Signal(
                signal_type=SignalType.REGULATORY,
                description=(
                    f"Tier-1 capital ratio of {tier1*100:.1f}% is below the "
                    f"{self._t.tier1_capital_ratio_min*100:.0f}% regulatory minimum threshold."
                ),
                severity=RiskLevel.HIGH,
                metric_name="tier1_capital_ratio",
                metric_value=tier1,
                triggered_by_rule=True,
                source_refs=[SourceRef(
                    document_type="10-Q",
                    document_date=metrics.get("data_as_of", ""),
                    section="Capital Ratios",
                )],
            ))

        # ── Derive minimum risk floor from signals ────────────────────────────
        if not signals:
            floor = RiskLevel.LOW
        else:
            severities = [s.severity for s in signals]
            if RiskLevel.HIGH in severities:
                floor = RiskLevel.HIGH
            elif RiskLevel.ELEVATED in severities:
                floor = RiskLevel.ELEVATED
            else:
                floor = RiskLevel.LOW

        logger.info(
            "Rule engine: %d signal(s) for %s — floor risk level: %s",
            len(signals),
            metrics.get("ticker", "?"),
            floor.value,
        )
        return signals, floor

    def _extract_htm_loss(self, market_docs: list[Document]) -> Optional[float]:
        """
        Attempt to extract HTM unrealized loss from market metric Documents.
        The value is stored in the JSON content of documents with
        source_type == "market_metrics".
        """
        for doc in market_docs:
            if doc.metadata.get("source_type") == "market_metrics":
                try:
                    data = json.loads(doc.page_content)
                    if "htm_unrealized_loss" in data:
                        return float(data["htm_unrealized_loss"])
                except (json.JSONDecodeError, ValueError):
                    pass
        return None

    def flag_names(self, signals: list[Signal]) -> list[str]:
        """Return a list of metric names that were flagged by the rule engine."""
        return [
            s.metric_name or s.signal_type.value
            for s in signals
            if s.triggered_by_rule
        ]
