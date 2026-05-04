"""
Discriminatory pattern monitoring via disparity ratios.

Calculates disparity ratios across industries and sectors to detect
whether certain groups of companies are being flagged disproportionately.

Methodology: Four-Fifths Rule adapted for algorithmic flagging.
    disparity_ratio(group_A, group_B) =
        (flagged_HIGH_or_ELEVATED / total_assessed for group_A) /
        (flagged_HIGH_or_ELEVATED / total_assessed for group_B)

A ratio > DISPARITY_RATIO_THRESHOLD (1.25) triggers a logged warning.

Note: this monitor compares companies within the same system run.
Meaningful disparity analysis requires a sufficiently large sample of
companies assessed over time. With a small watchlist, ratios may be
statistically unreliable — this is flagged in the output.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass

from config import DISPARITY_RATIO_THRESHOLD
from output.audit_log import AuditLog

logger = logging.getLogger(__name__)

MIN_GROUP_SIZE = 3  # minimum assessments per group for a reliable ratio


@dataclass
class DisparityResult:
    group_a: str
    group_b: str
    dimension: str          # "industry", "sector", or "market_cap_bucket"
    flagged_rate_a: float
    flagged_rate_b: float
    disparity_ratio: float
    count_a: int
    count_b: int
    is_flagged: bool
    low_sample_warning: bool


def calculate_disparity_ratios(audit_log: AuditLog | None = None) -> list[DisparityResult]:
    """
    Calculate disparity ratios across all dimension groups in the audit log.

    Args:
        audit_log: AuditLog instance (creates a new one if not provided)

    Returns:
        List of DisparityResult objects, one per pair that was compared.
    """
    log = audit_log or AuditLog()
    records = log.get_all_records(limit=10_000)

    if not records:
        logger.info("No audit records found — disparity monitoring skipped.")
        return []

    results: list[DisparityResult] = []

    for dimension in ("industry", "sector", "market_cap_bucket"):
        results.extend(_ratios_for_dimension(records, dimension))

    flagged = [r for r in results if r.is_flagged and not r.low_sample_warning]
    if flagged:
        for r in flagged:
            logger.warning(
                "DISPARITY ALERT: %s '%s' flagged at %.1fx the rate of '%s' "
                "(%.1f%% vs %.1f%%, n=%d vs n=%d) [dimension: %s]",
                dimension,
                r.group_a,
                r.disparity_ratio,
                r.group_b,
                r.flagged_rate_a * 100,
                r.flagged_rate_b * 100,
                r.count_a,
                r.count_b,
                r.dimension,
            )
    else:
        logger.info(
            "Disparity monitoring: %d group pairs analysed — no alerts triggered.",
            len(results),
        )

    return results


def _flagging_rate(records: list[dict]) -> float:
    """Fraction of records with risk_level ELEVATED or HIGH."""
    if not records:
        return 0.0
    flagged = sum(
        1 for r in records if r.get("risk_level") in ("ELEVATED", "HIGH")
    )
    return flagged / len(records)


def _ratios_for_dimension(
    records: list[dict],
    dimension: str,
) -> list[DisparityResult]:
    """Calculate disparity ratios for all group pairs within a dimension."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        val = record.get(dimension) or "Unknown"
        groups[val].append(record)

    group_names = sorted(groups.keys())
    if len(group_names) < 2:
        return []

    results: list[DisparityResult] = []

    for i, name_a in enumerate(group_names):
        for name_b in group_names[i + 1:]:
            recs_a = groups[name_a]
            recs_b = groups[name_b]

            rate_a = _flagging_rate(recs_a)
            rate_b = _flagging_rate(recs_b)

            # Skip pairs where both rates are zero (no signal to compare)
            if rate_a == 0 and rate_b == 0:
                continue

            # Denominator protection
            if rate_b == 0:
                ratio = float("inf") if rate_a > 0 else 1.0
            else:
                ratio = rate_a / rate_b

            low_sample = len(recs_a) < MIN_GROUP_SIZE or len(recs_b) < MIN_GROUP_SIZE

            results.append(
                DisparityResult(
                    group_a=name_a,
                    group_b=name_b,
                    dimension=dimension,
                    flagged_rate_a=rate_a,
                    flagged_rate_b=rate_b,
                    disparity_ratio=ratio,
                    count_a=len(recs_a),
                    count_b=len(recs_b),
                    is_flagged=ratio > DISPARITY_RATIO_THRESHOLD,
                    low_sample_warning=low_sample,
                )
            )

    return results


def disparity_summary_table(results: list[DisparityResult]) -> list[dict]:
    """
    Convert disparity results to a list of dicts suitable for a Streamlit DataFrame.
    """
    return [
        {
            "Dimension": r.dimension,
            "Group A": r.group_a,
            "Group B": r.group_b,
            "Rate A (%)": round(r.flagged_rate_a * 100, 1),
            "Rate B (%)": round(r.flagged_rate_b * 100, 1),
            "Disparity Ratio": round(r.disparity_ratio, 2),
            "n (A)": r.count_a,
            "n (B)": r.count_b,
            "Alert": "⚠️ FLAGGED" if r.is_flagged and not r.low_sample_warning else (
                "⚠️ Low sample" if r.low_sample_warning else "OK"
            ),
        }
        for r in results
    ]
