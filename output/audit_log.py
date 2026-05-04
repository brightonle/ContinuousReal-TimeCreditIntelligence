"""
Immutable audit log backed by SQLite.

Design principles:
  - INSERT-only: no UPDATE or DELETE methods are exposed
  - WAL mode: allows concurrent reads from the dashboard while the agent writes
  - All risk assessments are recorded as a complete JSON snapshot, not derived fields
  - The audit log is the source of truth for the dashboard and disparity monitor

Usage:
    from output.audit_log import AuditLog
    log = AuditLog()
    log.write(risk_output)                         # append record
    latest = log.get_latest_per_company()          # most recent per ticker
    history = log.get_history(ticker="SIVB")       # all records for one company
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

from config import AUDIT_DB_PATH
from risk.models import RiskLevel, RiskOutput

logger = logging.getLogger(__name__)


class AuditLog:
    """
    Append-only audit log for all risk assessments.

    Thread-safe for single-process use (SQLite WAL mode).
    """

    def __init__(self, db_path: Path = AUDIT_DB_PATH) -> None:
        self._db_path = db_path
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _init_db(self) -> None:
        schema_path = Path(__file__).parent / "db_schema.sql"
        with self._connect() as conn:
            if schema_path.exists():
                conn.executescript(schema_path.read_text())
            else:
                # Inline fallback if schema file is missing
                conn.executescript("""
                    CREATE TABLE IF NOT EXISTS audit_log (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        ticker TEXT NOT NULL,
                        company TEXT NOT NULL,
                        industry TEXT,
                        sector TEXT,
                        market_cap_bucket TEXT,
                        risk_level TEXT NOT NULL,
                        confidence_score REAL NOT NULL,
                        rule_engine_flags TEXT NOT NULL DEFAULT '[]',
                        signals TEXT NOT NULL DEFAULT '[]',
                        narrative_summary TEXT,
                        human_review_status TEXT NOT NULL DEFAULT 'PENDING',
                        raw_json TEXT NOT NULL
                    );
                    CREATE INDEX IF NOT EXISTS idx_audit_ticker ON audit_log (ticker);
                    CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_log (timestamp);
                    CREATE INDEX IF NOT EXISTS idx_audit_risk_level ON audit_log (risk_level);
                """)

    def write(self, risk_output: RiskOutput) -> int:
        """
        Append a risk assessment record to the audit log.

        Args:
            risk_output: the structured risk assessment to record

        Returns:
            The rowid of the newly inserted record.
        """
        raw_json = json.dumps(risk_output.model_dump_audit())
        signals_json = json.dumps([
            s.model_dump(mode="json") for s in risk_output.signals
        ])
        flags_json = json.dumps(risk_output.rule_engine_flags)

        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO audit_log (
                    timestamp, ticker, company, industry, sector,
                    market_cap_bucket, risk_level, confidence_score,
                    rule_engine_flags, signals, narrative_summary,
                    human_review_status, raw_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    risk_output.assessment_date,
                    risk_output.ticker,
                    risk_output.company,
                    risk_output.industry,
                    risk_output.sector,
                    risk_output.market_cap_bucket,
                    risk_output.risk_level.value,
                    risk_output.confidence_score,
                    flags_json,
                    signals_json,
                    risk_output.narrative_summary,
                    risk_output.human_review_status,
                    raw_json,
                ),
            )
            rowid = cursor.lastrowid

        logger.debug(
            "Audit log row %d: %s %s %s",
            rowid,
            risk_output.ticker,
            risk_output.risk_level.value,
            risk_output.assessment_date[:10],
        )
        return rowid

    def get_latest_per_company(self) -> list[RiskOutput]:
        """
        Return the most recent audit record for each ticker.
        Used by the dashboard's watchlist view.
        """
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT raw_json FROM audit_log
                WHERE id IN (
                    SELECT MAX(id) FROM audit_log GROUP BY ticker
                )
                ORDER BY ticker
                """
            ).fetchall()

        return [RiskOutput.model_validate(json.loads(r["raw_json"])) for r in rows]

    def get_history(
        self,
        ticker: str,
        limit: int = 100,
    ) -> list[RiskOutput]:
        """Return all audit records for a specific company, most recent first."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT raw_json FROM audit_log
                WHERE ticker = ?
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (ticker, limit),
            ).fetchall()

        return [RiskOutput.model_validate(json.loads(r["raw_json"])) for r in rows]

    def get_all_records(self, limit: int = 1000) -> list[dict]:
        """
        Return raw audit records as dicts (for the disparity monitor and audit panel).
        """
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, timestamp, ticker, company, industry, sector,
                       market_cap_bucket, risk_level, confidence_score,
                       rule_engine_flags, narrative_summary, human_review_status
                FROM audit_log
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

        return [dict(r) for r in rows]

    def mark_reviewed(self, record_id: int, status: str = "REVIEWED") -> None:
        """
        Update the human review status for a record.

        NOTE: This is the ONLY mutation method on the audit log, and it only
        updates the review status column — it does not modify risk data.
        """
        valid_statuses = {"REVIEWED", "ESCALATED", "PENDING"}
        if status not in valid_statuses:
            raise ValueError(f"status must be one of {valid_statuses}")
        with self._connect() as conn:
            conn.execute(
                "UPDATE audit_log SET human_review_status = ? WHERE id = ?",
                (status, record_id),
            )

    def count(self) -> int:
        """Return the total number of audit records."""
        with self._connect() as conn:
            return conn.execute("SELECT COUNT(*) FROM audit_log").fetchone()[0]
