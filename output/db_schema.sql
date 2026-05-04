-- Audit log schema for the Early Warning Financial Intelligence System
-- SQLite, WAL mode, INSERT-only (immutable append-only pattern)
--
-- Enable WAL mode on connection:
--   PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS audit_log (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       TEXT    NOT NULL,               -- ISO-8601 UTC datetime
    ticker          TEXT    NOT NULL,
    company         TEXT    NOT NULL,
    industry        TEXT,
    sector          TEXT,
    market_cap_bucket TEXT,
    risk_level      TEXT    NOT NULL,               -- LOW | ELEVATED | HIGH
    confidence_score REAL   NOT NULL,
    rule_engine_flags TEXT  NOT NULL DEFAULT '[]',  -- JSON array of flag names
    signals         TEXT    NOT NULL DEFAULT '[]',  -- JSON array of Signal objects
    narrative_summary TEXT,                         -- plain-English briefing
    human_review_status TEXT NOT NULL DEFAULT 'PENDING',  -- PENDING | REVIEWED | ESCALATED
    raw_json        TEXT    NOT NULL                -- full RiskOutput serialized as JSON
);

CREATE INDEX IF NOT EXISTS idx_audit_ticker     ON audit_log (ticker);
CREATE INDEX IF NOT EXISTS idx_audit_timestamp  ON audit_log (timestamp);
CREATE INDEX IF NOT EXISTS idx_audit_risk_level ON audit_log (risk_level);
