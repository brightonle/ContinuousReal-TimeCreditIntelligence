"""
LangGraph stateful agent for risk detection.

Loops over all companies in the watchlist. For each company:
  1. Fetch fresh market metrics
  2. Run the hard-threshold rule engine
  3. Query ChromaDB for relevant document chunks (RAG)
  4. Call Claude for structured risk assessment
  5. Write result to audit log

The LangGraph graph provides explicit state, resumability after failure,
and full visibility into mid-loop execution.

State flow:
  START → fetch_company → run_rules → rag_retrieval → claude_assess → write_audit
        ↑_____________________________________________|  (loop to next company)
                                                      ↓ (all companies done)
                                                      END
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, Optional

from langchain_core.documents import Document
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from config import DEFAULT_WATCHLIST, WatchlistEntry
from ingestion.pipeline import get_latest_market_docs
from risk.claude_risk import assess_company_risk
from risk.models import RiskLevel, RiskOutput, Signal
from risk.retriever import FinancialRiskRetriever
from risk.rule_engine import RuleEngine

logger = logging.getLogger(__name__)


# ── LangGraph State ────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    """Mutable state that flows through the risk detection graph."""
    watchlist: list[WatchlistEntry]
    current_index: int
    current_company: Optional[WatchlistEntry]
    market_docs: list[Document]
    market_metrics: dict
    rule_signals: list[Signal]
    rule_floor: RiskLevel
    rule_flags: list[str]
    dimension_docs: dict[str, list[Document]]
    risk_output: Optional[RiskOutput]
    completed_outputs: list[RiskOutput]
    errors: list[str]


# ── Graph Nodes ────────────────────────────────────────────────────────────────

def fetch_company(state: AgentState) -> AgentState:
    """Load the current company and fetch fresh market metrics."""
    idx = state["current_index"]
    watchlist = state["watchlist"]

    if idx >= len(watchlist):
        # Sentinel: all companies processed
        return {**state, "current_company": None}

    company = watchlist[idx]
    logger.info(
        "[%d/%d] Fetching market data for %s",
        idx + 1,
        len(watchlist),
        company.ticker,
    )

    try:
        market_docs = get_latest_market_docs(company)
    except Exception as exc:
        logger.error("Market fetch failed for %s: %s", company.ticker, exc)
        market_docs = []
        state["errors"].append(f"{company.ticker} market_fetch: {exc}")

    # Extract market metrics dict from the market_metrics Document
    market_metrics: dict = {}
    for doc in market_docs:
        if doc.metadata.get("source_type") == "market_metrics":
            try:
                raw = json.loads(doc.page_content)
                market_metrics = raw
                break
            except json.JSONDecodeError:
                pass

    return {
        **state,
        "current_company": company,
        "market_docs": market_docs,
        "market_metrics": market_metrics,
    }


def run_rules(state: AgentState) -> AgentState:
    """Run the hard-threshold rule engine against market metrics."""
    company = state["current_company"]
    if company is None:
        return state

    rule_engine = RuleEngine()
    try:
        signals, floor = rule_engine.check(
            metrics=dict(state["market_metrics"]),
            market_docs=state["market_docs"],
        )
        flags = rule_engine.flag_names(signals)
    except Exception as exc:
        logger.error("Rule engine failed for %s: %s", company.ticker, exc)
        signals, floor, flags = [], RiskLevel.LOW, []
        state["errors"].append(f"{company.ticker} rule_engine: {exc}")

    if signals:
        logger.info(
            "Rule engine flagged %d signal(s) for %s (floor: %s): %s",
            len(signals),
            company.ticker,
            floor.value,
            ", ".join(flags),
        )

    return {
        **state,
        "rule_signals": signals,
        "rule_floor": floor,
        "rule_flags": flags,
    }


def rag_retrieval(state: AgentState) -> AgentState:
    """Retrieve relevant document chunks from ChromaDB for all risk dimensions."""
    company = state["current_company"]
    if company is None:
        return state

    logger.info("Running RAG retrieval for %s", company.ticker)

    try:
        retriever = FinancialRiskRetriever(ticker=company.ticker, collection="both")
        dimension_docs = retriever.query_all_dimensions()
        total_chunks = sum(len(docs) for docs in dimension_docs.values())
        logger.info(
            "Retrieved %d total chunks across %d dimensions for %s",
            total_chunks,
            len(dimension_docs),
            company.ticker,
        )
    except Exception as exc:
        logger.error("RAG retrieval failed for %s: %s", company.ticker, exc)
        dimension_docs = {}
        state["errors"].append(f"{company.ticker} rag_retrieval: {exc}")

    return {**state, "dimension_docs": dimension_docs}


def claude_assess(state: AgentState) -> AgentState:
    """Call Claude for structured risk assessment."""
    company = state["current_company"]
    if company is None:
        return state

    logger.info("Calling Claude risk assessment for %s", company.ticker)

    try:
        risk_output = assess_company_risk(
            company=company,
            rule_signals=state["rule_signals"],
            rule_floor=state["rule_floor"],
            rule_flags=state["rule_flags"],
            dimension_docs=state["dimension_docs"],
            market_metrics=state["market_metrics"],
            assessment_date=datetime.utcnow().isoformat(),
        )
        logger.info(
            "Risk assessment complete for %s: %s (confidence: %.2f)",
            company.ticker,
            risk_output.risk_level.value,
            risk_output.confidence_score,
        )
    except Exception as exc:
        logger.error("Claude risk assessment failed for %s: %s", company.ticker, exc)
        risk_output = None
        state["errors"].append(f"{company.ticker} claude_assess: {exc}")

    return {**state, "risk_output": risk_output}


def write_audit(state: AgentState) -> AgentState:
    """Write the risk output to the audit log and advance to the next company."""
    risk_output = state["risk_output"]
    company = state["current_company"]

    completed = list(state["completed_outputs"])

    if risk_output is not None:
        try:
            # Import here to avoid circular deps — output module imports models
            from output.audit_log import AuditLog
            log = AuditLog()
            log.write(risk_output)
            logger.info(
                "Audit log entry written for %s (%s)",
                risk_output.ticker,
                risk_output.risk_level.value,
            )
        except Exception as exc:
            logger.error(
                "Failed to write audit log for %s: %s",
                company.ticker if company else "unknown",
                exc,
            )
            state["errors"].append(
                f"{company.ticker if company else '?'} audit_log: {exc}"
            )

        completed.append(risk_output)

    return {
        **state,
        "completed_outputs": completed,
        "current_index": state["current_index"] + 1,
        # Reset per-company state
        "current_company": None,
        "market_docs": [],
        "market_metrics": {},
        "rule_signals": [],
        "rule_floor": RiskLevel.LOW,
        "rule_flags": [],
        "dimension_docs": {},
        "risk_output": None,
    }


# ── Routing ────────────────────────────────────────────────────────────────────

def should_continue(state: AgentState) -> str:
    """Route back to fetch_company if more companies remain, else END."""
    idx = state["current_index"]
    if idx < len(state["watchlist"]):
        return "fetch_company"
    return END


# ── Graph Construction ─────────────────────────────────────────────────────────

def build_risk_graph() -> Any:
    """Build and compile the LangGraph risk detection graph."""
    graph = StateGraph(AgentState)

    graph.add_node("fetch_company", fetch_company)
    graph.add_node("run_rules", run_rules)
    graph.add_node("rag_retrieval", rag_retrieval)
    graph.add_node("claude_assess", claude_assess)
    graph.add_node("write_audit", write_audit)

    graph.add_edge(START, "fetch_company")
    graph.add_edge("fetch_company", "run_rules")
    graph.add_edge("run_rules", "rag_retrieval")
    graph.add_edge("rag_retrieval", "claude_assess")
    graph.add_edge("claude_assess", "write_audit")
    graph.add_conditional_edges("write_audit", should_continue, {
        "fetch_company": "fetch_company",
        END: END,
    })

    return graph.compile()


# Module-level compiled graph — build once, reuse
_compiled_graph = None


def get_risk_graph() -> Any:
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_risk_graph()
    return _compiled_graph


def run_risk_detection(
    watchlist: list[WatchlistEntry] | None = None,
) -> list[RiskOutput]:
    """
    Run the risk detection agent for all companies in the watchlist.

    Args:
        watchlist: companies to assess; defaults to DEFAULT_WATCHLIST

    Returns:
        List of RiskOutput objects (one per company that succeeded).
    """
    companies = watchlist or DEFAULT_WATCHLIST
    graph = get_risk_graph()

    initial_state: AgentState = {
        "watchlist": companies,
        "current_index": 0,
        "current_company": None,
        "market_docs": [],
        "market_metrics": {},
        "rule_signals": [],
        "rule_floor": RiskLevel.LOW,
        "rule_flags": [],
        "dimension_docs": {},
        "risk_output": None,
        "completed_outputs": [],
        "errors": [],
    }

    logger.info(
        "Starting risk detection for %d companies: %s",
        len(companies),
        [c.ticker for c in companies],
    )

    final_state = graph.invoke(initial_state)

    if final_state.get("errors"):
        for err in final_state["errors"]:
            logger.warning("Risk detection error: %s", err)

    outputs = final_state.get("completed_outputs", [])
    logger.info(
        "Risk detection complete: %d/%d companies assessed.",
        len(outputs),
        len(companies),
    )
    return outputs
