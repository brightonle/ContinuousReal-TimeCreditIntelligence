"""
Plain-English briefing generator.

Takes a structured RiskOutput JSON and calls Claude (Haiku — fast and cheap)
to convert it into a plain-English early warning briefing for the
credit risk committee.

Prompt caching is enabled on the static system prompt.
"""

from __future__ import annotations

import json
import logging

import anthropic

from config import ANTHROPIC_API_KEY, CLAUDE_BRIEFING_MODEL
from output.prompts import BRIEFING_SYSTEM_PROMPT, BRIEFING_USER_TEMPLATE
from risk.models import RiskOutput

logger = logging.getLogger(__name__)


def generate_briefing(risk_output: RiskOutput) -> str:
    """
    Generate a plain-English early warning briefing from a RiskOutput.

    Args:
        risk_output: the structured risk assessment

    Returns:
        Plain-English briefing string (under 300 words).
    """
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    # Serialize RiskOutput to a readable JSON (exclude verbose fields)
    output_dict = risk_output.model_dump(mode="json")
    # Remove the raw narrative_summary if already set (we're generating it)
    output_dict.pop("narrative_summary", None)
    risk_json = json.dumps(output_dict, indent=2)

    user_message = BRIEFING_USER_TEMPLATE.format(risk_output_json=risk_json)

    response = client.messages.create(
        model=CLAUDE_BRIEFING_MODEL,
        max_tokens=512,
        system=[
            {
                "type": "text",
                "text": BRIEFING_SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[{"role": "user", "content": user_message}],
    )

    logger.info(
        "Briefing generated for %s — input: %d tokens, output: %d tokens, "
        "cache_read: %d tokens",
        risk_output.ticker,
        response.usage.input_tokens,
        response.usage.output_tokens,
        getattr(response.usage, "cache_read_input_tokens", 0),
    )

    text_blocks = [b for b in response.content if b.type == "text"]
    if not text_blocks:
        logger.warning("No text in briefing response for %s", risk_output.ticker)
        return f"Risk assessment for {risk_output.company}: {risk_output.risk_level.value}"

    return text_blocks[0].text.strip()


def generate_and_attach_briefing(risk_output: RiskOutput) -> RiskOutput:
    """
    Generate a briefing and attach it to the RiskOutput as narrative_summary.

    Returns a new RiskOutput with narrative_summary populated.
    """
    briefing = generate_briefing(risk_output)
    return risk_output.model_copy(update={"narrative_summary": briefing})
