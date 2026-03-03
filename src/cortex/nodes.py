"""LangGraph node functions for the Cortex error-handling pipeline.

Nodes (in order):
  classify_node          — LLM: classify intent, detect off_topic early
  resolve_guard_node     — LLM extract + deterministic guard (4 checks)
  sql_agent_node         — LLM tool-calling loop (bound to execute_sql)
  post_router_node       — deterministic: read ToolMessage JSON, set next_action
  answer_node            — LLM: narrate successful SQL result
  clarify_or_fallback_node — interrupt (clarify) or terminal explanation (fallback)
"""

import json
from typing import Literal

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, RemoveMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.types import interrupt
from pydantic import BaseModel, Field

from cortex.config import MODEL_NAME, VALID_QUARTERS, VALID_YEARS
from cortex.prompts import (
    CLARIFY_SYSTEM,
    CLASSIFY_SYSTEM,
    EXTRACT_SYSTEM,
    FALLBACK_SYSTEM,
    RESPONSE_SYSTEM,
    SQL_AGENT_SYSTEM,
)
from cortex.state import AssetState, Timeframe
from cortex.tools import execute_sql, resolve_property

load_dotenv()

_llm = ChatOpenAI(model=MODEL_NAME, temperature=0)
_sql_llm = _llm.bind_tools([execute_sql])

MAX_CLARIFY_ATTEMPTS = 2


# ---------------------------------------------------------------------------
# Pydantic schemas for structured LLM outputs
# ---------------------------------------------------------------------------


class _Classification(BaseModel):
    request_type: Literal[
        "comparison", "pnl", "details", "general", "unclear", "off_topic"
    ] = Field(description="Type of asset-management request")


class _Extraction(BaseModel):
    properties: list[str] = Field(
        default_factory=list,
        description="Property names or identifiers mentioned in the query",
    )
    year: str | None = Field(None, description="4-digit year string, e.g. '2024'")
    quarter: str | None = Field(
        None, description="Quarter string in YYYY-QN format, e.g. '2024-Q3'"
    )
    month: str | None = Field(
        None, description="Month string in YYYY-MNN format, e.g. '2025-M01'"
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_sql_intent(state: AssetState) -> str:
    """Build the human message content for the sql_agent."""
    tf = state.get("timeframe") or {}
    lines = [
        f"User query: {state['user_query']}",
        f"Intent: {state.get('request_type', 'general')}",
        f"Resolved properties: {state.get('properties', [])}",
        f"Timeframe: year={tf.get('year')}, quarter={tf.get('quarter')}, month={tf.get('month')}",
    ]
    return "\n".join(lines)


def _build_clarify_message(state: AssetState) -> str:
    """Build the user context message for the clarify LLM call."""
    bucket = state.get("error_bucket", "CLARIFY_QUERY")
    unresolved = state.get("unresolved_entities", [])
    lines = [
        f"Error type: {bucket}",
        f"Original query: {state.get('user_query', '')}",
    ]
    if unresolved:
        lines.append(f"Unresolved property names mentioned: {', '.join(unresolved)}")
    return "\n".join(lines)


def _build_fallback_message(state: AssetState) -> str:
    """Build the user context message for the fallback LLM call."""
    bucket = state.get("error_bucket", "")
    lines = [
        f"Failure reason: {bucket}",
        f"Original query: {state.get('user_query', '')}",
        f"Classified as: {state.get('request_type', 'unknown')}",
        f"Clarification attempts made: {state.get('clarify_attempts', 0)}",
    ]
    tool_result = state.get("tool_result") or {}
    if tool_result.get("error_message"):
        lines.append(f"Technical detail: {tool_result['error_message']}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Node: classify_node
# ---------------------------------------------------------------------------


def classify_node(state: AssetState) -> dict:
    result: _Classification = _llm.with_structured_output(_Classification).invoke(  # type: ignore[assignment]
        [
            {"role": "system", "content": CLASSIFY_SYSTEM},
            {"role": "user", "content": state["user_query"]},
        ]
    )
    error_bucket = "FALLBACK_OFF_TOPIC" if result.request_type == "off_topic" else ""
    return {"request_type": result.request_type, "error_bucket": error_bucket}


# ---------------------------------------------------------------------------
# Node: resolve_guard_node
# LLM extract (Step 1) + deterministic guard checks (Step 2).
# ---------------------------------------------------------------------------


def resolve_guard_node(state: AssetState) -> dict:
    # Step 1 — LLM extraction
    user_content = f"Request type: {state['request_type']}\nQuery: {state['user_query']}"
    result: _Extraction = _llm.with_structured_output(_Extraction).invoke(  # type: ignore[assignment]
        [
            {"role": "system", "content": EXTRACT_SYSTEM},
            {"role": "user", "content": user_content},
        ]
    )

    timeframe: Timeframe = {
        "year": result.year,
        "quarter": result.quarter,
        "month": result.month,
    }

    resolved: list[str] = []
    unresolved: list[str] = []
    for name in [p.strip() for p in result.properties if p.strip()]:
        try:
            resolved.append(resolve_property(name))
        except ValueError:
            unresolved.append(name)

    # Step 2 — deterministic guard checks

    # Guard 1: trivially short / garbage input + unclear intent
    query_words = state["user_query"].strip().split()
    if len(query_words) < 3 and state.get("request_type") == "unclear":
        return {
            "properties": resolved,
            "timeframe": timeframe,
            "unresolved_entities": unresolved,
            "next_action": "clarify",
            "error_bucket": "CLARIFY_QUERY",
        }

    # Guard 2: requested time period outside available data
    if result.year and result.year not in VALID_YEARS:
        return {
            "properties": resolved,
            "timeframe": timeframe,
            "unresolved_entities": unresolved,
            "next_action": "fallback",
            "error_bucket": "FALLBACK_NO_DATA",
        }
    if result.quarter and result.quarter not in VALID_QUARTERS:
        return {
            "properties": resolved,
            "timeframe": timeframe,
            "unresolved_entities": unresolved,
            "next_action": "fallback",
            "error_bucket": "FALLBACK_NO_DATA",
        }

    # Guard 3: property mentioned but could not be resolved
    if unresolved:
        return {
            "properties": resolved,
            "timeframe": timeframe,
            "unresolved_entities": unresolved,
            "next_action": "clarify",
            "error_bucket": "CLARIFY_PROPERTY",
        }

    # Guard 4: intent still unclear (longer input — ask for specifics)
    if state.get("request_type") == "unclear":
        return {
            "properties": resolved,
            "timeframe": timeframe,
            "unresolved_entities": unresolved,
            "next_action": "clarify",
            "error_bucket": "CLARIFY_QUERY",
        }

    # All guards passed — proceed to SQL agent
    return {
        "properties": resolved,
        "timeframe": timeframe,
        "unresolved_entities": unresolved,
        "next_action": "sql",
        "error_bucket": "",
    }


# ---------------------------------------------------------------------------
# Node: sql_agent_node
# LLM tool-calling loop — bound to execute_sql.
# ---------------------------------------------------------------------------


def sql_agent_node(state: AssetState) -> dict:
    existing_messages = state.get("messages") or []
    # On first entry, inject the human intent message
    if not existing_messages:
        messages = [HumanMessage(content=_build_sql_intent(state))]
    else:
        messages = list(existing_messages)

    response = _sql_llm.invoke(
        [SystemMessage(content=SQL_AGENT_SYSTEM)] + messages
    )
    return {"messages": [response]}


# ---------------------------------------------------------------------------
# Node: post_router_node
# Deterministic — reads the last ToolMessage (execute_sql output).
# Sets next_action and tool_result in state.
# ---------------------------------------------------------------------------


def post_router_node(state: AssetState) -> dict:
    messages = state.get("messages") or []

    # Find the last ToolMessage
    tool_msg: ToolMessage | None = None
    for msg in reversed(messages):
        if isinstance(msg, ToolMessage):
            tool_msg = msg
            break

    if tool_msg is None:
        # No tool was called — sql_agent gave up; treat as fallback
        return {
            "next_action": "fallback",
            "error_bucket": "FALLBACK_NO_DATA",
            "tool_result": {},
        }

    try:
        payload = json.loads(str(tool_msg.content))
    except (json.JSONDecodeError, TypeError):
        return {
            "next_action": "fallback",
            "error_bucket": "FALLBACK_NO_DATA",
            "tool_result": {},
        }

    status = payload.get("status", "")

    if status == "ok":
        return {
            "next_action": "answer",
            "tool_result": payload,
            "error_bucket": "",
        }

    if status == "no_data":
        # Unresolved property names → ask for clarification
        bucket = (
            "CLARIFY_PROPERTY"
            if state.get("unresolved_entities")
            else "FALLBACK_NO_DATA"
        )
        return {
            "next_action": "fallback",
            "error_bucket": bucket,
            "tool_result": payload,
        }

    # exec_error or bad_sql
    return {
        "next_action": "fallback",
        "error_bucket": "FALLBACK_NO_DATA",
        "tool_result": payload,
    }


# ---------------------------------------------------------------------------
# Node: answer_node
# ---------------------------------------------------------------------------


def answer_node(state: AssetState) -> dict:
    tool_result = state.get("tool_result") or {}
    tf = state.get("timeframe") or {}
    period_parts = [v for v in [tf.get("month"), tf.get("quarter"), tf.get("year")] if v]
    period = ", ".join(period_parts) if period_parts else "all available periods"

    user_message = (
        f"User query: {state['user_query']}\n"
        f"Time period covered: {period}\n"
        f"Query result (JSON):\n"
        f"{json.dumps(tool_result.get('rows', []), indent=2, default=str)}"
    )
    response = _llm.invoke(
        [
            {"role": "system", "content": RESPONSE_SYSTEM},
            {"role": "user", "content": user_message},
        ]
    )
    rows = tool_result.get("rows", [])
    columns = tool_result.get("columns", [])
    return {
        "result": response.content,
        "result_type": "answer",
        "raw_data": {"rows": rows, "columns": columns},
    }


# ---------------------------------------------------------------------------
# Node: clarify_or_fallback_node
# Single interrupt location + single stop check (MAX_CLARIFY_ATTEMPTS).
# ---------------------------------------------------------------------------


def clarify_or_fallback_node(state: AssetState) -> dict:
    bucket = state.get("error_bucket", "FALLBACK_NO_DATA")
    attempts = state.get("clarify_attempts", 0)

    # Stop check: terminal buckets or too many attempts → fallback (no interrupt)
    go_fallback = bucket.startswith("FALLBACK") or attempts >= MAX_CLARIFY_ATTEMPTS

    if go_fallback:
        response = _llm.invoke(
            [
                {"role": "system", "content": FALLBACK_SYSTEM},
                {"role": "user", "content": _build_fallback_message(state)},
            ]
        )
        return {
            "result": response.content,
            "result_type": "fallback",
        }

    # Clarify path — generate question, interrupt, await user answer
    clarify_response = _llm.invoke(
        [
            {"role": "system", "content": CLARIFY_SYSTEM},
            {"role": "user", "content": _build_clarify_message(state)},
        ]
    )
    question = str(clarify_response.content)

    # Interrupt: pause graph and surface question to UI
    user_answer = interrupt({"question": question})

    # Clear message history so sql_agent starts fresh with new user input
    existing_messages = state.get("messages") or []
    messages_to_remove = [
        RemoveMessage(id=m.id)
        for m in existing_messages
        if hasattr(m, "id") and m.id is not None
    ]

    original_query = state.get("user_query", "")
    combined_query = f"{original_query} — {user_answer}" if original_query else user_answer

    return {
        "user_query": combined_query,
        "request_type": "general",  # prevent Guard 4 re-firing on the resumed query
        "messages": messages_to_remove,
        "clarify_attempts": attempts + 1,
        "result": question,
        "result_type": "clarify",
        # Reset routing signals
        "next_action": "",
        "error_bucket": "",
        "unresolved_entities": [],
        "tool_result": {},
    }
