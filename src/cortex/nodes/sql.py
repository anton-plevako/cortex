"""sql_agent_node + handle_sql_result_node — LLM tool-calling loop and result routing."""

import json

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from cortex.prompts import SQL_AGENT_SYSTEM
from cortex.state import AssetState

from ._shared import _sql_llm


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


def handle_sql_result_node(state: AssetState) -> dict:
    messages = state.get("messages") or []

    # Find the last ToolMessage
    tool_msg: ToolMessage | None = None
    for msg in reversed(messages):
        if isinstance(msg, ToolMessage):
            tool_msg = msg
            break

    if tool_msg is None:
        # No tool was called — sql_agent exhausted retries without generating valid SQL
        return {
            "next_action": "clarify",
            "error_bucket": "FALLBACK_EXEC_ERROR",
            "tool_result": {"status": "bad_sql", "error_message": "SQL could not be generated after retries."},
        }

    try:
        payload = json.loads(str(tool_msg.content))
    except (json.JSONDecodeError, TypeError):
        return {
            "next_action": "clarify",
            "error_bucket": "FALLBACK_EXEC_ERROR",
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
        bucket = (
            "CLARIFY_PROPERTY"
            if state.get("unresolved_entities")
            else "FALLBACK_NO_DATA"
        )
        return {
            "next_action": "clarify",
            "error_bucket": bucket,
            "tool_result": payload,
        }

    # exec_error or bad_sql reaching here means retries are exhausted
    return {
        "next_action": "clarify",
        "error_bucket": "FALLBACK_EXEC_ERROR",
        "tool_result": payload,
    }
