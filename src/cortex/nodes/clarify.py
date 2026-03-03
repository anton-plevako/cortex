"""clarify_agent_node + human_interrupt_node — clarification hub and human-in-the-loop."""

from typing import Literal

from langchain_core.messages import RemoveMessage
from langgraph.types import interrupt
from pydantic import BaseModel

from cortex.prompts import (
    CLARIFY_AGENT_SYSTEM,
    CLARIFY_FALLBACK_SYSTEM,
    CLARIFY_QUESTION_SYSTEM,
)
from cortex.state import AssetState

from ._shared import MAX_CLARIFY_ATTEMPTS, _llm


class ClarifyDecision(BaseModel):
    action: Literal["ask_human", "fallback", "done"]
    message: str  # question text (ask_human) / explanation (fallback) / empty string (done)


def _build_messages(state: AssetState, mode: str) -> list[dict]:
    """Build [system, user] message list for the given clarify mode."""
    bucket = state.get("error_bucket", "")
    query = state.get("user_query", "")
    tool_result = state.get("tool_result") or {}

    if mode == "fallback":
        system = CLARIFY_FALLBACK_SYSTEM
        lines = [
            f"Original query: {query}",
            f"Error type: {bucket}",
            f"SQL status: {tool_result.get('status', '')}",
        ]
        if tool_result.get("error_message"):
            lines.append(f"Technical detail: {tool_result['error_message']}")
        attempts = state.get("clarify_attempts", 0)
        if attempts > 0 and state.get("last_clarify_answer"):
            lines.append(f"Clarification question asked: {state.get('last_clarify_question', '')}")
            lines.append(f"User answered: {state['last_clarify_answer']}")
        user_msg = "\n".join(lines)

    elif mode == "clarify":
        system = CLARIFY_QUESTION_SYSTEM
        unresolved = state.get("unresolved_entities", [])
        lines = [
            f"Error type: {bucket}",
            f"Original query: {query}",
        ]
        if unresolved:
            lines.append(f"Unresolved property names mentioned: {', '.join(unresolved)}")
        user_msg = "\n".join(lines)

    else:  # mode == "agent"
        system = CLARIFY_AGENT_SYSTEM
        lines = [
            f"Original query: {query}",
            f"Error type: {bucket}",
            f"Question asked: {state.get('last_clarify_question', '')}",
            f"User answered: {state.get('last_clarify_answer', '')}",
            f"Attempts so far: {state.get('clarify_attempts', 0)}",
        ]
        user_msg = "\n".join(lines)

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_msg},
    ]


def clarify_agent_node(state: AssetState) -> dict:
    bucket = state.get("error_bucket", "FALLBACK_NO_DATA")
    attempts = state.get("clarify_attempts", 0)

    # Mode 1: replay safety — pending_question already set means the graph resumed
    # after interrupt; skip LLM and re-signal ask_human so route_clarify_agent
    # sends us back to human_interrupt.
    if state.get("pending_question"):
        return {"next_action": "ask_human"}

    # Mode 2a: terminal — FALLBACK_* bucket or attempts exhausted
    if bucket.startswith("FALLBACK") or attempts >= MAX_CLARIFY_ATTEMPTS:
        message = _llm.invoke(_build_messages(state, mode="fallback")).content
        return {
            "result": message,
            "result_type": "fallback",
            "next_action": "fallback",
            "pending_question": None,
            "last_clarify_question": None,
            "last_clarify_answer": None,
        }

    # Mode 2b: first clarify attempt — generate question, action forced by code
    if attempts == 0:
        question = _llm.invoke(_build_messages(state, mode="clarify")).content
        return {
            "pending_question": question,
            "next_action": "ask_human",
            "result": question,
            "result_type": "clarify",
        }

    # Mode 3: re-entry after human answered — LLM owns the decision
    response: ClarifyDecision = _llm.with_structured_output(ClarifyDecision).invoke(  # type: ignore[assignment]
        _build_messages(state, mode="agent")
    )
    action, message = response.action, response.message

    if action == "ask_human":
        return {
            "pending_question": message,
            "next_action": "ask_human",
            "result": message,
            "result_type": "clarify",
        }

    if action == "done":
        original = state.get("user_query", "")
        answer = state.get("last_clarify_answer", "")
        combined = f"{original} — {answer}" if answer else original
        return {
            "user_query": combined,
            "next_action": "done",
            "pending_question": None,
            # Clear ALL stale routing signals so re-entry to parse_and_validate is clean
            "error_bucket": "",
            "unresolved_entities": [],
            "tool_result": {},
            "last_clarify_question": None,
            "last_clarify_answer": None,
            # Reset so "unclear" doesn't immediately re-trigger CLARIFY_QUERY on re-parse
            "request_type": "general",
        }

    # action == "fallback"
    return {
        "result": message,
        "result_type": "fallback",
        "next_action": "fallback",
        "pending_question": None,
        "last_clarify_question": None,
        "last_clarify_answer": None,
    }


def human_interrupt_node(state: AssetState) -> dict:
    """Pause the graph and surface the clarification question to the user.
    Stores the Q/A pair in state; does NOT mutate user_query."""
    question = state.get("pending_question") or "Could you clarify your question?"
    user_answer = interrupt({"question": question})

    # Clear message history so sql_agent starts fresh on the next attempt
    existing_messages = state.get("messages") or []
    messages_to_remove = [
        RemoveMessage(id=m.id)
        for m in existing_messages
        if hasattr(m, "id") and m.id is not None
    ]

    return {
        "last_clarify_question": question,
        "last_clarify_answer": user_answer,
        "clarify_attempts": state.get("clarify_attempts", 0) + 1,
        "pending_question": None,
        "messages": messages_to_remove,
        # Preserve for hub context on re-entry
        "error_bucket": state.get("error_bucket", ""),
        "unresolved_entities": state.get("unresolved_entities", []),
    }
