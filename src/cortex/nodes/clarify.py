"""clarify_agent_node + human_interrupt_node — clarification hub and human-in-the-loop."""

import re
from typing import Literal

from langchain_core.messages import RemoveMessage
from langgraph.types import interrupt
from pydantic import BaseModel

from cortex.config import PROPERTY_NAMES
from cortex.prompts import (
    CLARIFY_AGENT_SYSTEM,
    CLARIFY_FALLBACK_SYSTEM,
    CLARIFY_QUESTION_SYSTEM,
)
from cortex.state import AssetState
from cortex.tools import resolve_property

from ._shared import MAX_CLARIFY_ATTEMPTS, _llm


class ClarifyDecision(BaseModel):
    action: Literal["ask_human", "fallback", "done"]
    message: str  # question text (ask_human) / explanation (fallback) / empty string (done)


def _resolve_from_answer(answer: str, expected_count: int) -> list[str]:
    """Resolve canonical property names from a free-form clarification answer.

    For a single expected property, tries the full answer string.
    For multiple, splits on common separators (', ', ' and ', ' & ') and
    resolves each part independently.  Raises ValueError if the right number
    of canonical names cannot be extracted.
    """
    if expected_count <= 1:
        return [resolve_property(answer)]  # let ValueError propagate

    parts = [p.strip() for p in re.split(r"\s*(?:and|&|,)\s*", answer, flags=re.IGNORECASE) if p.strip()]
    resolved: list[str] = []
    for part in parts:
        try:
            resolved.append(resolve_property(part))
        except ValueError:
            pass

    if len(resolved) == expected_count:
        return resolved
    raise ValueError(f"Could not resolve {expected_count} properties from '{answer}'")


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
    unresolved = state.get("unresolved_entities") or []

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
            "clarify_attempts": 0,  # reset: prevents poisoned attempt count on next query
            "raw_properties": [],   # clear: prevents stale properties leaking into new query
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
        bucket = state.get("error_bucket", "")

        if bucket == "CLARIFY_PROPERTY":
            # Deterministic bypass: resolve the user's answer directly, skip classify.
            # This prevents the bad property name(s) from re-entering classify and
            # triggering another CLARIFY_PROPERTY cycle.
            answer = state.get("last_clarify_answer") or ""
            unresolved = state.get("unresolved_entities") or []
            try:
                canonicals = _resolve_from_answer(answer, len(unresolved))
                # Patch bad property name(s) in user_query; use negative lookahead so
                # "Building 12" does not corrupt "Building 120" (prefix collision).
                original = state.get("user_query", "")
                new_query = original
                for bad_name, canonical in zip(unresolved, canonicals):
                    new_query = re.sub(re.escape(bad_name) + r'(?!\d)', canonical, new_query)
                return {
                    "raw_properties": canonicals,
                    "user_query": new_query,          # corrected for SQL agent framing
                    "user_query_original": new_query, # corrected so answer_node narrates right names
                    "next_action": "validate",        # route_clarify_agent → validate_and_resolve
                    "pending_question": None,         # clear replay-safety guard
                    "error_bucket": "",
                    "unresolved_entities": [],
                    "tool_result": {},                # clear stale SQL errors
                    "last_clarify_question": None,    # clear stale prompt context
                    "last_clarify_answer": None,
                    "clarify_attempts": 0,
                    # request_type kept unchanged — classify is skipped
                }
            except ValueError:
                # Answer still unresolvable; ask once more with explicit property list
                count = len(unresolved)
                which = "ones" if count > 1 else "one"
                question = (
                    f"Sorry, I still couldn't match that to a propert{'y' if count == 1 else 'ies'} "
                    f"in our portfolio. We manage: {', '.join(PROPERTY_NAMES)}. "
                    f"Which {which} did you mean?"
                )
                return {
                    "pending_question": question,
                    "next_action": "ask_human",
                    "result": question,
                    "result_type": "clarify",
                }

        # CLARIFY_QUERY (and any other bucket): concatenate + re-classify
        # Safe because the original query has no invalid property reference.
        original = state.get("user_query", "")
        answer = state.get("last_clarify_answer", "")
        combined = f"{original} — {answer}" if answer else original
        return {
            "user_query": combined,
            "next_action": "done",
            "pending_question": None,
            "error_bucket": "",
            "unresolved_entities": [],
            "tool_result": {},
            "last_clarify_question": None,
            "last_clarify_answer": None,
            "clarify_attempts": 0,
        }

    # action == "fallback"
    return {
        "result": message,
        "result_type": "fallback",
        "next_action": "fallback",
        "pending_question": None,
        "last_clarify_question": None,
        "last_clarify_answer": None,
        "clarify_attempts": 0,  # reset: prevents poisoned attempt count on next query
        "raw_properties": [],   # clear: prevents stale properties leaking into new query
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
