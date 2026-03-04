"""Clarify hub — 6 single-responsibility nodes.

Flow:
  clarify_entry  ──(question)──▶ clarify_question ──▶ clarify_interrupt ──▶ clarify_entry
  clarify_entry  ──(policy)───▶ clarify_policy
  clarify_policy ──(interrupt)▶ clarify_interrupt ──▶ clarify_entry
  clarify_policy ──(apply)────▶ clarify_apply
  clarify_apply  ──(validate)─▶ validate_and_resolve
  clarify_apply  ──(classify)─▶ classify
  clarify_apply  ──(interrupt)▶ clarify_interrupt ──▶ clarify_entry
  clarify_entry | clarify_policy | clarify_apply ──(fallback)──▶ clarify_fallback ──▶ END
"""

import re
from typing import Any, Literal

from langgraph.types import Overwrite, interrupt
from pydantic import BaseModel

from cortex.config import PROPERTY_NAMES
from cortex.prompts import (
    CLARIFY_AGENT_SYSTEM,
    CLARIFY_FALLBACK_SYSTEM,
    CLARIFY_QUESTION_SYSTEM,
)
from cortex.state import AssetState
from cortex.tools import resolve_property

from ._shared import MAX_CLARIFY_ATTEMPTS, _llm, invoke_with_retry, sanitize_error


class ClarifyDecision(BaseModel):
    action: Literal["ask_human", "fallback", "done"]
    message: str  # follow-up question (ask_human) | empty string (done) | explanation (fallback)


def _resolve_from_answer(answer: str, expected_count: int) -> list[str]:
    """Resolve canonical property names from a free-form clarification answer."""
    if expected_count <= 1:
        return [resolve_property(answer)]  # let ValueError propagate

    parts = [
        p.strip()
        for p in re.split(r"\s*(?:and|&|,)\s*", answer, flags=re.IGNORECASE)
        if p.strip()
    ]
    resolved: list[str] = []
    for part in parts:
        try:
            resolved.append(resolve_property(part))
        except ValueError:
            pass

    if len(resolved) == expected_count:
        return resolved
    raise ValueError(f"Could not resolve {expected_count} properties from '{answer}'")


# ── Node 1: Entry / dispatch ─────────────────────────────────────────────────

def clarify_entry_node(state: AssetState) -> dict[str, Any]:
    """Dispatch to the right clarify phase based on current state.

    Terminal conditions checked first, then staged-question check, then first-ask vs re-entry.
    """
    bucket = state.get("error_bucket", "FALLBACK_NO_DATA")
    attempts = state.get("clarify_attempts", 0)

    # Terminal: FALLBACK_* bucket or attempts exhausted
    if bucket.startswith("FALLBACK") or attempts >= MAX_CLARIFY_ATTEMPTS:
        return {"next_action": "fallback"}

    # Staged question already exists (set by policy/apply before a resume) — go straight to interrupt.
    # This makes entry idempotent on node re-runs: don't regenerate a question that was already staged.
    if state.get("pending_question"):
        return {"next_action": "interrupt"}

    # First ask: no answer yet.
    # Exact None check: empty string = "answered but useless" → policy
    if state.get("last_clarify_answer") is None:
        return {"next_action": "question"}

    # Re-entry: user has answered — let policy evaluate it
    return {"next_action": "policy"}


# ── Node 2: Generate clarification question ──────────────────────────────────

def clarify_question_node(state: AssetState) -> dict[str, Any]:
    """Generate a targeted clarification question (first ask only).

    Writes pending_question so clarify_interrupt_node knows what to surface.
    """
    bucket = state.get("error_bucket", "")
    query = state.get("user_query_working") or state.get("user_query", "")
    unresolved = state.get("unresolved_entities") or []

    lines = [
        f"Error type: {bucket}",
        f"Original query: {query}",
    ]
    if unresolved:
        lines.append(f"Unresolved property names mentioned: {', '.join(unresolved)}")

    messages = [
        {"role": "system", "content": CLARIFY_QUESTION_SYSTEM},
        {"role": "user", "content": "\n".join(lines)},
    ]
    try:
        question = str(invoke_with_retry(lambda: _llm.invoke(messages)).content)  # type: ignore[union-attr]
    except Exception as e:
        question = "Could you clarify your request?"
        return {"pending_question": question, "error_source": "clarify_question_node", "error_detail": sanitize_error(e)}
    return {"pending_question": question}


# ── Node 3: Human interrupt ──────────────────────────────────────────────────

def clarify_interrupt_node(state: AssetState) -> dict[str, Any]:
    """Pause the graph, surface pending_question, and store the user's answer.

    Uses Overwrite([]) to reset the messages channel (add_messages reducer)
    so sql_agent starts clean on the next attempt.
    """
    question = state.get("pending_question") or "Could you clarify your question?"
    user_answer = interrupt({"question": question})

    return {
        "last_clarify_question": question,
        "last_clarify_answer": user_answer,
        "clarify_attempts": state.get("clarify_attempts", 0) + 1,
        "pending_question": None,
        "messages": Overwrite([]),          # bypass add_messages reducer; reset channel
        # Preserve for hub context on re-entry
        "error_bucket": state.get("error_bucket", ""),
        "unresolved_entities": state.get("unresolved_entities", []),
    }


# ── Node 4: Policy — LLM evaluates the user's answer ────────────────────────

def clarify_policy_node(state: AssetState) -> dict[str, Any]:
    """LLM decides: ask again, apply the answer, or give up.

    On ask_human: writes pending_question for the next interrupt.
    On fallback: only signals next_action — clarify_fallback_node writes result/result_type.
    """
    bucket = state.get("error_bucket", "")
    query = state.get("user_query_working") or state.get("user_query", "")

    lines = [
        f"Original query: {query}",
        f"Error type: {bucket}",
        f"Question asked: {state.get('last_clarify_question', '')}",
        f"User answered: {state.get('last_clarify_answer', '')}",
        f"Attempts so far: {state.get('clarify_attempts', 0)}",
    ]
    messages = [
        {"role": "system", "content": CLARIFY_AGENT_SYSTEM},
        {"role": "user", "content": "\n".join(lines)},
    ]
    try:
        response: ClarifyDecision = invoke_with_retry(  # type: ignore[assignment]
            lambda: _llm.with_structured_output(ClarifyDecision).invoke(messages)
        )
    except Exception as e:
        return {"next_action": "fallback", "error_source": "clarify_policy_node", "error_detail": sanitize_error(e)}

    if response.action == "ask_human":
        return {
            "pending_question": response.message,
            "next_action": "interrupt",
        }

    if response.action == "done":
        return {"next_action": "apply"}

    # action == "fallback": only signal — clarify_fallback_node owns result/result_type
    return {"next_action": "fallback"}


# ── Node 5: Apply — patch state from the resolved answer ────────────────────

def clarify_apply_node(state: AssetState) -> dict[str, Any]:
    """Apply the user's answer: resolve properties or combine the query.

    Writes user_query_working (never mutates user_query).
    For unresolvable properties: writes pending_question and loops back via interrupt.
    """
    bucket = state.get("error_bucket", "")

    if bucket == "CLARIFY_PROPERTY":
        answer = state.get("last_clarify_answer") or ""
        unresolved = state.get("unresolved_entities") or []
        try:
            canonicals = _resolve_from_answer(answer, len(unresolved))
            # Patch bad property name(s) in working query; negative lookahead prevents
            # "Building 12" from corrupting "Building 120" (prefix collision guard).
            working = state.get("user_query_working") or state.get("user_query", "")
            new_working = working
            for bad_name, canonical in zip(unresolved, canonicals):
                new_working = re.sub(re.escape(bad_name) + r'(?!\d)', canonical, new_working)
            return {
                "raw_properties": canonicals,
                "user_query_working": new_working,
                "next_action": "validate",      # skip classify — straight to validate_and_resolve
                "error_bucket": "",
                "unresolved_entities": [],
                "tool_result": {},
                "last_clarify_question": None,
                "last_clarify_answer": None,
                "clarify_attempts": 0,
            }
        except ValueError:
            # Still unresolvable — ask once more with the explicit property list
            count = len(unresolved)
            question = (
                f"Sorry, I still couldn't match that to a propert"
                f"{'y' if count == 1 else 'ies'} in our portfolio. "
                f"We manage: {', '.join(PROPERTY_NAMES)}. "
                f"Which {'one' if count == 1 else 'ones'} did you mean?"
            )
            return {
                "pending_question": question,
                "next_action": "interrupt",
            }

    # CLARIFY_QUERY (and anything else): concatenate answer + re-classify
    working = state.get("user_query_working") or state.get("user_query", "")
    answer = state.get("last_clarify_answer", "")
    combined = f"{working} — {answer}" if answer else working
    return {
        "user_query_working": combined,
        "next_action": "classify",
        "error_bucket": "",
        "unresolved_entities": [],
        "tool_result": {},
        "last_clarify_question": None,
        "last_clarify_answer": None,
        "clarify_attempts": 0,
    }


# ── Node 6: Fallback — terminal failure message ──────────────────────────────

def clarify_fallback_node(state: AssetState) -> dict[str, Any]:
    """Generate a terminal failure explanation; always routes to END."""
    bucket = state.get("error_bucket", "")
    query = state.get("user_query_working") or state.get("user_query", "")
    tool_result = state.get("tool_result") or {}

    lines = [
        f"Original query: {query}",
        f"Error type: {bucket}",
        f"SQL status: {tool_result.get('status', '')}",
    ]
    if tool_result.get("error_message"):
        lines.append(f"Technical detail: {tool_result['error_message']}")
    if state.get("clarify_attempts", 0) > 0 and state.get("last_clarify_answer"):
        last_q: str = state.get("last_clarify_question") or ""  # type: ignore[assignment]
        last_a: str = state.get("last_clarify_answer") or ""    # type: ignore[assignment]
        lines.append(f"Clarification question asked: {last_q}")
        lines.append(f"User answered: {last_a}")

    messages = [
        {"role": "system", "content": CLARIFY_FALLBACK_SYSTEM},
        {"role": "user", "content": "\n".join(lines)},
    ]
    try:
        message = str(invoke_with_retry(lambda: _llm.invoke(messages)).content)  # type: ignore[union-attr]
    except Exception as e:
        prior_detail = state.get("error_detail") or ""
        detail = prior_detail or sanitize_error(e)
        message = f"I'm sorry, I wasn't able to process that request. ({detail})"
        return {
            "result": message,
            "result_type": "fallback",
            "error_source": "clarify_fallback_node",
            "error_detail": sanitize_error(e),
            "last_clarify_question": None,
            "last_clarify_answer": None,
            "pending_question": None,
            "clarify_attempts": 0,
            "raw_properties": [],
            "unresolved_entities": [],
            "tool_result": {},
            "messages": Overwrite([]),
        }
    return {
        "result": message,
        "result_type": "fallback",
        "last_clarify_question": None,
        "last_clarify_answer": None,
        "pending_question": None,
        "clarify_attempts": 0,
        "raw_properties": [],
        "unresolved_entities": [],
        "tool_result": {},
        "messages": Overwrite([]),
    }
