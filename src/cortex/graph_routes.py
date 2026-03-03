"""
Routing functions for the Cortex LangGraph pipeline.
Each function reads state and returns the name of the next node.
"""

from langgraph.graph import END

from cortex.state import AssetState


def route_after_classify(state: AssetState) -> str:
    """off_topic goes to clarify_entry hub; everything else goes to validate_and_resolve."""
    if state.get("request_type") == "off_topic":
        return "clarify_entry"
    return "validate_and_resolve"


def route_on_next_action(state: AssetState) -> str:
    """Route based on next_action set by validate_and_resolve or handle_sql_result.
    Maps 'clarify' → clarify_entry. 'fallback' is never emitted here — only by clarify nodes.
    """
    na = state.get("next_action", "clarify")
    if na == "clarify":
        return "clarify_entry"
    return na  # "sql" or "answer"


def route_sql_agent(state: AssetState) -> str:
    """If last message has tool calls, execute SQL; otherwise read result."""
    messages = state.get("messages") or []
    if messages:
        last = messages[-1]
        if getattr(last, "tool_calls", None):
            return "sql_executor"
    return "handle_sql_result"


def route_clarify_entry(state: AssetState) -> str:
    """Dispatch from the clarify hub entry point."""
    na = state.get("next_action", "fallback")
    if na == "question":
        return "clarify_question"
    if na == "policy":
        return "clarify_policy"
    if na == "interrupt":
        return "clarify_interrupt"   # staged question exists — skip regeneration
    return "clarify_fallback"  # "fallback" or anything unexpected


def route_clarify_policy(state: AssetState) -> str:
    """Route after LLM policy decision."""
    na = state.get("next_action", "fallback")
    if na == "interrupt":
        return "clarify_interrupt"
    if na == "apply":
        return "clarify_apply"
    return "clarify_fallback"  # "fallback"


def route_clarify_apply(state: AssetState) -> str:
    """Route after applying a resolved clarification answer."""
    na = state.get("next_action", "fallback")
    if na == "validate":
        return "validate_and_resolve"
    if na == "classify":
        return "classify"
    if na == "interrupt":
        return "clarify_interrupt"
    return "clarify_fallback"  # anything unexpected
