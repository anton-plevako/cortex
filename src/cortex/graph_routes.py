"""
Routing functions for the Cortex LangGraph pipeline.
Each function reads state and returns the name of the next node.
"""

from langgraph.graph import END

from cortex.state import AssetState


def route_after_classify(state: AssetState) -> str:
    """off_topic goes to clarify_agent hub; everything else attempts extract+SQL."""
    if state.get("request_type") == "off_topic":
        return "clarify_agent"
    return "parse_and_validate"


def route_on_next_action(state: AssetState) -> str:
    """Route based on next_action set by parse_and_validate or handle_sql_result.
    Only maps 'clarify' → clarify_agent. 'fallback' is never emitted by upstream
    nodes — it is only emitted by clarify_agent and consumed by route_clarify_agent.
    """
    na = state.get("next_action", "clarify")
    if na == "clarify":
        return "clarify_agent"
    return na  # "sql" or "answer"


def route_sql_agent(state: AssetState) -> str:
    """If last message has tool calls, execute SQL; otherwise read result."""
    messages = state.get("messages") or []
    if messages:
        last = messages[-1]
        if getattr(last, "tool_calls", None):
            return "sql_executor"
    return "handle_sql_result"


def route_clarify_agent(state: AssetState) -> str:
    """Sole owner of fallback → END routing. Also routes ask_human and done."""
    na = state.get("next_action", "fallback")
    if na == "ask_human":
        return "human_interrupt"
    if na == "done":
        return "parse_and_validate"
    return END  # "fallback" or anything unexpected → terminal
