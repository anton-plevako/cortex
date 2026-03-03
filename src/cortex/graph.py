from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from cortex.nodes import (
    answer_node,
    clarify_or_fallback_node,
    classify_node,
    parse_and_validate_node,
    sql_agent_node,
    handle_sql_result_node,
)
from cortex.state import AssetState
from cortex.tools import execute_sql


# ---------------------------------------------------------------------------
# Routing functions
# ---------------------------------------------------------------------------


def _route_after_classify(state: AssetState) -> str:
    """off_topic goes straight to fallback; everything else attempts extract+SQL."""
    if state.get("request_type") == "off_topic":
        return "clarify_or_fallback"
    return "parse_and_validate"


def _route_on_next_action(state: AssetState) -> str:
    """Route based on next_action set by parse_and_validate or handle_sql_result."""
    na = state.get("next_action", "fallback")
    if na in ("clarify", "fallback"):
        return "clarify_or_fallback"
    return na  # "sql" or "answer"


def _route_sql_agent(state: AssetState) -> str:
    """If last message has tool calls, execute SQL; otherwise read result."""
    messages = state.get("messages") or []
    if messages:
        last = messages[-1]
        if getattr(last, "tool_calls", None):
            return "sql_executor"
    return "handle_sql_result"


def _route_after_clarify_or_fallback(state: AssetState) -> str:
    """Fallback is terminal; clarify loops back to parse_and_validate with fresh input."""
    if state.get("result_type") == "fallback":
        return END
    return "parse_and_validate"


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


def build_graph() -> StateGraph:
    graph = StateGraph(AssetState)

    graph.add_node("classify", classify_node)
    graph.add_node("parse_and_validate", parse_and_validate_node)
    graph.add_node("sql_agent", sql_agent_node)
    graph.add_node("sql_executor", ToolNode([execute_sql]))
    graph.add_node("handle_sql_result", handle_sql_result_node)
    graph.add_node("answer", answer_node)
    graph.add_node("clarify_or_fallback", clarify_or_fallback_node)

    graph.add_edge(START, "classify")

    graph.add_conditional_edges(
        "classify",
        _route_after_classify,
        {"parse_and_validate": "parse_and_validate", "clarify_or_fallback": "clarify_or_fallback"},
    )

    graph.add_conditional_edges(
        "parse_and_validate",
        _route_on_next_action,
        {"sql": "sql_agent", "clarify_or_fallback": "clarify_or_fallback"},
    )

    graph.add_conditional_edges(
        "sql_agent",
        _route_sql_agent,
        {"sql_executor": "sql_executor", "handle_sql_result": "handle_sql_result"},
    )

    graph.add_edge("sql_executor", "sql_agent")

    graph.add_conditional_edges(
        "handle_sql_result",
        _route_on_next_action,
        {"answer": "answer", "clarify_or_fallback": "clarify_or_fallback"},
    )

    graph.add_edge("answer", END)

    graph.add_conditional_edges(
        "clarify_or_fallback",
        _route_after_clarify_or_fallback,
        {"parse_and_validate": "parse_and_validate", END: END},
    )

    return graph


app_graph = build_graph().compile(checkpointer=MemorySaver())
