from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from cortex.nodes import (
    answer_node,
    clarify_or_fallback_node,
    classify_node,
    post_router_node,
    resolve_guard_node,
    sql_agent_node,
)
from cortex.state import AssetState
from cortex.tools import execute_sql


# ---------------------------------------------------------------------------
# Routing functions
# ---------------------------------------------------------------------------


def _route_after_classify(state: AssetState) -> str:
    """off_topic goes straight to fallback; everything else attempts resolve+SQL."""
    if state.get("request_type") == "off_topic":
        return "clarify_or_fallback"
    return "resolve_guard"


def _route_on_next_action(state: AssetState) -> str:
    """Route based on next_action set by resolve_guard or post_router."""
    na = state.get("next_action", "fallback")
    if na in ("clarify", "fallback"):
        return "clarify_or_fallback"
    return na  # "sql" or "answer"


def _route_sql_agent(state: AssetState) -> str:
    """If last message has tool calls, run ToolNode; otherwise read result."""
    messages = state.get("messages") or []
    if messages:
        last = messages[-1]
        if getattr(last, "tool_calls", None):
            return "tool_node"
    return "post_router"


def _route_after_clarify_or_fallback(state: AssetState) -> str:
    """Fallback is terminal; clarify loops back to resolve_guard with fresh input."""
    if state.get("result_type") == "fallback":
        return END
    return "resolve_guard"


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


def build_graph() -> StateGraph:
    graph = StateGraph(AssetState)

    tool_node = ToolNode([execute_sql])

    graph.add_node("classify", classify_node)
    graph.add_node("resolve_guard", resolve_guard_node)
    graph.add_node("sql_agent", sql_agent_node)
    graph.add_node("tool_node", tool_node)
    graph.add_node("post_router", post_router_node)
    graph.add_node("answer", answer_node)
    graph.add_node("clarify_or_fallback", clarify_or_fallback_node)

    graph.add_edge(START, "classify")

    graph.add_conditional_edges(
        "classify",
        _route_after_classify,
        {"resolve_guard": "resolve_guard", "clarify_or_fallback": "clarify_or_fallback"},
    )

    graph.add_conditional_edges(
        "resolve_guard",
        _route_on_next_action,
        {"sql": "sql_agent", "clarify_or_fallback": "clarify_or_fallback"},
    )

    graph.add_conditional_edges(
        "sql_agent",
        _route_sql_agent,
        {"tool_node": "tool_node", "post_router": "post_router"},
    )

    graph.add_edge("tool_node", "sql_agent")

    graph.add_conditional_edges(
        "post_router",
        _route_on_next_action,
        {"answer": "answer", "clarify_or_fallback": "clarify_or_fallback"},
    )

    graph.add_edge("answer", END)

    graph.add_conditional_edges(
        "clarify_or_fallback",
        _route_after_clarify_or_fallback,
        {"resolve_guard": "resolve_guard", END: END},
    )

    return graph


app_graph = build_graph().compile(checkpointer=MemorySaver())
