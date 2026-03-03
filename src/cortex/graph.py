from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from cortex.graph_routes import (
    route_after_classify,
    route_clarify_apply,
    route_clarify_entry,
    route_clarify_policy,
    route_on_next_action,
    route_sql_agent,
)
from cortex.nodes import (
    answer_node,
    clarify_apply_node,
    clarify_entry_node,
    clarify_fallback_node,
    clarify_interrupt_node,
    clarify_policy_node,
    clarify_question_node,
    classify_node,
    handle_sql_result_node,
    sql_agent_node,
    validate_and_resolve_node,
)
from cortex.state import AssetState
from cortex.tools import execute_sql


def build_graph() -> StateGraph:
    graph = StateGraph(AssetState)

    # ── Core pipeline nodes ───────────────────────────────────────────────────
    graph.add_node("classify", classify_node)
    graph.add_node("validate_and_resolve", validate_and_resolve_node)
    graph.add_node("sql_agent", sql_agent_node)
    graph.add_node("sql_executor", ToolNode([execute_sql]))
    graph.add_node("handle_sql_result", handle_sql_result_node)
    graph.add_node("answer", answer_node)

    # ── Clarify hub nodes ─────────────────────────────────────────────────────
    graph.add_node("clarify_entry", clarify_entry_node)
    graph.add_node("clarify_question", clarify_question_node)
    graph.add_node("clarify_interrupt", clarify_interrupt_node)
    graph.add_node("clarify_policy", clarify_policy_node)
    graph.add_node("clarify_apply", clarify_apply_node)
    graph.add_node("clarify_fallback", clarify_fallback_node)

    # ── Core pipeline edges ───────────────────────────────────────────────────
    graph.add_edge(START, "classify")

    graph.add_conditional_edges(
        "classify",
        route_after_classify,
        {"validate_and_resolve": "validate_and_resolve", "clarify_entry": "clarify_entry"},
    )

    graph.add_conditional_edges(
        "validate_and_resolve",
        route_on_next_action,
        {"sql": "sql_agent", "clarify_entry": "clarify_entry"},
    )

    graph.add_conditional_edges(
        "sql_agent",
        route_sql_agent,
        {"sql_executor": "sql_executor", "handle_sql_result": "handle_sql_result"},
    )

    graph.add_edge("sql_executor", "sql_agent")

    graph.add_conditional_edges(
        "handle_sql_result",
        route_on_next_action,
        {"answer": "answer", "clarify_entry": "clarify_entry"},
    )

    graph.add_edge("answer", END)

    # ── Clarify hub edges ─────────────────────────────────────────────────────
    graph.add_conditional_edges(
        "clarify_entry",
        route_clarify_entry,
        {
            "clarify_question": "clarify_question",
            "clarify_policy": "clarify_policy",
            "clarify_interrupt": "clarify_interrupt",  # staged question — bypass regeneration
            "clarify_fallback": "clarify_fallback",
        },
    )

    # question → interrupt (always); interrupt → entry (always, loops back)
    graph.add_edge("clarify_question", "clarify_interrupt")
    graph.add_edge("clarify_interrupt", "clarify_entry")

    graph.add_conditional_edges(
        "clarify_policy",
        route_clarify_policy,
        {
            "clarify_interrupt": "clarify_interrupt",
            "clarify_apply": "clarify_apply",
            "clarify_fallback": "clarify_fallback",
        },
    )

    graph.add_conditional_edges(
        "clarify_apply",
        route_clarify_apply,
        {
            "validate_and_resolve": "validate_and_resolve",
            "classify": "classify",
            "clarify_interrupt": "clarify_interrupt",
            "clarify_fallback": "clarify_fallback",
        },
    )

    graph.add_edge("clarify_fallback", END)

    return graph


app_graph = build_graph().compile(checkpointer=MemorySaver())
