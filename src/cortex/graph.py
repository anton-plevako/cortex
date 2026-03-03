from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from cortex.graph_routes import (
    route_after_classify,
    route_clarify_agent,
    route_on_next_action,
    route_sql_agent,
)
from cortex.nodes import (
    answer_node,
    clarify_agent_node,
    classify_node,
    handle_sql_result_node,
    human_interrupt_node,
    parse_and_validate_node,
    sql_agent_node,
)
from cortex.state import AssetState
from cortex.tools import execute_sql


def build_graph() -> StateGraph:
    graph = StateGraph(AssetState)

    graph.add_node("classify", classify_node)
    graph.add_node("parse_and_validate", parse_and_validate_node)
    graph.add_node("sql_agent", sql_agent_node)
    graph.add_node("sql_executor", ToolNode([execute_sql]))
    graph.add_node("handle_sql_result", handle_sql_result_node)
    graph.add_node("answer", answer_node)
    graph.add_node("clarify_agent", clarify_agent_node)
    graph.add_node("human_interrupt", human_interrupt_node)

    graph.add_edge(START, "classify")

    graph.add_conditional_edges(
        "classify",
        route_after_classify,
        {"parse_and_validate": "parse_and_validate", "clarify_agent": "clarify_agent"},
    )

    graph.add_conditional_edges(
        "parse_and_validate",
        route_on_next_action,
        {"sql": "sql_agent", "clarify_agent": "clarify_agent"},
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
        {"answer": "answer", "clarify_agent": "clarify_agent"},
    )

    graph.add_edge("answer", END)

    graph.add_conditional_edges(
        "clarify_agent",
        route_clarify_agent,
        {
            "human_interrupt": "human_interrupt",
            "parse_and_validate": "parse_and_validate",
            END: END,
        },
    )

    graph.add_edge("human_interrupt", "clarify_agent")

    return graph


app_graph = build_graph().compile(checkpointer=MemorySaver())
