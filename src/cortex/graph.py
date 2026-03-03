from pathlib import Path

from langgraph.graph import END, START, StateGraph

from cortex.nodes import (
    _MAX_EXTRACT_ATTEMPTS,
    _MAX_SQL_ATTEMPTS,
    clarify_node,
    classify_node,
    execute_node,
    extract_node,
    plan_sql_node,
    response_node,
    validate_node,
)
from cortex.state import AssetState


# ---------------------------------------------------------------------------
# Routing functions
# ---------------------------------------------------------------------------


def _route_after_extract(state: AssetState) -> str:
    """After extract: retry on entity error, else go to SQL planner."""
    if state.get("error"):
        if state.get("extract_attempts", 0) < _MAX_EXTRACT_ATTEMPTS:
            return "extract"       # retry with error context
        return "clarify"           # genuinely unresolvable entity
    return "plan_sql"


def _route_after_validate(state: AssetState) -> str:
    """After validate: loop back to planner on bad SQL, else execute."""
    if state.get("sql_error"):
        if state.get("plan_attempts", 0) < _MAX_SQL_ATTEMPTS:
            return "plan_sql"      # retry with validation error context
        return "clarify"
    return "execute"


def _route_after_execute(state: AssetState) -> str:
    """After execute: loop back to planner on DB error, else respond."""
    if state.get("sql_error"):
        if state.get("plan_attempts", 0) < _MAX_SQL_ATTEMPTS:
            return "plan_sql"      # retry with execution error context
        return "clarify"
    return "respond"


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


def build_graph() -> StateGraph:
    graph = StateGraph(AssetState)

    graph.add_node("classify", classify_node)
    graph.add_node("extract", extract_node)
    graph.add_node("plan_sql", plan_sql_node)
    graph.add_node("validate", validate_node)
    graph.add_node("execute", execute_node)
    graph.add_node("respond", response_node)
    graph.add_node("clarify", clarify_node)

    # classify → extract (all types — classification is now a hint, not a gate)
    graph.add_edge(START, "classify")
    graph.add_edge("classify", "extract")

    # extract → [retry | clarify | plan_sql]
    graph.add_conditional_edges(
        "extract", _route_after_extract,
        {"extract": "extract", "clarify": "clarify", "plan_sql": "plan_sql"},
    )

    # plan_sql → validate (always)
    graph.add_edge("plan_sql", "validate")

    # validate → [retry plan_sql | clarify | execute]
    graph.add_conditional_edges(
        "validate", _route_after_validate,
        {"plan_sql": "plan_sql", "clarify": "clarify", "execute": "execute"},
    )

    # execute → [retry plan_sql | clarify | respond]
    graph.add_conditional_edges(
        "execute", _route_after_execute,
        {"plan_sql": "plan_sql", "clarify": "clarify", "respond": "respond"},
    )

    graph.add_edge("respond", END)
    graph.add_edge("clarify", END)

    return graph


app_graph = build_graph().compile()

_PNG_PATH = Path(__file__).parent.parent.parent / "cortex_flow.png"
app_graph.get_graph().draw_mermaid_png(output_file_path=str(_PNG_PATH))
