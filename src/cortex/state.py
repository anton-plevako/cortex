from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages


class Timeframe(TypedDict, total=False):
    year: str | None
    quarter: str | None
    month: str | None


class AssetState(TypedDict, total=False):
    # ── Input ─────────────────────────────────────────────────────────────────
    user_query: str

    # ── LLM tool-calling loop (sql_agent ↔ ToolNode) ──────────────────────────
    messages: Annotated[list, add_messages]

    # ── Extracted slots — preserved across clarify cycles ─────────────────────
    request_type: str             # from classify_node
    properties: list[str]         # resolved canonical property names
    timeframe: Timeframe          # year / quarter / month extracted from query
    unresolved_entities: list[str]  # property names mentioned but not matched

    # ── Routing signals — set by resolve_guard and post_router ────────────────
    next_action: str              # "sql" | "clarify" | "fallback" | "answer"
    error_bucket: str             # "CLARIFY_PROPERTY" | "CLARIFY_QUERY" |
                                  # "FALLBACK_OFF_TOPIC" | "FALLBACK_NO_DATA" |
                                  # "FALLBACK_EXEC_ERROR" | ""

    # ── SQL execution output — written by post_router after reading ToolMessage
    tool_result: dict             # {status, error_message, rows, row_count, columns, cleaned_sql}

    # ── Clarification loop control ─────────────────────────────────────────────
    clarify_attempts: int         # incremented on each interrupt
    pending_question: str | None  # set by clarify_agent before interrupt; replay-safety guard
    last_clarify_question: str | None  # question asked last turn
    last_clarify_answer: str | None    # user's answer last turn

    # ── Output ────────────────────────────────────────────────────────────────
    result: str                   # final response text shown to user
    result_type: str              # "answer" | "clarify" | "fallback" — Streamlit display logic
    raw_data: dict                # {rows, columns} for Streamlit table display
