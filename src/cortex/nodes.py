from typing import Literal

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from cortex.config import MODEL_NAME, PROPERTY_NAMES
from cortex.prompts import (
    CLARIFY_SYSTEM,
    CLASSIFY_SYSTEM,
    EXTRACT_SYSTEM,
    RESPONSE_SYSTEM,
    SQL_PLAN_SYSTEM,
)
from cortex.state import AssetState, Timeframe
from cortex.tools import resolve_property

load_dotenv()

_llm = ChatOpenAI(model=MODEL_NAME, temperature=0)

# ---------------------------------------------------------------------------
# Pydantic schemas for structured LLM outputs
# ---------------------------------------------------------------------------


class _Classification(BaseModel):
    request_type: Literal["comparison", "pnl", "details", "general", "unknown"] = Field(
        description="Type of asset-management request"
    )


class _Extraction(BaseModel):
    properties: list[str] = Field(
        default_factory=list,
        description="Property names or identifiers mentioned in the query",
    )
    year: str | None = Field(None, description="4-digit year string, e.g. '2024'")
    quarter: str | None = Field(
        None, description="Quarter string in YYYY-QN format, e.g. '2024-Q3'"
    )
    month: str | None = Field(
        None, description="Month string in YYYY-MNN format, e.g. '2025-M01'"
    )


class _SqlPlan(BaseModel):
    sql: str = Field(description="A valid DuckDB SELECT statement")
    explanation: str = Field(description="One-sentence description of what the SQL does")


# ---------------------------------------------------------------------------
# Node: classify
# ---------------------------------------------------------------------------


def classify_node(state: AssetState) -> AssetState:
    result: _Classification = _llm.with_structured_output(_Classification).invoke(  # type: ignore[assignment]
        [
            {"role": "system", "content": CLASSIFY_SYSTEM},
            {"role": "user", "content": state["user_query"]},
        ]
    )
    return {"request_type": result.request_type}


# ---------------------------------------------------------------------------
# Node: extract
# ---------------------------------------------------------------------------

_MAX_EXTRACT_ATTEMPTS = 2


def extract_node(state: AssetState) -> AssetState:
    attempt = state.get("extract_attempts", 0) + 1
    previous_error = state.get("error", "") if attempt > 1 else ""

    user_content = f"Request type: {state['request_type']}\nQuery: {state['user_query']}"
    if previous_error:
        user_content += (
            f"\n\nPrevious attempt failed: {previous_error}"
            "\nPlease re-examine the query and try harder to map any identifiers"
            " to a known portfolio property or address."
        )

    result: _Extraction = _llm.with_structured_output(_Extraction).invoke(  # type: ignore[assignment]
        [
            {"role": "system", "content": EXTRACT_SYSTEM},
            {"role": "user", "content": user_content},
        ]
    )

    timeframe: Timeframe = {
        "year": result.year,
        "quarter": result.quarter,
        "month": result.month,
    }

    raw_properties = [p.strip() for p in result.properties if p.strip()]

    resolved: list[str] = []
    unresolved: list[str] = []
    for name in raw_properties:
        try:
            resolved.append(resolve_property(name))
        except ValueError:
            unresolved.append(name)

    if unresolved:
        return {
            "extract_attempts": attempt,
            "error": (
                f"Could not match '{', '.join(unresolved)}' to any portfolio property. "
                f"Available: {', '.join(PROPERTY_NAMES)}."
            ),
        }

    return {
        "properties": resolved,
        "timeframe": timeframe,
        "extract_attempts": attempt,
        "error": "",
    }


# ---------------------------------------------------------------------------
# Node: plan_sql  (NEW — Phase 2)
# ---------------------------------------------------------------------------

_MAX_SQL_ATTEMPTS = 3


def plan_sql_node(state: AssetState) -> AssetState:
    attempt = state.get("plan_attempts", 0) + 1
    previous_error = state.get("sql_error", "")

    tf = state.get("timeframe") or {}
    user_content = (
        f"User query: {state['user_query']}\n"
        f"Intent hint: {state.get('request_type', 'unknown')}\n"
        f"Properties hint: {state.get('properties', [])}\n"
        f"Timeframe hint: year={tf.get('year')}, "
        f"quarter={tf.get('quarter')}, month={tf.get('month')}"
    )
    if previous_error:
        user_content += (
            f"\n\nPrevious SQL failed with error:\n{previous_error}"
            "\nPlease fix the SQL and try again."
        )

    result: _SqlPlan = _llm.with_structured_output(_SqlPlan).invoke(  # type: ignore[assignment]
        [
            {"role": "system", "content": SQL_PLAN_SYSTEM},
            {"role": "user", "content": user_content},
        ]
    )

    return {
        "sql_query": result.sql,
        "plan_attempts": attempt,
        "sql_error": "",
    }


# ---------------------------------------------------------------------------
# Node: validate  (NEW — Phase 3)
# Deterministic safety gate — no LLM call.
# ---------------------------------------------------------------------------

# All column names present in the two DuckDB views.
_LEDGER_COLUMNS = {
    "entity_name", "property_name", "tenant_name", "ledger_type", "ledger_group",
    "ledger_category", "ledger_code", "ledger_description", "profit",
    "month", "quarter", "year",
}
_META_COLUMNS = {
    "property_name", "address", "current_value", "purchase_price", "last_appraisal_date",
}
_ALL_COLUMNS = _LEDGER_COLUMNS | _META_COLUMNS

_VALID_YEARS = {"2024", "2025"}
_VALID_QUARTERS = {f"{y}-Q{q}" for y in ("2024", "2025") for q in "1234"} | {"2025-Q1"}
_VALID_MONTHS = {f"2024-M{m:02d}" for m in range(1, 13)} | {f"2025-M{m:02d}" for m in range(1, 4)}
_FORBIDDEN = ("insert", "update", "delete", "drop", "create", "alter", "copy", "export")


def validate_node(state: AssetState) -> AssetState:
    sql = (state.get("sql_query") or "").strip()

    # 1. Must be a SELECT statement
    if not sql.upper().lstrip().startswith("SELECT"):
        return {"sql_error": "Only SELECT statements are allowed. Rewrite as a SELECT."}

    # 2. No forbidden keywords (case-insensitive word boundary check)
    sql_lower = sql.lower()
    for kw in _FORBIDDEN:
        if kw in sql_lower:
            return {"sql_error": f"Forbidden keyword '{kw}' found. Use SELECT only."}

    # 3. Add LIMIT if absent to guard against huge result sets
    if "limit" not in sql_lower:
        sql = sql.rstrip(";") + "\nLIMIT 200"

    return {"sql_query": sql, "sql_error": ""}


# ---------------------------------------------------------------------------
# Node: execute  (NEW — Phase 3)
# ---------------------------------------------------------------------------


def execute_node(state: AssetState) -> AssetState:
    import json as _json

    from cortex.db import get_connection

    sql = state.get("sql_query", "")
    try:
        conn = get_connection()
        df = conn.execute(sql).df()

        if df.empty:
            return {
                "sql_error": (
                    "The query returned no rows. The requested data may not exist "
                    "for the specified property or time period."
                )
            }

        # Serialise: round floats, convert to records
        df = df.round(2)
        records = df.to_dict(orient="records")
        sql_result = _json.dumps(records, indent=2, default=str)

        return {
            "sql_result": sql_result,
            "raw_data": {"rows": records, "columns": list(df.columns)},
            "sql_error": "",
        }

    except Exception as exc:
        return {"sql_error": f"DuckDB execution error: {exc}"}


# ---------------------------------------------------------------------------
# Node: respond
# ---------------------------------------------------------------------------


def response_node(state: AssetState) -> AssetState:
    tf = state.get("timeframe") or {}
    period_parts = [v for v in [tf.get("month"), tf.get("quarter"), tf.get("year")] if v]
    period = ", ".join(period_parts) if period_parts else "all available periods"

    # Prefer sql_result (new pipeline) over result (legacy)
    data_payload = state.get("sql_result") or state.get("result", "")

    user_message = (
        f"User query: {state['user_query']}\n"
        f"Time period covered: {period}\n"
        f"Query result:\n{data_payload}"
    )
    response = _llm.invoke(
        [
            {"role": "system", "content": RESPONSE_SYSTEM},
            {"role": "user", "content": user_message},
        ]
    )
    return {"result": response.content}


# ---------------------------------------------------------------------------
# Node: clarify  (genuine last resort)
# ---------------------------------------------------------------------------


def clarify_node(state: AssetState) -> AssetState:
    error = state.get("error", "") or state.get("sql_error", "")
    req = state.get("request_type", "unknown")

    user_message = (
        f"Original query: {state.get('user_query', '')}\n"
        f"Classified as: {req}\n"
        f"Failure reason: {error if error else 'could not generate a valid answer'}"
    )
    response = _llm.invoke(
        [
            {"role": "system", "content": CLARIFY_SYSTEM},
            {"role": "user", "content": user_message},
        ]
    )
    return {"result": response.content}
