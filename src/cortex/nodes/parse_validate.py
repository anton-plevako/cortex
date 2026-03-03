"""parse_and_validate_node — LLM extraction + deterministic guard checks."""

from pydantic import BaseModel, Field

from cortex.config import VALID_MONTHS, VALID_QUARTERS, VALID_YEARS
from cortex.prompts import EXTRACT_SYSTEM
from cortex.state import AssetState, Timeframe
from cortex.tools import resolve_property

from ._shared import _llm


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


def parse_and_validate_node(state: AssetState) -> dict:
    # Step 1 — LLM extraction
    user_content = f"Request type: {state['request_type']}\nQuery: {state['user_query']}"
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

    resolved: list[str] = []
    unresolved: list[str] = []
    for name in [p.strip() for p in result.properties if p.strip()]:
        try:
            resolved.append(resolve_property(name))
        except ValueError:
            unresolved.append(name)

    # Step 2 — deterministic guard checks (order: unclear → invalid_time → unresolved)

    next_action = "sql"
    error_bucket = ""
    request_type: str = state.get("request_type") or ""  # type: ignore[assignment]

    if request_type == "unclear":
        next_action = "clarify"
        error_bucket = "CLARIFY_QUERY"
    elif (
        (result.year and result.year not in VALID_YEARS)
        or (result.quarter and result.quarter not in VALID_QUARTERS)
        or (result.month and result.month not in VALID_MONTHS)
    ):
        next_action = "clarify"   # clarify_agent decides this is terminal via FALLBACK_* prefix
        error_bucket = "FALLBACK_NO_DATA"
    elif unresolved:
        next_action = "clarify"
        error_bucket = "CLARIFY_PROPERTY"

    return {
        "properties": resolved,
        "timeframe": timeframe,
        "unresolved_entities": unresolved,
        "next_action": next_action,
        "error_bucket": error_bucket,
    }
