"""classify_node — LLM intent classification + slot extraction (single call)."""

from typing import Literal

from pydantic import BaseModel, Field

from cortex.prompts import UNDERSTAND_SYSTEM
from cortex.state import AssetState

from ._shared import _llm, invoke_with_retry, sanitize_error


class _Classification(BaseModel):
    request_type: Literal[
        "comparison", "pnl", "details", "general", "unclear", "off_topic"
    ] = Field(description="Type of asset-management request")
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


def classify_node(state: AssetState) -> dict:
    # Use the mutable working copy if available (set after clarify loops); fall back to raw input
    query_text = state.get("user_query_working") or state["user_query"]
    messages = [
        {"role": "system", "content": UNDERSTAND_SYSTEM},
        {"role": "user", "content": query_text},
    ]
    try:
        result: _Classification = invoke_with_retry(  # type: ignore[assignment]
            lambda: _llm.with_structured_output(_Classification).invoke(messages)
        )
    except Exception as e:
        return {
            "request_type": state.get("request_type") or "fallback",  # preserve last-known; never "off_topic"
            "error_bucket": "FALLBACK_EXEC_ERROR",
            "error_source": "classify_node",
            "error_detail": sanitize_error(e),
            "raw_properties": [],
            "timeframe": {},
        }
    error_bucket = "FALLBACK_OFF_TOPIC" if result.request_type == "off_topic" else ""
    out: dict = {
        "request_type": result.request_type,
        "error_bucket": error_bucket,
        "error_source": "",
        "error_detail": "",
        "raw_properties": result.properties,
        "timeframe": {
            "year": result.year,
            "quarter": result.quarter,
            "month": result.month,
        },
    }
    # On first call: pin original query and seed working copy — never overwritten after that
    if not state.get("user_query_original"):
        out["user_query_original"] = state["user_query"]
        out["user_query_working"] = state["user_query"]
    return out
