"""classify_node — LLM intent classification + slot extraction (single call)."""

from typing import Literal

from pydantic import BaseModel, Field

from cortex.prompts import UNDERSTAND_SYSTEM
from cortex.state import AssetState

from ._shared import _llm


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
    result: _Classification = _llm.with_structured_output(_Classification).invoke(  # type: ignore[assignment]
        [
            {"role": "system", "content": UNDERSTAND_SYSTEM},
            {"role": "user", "content": state["user_query"]},
        ]
    )
    error_bucket = "FALLBACK_OFF_TOPIC" if result.request_type == "off_topic" else ""
    out: dict = {
        "request_type": result.request_type,
        "error_bucket": error_bucket,
        "raw_properties": result.properties,
        "timeframe": {
            "year": result.year,
            "quarter": result.quarter,
            "month": result.month,
        },
    }
    # Preserve original query on first call only — never overwritten on re-classify
    if not state.get("user_query_original"):
        out["user_query_original"] = state["user_query"]
    return out
