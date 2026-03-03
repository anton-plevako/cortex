"""classify_node — LLM intent classification."""

from typing import Literal

from pydantic import BaseModel, Field

from cortex.prompts import CLASSIFY_SYSTEM
from cortex.state import AssetState

from ._shared import _llm


class _Classification(BaseModel):
    request_type: Literal[
        "comparison", "pnl", "details", "general", "unclear", "off_topic"
    ] = Field(description="Type of asset-management request")


def classify_node(state: AssetState) -> dict:
    result: _Classification = _llm.with_structured_output(_Classification).invoke(  # type: ignore[assignment]
        [
            {"role": "system", "content": CLASSIFY_SYSTEM},
            {"role": "user", "content": state["user_query"]},
        ]
    )
    error_bucket = "FALLBACK_OFF_TOPIC" if result.request_type == "off_topic" else ""
    return {"request_type": result.request_type, "error_bucket": error_bucket}
