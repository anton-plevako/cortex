"""answer_node — LLM narration of a successful SQL result."""

import json

from cortex.prompts import RESPONSE_SYSTEM
from cortex.state import AssetState

from ._shared import _llm


def answer_node(state: AssetState) -> dict:
    tool_result = state.get("tool_result") or {}
    tf = state.get("timeframe") or {}
    period_parts = [v for v in [tf.get("month"), tf.get("quarter"), tf.get("year")] if v]
    period = ", ".join(period_parts) if period_parts else "all available periods"

    user_message = (
        f"User query: {state['user_query']}\n"
        f"Time period covered: {period}\n"
        f"Query result (JSON):\n"
        f"{json.dumps(tool_result.get('rows', []), indent=2, default=str)}"
    )
    response = _llm.invoke(
        [
            {"role": "system", "content": RESPONSE_SYSTEM},
            {"role": "user", "content": user_message},
        ]
    )
    rows = tool_result.get("rows", [])
    columns = tool_result.get("columns", [])
    return {
        "result": response.content,
        "result_type": "answer",
        "raw_data": {"rows": rows, "columns": columns},
    }
