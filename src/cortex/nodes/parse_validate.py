"""validate_and_resolve_node — deterministic property resolution and guard checks.
No LLM call. Reads raw_properties + timeframe written by classify_node."""

from cortex.config import VALID_MONTHS, VALID_QUARTERS, VALID_YEARS
from cortex.state import AssetState
from cortex.tools import resolve_property


def validate_and_resolve_node(state: AssetState) -> dict:
    tf = state.get("timeframe") or {}
    request_type = state.get("request_type") or ""

    # Resolve raw property names extracted by classify_node
    resolved: list[str] = []
    unresolved: list[str] = []
    for name in [p.strip() for p in (state.get("raw_properties") or []) if p.strip()]:
        try:
            resolved.append(resolve_property(name))
        except ValueError:
            unresolved.append(name)

    # Guard checks (order: unclear → invalid_time → unresolved)
    next_action = "sql"
    error_bucket = ""

    if request_type == "unclear":
        next_action = "clarify"
        error_bucket = "CLARIFY_QUERY"
    elif (
        (tf.get("year") and tf["year"] not in VALID_YEARS)
        or (tf.get("quarter") and tf["quarter"] not in VALID_QUARTERS)
        or (tf.get("month") and tf["month"] not in VALID_MONTHS)
    ):
        next_action = "clarify"   # clarify_agent decides this is terminal via FALLBACK_* prefix
        error_bucket = "FALLBACK_NO_DATA"
    elif unresolved:
        next_action = "clarify"
        error_bucket = "CLARIFY_PROPERTY"

    return {
        "properties": resolved,
        "unresolved_entities": unresolved,
        "timeframe": tf,            # pass-through: prevents stale None downstream
        "request_type": request_type,  # pass-through: same reason
        "next_action": next_action,
        "error_bucket": error_bucket,
    }
