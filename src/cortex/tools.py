"""
Tools for the Cortex real estate assistant.

resolve_property — fuzzy-match user input to canonical property names (used by resolve_guard_node).
execute_sql      — validate + execute a DuckDB SELECT query; returns structured JSON string
                   (used by sql_agent via ToolNode).
"""

import difflib
import json
import re

from langchain_core.tools import tool

from cortex.config import PROPERTY_NAMES
from cortex.db import get_connection

# ---------------------------------------------------------------------------
# Property name resolution
# ---------------------------------------------------------------------------

_FORBIDDEN_SQL = ("insert", "update", "delete", "drop", "create", "alter", "copy", "export")

# Maps building number string → canonical name, derived from PROPERTY_NAMES.
# e.g. {"17": "Building 17", "120": "Building 120", ...}
_NUMBER_TO_PROPERTY: dict[str, str] = {
    m.group(): p
    for p in PROPERTY_NAMES
    if (m := re.search(r"\d+", p))
}

_AVAILABLE = ", ".join(PROPERTY_NAMES)


def resolve_property(name: str) -> str:
    """
    Return the canonical property name for `name`, or raise ValueError.

    Stage 1 (numeric inputs): extract all digit runs from `name`.
      - More than one number found → raise (ambiguous input, e.g. "120 and 17").
      - Exactly one number found → look up in _NUMBER_TO_PROPERTY.
          Match  → return canonical name.
          No match → raise immediately. No fuzzy fallback for numeric inputs,
                     so "Building 10" never silently maps to "Building 180".
    Stage 2 (non-numeric / descriptive inputs): difflib fuzzy matching, then
      substring containment check — same logic as before.
    """
    normalised = name.strip().lower()

    # Stage 1: numeric path
    nums = re.findall(r"\d+", normalised)
    if nums:
        if len(nums) != 1:
            raise ValueError(
                f"Ambiguous property reference '{name}' (multiple numbers found). "
                f"Available: {_AVAILABLE}"
            )
        canonical = _NUMBER_TO_PROPERTY.get(nums[0])
        if canonical:
            return canonical
        raise ValueError(
            f"Unknown property '{name}'. Available: {_AVAILABLE}"
        )

    # Stage 2: descriptive / non-numeric — fuzzy match
    candidates: dict[str, str] = {p.lower(): p for p in PROPERTY_NAMES}

    matches = difflib.get_close_matches(normalised, candidates.keys(), n=1, cutoff=0.4)
    if matches:
        return candidates[matches[0]]

    for key, canonical in candidates.items():
        if normalised in key or key in normalised:
            return canonical

    raise ValueError(
        f"Unknown property '{name}'. Available: {_AVAILABLE}"
    )


# ---------------------------------------------------------------------------
# SQL execution tool — used by sql_agent via ToolNode
# ---------------------------------------------------------------------------


@tool
def execute_sql(sql: str) -> str:
    """Validate and execute a DuckDB SELECT query against the real estate ledger.

    Always call this with a complete SELECT statement.
    Returns a JSON string with keys:
      status       : 'ok' | 'no_data' | 'exec_error' | 'bad_sql'
      error_message: description of the problem (empty string on success)
      rows         : list of row dicts (empty on error)
      row_count    : number of rows returned
      columns      : list of column names
      cleaned_sql  : the SQL actually executed (trailing semicolons stripped)

    If status is 'bad_sql', fix the SQL and call this tool again.
    Do not call other tools before this one succeeds with status='ok' or a non-bad_sql status.
    """
    stripped = sql.strip()

    # --- Validation (deterministic) -----------------------------------------

    if not stripped.upper().lstrip().startswith("SELECT"):
        return json.dumps({
            "status": "bad_sql",
            "error_message": "Only SELECT statements are allowed. Rewrite as a SELECT.",
            "rows": [], "row_count": 0, "columns": [], "cleaned_sql": sql,
        })

    sql_lower = stripped.lower()
    for kw in _FORBIDDEN_SQL:
        if kw in sql_lower:
            return json.dumps({
                "status": "bad_sql",
                "error_message": f"Forbidden keyword '{kw}' found. Use SELECT only.",
                "rows": [], "row_count": 0, "columns": [], "cleaned_sql": sql,
            })

    cleaned_sql = stripped.rstrip(";")

    # --- Execution -----------------------------------------------------------

    try:
        conn = get_connection()
        df = conn.execute(cleaned_sql).df()

        if df.empty:
            return json.dumps({
                "status": "no_data",
                "error_message": "Query returned no rows for the requested filters.",
                "rows": [], "row_count": 0,
                "columns": list(df.columns),
                "cleaned_sql": cleaned_sql,
            })

        df = df.round(2)
        records = df.to_dict(orient="records")
        return json.dumps({
            "status": "ok",
            "error_message": "",
            "rows": records,
            "row_count": len(records),
            "columns": list(df.columns),
            "cleaned_sql": cleaned_sql,
        }, default=str)

    except Exception as exc:
        return json.dumps({
            "status": "exec_error",
            "error_message": f"DuckDB execution error: {exc}",
            "rows": [], "row_count": 0, "columns": [], "cleaned_sql": cleaned_sql,
        })
