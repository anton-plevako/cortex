"""
DuckDB connection and schema setup for Cortex.

Exposes:
  get_connection()  — returns the shared in-memory DuckDB connection
  SCHEMA_SUMMARY    — human-readable schema string injected into LLM prompts
"""

import duckdb

from cortex.config import DATA_PATH

# ---------------------------------------------------------------------------
# Shared in-memory connection (created once at import time)
# ---------------------------------------------------------------------------

_conn: duckdb.DuckDBPyConnection | None = None


def get_connection() -> duckdb.DuckDBPyConnection:
    """Return the shared DuckDB connection, initialising it on first call."""
    global _conn
    if _conn is None:
        _conn = _build_connection()
    return _conn


def _build_connection() -> duckdb.DuckDBPyConnection:
    conn = duckdb.connect(database=":memory:")
    conn.execute(
        f"CREATE VIEW ledger AS SELECT * FROM read_parquet('{DATA_PATH.as_posix()}')"
    )
    return conn


# ---------------------------------------------------------------------------
# Schema summary — injected into plan_sql_node system prompt
# ---------------------------------------------------------------------------

SCHEMA_SUMMARY = """\
You have access to a DuckDB in-memory database with one table:

TABLE: ledger
  Columns:
    entity_name      VARCHAR  — always "PropCo"
    property_name    VARCHAR  — one of: Building 17, Building 120, Building 140,
                                Building 160, Building 180
                                NULL = entity-level costs (mortgage, management fees, etc.)
    tenant_name      VARCHAR
    ledger_type      VARCHAR  — "revenue" | "expenses"
    ledger_group     VARCHAR  — rental_income | sales_discounts | general_expenses |
                                management_fees | taxes_and_insurances
    ledger_category  VARCHAR  — e.g. revenue_rent_taxed, interest_mortgage,
                                proceeds_parking_taxed, insurance_in_general,
                                rent_discount_taxed, asset_management_fees
    ledger_code      VARCHAR
    ledger_description VARCHAR
    profit           DOUBLE   — single sign-encoded column:
                                positive = revenue inflow
                                negative = expense / cost
    month            VARCHAR  — format "2024-M01" through "2025-M03"
    quarter          VARCHAR  — format "2024-Q1" through "2025-Q1"
    year             VARCHAR  — "2024" | "2025"

  Important notes:
    - Rows with property_name IS NULL are entity-level costs for the whole portfolio
      (interest_mortgage, success_fees, asset_management_fees, etc.).
      Include these rows when computing true portfolio-wide totals.
    - Revenue rows may have negative profit values (e.g. rent_discount_taxed = discount given).
    - Available data: full year 2024, Q1 2025 only (months 2025-M01 to 2025-M03).

Time filter examples:
  year = '2024'         full year 2024
  quarter = '2025-Q1'   Q1 2025 (Jan–Mar)
  month = '2024-M10'    October 2024
  year = '2025'         same as quarter = '2025-Q1' (all available 2025 data)
"""
