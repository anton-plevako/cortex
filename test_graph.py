"""
35-question robustness test for the LangGraph NL→SQL pipeline.
Run with: uv run python test_graph.py

Questions are grouped by path and complexity:
  P  — P&L queries (portfolio-wide and per-property)
  D  — Detail / breakdown queries
  C  — Comparison queries
  A  — Asset value / address / metadata queries
  G  — General / novel queries (new — these failed with the old rigid-tool system)
  U  — Unknown / edge / adversarial inputs
"""
import sys

sys.path.insert(0, "src")

from cortex.graph import app_graph

QUERIES = [
    # ── P&L: portfolio-wide ──────────────────────────────────────────
    ("P1", "What is the total P&L for all properties in 2024?"),
    ("P2", "What is the total P&L for all my properties this year?"),
    ("P3", "Show me revenue and expenses for the whole portfolio in Q1 2025"),
    ("P4", "What are the mortgage costs for the portfolio?"),
    ("P5", "What were the asset management fees in 2024?"),

    # ── P&L: single property ─────────────────────────────────────────
    ("P6", "What did Building 140 earn in 2024?"),
    ("P7", "Show me the P&L for Building 17 in Q4 2024"),
    ("P8", "What is the net profit for the 120 building this year?"),
    ("P9", "How much revenue did Building 180 generate in Q1 2025?"),

    # ── Details ──────────────────────────────────────────────────────
    ("D1", "Give me details for Building 160 in Q4 2024"),
    ("D2", "Full financial breakdown for Building 17 in 2024"),
    ("D3", "Show me income sources and costs for Building 120 in Q1 2025"),
    ("D4", "What are the expense categories for Building 140?"),
    ("D5", "Give me details for all buildings in Q4 2024"),

    # ── Comparison ───────────────────────────────────────────────────
    ("C1", "Compare Building 120 and Building 17 for 2025"),
    ("C2", "Compare Building 140 and Building 160 in Q4 2024"),
    ("C3", "Which performed better in Q1 2025, Building 17 or Building 180?"),
    ("C4", "How does Building 180 compare to Building 140 for 2024?"),
    ("C5", "Compare the 120 to the 17 for last year"),

    # ── Asset value / address / metadata ─────────────────────────────
    ("A1", "What is the current value of Building 120?"),
    ("A2", "What is the purchase price of Building 180?"),
    ("A3", "Compare the harbor building with Building 17 this year"),
    ("A4", "What did the 17 building earn last year?"),

    # ── General / novel (failed with old rigid-tool system) ──────────
    ("G1", "Which property made the most money in 2024?"),          # superlative
    ("G2", "Rank all properties by net profit in 2024"),            # ranking
    ("G3", "Show me Building 120 profit trend across all quarters"),# time series
    ("G4", "How much parking revenue did we earn in 2024?"),        # category-specific
    ("G5", "How much rent discount did Building 180 give in 2024?"),# neg-revenue category
    ("G6", "Compare Building 120 in Q1 2024 vs Q1 2025"),          # cross-period
    ("G7", "Show me January 2025 P&L for the whole portfolio"),     # month-level
    ("G8", "What is our total equity across the portfolio?"),       # metadata math
    ("G9", "Show me last quarter performance"),                     # relative time

    # ── Unknown / edge / adversarial ─────────────────────────────────
    ("U1", "How is the weather today?"),                            # off-topic
    ("U2", "Compare Main Street with the other building"),          # unresolvable
    ("U3", "Show me Q3 2025 data"),                                 # out-of-range period
    ("U4", "bldg 120 q4 profit"),                                   # terse shorthand
    ("U5", "What will we earn next year?"),                         # future projection
]


def run() -> None:
    passed = 0
    failed = 0
    failures: list[str] = []

    for label, query in QUERIES:
        print(f"\n{'='*70}")
        print(f"[{label}] {query}")
        print("=" * 70)

        final = app_graph.invoke({"user_query": query})
        rt = final.get("request_type", "—")
        props = final.get("properties", [])
        tf = final.get("timeframe", {})
        sql = final.get("sql_query", "")
        err = final.get("error", "") or final.get("sql_error", "")
        result = final.get("result", "")

        print(f"  type      : {rt}")
        print(f"  properties: {props}")
        print(f"  timeframe : {tf}")
        if sql:
            print(f"  sql       :\n    " + sql.replace("\n", "\n    "))
        if err:
            print(f"  error     : {err}")
        print(f"\n  RESULT:\n  {result[:500].replace(chr(10), chr(10) + '  ')}")

        ok = bool(result and result.strip())
        status = "PASS" if ok else "FAIL"
        if ok:
            passed += 1
        else:
            failed += 1
            failures.append(f"[{label}] {query}")
        print(f"\n  [{status}]")

    print(f"\n{'='*70}")
    print(f"TOTAL: {passed} passed / {failed} failed out of {len(QUERIES)}")
    if failures:
        print("\nFailed:")
        for f in failures:
            print(f"  {f}")
    print("=" * 70)


if __name__ == "__main__":
    run()
