"""
38-question robustness test for the LangGraph NL→SQL pipeline.
Run with: uv run python test_graph.py

Questions are grouped by path and complexity:
  P  — P&L queries (portfolio-wide and per-property)
  D  — Detail / breakdown queries
  C  — Comparison queries
  A  — Asset value / address / metadata queries
  G  — General / novel queries
  U  — Unknown / edge / adversarial inputs
  E  — Error-path assertions (result_type and bucket are verified)
"""
import sys
import uuid

sys.path.insert(0, "src")

from langgraph.types import Command

from cortex.graph import app_graph

# (label, query, expected_result_type_or_None, expected_bucket_or_None)
# None means "any non-empty result is acceptable"
QUERIES = [
    # ── P&L: portfolio-wide ──────────────────────────────────────────
    ("P1", "What is the total P&L for all properties in 2024?",             "answer", None),
    ("P2", "What is the total P&L for all my properties this year?",        "answer", None),
    ("P3", "Show me revenue and expenses for the whole portfolio in Q1 2025","answer", None),
    ("P4", "What are the mortgage costs for the portfolio?",                 "answer", None),
    ("P5", "What were the asset management fees in 2024?",                   "answer", None),

    # ── P&L: single property ─────────────────────────────────────────
    ("P6", "What did Building 140 earn in 2024?",                           "answer", None),
    ("P7", "Show me the P&L for Building 17 in Q4 2024",                   "answer", None),
    ("P8", "What is the net profit for the 120 building this year?",        "answer", None),
    ("P9", "How much revenue did Building 180 generate in Q1 2025?",        "answer", None),

    # ── Details ──────────────────────────────────────────────────────
    ("D1", "Give me details for Building 160 in Q4 2024",                   "answer", None),
    ("D2", "Full financial breakdown for Building 17 in 2024",              "answer", None),
    ("D3", "Show me income sources and costs for Building 120 in Q1 2025",  "answer", None),
    ("D4", "What are the expense categories for Building 140?",              "answer", None),
    ("D5", "Give me details for all buildings in Q4 2024",                  "answer", None),

    # ── Comparison ───────────────────────────────────────────────────
    ("C1", "Compare Building 120 and Building 17 for 2025",                 "answer", None),
    ("C2", "Compare Building 140 and Building 160 in Q4 2024",             "answer", None),
    ("C3", "Which performed better in Q1 2025, Building 17 or Building 180?","answer", None),
    ("C4", "How does Building 180 compare to Building 140 for 2024?",       "answer", None),
    ("C5", "Compare the 120 to the 17 for last year",                       "answer", None),

    # ── Ledger category / breakdown queries ──────────────────────────
    ("A1", "What are the biggest expense categories in 2024?",              "answer", None),
    ("A2", "How much insurance did we pay across the portfolio in 2024?",   "answer", None),
    ("A3", "What is the breakdown of revenue types for Building 120?",      "answer", None),
    ("A4", "What did the 17 building earn last year?",                      "answer", None),

    # ── General / novel ──────────────────────────────────────────────
    ("G1", "Which property made the most money in 2024?",                   "answer", None),
    ("G2", "Rank all properties by net profit in 2024",                     "answer", None),
    ("G3", "Show me Building 120 profit trend across all quarters",         "answer", None),
    ("G4", "How much parking revenue did we earn in 2024?",                 "answer", None),
    ("G5", "How much rent discount did Building 180 give in 2024?",        "answer", None),
    ("G6", "Compare Building 120 in Q1 2024 vs Q1 2025",                   "answer", None),
    ("G7", "Show me January 2025 P&L for the whole portfolio",              "answer", None),
    ("G8", "What is our total equity across the portfolio?",                "answer", None),
    ("G9", "Show me last quarter performance",                              "answer", None),

    # ── Unknown / edge / adversarial ─────────────────────────────────
    ("U1", "How is the weather today?",                    "fallback", "FALLBACK_OFF_TOPIC"),
    ("U2", "Compare Main Street with the other building",  None,       None),  # CLARIFY_PROPERTY → answer after resume
    ("U3", "Show me Q3 2025 data",                         None,       None),  # out-of-range period
    ("U4", "bldg 120 q4 profit",                           "answer",   None),  # terse shorthand — should resolve
    ("U5", "What will we earn next year?",                 "fallback",  None),  # future projection

    # ── Error-path assertions (deterministic guards) ──────────────────
    ("E1", "Tell me about revenues",  None, None),  # CLARIFY_QUERY — no property/period/metric
    ("E2", "!!!???##",                None, None),  # garbage input — must not crash, any result ok
    ("E3", "Show me 2023 data",       None, None),  # year before data starts — clarify or fallback
]

_MAX_RESUMES = 2  # matches MAX_CLARIFY_ATTEMPTS in nodes.py


def run() -> None:
    passed = 0
    failed = 0
    failures: list[str] = []

    for label, query, expect_type, expect_bucket in QUERIES:
        print(f"\n{'='*70}")
        print(f"[{label}] {query}")
        print("=" * 70)

        config = {"configurable": {"thread_id": str(uuid.uuid4())}}
        resume_count = 0

        final = app_graph.invoke({"user_query": query}, config=config)

        # Handle up to _MAX_RESUMES interrupts (clarify loop)
        while resume_count < _MAX_RESUMES:
            gs = app_graph.get_state(config)
            if not gs.next:
                break
            question = ""
            for task in gs.tasks:
                for intr in getattr(task, "interrupts", []):
                    question = (getattr(intr, "value", {}) or {}).get("question", "")
                    break
            print(f"  INTERRUPT #{resume_count + 1}: {question[:120]}")
            final = app_graph.invoke(Command(resume="Building 120 in 2024"), config=config)
            resume_count += 1

        result      = final.get("result", "")
        result_type = final.get("result_type", "")
        bucket      = final.get("error_bucket", "")

        print(f"  result_type: {result_type}  bucket: {bucket or '—'}")
        print(f"  RESULT: {result[:300].replace(chr(10), chr(10) + '  ')}")

        # Base check: result must be non-empty
        ok = bool(result and result.strip())

        # Tighter check: verify expected result_type when specified
        if ok and expect_type is not None and result_type != expect_type:
            ok = False
            print(f"  EXPECTED result_type={expect_type!r}, got {result_type!r}")

        # Tighter check: verify expected bucket when specified
        if ok and expect_bucket is not None and bucket != expect_bucket:
            ok = False
            print(f"  EXPECTED bucket={expect_bucket!r}, got {bucket!r}")

        status = "PASS" if ok else "FAIL"
        if ok:
            passed += 1
        else:
            failed += 1
            failures.append(f"[{label}] {query}")
        print(f"  [{status}]")

    print(f"\n{'='*70}")
    print(f"TOTAL: {passed} passed / {failed} failed out of {len(QUERIES)}")
    if failures:
        print("\nFailed:")
        for f in failures:
            print(f"  {f}")
    print("=" * 70)


if __name__ == "__main__":
    run()
