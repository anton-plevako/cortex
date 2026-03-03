import sys
import warnings

import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

sys.path.insert(0, "src")

from cortex.config import PROPERTY_NAMES, VALID_QUARTERS, VALID_YEARS
from cortex.graph import app_graph

st.set_page_config(page_title="Cortex – Asset Management", page_icon="🏢", layout="wide")

st.title("🏢 Cortex – Real Estate Asset Management")
st.caption("Ask natural-language questions about your property portfolio.")

with st.sidebar:
    st.header("Portfolio")
    st.markdown("**Properties**")
    for p in PROPERTY_NAMES:
        st.markdown(f"- {p}")
    st.markdown("**Available periods**")
    st.markdown(f"Years: {', '.join(sorted(VALID_YEARS))}")
    st.markdown(f"Quarters: {', '.join(sorted(VALID_QUARTERS))}")
    st.divider()
    st.markdown("**Example queries**")
    examples = [
        "What is the total P&L for all properties in 2024?",
        "Which property made the most money in 2024?",
        "Compare Building 120 and Building 17 for 2025",
        "Show Building 120 profit trend across all quarters",
        "What are our mortgage costs for the portfolio?",
        "Compare Building 120 in Q1 2024 vs Q1 2025",
        "What is the current value of Building 180?",
        "Give me details for Building 160 in Q4 2024",
    ]
    for ex in examples:
        if st.button(ex, key=ex, use_container_width=True):
            st.session_state["query_input"] = ex

query = st.text_area(
    "Your question",
    value=st.session_state.get("query_input", ""),
    placeholder="Ask anything about your portfolio…",
    height=80,
    key="query_input",
)

submitted = st.button("Ask", type="primary", disabled=not query.strip())

if submitted and query.strip():
    with st.spinner("Running agents…"):
        final = app_graph.invoke({"user_query": query.strip()})

    st.markdown(final.get("result", "No result returned."))

    # Show tabular result when the SQL returned multiple rows with numeric columns
    raw_data = final.get("raw_data") or {}
    rows = raw_data.get("rows", [])
    columns = raw_data.get("columns", [])
    if rows and len(rows) > 1 and columns:
        try:
            df = pd.DataFrame(rows, columns=columns)
            numeric_cols = df.select_dtypes("number").columns.tolist()
            if numeric_cols:
                st.divider()
                st.subheader("Query result")
                st.dataframe(
                    df.style.format(
                        {c: "€{:,.0f}" for c in numeric_cols}, na_rep="—"
                    ),
                    use_container_width=True,
                )
        except Exception:
            pass  # table display is best-effort

    with st.expander("Debug – pipeline state"):
        st.markdown("**Generated SQL**")
        st.code(final.get("sql_query", ""), language="sql")
        st.markdown("**State**")
        st.json(
            {
                "request_type": final.get("request_type"),
                "properties": final.get("properties"),
                "timeframe": final.get("timeframe"),
                "plan_attempts": final.get("plan_attempts"),
                "error": final.get("error"),
                "sql_error": final.get("sql_error"),
            }
        )
