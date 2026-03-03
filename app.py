import sys
import uuid
import warnings

import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

sys.path.insert(0, "src")

from langgraph.types import Command

from cortex.config import PROPERTY_NAMES, VALID_QUARTERS, VALID_YEARS
from cortex.graph import app_graph

st.set_page_config(page_title="Cortex – Asset Management", page_icon="🏢", layout="wide")

st.title("🏢 Cortex – Real Estate Asset Management")
st.caption("Ask natural-language questions about your property portfolio.")

# ── Session state initialisation ────────────────────────────────────────────
st.session_state.setdefault("thread_id", str(uuid.uuid4()))
st.session_state.setdefault("awaiting_clarification", False)
st.session_state.setdefault("clarify_question", "")

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
        "What are the biggest expense categories in 2024?",
        "Give me details for Building 160 in Q4 2024",
    ]
    for ex in examples:
        if st.button(ex, key=ex, use_container_width=True):
            st.session_state["query_input"] = ex
            st.session_state["thread_id"] = str(uuid.uuid4())
            st.session_state["awaiting_clarification"] = False

    st.divider()
    if st.button("New conversation", use_container_width=True):
        st.session_state["thread_id"] = str(uuid.uuid4())
        st.session_state["awaiting_clarification"] = False
        st.session_state["clarify_question"] = ""

# ── Clarification prompt (shown when graph is waiting for more info) ─────────
clarify_placeholder = st.empty()
if st.session_state["awaiting_clarification"]:
    clarify_placeholder.info(st.session_state["clarify_question"])
    placeholder = "Your answer…"
    label = "Clarification"
else:
    placeholder = "Ask anything about your portfolio…"
    label = "Your question"

query = st.text_area(
    label,
    value=st.session_state.get("query_input", ""),
    placeholder=placeholder,
    height=80,
    key="query_input",
)

submitted = st.button("Ask", type="primary", disabled=not query.strip())

if submitted and query.strip():
    config = {"configurable": {"thread_id": st.session_state["thread_id"]}}

    with st.spinner("Running agents…"):
        if st.session_state["awaiting_clarification"]:
            # Resume the paused graph with the user's clarification answer
            final = app_graph.invoke(Command(resume=query.strip()), config=config)
        else:
            # Fresh question — start a new thread
            st.session_state["thread_id"] = str(uuid.uuid4())
            config = {"configurable": {"thread_id": st.session_state["thread_id"]}}
            final = app_graph.invoke({"user_query": query.strip()}, config=config)

        # Check if the graph is now paused at another interrupt
        graph_state = app_graph.get_state(config)
        if graph_state.next:
            # Extract the clarification question
            question = ""
            for task in graph_state.tasks:
                for intr in getattr(task, "interrupts", []):
                    val = getattr(intr, "value", {}) or {}
                    question = val.get("question", "")
                    break
            st.session_state["awaiting_clarification"] = True
            st.session_state["clarify_question"] = question or "Could you clarify your question?"
            st.rerun()
        else:
            st.session_state["awaiting_clarification"] = False
            st.session_state["clarify_question"] = ""
            clarify_placeholder.empty()  # remove the blue question box

    result_type = final.get("result_type", "answer")
    result_text = final.get("result", "No result returned.")

    if result_type == "fallback":
        st.warning(result_text)
    elif result_type == "clarify":
        st.info(result_text)
    else:
        st.markdown(result_text)

    # Tabular result for multi-row SQL answers
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
        tool_result = final.get("tool_result") or {}
        if tool_result.get("cleaned_sql"):
            st.markdown("**Generated SQL**")
            st.code(tool_result["cleaned_sql"], language="sql")
        st.markdown("**State**")
        st.json(
            {
                "request_type": final.get("request_type"),
                "result_type": final.get("result_type"),
                "error_bucket": final.get("error_bucket"),
                "properties": final.get("properties"),
                "timeframe": final.get("timeframe"),
                "unresolved_entities": final.get("unresolved_entities"),
                "clarify_attempts": final.get("clarify_attempts"),
            }
        )
