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
st.session_state.setdefault("history", [])
st.session_state.setdefault("current_question", "")
st.session_state.setdefault("was_clarified", False)
st.session_state.setdefault("query_input", "")
st.session_state.setdefault("pending_query", None)

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
        "Compare Building 120 and Building 160 across all quarters in 2024",
        "Show me the financials for Building 12 in 2024",
    ]
    for ex in examples:
        if st.button(ex, key=ex, use_container_width=True):
            st.session_state["query_input"] = ex
            st.session_state["thread_id"] = str(uuid.uuid4())
            st.session_state["awaiting_clarification"] = False
            st.session_state["clarify_question"] = ""
            st.session_state["current_question"] = ""
            st.session_state["was_clarified"] = False
            st.rerun()

# ── Conversation history ─────────────────────────────────────────────────────
for item in st.session_state["history"]:
    question_label = item["question"] + (" *(clarified)*" if item.get("was_clarified") else "")
    with st.chat_message("user"):
        st.markdown(question_label)
    with st.chat_message("assistant"):
        if item["result_type"] == "fallback":
            st.warning(item["result"])
        else:
            st.markdown(item["result"])
        rows = item.get("raw_data", {}).get("rows", [])
        columns = item.get("raw_data", {}).get("columns", [])
        if rows and len(rows) > 1 and columns:
            try:
                df = pd.DataFrame(rows, columns=columns)
                numeric_cols = df.select_dtypes("number").columns.tolist()
                if numeric_cols:
                    st.dataframe(
                        df.style.format(
                            {c: "€{:,.0f}" for c in numeric_cols}, na_rep="—"
                        ),
                        use_container_width=True,
                    )
            except Exception:
                pass
        with st.expander("Debug – pipeline state"):
            if item.get("sql"):
                st.code(item["sql"], language="sql")
            st.json(item["state"])

# ── Clarification prompt (shown when graph is waiting for more info) ─────────
clarify_placeholder = st.empty()
if st.session_state["awaiting_clarification"]:
    clarify_placeholder.info(st.session_state["clarify_question"])
    placeholder = "Your answer…"
    label = "Clarification"
else:
    placeholder = "Ask anything about your portfolio…"
    label = "Your question"

with st.form("ask_form", clear_on_submit=False):
    st.text_area(label, placeholder=placeholder, height=80, key="query_input")
    submitted = st.form_submit_button("Ask", type="primary")

if submitted:
    st.session_state["pending_query"] = (st.session_state["query_input"] or "").strip()
    st.rerun()

_btn_label = "✕ Cancel & start over" if st.session_state["awaiting_clarification"] else "+ New conversation"
if st.button(_btn_label):
    st.session_state["thread_id"] = str(uuid.uuid4())
    st.session_state["awaiting_clarification"] = False
    st.session_state["clarify_question"] = ""
    st.session_state["history"] = []
    st.session_state["current_question"] = ""
    st.session_state["was_clarified"] = False
    st.rerun()

query = st.session_state["pending_query"]
if query:
    st.session_state["pending_query"] = None

    if not st.session_state["awaiting_clarification"]:
        st.session_state["current_question"] = query
        st.session_state["was_clarified"] = False

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
            st.session_state["was_clarified"] = True
            st.session_state["awaiting_clarification"] = True
            st.session_state["clarify_question"] = question or "Could you clarify your question?"
            st.rerun()
        else:
            st.session_state["awaiting_clarification"] = False
            st.session_state["clarify_question"] = ""
            clarify_placeholder.empty()  # remove the blue question box

    result_type = final.get("result_type", "answer")
    tool_result = final.get("tool_result") or {}
    st.session_state["history"].append({
        "question":      st.session_state["current_question"],
        "result_type":   result_type,
        "result":        final.get("result", "No result returned."),
        "raw_data":      final.get("raw_data") or {},
        "was_clarified": st.session_state["was_clarified"],
        "sql":           tool_result.get("cleaned_sql"),
        "state": {
            "request_type":        final.get("request_type"),
            "result_type":         result_type,
            "error_bucket":        final.get("error_bucket"),
            "properties":          final.get("properties"),
            "timeframe":           final.get("timeframe"),
            "unresolved_entities": final.get("unresolved_entities"),
            "clarify_attempts":    final.get("clarify_attempts"),
        },
    })
    st.rerun()
