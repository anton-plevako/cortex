````md
# Cortex — Real Estate Asset Management Assistant

A conversational AI assistant for querying a commercial property portfolio. You ask questions in plain English, it returns precise answers backed by actual data.

Built with LangGraph, GPT-4o / GPT-5.2, DuckDB, and Streamlit.

---

## Setup

Requirements: Python 3.13, [uv](https://docs.astral.sh/uv/)

```bash
git clone https://github.com/anton-plevako/cortex.git
cd cortex
uv sync
```

Create a `.env` file:

```env
OPENAI_API_KEY=sk-...
LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_API_KEY=lsv2_...
LANGSMITH_PROJECT=cortex
```

Run the app:

```bash
uv run streamlit run app.py
```

---

## What it does

The system answers natural-language questions about a portfolio of commercial properties. Some examples of what works:

- "Which property made the most money in 2024?"
- "Compare Building 120 and Building 17 for Q4 2024."
- "What are our biggest expense categories this year?"
- "Show me Building 160's profit trend across all quarters."
- "How much parking revenue did we earn in 2024?"

Under the hood, each question gets translated into a DuckDB SQL query, executed against the ledger, and narrated back in plain language. When a query is too vague or references an unknown property, the system asks a clarifying question rather than guessing.

The dataset is `data/cortex.parquet` — a P&L ledger for a single entity (PropCo) with 5 properties, 18 tenants, 29 ledger categories, and 15 months of data (2024 full year + Q1 2025).

---

## Architecture

The core idea: all numbers come from SQL execution, not from the LLM. GPT handles language (intent, SQL drafting, narration), and DuckDB is the source of truth.

High-level flow:

- `classify` (LLM) extracts `request_type`, `timeframe`, and `raw_properties` (may be imperfect)
- `validate_and_resolve` (code) resolves properties deterministically and runs guard checks:
  - `unclear` → `CLARIFY_QUERY`
  - invalid timeframe → `FALLBACK_NO_DATA` (terminal)
  - unresolved property names → `CLARIFY_PROPERTY`
  - otherwise routes to SQL
- `sql_agent` ↔ `sql_executor` loops until `execute_sql` returns success or a terminal failure status
- `handle_sql_result` decides whether to answer or route into the clarify hub
- Clarify hub handles both clarification (`CLARIFY_*`) and terminal failures (`FALLBACK_*`) via `interrupt()` + `MemorySaver` resume

### File layout

```text
src/cortex/
  config.py        — constants and valid time periods loaded from parquet at startup
  state.py         — AssetState TypedDict passed between all nodes
  db.py            — DuckDB connection + schema summary injected into prompts
  tools.py         — execute_sql tool + resolve_property helper
  prompts.py       — all system prompts in one place
  nodes/           — one file per node (incl. clarify hub nodes)
  graph_routes.py  — routing functions (state -> next node)
  graph.py         — graph wiring and compilation
app.py             — Streamlit UI
test_graph.py      — test suite
data/cortex.parquet
```

---

## LangGraph pipeline

Each node has a single job. Routing is driven by state (`next_action`, `error_bucket`, etc.), using small route functions.

- `classify` (LLM) — classify intent + extract raw slots (property mentions, timeframe, etc.)
- `validate_and_resolve` (code) — resolves `raw_properties` → `properties`, validates timeframe, and sets `next_action` + `error_bucket`
- `sql_agent` (LLM) — generate DuckDB `SELECT` + tool-call `execute_sql`
- `sql_executor` (ToolNode) — executes `execute_sql`
- `handle_sql_result` (code) — interprets tool output, sets `next_action` (answer vs clarify)
- `answer` (LLM) — narrates the final result (uses `user_query_original` for display)

Clarification / error handling is a small hub (6 nodes):

- `clarify_entry` (code) — dispatches based on `error_bucket`, attempts, and whether a question is already staged
- `clarify_question` (LLM) — generates the first clarification question for `CLARIFY_*`
- `clarify_interrupt` (`interrupt`) — pauses + stores Q/A + resets messages so SQL restarts clean
- `clarify_policy` (LLM, structured output) — decides: ask again / apply / fallback
- `clarify_apply` (code) — applies the answer (patches `user_query_working`, or sets `raw_properties`) then routes back to `validate_and_resolve` or `classify`
- `clarify_fallback` (LLM) — terminal explanation for `FALLBACK_*` or max attempts

Loops:

- `sql_agent` ↔ `sql_executor` loops until SQL is terminal (`ok` / `no_data` / `exec_error`, etc.)
- Clarify hub loops through interrupt/resume until resolved or attempts are exhausted (uses `pending_question` to avoid regenerating questions on reruns). Interrupt/resume behavior is the standard LangGraph pattern.
- `messages` is reset via `Overwrite([])` to bypass the reducer and start clean after clarification.

---

## Shared state

All nodes read and write a single `AssetState` `TypedDict`:

```text
user_query              # original user input (kept as-is)
user_query_original     # immutable copy used for display/narration
user_query_working      # mutable working copy used for re-classify / SQL after clarification

request_type            # "pnl" | "comparison" | "details" | "general" | "unclear" | "off_topic"
raw_properties          # extracted property strings (may include unresolved); can be overwritten after CLARIFY_PROPERTY
properties              # canonical property names (written by validate_and_resolve)

timeframe               # {year, quarter, month}

next_action             # routing signal used across nodes:
                        # core: "sql" | "answer" | "clarify"
                        # clarify hub internal: "question" | "policy" | "interrupt" | "apply" |
                        #                      "validate" | "classify" | "fallback"

error_bucket            # "CLARIFY_PROPERTY" | "CLARIFY_QUERY" |
                        # "FALLBACK_OFF_TOPIC" | "FALLBACK_NO_DATA" | "FALLBACK_EXEC_ERROR" | ""

tool_result             # {status, error_message, rows, row_count, columns, cleaned_sql}
result                  # final text shown to user
result_type             # "answer" | "clarify" | "fallback" (Streamlit display)

messages                # message history for the sql_agent <-> tool loop (reset after interrupt)
pending_question        # staged question to surface via interrupt() (replay-safe)
last_clarify_question   # last question shown to the user
last_clarify_answer     # user’s last answer
clarify_attempts        # incremented on each interrupt; gates MAX_CLARIFY_ATTEMPTS
```

---

## Design decisions

- **Schema injection vs. RAG:** The schema is small (one table), but the data has quirks so I inject the full schema + rules into the SQL prompt so every query has the same context and there are no retrieval misses.

- **Structured output:** For anything that drives control flow (intent, extracted slots, clarify decisions), I use `with_structured_output(PydanticModel)` so state stays typed and routing stays deterministic.

- **MemorySaver + interrupt:** Clarifications use LangGraph `interrupt()` and Streamlit resumes with `Command(resume=answer)` on the same `thread_id`, so the conversation continues instead of restarting.

- **LLM vs code split:** LLMs handle language (intent, SQL drafting, narration). Code handles correctness (validation, execution, retries, routing).

- **Centralized error handling:** Errors are routed into a single clarify hub. I refactored it from a monolith into smaller nodes to fix an infinite clarify loop and make resume/replay safe.

---

## Challenges & how I solved them

- **Tool-per-intent didn’t scale:** Started with multiple rigid tools (`get_property_pnl`, `compare_properties`, etc.). It broke quickly on novel questions (rankings, trends, category breakdowns). Switched to a single SQL-generating agent + one tool: `execute_sql`.

- **SQL correctness required data-specific rules:** The ledger has quirks (sign-encoded profit, entity-level costs with `NULL property_name`, etc.). I explored the data and wrote an explicit schema summary + SQL conventions into the SQL agent prompt, with examples. The agent also retries when `execute_sql` returns `bad_sql`.
- **Property matching could silently produce wrong answers:** Fuzzy matching on numeric names can be dangerous (“10” looking like “180”) in the initial approach. I fixed this with deterministic numeric-first resolution and only using fuzzy matching for non-numeric inputs.

- **Clarification + error handling loops were tricky:** I bucket failures into `CLARIFY_PROPERTY`, `CLARIFY_QUERY`, and terminal `FALLBACK_*` cases. `validate_and_resolve` sets these buckets deterministically (e.g. invalid time → `FALLBACK_NO_DATA`, unresolved names → `CLARIFY_PROPERTY`). I originally hit an infinite clarify-loop edge case, so I refactored the logic into a small clarify hub (`clarify_entry → clarify_question → interrupt → clarify_policy → clarify_apply/clarify_fallback`). The hub stages questions via `pending_question` to stay replay-safe on reruns, resets `messages` after a human answer with `Overwrite([])` so the SQL agent restarts clean, and applies the result deterministically (patch `user_query_working` / override `raw_properties`, or fall back).

- **Streamlit + interrupt/resume integration:** Streamlit reruns the script, so I had to be careful with session state and stable `thread_id`. The graph uses `MemorySaver`, and the UI resumes with `Command(resume=answer)` so the conversation continues instead of restarting.
````
