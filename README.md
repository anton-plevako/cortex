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

```
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

The core idea: all numbers come from SQL execution, not from the LLM. GPT-4o handles intent classification and response narration; GPT-5.2 handles SQL generation. Neither model does arithmetic.

### File layout

```
src/cortex/
  config.py        — constants and valid time periods loaded from parquet at startup
  state.py         — AssetState TypedDict passed between all nodes
  db.py            — DuckDB connection + schema description injected into prompts
  tools.py         — execute_sql tool + resolve_property helper
  prompts.py       — all system prompts in one place
  nodes/           — one file per node
  graph_routes.py  — routing functions (state -> next node)
  graph.py         — graph wiring and compilation
app.py             — Streamlit UI
test_graph.py      — 38-question test suite
data/cortex.parquet
```

### LangGraph pipeline

Each node has a single job. Routing between nodes is driven entirely by state fields — no logic lives in the graph itself.


| `classify` | LLM (GPT-4o) | Classifies intent: pnl / comparison / details / general / unclear / off_topic |
| `parse_and_validate` | LLM + code | Extracts properties and timeframe; validates them against known data |
| `sql_agent` | LLM (GPT-4o) | Writes a DuckDB SELECT and calls `execute_sql`; retries on bad SQL |
| `sql_executor` | ToolNode | Runs the tool call |
| `handle_sql_result` | Code | Reads the result JSON and sets the next routing signal |
| `answer` | LLM (GPT-4o) | Narrates the result in plain language |
| `clarify_agent` | LLM (GPT-4o) | Unified error/clarification hub: routes to question, fallback, or re-parse based on error bucket and prior attempts |
| `human_interrupt` | LangGraph interrupt | Pauses the graph, surfaces the question to the UI, stores the Q&A pair in state |

`sql_agent` and `sql_executor` loop until the SQL succeeds or retries are exhausted. `clarify_agent` and `human_interrupt` loop until the user's answer resolves the issue or the attempt limit is reached.

### Shared state

All nodes read and write a single `AssetState` TypedDict:

```python
user_query              # original question (mutated to include clarify answer only on re-parse)
request_type            # "pnl" | "comparison" | "details" | "general" | "unclear" | "off_topic"
properties              # resolved property names
timeframe               # {year, quarter, month}
next_action             # "sql" | "clarify" | "fallback" | "answer" | "ask_human" | "done"
error_bucket            # "CLARIFY_PROPERTY" | "CLARIFY_QUERY" | "FALLBACK_OFF_TOPIC" |
                        # "FALLBACK_NO_DATA" | "FALLBACK_EXEC_ERROR" | ""
tool_result             # raw JSON from execute_sql
result                  # final text shown to user
messages                # message history for the sql_agent loop
pending_question        # set by clarify_agent before interrupt; replay-safety guard
last_clarify_question   # question asked last turn
last_clarify_answer     # user's answer last turn
clarify_attempts        # incremented by human_interrupt; gates MAX_CLARIFY_ATTEMPTS
```

### Design decisions

**Schema injection vs. retrieval.**
The SQL agent needs to know the table structure, column names, and data quirks to write correct queries. One option is RAG — embed schema chunks and retrieve relevant ones per query. That's the right approach when schemas are too large to fit in a prompt. Here the schema is small enough (one table, 12 columns, 29 known category values, a few SQL examples) that full injection is simpler, faster, and more reliable. Every query gets the complete picture, with no risk of a retrieval miss leaving out something critical.

**Structured output for classification and extraction.**
The classify and parse_and_validate nodes need to return structured data — intent labels, property names, time ranges — not free text. LangChain's `with_structured_output(PydanticModel)` attaches a Pydantic schema to the model call, so the response is validated and typed before it touches the graph. No string parsing, no defensive checks for missing keys. Failures are loud and early rather than silent and downstream.

**MemorySaver for conversation continuity.**
LangGraph's `MemorySaver` checkpointer persists graph state between invocations, keyed by `thread_id`. This is what makes interrupt/resume work: when the graph pauses to ask a clarifying question, the UI collects the answer and calls `graph.invoke(Command(resume=answer), config)` — the graph picks up exactly where it left off. Each Streamlit session gets its own UUID thread ID so conversations don't bleed into each other.

**Where to trust the LLM vs. the code.**
The general rule here: LLMs handle language (intent classification, entity extraction, query narration), code handles everything else. Property validation, time range checks, SQL execution, routing decisions, and retry limits are all deterministic. The LLM never decides whether a property name is valid — `resolve_property` does. The LLM never decides whether to retry — the retry counter in `handle_sql_result` does. This makes the system predictable and testable: the parts that can be wrong are the LLM calls, and those are the parts with structured output constraints and explicit fallback paths.

---

## Challenges

**Choosing the right query mechanism.**
The first design used five rigid `@tool` functions — `get_property_pnl`, `compare_properties`, `get_property_details`, etc. This worked for the obvious cases but broke immediately on anything slightly novel: superlatives ("which property performed best?"), cross-period comparisons, category-level questions. The tool approach required writing a new function for every query shape, which doesn't scale.

The next idea was a code-generation agent — have the LLM write Python that runs against the dataframe. That's flexible but carries real risks: arbitrary code execution is hard to sandbox, errors are opaque, and it's not idiomatic for what is essentially a structured query problem. The data is already in a Parquet file, DuckDB reads Parquet natively, and LLMs have strong SQL training. Switching to a SQL-generating agent meant the only "tool" needed was `execute_sql` — a thin wrapper that validates the statement is a SELECT, runs it, and returns structured JSON. The LLM handles any question expressible as a SELECT with no additional code per query type, and the SQL is inspectable, loggable, and easy to test.

**Getting the SQL agent to produce correct queries.**
Once the architecture was decided, the actual work was understanding the data well enough to write a schema description the SQL agent could reliably use. The ledger has a few non-obvious properties that the model needs to know about explicitly: the `profit` column is sign-encoded (positive for revenue, negative for expenses), so a naive SUM gives net profit but a cost breakdown requires flipping the sign; entity-level costs like mortgage and management fees are stored with a NULL `property_name`, meaning portfolio-wide totals must include NULL rows or they're wrong; and revenue types like parking and rent are each split across two `ledger_category` values (taxed + untaxed), so querying only one returns half the actual figure.

None of this is guessable from column names alone. The solution was to explore the data first — check unique values, understand the groupings, verify the sign convention with actual numbers — then write the schema description to cover every edge case explicitly, with SQL examples that demonstrate the correct patterns. The agent also gets the full list of 29 `ledger_category` values so it never has to guess a category name.

On top of that, the agent needs a self-correction loop: if it generates invalid SQL, `execute_sql` returns `status=bad_sql` with the error, and the agent retries. Getting the prompt right meant being explicit about when to stop (after receiving a terminal status), what column aliases to use, and when to use GROUP BY vs. not.

**Building the error handling and clarification flow.**
The clarification logic turned out to be more involved than expected. A query can fail for several different reasons — genuinely off-topic, ambiguous intent, unknown property, out-of-range time period — and each warrants a different response. Off-topic is terminal. An unknown property is fixable with a follow-up question. An out-of-range time period isn't fixable — the data simply doesn't exist, so asking the user to clarify would be misleading.

The solution was to classify failure reasons into named buckets (`FALLBACK_OFF_TOPIC`, `FALLBACK_NO_DATA`, `FALLBACK_EXEC_ERROR`, `CLARIFY_PROPERTY`, `CLARIFY_QUERY`) and route everything to a single `clarify_agent` hub. Upstream nodes — `classify`, `parse_and_validate`, `handle_sql_result` — only report facts by setting `error_bucket`; the hub owns all routing policy decisions.

`clarify_agent` operates in three modes: (1) a replay-safety short-circuit that re-signals `ask_human` if the graph resumed mid-interrupt without the LLM having to run again; (2) a deterministic first-entry mode that either generates a question (`CLARIFY_*` buckets) or a terminal explanation (`FALLBACK_*`) using plain `_llm.invoke`; and (3) a re-entry mode after the user has answered, where `with_structured_output` is used so the LLM can decide whether to ask a follow-up, re-parse the query, or give up. The `human_interrupt` node stores the Q&A pair in dedicated state fields rather than appending to `user_query`, which keeps the original query clean and prevents the classifier from misreading a concatenated string on re-entry.

**Prompts referencing stale dates.**
"Last year" and "this quarter" were initially hardcoded, which meant they'd produce wrong SQL filters over time. Fixed by computing them from `datetime.now()` at startup and injecting the resolved values into the extraction prompt.

**Silent wrong answers from property name matching.**
`difflib` on full property names like "Building 180" is vulnerable to prefix dominance — "building 10" matched "building 180" at ~0.96 similarity. Fixed with a two-stage resolver: numeric inputs go through exact lookup first, fuzzy matching only applies to non-numeric inputs.

**Clarification without losing context.**
When a query can't proceed (unknown property, ambiguous intent), the system needs to ask a follow-up and resume — not restart. LangGraph's `interrupt()` handles this cleanly: the graph pauses, the UI surfaces the question, and `Command(resume=answer)` picks up the same thread with updated state.

---

## Tests

38 questions in `test_graph.py` covering the main paths and edge cases. Where the outcome is deterministic (off-topic queries, out-of-range dates, garbage input), the expected `result_type` and `error_bucket` are asserted explicitly.

```bash
uv run python test_graph.py
```
