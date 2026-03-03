# Cortex – Refactoring Plan
## From rigid-tool pipeline → SoTA NL→SQL multi-agent system

---

## 0. Original Goals (non-negotiable)

From the spec:
- Multi-agent system orchestrated by **LangGraph**
- Handles **all types** of natural-language questions about the portfolio
  ("All types of question need to be handled" — spec §2)
- Detect request type, extract details, retrieve from dataset, perform calculations,
  generate a clear step-by-step response
- **Robust to vague, incomplete, and unexpected inputs** — no dead-end "unknown" path
- Streamlit UI for basic interaction
- Python + GPT-4o

Example outputs the spec expects:
```
"The asset at 123 Main St is priced at $500,000, while the asset at 456 Oak Ave is priced at $450,000."
"The total P&L for all your properties this year is $1,200,000."
"Details for the property at 789 Pine Ln: Address - 789 Pine Ln, Value - $300,000, Last Appraisal Date - 2024-01-15."
```

---

## 1. What we have now (v1 audit)

### Strengths
- classify_node: LLM structured output, few-shot, works well
- extract_node: entity + time resolution, fuzzy name matching, retry loop
- response_node: grounded LLM narration, uses raw JSON data
- clarify_node: LLM-reasoned, context-aware
- Streamlit UI, config, state, tests — all solid

### Gaps vs. spec and SoTA
| Gap | Root cause |
|---|---|
| `unknown` → clarify (dead end) | Classifier gates access, not just guides it |
| Only 3 query types supported | compute_node is a hardcoded if/elif |
| Cross-period comparison impossible | extract_node holds only one timeframe |
| Superlatives, trends, rankings fail | No flexible query mechanism |
| Category-specific queries unreliable | No column-level filter in tools |
| Compute is rigid, not intelligent | 5 fixed @tool functions, not SQL |
| compute_node does arithmetic | Spec: LLM should narrate, not calculate |

---

## 2. Target Architecture

### Core principle (from SoTA)
> All numeric answers come from **deterministic execution** (DuckDB).
> The LLM is used for intent, extraction, query planning, and narration — not arithmetic.

### Execution substrate: DuckDB
DuckDB reads Parquet directly. SQL is safer than Python exec, easier to validate,
and the LLM has deep SQL training. One in-memory connection, two views:
- `ledger` — `read_parquet('cortex.parquet')` — the full P&L data
- `property_metadata` — from properties.json — addresses, values, appraisal dates

### New graph
```
START
  │
  ▼
classify_node          keeps existing — sets request_type (hint, not gate)
  │
  ▼
extract_node           keeps existing — sets properties, timeframe (context for SQL planner)
  │  [exhausted retries on entity error → clarify]
  ▼
plan_sql_node          NEW — LLM generates DuckDB SQL from schema + context
  ▲                         request_type and properties are HINTS, not constraints
  │ [invalid SQL]            unknown type → still attempts SQL (no dead end)
  │
validate_node          NEW — deterministic safety gate (columns, timeframe, SELECT-only)
  │  [valid]
  ▼
execute_node           NEW — runs SQL via DuckDB, serialises result
  │  [runtime error → back to plan_sql up to MAX_SQL_ATTEMPTS]
  │  [exhausted → clarify]
  ▼
response_node          keeps existing — LLM narrates the execution result
  │
  ▼
 END

clarify_node           genuine last resort only:
                         - entity resolution failed after retries, OR
                         - SQL generation + execution exhausted all retries
```

### Key change in routing philosophy
| Before | After |
|---|---|
| classify gates: unknown → hard stop | classify hints: unknown → plan_sql attempts answer |
| 3 supported types, rest fail | All types route through SQL planner |
| extract_node validates query type | extract_node provides context, not control |

---

## 3. File-by-file changes

### NEW: `src/cortex/db.py`
DuckDB connection factory. Called once at import, cached.
```python
# Creates in-memory DuckDB connection with:
#   VIEW ledger   → read_parquet(DATA_PATH)
#   TABLE property_metadata → from PROPERTY_METADATA dict
# Exposes: get_connection() -> duckdb.DuckDBPyConnection
# Exposes: SCHEMA_SUMMARY (string injected into SQL planner prompt)
```

Schema summary the planner will receive:
```
TABLE: ledger
  entity_name     VARCHAR    — always "PropCo"
  property_name   VARCHAR    — Building 17|120|140|160|180, NULL = entity-level costs
  tenant_name     VARCHAR
  ledger_type     VARCHAR    — "revenue" | "expenses"
  ledger_group    VARCHAR    — rental_income | sales_discounts | general_expenses |
                               management_fees | taxes_and_insurances
  ledger_category VARCHAR    — e.g. revenue_rent_taxed, interest_mortgage, insurance_in_general
  profit          DOUBLE     — positive=revenue, negative=expenses (single sign-encoded column)
  month           VARCHAR    — "2024-M01" format
  quarter         VARCHAR    — "2024-Q1" format
  year            VARCHAR    — "2024" | "2025"

TABLE: property_metadata
  property_name       VARCHAR
  address             VARCHAR
  current_value       BIGINT
  purchase_price      BIGINT
  last_appraisal_date VARCHAR

NOTE: rows with property_name IS NULL are entity-level costs (mortgage, management fees, etc.)
Available years: 2024, 2025 (2025 = Q1 only: Jan–Mar)
```

### MODIFY: `src/cortex/state.py`
Add fields for the new nodes:
```python
class AssetState(TypedDict, total=False):
    # existing
    user_query: str
    request_type: str
    properties: list[str]
    timeframe: Timeframe
    extract_attempts: int
    raw_data: dict
    result: str
    error: str
    # new
    sql_query: str        # generated SQL string
    sql_result: str       # JSON-serialised query result
    plan_attempts: int    # SQL generation + execution retries
```

### MODIFY: `src/cortex/nodes.py`

#### ADD: `plan_sql_node`
- System prompt: full schema summary + property names + date format examples + 3-4 SQL examples
- Context injected: `request_type`, `properties`, `timeframe` as hints
- For `unknown` type: prompt explicitly says "the intent is unclear — write the most useful SQL you can"
- Structured output: `{"sql": "SELECT ...", "explanation": "..."}`
- MAX_SQL_ATTEMPTS = 3

```
Prompt strategy:
  - Tell LLM it is writing DuckDB SQL
  - Schema summary (from db.py)
  - Hint: request_type, extracted entities, timeframe
  - Rule: always end with a SELECT that returns a result
  - Rule: use property_metadata JOIN for value/address queries
  - Rule: NULL property_name rows = entity-level costs (include for portfolio totals)
  - Examples covering: P&L, comparison, details, cross-period, category-specific,
    superlatives, metadata lookups
```

#### ADD: `validate_node`
Deterministic checks (no LLM):
1. Statement starts with SELECT (no INSERT/UPDATE/DROP/EXEC)
2. No system calls: no `read_csv`, `read_json` with external paths, no `COPY TO`
3. All column names referenced exist in ledger or property_metadata schema
4. If year/quarter/month filter present: value is within known valid range
5. Adds `LIMIT 500` if no LIMIT present (safety)

On failure: sets `sql_error`, routes back to `plan_sql_node` with error context.
On success: routes to `execute_node`.

#### ADD: `execute_node`
```python
conn = get_connection()
relation = conn.execute(state["sql_query"])
df = relation.df()
# Serialise: DataFrame → dict (orient="records" or summary dict)
# Store in state["sql_result"] (JSON string) and state["raw_data"] (dict for Streamlit)
```
On exception: sets `sql_error`, increments `plan_attempts`, routes to `plan_sql_node` or clarify.

#### KEEP + UPDATE: `classify_node`
No structural change. Add `general` to the type enum alongside unknown to distinguish
"off-topic" from "valid but unclassified":
```python
request_type: Literal["comparison", "pnl", "details", "general", "unknown"]
```
`general` — portfolio/market questions answerable from data but not fitting the three types.
`unknown` — genuinely off-topic (weather, etc.).
Both route to `plan_sql_node`; only `unknown` gets a weaker SQL attempt hint.

#### KEEP + UPDATE: `extract_node`
- Remove the `unknown` guard — extract runs for all types (provides useful hints)
- Enrich output: add `metrics` (profit/revenue/expenses), `group_by` (ledger_category/quarter)
  so the SQL planner has richer context
- Keep fuzzy name resolution (still needed for entity mapping)
- Keep retry loop for unresolvable entity errors

#### KEEP + UPDATE: `response_node`
- Now receives `sql_result` (JSON) instead of `raw_data` from hardcoded tools
- Prompt addition: "include the key SQL filters used (time period, property) in your response"
- Prompt addition: "for property value queries, cite the appraisal date"

#### KEEP + UPDATE: `clarify_node`
- Fix the lie: remove "There are NO asset prices, valuations" — we have PROPERTY_METADATA
- Updated scope: "This assistant can answer any question about the portfolio's
  P&L, property values, comparisons, and trends from the available data."
- New trigger context: "SQL generation failed after N attempts" is a new failure reason

#### REMOVE: `compute_node`
Replaced entirely by plan_sql + validate + execute.

### MODIFY: `src/cortex/tools.py`
- Remove all `@tool` decorated functions (calculate_total_pnl, get_property_pnl,
  get_property_details, compare_properties, list_properties, get_asset_value)
- Keep `_resolve_property()` — still used by extract_node for fuzzy entity mapping
- Keep `_load_df()` — used by config.py (or migrate to DuckDB)
- File may be renamed to `resolution.py` for clarity

### MODIFY: `src/cortex/graph.py`
New edges:
```python
graph.add_node("classify", classify_node)
graph.add_node("extract", extract_node)
graph.add_node("plan_sql", plan_sql_node)
graph.add_node("validate", validate_node)
graph.add_node("execute", execute_node)
graph.add_node("respond", response_node)
graph.add_node("clarify", clarify_node)

graph.add_edge(START, "classify")
graph.add_edge("classify", "extract")           # all types — extract is now always a hint
graph.add_conditional_edges("extract", _route_after_extract,
    {"plan_sql": "plan_sql", "clarify": "clarify"})
graph.add_conditional_edges("validate", _route_after_validate,
    {"plan_sql": "plan_sql", "execute": "execute"})
graph.add_conditional_edges("execute", _route_after_execute,
    {"plan_sql": "plan_sql", "respond": "respond", "clarify": "clarify"})
graph.add_edge("plan_sql", "validate")
graph.add_edge("respond", END)
graph.add_edge("clarify", END)
```

### MODIFY: `app.py`
- Debug expander: add `st.code(final.get("sql_query"), language="sql")`
- Table display: generalise from `by_property` dict check to handle any tabular sql_result
- Sidebar: unchanged

### MODIFY: `test_graph.py`
- Keep existing 25 questions (regression)
- Add 10 odd-ball questions from the analysis (cross-period, superlatives, category-specific,
  out-of-range dates, terse input)

---

## 4. Implementation phases

### Phase 1 — Infrastructure (no behaviour change)
- [ ] `uv add duckdb`
- [ ] Create `src/cortex/db.py`: DuckDB connection, ledger view, property_metadata table,
      SCHEMA_SUMMARY string constant
- [ ] Update `src/cortex/state.py`: add `sql_query`, `sql_result`, `plan_attempts`
- [ ] Verify DuckDB can query parquet and return results matching current pandas output

### Phase 2 — SQL planning node
- [ ] Write `plan_sql_node` in nodes.py
- [ ] Design system prompt with schema, examples, rules
- [ ] Add `_SQL_PLAN_SYSTEM` prompt constant
- [ ] Structured output: `_SqlPlan(BaseModel)` with `sql: str` field
- [ ] Manual test: invoke plan_sql_node in isolation on 5 diverse queries

### Phase 3 — Validation + execution nodes
- [ ] Write `validate_node`: column check, SELECT-only, LIMIT guard
- [ ] Write `execute_node`: DuckDB run, DataFrame → dict serialisation, error handling
- [ ] Unit test validate + execute against known good and bad SQL strings

### Phase 4 — Graph rewiring
- [ ] Update graph.py with new nodes and edges
- [ ] Update routing functions (_route_after_extract, _route_after_validate, _route_after_execute)
- [ ] Remove compute_node and its edges
- [ ] Route unknown/general through plan_sql (not clarify)
- [ ] Regenerate cortex_flow.png

### Phase 5 — Node updates
- [ ] Update classify_node: add `general` type to Literal
- [ ] Update extract_node: remove unknown gate, enrich with metrics/group_by fields
- [ ] Update response_node: use sql_result, add citation instructions
- [ ] Update clarify_node: fix false "no valuations" claim, update scope description

### Phase 6 — Cleanup
- [ ] Remove @tool functions from tools.py (keep _resolve_property, _load_df)
- [ ] Remove compute_node from nodes.py
- [ ] Confirm no broken imports

### Phase 7 — Testing + Streamlit
- [ ] Run 25-question regression suite — all should still pass
- [ ] Add 10 odd-ball questions, run full 35-question suite
- [ ] Update app.py: SQL in debug expander, generalised table display
- [ ] Manual Streamlit smoke test: 5 queries covering each path

---

## 5. SQL prompt examples (seed for plan_sql_node prompt)

```sql
-- P&L for one property, one year
SELECT property_name, SUM(CASE WHEN ledger_type='revenue' THEN profit ELSE 0 END) AS revenue,
       SUM(CASE WHEN ledger_type='expenses' THEN profit ELSE 0 END) AS expenses,
       SUM(profit) AS net_profit
FROM ledger WHERE property_name='Building 120' AND year='2024' GROUP BY property_name;

-- Portfolio total including entity-level costs
SELECT SUM(profit) AS net_profit,
       SUM(CASE WHEN ledger_type='revenue' THEN profit ELSE 0 END) AS revenue,
       SUM(CASE WHEN ledger_type='expenses' THEN profit ELSE 0 END) AS expenses
FROM ledger WHERE year='2024';

-- Cross-period comparison (impossible with old tools)
SELECT quarter, SUM(profit) AS net_profit FROM ledger
WHERE property_name='Building 120' AND quarter IN ('2024-Q1','2025-Q1') GROUP BY quarter;

-- Most profitable property (superlative — impossible with old tools)
SELECT property_name, SUM(profit) AS net_profit FROM ledger
WHERE year='2024' AND property_name IS NOT NULL
GROUP BY property_name ORDER BY net_profit DESC LIMIT 1;

-- Property value + P&L joined (metadata query)
SELECT m.property_name, m.current_value, m.purchase_price,
       m.current_value - m.purchase_price AS unrealised_gain,
       SUM(l.profit) AS annual_pnl
FROM property_metadata m
LEFT JOIN ledger l ON l.property_name = m.property_name AND l.year = '2024'
GROUP BY m.property_name, m.current_value, m.purchase_price;

-- Category-specific across portfolio (impossible with old tools)
SELECT ledger_category, SUM(profit) AS total FROM ledger
WHERE ledger_category LIKE '%parking%' AND year='2024' GROUP BY ledger_category;

-- Quarter trend for one building
SELECT quarter, SUM(profit) AS net_profit FROM ledger
WHERE property_name='Building 120' GROUP BY quarter ORDER BY quarter;
```

---

## 6. Risk log

| Risk | Mitigation |
|---|---|
| LLM generates wrong column name | validate_node catches it; retry with error |
| LLM generates non-SELECT SQL | validate_node blocks it |
| Result too large (thousands of rows) | validate_node adds LIMIT 500 |
| property_metadata not in DuckDB | Register as virtual table from dict at startup |
| _resolve_property still needed | Keep in tools.py / resolution.py |
| response_node prompt misreads tabular result | Update prompt to handle both scalar and tabular JSON |
| Regression on existing 25 tests | Run regression suite after Phase 4 before cleanup |

---

## 7. What does NOT change

- Python 3.13, UV, GPT-4o, temperature=0
- Streamlit as the UI
- AssetState as shared state TypedDict
- LangGraph StateGraph + conditional edges pattern
- Fuzzy property name resolution in extract_node
- The clarify_node as the terminal fallback
- data/cortex.parquet and data/properties.json (unchanged datasets)
- The retry-with-error-context pattern (proven to work in extract_node)
