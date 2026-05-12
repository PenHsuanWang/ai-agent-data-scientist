"""
PHASE 2 — STEP 4: ACI — Safe SQL Read Tool
============================================
Harness Pillar: ACI (Agent-Computer Interface) → Observation Layer
SDK Docs: https://docs.anthropic.com/en/agents-and-tools/tool-use/define-tools

Goal: Build a read-only SQL tool that an agent can safely use.
      Implement essential ACI guardrails:
        - Enforce SELECT-only (no mutations)
        - Enforce LIMIT clause (no full-table scans)
        - Catch SQL errors and return them as structured JSON
          so the agent can self-correct (refinement loop)

ACI Hierarchy:
  Context/Maps → Observation (READ) → Compute → Action (WRITE)
                      ↑
                 We build this layer
"""

import json
import re
import sqlite3
import tempfile
import os
import anthropic
from dotenv import load_dotenv

load_dotenv()
client = anthropic.Anthropic()


# ─────────────────────────────────────────────
# Setup: Create an in-memory demo SQLite database
# (In production: replace with your real DB connection)
# ─────────────────────────────────────────────
def create_demo_db() -> str:
    """Create a temporary SQLite database for demo purposes."""
    db_path = tempfile.mktemp(suffix=".db")
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.executescript("""
        CREATE TABLE employees (
            id INTEGER PRIMARY KEY,
            name TEXT,
            department TEXT,
            salary REAL,
            hire_date TEXT
        );

        CREATE TABLE sales (
            id INTEGER PRIMARY KEY,
            employee_id INTEGER,
            amount REAL,
            sale_date TEXT,
            product TEXT
        );

        INSERT INTO employees VALUES
            (1, 'Alice Chen', 'Engineering', 120000, '2021-03-15'),
            (2, 'Bob Smith', 'Sales', 85000, '2020-07-01'),
            (3, 'Carol White', 'Engineering', 135000, '2019-11-20'),
            (4, 'David Lee', 'Marketing', 78000, '2022-01-10'),
            (5, 'Eva Brown', 'Sales', 92000, '2020-09-05');

        INSERT INTO sales VALUES
            (1, 2, 15000.00, '2024-01-15', 'Enterprise License'),
            (2, 5, 8500.00, '2024-01-20', 'Pro Subscription'),
            (3, 2, 22000.00, '2024-02-01', 'Enterprise License'),
            (4, 5, 6000.00, '2024-02-10', 'Consulting'),
            (5, 2, 18000.00, '2024-03-05', 'Enterprise License');
    """)
    conn.commit()
    conn.close()
    return db_path


# ─────────────────────────────────────────────
# ACI GUARDRAIL LAYER — This is the harness
# that prevents the agent from doing damage
# ─────────────────────────────────────────────
MAX_ROWS = 50
FORBIDDEN_PATTERNS = [
    r"\bDROP\b", r"\bDELETE\b", r"\bINSERT\b", r"\bUPDATE\b",
    r"\bALTER\b", r"\bTRUNCATE\b", r"\bCREATE\b", r"\bREPLACE\b",
]

def safe_sql_query(query: str, db_path: str) -> dict:
    """
    Execute a SQL query with safety guardrails:
      1. Block all non-SELECT statements
      2. Inject LIMIT if missing
      3. Return structured JSON (success or error)
      → The agent receives this JSON and can self-correct
    """
    query = query.strip()

    # Guardrail 1: Only SELECT allowed
    if not query.upper().lstrip().startswith("SELECT"):
        return {
            "success": False,
            "error": "FORBIDDEN: Only SELECT statements are allowed. Mutations are not permitted.",
            "query": query,
        }

    # Guardrail 2: Block mutation keywords (even inside subqueries)
    for pattern in FORBIDDEN_PATTERNS:
        if re.search(pattern, query, re.IGNORECASE):
            return {
                "success": False,
                "error": f"FORBIDDEN: Query contains disallowed keyword matching '{pattern}'.",
                "query": query,
            }

    # Guardrail 3: Inject LIMIT if missing (protect against full-table scans)
    if "LIMIT" not in query.upper():
        query = f"{query.rstrip(';')} LIMIT {MAX_ROWS};"

    # Execute
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute(query)
        rows = [dict(row) for row in cur.fetchall()]
        conn.close()
        return {
            "success": True,
            "row_count": len(rows),
            "rows": rows,
            "note": f"Results capped at {MAX_ROWS} rows." if len(rows) == MAX_ROWS else None,
        }
    except sqlite3.Error as e:
        # Return the error as structured data — the agent can use this to self-correct
        return {
            "success": False,
            "error": str(e),
            "error_type": "SQL_SYNTAX_ERROR",
            "query": query,
            "hint": "Check column names and table names. Use the list_tables tool first.",
        }


def list_tables(db_path: str) -> dict:
    """Return the database schema — the 'Map' in the ACI hierarchy."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cur.fetchall()]

    schema = {}
    for table in tables:
        cur.execute(f"PRAGMA table_info({table});")
        schema[table] = [
            {"column": row[1], "type": row[2], "nullable": not row[3]}
            for row in cur.fetchall()
        ]
    conn.close()
    return {"tables": tables, "schema": schema}


# ─────────────────────────────────────────────
# Tool definitions for the agent
# ─────────────────────────────────────────────
def make_tools() -> list:
    return [
        {
            "name": "list_tables",
            "description": (
                "Get the database schema: all table names and their column definitions. "
                "ALWAYS call this first before writing any SQL query. "
                "This is your map of the data available to you."
            ),
            "input_schema": {"type": "object", "properties": {}, "required": []},
        },
        {
            "name": "run_sql",
            "description": (
                "Execute a read-only SQL SELECT query against the database. "
                "Returns rows as a list of JSON objects. "
                "Only SELECT statements are allowed — no INSERT, UPDATE, or DELETE. "
                "A LIMIT clause will be automatically added if you forget it. "
                "If the query fails, you will receive the SQL error — fix it and retry."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "A valid SQL SELECT statement."
                    }
                },
                "required": ["query"],
            },
        },
    ]


# ─────────────────────────────────────────────
# Agent runner with SQL dispatch
# ─────────────────────────────────────────────
def run_sql_agent(question: str, db_path: str, verbose: bool = True) -> str:
    messages = [{"role": "user", "content": question}]
    tools = make_tools()
    iteration = 0

    while iteration < 10:
        iteration += 1
        response = client.messages.create(
            model="claude-opus-4-5-20251101",
            max_tokens=1024,
            tools=tools,
            system=(
                "You are a data analyst with read-only SQL access. "
                "Always call list_tables first to understand the schema. "
                "If a SQL query fails, read the error and fix it — do not give up. "
                "Present your final answer in a clear, formatted way."
            ),
            messages=messages,
        )

        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn":
            return next((b.text for b in response.content if hasattr(b, "text")), "")

        if response.stop_reason == "tool_use":
            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue

                if verbose:
                    print(f"\n[Tool] {block.name}({json.dumps(block.input)[:100]})")

                # Dispatch to the appropriate function
                if block.name == "list_tables":
                    result = list_tables(db_path)
                elif block.name == "run_sql":
                    result = safe_sql_query(block.input["query"], db_path)
                else:
                    result = {"error": f"Unknown tool: {block.name}"}

                result_str = json.dumps(result, indent=2)
                if verbose:
                    print(f"[Result] {result_str[:300]}")

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result_str,
                })

            messages.append({"role": "user", "content": tool_results})

    return "Agent reached iteration limit."


# ─────────────────────────────────────────────
# Demo
# ─────────────────────────────────────────────
if __name__ == "__main__":
    db_path = create_demo_db()

    questions = [
        "Who are the top 3 salespeople by total sales amount?",
        "What is the average salary by department?",
        # This one should trigger the self-correction loop:
        "Show me records from the 'orders' table",  # table doesn't exist!
    ]

    for question in questions:
        print("\n" + "=" * 60)
        print(f"QUESTION: {question}")
        print("=" * 60)
        answer = run_sql_agent(question, db_path, verbose=True)
        print(f"\nANSWER:\n{answer}")

    os.unlink(db_path)


print("""
KEY TAKEAWAYS
=============
1. The ACI Observation Layer is READ-ONLY by default — protect your data.
2. Always return errors as structured JSON, not exceptions — the agent
   reads the error and self-corrects (the refinement loop).
3. list_tables = the "Map" tool — Claude always calls this first to
   understand what data is available (Cognitive Framework pillar).
4. Inject LIMIT automatically — never trust the agent to remember this.
5. The forbidden keyword check is a deterministic guardrail that runs
   BEFORE the LLM sees any data — it's the first line of defense.
""")
