"""run_sql tool: read-only SELECT against the session's DuckDB (PRD FR4)."""
from __future__ import annotations

import threading

import pandas as pd
import sqlglot
from sqlglot import exp

from analyst_agent.config import SQL_ROW_LIMIT, SQL_TIMEOUT_S


class UnsafeSQLError(ValueError):
    pass


_FORBIDDEN = (exp.Insert, exp.Update, exp.Delete, exp.Drop, exp.Create, exp.Alter,
              exp.Command, exp.Merge, exp.TruncateTable)


def validate_select(query: str) -> str:
    """Allow exactly one SELECT/WITH statement; raise UnsafeSQLError otherwise."""
    try:
        statements = [s for s in sqlglot.parse(query, read="duckdb") if s is not None]
    except sqlglot.errors.ParseError as e:
        raise UnsafeSQLError(f"Could not parse SQL: {e}") from e
    if len(statements) != 1:
        raise UnsafeSQLError("Send exactly one SQL statement.")
    stmt = statements[0]
    if not isinstance(stmt, exp.Query):
        raise UnsafeSQLError("Only SELECT queries are allowed.")
    for node in stmt.walk():
        if isinstance(node, _FORBIDDEN):
            raise UnsafeSQLError(f"Statement type {type(node).__name__} is not allowed.")
        fname = type(node).__name__.lower() if isinstance(node, exp.Func) else ""
        if isinstance(node, exp.Anonymous):
            fname = node.name.lower()
        if fname.startswith("read") or fname in {"glob", "getenv", "query", "query_table"}:
            raise UnsafeSQLError("Reading files or environment from SQL is not allowed; use the loaded tables.")
    return query.strip().rstrip(";")


def run_sql(con, query: str) -> pd.DataFrame:
    """Execute a validated query with a timeout. Returns the full result as a DataFrame."""
    q = validate_select(query)
    result: dict = {}
    cur = con.cursor()

    def _work():
        try:
            result["df"] = cur.execute(q).df()
        except Exception as e:  # surfaced to the LLM so it can fix the query
            result["err"] = e

    t = threading.Thread(target=_work, daemon=True)
    t.start()
    t.join(SQL_TIMEOUT_S)
    if t.is_alive():
        cur.interrupt()
        raise TimeoutError(f"Query exceeded {SQL_TIMEOUT_S}s and was cancelled.")
    if "err" in result:
        raise result["err"]
    return result["df"]


def df_preview(df: pd.DataFrame, limit: int = SQL_ROW_LIMIT) -> dict:
    """What the LLM sees: columns, row count and the first `limit` rows."""
    shown = df.head(limit)
    return {
        "columns": [str(c) for c in df.columns],
        "row_count": int(len(df)),
        "rows": shown.astype(object).where(shown.notna(), None).map(_scalar).values.tolist(),
        "truncated": len(df) > limit,
    }


def _scalar(v):
    if isinstance(v, pd.Timestamp):
        return v.isoformat()[:19]
    if isinstance(v, float):
        return round(v, 6)
    if hasattr(v, "item"):
        return v.item()
    return v
