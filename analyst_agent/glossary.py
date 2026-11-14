"""Business glossary: metric definitions learned from user corrections (PRD FR10).

"No, revenue should exclude refunds" -> the agent calls save_definition, and every later
question on that dataset gets the definition in its system prompt.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from analyst_agent.config import DATA_DIR

DB_PATH = DATA_DIR / "glossary.sqlite"


class Glossary:
    def __init__(self, path: Path = DB_PATH):
        self.path = Path(path)
        with self._db() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS definitions (
                dataset TEXT NOT NULL, term TEXT NOT NULL, definition TEXT NOT NULL,
                sql_expression TEXT, created_at TEXT NOT NULL,
                PRIMARY KEY (dataset, term))""")

    def _db(self):
        return sqlite3.connect(self.path)

    def save(self, dataset: str, term: str, definition: str, sql_expression: str | None = None) -> dict:
        term = term.strip().lower()
        row = (dataset, term, definition.strip(), (sql_expression or "").strip() or None,
               datetime.now(timezone.utc).isoformat(timespec="seconds"))
        with self._db() as db:
            db.execute("INSERT OR REPLACE INTO definitions VALUES (?,?,?,?,?)", row)
        return {"dataset": dataset, "term": term, "definition": row[2], "sql_expression": row[3]}

    def list(self, dataset: str | None = None) -> list[dict]:
        q, args = "SELECT dataset, term, definition, sql_expression, created_at FROM definitions", ()
        if dataset:
            q, args = q + " WHERE dataset = ?", (dataset,)
        with self._db() as db:
            rows = db.execute(q + " ORDER BY term", args).fetchall()
        keys = ("dataset", "term", "definition", "sql_expression", "created_at")
        return [dict(zip(keys, r)) for r in rows]

    def delete(self, dataset: str, term: str) -> bool:
        with self._db() as db:
            cur = db.execute("DELETE FROM definitions WHERE dataset = ? AND term = ?", (dataset, term.strip().lower()))
        return cur.rowcount > 0

    def prompt_text(self, datasets: list[str]) -> str:
        defs = [d for ds in datasets for d in self.list(ds)]
        if not defs:
            return "(none saved yet)"
        lines = []
        for d in defs:
            line = f'- "{d["term"]}" on {d["dataset"]}: {d["definition"]}'
            if d["sql_expression"]:
                line += f"  SQL: {d['sql_expression']}"
            lines.append(line)
        return "\n".join(lines)
