"""Business glossary: metric definitions learned from user corrections (PRD FR10).

"No, revenue should exclude refunds" -> the agent calls save_definition, and every later
question on that dataset gets the definition in its system prompt.
"""
from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from analyst_agent.config import DATA_DIR

DB_PATH = DATA_DIR / "glossary.sqlite"


class Glossary:
    """Definitions are scoped to an owner: "local" for a single-user install, the browser session
    id in public mode, so one visitor's definitions never leak into another's answers."""

    def __init__(self, path: Path = DB_PATH, owner: str = "local"):
        self.path = Path(path)
        self.owner = owner
        with self._db() as db:
            cols = [r[1] for r in db.execute("PRAGMA table_info(definitions)")]
            if cols and "owner" not in cols:                 # migrate a pre-owner database
                db.execute("ALTER TABLE definitions RENAME TO definitions_old")
            db.execute("""CREATE TABLE IF NOT EXISTS definitions (
                owner TEXT NOT NULL, dataset TEXT NOT NULL, term TEXT NOT NULL, definition TEXT NOT NULL,
                sql_expression TEXT, created_at TEXT NOT NULL,
                PRIMARY KEY (owner, dataset, term))""")
            if cols and "owner" not in cols:
                db.execute("INSERT OR IGNORE INTO definitions SELECT 'local', * FROM definitions_old")
                db.execute("DROP TABLE definitions_old")

    def _db(self):
        return sqlite3.connect(self.path, timeout=10)

    def save(self, dataset: str, term: str, definition: str, sql_expression: str | None = None) -> dict:
        term = term.strip().lower()
        row = (self.owner, dataset, term, definition.strip(), (sql_expression or "").strip() or None,
               datetime.now(UTC).isoformat(timespec="seconds"))
        with self._db() as db:
            db.execute("INSERT OR REPLACE INTO definitions VALUES (?,?,?,?,?,?)", row)
        return {"dataset": dataset, "term": term, "definition": row[3], "sql_expression": row[4]}

    def list(self, dataset: str | None = None) -> list[dict]:
        q = "SELECT dataset, term, definition, sql_expression, created_at FROM definitions WHERE owner = ?"
        args: tuple = (self.owner,)
        if dataset:
            q, args = q + " AND dataset = ?", (self.owner, dataset)
        with self._db() as db:
            rows = db.execute(q + " ORDER BY term", args).fetchall()
        keys = ("dataset", "term", "definition", "sql_expression", "created_at")
        return [dict(zip(keys, r, strict=True)) for r in rows]

    def delete(self, dataset: str, term: str) -> bool:
        with self._db() as db:
            cur = db.execute("DELETE FROM definitions WHERE owner = ? AND dataset = ? AND term = ?",
                             (self.owner, dataset, term.strip().lower()))
        return cur.rowcount > 0

    def forget_owner(self) -> int:
        """Remove everything this owner saved (used when a public session expires)."""
        with self._db() as db:
            return db.execute("DELETE FROM definitions WHERE owner = ?", (self.owner,)).rowcount

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
