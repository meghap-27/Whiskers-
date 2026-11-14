"""Business glossary: metric definitions learned from user corrections (PRD FR10).

"No, revenue should exclude refunds" -> the agent calls save_definition, and every later
question on that dataset gets the definition in its system prompt.
"""
from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from analyst_agent.config import DATA_DIR
from analyst_agent.storage import Store

DB_PATH = DATA_DIR / "glossary.sqlite"


class Glossary:
    """Definitions are scoped to an owner: "local" for a single-user install, the browser session
    id in public mode, so one visitor's definitions never leak into another's answers.
    Stored in SQLite by default, or in Postgres when DATABASE_URL is set."""

    def __init__(self, path: Path = DB_PATH, owner: str = "local", url: str | None = None):
        self.path = Path(path)
        self.owner = owner
        self.db = Store(self.path, url)
        cols = self.db.columns("definitions")
        legacy = bool(cols) and "owner" not in cols          # SQLite file from before owners existed
        if legacy:
            self.db.run("ALTER TABLE definitions RENAME TO definitions_old")
        self.db.run("""CREATE TABLE IF NOT EXISTS definitions (
            owner TEXT NOT NULL, dataset TEXT NOT NULL, term TEXT NOT NULL, definition TEXT NOT NULL,
            sql_expression TEXT, created_at TEXT NOT NULL,
            PRIMARY KEY (owner, dataset, term))""")
        if legacy:
            self.db.run("INSERT INTO definitions SELECT 'local', * FROM definitions_old "
                        "WHERE true ON CONFLICT DO NOTHING")
            self.db.run("DROP TABLE definitions_old")

    def save(self, dataset: str, term: str, definition: str, sql_expression: str | None = None) -> dict:
        term = term.strip().lower()
        row = (self.owner, dataset, term, definition.strip(), (sql_expression or "").strip() or None,
               datetime.now(UTC).isoformat(timespec="seconds"))
        self.db.run("""INSERT INTO definitions VALUES (?,?,?,?,?,?)
            ON CONFLICT (owner, dataset, term) DO UPDATE SET definition = excluded.definition,
            sql_expression = excluded.sql_expression, created_at = excluded.created_at""", row)
        return {"dataset": dataset, "term": term, "definition": row[3], "sql_expression": row[4]}

    def list(self, dataset: str | None = None) -> list[dict]:
        q = "SELECT dataset, term, definition, sql_expression, created_at FROM definitions WHERE owner = ?"
        args: tuple = (self.owner,)
        if dataset:
            q, args = q + " AND dataset = ?", (self.owner, dataset)
        keys = ("dataset", "term", "definition", "sql_expression", "created_at")
        return [dict(zip(keys, r, strict=True)) for r in self.db.rows(q + " ORDER BY term", args)]

    def delete(self, dataset: str, term: str) -> bool:
        return self.db.count("DELETE FROM definitions WHERE owner = ? AND dataset = ? AND term = ?",
                             (self.owner, dataset, term.strip().lower())) > 0

    def forget_owner(self) -> int:
        """Remove everything this owner saved (used when a public session expires)."""
        return self.db.count("DELETE FROM definitions WHERE owner = ?", (self.owner,))

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
