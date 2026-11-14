"""Pin an answer to a dashboard; pinned answers re-run their saved SQL/chart steps on refresh (PRD FR11)."""
from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path

from analyst_agent.config import DATA_DIR
from analyst_agent.storage import Store
from analyst_agent.tools.chart import make_chart
from analyst_agent.tools.sql import run_sql

PINS_PATH = DATA_DIR / "pins.sqlite"


class PinBoard:
    """Pinned answers, scoped per owner like the glossary. SQLite by default, Postgres with DATABASE_URL."""

    def __init__(self, path: Path = PINS_PATH, owner: str = "local", url: str | None = None):
        self.path = Path(path)
        self.owner = owner
        self.db = Store(self.path, url)
        self.db.run("""CREATE TABLE IF NOT EXISTS pins (
            id TEXT PRIMARY KEY, owner TEXT NOT NULL, pinned_at TEXT NOT NULL, data TEXT NOT NULL)""")

    def pin(self, run, title: str | None = None) -> dict:
        """Store the replayable parts of a run: its last successful SQL and its chart settings."""
        sql = next((s.input["query"] for s in reversed(run.steps) if s.tool == "run_sql" and not s.error), None)
        if sql is None:
            raise ValueError("Only answers backed by a SQL query can be pinned (it is re-run on refresh).")
        chart = next((s.input for s in reversed(run.steps) if s.tool == "make_chart" and not s.error), None)
        pin = {"id": uuid.uuid4().hex[:8], "title": title or run.question, "question": run.question,
               "sql": sql, "chart": chart, "answer": run.answer,
               "pinned_at": datetime.now(UTC).isoformat(timespec="seconds")}
        self.db.run("INSERT INTO pins VALUES (?,?,?,?)", (pin["id"], self.owner, pin["pinned_at"], json.dumps(pin)))
        return pin

    def list(self) -> list[dict]:
        rows = self.db.rows("SELECT data FROM pins WHERE owner = ? ORDER BY pinned_at", (self.owner,))
        return [json.loads(r[0]) for r in rows]

    def remove(self, pin_id: str) -> bool:
        return self.db.count("DELETE FROM pins WHERE id = ? AND owner = ?", (pin_id, self.owner)) > 0

    def forget_owner(self) -> int:
        return self.db.count("DELETE FROM pins WHERE owner = ?", (self.owner,))

    def refresh(self, pin: dict, con) -> dict:
        """Re-run the pinned query on current data. Returns {"table", "chart"}."""
        df = run_sql(con, pin["sql"])
        chart = None
        if pin.get("chart"):
            c = pin["chart"]
            chart = make_chart(df, c["chart_type"], c["x"], c.get("y"), c.get("color"), c.get("title", ""))
        return {"table": df, "chart": chart}
