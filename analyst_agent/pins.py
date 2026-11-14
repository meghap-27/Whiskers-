"""Pin an answer to a dashboard; pinned answers re-run their saved SQL/chart steps on refresh (PRD FR11)."""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from analyst_agent.config import DATA_DIR
from analyst_agent.tools.chart import make_chart
from analyst_agent.tools.sql import run_sql

PINS_PATH = DATA_DIR / "pins.json"


class PinBoard:
    def __init__(self, path: Path = PINS_PATH):
        self.path = Path(path)

    def _load(self) -> list[dict]:
        return json.loads(self.path.read_text()) if self.path.exists() else []

    def _save(self, pins: list[dict]):
        self.path.write_text(json.dumps(pins, indent=2))

    def pin(self, run, title: str | None = None) -> dict:
        """Store the replayable parts of a run: its last successful SQL and its chart settings."""
        sql = next((s.input["query"] for s in reversed(run.steps) if s.tool == "run_sql" and not s.error), None)
        if sql is None:
            raise ValueError("Only answers backed by a SQL query can be pinned (it is re-run on refresh).")
        chart = next((s.input for s in reversed(run.steps) if s.tool == "make_chart" and not s.error), None)
        pin = {"id": uuid.uuid4().hex[:8], "title": title or run.question, "question": run.question,
               "sql": sql, "chart": chart, "answer": run.answer,
               "pinned_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        pins = self._load()
        pins.append(pin)
        self._save(pins)
        return pin

    def list(self) -> list[dict]:
        return self._load()

    def remove(self, pin_id: str) -> bool:
        pins = self._load()
        kept = [p for p in pins if p["id"] != pin_id]
        self._save(kept)
        return len(kept) < len(pins)

    def refresh(self, pin: dict, con) -> dict:
        """Re-run the pinned query on current data. Returns {"table", "chart"}."""
        df = run_sql(con, pin["sql"])
        chart = None
        if pin.get("chart"):
            c = pin["chart"]
            chart = make_chart(df, c["chart_type"], c["x"], c.get("y"), c.get("color"), c.get("title", ""))
        return {"table": df, "chart": chart}
