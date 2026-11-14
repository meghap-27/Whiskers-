import json
from pathlib import Path

from fastapi.testclient import TestClient

import analyst_agent.api as api
from analyst_agent.demo import DEMO_QUESTION, demo_llm

SAMPLE = Path(__file__).resolve().parent.parent / "sample_data" / "retail_orders.csv"


def test_api_flow(monkeypatch):
    monkeypatch.setattr(api, "llm", demo_llm)
    c = TestClient(api.app)
    sid = c.post("/sessions").json()["session_id"]

    with open(SAMPLE, "rb") as f:
        up = c.post(f"/sessions/{sid}/upload", files={"file": ("retail_orders.csv", f, "text/csv")}).json()
    assert up["table"] == "retail_orders" and up["profile"]["quality"]["issue_count"] >= 3

    with c.stream("POST", f"/sessions/{sid}/ask", json={"question": DEMO_QUESTION}) as r:
        events = [json.loads(line[6:]) for line in r.iter_lines() if line.startswith("data: ")]
    done = events[-1]
    assert done["type"] == "done" and done["verification"]["verified"]
    assert any(e["type"] == "step_end" and e["chart"] for e in events)

    nb = c.get(f"/sessions/{sid}/runs/{done['run_id']}/notebook")
    assert nb.status_code == 200 and json.loads(nb.content)["nbformat"] == 4

    pin = c.post("/pins", json={"session_id": sid, "run_id": done["run_id"]}).json()
    refreshed = c.get(f"/pins/{pin['id']}/refresh", params={"session_id": sid}).json()
    assert refreshed["table"]["row_count"] == 8

    assert c.post("/sessions/nope/ask", json={"question": "x"}).status_code == 404
