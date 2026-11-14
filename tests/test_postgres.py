"""Glossary, pins and usage limits on a real Postgres (e.g. Neon).

Skipped unless TEST_DATABASE_URL is set, for example:
    TEST_DATABASE_URL=postgresql://postgres@localhost:5432/postgres pytest tests/test_postgres.py
"""
import os
import uuid

import pytest

URL = os.getenv("TEST_DATABASE_URL", "")
pytestmark = pytest.mark.skipif(not URL, reason="TEST_DATABASE_URL not set")


@pytest.fixture(scope="module", autouse=True)
def clean_tables():
    from analyst_agent.storage import Store
    s = Store("unused.sqlite", URL)
    for t in ("definitions", "pins", "daily_usage"):
        s.run(f"DROP TABLE IF EXISTS {t}")
    yield


def test_store_is_postgres():
    from analyst_agent.storage import Store
    s = Store("unused.sqlite", URL)
    assert s.kind == "postgres" and s.rows("SELECT 1 + 1")[0][0] == 2


def test_glossary_on_postgres(tmp_path):
    from analyst_agent.glossary import Glossary
    a = Glossary(tmp_path / "x.sqlite", owner=f"a-{uuid.uuid4().hex[:6]}", url=URL)
    b = Glossary(tmp_path / "x.sqlite", owner=f"b-{uuid.uuid4().hex[:6]}", url=URL)
    a.save("retail_orders", "Revenue", "net of refunds", "SUM(order_value - refund_amount)")
    a.save("retail_orders", "revenue", "updated definition")           # upsert, not a duplicate
    assert [d["definition"] for d in a.list("retail_orders")] == ["updated definition"]
    assert b.list() == []
    assert "updated definition" in a.prompt_text(["retail_orders"])
    assert a.delete("retail_orders", "REVENUE") and a.list() == []
    assert not (tmp_path / "x.sqlite").exists()                          # nothing written locally


def test_pins_on_postgres(tmp_path, session):
    from analyst_agent.agent.loop import Agent
    from analyst_agent.demo import DEMO_QUESTION, demo_llm
    from analyst_agent.glossary import Glossary
    from analyst_agent.pins import PinBoard
    run = Agent(session, demo_llm(), Glossary(tmp_path / "g.sqlite")).ask(DEMO_QUESTION)
    owner = f"p-{uuid.uuid4().hex[:6]}"
    board = PinBoard(tmp_path / "p.sqlite", owner=owner, url=URL)
    pin = board.pin(run)
    again = PinBoard(tmp_path / "other.sqlite", owner=owner, url=URL)   # new process / redeploy
    assert [p["id"] for p in again.list()] == [pin["id"]]
    assert len(again.refresh(pin, session.con)["table"]) == 8
    assert again.remove(pin["id"]) and again.list() == []


def test_daily_cap_on_postgres(tmp_path):
    from analyst_agent.limits import RateLimiter
    rl = RateLimiter(per_hour=0, daily_cap=2, path=tmp_path / "u.sqlite", url=URL)
    assert rl.check("x") is None and rl.check("y") is None
    assert "today's shared question limit" in rl.check("z")
    # a restarted app sees the same count
    assert "today's shared question limit" in RateLimiter(0, 2, tmp_path / "v.sqlite", URL).check("w")


def test_reconnects_after_dropped_connection():
    from analyst_agent import storage
    s = storage.Store("unused.sqlite", URL)
    s.rows("SELECT 1")
    storage._pg_conns[URL].close()                                      # like Neon suspending compute
    assert s.rows("SELECT 2")[0][0] == 2
