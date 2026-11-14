"""Production hardening: code guard, public mode, limits, per-owner data, fallback, cleanup."""
import json
import os
import time

import httpx
import pytest

from analyst_agent.agent.llm import FallbackLLM, OpenAICompatLLM, ProviderError, ScriptedLLM, final, tool_call
from analyst_agent.agent.loop import Agent
from analyst_agent.glossary import Glossary
from analyst_agent.limits import RateLimiter
from analyst_agent.pins import PinBoard
from analyst_agent.tools.code_guard import UnsafeCodeError, check_code
from analyst_agent.tools.python_exec import run_python


@pytest.mark.parametrize("code", [
    "import os\nos.listdir('.')", "open('/etc/passwd').read()", "import subprocess",
    "pd.read_csv('http://x')", "df.to_csv('out.csv')", "().__class__.__base__.__subclasses__()",
    "__import__('socket')", "getattr(pd, 'read_csv')('x')",
])
def test_code_guard_blocks(code):
    with pytest.raises(UnsafeCodeError):
        check_code(code)


def test_code_guard_allows_analysis():
    check_code("import numpy as np\nfrom scipy import stats\nstats.pearsonr(df.a, df.b)\ndf.groupby('x').y.mean()")


def test_blocked_code_never_runs():
    res = run_python("import os\nprint(os.getcwd())", {})
    assert not res["ok"] and res["error"].startswith("Blocked:")


def test_python_disabled_in_public_mode(session, glossary):
    llm = ScriptedLLM([tool_call("run_python", {"code": "print(1)"}), final("Used SQL instead: 3025 rows.")])
    run = Agent(session, llm, glossary, allow_python=False).ask("rows?")
    assert "disabled" in run.steps[0].error
    assert all(t["name"] != "run_python" for t in Agent(session, llm, glossary, allow_python=False).tools)
    assert "run_python is disabled" in llm.calls[0]["system"]


def test_run_timeout(session, glossary):
    class Slow(ScriptedLLM):
        def create(self, *a):
            time.sleep(0.6)
            return super().create(*a)
    llm = Slow([tool_call("run_sql", {"query": "SELECT 1 AS x"}, f"c{i}") for i in range(5)])
    run = Agent(session, llm, glossary, timeout_s=1).ask("slow")
    assert run.status == "timeout" and "stopped after" in run.answer


def test_glossary_and_pins_are_per_owner(tmp_path, session):
    a, b = Glossary(tmp_path / "g.sqlite", owner="alice"), Glossary(tmp_path / "g.sqlite", owner="bob")
    a.save("retail_orders", "revenue", "net of refunds")
    assert a.list() and not b.list()
    from analyst_agent.demo import DEMO_QUESTION, demo_llm
    run = Agent(session, demo_llm(), a).ask(DEMO_QUESTION)
    pa, pb = PinBoard(tmp_path / "p.sqlite", owner="alice"), PinBoard(tmp_path / "p.sqlite", owner="bob")
    pa.pin(run)
    assert len(pa.list()) == 1 and pb.list() == []
    assert not pb.remove(pa.list()[0]["id"])            # can't delete someone else's pin


def test_old_glossary_database_is_migrated(tmp_path):
    import sqlite3
    db = sqlite3.connect(tmp_path / "old.sqlite")
    db.execute("CREATE TABLE definitions (dataset TEXT, term TEXT, definition TEXT, sql_expression TEXT, "
               "created_at TEXT, PRIMARY KEY (dataset, term))")
    db.execute("INSERT INTO definitions VALUES ('t', 'revenue', 'x', NULL, '2026-01-01')")
    db.commit(); db.close()
    assert Glossary(tmp_path / "old.sqlite").list()[0]["term"] == "revenue"


def test_rate_limits(tmp_path):
    rl = RateLimiter(per_hour=2, daily_cap=3, path=tmp_path / "u.sqlite")
    assert rl.check("s1") is None and rl.check("s1") is None
    assert "questions for the hour" in rl.check("s1")
    assert rl.remaining("s1") == 0 and rl.remaining("s2") == 2
    assert rl.check("s2") is None
    assert "today's shared question limit" in rl.check("s3")
    assert RateLimiter(per_hour=0, daily_cap=0, path=tmp_path / "v.sqlite").check("x") is None


def test_fallback_moves_to_next_provider_on_rate_limit():
    def busy(request):
        return httpx.Response(429, headers={"retry-after": "0"}, json={"error": "busy"})

    def ok(request):
        return httpx.Response(200, json={"choices": [{"message": {"content": "from gemini"}}]})
    groq = OpenAICompatLLM("groq", api_key="gsk_test_0123456789abcdefghij", client=httpx.Client(transport=httpx.MockTransport(busy)))
    gem = OpenAICompatLLM("gemini", api_key="gsk_test_0123456789abcdefghij", client=httpx.Client(transport=httpx.MockTransport(ok)))
    llm = FallbackLLM([groq, gem])
    import analyst_agent.agent.llm as L
    real_sleep, L.time.sleep = L.time.sleep, lambda s: None
    try:
        out = llm.create("s", [{"role": "user", "content": "hi"}], [])
    finally:
        L.time.sleep = real_sleep
    assert out["content"][0]["text"] == "from gemini" and llm.provider == "gemini"


def test_fallback_does_not_hide_bad_key():
    def denied(request):
        return httpx.Response(401, json={"error": "bad key"})
    groq = OpenAICompatLLM("groq", api_key="gsk_test_0123456789abcdefghij", client=httpx.Client(transport=httpx.MockTransport(denied)))
    with pytest.raises(ProviderError):
        FallbackLLM([groq, groq]).create("s", [{"role": "user", "content": "hi"}], [])


def test_upload_limits(tmp_path, monkeypatch):
    import analyst_agent.data.store as store
    monkeypatch.setattr(store, "MAX_ROWS", 10)
    f = tmp_path / "big.csv"
    f.write_text("a\n" + "\n".join(str(i) for i in range(20)))
    with pytest.raises(store.UploadTooLarge):
        store.Session().load_file(f)
    monkeypatch.setattr(store, "MAX_UPLOAD_MB", 0.00001)
    with pytest.raises(store.UploadTooLarge):
        store.Session().load_file(f)


def test_old_sessions_are_cleaned_up():
    from analyst_agent.data.store import cleanup_old_sessions, get_or_create
    s = get_or_create()
    old = time.time() - 3 * 3600
    os.utime(s.dir, (old, old))
    assert cleanup_old_sessions(ttl_min=120) >= 1
    assert not s.dir.exists()


def test_run_log_has_no_user_data(session, glossary):
    from analyst_agent.demo import DEMO_QUESTION, demo_llm
    from analyst_agent.telemetry import log_run
    run = Agent(session, demo_llm(), glossary).ask(DEMO_QUESTION)
    rec = log_run(run, session.id, "demo", "")
    text = json.dumps(rec)
    assert rec["status"] == "done" and rec["steps"][0] == "run_sql" and "simpsons_paradox" in rec["warnings"]
    assert "Online" not in text and "SELECT" not in text and session.id not in text
