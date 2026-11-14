"""The E2B path of run_python, with a fake sandbox that mimics E2B's file and run_code API
but executes locally. The real service needs E2B_API_KEY (see test_e2b_live at the bottom)."""
import os
import subprocess
import sys
from types import SimpleNamespace

import pandas as pd
import pytest

import analyst_agent.tools.python_exec as px


class FakeSandbox:
    """Same surface as e2b_code_interpreter.Sandbox: .files.write/.read, .run_code, .kill."""

    def __init__(self, root):
        self.root, self.files, self.killed, self.written = root, self, False, []

    def _p(self, path):
        assert path.startswith(px.E2B_DIR + "/"), path          # nothing written outside the sandbox dir
        return self.root / path[len(px.E2B_DIR) + 1:]

    def write(self, path, data):
        self.written.append(path)
        if isinstance(data, str):
            data = data.replace(px.E2B_DIR, str(self.root)).encode()
        self._p(path).write_bytes(data)

    def read(self, path, format="text"):
        return self._p(path).read_bytes()

    def run_code(self, code, timeout=None):
        code = code.replace(px.E2B_DIR, str(self.root))
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
        return SimpleNamespace(logs=SimpleNamespace(stdout=[r.stdout], stderr=[r.stderr]), error=None)

    def kill(self):
        self.killed = True


@pytest.fixture
def fake(tmp_path, monkeypatch):
    box = FakeSandbox(tmp_path)
    monkeypatch.setattr(px, "_e2b_sandbox", lambda: box)
    return box


def test_e2b_runs_code_and_returns_table(fake, sample_session):
    paths = {k: d.path for k, d in sample_session.datasets.items()}
    res = px.run_python("output = df.groupby('region').order_value.median().reset_index()\nprint('hi')",
                        paths, main_table="retail_orders", sandbox="e2b")
    assert res["ok"] and res["sandbox"] == "e2b" and "hi" in res["stdout"]
    assert list(res["output_df"].columns) == ["region", "order_value"] and len(res["output_df"]) == 4
    assert fake.killed                                          # sandbox always shut down
    assert any(p.endswith("retail_orders.parquet") for p in fake.written)


def test_e2b_sees_last_sql_result(fake):
    res = px.run_python("result['x'].sum()", {}, last_result=pd.DataFrame({"x": [1, 2, 3]}), sandbox="e2b")
    assert res["last_value"] == "6"


def test_e2b_reports_code_errors(fake):
    res = px.run_python("1/0", {}, sandbox="e2b")
    assert not res["ok"] and "ZeroDivisionError" in res["error"] and fake.killed


def test_guard_still_applies_before_e2b(fake):
    res = px.run_python("import os", {}, sandbox="e2b")
    assert res["error"].startswith("Blocked:") and fake.written == []


def test_e2b_start_failure_is_friendly(monkeypatch):
    def boom():
        raise RuntimeError("401 invalid API key")
    monkeypatch.setattr(px, "_e2b_sandbox", boom)
    res = px.run_python("print(1)", {}, sandbox="e2b")
    assert not res["ok"] and "Couldn't start the E2B sandbox" in res["error"]


def test_public_mode_enables_python_with_e2b(monkeypatch):
    import importlib

    import analyst_agent.config as cfg
    for env, expected in [({"PUBLIC_MODE": "true"}, False),
                          ({"PUBLIC_MODE": "true", "E2B_API_KEY": "e2b_x"}, True),
                          ({"PUBLIC_MODE": "false"}, True)]:
        for k in ("PUBLIC_MODE", "E2B_API_KEY", "ALLOW_PYTHON", "PYTHON_SANDBOX"):
            monkeypatch.delenv(k, raising=False)
        for k, v in env.items():
            monkeypatch.setenv(k, v)
        assert importlib.reload(cfg).ALLOW_PYTHON is expected, env
    monkeypatch.delenv("PUBLIC_MODE", raising=False)
    monkeypatch.delenv("E2B_API_KEY", raising=False)
    importlib.reload(cfg)


@pytest.mark.skipif(not os.getenv("E2B_API_KEY"), reason="needs a real E2B_API_KEY")
def test_e2b_live(sample_session):
    paths = {k: d.path for k, d in sample_session.datasets.items()}
    res = px.run_python("output = df.groupby('channel').order_value.mean().reset_index()", paths,
                        main_table="retail_orders", sandbox="e2b")
    assert res["ok"], res.get("error")
    assert set(res["output_df"]["channel"]) == {"Online", "Store", "B2B"}
