"""run_python tool: execute LLM-written analysis code away from the app (PRD FR5).

Every call is first checked by code_guard (allow-listed imports, no file/network/interpreter
access). Then it runs in one of two places, chosen by PYTHON_SANDBOX:

- "local": a fresh `python -I` subprocess with CPU, memory and time limits, an empty environment
  and a scratch directory. Good for a laptop; not a real sandbox, so public mode turns Python off.
- "e2b": an E2B cloud sandbox (a Firecracker microVM per call). The same runner script runs
  there, so behaviour is identical, but the code never touches this server, its files or its keys.
  Needs E2B_API_KEY; public mode then allows Python again.
"""
from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

from analyst_agent.config import PY_MEMORY_MB, PY_TIMEOUT_S, PYTHON_SANDBOX
from analyst_agent.tools.code_guard import UnsafeCodeError, check_code

RUNNER = Path(__file__).with_name("_runner.py")
MARK = "__AGENT_RESULT__"


def run_python(code: str, datasets: dict[str, Path], last_result: pd.DataFrame | None = None,
               main_table: str | None = None, sandbox: str | None = None) -> dict:
    """Run `code` with each dataset loaded as a DataFrame named after its table.

    `df` is the main table and `result` the previous SQL result. If the code leaves a
    DataFrame in a variable called `output` (or as its last expression), it is returned
    as a table so it can be charted.
    """
    try:
        check_code(code)
    except UnsafeCodeError as e:
        return {"ok": False, "error": f"Blocked: {e}"}
    if (sandbox or PYTHON_SANDBOX) == "e2b":
        return _run_in_e2b(code, datasets, last_result, main_table)
    with tempfile.TemporaryDirectory(prefix="agent_py_") as tmp:
        tmp = Path(tmp)
        result_path = None
        if last_result is not None:
            result_path = tmp / "result.parquet"
            last_result.to_parquet(result_path, index=False)
        spec = {
            "code": code,
            "datasets": {k: str(v) for k, v in datasets.items()},
            "main": main_table or (next(iter(datasets)) if datasets else None),
            "result": str(result_path) if result_path else None,
            "output_path": str(tmp / "output.parquet"),
            "memory_mb": PY_MEMORY_MB,
            "cpu_s": PY_TIMEOUT_S,
        }
        try:
            proc = subprocess.run(
                [sys.executable, "-I", str(RUNNER)],
                input=json.dumps(spec), capture_output=True, text=True,
                timeout=PY_TIMEOUT_S + 5, cwd=tmp, env=_child_env(),
            )
        except subprocess.TimeoutExpired:
            return {"ok": False, "error": f"Code exceeded {PY_TIMEOUT_S}s and was stopped."}

        payload = _parse(proc.stdout)
        if payload is None:
            err = (proc.stderr or "").strip()[-1500:] or f"Process exited with code {proc.returncode}"
            return {"ok": False, "error": err}
        out_df = None
        if payload.get("has_output") and Path(spec["output_path"]).exists():
            out_df = pd.read_parquet(spec["output_path"])
        payload["output_df"] = out_df
        return payload


def _child_env() -> dict:
    """A near-empty environment: no API keys or secrets leak into analysis code."""
    env = {"PATH": os.defpath, "MPLBACKEND": "Agg"}
    if os.name == "nt":  # Windows needs these for Python to start
        for k in ("SYSTEMROOT", "TEMP", "TMP", "PATH"):
            if k in os.environ:
                env[k] = os.environ[k]
    return env


def _parse(stdout: str) -> dict | None:
    if MARK not in stdout:
        return None
    before, _, after = stdout.rpartition(MARK)
    try:
        data = json.loads(after.strip())
    except json.JSONDecodeError:
        return None
    data["stdout"] = (data.get("stdout") or "") + before
    return data


# ---------------------------------------------------------------------------- E2B sandbox
E2B_DIR = "/home/user/agent"          # working folder inside the sandbox VM

# Runs inside the sandbox's Jupyter kernel: make sure pyarrow exists (for parquet), then run the
# same _runner.py used locally and print its output for us to parse.
_E2B_DRIVER = """
import subprocess, sys
try:
    import pyarrow  # noqa: F401
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pyarrow"], check=False)
r = subprocess.run([sys.executable, "{dir}/_runner.py"], input=open("{dir}/spec.json").read(),
                   capture_output=True, text=True, timeout={timeout})
print(r.stdout)
if r.returncode:
    print(r.stderr[-1500:], file=sys.stderr)
"""


def _e2b_sandbox():
    """Create a sandbox. Replaced in tests by a fake that runs locally."""
    from e2b_code_interpreter import Sandbox
    return Sandbox.create(timeout=PY_TIMEOUT_S + 90)


def _run_in_e2b(code: str, datasets: dict[str, Path], last_result: pd.DataFrame | None,
                main_table: str | None) -> dict:
    try:
        sbx = _e2b_sandbox()
    except ImportError:
        return {"ok": False, "error": "PYTHON_SANDBOX=e2b needs `pip install e2b-code-interpreter`."}
    except Exception as e:  # bad key, out of credit, E2B outage
        return {"ok": False, "error": f"Couldn't start the E2B sandbox ({type(e).__name__}: {str(e)[:200]})."}
    try:
        remote = {}
        for name, path in datasets.items():
            remote[name] = f"{E2B_DIR}/{name}.parquet"
            sbx.files.write(remote[name], Path(path).read_bytes())
        result_path = None
        if last_result is not None:
            buf = io.BytesIO()
            last_result.to_parquet(buf, index=False)
            result_path = f"{E2B_DIR}/result.parquet"
            sbx.files.write(result_path, buf.getvalue())
        spec = {"code": code, "datasets": remote,
                "main": main_table or (next(iter(datasets)) if datasets else None),
                "result": result_path, "output_path": f"{E2B_DIR}/output.parquet",
                "memory_mb": PY_MEMORY_MB, "cpu_s": PY_TIMEOUT_S}
        sbx.files.write(f"{E2B_DIR}/spec.json", json.dumps(spec))
        sbx.files.write(f"{E2B_DIR}/_runner.py", RUNNER.read_text())

        execution = sbx.run_code(_E2B_DRIVER.format(dir=E2B_DIR, timeout=PY_TIMEOUT_S + 5),
                                 timeout=PY_TIMEOUT_S + 30)
        stdout = "".join(execution.logs.stdout)
        payload = _parse(stdout)
        if payload is None:
            err = execution.error
            detail = f"{err.name}: {err.value}" if err else "".join(execution.logs.stderr)[-1500:]
            return {"ok": False, "error": detail or "The sandbox returned no result."}
        payload["output_df"] = None
        if payload.get("has_output"):
            data = sbx.files.read(spec["output_path"], format="bytes")
            payload["output_df"] = pd.read_parquet(io.BytesIO(bytes(data)))
        payload["sandbox"] = "e2b"
        return payload
    finally:
        try:
            sbx.kill()
        except Exception:
            pass
