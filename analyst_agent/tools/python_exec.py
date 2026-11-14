"""run_python tool: execute LLM-written analysis code in a separate process (PRD FR5).

v1 isolation: a fresh `python -I` subprocess per call with CPU, memory and wall-clock limits,
an empty environment and a scratch working directory. This keeps bad code out of the API
process. It does NOT block network access: before deploying publicly, run the same runner
inside the Docker image in sandbox/ (network disabled) or a hosted sandbox such as E2B.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

from analyst_agent.config import PY_MEMORY_MB, PY_TIMEOUT_S

RUNNER = Path(__file__).with_name("_runner.py")
MARK = "__AGENT_RESULT__"


def run_python(code: str, datasets: dict[str, Path], last_result: pd.DataFrame | None = None,
               main_table: str | None = None) -> dict:
    """Run `code` with each dataset loaded as a DataFrame named after its table.

    `df` is the main table and `result` the previous SQL result. If the code leaves a
    DataFrame in a variable called `output` (or as its last expression), it is returned
    as a table so it can be charted.
    """
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
