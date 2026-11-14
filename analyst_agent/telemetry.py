"""Structured run logs and optional error tracking.

One JSON line per question goes to the `analyst_agent.runs` logger (stdout by default, which
hosting platforms collect). It records how the run went, never the user's data: no question
text, SQL, answers or values. Set LOG_RUNS_FILE to also append to a file, and SENTRY_DSN (with
`pip install sentry-sdk`) to send crashes to Sentry's free tier.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
from datetime import UTC, datetime

log = logging.getLogger("analyst_agent.runs")
if not log.handlers:
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter("%(message)s"))
    log.addHandler(h)
    log.setLevel(logging.INFO)
    log.propagate = False
    if os.getenv("LOG_RUNS_FILE"):
        log.addHandler(logging.FileHandler(os.environ["LOG_RUNS_FILE"]))

_sentry_ready = False


def init_error_tracking() -> bool:
    """Start Sentry if SENTRY_DSN is set and sentry-sdk is installed. Safe to call repeatedly."""
    global _sentry_ready
    dsn = os.getenv("SENTRY_DSN")
    if _sentry_ready or not dsn:
        return _sentry_ready
    try:
        import sentry_sdk
    except ImportError:
        return False
    sentry_sdk.init(dsn=dsn, traces_sample_rate=0.0, send_default_pii=False)
    _sentry_ready = True
    return True


def capture(e: Exception) -> None:
    if _sentry_ready:
        import sentry_sdk
        sentry_sdk.capture_exception(e)


def anon(key: str) -> str:
    """Short one-way hash so logs can group a session's runs without storing its id."""
    return hashlib.sha256(key.encode()).hexdigest()[:10]


def log_run(run, session_key: str, provider: str = "", model: str = "", error: str | None = None) -> dict:
    record = {
        "ts": datetime.now(UTC).isoformat(timespec="seconds"),
        "event": "run",
        "session": anon(session_key),
        "provider": provider,
        "model": model,
        "status": "error" if error else getattr(run, "status", "unknown"),
        "error": error,
        "steps": [s.tool for s in run.steps] if run else [],
        "step_errors": sum(1 for s in run.steps if s.error) if run else 0,
        "verified": run.verification.get("verified") if run else None,
        "warnings": sorted({w["type"] for w in run.warnings}) if run else [],
        "latency_ms": run.latency_ms if run else 0,
        "tokens_in": run.tokens_in if run else 0,
        "tokens_out": run.tokens_out if run else 0,
    }
    log.info(json.dumps(record))
    return record
