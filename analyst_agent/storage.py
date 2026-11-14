"""One small storage layer for the glossary, pins and usage counts.

- Default: a local SQLite file (fine on a laptop).
- DATABASE_URL=postgresql://...: a hosted Postgres such as Neon's free plan, so data survives
  restarts and redeploys of a free host.

Callers write SQL with `?` placeholders and portable syntax (INSERT ... ON CONFLICT ... DO UPDATE
works in both); Store converts placeholders for Postgres.
"""
from __future__ import annotations

import os
import sqlite3
import threading
from pathlib import Path

_pg_conns: dict[str, object] = {}
_pg_lock = threading.Lock()


class Store:
    def __init__(self, sqlite_path: Path, url: str | None = None):
        url = url if url is not None else os.getenv("DATABASE_URL", "")
        self.url = url.strip()
        self.kind = "postgres" if self.url.startswith(("postgres://", "postgresql://")) else "sqlite"
        self.sqlite_path = Path(sqlite_path)

    # ------------------------------------------------------------------ public
    def run(self, sql: str, params: tuple = ()) -> tuple[list[tuple], int]:
        """Execute one statement. Returns (rows, rowcount)."""
        if self.kind == "sqlite":
            with sqlite3.connect(self.sqlite_path, timeout=10) as db:
                cur = db.execute(sql, params)
                return cur.fetchall(), cur.rowcount
        return self._pg(sql.replace("?", "%s"), params)

    def rows(self, sql: str, params: tuple = ()) -> list[tuple]:
        return self.run(sql, params)[0]

    def count(self, sql: str, params: tuple = ()) -> int:
        return self.run(sql, params)[1]

    def columns(self, table: str) -> list[str]:
        """Column names of a table ([] if it doesn't exist)."""
        if self.kind == "sqlite":
            return [r[1] for r in self.rows(f"PRAGMA table_info({table})")]
        return [r[0] for r in self.rows(
            "SELECT column_name FROM information_schema.columns WHERE table_name = ? ORDER BY ordinal_position",
            (table,))]

    # ------------------------------------------------------------------ postgres
    def _pg(self, sql: str, params: tuple):
        import psycopg
        with _pg_lock:                                   # one shared connection per URL, used serially
            for attempt in range(2):
                conn = _pg_conns.get(self.url)
                if conn is None or conn.closed:
                    conn = psycopg.connect(self.url, autocommit=True, connect_timeout=10)
                    _pg_conns[self.url] = conn
                try:
                    with conn.cursor() as cur:
                        cur.execute(sql, params)
                        rows = cur.fetchall() if cur.description else []
                        return rows, cur.rowcount
                except psycopg.OperationalError:
                    # Neon's free compute sleeps when idle and drops connections: reconnect once
                    _pg_conns.pop(self.url, None)
                    if attempt:
                        raise
        raise RuntimeError("unreachable")
