"""Question limits that protect the server's free API key in a public deployment.

- per session: QUESTIONS_PER_HOUR in a sliding one-hour window (in memory)
- whole app:   DAILY_QUESTION_CAP per UTC day (stored via Store: SQLite, or Postgres with DATABASE_URL,
               so it survives restarts)
Visitors who bring their own key are not counted. 0 disables a limit.
"""
from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from datetime import UTC, datetime
from pathlib import Path

from analyst_agent.config import DAILY_QUESTION_CAP, DATA_DIR, QUESTIONS_PER_HOUR
from analyst_agent.storage import Store


class RateLimiter:
    def __init__(self, per_hour: int = QUESTIONS_PER_HOUR, daily_cap: int = DAILY_QUESTION_CAP,
                 path: Path = DATA_DIR / "usage.sqlite", url: str | None = None):
        self.per_hour, self.daily_cap = per_hour, daily_cap
        self._recent: dict[str, deque] = defaultdict(deque)
        self._lock = threading.Lock()
        self.db = Store(Path(path), url)
        self.db.run("CREATE TABLE IF NOT EXISTS daily_usage (day TEXT PRIMARY KEY, n INTEGER NOT NULL)")

    @staticmethod
    def _today() -> str:
        return datetime.now(UTC).strftime("%Y-%m-%d")

    def _trim(self, key: str, now: float) -> deque:
        q = self._recent[key]
        while q and now - q[0] > 3600:
            q.popleft()
        return q

    def remaining(self, key: str) -> int | None:
        """Questions left this hour for this session (None = unlimited)."""
        if not self.per_hour:
            return None
        with self._lock:
            return max(0, self.per_hour - len(self._trim(key, time.time())))

    def check(self, key: str) -> str | None:
        """None if the question may go ahead (and records it), else a message for the user."""
        now = time.time()
        with self._lock:
            q = self._trim(key, now)
            if self.per_hour and len(q) >= self.per_hour:
                wait = int((3600 - (now - q[0])) // 60) + 1
                noun = "question" if self.per_hour == 1 else "questions"
                return (f"You've used this demo's {self.per_hour} {noun} for the hour. Try again in about "
                        f"{wait} min, or add your own free API key in the sidebar.")
            if self.daily_cap:
                row = self.db.rows("SELECT n FROM daily_usage WHERE day = ?", (self._today(),))
                if row and row[0][0] >= self.daily_cap:
                    return ("The app has reached today's shared question limit. Add your own free API key "
                            "in the sidebar to keep going, or come back tomorrow.")
                self.db.run("INSERT INTO daily_usage VALUES (?, 1) "
                            "ON CONFLICT (day) DO UPDATE SET n = daily_usage.n + 1", (self._today(),))
            q.append(now)
            return None
