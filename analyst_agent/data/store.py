"""Session state: one in-memory DuckDB per session plus the datasets loaded into it."""
from __future__ import annotations

import re
import shutil
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import duckdb
import pandas as pd

from analyst_agent.config import DATA_DIR, DUCKDB_MEMORY, MAX_ROWS, MAX_UPLOAD_MB, SESSION_TTL_MIN
from analyst_agent.data.profile import profile_dataframe


class UploadTooLarge(ValueError):
    pass


def _table_name(name: str) -> str:
    stem = Path(name).stem.lower()
    stem = re.sub(r"[^a-z0-9_]+", "_", stem).strip("_") or "data"
    if stem[0].isdigit():
        stem = "t_" + stem
    return stem


@dataclass
class Dataset:
    name: str
    table: str
    path: Path            # parquet copy used by the Python sandbox and notebook export
    source_path: Path     # original file
    profile: dict


@dataclass
class Session:
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    datasets: dict[str, Dataset] = field(default_factory=dict)
    history: list[dict] = field(default_factory=list)   # LLM message history
    runs: list = field(default_factory=list)

    def __post_init__(self):
        self.dir = DATA_DIR / "sessions" / self.id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.con = duckdb.connect(":memory:")
        self.con.execute(f"SET memory_limit = '{DUCKDB_MEMORY}'")
        self.con.execute("SET threads = 2")
        # defence in depth: generated SQL can never touch the file system or re-enable it
        self.con.execute("SET enable_external_access = false")
        self.con.execute("SET lock_configuration = true")

    # ---- loading -------------------------------------------------------
    def load_file(self, path: str | Path, name: str | None = None) -> Dataset:
        path = Path(path)
        name = name or path.name
        size_mb = path.stat().st_size / 1e6
        if size_mb > MAX_UPLOAD_MB:
            raise UploadTooLarge(f"{name} is {size_mb:.1f} MB; the limit here is {MAX_UPLOAD_MB:g} MB.")
        if path.suffix.lower() in (".xlsx", ".xls"):
            df = pd.read_excel(path)
        else:
            # read everything as text first so the profiler can spot mixed formats
            raw = pd.read_csv(path, dtype=str, keep_default_na=True)
            df = _infer_types(raw)
            return self.load_dataframe(df, name=name, source_path=path, raw=raw)
        return self.load_dataframe(df, name=name, source_path=path)

    def load_dataframe(self, df: pd.DataFrame, name: str, source_path: Path | None = None,
                       raw: pd.DataFrame | None = None) -> Dataset:
        if len(df) > MAX_ROWS:
            raise UploadTooLarge(f"{name} has {len(df):,} rows; the limit here is {MAX_ROWS:,}. "
                                 "Upload a sample or a filtered extract.")
        table = _table_name(name)
        base, i = table, 2
        while table in self.datasets:
            table, i = f"{base}_{i}", i + 1
        pq = self.dir / f"{table}.parquet"
        df.to_parquet(pq, index=False)
        self.con.register(f"_{table}_df", df)
        self.con.execute(f'CREATE OR REPLACE TABLE "{table}" AS SELECT * FROM "_{table}_df"')
        self.con.unregister(f"_{table}_df")
        ds = Dataset(name=name, table=table, path=pq,
                     source_path=Path(source_path) if source_path else pq,
                     profile=profile_dataframe(df, raw=raw))
        self.datasets[table] = ds
        return ds

    @property
    def main_table(self) -> str | None:
        """The most recently loaded table: questions are about it unless the user names another."""
        return next(reversed(self.datasets), None)

    def remove(self, table: str) -> bool:
        ds = self.datasets.pop(table, None)
        if ds is None:
            return False
        self.con.execute(f'DROP TABLE IF EXISTS "{table}"')
        ds.path.unlink(missing_ok=True)
        return True

    def schema_text(self) -> str:
        """Compact schema + profile for the system prompt. The main table is listed first."""
        parts = []
        order = sorted(self.datasets.values(), key=lambda d: d.table != self.main_table)
        if len(order) > 1:
            parts.append(f'Main table: "{self.main_table}" (the latest upload). Answer about it unless the '
                         "user names another table.")
        for ds in order:
            p = ds.profile
            parts.append(f'Table "{ds.table}" (from {ds.name}): {p["row_count"]} rows')
            for c in p["columns"]:
                line = f'  - {c["name"]} [{c["dtype"]}] nulls={c["null_pct"]}%'
                if c.get("min") is not None:
                    line += f' range={c["min"]}..{c["max"]}'
                if c.get("top_values"):
                    line += " top=" + ", ".join(map(str, list(c["top_values"])[:5]))
                parts.append(line)
            issues = p.get("quality", {}).get("issues", [])
            if issues:
                parts.append("  Data-quality issues: " + "; ".join(i["message"] for i in issues))
        return "\n".join(parts) or "(no datasets loaded)"


def _infer_types(raw: pd.DataFrame) -> pd.DataFrame:
    """Convert text columns to numbers/dates when (almost) every value parses."""
    df = raw.copy()
    for col in df.columns:
        s = raw[col].dropna().str.strip()
        if s.empty:
            continue
        num = pd.to_numeric(s.str.replace(",", "", regex=False), errors="coerce")
        if num.notna().mean() >= 0.98:
            df[col] = pd.to_numeric(raw[col].str.replace(",", "", regex=False).str.strip(), errors="coerce")
            continue
        if _looks_like_date(s):
            parsed = parse_dates(raw[col])
            if parsed.notna().sum() >= 0.9 * len(s):
                df[col] = parsed
    return df


def parse_dates(col: pd.Series) -> pd.Series:
    """Parse a text column that may mix ISO dates with DD-MM-YYYY or MM/DD/YYYY.

    ISO (2024-06-01) is unambiguous. For the rest, the values decide: a first number above 12
    means day-first, a second number above 12 means month-first. If no value settles it,
    '-' and '.' separators are read day-first (Indian/European style) and '/' month-first.
    """
    s = col.astype("string").str.strip()
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    iso = s.str.match(r"^\d{4}[-/.]\d{1,2}[-/.]\d{1,2}", na=False)
    out[iso] = pd.to_datetime(s[iso], errors="coerce", format="mixed", yearfirst=True)
    rest = s[~iso & s.notna()]
    if not rest.empty:
        parts = rest.str.extract(r"^(\d{1,2})([-/.])(\d{1,2})")
        first, second = pd.to_numeric(parts[0], errors="coerce"), pd.to_numeric(parts[2], errors="coerce")
        if (first > 12).any() and not (second > 12).any():
            dayfirst = True
        elif (second > 12).any():
            dayfirst = False
        else:
            dayfirst = parts[1].mode().iat[0] in "-." if not parts[1].dropna().empty else False
        out[rest.index] = pd.to_datetime(rest, errors="coerce", format="mixed", dayfirst=dayfirst)
    return out


def _looks_like_date(s: pd.Series) -> bool:
    sample = s.head(200)
    pattern = r"^\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}"
    return sample.str.match(pattern).mean() >= 0.9


def cleanup_old_sessions(ttl_min: int = SESSION_TTL_MIN) -> int:
    """Delete session folders (uploaded files) untouched for ttl_min minutes; drop them from memory."""
    root, removed = DATA_DIR / "sessions", 0
    if not root.exists():
        return 0
    cutoff = time.time() - ttl_min * 60
    for d in root.iterdir():
        try:
            if d.is_dir() and d.stat().st_mtime < cutoff:
                shutil.rmtree(d, ignore_errors=True)
                s = SESSIONS.pop(d.name, None)
                if s is not None:
                    s.con.close()
                removed += 1
        except OSError:
            pass
    return removed


# simple process-wide registry
SESSIONS: dict[str, Session] = {}


def get_or_create(session_id: str | None = None) -> Session:
    if session_id and session_id in SESSIONS:
        return SESSIONS[session_id]
    s = Session()
    SESSIONS[s.id] = s
    return s
