"""Session state: one in-memory DuckDB per session plus the datasets loaded into it."""
from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import duckdb
import pandas as pd

from analyst_agent.config import DATA_DIR
from analyst_agent.data.profile import profile_dataframe


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
        # defence in depth: generated SQL can never touch the file system or re-enable it
        self.con.execute("SET enable_external_access = false")
        self.con.execute("SET lock_configuration = true")

    # ---- loading -------------------------------------------------------
    def load_file(self, path: str | Path, name: str | None = None) -> Dataset:
        path = Path(path)
        name = name or path.name
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

    def schema_text(self) -> str:
        """Compact schema + profile for the system prompt."""
        parts = []
        for ds in self.datasets.values():
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
            parsed = pd.to_datetime(raw[col].str.strip(), errors="coerce", format="mixed", dayfirst=False)
            if parsed.notna().sum() >= 0.9 * len(s):
                df[col] = parsed
    return df


def _looks_like_date(s: pd.Series) -> bool:
    sample = s.head(200)
    pattern = r"^\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}"
    return sample.str.match(pattern).mean() >= 0.9


# simple process-wide registry (fine for a single-user demo)
SESSIONS: dict[str, Session] = {}


def get_or_create(session_id: str | None = None) -> Session:
    if session_id and session_id in SESSIONS:
        return SESSIONS[session_id]
    s = Session()
    SESSIONS[s.id] = s
    return s
