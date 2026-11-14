"""Dataset profiling and the data-quality report shown on upload (PRD FR1, FR9)."""
from __future__ import annotations

import difflib
import re

import pandas as pd


def _jsonable(v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    if isinstance(v, pd.Timestamp):
        return v.isoformat()[:19]
    if hasattr(v, "item"):
        return v.item()
    return v


def profile_dataframe(df: pd.DataFrame, raw: pd.DataFrame | None = None) -> dict:
    cols = []
    for c in df.columns:
        s = df[c]
        info = {
            "name": str(c),
            "dtype": _kind(s),
            "null_pct": round(float(s.isna().mean() * 100), 1),
            "unique": int(s.nunique(dropna=True)),
        }
        if info["dtype"] in ("number", "date") and s.notna().any():
            info["min"], info["max"] = _jsonable(s.min()), _jsonable(s.max())
        if info["dtype"] in ("number",) and s.notna().any():
            info["mean"] = round(float(s.mean()), 4)
            info["median"] = round(float(s.median()), 4)
        if info["dtype"] == "text":
            info["top_values"] = {str(k): int(v) for k, v in s.value_counts().head(8).items()}
        cols.append(info)
    return {
        "row_count": int(len(df)),
        "columns": cols,
        "sample": [{k: _jsonable(v) for k, v in r.items()} for r in df.head(5).to_dict("records")],
        "quality": quality_report(df, raw),
    }


def _kind(s: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(s):
        return "bool"
    if pd.api.types.is_numeric_dtype(s):
        return "number"
    if pd.api.types.is_datetime64_any_dtype(s):
        return "date"
    return "text"


# --------------------------------------------------------------------------
# Data-quality checks
# --------------------------------------------------------------------------
def quality_report(df: pd.DataFrame, raw: pd.DataFrame | None = None) -> dict:
    issues: list[dict] = []

    dups = int(df.duplicated().sum())
    if dups:
        issues.append({"type": "duplicate_rows", "count": dups,
                       "message": f"{dups} fully duplicated rows ({dups / len(df):.1%})"})

    for c in df.columns:
        s = df[c]
        null_pct = s.isna().mean()
        if null_pct >= 0.3:
            issues.append({"type": "high_nulls", "column": c,
                           "message": f"'{c}' is {null_pct:.0%} empty"})
        if s.nunique(dropna=True) == 1 and len(s) > 1:
            issues.append({"type": "constant_column", "column": c,
                           "message": f"'{c}' has a single value for every row"})

    text_source = raw if raw is not None else df
    for c in text_source.columns:
        s = text_source[c]
        if not (s.dtype == object or pd.api.types.is_string_dtype(s)):
            continue
        fmts = date_formats(s)
        if len(fmts) > 1:
            shown = ", ".join(f"{k} ({v})" for k, v in fmts.items())
            issues.append({"type": "mixed_date_formats", "column": c, "formats": fmts,
                           "message": f"'{c}' mixes date formats: {shown}"})
        if c in df.columns and _kind(df[c]) == "text":
            groups = near_duplicate_categories(s)
            if groups:
                ex = "; ".join(" / ".join(repr(x) for x in g) for g in groups[:3])
                issues.append({"type": "inconsistent_categories", "column": c, "groups": groups,
                               "message": f"'{c}' has spellings that look like the same value: {ex}"})

    return {"issue_count": len(issues), "issues": issues}


_DATE_RE = re.compile(r"^\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}")


def date_formats(s: pd.Series) -> dict[str, int]:
    """Count shape patterns like 'dddd-dd-dd' vs 'dd/dd/dddd' in a text column."""
    vals = s.dropna().astype(str).str.strip()
    vals = vals[vals.str.match(_DATE_RE)]
    if len(vals) < 5 or len(vals) < 0.8 * s.notna().sum():
        return {}
    shapes = vals.str.extract(r"^(\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4})")[0]
    # 4-digit runs -> YYYY, other digit runs -> N, so 1/5/2024 and 01/05/2024 count as one format
    shapes = shapes.str.replace(r"\d{4}", "YYYY", regex=True).str.replace(r"\d{1,2}", "N", regex=True)
    counts = shapes.value_counts()
    counts = counts[counts >= max(2, 0.02 * len(vals))]
    return {str(k): int(v) for k, v in counts.items()} if len(counts) > 1 else {}


def _typo_distance(a: str, b: str) -> int:
    """Edit distance where swapping two neighbouring letters counts as one edit (Delhi/Dehli = 1)."""
    d = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a) + 1):
        d[i][0] = i
    for j in range(len(b) + 1):
        d[0][j] = j
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            cost = a[i - 1] != b[j - 1]
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                d[i][j] = min(d[i][j], d[i - 2][j - 2] + 1)
    return d[len(a)][len(b)]


def _looks_like_typo(a: str, b: str, cutoff: float) -> bool:
    """Close spellings only: one edit for short words, two for long ones. Male/Female is two edits
    on a short word, so it is not flagged."""
    if min(len(a), len(b)) < 4 or difflib.SequenceMatcher(None, a, b).ratio() < cutoff:
        return False
    return _typo_distance(a, b) <= (1 if min(len(a), len(b)) < 8 else 2)


def _norm(x: str) -> str:
    return re.sub(r"\s+", " ", str(x).strip().lower())


def near_duplicate_categories(s: pd.Series, max_unique: int = 150, cutoff: float = 0.8) -> list[list[str]]:
    """Group values that differ only by case/whitespace or a small typo."""
    vals = s.dropna().astype(str)
    uniq = vals.unique()
    if len(uniq) < 2 or len(uniq) > max_unique:
        return []
    counts = vals.value_counts()
    by_norm: dict[str, list[str]] = {}
    for v in uniq:
        by_norm.setdefault(_norm(v), []).append(v)

    # merge normalised keys that are close (typos), most common first
    keys = sorted(by_norm, key=lambda k: -sum(counts[v] for v in by_norm[k]))
    merged: list[list[str]] = []
    used: set[str] = set()
    for k in keys:
        if k in used:
            continue
        group = list(by_norm[k])
        used.add(k)
        if len(k) >= 4:
            for k2 in keys:
                if k2 not in used and _looks_like_typo(k, k2, cutoff):
                    group += by_norm[k2]
                    used.add(k2)
        if len(group) > 1:
            merged.append(sorted(group, key=lambda v: -counts[v]))
    return merged
