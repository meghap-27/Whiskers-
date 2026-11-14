"""Suggested questions for the home-page folders, built from the uploaded file's profile.

No LLM call: the columns, their types and the data-quality report are enough to write useful
questions instantly and for free. Three folders, four questions each:
  - Data checks:        from the quality report (duplicates, mixed dates, misspellings, blanks)
  - Is it real?:        group comparisons, correlation and skew, which trigger the stats checks
  - Business questions: rankings, trends over time, breakdowns
"""
from __future__ import annotations

import re

_MEASURE_HINTS = ("revenue", "sales", "amount", "value", "price", "cost", "profit", "score", "income",
                  "spend", "salary", "total", "count", "qty", "quantity", "rating", "hours", "duration", "age")
_ADDITIVE = ("revenue", "sales", "amount", "value", "cost", "profit", "spend", "income", "total", "qty",
             "quantity", "count", "units", "orders")
_UNIT_WORDS = {"usd", "inr", "eur", "pct", "percent", "week", "weekly", "90d", "30d", "7d", "num", "amt"}
_ID_HINTS = re.compile(r"(^id$|_id$|^id_|code$|zip|pin|phone|number$|^no$)", re.I)


def _pretty(col: str) -> str:
    """order_value -> 'order value'; monthly_revenue_usd -> 'monthly revenue' (unit words dropped)."""
    words = [w for w in re.split(r"[_\-\s]+", col.strip()) if w]
    kept = [w for w in words if w.lower() not in _UNIT_WORDS] or words
    return " ".join(kept).lower() if col.islower() or "_" in col else " ".join(kept)


def _agg(measure: str) -> str:
    """'total' for things that add up (revenue, quantity), 'average' for the rest (scores, ratings)."""
    return "total" if any(h in measure.lower() for h in _ADDITIVE) else "average"


def _rank(name: str) -> int:
    low = name.lower()
    return min((i for i, h in enumerate(_MEASURE_HINTS) if h in low), default=len(_MEASURE_HINTS))


def _short(text: str, n: int = 26) -> str:
    text = text[:1].upper() + text[1:]
    if len(text) <= n:
        return text
    words = text[: n - 1].rsplit(" ", 1)[0].split()      # don't cut a word in half
    while words and words[-1] in {"by", "×", "→", "of", "in", "and"}:
        words.pop()                                      # nor end on a connector
    return " ".join(words) + "…"


def _measures(profile: dict) -> list[str]:
    rows = profile["row_count"] or 1
    cols = [c for c in profile["columns"] if c["dtype"] == "number" and not _ID_HINTS.search(c["name"])
            and not (c["unique"] >= rows * 0.98 and float(c.get("min") or 0).is_integer())]
    # prefer business-sounding names, then columns with more distinct values
    return [c["name"] for c in sorted(cols, key=lambda c: (_rank(c["name"]), -c["unique"]))]


def _groups(profile: dict) -> list[str]:
    cols = [c for c in profile["columns"] if c["dtype"] in ("text", "bool") and 2 <= c["unique"] <= 12
            and not _ID_HINTS.search(c["name"])]
    return [c["name"] for c in sorted(cols, key=lambda c: abs(c["unique"] - 4))]


def _wide_groups(profile: dict) -> list[str]:
    """Text columns with many values (cities, products): good for 'top N' questions."""
    rows = profile["row_count"] or 1
    return [c["name"] for c in profile["columns"] if c["dtype"] == "text" and 12 < c["unique"] < rows * 0.5
            and not _ID_HINTS.search(c["name"])]


def _dates(profile: dict) -> list[str]:
    return [c["name"] for c in profile["columns"] if c["dtype"] == "date"]


def _q(label: str, value: str, hint: str, icon: str) -> dict:
    return {"label": _short(label), "value": value, "hint": hint, "icon": icon}


def suggest_folders(profile: dict, dataset_name: str = "your data") -> list[dict]:
    m, g, wide, d = _measures(profile), _groups(profile), _wide_groups(profile), _dates(profile)
    m1 = m[0] if m else None
    m2 = m[1] if len(m) > 1 else None
    g1 = g[0] if g else None
    g2 = g[1] if len(g) > 1 else None
    rank_by = (wide or g or [None])[0]

    # ---- Data checks: from the quality report first
    checks = []
    for issue in profile.get("quality", {}).get("issues", []):
        t, col = issue["type"], issue.get("column", "")
        if t == "duplicate_rows":
            checks.append(_q("Duplicate rows?", "How many duplicate rows are there, and how do they change the totals?",
                             issue["message"], "⧉"))
        elif t == "inconsistent_categories":
            checks.append(_q(f"Messy {_pretty(col)}?", f"Which {_pretty(col)} values look misspelled or inconsistent, "
                             "and what are the totals after cleaning them?", issue["message"], "✎"))
        elif t == "mixed_date_formats":
            checks.append(_q(f"{_pretty(col)} formats?",
                             f"Is {_pretty(col)} stored in more than one date format?", issue["message"], "◷"))
        elif t == "high_nulls":
            checks.append(_q(f"Blank {_pretty(col)}?", f"How much of {_pretty(col)} is missing, and does it bias "
                             "the results?", issue["message"], "▢"))
        elif t == "constant_column":
            checks.append(_q(f"{_pretty(col)} useful?", f"Does {_pretty(col)} ever change?",
                             issue["message"], "▢"))
    fillers = [
        _q("Missing values?", "Which columns have missing values, and how many?",
           "Counts blanks in every column", "▢"),
        _q("Any duplicates?", "Are there duplicate rows?", "Checks for exact duplicates", "⧉"),
    ]
    if m1:
        fillers.insert(0, _q(f"Outliers in {_pretty(m1)}?", f"Are there outliers in {_pretty(m1)}?",
                             "Flags extreme values and how much they move the average", "⚠"))
    fillers.append(_q("Summarise columns", "Give me a quick summary of each column.", "Types, ranges and blanks", "≡"))
    checks = _dedupe(checks + fillers)[:4]

    # ---- Is it real?: questions that trigger significance tests and trap checks
    stats = []
    if m1 and g1:
        stats.append(_q(f"{_pretty(m1)} by {_pretty(g1)}",
                        f"Does the average {_pretty(m1)} really differ by {_pretty(g1)}?",
                        "Significance test + Simpson's paradox check", "⇄"))
    if m1 and g2:
        stats.append(_q(f"{_pretty(m1)} by {_pretty(g2)}",
                        f"Does the average {_pretty(m1)} really differ by {_pretty(g2)}?",
                        "ANOVA or t-test with effect size", "★"))
    if m1 and m2:
        stats.append(_q(f"{_pretty(m2)} → {_pretty(m1)}?", f"Is {_pretty(m2)} related to {_pretty(m1)}?",
                        "Correlation, with a caveat that it isn't causation", "∿"))
    if m1:
        stats.append(_q(f"Skewed {_pretty(m1)}?", f"Is the average {_pretty(m1)} skewed by outliers?",
                        "Mean vs median and the share held by the top 1%", "⚖"))
    if g1 and len(g) > 1 and not m1:
        stats.append(_q(f"{_pretty(g1)} vs {_pretty(g[1])}", f"Is {_pretty(g1)} related to {_pretty(g[1])}?",
                        "Chi-square test of independence", "⇄"))
    stats.append(_q("Strongest links", "Which columns are most strongly related to each other?",
                    "Correlation scan across numeric columns", "∿"))
    stats = _dedupe(stats)[:4]

    # ---- Business questions
    biz = []
    if rank_by and m1:
        agg = _agg(m1)
        if rank_by in wide:
            biz.append(_q(f"Top {_pretty(rank_by)} by {_pretty(m1)}",
                          f"Which 5 {_pretty(rank_by)} values have the highest {agg} {_pretty(m1)}?",
                          "Ranked table and bar chart", "▤"))
        else:
            biz.append(_q(f"Best {_pretty(rank_by)}", f"Which {_pretty(rank_by)} has the highest {agg} {_pretty(m1)}?",
                          "Ranked table and bar chart", "▤"))
    if d and m1:
        biz.append(_q(f"{_pretty(m1)} over time", f"Show the monthly trend of {_agg(m1)} {_pretty(m1)}.",
                      f"Line chart by month of {_pretty(d[0])}", "↗"))
    elif d:
        biz.append(_q("Rows over time", f"How many rows are there per month of {_pretty(d[0])}?",
                      "Line chart by month", "↗"))
    if g1:
        biz.append(_q(f"Share by {_pretty(g1)}", f"What share of rows falls in each {_pretty(g1)}?",
                      "Counts and percentages", "◔"))
    if g1 and m1 and g2:
        biz.append(_q(f"{_pretty(g1)} × {_pretty(g2)}", f"Compare average {_pretty(m1)} across {_pretty(g1)} "
                      f"and {_pretty(g2)}.", "Grouped bar chart", "▦"))
    biz.append(_q("Key takeaways", f"What are the three most important insights in {dataset_name}?",
                  "A short overview with numbers", "✦"))
    biz = _dedupe(biz)[:4]

    return [
        {"coat": "ginger", "label": "Data checks", "caption": "CLEAN BEFORE YOU COUNT", "items": checks},
        {"coat": "chocolate", "label": "Is it real?", "caption": "STATS BEFORE CLAIMS", "items": stats},
        {"coat": "cream", "label": "Business questions", "caption": "THE EVERYDAY STUFF", "items": biz},
    ]


def _dedupe(items: list[dict]) -> list[dict]:
    seen, out = set(), []
    for it in items:
        if it["value"] not in seen:
            seen.add(it["value"])
            out.append(it)
    return out
