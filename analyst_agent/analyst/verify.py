"""Check the final answer before it is shown (PRD FR7, FR9).

1. Every number in the answer must be traceable to a tool output (allowing for rounding,
   thousands separators, percentages and k/M/B/lakh/crore suffixes).
2. Causal wording ("leads to", "drives") gets a caveat unless the run tested it.
"""
from __future__ import annotations

import math
import re

from analyst_agent.analyst.traps import causal_language

_NUM = re.compile(r"(?<![\w.])[-+]?(?:\d{1,3}(?:,\d{2,3})+|\d+)(?:\.\d+)?\s*(?:%|k|m|bn|b|lakh|crore|cr)?(?![\w])",
                   re.I)
_SCALE = {"k": 1e3, "m": 1e6, "b": 1e9, "bn": 1e9, "lakh": 1e5, "crore": 1e7, "cr": 1e7}


def numbers_in(text: str) -> list[tuple[str, float, int, str]]:
    """(raw, value, decimals shown, suffix) for each number in text, skipping ranking words
    such as "top 5" / "bottom 10%" that describe the question rather than a result."""
    out = []
    for m in _NUM.finditer(text or ""):
        raw = m.group(0).strip()
        if re.search(r"\b(top|bottom|first|last)\s*$", text[max(0, m.start() - 10):m.start()], re.I):
            continue
        mnum = re.match(r"[-+]?[\d,]+(?:\.(\d+))?", raw)
        if not mnum:
            continue
        decimals = len(mnum.group(1) or "")
        value = float(mnum.group(0).replace(",", ""))
        suffix = raw[mnum.end():].strip().lower()
        out.append((raw, value, decimals, suffix))
    return out


def _source_values(tool_texts: list[str]) -> list[float]:
    vals = []
    for t in tool_texts:
        for m in re.finditer(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", (t or "").replace(",", "")):
            try:
                vals.append(float(m.group(0)))
            except ValueError:
                pass
    return vals


def _matches(value: float, decimals: int, suffix: str, sources: list[float]) -> bool:
    candidates = [value]
    if suffix in _SCALE:
        candidates = [value * _SCALE[suffix]]
    if suffix == "%":
        candidates = [value, value / 100]
    for target in candidates:
        tol = 0.5 * 10 ** (-decimals) if suffix not in _SCALE else 0.5 * 10 ** (-decimals) * _SCALE[suffix]
        if suffix == "%" and target == value / 100:
            tol = tol / 100
        for s in sources:
            if math.isclose(s, target, rel_tol=1e-9, abs_tol=tol + 1e-12):
                return True
            # sources often carry more precision than the answer; also accept a 0.5% relative gap
            if target != 0 and abs(s - target) / abs(target) <= 0.005:
                return True
    return False


def verify_answer(answer: str, tool_texts: list[str], ignore: set[float] | None = None,
                  causal_tested: bool = False) -> dict:
    sources = _source_values(tool_texts)
    ignore = ignore or set()
    unverified = []
    for raw, value, decimals, suffix in numbers_in(answer):
        if value in ignore or (suffix == "" and decimals == 0 and abs(value) < 10):
            continue  # small counting words like "3 regions" / "top 5"
        if 1900 <= value <= 2100 and decimals == 0 and suffix == "":
            continue  # years
        if not _matches(value, decimals, suffix, sources):
            unverified.append(raw)
    causal = [] if causal_tested else causal_language(answer)
    return {
        "verified": not unverified,
        "unverified_numbers": unverified,
        "causal_phrases": causal,
        "caveat": ("These results show association, not cause and effect; other factors could explain them."
                   if causal else None),
    }
