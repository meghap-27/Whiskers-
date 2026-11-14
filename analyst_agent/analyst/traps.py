"""Detect common analysis traps before an answer goes out (PRD FR9).

- Simpson's paradox: a gap between two groups that reverses inside most subgroups.
- Outliers driving an average: mean far from the median, or a few rows holding much of the total.
- Small samples: groups too small to compare.
- Causal language: "X causes Y" claims made from observational comparisons.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from analyst_agent.config import SMALL_SAMPLE_N


def check_comparison(df: pd.DataFrame, value_col: str, group_col: str,
                     split_candidates: list[str] | None = None) -> list[dict]:
    """Run every trap check for 'compare value_col across group_col' on row-level data."""
    warnings: list[dict] = []
    warnings += small_samples(df, value_col, group_col)
    warnings += outlier_influence(df, value_col, group_col)
    warnings += simpsons_paradox(df, value_col, group_col, split_candidates)
    return warnings


def small_samples(df, value_col, group_col, n_min: int = SMALL_SAMPLE_N) -> list[dict]:
    counts = df.dropna(subset=[value_col]).groupby(group_col).size()
    small = counts[counts < n_min]
    if small.empty:
        return []
    detail = ", ".join(f"{k} (n={v})" for k, v in small.items())
    return [{"type": "small_sample", "groups": {str(k): int(v) for k, v in small.items()},
             "message": f"Some groups are too small to compare reliably: {detail}."}]


def outlier_influence(df, value_col, group_col=None, ratio: float = 1.5, top_share: float = 0.2) -> list[dict]:
    s = pd.to_numeric(df[value_col], errors="coerce").dropna()
    if len(s) < 20 or not pd.api.types.is_numeric_dtype(s):
        return []
    out = []
    mean, median = float(s.mean()), float(s.median())
    k = max(1, int(np.ceil(len(s) * 0.01)))
    total = float(s.sum())
    share = float(s.nlargest(k).sum() / total) if total > 0 else 0.0
    q1, q3 = s.quantile([0.25, 0.75])
    n_out = int(((s < q1 - 3 * (q3 - q1)) | (s > q3 + 3 * (q3 - q1))).sum())
    if (median > 0 and mean / median > ratio) or share > top_share:
        out.append({
            "type": "outliers_drive_mean", "mean": round(mean, 2), "median": round(median, 2),
            "top_1pct_share": round(share, 3), "extreme_rows": n_out,
            "message": (f"The average of {value_col} ({mean:,.2f}) is pulled by a few extreme rows: the median is "
                        f"{median:,.2f} and the top 1% of rows hold {share:.0%} of the total. "
                        f"Report the median too, or check those {n_out} extreme rows."),
        })
    return out


def simpsons_paradox(df, value_col, group_col, split_candidates=None,
                     min_rows: int = 5, max_levels: int = 12) -> list[dict]:
    """Compare the two groups with the largest overall gap inside each level of other columns."""
    data = df.dropna(subset=[value_col, group_col])
    if not pd.api.types.is_numeric_dtype(data[value_col]):
        return []
    overall = data.groupby(group_col)[value_col].mean()
    counts = data.groupby(group_col).size()
    overall = overall[counts >= min_rows * 2]
    if len(overall) < 2:
        return []
    hi, lo = overall.idxmax(), overall.idxmin()

    if split_candidates is None:
        split_candidates = [c for c in data.columns
                            if c not in (value_col, group_col)
                            and not pd.api.types.is_numeric_dtype(data[c])
                            and not pd.api.types.is_datetime64_any_dtype(data[c])
                            and 2 <= data[c].nunique() <= max_levels]
    out = []
    for split in split_candidates:
        sub = data[data[group_col].isin([hi, lo])]
        means = sub.groupby([split, group_col])[value_col].agg(["mean", "size"]).reset_index()
        means = means[means["size"] >= min_rows]
        wide = means.pivot(index=split, columns=group_col, values="mean").dropna()
        if len(wide) < 2 or hi not in wide or lo not in wide:
            continue
        reversed_in = wide[wide[hi] < wide[lo]]
        if len(reversed_in) / len(wide) >= 0.66:
            out.append({
                "type": "simpsons_paradox", "split_by": split, "higher_overall": str(hi), "lower_overall": str(lo),
                "overall_means": {str(hi): round(float(overall[hi]), 2), str(lo): round(float(overall[lo]), 2)},
                "reversed_in": [str(x) for x in reversed_in.index], "subgroups_checked": int(len(wide)),
                "message": (f"Simpson's paradox: '{hi}' has a higher average {value_col} than '{lo}' overall, "
                            f"but '{lo}' is higher within {len(reversed_in)} of {len(wide)} '{split}' groups. "
                            f"The overall gap comes from how the groups are spread across '{split}'; "
                            f"report the split-by-{split} comparison."),
            })
    return out


_CAUSAL = re.compile(
    r"\b(causes?|caused|causing|leads? to|led to|drives?|driven by|results? in|because of|due to|"
    r"impacts?|effect of|increases?|decreases?|boosts?|reduces?)\b", re.I)


def causal_language(text: str) -> list[str]:
    """Phrases that imply causation. The caller decides whether the analysis supports them."""
    return sorted({m.group(0).lower() for m in _CAUSAL.finditer(text or "")})
