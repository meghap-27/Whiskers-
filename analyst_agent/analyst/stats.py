"""Significance tests run before the agent claims "X is higher than Y" (PRD FR9)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from analyst_agent.config import SIGNIFICANCE_ALPHA, SMALL_SAMPLE_N


def compare_groups(df: pd.DataFrame, value_col: str, group_col: str, max_groups: int = 20) -> dict:
    """Pick and run the right test for value_col across the levels of group_col."""
    for c in (value_col, group_col):
        if c not in df.columns:
            raise ValueError(f"Column '{c}' not found. Available: {list(df.columns)}")
    data = df[[group_col, value_col]].dropna()
    if data[group_col].nunique() > max_groups:
        raise ValueError(f"'{group_col}' has more than {max_groups} groups; filter or bucket it first.")
    if pd.api.types.is_numeric_dtype(data[value_col]) and data[value_col].nunique() > 2:
        return _numeric(data, value_col, group_col)
    return _categorical(data, value_col, group_col)


def _numeric(data, value_col, group_col) -> dict:
    groups = {str(k): g[value_col].to_numpy(dtype=float) for k, g in data.groupby(group_col)}
    groups = {k: v for k, v in groups.items() if len(v) >= 2}
    if len(groups) < 2:
        raise ValueError("Need at least two groups with 2+ rows each.")
    summary = [{"group": k, "n": int(len(v)), "mean": round(float(v.mean()), 4),
                "median": round(float(np.median(v)), 4), "std": round(float(v.std(ddof=1)), 4)}
               for k, v in groups.items()]
    summary.sort(key=lambda r: -r["mean"])
    arrays = list(groups.values())

    if len(groups) == 2:
        (a_name, a), (b_name, b) = groups.items()
        t = stats.ttest_ind(a, b, equal_var=False)
        mw = stats.mannwhitneyu(a, b, alternative="two-sided")
        pooled = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
        d = float((a.mean() - b.mean()) / pooled) if pooled > 0 else 0.0
        res = {"test": "Welch's t-test", "statistic": float(t.statistic), "p_value": float(t.pvalue),
               "effect_size": {"cohens_d": round(d, 3), "magnitude": _cohen_label(d)},
               "robustness_check": {"test": "Mann-Whitney U", "p_value": float(mw.pvalue)}}
    else:
        f = stats.f_oneway(*arrays)
        kw = stats.kruskal(*arrays)
        allv = np.concatenate(arrays)
        ss_between = sum(len(v) * (v.mean() - allv.mean()) ** 2 for v in arrays)
        ss_total = ((allv - allv.mean()) ** 2).sum()
        eta2 = float(ss_between / ss_total) if ss_total > 0 else 0.0
        res = {"test": "One-way ANOVA", "statistic": float(f.statistic), "p_value": float(f.pvalue),
               "effect_size": {"eta_squared": round(eta2, 3)},
               "robustness_check": {"test": "Kruskal-Wallis", "p_value": float(kw.pvalue)}}
    return _finish(res, summary, value_col, group_col)


def _categorical(data, value_col, group_col) -> dict:
    table = pd.crosstab(data[group_col], data[value_col])
    if table.shape[0] < 2 or table.shape[1] < 2:
        raise ValueError("Need at least two groups and two outcome values.")
    chi2, p, dof, _ = stats.chi2_contingency(table)
    n = table.to_numpy().sum()
    v = float(np.sqrt(chi2 / (n * (min(table.shape) - 1))))
    summary = [{"group": str(k), "n": int(row.sum()),
                **{f"share_{c}": round(float(row[c] / row.sum()), 4) for c in table.columns}}
               for k, row in table.iterrows()]
    res = {"test": "Chi-square test of independence", "statistic": float(chi2), "p_value": float(p),
           "effect_size": {"cramers_v": round(v, 3)}}
    return _finish(res, summary, value_col, group_col)


def _finish(res, summary, value_col, group_col) -> dict:
    p = res["p_value"]
    res["p_value"] = float(f"{p:.4g}")
    res["significant"] = bool(p < SIGNIFICANCE_ALPHA)
    res["groups"] = summary
    small = [g["group"] for g in summary if g["n"] < SMALL_SAMPLE_N]
    res["small_groups"] = small
    verdict = ("a statistically significant difference" if res["significant"]
               else "no statistically significant difference")
    res["summary"] = (f"{res['test']}: {verdict} in {value_col} across {group_col} "
                      f"(p = {res['p_value']}, alpha = {SIGNIFICANCE_ALPHA}).")
    if small:
        res["summary"] += f" Small groups (n < {SMALL_SAMPLE_N}): {', '.join(small)}."
    return res


def _cohen_label(d: float) -> str:
    d = abs(d)
    return "negligible" if d < 0.2 else "small" if d < 0.5 else "medium" if d < 0.8 else "large"
