"""make_chart tool: turn a result table into a Plotly figure (PRD FR6)."""
from __future__ import annotations

import pandas as pd
import plotly.express as px

CHART_TYPES = ("bar", "line", "scatter", "histogram", "box", "pie", "heatmap")


def make_chart(df: pd.DataFrame, chart_type: str, x: str, y: str | None = None,
               color: str | None = None, title: str = "") -> dict:
    """Return {"figure_json": ..., "code": ...}. `code` is reused by notebook export."""
    if df is None or df.empty:
        raise ValueError("No data to chart: run a query or Python step that returns rows first.")
    if chart_type not in CHART_TYPES:
        raise ValueError(f"chart_type must be one of {CHART_TYPES}")
    for col in (x, y, color):
        if col and col not in df.columns:
            raise ValueError(f"Column '{col}' not in data. Available: {list(df.columns)}")

    kw = {"title": title}
    if color:
        kw["color"] = color
    if chart_type == "bar":
        fig = px.bar(df, x=x, y=y, **kw)
    elif chart_type == "line":
        fig = px.line(df.sort_values(x), x=x, y=y, markers=True, **kw)
    elif chart_type == "scatter":
        fig = px.scatter(df, x=x, y=y, **kw)
    elif chart_type == "histogram":
        fig = px.histogram(df, x=x, **kw)
    elif chart_type == "box":
        fig = px.box(df, x=x, y=y, **kw)
    elif chart_type == "pie":
        kw.pop("color", None)
        fig = px.pie(df, names=x, values=y, **kw)
    else:  # heatmap: x, y categorical, color = value column
        if not (y and color):
            raise ValueError("heatmap needs x, y and color (the value column).")
        pivot = df.pivot_table(index=y, columns=x, values=color, aggfunc="mean")
        fig = px.imshow(pivot, text_auto=".2s", aspect="auto", title=title)

    fig.update_layout(template="plotly_white", margin=dict(l=40, r=20, t=60, b=40))
    return {"figure_json": fig.to_json(), "code": chart_code(chart_type, x, y, color, title)}


def chart_code(chart_type: str, x: str, y: str | None, color: str | None, title: str) -> str:
    """Equivalent Plotly Express code, for the exported notebook."""
    args = [f"x={x!r}"]
    if chart_type == "pie":
        args = [f"names={x!r}", f"values={y!r}"]
    elif chart_type == "heatmap":
        return (f"pivot = data.pivot_table(index={y!r}, columns={x!r}, values={color!r}, aggfunc='mean')\n"
                f"px.imshow(pivot, text_auto='.2s', aspect='auto', title={title!r})")
    else:
        if y and chart_type != "histogram":
            args.append(f"y={y!r}")
        if color:
            args.append(f"color={color!r}")
    fn = {"line": "line", "bar": "bar", "scatter": "scatter", "histogram": "histogram",
          "box": "box", "pie": "pie"}[chart_type]
    return f"px.{fn}(data, {', '.join(args)}, title={title!r})"
