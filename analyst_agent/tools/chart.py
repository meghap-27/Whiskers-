"""make_chart tool: turn a result table into a Plotly figure (PRD FR6)."""
from __future__ import annotations

import pandas as pd
import plotly.express as px

CHART_TYPES = ("bar", "line", "scatter", "histogram", "box", "pie", "heatmap")
# cat-coat palette: ginger, chocolate, cream, rust, smoke grey, black, peach, tortoiseshell
PALETTE = ["#E07A2E", "#7B4A2C", "#E9B97A", "#B4531A", "#8C8279", "#3A2417", "#F2A889", "#5E6B4E"]


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

    kw = {"title": title, "color_discrete_sequence": PALETTE}
    if color:
        kw["color"] = color
    if chart_type == "bar":
        fig = px.bar(df, x=x, y=y, barmode="group", **kw)   # side by side, so groups compare
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
        fig = px.imshow(pivot, text_auto=".2s", aspect="auto", title=title,
                        color_continuous_scale=["#FBF1E3", "#E9B97A", "#E07A2E", "#7B4A2C"])

    fig.update_layout(template="plotly_white", margin=dict(l=40, r=20, t=60, b=40),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="JetBrains Mono, monospace", color="#3A2417"))
    fig.update_xaxes(gridcolor="#EAD3B2")
    fig.update_yaxes(gridcolor="#EAD3B2")
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
        if chart_type == "bar":
            args.append("barmode='group'")
    fn ={"line": "line", "bar": "bar", "scatter": "scatter", "histogram": "histogram",
          "box": "box", "pie": "pie"}[chart_type]
    return f"px.{fn}(data, {', '.join(args)}, title={title!r})"
