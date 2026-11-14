"""Offline demo: a scripted run on the sample data, so the app works without an API key."""
from analyst_agent.agent.llm import ScriptedLLM, final, tool_call

DEMO_QUESTION = "Do Online or Store orders have a higher average order value?"


def demo_llm() -> ScriptedLLM:
    return ScriptedLLM([
        tool_call("run_sql", {
            "query": "SELECT channel, ROUND(AVG(order_value), 2) AS avg_order_value, COUNT(*) AS orders\n"
                     "FROM retail_orders\nWHERE channel IN ('Online', 'Store')\nGROUP BY channel\nORDER BY 2 DESC",
            "purpose": "Average order value by channel (excluding the 8 bulk B2B orders)",
        }, text="I'll compare average order value by channel, excluding bulk B2B orders."),
        tool_call("check_findings", {
            "value_column": "order_value", "group_column": "channel",
            "sql": "SELECT * FROM retail_orders WHERE channel IN ('Online', 'Store')",
        }),
        tool_call("run_sql", {
            "query": "SELECT region, channel, ROUND(AVG(order_value), 2) AS avg_order_value\n"
                     "FROM retail_orders\nWHERE channel IN ('Online', 'Store')\nGROUP BY region, channel\n"
                     "ORDER BY region, channel",
            "purpose": "Same comparison inside each region, because the check flagged Simpson's paradox",
        }),
        tool_call("make_chart", {"chart_type": "bar", "x": "region", "y": "avg_order_value", "color": "channel",
                                 "title": "Online beats Store in every region"}),
        final("Overall, Store orders look bigger: 1496.32 on average vs 1109.38 for Online. But that is "
              "Simpson's paradox. Inside every region Online is higher (West 1932.53 vs 1868.95, North 1254.68 "
              "vs 1125.52, South 980.12 vs 842.81, East 785.47 vs 651.82). Store orders are concentrated in the "
              "high-spending West, which inflates the Store average. Compare channels within a region, not overall."),
    ])
