import pandas as pd
import pytest

from analyst_agent.tools.chart import make_chart
from analyst_agent.tools.python_exec import run_python
from analyst_agent.tools.sql import UnsafeSQLError, run_sql, validate_select


@pytest.mark.parametrize("q", [
    "DROP TABLE retail_orders",
    "DELETE FROM retail_orders",
    "INSERT INTO retail_orders VALUES (1)",
    "SELECT 1; DROP TABLE retail_orders",
    "SELECT * FROM read_csv('/etc/passwd')",
    "SELECT getenv('HOME')",
    "COPY retail_orders TO 'out.csv'",
])
def test_sql_guard_blocks_unsafe(q):
    with pytest.raises(UnsafeSQLError):
        validate_select(q)


def test_sql_runs_select(sample_session):
    df = run_sql(sample_session.con, "SELECT region, COUNT(*) n FROM retail_orders GROUP BY 1")
    assert set(df["region"]) == {"West", "North", "South", "East"}
    assert df["n"].sum() == 3025


def test_duckdb_file_access_disabled(sample_session):
    import duckdb
    with pytest.raises(duckdb.Error):
        sample_session.con.execute("SELECT * FROM '/etc/hostname'").fetchall()


def test_python_tool_returns_output_table(sample_session):
    paths = {k: d.path for k, d in sample_session.datasets.items()}
    res = run_python("output = df.groupby('region').order_value.median().reset_index()\nprint('ok')",
                     paths, main_table="retail_orders")
    assert res["ok"] and "ok" in res["stdout"]
    assert list(res["output_df"].columns) == ["region", "order_value"]


def test_python_tool_reports_errors():
    res = run_python("1/0", {})
    assert not res["ok"] and "ZeroDivisionError" in res["error"]


def test_python_tool_sees_last_sql_result():
    res = run_python("result['x'].sum()", {}, last_result=pd.DataFrame({"x": [1, 2, 3]}))
    assert res["last_value"] == "6"


def test_chart_validates_columns():
    df = pd.DataFrame({"a": ["x", "y"], "b": [1, 2]})
    assert "figure_json" in make_chart(df, "bar", "a", "b", title="t")
    with pytest.raises(ValueError):
        make_chart(df, "bar", "a", "missing")
