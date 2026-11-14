import pandas as pd

from analyst_agent.data.profile import date_formats, near_duplicate_categories, quality_report


def _types(report):
    return {i["type"] for i in report["issues"]}


def test_sample_quality_report_finds_planted_issues(sample_session):
    q = sample_session.datasets["retail_orders"].profile["quality"]
    assert {"duplicate_rows", "mixed_date_formats", "inconsistent_categories"} <= _types(q)
    dup = next(i for i in q["issues"] if i["type"] == "duplicate_rows")
    assert dup["count"] == 25


def test_mixed_date_formats():
    s = pd.Series(["2024-01-05"] * 10 + ["01/06/2024"] * 5)
    assert set(date_formats(s)) == {"YYYY-N-N", "N/N/YYYY"}
    assert date_formats(pd.Series(["2024-01-05", "2024-1-6"] * 5)) == {}


def test_near_duplicate_categories():
    s = pd.Series(["Delhi"] * 20 + ["delhi "] * 2 + ["Dehli"] + ["Pune"] * 10 + ["Patna"] * 10)
    groups = near_duplicate_categories(s)
    assert len(groups) == 1 and groups[0][0] == "Delhi"
    assert set(groups[0]) == {"Delhi", "delhi ", "Dehli"}


def test_clean_data_has_no_issues():
    df = pd.DataFrame({"a": range(10), "b": list("abcdefghij")})
    assert quality_report(df)["issue_count"] == 0


def test_dates_and_numbers_are_typed(sample_session):
    types = {c["name"]: c["dtype"] for c in sample_session.datasets["retail_orders"].profile["columns"]}
    assert types["order_date"] == "date"
    assert types["order_value"] == "number"
    assert types["city"] == "text"
