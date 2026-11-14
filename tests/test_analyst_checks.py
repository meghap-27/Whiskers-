import numpy as np
import pandas as pd

from analyst_agent.analyst.stats import compare_groups
from analyst_agent.analyst.traps import (
    causal_language,
    check_comparison,
    outlier_influence,
    simpsons_paradox,
    small_samples,
)
from analyst_agent.analyst.verify import verify_answer


def _rows(session, where="channel IN ('Online','Store')"):
    return session.con.execute(f"SELECT * FROM retail_orders WHERE {where}").df()


def test_simpsons_paradox_detected_on_sample(sample_session):
    w = simpsons_paradox(_rows(sample_session), "order_value", "channel")
    assert len(w) == 1
    assert w[0]["split_by"] == "region"
    assert w[0]["higher_overall"] == "Store" and len(w[0]["reversed_in"]) == 4


def test_no_paradox_when_effect_is_consistent():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"g": ["a", "b"] * 200, "seg": ["x"] * 200 + ["y"] * 200})
    df["v"] = np.where(df.g == "a", 10, 5) + rng.normal(0, 1, 400)
    assert simpsons_paradox(df, "v", "g") == []


def test_outliers_flagged(sample_session):
    all_rows = sample_session.con.execute("SELECT * FROM retail_orders").df()
    w = outlier_influence(all_rows, "order_value")
    assert w and w[0]["top_1pct_share"] > 0.2


def test_small_samples(sample_session):
    all_rows = sample_session.con.execute("SELECT * FROM retail_orders").df()
    w = small_samples(all_rows, "order_value", "channel")
    assert w and "B2B" in w[0]["groups"]


def test_significance_two_groups(sample_session):
    r = compare_groups(_rows(sample_session), "order_value", "channel")
    assert r["test"] == "Welch's t-test" and r["significant"]


def test_no_significant_difference_reported():
    # every group gets exactly the same values, so there is truly no difference
    values = np.random.default_rng(1).normal(5, 1, 100)
    df = pd.DataFrame({"g": ["a"] * 100 + ["b"] * 100 + ["c"] * 100, "v": np.tile(values, 3)})
    r = compare_groups(df, "v", "g")
    assert r["test"] == "One-way ANOVA" and not r["significant"]
    assert "no statistically significant" in r["summary"]


def test_chi_square_for_categorical():
    df = pd.DataFrame({"g": ["a"] * 50 + ["b"] * 50, "churned": ["yes"] * 40 + ["no"] * 10 + ["yes"] * 10 + ["no"] * 40})
    r = compare_groups(df, "churned", "g")
    assert r["test"].startswith("Chi-square") and r["significant"]


def test_check_comparison_bundles_all(sample_session):
    types = {w["type"] for w in check_comparison(sample_session.con.execute("SELECT * FROM retail_orders").df(),
                                                 "order_value", "channel")}
    assert {"simpsons_paradox", "outliers_drive_mean", "small_sample"} <= types


def test_causal_language():
    assert causal_language("Discounts lead to higher ratings") == ["lead to"]
    assert causal_language("Region West has the highest average") == []


def test_verifier_catches_invented_numbers():
    src = ['{"rows": [["West", 1873.4567]], "share": 0.4712}']
    ok = verify_answer("West averages 1,873.46 and the top 1% hold 47%.", src)
    assert ok["verified"]
    bad = verify_answer("West averages 2,950.10.", src)
    assert not bad["verified"] and bad["unverified_numbers"] == ["2,950.10"]
