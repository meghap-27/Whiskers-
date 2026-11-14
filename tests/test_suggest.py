"""Folder questions come from the uploaded file, and the latest upload is the main table."""
from pathlib import Path

from analyst_agent.data.store import Session
from analyst_agent.suggest import _short, suggest_folders

DATA = Path(__file__).resolve().parent.parent / "sample_data"


def _all(folders):
    return [i for f in folders for i in f["items"]]


def test_questions_follow_the_uploaded_file():
    s = Session()
    saas = s.load_file(DATA / "saas_subscriptions.xlsx")
    qs = " ".join(i["value"] for i in _all(suggest_folders(saas.profile, saas.name)))
    assert "monthly revenue" in qs and "plan" in qs and "nps" in qs
    assert "order value" not in qs and "region" not in qs          # nothing from the retail sample

    students = s.load_file(DATA / "student_performance.csv")
    qs = " ".join(i["value"] for i in _all(suggest_folders(students.profile, students.name)))
    assert "score" in qs and "school" in qs and "duplicate" in qs and "city" in qs


def test_three_folders_of_four_short_labels():
    s = Session()
    ds = s.load_file(DATA / "retail_orders.csv")
    folders = suggest_folders(ds.profile, ds.name)
    assert [f["label"] for f in folders] == ["Data checks", "Is it real?", "Business questions"]
    for f in folders:
        assert len(f["items"]) == 4
        for item in f["items"]:
            assert len(item["label"]) <= 26 and item["value"].endswith(("?", "."))


def test_tiny_or_text_only_files_still_get_questions():
    import pandas as pd
    s = Session()
    ds = s.load_dataframe(pd.DataFrame({"name": ["a", "b", "c"], "team": ["x", "y", "x"]}), "people.csv")
    folders = suggest_folders(ds.profile, ds.name)
    assert all(len(f["items"]) >= 1 for f in folders)


def test_short_labels_cut_on_words():
    assert _short("Outliers in monthly revenue?") == "Outliers in monthly…"
    assert _short("Reading score by parent education") == "Reading score by parent…"


def test_latest_upload_is_main_and_listed_first():
    s = Session()
    s.load_file(DATA / "retail_orders.csv")
    s.load_file(DATA / "student_performance.csv")
    assert s.main_table == "student_performance"
    text = s.schema_text()
    assert text.index('"student_performance"') < text.index('"retail_orders"')
    assert s.remove("student_performance") and s.main_table == "retail_orders"
    assert "student_performance" not in [r[0] for r in s.con.execute("SHOW TABLES").fetchall()]
