"""Export a run as a Jupyter notebook that reproduces the analysis end to end (PRD FR11)."""
from __future__ import annotations

import json
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook


def run_to_notebook(run, session, data_dir: str = ".") -> nbformat.NotebookNode:
    """`data_dir` is where the notebook will find the original data files (default: next to it)."""
    cells = [new_markdown_cell(f"# {run.question}\n\nExported from AI Data Analyst Agent · run `{run.id}`")]

    load = ["import duckdb", "import pandas as pd", "import plotly.express as px", "from scipy import stats", "",
            "con = duckdb.connect()"]
    for ds in session.datasets.values():
        fname = Path(ds.source_path).name
        path = f"{data_dir.rstrip('/')}/{fname}"
        if fname.endswith(".parquet"):
            load.append(f'{ds.table} = pd.read_parquet("{path}")')
        elif fname.endswith((".xlsx", ".xls")):
            load.append(f'{ds.table} = pd.read_excel("{path}")')
        else:
            load.append(f'{ds.table} = pd.read_csv("{path}")')
        load.append(f'con.register("{ds.table}", {ds.table})')
    main = next(iter(session.datasets), None)
    if main:
        load.append(f"df = {main}")
    load.append("result = None")
    cells.append(new_markdown_cell("## Load the data"))
    cells.append(new_code_cell("\n".join(load)))

    for s in run.steps:
        if s.error:
            continue
        if s.tool == "run_sql":
            if s.input.get("purpose"):
                cells.append(new_markdown_cell(f"**Step {s.idx}.** {s.input['purpose']}"))
            q = s.input["query"].replace('"""', '\\"\\"\\"')
            cells.append(new_code_cell(f'result = con.execute("""\n{q}\n""").df()\ndata = result\nresult'))
        elif s.tool == "run_python":
            cells.append(new_markdown_cell(f"**Step {s.idx}.** Python analysis"))
            code = s.input["code"]
            if s.table is not None:
                code += "\ndata = output if 'output' in dir() else result"
            cells.append(new_code_cell(code))
        elif s.tool == "make_chart" and s.chart:
            cells.append(new_markdown_cell(f"**Step {s.idx}.** Chart"))
            cells.append(new_code_cell(s.chart["code"]))
        elif s.tool == "check_findings":
            a = s.input
            sql = (a.get("sql") or f'SELECT * FROM "{main}"').replace('"""', '\\"\\"\\"')
            cells.append(new_markdown_cell(f"**Step {s.idx}.** Is the difference in `{a['value_column']}` across "
                                           f"`{a['group_column']}` real? (significance test)"))
            cells.append(new_code_cell(
                f'rows = con.execute("""\n{sql}\n""").df()\n'
                f'groups = [g[{a["value_column"]!r}].dropna() for _, g in rows.groupby({a["group_column"]!r})]\n'
                "test = stats.ttest_ind(*groups, equal_var=False) if len(groups) == 2 else stats.f_oneway(*groups)\n"
                "test"))

    answer = run.answer or "(no answer)"
    if run.warnings:
        answer += "\n\n**Caveats**\n" + "\n".join(f"- {w['message']}" for w in run.warnings)
    if run.verification.get("caveat"):
        answer += f"\n- {run.verification['caveat']}"
    cells.append(new_markdown_cell("## Answer\n\n" + answer))

    nb = new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    return nb


def save_notebook(run, session, path: str | Path, data_dir: str = ".") -> Path:
    path = Path(path)
    nbformat.write(run_to_notebook(run, session, data_dir), path)
    return path


def notebook_bytes(run, session, data_dir: str = ".") -> bytes:
    return nbformat.writes(run_to_notebook(run, session, data_dir)).encode()
