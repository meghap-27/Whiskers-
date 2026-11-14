"""Streamlit UI. Run from the repo root:  streamlit run ui/streamlit_app.py"""
import json
import os
import sys
from pathlib import Path

import plotly.io as pio
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analyst_agent.agent.llm import AnthropicLLM  # noqa: E402
from analyst_agent.agent.loop import Agent  # noqa: E402
from analyst_agent.data.store import Session  # noqa: E402
from analyst_agent.demo import DEMO_QUESTION, demo_llm  # noqa: E402
from analyst_agent.export.notebook import notebook_bytes  # noqa: E402
from analyst_agent.glossary import Glossary  # noqa: E402
from analyst_agent.pins import PinBoard  # noqa: E402

SAMPLE = Path(__file__).resolve().parent.parent / "sample_data" / "retail_orders.csv"
TOOL_LABEL = {"run_sql": "Query database", "run_python": "Run code", "make_chart": "Generate chart",
              "check_findings": "Check findings", "save_definition": "Save definition"}

st.set_page_config(page_title="AI Data Analyst Agent", layout="wide")
ss = st.session_state
ss.setdefault("session", Session())
ss.setdefault("runs", [])
glossary, pins = Glossary(), PinBoard()
session: Session = ss.session

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Data")
    up = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"])
    if up and up.name not in [d.name for d in session.datasets.values()]:
        path = session.dir / up.name
        path.write_bytes(up.getvalue())
        session.load_file(path, name=up.name)
    if st.button("Use sample retail dataset", width="stretch") and "retail_orders" not in session.datasets:
        session.load_file(SAMPLE)

    for ds in session.datasets.values():
        p = ds.profile
        with st.expander(f"{ds.table} · {p['row_count']:,} rows · {len(p['columns'])} cols", expanded=True):
            issues = p["quality"]["issues"]
            if issues:
                st.markdown(f"**{len(issues)} data-quality issue(s)**")
                for i in issues:
                    st.warning(i["message"], icon="⚠️")
            else:
                st.success("No data-quality issues found")
            st.dataframe([{"column": c["name"], "type": c["dtype"], "nulls %": c["null_pct"],
                           "unique": c["unique"]} for c in p["columns"]], hide_index=True)

    st.header("Model")
    key = st.text_input("Anthropic API key", type="password", value=os.getenv("ANTHROPIC_API_KEY", ""))
    st.caption("No key? Use the offline demo question on the sample dataset.")

    st.header("Business glossary")
    defs = glossary.list()
    if not defs:
        st.caption('Correct the agent ("revenue should exclude refunds") and it saves the definition here.')
    for d in defs:
        c1, c2 = st.columns([5, 1])
        c1.markdown(f"**{d['term']}** ({d['dataset']}): {d['definition']}"
                    + (f"  \n`{d['sql_expression']}`" if d["sql_expression"] else ""))
        if c2.button("✕", key=f"del_{d['dataset']}_{d['term']}"):
            glossary.delete(d["dataset"], d["term"])
            st.rerun()


# ------------------------------------------------------------------ helpers
def show_run(run, i):
    for w in run.warnings:
        st.warning(w["message"], icon="🔎")
    st.markdown(run.answer)
    v = run.verification
    if v.get("caveat"):
        st.info(v["caveat"])
    if v and not v.get("verified", True):
        st.error("Could not trace these numbers to a tool result: " + ", ".join(v["unverified_numbers"]))
    if run.definitions_used:
        st.caption("Definitions applied: " + ", ".join(run.definitions_used))
    if run.chart:
        st.plotly_chart(pio.from_json(run.chart["figure_json"]), width="stretch", key=f"chart_{i}")
    if run.table is not None:
        st.dataframe(run.table.head(200), hide_index=True)

    with st.expander(f"How I got this · {len(run.steps)} steps · {run.latency_ms / 1000:.1f}s"):
        for s in run.steps:
            st.markdown(f"**{s.idx}. {TOOL_LABEL.get(s.tool, s.tool)}** · {s.duration_ms} ms"
                        + (f" · {s.input.get('purpose')}" if s.input.get("purpose") else ""))
            if s.tool == "run_sql":
                st.code(s.input["query"], language="sql")
            elif s.tool == "run_python":
                st.code(s.input["code"], language="python")
            elif s.tool == "check_findings" and s.extra.get("test"):
                st.caption(s.extra["test"]["summary"])
            else:
                st.code(json.dumps(s.input, indent=2), language="json")
            if s.error:
                st.error(s.error)

    c1, c2, _ = st.columns([1, 1, 4])
    c1.download_button("Notebook", notebook_bytes(run, session, data_dir="."), f"analysis_{run.id}.ipynb",
                       key=f"nb_{i}")
    if c2.button("Pin to dashboard", key=f"pin_{i}"):
        try:
            pins.pin(run)
            st.toast("Pinned")
        except ValueError as e:
            st.toast(str(e))


# ------------------------------------------------------------------ main
st.title("AI Data Analyst Agent")
tab_ask, tab_dash = st.tabs(["Ask", "Dashboard"])

with tab_ask:
    if not session.datasets:
        st.info("Upload a CSV or load the sample dataset from the sidebar to start.")
    for i, (q, run) in enumerate(ss.runs):
        with st.chat_message("user"):
            st.markdown(q)
        with st.chat_message("assistant"):
            show_run(run, i)

    demo = False
    if "retail_orders" in session.datasets and not key:
        demo = st.button(f"▶ Offline demo: “{DEMO_QUESTION}”")
    question = st.chat_input("Ask a question about your data", disabled=not session.datasets)
    if demo:
        question = DEMO_QUESTION

    if question:
        if demo:
            llm = demo_llm()
        elif key:
            llm = AnthropicLLM(api_key=key)
        else:
            st.error("Add an Anthropic API key in the sidebar (or run the offline demo).")
            st.stop()
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            status = st.status("Analyzing…", expanded=True)
            run = None
            for ev in Agent(session, llm, glossary).stream(question):
                if ev["type"] == "step_start":
                    status.write(f"{TOOL_LABEL.get(ev['tool'], ev['tool'])}…")
                elif ev["type"] == "reanalyze":
                    status.write("Verifier found unsupported numbers — re-checking…")
                elif ev["type"] == "done":
                    run = ev["run"]
            status.update(label=f"Done in {run.latency_ms / 1000:.1f}s", state="complete", expanded=False)
        ss.runs.append((question, run))
        st.rerun()

with tab_dash:
    board = pins.list()
    if not board:
        st.caption("Pin an answer to see it here. Pinned answers re-run their query on the current data.")
    for p in board:
        st.subheader(p["title"])
        try:
            out = pins.refresh(p, session.con)
            if out["chart"]:
                st.plotly_chart(pio.from_json(out["chart"]["figure_json"]), width="stretch",
                                key=f"pinchart_{p['id']}")
            else:
                st.dataframe(out["table"], hide_index=True)
        except Exception as e:
            st.caption(f"Load the dataset this pin uses to refresh it ({e}).")
        if st.button("Remove", key=f"unpin_{p['id']}"):
            pins.remove(p["id"])
            st.rerun()
