"""Streamlit UI. Run from the repo root:  streamlit run ui/streamlit_app.py"""
import json
import os
import sys
from pathlib import Path

import plotly.io as pio
import streamlit as st
import streamlit.components.v1 as components

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# On Streamlit Community Cloud, keys live in the app's Secrets; copy them into the environment
# before the agent's config is imported.
try:
    for _k, _v in st.secrets.items():
        if isinstance(_v, (str, int, float, bool)):
            os.environ.setdefault(_k, str(_v))
except Exception:  # no secrets file locally
    pass

from analyst_agent import config  # noqa: E402
from analyst_agent.agent.llm import PROVIDERS, configured_llm, friendly_error  # noqa: E402
from analyst_agent.agent.loop import Agent  # noqa: E402
from analyst_agent.data.store import Session, UploadTooLarge, cleanup_old_sessions  # noqa: E402
from analyst_agent.demo import DEMO_QUESTION, demo_llm  # noqa: E402
from analyst_agent.export.notebook import notebook_bytes  # noqa: E402
from analyst_agent.glossary import Glossary  # noqa: E402
from analyst_agent.limits import RateLimiter  # noqa: E402
from analyst_agent.pins import PinBoard  # noqa: E402
from analyst_agent.telemetry import capture, init_error_tracking, log_run  # noqa: E402

SAMPLE = Path(__file__).resolve().parent.parent / "sample_data" / "retail_orders.csv"
# the home page (scanline cat hero + FolderFloat shelf) is a small HTML/JS component;
# it sends back {"action": "demo"} or {"action": "ask", "question": ...}
home = components.declare_component("analyst_home", path=str(Path(__file__).resolve().parent / "home_component"))
TOOL_LABEL = {"run_sql": "Query database", "run_python": "Run code", "make_chart": "Generate chart",
              "check_findings": "Check findings", "save_definition": "Save definition"}

st.set_page_config(page_title="Data Analyst Agent", page_icon="🐈", layout="wide")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Silkscreen:wght@700&family=JetBrains+Mono:wght@400;600&display=swap');
html, body, [class*="css"], .stMarkdown, .stButton button, .stTextInput input { font-family: 'JetBrains Mono', monospace; }
h1, h2, h3 { font-family: 'Silkscreen', monospace !important; letter-spacing: -.01em; }
.stButton button[kind="primary"], .stDownloadButton button { background: #3A2417; color: #FBF1E3; border: 0; border-radius: 2px; }
.stButton button[kind="primary"]:hover, .stDownloadButton button:hover { background: #E07A2E; color: #FBF1E3; }
[data-testid="stSidebar"] { border-right: 1px solid #EAD3B2; }
[data-testid="stChatMessage"] { background: #FFF8EE; border: 1px solid #F3E0C4; border-radius: 12px; }
.block-container { padding-top: 1.5rem; }
</style>""", unsafe_allow_html=True)
init_error_tracking()


@st.cache_resource
def limiter() -> RateLimiter:          # one limiter shared by every visitor
    return RateLimiter()


@st.cache_data(ttl=600, show_spinner=False)
def sweep_old_sessions() -> int:       # at most every 10 minutes: delete expired uploads
    return cleanup_old_sessions()


sweep_old_sessions()
ss = st.session_state
ss.setdefault("session", Session())
ss.setdefault("runs", [])
ss.setdefault("show_home", True)
ss.setdefault("last_home_event", None)
ss.setdefault("pending_question", None)
session: Session = ss.session
# in public mode every browser session gets its own glossary and pins
owner = session.id if config.PUBLIC_MODE else "local"
glossary, pins = Glossary(owner=owner), PinBoard(owner=owner)

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Data")
    up = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"])
    if up and up.name not in [d.name for d in session.datasets.values()] and ss.get("rejected") != up.name:
        path = session.dir / Path(up.name).name
        path.write_bytes(up.getvalue())
        try:
            session.load_file(path, name=up.name)
        except UploadTooLarge as e:
            path.unlink(missing_ok=True)
            ss.rejected = up.name
            st.error(str(e))
        except Exception as e:
            path.unlink(missing_ok=True)
            ss.rejected = up.name
            capture(e)
            st.error(f"Couldn't read {up.name}. Is it a valid CSV or Excel file? ({type(e).__name__})")
    if st.button("Use sample retail dataset", width="stretch") and "retail_orders" not in session.datasets:
        session.load_file(SAMPLE)
    st.caption(f"Up to {config.MAX_UPLOAD_MB:g} MB / {config.MAX_ROWS:,} rows. "
               + (f"Uploads are deleted after {config.SESSION_TTL_MIN // 60} hours. " if config.PUBLIC_MODE else "")
               + "Column summaries and query results are sent to the AI provider to answer your questions, "
               "so don't upload confidential data.")

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
    choices = {"groq": "Groq · free", "gemini": "Gemini · free", "ollama": "Ollama · free, on your PC",
               "anthropic": "Claude · paid"}
    server_has_key = any(os.getenv(PROVIDERS[p][2]) for p in ("groq", "gemini"))
    byok = True
    if config.PUBLIC_MODE:
        choices.pop("ollama")                 # a hosted app can't reach the visitor's computer
        if server_has_key:
            left = limiter().remaining(session.id)
            st.caption("Using this app's free AI key"
                       + (f" · {left} question{'s' if left != 1 else ''} left this hour" if left is not None else ""))
        byok = st.toggle("Use my own API key", value=not server_has_key,
                         help="Your key is used only for your questions in this browser tab and is never stored.")
    provider = os.getenv("LLM_PROVIDER", "groq")
    key, model = None, None
    if byok:
        provider = st.selectbox("AI provider", list(choices), format_func=choices.get,
                                index=list(choices).index(provider) if provider in choices else 0)
        if provider == "ollama":
            key = "ollama"
            st.caption("Install Ollama, then run `ollama pull qwen2.5:7b`. Nothing leaves your computer.")
        else:
            env = "ANTHROPIC_API_KEY" if provider == "anthropic" else PROVIDERS[provider][2]
            key = st.text_input(f"{choices[provider].split(' ·')[0]} API key", type="password",
                                value="" if config.PUBLIC_MODE else os.getenv(env, ""))
            if provider != "anthropic":
                st.caption(f"Free key, no card: {PROVIDERS[provider][3]}")
        model = st.text_input("Model (optional)", value="", placeholder="provider default") or None
    have_llm = bool(key) or (not byok and server_has_key)
    if not have_llm:
        st.caption("No key yet? The offline demo works without one.")

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
if ss.show_home:
    event = home(key="home", default=None)
    if event and event.get("ts") != ss.last_home_event:      # each click carries a fresh timestamp
        ss.last_home_event = event["ts"]
        if "retail_orders" not in session.datasets:
            session.load_file(SAMPLE)
        ss.pending_question = DEMO_QUESTION if event["action"] == "demo" else event.get("question")
        ss.pending_is_demo = event["action"] == "demo"
        ss.show_home = False
        st.rerun()
else:
    if st.sidebar.button("🏠 Back to home", width="stretch"):
        ss.show_home = True
        st.rerun()

tab_ask, tab_dash = st.tabs(["Ask", "Dashboard"])

with tab_ask:
    if not session.datasets:
        st.info("Upload a CSV or load the sample dataset from the sidebar to start.")
    for i, (q, run) in enumerate(ss.runs):
        with st.chat_message("user"):
            st.markdown(q)
        with st.chat_message("assistant", avatar="🐈"):
            show_run(run, i)

    demo = False
    if "retail_orders" in session.datasets and not have_llm:
        demo = st.button(f"▶ Offline demo: “{DEMO_QUESTION}”")
    question = st.chat_input("Ask Whiskers about your data", disabled=not session.datasets)
    if demo:
        question = DEMO_QUESTION
    if ss.pending_question:                      # a question picked on the home page
        question, demo = ss.pending_question, ss.pop("pending_is_demo", False) or not have_llm
        ss.pending_question = None
        if demo and question != DEMO_QUESTION:
            st.warning(f"“{question}” needs a free API key (sidebar → Model). Showing the offline demo instead.")
            question = DEMO_QUESTION

    if question:
        if demo:
            llm = demo_llm()
        elif have_llm:
            if config.PUBLIC_MODE and not byok:          # shared key: enforce the question limits
                blocked = limiter().check(session.id)
                if blocked:
                    st.warning(blocked, icon="⏳")
                    st.stop()
            try:
                # a visitor's own key must not silently fall back to the server's keys
                llm = configured_llm(provider, api_key=key if byok else None, model=model,
                                     fallback=not (config.PUBLIC_MODE and byok))
            except RuntimeError as e:
                st.error(str(e))
                st.stop()
        else:
            st.error("Add a free API key in the sidebar (Model section), or run the offline demo.")
            st.stop()
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant", avatar="🐈"):
            status = st.status("Whiskers is sniffing around…", expanded=True)
            run = None
            os.utime(session.dir)                          # keep this session's uploads alive
            try:
                for ev in Agent(session, llm, glossary).stream(question):
                    if ev["type"] == "step_start":
                        status.write(f"{TOOL_LABEL.get(ev['tool'], ev['tool'])}…")
                    elif ev["type"] == "reanalyze":
                        status.write("Verifier found unsupported numbers — re-checking…")
                    elif ev["type"] == "done":
                        run = ev["run"]
            except Exception as e:  # show API problems (no credit, bad key…) as a message, not a traceback
                msg = friendly_error(e)
                log_run(run, session.id, getattr(llm, "provider", ""), getattr(llm, "model", ""),
                        error=type(e).__name__)
                if msg is None:
                    capture(e)
                    raise
                status.update(label="Couldn't get an answer from the model", state="error", expanded=False)
                st.error(msg, icon="🙀")
                st.stop()
            log_run(run, session.id, "demo" if demo else getattr(llm, "provider", ""), getattr(llm, "model", ""))
            status.update(label=f"Done in {run.latency_ms / 1000:.1f}s", state="complete", expanded=False)
            if getattr(llm, "retired_model", None):
                st.toast(f"{llm.retired_model} was retired by the provider; used {llm.model} instead. "
                         f"Update the model in .env to skip this check.")
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
