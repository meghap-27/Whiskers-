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
from analyst_agent.suggest import suggest_folders  # noqa: E402
from analyst_agent.telemetry import capture, init_error_tracking, log_run  # noqa: E402

SAMPLE = Path(__file__).resolve().parent.parent / "sample_data" / "retail_orders.csv"
# the home page (scanline cat hero + FolderFloat shelf) is a small HTML/JS component;
# it sends back {"action": "demo"} or {"action": "ask", "question": ...}
home = components.declare_component("analyst_home", path=str(Path(__file__).resolve().parent / "home_component"))
TOOL_LABEL = {"run_sql": "Query database", "run_python": "Run code", "make_chart": "Generate chart",
              "check_findings": "Check findings", "save_definition": "Save definition"}

STATIC = Path(__file__).resolve().parent / "static"
WHISKERS_AVATAR = str(STATIC / "img" / "whiskers.png")
YOU_AVATAR = str(STATIC / "img" / "you.png")
st.set_page_config(page_title="Whiskers, a data analyst", page_icon=str(STATIC / "img" / "favicon.png"), layout="wide")
st.markdown("""
<style>
/* cardboard grain on the desk */
.stApp {
  background-color: #C9A37C;
  background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='240' height='240'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='.85' numOctaves='3' stitchTiles='stitch'/%3E%3CfeColorMatrix values='0 0 0 0 .35 0 0 0 0 .22 0 0 0 0 .1 0 0 0 .09 0'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E");
}
.block-container { padding-top: 1.2rem; max-width: 1180px; }

/* answers and questions are sheets of paper resting on the desk */
[data-testid="stChatMessage"] {
  background: #FBF7F0; border: 0; border-radius: 6px; padding: 18px 20px;
  box-shadow: 0 1px 0 rgba(43,27,18,.10), 0 14px 26px -14px rgba(60,35,15,.55);
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarCustom"] img[src*="you"]) { background: #EFE3D2; }

/* tactile buttons: they press down */
.stButton > button, .stDownloadButton > button {
  font-weight: 700; border: 1.5px solid #2B1B12; background: #FBF7F0; color: #2B1B12;
  box-shadow: 0 3px 0 #2B1B12; transform: translateY(0); transition: transform .08s ease, box-shadow .08s ease;
}
.stButton > button:hover, .stDownloadButton > button:hover { border-color: #2B1B12; color: #2B1B12; background: #fff; }
.stButton > button:active, .stDownloadButton > button:active { transform: translateY(3px); box-shadow: 0 0 0 #2B1B12; }
.stButton > button:focus-visible, .stDownloadButton > button:focus-visible { outline: 3px solid #3557A0; outline-offset: 2px; }

/* the chat box is the one place you type: give it weight */
[data-testid="stChatInput"] { border: 2px solid #2B1B12 !important; border-radius: 8px !important;
  box-shadow: 0 4px 0 #2B1B12; background: #FBF7F0; }
[data-testid="stBottomBlockContainer"], [data-testid="stBottom"], [data-testid="stBottom"] > div,
[data-testid="stHeader"], [data-testid="stMain"] { background: transparent; }

/* tabs: ink underline */
.stTabs [data-baseweb="tab-list"] { gap: 18px; }
.stTabs [data-baseweb="tab"] { font-weight: 700; color: #5E4636; }
.stTabs [aria-selected="true"] { color: #2B1B12; }
.stTabs [data-baseweb="tab-highlight"] { background: #2B1B12; height: 3px; }

/* sidebar sits on darker card stock */
[data-testid="stSidebar"] { background-color: #B48C63; }
[data-testid="stSidebar"] h2 { font-size: 1.05rem; margin-top: .6rem; }
[data-testid="stSidebar"] [data-testid="stExpander"] { background: #FBF7F0; border-radius: 6px; border: 0;
  box-shadow: 0 8px 18px -12px rgba(43,27,18,.6); }
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] { background: #FBF7F0; border: 1.5px dashed #2B1B12; }

/* notes the agent raises read like margin notes, not traffic signs */
[data-testid="stAlert"] { border-radius: 4px; }
[data-testid="stExpander"] details { border-radius: 6px; }
@media (prefers-reduced-motion: reduce) {
  .stButton > button, .stDownloadButton > button { transition: none; }
}
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


def demo_session() -> Session:
    """The scripted demo runs on the retail sample in its own workspace, so it never mixes
    with (or replaces) the visitor's uploaded file."""
    if "demo_session" not in ss:
        ss.demo_session = Session()
        ss.demo_session.load_file(SAMPLE)
    return ss.demo_session
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

    for ds in list(session.datasets.values()):
        p = ds.profile
        main = ", latest upload" if ds.table == session.main_table and len(session.datasets) > 1 else ""
        with st.expander(f"{ds.name} ({p['row_count']:,} rows, {len(p['columns'])} columns{main})", expanded=True):
            if st.button("Remove this file", key=f"rm_{ds.table}"):
                session.remove(ds.table)
                if ss.get("rejected") == ds.name:
                    ss.rejected = None
                st.rerun()
            issues = p["quality"]["issues"]
            if issues:
                st.markdown(f"**{len(issues)} data-quality issue{'s' if len(issues) != 1 else ''}**")
                for i in issues:
                    st.warning(i["message"])
            else:
                st.success("No data-quality issues found")
            st.dataframe([{"column": c["name"], "type": c["dtype"], "nulls %": c["null_pct"],
                           "unique": c["unique"]} for c in p["columns"]], hide_index=True)

    st.header("Model")
    choices = {"groq": "Groq (free)", "gemini": "Gemini (free)", "ollama": "Ollama (free, runs on your computer)",
               "anthropic": "Claude (paid)"}
    server_has_key = any(os.getenv(PROVIDERS[p][2]) for p in ("groq", "gemini"))
    byok = True
    if config.PUBLIC_MODE:
        choices.pop("ollama")                 # a hosted app can't reach the visitor's computer
        if server_has_key:
            left = limiter().remaining(session.id)
            st.caption("Using this app's free AI key."
                       + (f" You have {left} question{'s' if left != 1 else ''} left this hour." if left is not None
                          else ""))
        byok = st.toggle("Use my own API key", value=not server_has_key,
                         help="Your key is used only for your questions in this browser tab and is never stored.")
    provider = os.getenv("LLM_PROVIDER", "groq")
    key, model = None, None
    if byok:
        provider = st.selectbox("AI provider", list(choices), format_func=choices.get,
                                index=list(choices).index(provider) if provider in choices else 0)
        name = choices[provider].split(" (")[0]
        if provider == "ollama":
            key = "ollama"
            st.caption("Install Ollama, then run `ollama pull qwen2.5:7b`. Nothing leaves your computer.")
        else:
            env = "ANTHROPIC_API_KEY" if provider == "anthropic" else PROVIDERS[provider][2]
            # A key from .env / secrets is used server-side only and never sent to the page.
            saved = "" if config.PUBLIC_MODE else os.getenv(env, "")
            if saved:
                st.caption(f"Using the {name} key from your .env file.")
            typed = st.text_input(f"{name} API key" + (" (to use a different one)" if saved else ""),
                                  type="password", value="", placeholder="paste key" if not saved else "optional")
            key = typed.strip() or saved
            if not key and provider != "anthropic":
                st.caption(f"Free key, no card: {PROVIDERS[provider][3]}")
        model = st.text_input("Model (optional)", value="", placeholder="provider default") or None
        default_model = (os.getenv("ANTHROPIC_MODEL", config.ANTHROPIC_MODEL) if provider == "anthropic"
                         else os.getenv(f"{provider.upper()}_MODEL") or PROVIDERS[provider][1])
        st.caption(f"Model in use: `{model or default_model}`")
    have_llm = bool(key) or (not byok and server_has_key)
    if config.ALLOW_PYTHON and config.PYTHON_SANDBOX == "e2b":
        st.caption("Python analysis runs in an isolated E2B sandbox.")
    elif not config.ALLOW_PYTHON:
        st.caption("Python is off on this server; the agent answers with SQL and statistics checks.")
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
        if c2.button("Delete", key=f"del_{d['dataset']}_{d['term']}"):
            glossary.delete(d["dataset"], d["term"])
            st.rerun()


# ------------------------------------------------------------------ helpers
def show_run(run, i, work: Session):
    for w in run.warnings:
        st.warning(w["message"])
    st.markdown(run.answer)
    v = run.verification
    if v.get("caveat"):
        st.info(v["caveat"])
    if v and not v.get("verified", True):
        st.error("Could not trace these numbers to a tool result: " + ", ".join(v["unverified_numbers"]))
    if run.definitions_used:
        st.caption("Definitions applied: " + ", ".join(run.definitions_used))
    if run.chart:
        st.plotly_chart(pio.from_json(run.chart["figure_json"]), width="stretch", key=f"chart_{i}",
                        theme=None)
    if run.table is not None:
        st.dataframe(run.table.head(200), hide_index=True)

    with st.expander(f"How Whiskers got this ({len(run.steps)} steps, {run.latency_ms / 1000:.1f} s)"):
        for s in run.steps:
            st.markdown(f"**{s.idx}. {TOOL_LABEL.get(s.tool, s.tool)}** ({s.duration_ms} ms)"
                        + (f": {s.input.get('purpose')}" if s.input.get("purpose") else ""))
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
    c1.download_button("Notebook", notebook_bytes(run, work, data_dir="."), f"analysis_{run.id}.ipynb",
                       key=f"nb_{i}")
    if c2.button("Pin to dashboard", key=f"pin_{i}"):
        try:
            pins.pin(run)
            st.toast("Pinned")
        except ValueError as e:
            st.toast(str(e))


# ------------------------------------------------------------------ main
if ss.show_home:
    main_ds = session.datasets.get(session.main_table) if session.main_table else None
    folders = suggest_folders(main_ds.profile, main_ds.name) if main_ds else None
    event = home(key="home", default=None, folders=folders, dataset=main_ds.name if main_ds else None)
    if event and event.get("ts") != ss.last_home_event:      # each click carries a fresh timestamp
        ss.last_home_event = event["ts"]
        if event["action"] == "demo":
            ss.pending_question, ss.pending_is_demo = DEMO_QUESTION, True
        else:
            if not session.datasets:                         # nothing uploaded yet: use the sample
                session.load_file(SAMPLE)
            ss.pending_question, ss.pending_is_demo = event.get("question"), False
        ss.show_home = False
        st.rerun()
else:
    if st.sidebar.button("Back to home", width="stretch"):
        ss.show_home = True
        st.rerun()

tab_ask, tab_dash = st.tabs(["Ask", "Dashboard"])

with tab_ask:
    if ss.get("notice"):
        st.info(ss.pop("notice"))
    if not session.datasets:
        st.info("Upload a CSV or load the sample dataset from the sidebar to start.")
    for i, (q, run, was_demo) in enumerate(ss.runs):
        with st.chat_message("user", avatar=YOU_AVATAR):
            st.markdown(q + ("  \n*(offline demo on the sample retail data)*" if was_demo else ""))
        with st.chat_message("assistant", avatar=WHISKERS_AVATAR):
            show_run(run, i, demo_session() if was_demo else session)

    demo = False
    if not have_llm:
        demo = st.button(f"Run the offline demo on sample data: “{DEMO_QUESTION}”")
    question = st.chat_input("Ask Whiskers about your data", disabled=not session.datasets)
    if demo:
        question = DEMO_QUESTION
    if ss.pending_question:                      # a question picked on the home page
        question, demo = ss.pending_question, ss.pop("pending_is_demo", False) or not have_llm
        ss.pending_question = None
        if demo and question != DEMO_QUESTION:
            ss.notice = (f"“{question}” needs a free AI key (sidebar → Model) to run on your file. "
                         "Showing the offline demo on sample data instead.")
            question = DEMO_QUESTION

    if question:
        work = demo_session() if demo else session       # the demo never touches the user's data
        if demo:
            llm = demo_llm()
        elif have_llm:
            if config.PUBLIC_MODE and not byok:          # shared key: enforce the question limits
                blocked = limiter().check(session.id)
                if blocked:
                    st.warning(blocked)
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
        with st.chat_message("user", avatar=YOU_AVATAR):
            st.markdown(question)
        with st.chat_message("assistant", avatar=WHISKERS_AVATAR):
            status = st.status("Analysing your data", expanded=True)
            run = None
            os.utime(work.dir)                             # keep this session's uploads alive
            try:
                for ev in Agent(work, llm, glossary).stream(question):
                    if ev["type"] == "step_start":
                        status.write(f"{TOOL_LABEL.get(ev['tool'], ev['tool'])}…")
                    elif ev["type"] == "reanalyze":
                        status.write("Some numbers weren't backed by a query. Checking them again.")
                    elif ev["type"] == "done":
                        run = ev["run"]
            except Exception as e:  # show API problems (no credit, bad key…) as a message, not a traceback
                msg = friendly_error(e)
                log_run(run, session.id, getattr(llm, "provider", ""), getattr(llm, "model", ""),
                        error=type(e).__name__)
                if msg is None:
                    capture(e)
                    raise
                status.update(label="No answer from the AI provider", state="error", expanded=False)
                st.error(msg)
                st.stop()
            log_run(run, session.id, "demo" if demo else getattr(llm, "provider", ""), getattr(llm, "model", ""))
            status.update(label=f"Done in {run.latency_ms / 1000:.1f}s", state="complete", expanded=False)
            if getattr(llm, "retired_model", None):
                st.toast(f"{llm.retired_model} was retired by the provider; used {llm.model} instead. "
                         f"Update the model in .env to skip this check.")
        ss.runs.append((question, run, demo))
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
                                key=f"pinchart_{p['id']}", theme=None)
            else:
                st.dataframe(out["table"], hide_index=True)
        except Exception as e:
            st.caption(f"Load the dataset this pin uses to refresh it ({e}).")
        if st.button("Remove", key=f"unpin_{p['id']}"):
            pins.remove(p["id"])
            st.rerun()
