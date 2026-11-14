"""FastAPI backend. Run:  uvicorn analyst_agent.api:app --reload

POST /sessions                         -> {"session_id"}
POST /sessions/{id}/upload             -> dataset profile + data-quality report
POST /sessions/{id}/ask  {"question"}  -> Server-Sent Events: step_start, step_end, reanalyze, done
GET  /sessions/{id}/runs/{run}/notebook -> .ipynb download
GET/POST/DELETE /glossary              -> business definitions
GET/POST/DELETE /pins, GET /pins/{id}/refresh
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel

from analyst_agent.agent.llm import AnthropicLLM
from analyst_agent.agent.loop import Agent
from analyst_agent.data.store import SESSIONS, get_or_create
from analyst_agent.export.notebook import notebook_bytes
from analyst_agent.glossary import Glossary
from analyst_agent.pins import PinBoard
from analyst_agent.tools.sql import df_preview

app = FastAPI(title="AI Data Analyst Agent", version="0.1.0")
glossary = Glossary()
pins = PinBoard()
_llm = None


def llm():
    global _llm
    if _llm is None:
        _llm = AnthropicLLM()
    return _llm


def _session(sid: str):
    if sid not in SESSIONS:
        raise HTTPException(404, "Unknown session")
    return SESSIONS[sid]


class AskBody(BaseModel):
    question: str


class DefinitionBody(BaseModel):
    table: str
    term: str
    definition: str
    sql_expression: str | None = None


class PinBody(BaseModel):
    session_id: str
    run_id: str
    title: str | None = None


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/sessions")
def create_session():
    return {"session_id": get_or_create().id}


@app.post("/sessions/{sid}/upload")
async def upload(sid: str, file: UploadFile = File(...)):
    s = _session(sid)
    suffix = Path(file.filename or "data.csv").suffix or ".csv"
    if suffix.lower() not in (".csv", ".xlsx", ".xls"):
        raise HTTPException(400, "Upload a .csv or .xlsx file")
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=s.dir) as tmp:
        tmp.write(await file.read())
    ds = s.load_file(tmp.name, name=file.filename)
    return {"table": ds.table, "profile": ds.profile}


@app.get("/sessions/{sid}/datasets")
def datasets(sid: str):
    return {t: d.profile for t, d in _session(sid).datasets.items()}


def _event_json(ev: dict) -> dict:
    if ev["type"] == "step_end":
        st = ev["step"]
        return {"type": "step_end", "tool": st.tool, "input": st.input, "error": st.error,
                "duration_ms": st.duration_ms,
                "table": df_preview(st.table, 50) if st.table is not None else None,
                "chart": json.loads(st.chart["figure_json"]) if st.chart else None}
    if ev["type"] == "done":
        r = ev["run"]
        return {"type": "done", "run_id": r.id, "answer": r.answer, "warnings": r.warnings,
                "verification": r.verification, "definitions_used": r.definitions_used,
                "steps": len(r.steps), "latency_ms": r.latency_ms,
                "tokens": {"in": r.tokens_in, "out": r.tokens_out}}
    return ev


@app.post("/sessions/{sid}/ask")
def ask(sid: str, body: AskBody):
    s = _session(sid)
    if not s.datasets:
        raise HTTPException(400, "Upload a dataset first")
    agent = Agent(s, llm(), glossary)

    def gen():
        for ev in agent.stream(body.question):
            yield f"data: {json.dumps(_event_json(ev), default=str)}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.get("/sessions/{sid}/runs/{run_id}/notebook")
def notebook(sid: str, run_id: str):
    s = _session(sid)
    run = next((r for r in s.runs if r.id == run_id), None)
    if not run:
        raise HTTPException(404, "Unknown run")
    return Response(notebook_bytes(run, s), media_type="application/x-ipynb+json",
                    headers={"Content-Disposition": f'attachment; filename="analysis_{run_id}.ipynb"'})


@app.get("/glossary")
def list_definitions(table: str | None = None):
    return glossary.list(table)


@app.post("/glossary")
def add_definition(body: DefinitionBody):
    return glossary.save(body.table, body.term, body.definition, body.sql_expression)


@app.delete("/glossary/{table}/{term}")
def delete_definition(table: str, term: str):
    return {"deleted": glossary.delete(table, term)}


@app.get("/pins")
def list_pins():
    return pins.list()


@app.post("/pins")
def add_pin(body: PinBody):
    s = _session(body.session_id)
    run = next((r for r in s.runs if r.id == body.run_id), None)
    if not run:
        raise HTTPException(404, "Unknown run")
    try:
        return pins.pin(run, body.title)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.delete("/pins/{pin_id}")
def delete_pin(pin_id: str):
    return {"deleted": pins.remove(pin_id)}


@app.get("/pins/{pin_id}/refresh")
def refresh_pin(pin_id: str, session_id: str):
    s = _session(session_id)
    pin = next((p for p in pins.list() if p["id"] == pin_id), None)
    if not pin:
        raise HTTPException(404, "Unknown pin")
    out = pins.refresh(pin, s.con)
    return {"table": df_preview(out["table"], 200),
            "chart": json.loads(out["chart"]["figure_json"]) if out["chart"] else None}
