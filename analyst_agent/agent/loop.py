"""The agent orchestrator: plan -> call tool -> observe -> repeat -> verify -> answer (PRD FR3, FR7)."""
from __future__ import annotations

import json
import time
import uuid
from collections.abc import Iterator
from dataclasses import dataclass, field

import pandas as pd
import sqlglot
from sqlglot import exp

from analyst_agent.agent.llm import LLM
from analyst_agent.agent.prompts import SYSTEM_TEMPLATE, TOOLS
from analyst_agent.analyst.stats import compare_groups
from analyst_agent.analyst.traps import check_comparison
from analyst_agent.analyst.verify import verify_answer
from analyst_agent.config import ALLOW_PYTHON, MAX_STEPS, RUN_TIMEOUT_S, TOOL_OUTPUT_CHARS
from analyst_agent.data.store import Session
from analyst_agent.glossary import Glossary
from analyst_agent.tools.chart import make_chart
from analyst_agent.tools.python_exec import run_python
from analyst_agent.tools.sql import df_preview, run_sql


@dataclass
class Step:
    idx: int
    tool: str
    input: dict
    output: str = ""
    error: str | None = None
    duration_ms: int = 0
    table: pd.DataFrame | None = None     # result table produced by this step, if any
    chart: dict | None = None             # {"figure_json", "code"}
    extra: dict = field(default_factory=dict)


@dataclass
class Run:
    question: str
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:10])
    steps: list[Step] = field(default_factory=list)
    answer: str = ""
    warnings: list[dict] = field(default_factory=list)
    verification: dict = field(default_factory=dict)
    definitions_used: list[str] = field(default_factory=list)
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: int = 0
    status: str = "running"

    @property
    def chart(self):
        return next((s.chart for s in reversed(self.steps) if s.chart), None)

    @property
    def table(self):
        return next((s.table for s in reversed(self.steps) if s.table is not None and s.tool != "make_chart"), None)


class Agent:
    def __init__(self, session: Session, llm: LLM, glossary: Glossary | None = None, max_steps: int = MAX_STEPS,
                 allow_python: bool = ALLOW_PYTHON, timeout_s: int = RUN_TIMEOUT_S):
        self.session = session
        self.llm = llm
        self.glossary = glossary or Glossary()
        self.max_steps = max_steps
        self.allow_python = allow_python
        self.timeout_s = timeout_s
        self.tools = [t for t in TOOLS if allow_python or t["name"] != "run_python"]
        self._last_table: pd.DataFrame | None = None

    # ------------------------------------------------------------------ public
    def ask(self, question: str) -> Run:
        run = None
        for event in self.stream(question):
            if event["type"] == "done":
                run = event["run"]
        return run

    def stream(self, question: str) -> Iterator[dict]:
        """Yields progress events; the last one is {"type": "done", "run": Run}."""
        run = Run(question=question)
        t0 = time.time()
        system = self._system_prompt()
        run.definitions_used = [d["term"] for d in self.glossary.list() if d["dataset"] in self.session.datasets]
        messages = self.session.history + [{"role": "user", "content": question}]
        reanalyzed = False

        for _ in range(self.max_steps + 2):
            if time.time() - t0 > self.timeout_s:
                run.answer = (f"I stopped after {self.timeout_s}s without a complete answer. "
                              "Try a narrower question, e.g. one metric for one time period.")
                run.status = "timeout"
                break
            resp = self.llm.create(system, messages, self.tools)
            run.tokens_in += resp["usage"]["input_tokens"]
            run.tokens_out += resp["usage"]["output_tokens"]
            # the API rejects empty text blocks, so drop them from the history
            kept = [b for b in resp["content"] if b["type"] != "text" or b["text"].strip()]
            messages.append({"role": "assistant", "content": kept or [{"type": "text", "text": "(empty)"}]})
            text = "".join(b["text"] for b in resp["content"] if b["type"] == "text").strip()
            calls = [b for b in resp["content"] if b["type"] == "tool_use"]

            if resp["stop_reason"] == "tool_use" and calls:
                if len(run.steps) >= self.max_steps:
                    results = [{"type": "tool_result", "tool_use_id": c["id"], "is_error": True,
                                "content": "Step limit reached. Answer now with what you have."} for c in calls]
                    messages.append({"role": "user", "content": results})
                    continue
                if text:
                    yield {"type": "thinking", "text": text}
                results = []
                for c in calls:
                    yield {"type": "step_start", "tool": c["name"], "input": c["input"]}
                    step = self._execute(len(run.steps) + 1, c["name"], c["input"])
                    run.steps.append(step)
                    yield {"type": "step_end", "step": step}
                    results.append({"type": "tool_result", "tool_use_id": c["id"], "content": step.output,
                                    **({"is_error": True} if step.error else {})})
                messages.append({"role": "user", "content": results})
                continue

            # final answer: verify before showing
            check = self._verify(text, run)
            if not check["verified"] and not reanalyzed:
                reanalyzed = True
                yield {"type": "reanalyze", "unverified": check["unverified_numbers"]}
                messages.append({"role": "user", "content":
                                 "Verifier: these numbers in your answer do not appear in any tool result: "
                                 f"{', '.join(check['unverified_numbers'])}. Recompute them with a tool or remove "
                                 "them, then give the corrected final answer."})
                continue
            run.answer = text
            run.verification = check
            break
        else:
            run.answer = run.answer or "I could not finish this analysis within the step limit."

        run.warnings += self._auto_trap_check(run)
        run.warnings = _dedupe(run.warnings)
        if run.status != "timeout":
            run.status = "done"
        run.latency_ms = int((time.time() - t0) * 1000)
        # keep the conversation for follow-ups (text only, to keep the context small)
        self.session.history += [{"role": "user", "content": question},
                                 {"role": "assistant", "content": run.answer or "(no answer)"}]
        self.session.runs.append(run)
        yield {"type": "done", "run": run}

    # ------------------------------------------------------------------ internals
    def _system_prompt(self) -> str:
        prompt = SYSTEM_TEMPLATE.format(schema=self.session.schema_text(),
                                        glossary=self.glossary.prompt_text(list(self.session.datasets)))
        if not self.allow_python:
            prompt += ("\nNOTE: run_python is disabled in this deployment. Do all analysis with run_sql "
                       "(DuckDB has corr(), stddev(), quantile_cont(), regr_slope(), etc.) and check_findings.\n")
        return prompt

    @property
    def _main_table(self) -> str | None:
        return next(iter(self.session.datasets), None)

    def _execute(self, idx: int, name: str, args: dict) -> Step:
        step = Step(idx=idx, tool=name, input=args)
        t = time.time()
        try:
            if name == "run_sql":
                df = run_sql(self.session.con, args["query"])
                step.table, self._last_table = df, df
                step.output = json.dumps(df_preview(df), default=str)
            elif name == "run_python" and not self.allow_python:
                raise PermissionError("run_python is disabled here; use run_sql (DuckDB) instead.")
            elif name == "run_python":
                res = run_python(args["code"], {k: d.path for k, d in self.session.datasets.items()},
                                 last_result=self._last_table, main_table=self._main_table)
                if not res.get("ok"):
                    raise RuntimeError(res.get("error", "Python failed"))
                if res.get("output_df") is not None:
                    step.table = self._last_table = res["output_df"]
                payload = {"stdout": res.get("stdout", ""), "last_value": res.get("last_value")}
                if step.table is not None:
                    payload["output_table"] = df_preview(step.table, 50)
                step.output = json.dumps(payload, default=str)
            elif name == "make_chart":
                chart = make_chart(self._last_table, args["chart_type"], args["x"], args.get("y"),
                                   args.get("color"), args.get("title", ""))
                step.chart, step.table = chart, self._last_table
                step.output = json.dumps({"ok": True, "chart": args["chart_type"],
                                          "rows_plotted": len(self._last_table)})
            elif name == "check_findings":
                sql = args.get("sql") or f'SELECT * FROM "{self._main_table}"'
                df = run_sql(self.session.con, sql)
                test = compare_groups(df, args["value_column"], args["group_column"])
                warns = check_comparison(df, args["value_column"], args["group_column"])
                step.extra = {"test": test, "warnings": warns}
                step.output = json.dumps({"significance": test, "warnings": [w["message"] for w in warns]},
                                         default=str)
            elif name == "save_definition":
                table = args.get("table") or self._main_table
                saved = self.glossary.save(table, args["term"], args["definition"], args.get("sql_expression"))
                step.extra = {"saved": saved}
                step.output = json.dumps({"saved": saved})
            else:
                raise ValueError(f"Unknown tool {name}")
        except Exception as e:  # errors go back to the LLM so it can correct itself
            step.error = f"{type(e).__name__}: {e}"
            step.output = step.error
        step.duration_ms = int((time.time() - t) * 1000)
        step.output = step.output[:TOOL_OUTPUT_CHARS]
        return step

    def _verify(self, text: str, run: Run) -> dict:
        texts = [s.output for s in run.steps]
        texts += [json.dumps(s.extra, default=str) for s in run.steps if s.extra]
        texts += [s.table.to_csv(index=False) for s in run.steps if s.table is not None]
        texts.append(self.session.schema_text())
        # observational data never establishes cause, so causal wording always gets a caveat
        return verify_answer(text, texts, causal_tested=False)

    def _auto_trap_check(self, run: Run) -> list[dict]:
        """Collect trap warnings from check_findings, and run them ourselves if the agent
        compared averages across groups without calling check_findings."""
        warnings = [w for s in run.steps if s.tool == "check_findings" for w in s.extra.get("warnings", [])]
        if any(s.tool == "check_findings" and not s.error for s in run.steps):
            return warnings
        for s in run.steps:
            if s.tool != "run_sql" or s.error:
                continue
            target = _avg_by_group(s.input.get("query", ""))
            if not target:
                continue
            table, value_col, group_col, row_sql = target
            if table not in self.session.datasets:
                continue
            try:
                df = run_sql(self.session.con, row_sql)
                for w in check_comparison(df, value_col, group_col):
                    w["auto"] = True
                    warnings.append(w)
            except Exception:
                pass
        return warnings


def _avg_by_group(query: str):
    """(table, value_col, group_col, row_level_sql) for queries like
    SELECT g, AVG(v) FROM t WHERE ... GROUP BY g. The row-level SQL keeps the WHERE filter."""
    try:
        tree = sqlglot.parse_one(query, read="duckdb")
    except Exception:
        return None
    group = tree.args.get("group")
    avgs = list(tree.find_all(exp.Avg))
    tables = list(tree.find_all(exp.Table))
    if not group or len(group.expressions) != 1 or not avgs or len(tables) != 1:
        return None
    g = group.expressions[0]
    v = avgs[0].this
    if not isinstance(g, exp.Column) or not isinstance(v, exp.Column):
        return None
    rows = tree.copy()
    rows.set("expressions", [exp.Star()])
    for key in ("group", "having", "order", "limit"):
        rows.set(key, None)
    return tables[0].name, v.name, g.name, rows.sql(dialect="duckdb")


def _dedupe(warnings: list[dict]) -> list[dict]:
    seen, out = set(), []
    for w in warnings:
        if w["message"] not in seen:
            seen.add(w["message"])
            out.append(w)
    return out
