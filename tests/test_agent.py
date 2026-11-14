import nbformat

from analyst_agent.agent.llm import ScriptedLLM, final, tool_call
from analyst_agent.agent.loop import Agent
from analyst_agent.demo import DEMO_QUESTION, demo_llm
from analyst_agent.export.notebook import run_to_notebook
from analyst_agent.pins import PinBoard


def test_demo_run_end_to_end(session, glossary):
    run = Agent(session, demo_llm(), glossary).ask(DEMO_QUESTION)
    assert run.status == "done"
    assert [s.tool for s in run.steps] == ["run_sql", "check_findings", "run_sql", "make_chart"]
    assert all(s.error is None for s in run.steps)
    assert run.verification["verified"]
    assert any(w["type"] == "simpsons_paradox" for w in run.warnings)
    assert run.chart is not None


def test_sql_error_is_fed_back_and_agent_recovers(session, glossary):
    llm = ScriptedLLM([
        tool_call("run_sql", {"query": "SELECT nope FROM retail_orders"}, "c1"),
        tool_call("run_sql", {"query": "SELECT COUNT(*) AS n FROM retail_orders"}, "c2"),
        final("There are 3025 rows."),
    ])
    run = Agent(session, llm, glossary).ask("How many rows?")
    assert run.steps[0].error and not run.steps[1].error
    # the error text was sent back to the model as a tool_result
    second_call_msgs = llm.calls[1]["messages"]
    assert any(isinstance(m["content"], list) and m["content"][0].get("is_error") for m in second_call_msgs)
    assert run.verification["verified"]


def test_verifier_triggers_one_reanalysis(session, glossary):
    llm = ScriptedLLM([
        tool_call("run_sql", {"query": "SELECT COUNT(*) AS n FROM retail_orders"}),
        final("There are 9999 rows."),          # invented
        final("There are 3025 rows."),          # corrected after the verifier's message
    ])
    events = list(Agent(session, llm, glossary).stream("How many rows?"))
    assert any(e["type"] == "reanalyze" for e in events)
    run = events[-1]["run"]
    assert run.answer == "There are 3025 rows." and run.verification["verified"]


def test_auto_trap_check_without_check_findings(session, glossary):
    llm = ScriptedLLM([
        tool_call("run_sql", {"query": "SELECT channel, AVG(order_value) AS v FROM retail_orders "
                                       "WHERE channel <> 'B2B' GROUP BY channel"}),
        final("Store averages 1496.32 and Online 1109.38."),
    ])
    run = Agent(session, llm, glossary).ask("Which channel has a bigger AOV?")
    paradox = [w for w in run.warnings if w["type"] == "simpsons_paradox"]
    assert paradox and paradox[0].get("auto")


def test_glossary_learning_flows_into_next_prompt(session, glossary):
    llm = ScriptedLLM([
        tool_call("save_definition", {"term": "Revenue", "definition": "order value minus refunds",
                                      "sql_expression": "SUM(order_value - refund_amount)"}),
        final("Saved: revenue now excludes refunds."),
        final("ok"),
    ])
    agent = Agent(session, llm, glossary)
    agent.ask("No, revenue should exclude refunds")
    assert glossary.list("retail_orders")[0]["term"] == "revenue"
    agent.ask("What is total revenue?")
    assert "SUM(order_value - refund_amount)" in llm.calls[-1]["system"]


def test_step_limit(session, glossary):
    llm = ScriptedLLM([tool_call("run_sql", {"query": "SELECT 1 AS x"}, f"c{i}") for i in range(12)]
                      + [final("done")])
    run = Agent(session, llm, glossary, max_steps=3).ask("loop")
    assert len(run.steps) == 3


def test_notebook_export_and_pin(session, glossary, tmp_path):
    run = Agent(session, demo_llm(), glossary).ask(DEMO_QUESTION)
    nb = run_to_notebook(run, session, data_dir="sample_data")
    nbformat.validate(nb)
    src = "\n".join(c.source for c in nb.cells)
    assert "con.execute" in src and "px.bar" in src and "ttest_ind" in src

    board = PinBoard(tmp_path / "pins.json")
    pin = board.pin(run)
    out = board.refresh(pin, session.con)
    assert len(out["table"]) == 8 and out["chart"] is not None


def test_friendly_error_for_no_credit():
    import anthropic
    import httpx

    from analyst_agent.agent.llm import friendly_error

    req = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    body = {"type": "error", "error": {"type": "invalid_request_error",
                                       "message": "Your credit balance is too low to access the Anthropic API."}}
    err = anthropic.BadRequestError("credit balance is too low", response=httpx.Response(400, request=req, json=body),
                                    body=body)
    assert "no API credit" in friendly_error(err)
    assert friendly_error(ValueError("something else")) is None
