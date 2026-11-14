"""OpenAICompatLLM (Groq / Gemini / Ollama) against a fake OpenAI-style server."""
import json

import httpx

from analyst_agent.agent.llm import OpenAICompatLLM, ProviderError, friendly_error
from analyst_agent.agent.loop import Agent


def _tool_call(cid, name, args):
    return {"choices": [{"finish_reason": "tool_calls", "message": {"role": "assistant", "content": None,
            "tool_calls": [{"id": cid, "type": "function",
                            "function": {"name": name, "arguments": json.dumps(args)}}]}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 20}}


def _text(t):
    return {"choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": t}}],
            "usage": {"prompt_tokens": 50, "completion_tokens": 10}}


def _fake(replies, seen, status_first=None):
    replies = list(replies)

    def handler(request):
        body = json.loads(request.content)
        seen.append(body)
        if status_first:
            code = status_first.pop(0)
            return httpx.Response(code, headers={"retry-after": "0"}, json={"error": "x"})
        return httpx.Response(200, json=replies.pop(0))
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_agent_runs_on_openai_compatible_provider(session, glossary):
    seen = []
    client = _fake([
        _tool_call("call_1", "run_sql", {"query": "SELECT COUNT(*) AS n FROM retail_orders"}),
        _text("There are 3025 rows."),
    ], seen)
    llm = OpenAICompatLLM("groq", api_key="test", client=client)
    run = Agent(session, llm, glossary).ask("How many rows?")

    assert run.answer == "There are 3025 rows." and run.verification["verified"]
    assert run.tokens_in == 150
    first, second = seen
    assert first["tools"][0]["type"] == "function" and first["messages"][0]["role"] == "system"
    # the second request must carry the assistant tool call and a matching tool result
    roles = [m["role"] for m in second["messages"]]
    assert roles[-2:] == ["assistant", "tool"]
    assert second["messages"][-1]["tool_call_id"] == second["messages"][-2]["tool_calls"][0]["id"] == "call_1"


def test_rate_limit_is_retried(session, glossary):
    seen = []
    client = _fake([_text("hello")], seen, status_first=[429])
    llm = OpenAICompatLLM("groq", api_key="test", client=client)
    assert llm.create("sys", [{"role": "user", "content": "hi"}], [])["content"][0]["text"] == "hello"
    assert len(seen) == 2


def test_friendly_messages():
    assert "rate limit" in friendly_error(ProviderError("groq", 429, ""))
    assert "rejected the API key" in friendly_error(ProviderError("gemini", 401, ""))


def test_ollama_needs_no_key():
    assert OpenAICompatLLM("ollama").api_key == "ollama"


def test_network_errors_are_friendly():
    req = httpx.Request("POST", "http://localhost:11434/v1/chat/completions")
    assert "Ollama" in friendly_error(httpx.ConnectError("refused", request=req))
    req2 = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    assert "internet" in friendly_error(httpx.ProxyError("403", request=req2))
