"""Thin LLM wrapper so the agent loop can run against Claude or a scripted fake (tests, demos).

Every implementation returns a plain dict:
    {"stop_reason": "tool_use" | "end_turn" | ...,
     "content": [{"type": "text", "text": ...} | {"type": "tool_use", "id", "name", "input"}],
     "usage": {"input_tokens": int, "output_tokens": int}}
"""
from __future__ import annotations

import os
from typing import Protocol

from analyst_agent.config import ANTHROPIC_MODEL, MAX_TOKENS


class LLM(Protocol):
    def create(self, system: str, messages: list[dict], tools: list[dict]) -> dict: ...


class AnthropicLLM:
    def __init__(self, model: str = ANTHROPIC_MODEL, api_key: str | None = None):
        import anthropic
        key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError("Set ANTHROPIC_API_KEY (see .env.example) or pass api_key.")
        self.client = anthropic.Anthropic(api_key=key)
        self.model = model

    def create(self, system, messages, tools):
        resp = self.client.messages.create(model=self.model, max_tokens=MAX_TOKENS,
                                           system=system, messages=messages, tools=tools)
        data = resp.model_dump()
        content = []
        for b in data["content"]:
            if b["type"] == "text":
                content.append({"type": "text", "text": b["text"]})
            elif b["type"] == "tool_use":
                content.append({"type": "tool_use", "id": b["id"], "name": b["name"], "input": b["input"]})
        return {"stop_reason": data["stop_reason"], "content": content,
                "usage": {"input_tokens": data["usage"]["input_tokens"],
                          "output_tokens": data["usage"]["output_tokens"]}}


class ScriptedLLM:
    """Replays a fixed list of responses. Used by tests and the offline demo."""

    def __init__(self, responses: list[dict]):
        self.responses = list(responses)
        self.calls: list[dict] = []

    def create(self, system, messages, tools):
        self.calls.append({"system": system, "messages": [dict(m) for m in messages]})
        if not self.responses:
            return {"stop_reason": "end_turn", "content": [{"type": "text", "text": "(no more scripted replies)"}],
                    "usage": {"input_tokens": 0, "output_tokens": 0}}
        r = self.responses.pop(0)
        r.setdefault("usage", {"input_tokens": 0, "output_tokens": 0})
        return r


def tool_call(name: str, args: dict, call_id: str | None = None, text: str = "") -> dict:
    """Helper for building ScriptedLLM responses."""
    content = [{"type": "text", "text": text}] if text else []
    content.append({"type": "tool_use", "id": call_id or f"call_{name}", "name": name, "input": args})
    return {"stop_reason": "tool_use", "content": content}


def final(text: str) -> dict:
    return {"stop_reason": "end_turn", "content": [{"type": "text", "text": text}]}
