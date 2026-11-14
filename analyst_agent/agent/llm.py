"""Thin LLM wrappers so the agent loop can run on Claude, on free providers (Groq, Gemini,
a local Ollama model) or on a scripted fake (tests, demos).

The loop speaks Anthropic-style messages; OpenAICompatLLM translates to and from the OpenAI
Chat Completions format that Groq, Gemini and Ollama all accept.

Every implementation returns a plain dict:
    {"stop_reason": "tool_use" | "end_turn" | ...,
     "content": [{"type": "text", "text": ...} | {"type": "tool_use", "id", "name", "input"}],
     "usage": {"input_tokens": int, "output_tokens": int}}
"""
from __future__ import annotations

import json
import os
import re
import time
import uuid
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
        if looks_like_placeholder(key):
            raise RuntimeError("The Claude key is still the placeholder from .env.example. Paste your full key "
                               "into .env or the sidebar, then restart the app.")
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


# --------------------------------------------------------------------------- free providers
PROVIDERS = {
    # name: (base URL, default model, env var for the key, where to get a key)
    "groq": ("https://api.groq.com/openai/v1", "openai/gpt-oss-120b", "GROQ_API_KEY",
             "console.groq.com/keys"),
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai", "gemini-3.8-flash", "GEMINI_API_KEY",
               "aistudio.google.com/apikey"),
    "ollama": ("http://localhost:11434/v1", "qwen2.5:7b", None, "ollama.com/download (runs on your computer)"),
}


# If the configured model has been retired, the first of these that the provider still lists is used.
FALLBACK_MODELS = {
    "groq": ["openai/gpt-oss-120b", "qwen/qwen3.6-27b", "openai/gpt-oss-20b", "moonshotai/kimi-k2-instruct-0905"],
    "gemini": ["gemini-3.8-flash", "gemini-flash-latest", "gemini-2.5-flash"],
    "ollama": ["qwen2.5:7b", "qwen3:8b", "llama3.1:8b", "qwen2.5:3b"],
}


def looks_like_placeholder(key: str) -> bool:
    """Template values such as 'gsk_...' or 'sk-ant-...', or keys far too short to be real."""
    key = key.strip()
    return "..." in key or "your" in key.lower() or len(key) < 20


class ProviderError(RuntimeError):
    """An HTTP error from an OpenAI-compatible provider, with the provider name attached."""

    def __init__(self, provider: str, status: int, body: str):
        super().__init__(f"{provider} returned HTTP {status}: {body[:500]}")
        self.provider, self.status, self.body = provider, status, body


class OpenAICompatLLM:
    """Groq, Gemini or Ollama through their OpenAI-compatible Chat Completions endpoints."""

    def __init__(self, provider: str = "groq", api_key: str | None = None, model: str | None = None,
                 base_url: str | None = None, timeout: float = 120, client=None):
        import httpx
        if provider not in PROVIDERS:
            raise ValueError(f"provider must be one of {list(PROVIDERS)}")
        default_url, default_model, key_env, _ = PROVIDERS[provider]
        self.provider = provider
        self.base_url = (base_url or os.getenv(f"{provider.upper()}_BASE_URL") or default_url).rstrip("/")
        self.model = model or os.getenv(f"{provider.upper()}_MODEL") or default_model
        env_key = os.getenv(key_env) if key_env else None
        self.api_key = api_key or env_key or ("ollama" if provider == "ollama" else None)
        if not self.api_key:
            raise RuntimeError(f"Set {key_env} (free key from {PROVIDERS[provider][3]}) or pass api_key.")
        if provider != "ollama" and looks_like_placeholder(self.api_key):
            raise RuntimeError(f"The {provider.capitalize()} key is still the placeholder from .env.example "
                               f"(\"{self.api_key[:8]}\"). Paste your full key from {PROVIDERS[provider][3]} "
                               "into .env or the sidebar, then restart the app.")
        self.http = client or httpx.Client(timeout=timeout)
        self.retired_model = None   # set if we had to switch away from a retired model
        self.rate_retries = 3       # waits on HTTP 429; FallbackLLM sets 0 when a backup provider exists

    # ---- format translation
    @staticmethod
    def _tools(tools: list[dict]) -> list[dict]:
        return [{"type": "function", "function": {"name": t["name"], "description": t["description"],
                                                  "parameters": t["input_schema"]}} for t in tools]

    @staticmethod
    def _messages(system: str, messages: list[dict]) -> list[dict]:
        out = [{"role": "system", "content": system}]
        for m in messages:
            if isinstance(m["content"], str):
                out.append({"role": m["role"], "content": m["content"]})
                continue
            if m["role"] == "user":            # tool results (and any plain text)
                for b in m["content"]:
                    if b["type"] == "tool_result":
                        out.append({"role": "tool", "tool_call_id": b["tool_use_id"], "content": str(b["content"])})
                    elif b["type"] == "text":
                        out.append({"role": "user", "content": b["text"]})
            else:                              # assistant: text + tool calls
                text = "".join(b.get("text", "") for b in m["content"] if b["type"] == "text").strip()
                calls = [b.get("_raw") or {"id": b["id"], "type": "function",
                                           "function": {"name": b["name"], "arguments": json.dumps(b["input"])}}
                         for b in m["content"] if b["type"] == "tool_use"]
                msg = {"role": "assistant", "content": text or None}
                if calls:
                    msg["tool_calls"] = calls
                out.append(msg)
        return out

    def create(self, system, messages, tools):
        payload = {"model": self.model, "messages": self._messages(system, messages),
                   "tools": self._tools(tools), "max_tokens": MAX_TOKENS, "temperature": 0.1}
        data = self._post(payload)
        msg = data["choices"][0]["message"]
        content = []
        if msg.get("content"):
            content.append({"type": "text", "text": msg["content"]})
        for tc in msg.get("tool_calls") or []:
            try:
                args = json.loads(tc["function"].get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            content.append({"type": "tool_use", "id": tc.get("id") or f"call_{uuid.uuid4().hex[:8]}",
                            "name": tc["function"]["name"], "input": args if isinstance(args, dict) else {},
                            "_raw": tc})   # keeps provider extras (e.g. Gemini thought signatures)
        usage = data.get("usage") or {}
        has_calls = any(b["type"] == "tool_use" for b in content)
        return {"stop_reason": "tool_use" if has_calls else "end_turn",
                "content": content or [{"type": "text", "text": ""}],
                "usage": {"input_tokens": usage.get("prompt_tokens", 0),
                          "output_tokens": usage.get("completion_tokens", 0)}}

    def _post(self, payload: dict) -> dict:
        """POST with retries for free-tier rate limits (429) and malformed tool calls."""
        url = f"{self.base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        for attempt in range(4):
            r = self.http.post(url, json=payload, headers=headers)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 429 and attempt < self.rate_retries:   # free tiers: wait and retry
                wait = float(r.headers.get("retry-after", 0) or 0) or 5 * (attempt + 1)
                time.sleep(min(wait, 30))
                continue
            if r.status_code == 404 and "model" in r.text.lower() and attempt == 0:
                replacement = self._replacement_model()          # model retired: switch and retry once
                if replacement:
                    self.retired_model, self.model = self.model, replacement
                    payload = dict(payload, model=replacement)
                    continue
            if r.status_code == 400 and "tool_use_failed" in r.text:
                # The model wrote a tool call whose arguments aren't valid JSON (e.g. raw Python).
                rescued = _rescue_tool_call(r)
                if rescued:
                    return rescued
                if attempt < 2:                                   # ask it to resend the call as JSON
                    payload = dict(payload, temperature=0.0, messages=payload["messages"] + [{
                        "role": "user", "content": _JSON_NUDGE}])
                    continue
            raise ProviderError(self.provider, r.status_code, r.text)
        raise ProviderError(self.provider, 429, "rate limited")


    def available_models(self) -> list[str]:
        r = self.http.get(f"{self.base_url}/models", headers={"Authorization": f"Bearer {self.api_key}"})
        r.raise_for_status()
        return [m["id"].removeprefix("models/") for m in r.json().get("data", [])]

    def _replacement_model(self) -> str | None:
        try:
            have = set(self.available_models())
        except Exception:
            return None
        return next((m for m in FALLBACK_MODELS.get(self.provider, []) if m in have and m != self.model), None)


_JSON_NUDGE = ("Your last tool call was rejected because its arguments were not valid JSON. Call the tool again "
               'with a JSON object, e.g. {"code": "print(df.head())"} for run_python or '
               '{"query": "SELECT ..."} for run_sql. Escape newlines and quotes inside strings.')

# tools whose single required argument can be recovered from raw text
_RAW_ARG = {"run_python": "code", "run_sql": "query"}


def _rescue_tool_call(r) -> dict | None:
    """Turn Groq's `failed_generation` (a tool call with non-JSON arguments) into a normal response.

    Example failed_generation: {"name": "run_python", "arguments": import pandas as pd\nprint(df.head())}
    """
    try:
        failed = r.json()["error"].get("failed_generation") or ""
    except Exception:
        return None
    m = re.match(r'\s*\{\s*"name"\s*:\s*"([^"]+)"\s*,\s*"arguments"\s*:\s*(.*)\}\s*$', failed, re.S)
    if not m or m.group(1) not in _RAW_ARG:
        return None
    name, raw = m.group(1), m.group(2).strip()
    try:
        args = json.loads(raw)
        if isinstance(args, str):
            args = {_RAW_ARG[name]: args}
    except json.JSONDecodeError:
        if raw.startswith("{") and f'"{_RAW_ARG[name]}"' in raw:
            return None                       # a broken JSON object: safer to ask for a resend
        args = {_RAW_ARG[name]: raw}
    if not isinstance(args, dict) or not args.get(_RAW_ARG[name]):
        return None
    call = {"id": f"call_{uuid.uuid4().hex[:8]}", "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)}}
    return {"choices": [{"finish_reason": "tool_calls",
                         "message": {"role": "assistant", "content": None, "tool_calls": [call]}}],
            "usage": {}}


class FallbackLLM:
    """Try providers in order; move to the next on rate limits, outages or network errors.
    With Groq first and Gemini second, a busy free tier doesn't stop the app."""

    RETRYABLE_STATUS = {429, 500, 502, 503, 504}

    def __init__(self, llms: list):
        if not llms:
            raise RuntimeError("No AI provider is configured. Add a free GROQ_API_KEY or GEMINI_API_KEY.")
        self.llms = llms
        self.active = 0
        for llm in llms[:-1]:            # don't sit out a rate limit when another provider can answer
            if hasattr(llm, "rate_retries"):
                llm.rate_retries = 0

    @property
    def model(self):
        return getattr(self.llms[self.active], "model", "")

    @property
    def provider(self):
        return getattr(self.llms[self.active], "provider", "anthropic")

    @property
    def retired_model(self):
        return getattr(self.llms[self.active], "retired_model", None)

    def create(self, system, messages, tools):
        import httpx
        last = None
        for i in range(self.active, len(self.llms)):
            try:
                out = self.llms[i].create(system, messages, tools)
                self.active = i
                return out
            except ProviderError as e:
                if e.status not in self.RETRYABLE_STATUS:
                    raise
                last = e
            except httpx.TransportError as e:
                last = e
        raise last


def configured_llm(preferred: str | None = None, api_key: str | None = None, model: str | None = None,
                   fallback: bool = True):
    """The server's LLM: the preferred provider first, then any other free provider with a key in the
    environment as fallback. A key typed by the user (api_key) applies to the preferred provider only."""
    preferred = preferred or os.getenv("LLM_PROVIDER", "groq")
    order = [preferred] + [p for p in ("groq", "gemini") if p != preferred]
    llms, first_error = [], None
    for p in order:
        try:
            if p == preferred:
                llms.append(make_llm(p, api_key=api_key, model=model))
            elif fallback and p in PROVIDERS and os.getenv(PROVIDERS[p][2] or ""):
                llms.append(make_llm(p))
        except RuntimeError as e:
            first_error = first_error or e      # e.g. placeholder key: show this, not a generic message
    if not llms and first_error:
        raise first_error
    return FallbackLLM(llms) if len(llms) > 1 else (llms[0] if llms else FallbackLLM([]))


def make_llm(provider: str, api_key: str | None = None, model: str | None = None):
    """'anthropic', 'groq', 'gemini' or 'ollama'."""
    if provider == "anthropic":
        return AnthropicLLM(model=model or ANTHROPIC_MODEL, api_key=api_key)
    return OpenAICompatLLM(provider, api_key=api_key, model=model)


def friendly_error(e: Exception) -> str | None:
    """Plain-English message for common LLM API errors, or None if it isn't one."""
    if isinstance(e, ProviderError):
        name = e.provider.capitalize()
        if e.status in (401, 403):
            return f"{name} rejected the API key. Check you copied all of it (free key: {PROVIDERS[e.provider][3]})."
        if e.status == 429:
            return f"{name}'s free tier rate limit was reached. Wait a minute and ask again."
        if e.status == 404:
            return (f"{name} doesn't offer that model any more. Set {e.provider.upper()}_MODEL in .env or the "
                    f"sidebar's Model box to one listed at your {name} console.")
        return f"{name} returned an error ({e.status}): {e.body[:200]}"
    try:
        import httpx
        if isinstance(e, httpx.TimeoutException):
            return "The model took too long to answer. A local model may need a smaller size, e.g. qwen2.5:3b."
        if isinstance(e, httpx.TransportError):        # connection refused, proxy, DNS, TLS…
            url = str(e.request.url) if getattr(e, "_request", None) is not None else ""
            if "11434" in url:
                return "Can't reach Ollama. Install it from ollama.com, then run `ollama pull qwen2.5:7b`."
            return "Couldn't reach the AI provider. Check your internet connection (or a firewall/proxy)."
    except ImportError:
        pass
    if isinstance(e, RuntimeError) and "free key from" in str(e):
        return str(e)
    try:
        import anthropic
    except ImportError:
        return None
    text = str(e).lower()
    if isinstance(e, anthropic.AuthenticationError):
        return "The API key was rejected. Check that you copied the whole key (it starts with sk-ant-)."
    if isinstance(e, anthropic.PermissionDeniedError):
        return "This API key isn't allowed to use that model. Try ANTHROPIC_MODEL=claude-haiku-5-5 in .env."
    if isinstance(e, anthropic.BadRequestError) and "credit balance" in text:
        return ("Your Claude Console account has no API credit left. Add credit at "
                "platform.claude.com → Settings → Billing, or use the offline demo in the meantime.")
    if isinstance(e, anthropic.NotFoundError):
        return "That model name wasn't found. Check ANTHROPIC_MODEL in your .env file."
    if isinstance(e, anthropic.RateLimitError):
        return "Rate limit reached. Wait a minute and try again."
    if isinstance(e, anthropic.APIConnectionError):
        return "Couldn't reach the Claude API. Check your internet connection."
    if isinstance(e, (anthropic.InternalServerError, anthropic.APIStatusError)) and getattr(e, "status_code", 0) >= 500:
        return "The Claude API is having trouble right now. Try again shortly."
    return None


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
