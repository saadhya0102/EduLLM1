"""Provider-agnostic LLM client.

Supports Anthropic (via ANTHROPIC_AUTH_TOKEN + ANTHROPIC_BASE_URL proxy, or
ANTHROPIC_API_KEY) and OpenAI (OPENAI_API_KEY). We keep two providers on hand so
generation and judging can use *different* models (avoids the LLM-judge
circularity risk, PRD #7).

Usage:
    from src.llm import chat
    text = chat("Say hi", provider="anthropic", model="claude-opus-4-8")
"""
from __future__ import annotations

import json
import os
import time
from typing import Optional

# Sensible defaults; override per-call.
DEFAULT_ANTHROPIC_MODEL = "claude-opus-4-8"
DEFAULT_OPENAI_MODEL = "gpt-4o"


def _anthropic_client():
    import anthropic  # lazy import so the module loads without the SDK installed

    auth_token = os.getenv("ANTHROPIC_AUTH_TOKEN")
    base_url = os.getenv("ANTHROPIC_BASE_URL")
    api_key = os.getenv("ANTHROPIC_API_KEY")

    kwargs = {}
    if base_url:
        kwargs["base_url"] = base_url
    if auth_token and not api_key:
        # Proxy setup: Bearer auth instead of x-api-key.
        kwargs["auth_token"] = auth_token
    elif api_key:
        kwargs["api_key"] = api_key
    else:
        raise RuntimeError(
            "No Anthropic credentials found "
            "(set ANTHROPIC_AUTH_TOKEN+ANTHROPIC_BASE_URL or ANTHROPIC_API_KEY)."
        )
    return anthropic.Anthropic(**kwargs)


def _openai_client():
    from openai import OpenAI  # lazy import

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY not set.")
    return OpenAI()


def chat(
    prompt: str,
    *,
    provider: str = "anthropic",
    model: Optional[str] = None,
    system: Optional[str] = None,
    max_tokens: int = 2048,
    temperature: float = 0.0,
    retries: int = 4,
) -> str:
    """Single-turn chat. Returns the assistant text. Retries on transient errors."""
    last_err = None
    for attempt in range(retries):
        try:
            if provider == "anthropic":
                client = _anthropic_client()
                create_kwargs = dict(
                    model=model or DEFAULT_ANTHROPIC_MODEL,
                    max_tokens=max_tokens,
                    system=system or "",
                    messages=[{"role": "user", "content": prompt}],
                )
                if temperature is not None:
                    create_kwargs["temperature"] = temperature
                try:
                    resp = client.messages.create(**create_kwargs)
                except Exception as e:  # noqa: BLE001
                    # Some (reasoning) models / proxies reject `temperature`.
                    if "temperature" in str(e).lower() and "temperature" in create_kwargs:
                        create_kwargs.pop("temperature")
                        resp = client.messages.create(**create_kwargs)
                    else:
                        raise
                return "".join(
                    b.text for b in resp.content if getattr(b, "type", None) == "text"
                )
            elif provider == "openai":
                client = _openai_client()
                messages = []
                if system:
                    messages.append({"role": "system", "content": system})
                messages.append({"role": "user", "content": prompt})
                resp = client.chat.completions.create(
                    model=model or DEFAULT_OPENAI_MODEL,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    messages=messages,
                )
                return resp.choices[0].message.content or ""
            else:
                raise ValueError(f"Unknown provider: {provider}")
        except Exception as e:  # noqa: BLE001 - broad by design for retry
            last_err = e
            wait = 2 ** attempt
            print(f"[llm] {provider} attempt {attempt+1} failed: {e}; retry in {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"LLM call failed after {retries} retries: {last_err}")


def chat_json(prompt: str, **kwargs) -> dict:
    """chat() that extracts the first JSON object from the response.

    Robust to models that wrap JSON in prose or ```json fences.
    """
    raw = chat(prompt, **kwargs)
    return extract_json(raw)


def extract_json(text: str) -> dict:
    """Best-effort extraction of the first top-level JSON object in `text`."""
    # Strip code fences.
    t = text.strip()
    if "```" in t:
        # take content between first pair of fences if present
        parts = t.split("```")
        for p in parts:
            p = p.strip()
            if p.startswith("json"):
                p = p[4:].strip()
            if p.startswith("{"):
                t = p
                break
    # Find first balanced {...}.
    start = t.find("{")
    if start == -1:
        raise ValueError(f"No JSON object found in response: {text[:200]!r}")
    depth = 0
    for i in range(start, len(t)):
        if t[i] == "{":
            depth += 1
        elif t[i] == "}":
            depth -= 1
            if depth == 0:
                return json.loads(t[start : i + 1])
    raise ValueError(f"Unbalanced JSON in response: {text[:200]!r}")
