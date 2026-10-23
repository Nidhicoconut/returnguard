"""LLM proposals. Providers: anthropic | openai (any OpenAI-compatible endpoint) | rules (offline).

The LLM never has the last word (see decision.py), so if no provider is configured, or a call fails,
the app still works using the rule engine alone.
"""
import json
import logging
import os
import re

import httpx

from . import config

log = logging.getLogger("returnguard.llm")

SYSTEM = """You are the decision assistant at a retail returns desk. You are given structured FACTS about a
return request: the order, days since delivery, the customer's stated reason, evidence extracted from the
customer's photo by a vision model, and the policy clauses that apply.

Rules:
- Use ONLY the facts and clauses provided. Never invent order details, amounts or policy.
- Cite clause ids (like "R2") for every reason. Cite only ids that appear in the provided clauses.
- decision must be one of: approve, reject, escalate. Choose escalate when evidence is unclear or contradictory.
- Be brief and plain in the customer_message.

Reply with a single JSON object and nothing else:
{"decision": "approve|reject|escalate", "confidence": 0.0-1.0, "summary": "one or two sentences",
 "reasons": ["..."], "cited_clauses": ["R2"], "customer_message": "..."}"""


def provider() -> str:
    p = config.LLM_PROVIDER
    if p == "auto":
        if os.getenv("ANTHROPIC_API_KEY"):
            return "anthropic"
        if config.OPENAI_BASE_URL:
            return "openai"
        return "rules"
    return p


def _parse_json(text: str) -> dict:
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("no JSON object in model reply")
    return json.loads(text[start:end + 1])


def _user_prompt(facts: dict) -> str:
    slim = {
        "order": facts.get("order"),
        "days_since_delivery": facts.get("days_since_delivery"),
        "return_window_days": facts["return_window_days"],
        "customer_reason": facts["reason"],
        "customer_claims_defect": facts["claims_defect"],
        "photo_evidence": {k: facts["vision"].get(k) for k in ("label", "confidence", "defect_type", "note")} if facts.get("vision") else None,
        "policy_clauses": facts["clauses"],
    }
    return "FACTS:\n" + json.dumps(slim, indent=2, default=str)


def _call_anthropic(prompt: str) -> str:
    import anthropic

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY on the server
    msg = client.messages.create(model=config.LLM_MODEL, max_tokens=700, system=SYSTEM,
                                 messages=[{"role": "user", "content": prompt}])
    return "".join(b.text for b in msg.content if b.type == "text")


def _call_openai_compatible(prompt: str) -> str:
    if not config.OPENAI_BASE_URL:
        raise RuntimeError("OPENAI_BASE_URL is not set")
    headers = {"Authorization": f"Bearer {os.getenv('OPENAI_API_KEY', 'none')}"}
    body = {"model": config.OPENAI_MODEL, "temperature": 0, "max_tokens": 700,
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]}
    r = httpx.post(f"{config.OPENAI_BASE_URL}/chat/completions", json=body, headers=headers, timeout=90)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def propose(facts: dict) -> tuple[dict | None, str, str | None]:
    """Return (proposal | None, provider_name, error | None)."""
    name = provider()
    if name == "rules":
        return None, "rules", None
    try:
        prompt = _user_prompt(facts)
        text = _call_anthropic(prompt) if name == "anthropic" else _call_openai_compatible(prompt)
        return _parse_json(text), name, None
    except Exception as exc:  # network, quota, bad JSON ... never break the case flow
        log.warning("LLM call failed: %s", exc)
        return None, "rules", f"{type(exc).__name__}: {exc}"
