# app/utils/llm.py
from typing import Any, Dict, List
import json
from openai import OpenAI
from .config import OPENAI_API_KEY, OPENAI_MODEL

client = OpenAI(api_key=OPENAI_API_KEY)


def _normalize_messages(raw_messages: List[Any]) -> List[Dict[str, str]]:
    """
    LangGraph keeps messages as LangChain Message objects. OpenAI client
    expects a list of dicts {role, content}. Convert safely.
    """
    normalized: List[Dict[str, str]] = []
    for m in raw_messages or []:
        if isinstance(m, dict):
            role = m.get("role")
            content = m.get("content", "")
        else:
            m_type = getattr(m, "type", None)
            if m_type == "human":
                role = "user"
            elif m_type == "ai":
                role = "assistant"
            elif m_type == "system":
                role = "system"
            else:
                role = "user"
            content = getattr(m, "content", "") or ""

        if not role:
            role = "user"

        normalized.append({"role": role, "content": content})
    return normalized


async def chat_llm(messages: List[Dict[str, str]]) -> str:
    """
    Simple wrapper. messages can be LangChain messages or dicts.
    """
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=_normalize_messages(messages),
        temperature=0.2,
    )
    return resp.choices[0].message.content


async def llm_json(prompt: str) -> Dict[str, Any]:
    """
    Ask the model to return a JSON object. We parse with json.loads.
    """
    system = {
        "role": "system",
        "content": "You are a strict JSON generator. Always return ONLY a valid JSON object.",
    }
    user = {"role": "user", "content": prompt}
    content = await chat_llm([system, user])

    try:
        start = content.find("{")
        end = content.rfind("}")
        return json.loads(content[start : end + 1])
    except Exception:
        return {
            "task": "none",
            "needs_clarification": True,
            "clarification_question": "Could you clarify what you want me to do?",
            "reasoning": "Failed to parse JSON from model.",
        }
