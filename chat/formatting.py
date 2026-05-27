"""Format agent answers for plain terminals (Android Studio, WSL) — no markdown links."""
from __future__ import annotations

import json
import re
from typing import Any


def format_terminal_answer(text: str) -> str:
    """Convert markdown links to 'Label: https://...' so URLs are visible and copyable."""
    if not text:
        return text

    text = re.sub(
        r"\[([^\]]+)\]\((https?://[^)\s]+)\)",
        r"\1: \2",
        text,
    )
    text = re.sub(r"<(https?://[^>]+)>", r"\1", text)
    return text


def format_structured_answer(data: dict[str, Any]) -> str:
    """Turn common LLM JSON answer shapes into readable plain text."""
    if not data:
        return ""

    lines: list[str] = []

    strat = data.get("implementation_strategy")
    if strat is not None:
        lines.append("## Implementation strategy")
        if isinstance(strat, str):
            lines.append(strat.strip())
        elif isinstance(strat, dict):
            if strat.get("description"):
                lines.append(str(strat["description"]).strip())
            for i, step in enumerate(strat.get("steps") or [], 1):
                if isinstance(step, dict):
                    title = step.get("step") or step.get("title") or f"Step {i}"
                    lines.append(f"\n{i}. {title}")
                    details = step.get("details") or step.get("description")
                    if details:
                        lines.append(f"   {details}")
                elif isinstance(step, str):
                    lines.append(f"\n{i}. {step}")

    for key in ("strategy", "implementation", "answer", "content"):
        val = data.get(key)
        if isinstance(val, str) and val.strip() and key not in (
            "implementation_strategy",
        ):
            lines.append(f"\n## {key.replace('_', ' ').title()}\n{val.strip()}")

    doc = data.get("documentation") or data.get("supporting_documentation")
    if isinstance(doc, str) and doc.strip():
        lines.append(f"\n## Android documentation\n{doc.strip()}")

    links = data.get("documentation_links")
    if links:
        lines.append("\n## Links")
        if isinstance(links, list):
            for item in links:
                if isinstance(item, dict):
                    topic = item.get("topic") or item.get("name") or "Documentation"
                    url = item.get("link") or item.get("url") or ""
                    if url:
                        lines.append(f"{topic}: {url}")
                elif isinstance(item, str):
                    lines.append(item)
        elif isinstance(links, dict):
            for topic, url in links.items():
                lines.append(f"{topic}: {url}")

    if not lines:
        return json.dumps(data, ensure_ascii=False, indent=2)

    return format_terminal_answer("\n".join(lines).strip())


def coerce_answer_text(data: dict[str, Any] | Any) -> str:
    """Best-effort plain text from LLM JSON (with or without action/content wrapper)."""
    if isinstance(data, str):
        return format_terminal_answer(data.strip())
    if not isinstance(data, dict):
        return str(data)

    if data.get("action") == "final_answer":
        content = data.get("content")
        if isinstance(content, str) and content.strip():
            return format_terminal_answer(content.strip())
        if isinstance(content, dict):
            return format_structured_answer(content)

    if "content" in data and isinstance(data["content"], str):
        return format_terminal_answer(data["content"].strip())

    if any(
        k in data
        for k in (
            "implementation_strategy",
            "documentation_links",
            "strategy",
            "documentation",
        )
    ):
        return format_structured_answer(data)

    return format_structured_answer(data)
