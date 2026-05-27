"""Per-turn query_session.json — checklist + agent notes for downstream phases."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def path_for_session(session_jsonl: Path) -> Path:
    """e.g. 2026-05-26T23-23-24.session.jsonl → 2026-05-26T23-23-24.query_session.json"""
    stem = session_jsonl.stem
    base = stem.removesuffix(".session") if stem.endswith(".session") else stem
    return session_jsonl.with_name(f"{base}.query_session.json")


def load(session_jsonl: Path) -> dict[str, Any]:
    p = path_for_session(session_jsonl)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def save(session_jsonl: Path, state: dict[str, Any]) -> None:
    p = path_for_session(session_jsonl)
    p.write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def init_turn(
    session_jsonl: Path,
    user_query: str,
    checklist: list[dict[str, Any]],
    graph_summary: str = "",
) -> dict[str, Any]:
    state = {
        "session_id": session_jsonl.stem.replace(".session", ""),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "user_query": user_query,
        "checklist": checklist,
        "graph_summary_excerpt": (graph_summary or "")[:1500],
        "entries": [],
    }
    save(session_jsonl, state)
    return state


def append_entry(
    session_jsonl: Path,
    *,
    phase: str,
    task: str,
    notes: str,
    metadata: dict[str, Any] | None = None,
) -> None:
    state = load(session_jsonl)
    if not state:
        state = {"entries": []}
    state.setdefault("entries", []).append({
        "ts": datetime.now(timezone.utc).isoformat(),
        "phase": phase,
        "task": task,
        "notes": notes,
        "metadata": metadata or {},
    })
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    save(session_jsonl, state)


def format_for_prompt(session_jsonl: Path, max_entries: int = 12) -> str:
    state = load(session_jsonl)
    if not state:
        return ""
    lines = [
        f"Query: {state.get('user_query', '')}",
        "",
        "Checklist:",
    ]
    for item in state.get("checklist") or []:
        if isinstance(item, dict):
            mark = "x" if item.get("status") == "done" else " "
            lines.append(f"  [{mark}] {item.get('task', '')}")
    entries = state.get("entries") or []
    if entries:
        lines.append("\nAgent notes (from earlier phases this turn):")
        for e in entries[-max_entries:]:
            if not isinstance(e, dict):
                continue
            lines.append(f"  [{e.get('phase', '?')}] {e.get('task', '')}")
            n = (e.get("notes") or "").strip()
            if n:
                lines.append(f"      {n[:400]}{'…' if len(n) > 400 else ''}")
    excerpt = state.get("graph_summary_excerpt")
    if excerpt:
        lines.append("\nGraph (abbreviated):\n" + excerpt[:800])
    return "\n".join(lines)
