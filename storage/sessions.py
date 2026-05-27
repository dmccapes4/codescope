"""Append-only session log (sessions/<project>/<ts>.session.jsonl)."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _ts() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Session file management
# ---------------------------------------------------------------------------

def latest(session_dir: Path) -> Path | None:
    files = sorted(session_dir.glob("*.session.jsonl"))
    return files[-1] if files else None


def new_path(session_dir: Path) -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
    return session_dir / f"{ts}.session.jsonl"


def list_sessions(session_dir: Path) -> list[dict]:
    results = []
    for p in sorted(session_dir.glob("*.session.jsonl"), reverse=True):
        size  = p.stat().st_size
        mtime = datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc).isoformat()
        count = sum(1 for line in p.read_text(encoding="utf-8").splitlines() if line.strip())
        results.append({"file": p.name, "size": size, "modified": mtime, "entries": count})
    return results


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------

def read_all(path: Path) -> list[dict]:
    if not path.exists():
        return []
    entries = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return entries


def last_n_turns(path: Path, n: int) -> list[dict]:
    """Return the last n user+assistant pairs, oldest first."""
    entries = read_all(path)
    # Collect only user and assistant role entries (skip tool entries)
    convo = [e for e in entries if e.get("role") in ("user", "assistant")]
    # Pair up: take the last 2*n entries
    tail = convo[-(n * 2):]
    return tail


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------

def append(path: Path, record: dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()


def append_user(path: Path, content: str) -> None:
    append(path, {"role": "user", "ts": _ts(), "content": content})


def append_assistant(path: Path, content: str, thought: str = "") -> None:
    rec: dict[str, Any] = {"role": "assistant", "ts": _ts(), "content": content}
    if thought:
        rec["thought"] = thought
    append(path, rec)


def append_tool_call(path: Path, tool: str, args: dict, result: Any) -> None:
    append(path, {
        "role":   "tool",
        "ts":     _ts(),
        "name":   tool,
        "args":   args,
        "result": result,
    })
