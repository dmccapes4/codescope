"""Token helpers + graph summary. Kept deliberately small.

The chat pipeline (chat/agent.py) does its own prompt assembly now; this module
only holds the two things shared with the CLI: token counting and the graph
summary text.
"""
from __future__ import annotations

from pathlib import Path

import tiktoken

from ..config import GRAPH_SUMMARY_TOKENS
from ..storage.graph import GraphIndex

_enc = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_enc.encode(text))


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    tokens = _enc.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return _enc.decode(tokens[:max_tokens]) + "\n… [truncated]"


def build_graph_summary(cache_dir: Path, max_tokens: int = GRAPH_SUMMARY_TOKENS) -> str:
    """Human-readable codebase map: modules, files (with summaries), manifest entries."""
    try:
        idx = GraphIndex(cache_dir)
    except Exception:
        return "(graph not yet built — run `codescope index` first)"

    lines = idx.summary_lines(max_files=30)
    text = "\n".join(lines)
    if count_tokens(text) <= max_tokens:
        return text

    # Too big — drop the per-file summaries (everything after an em dash).
    lines2 = [line.split("—")[0].rstrip() if "—" in line else line for line in lines]
    text2 = "\n".join(lines2)
    if count_tokens(text2) <= max_tokens:
        return text2

    module_lines = [l for l in lines if l.startswith(("Modules:", "Files indexed:"))]
    return "\n".join(module_lines) + "\n(use the graph command for details)"
