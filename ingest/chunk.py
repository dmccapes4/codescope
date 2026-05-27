"""Split file content into embedding-sized chunks."""
from __future__ import annotations

import re
from pathlib import Path
from typing import List

from ..config import CHUNK_SIZE, CHUNK_OVERLAP


def _sliding_window(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Simple character-level sliding window."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start += size - overlap
        if start >= len(text):
            break
    return chunks


# ---------------------------------------------------------------------------
# Language-aware chunkers
# ---------------------------------------------------------------------------

_KT_TOP_LEVEL = re.compile(
    r"(?:^|\n)(?=\s*(?:@\w+\s+)*(?:(?:abstract|sealed|open|data|inner|enum|annotation|value)\s+)*"
    r"(?:class|interface|object|fun)\s+\w+)",
)

_MD_HEADING = re.compile(r"(?:^|\n)(?=#{1,3} )")


def chunk_kotlin(text: str, rel: str) -> List[dict]:
    """Split on top-level Kotlin declarations; fall back to sliding window."""
    parts = _KT_TOP_LEVEL.split(text)
    chunks = []
    line_cursor = 1

    for part in parts:
        if not part.strip():
            line_cursor += part.count("\n")
            continue
        # If a part is too large, sub-split with sliding window
        if len(part) > CHUNK_SIZE * 2:
            for sub in _sliding_window(part):
                if sub.strip():
                    chunks.append({
                        "file":       rel,
                        "lines":      [line_cursor, line_cursor + sub.count("\n")],
                        "chunk_text": sub,
                        "kind":       "code",
                    })
                line_cursor += sub.count("\n")
        else:
            chunks.append({
                "file":       rel,
                "lines":      [line_cursor, line_cursor + part.count("\n")],
                "chunk_text": part,
                "kind":       "code",
            })
            line_cursor += part.count("\n")

    return chunks or [{"file": rel, "lines": [1, text.count("\n") + 1], "chunk_text": text, "kind": "code"}]


def chunk_markdown(text: str, rel: str) -> List[dict]:
    parts = _MD_HEADING.split(text)
    chunks = []
    line_cursor = 1
    for part in parts:
        if not part.strip():
            line_cursor += part.count("\n")
            continue
        if len(part) > CHUNK_SIZE * 2:
            for sub in _sliding_window(part):
                if sub.strip():
                    chunks.append({"file": rel, "lines": [line_cursor, line_cursor + sub.count("\n")], "chunk_text": sub, "kind": "doc"})
                line_cursor += sub.count("\n")
        else:
            chunks.append({"file": rel, "lines": [line_cursor, line_cursor + part.count("\n")], "chunk_text": part, "kind": "doc"})
            line_cursor += part.count("\n")
    return chunks or [{"file": rel, "lines": [1, text.count("\n") + 1], "chunk_text": text, "kind": "doc"}]


def chunk_whole(text: str, rel: str, kind: str = "config") -> List[dict]:
    """Whole file as one chunk (XML, Gradle, TOML) — or split if very large."""
    if len(text) <= CHUNK_SIZE * 3:
        return [{"file": rel, "lines": [1, text.count("\n") + 1], "chunk_text": text, "kind": kind}]
    chunks = []
    line_cursor = 1
    for part in _sliding_window(text):
        chunks.append({"file": rel, "lines": [line_cursor, line_cursor + part.count("\n")], "chunk_text": part, "kind": kind})
        line_cursor += part.count("\n")
    return chunks


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def chunk_file(abs_path: Path, rel: str, category: str) -> List[dict]:
    """Return a list of chunk dicts for the given file."""
    try:
        text = abs_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []

    if category in ("source.kotlin", "source.java"):
        return chunk_kotlin(text, rel)
    if category == "markdown":
        return chunk_markdown(text, rel)
    if category in ("manifest.xml", "layout.xml", "resource.xml", "gradle.kts",
                    "gradle.kts.settings", "gradle.groovy", "versions.toml", "proto"):
        return chunk_whole(text, rel, kind="config")
    # For any other text category, use sliding window
    return [{"file": rel, "lines": [1, text.count("\n") + 1], "chunk_text": text[:CHUNK_SIZE * 3], "kind": "other"}]
