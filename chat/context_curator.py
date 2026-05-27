"""Hybrid-fusion context curation: per-item relevance scoring, floor guarantees, budget fill.

Architecture:
  1. Each pre-rendered section (graph, research, git, …) is split into scorable chunks.
     - Research and session_log are split on `### ` headers so each individual tool result
       (grep block, semantic hit, prior Q&A turn) is scored independently.
     - All other sections are treated as a single chunk (already small / single-topic).
  2. Every chunk is scored by cosine similarity against the query embedding.
  3. Mandatory floors: the top-scoring chunks from configured sources are pinned first,
     guaranteeing minimum coverage even if the budget is exhausted.
  4. Remaining budget is filled score-descending across all sources.
  5. Selected chunks are re-ordered by _SOURCE_ORDER for a coherent prompt structure.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import numpy as np

# Logical display order for sections in the assembled prompt
_SOURCE_ORDER = [
    "graph",
    "research",
    "git",
    "android",
    "session",
    "session_log",
    "docs",
    "history",
]

_SECTION_HEADERS: dict[str, str] = {
    "graph":       "=== CODEBASE GRAPH ===",
    "research":    "=== CODEBASE RESEARCH (grep / semantic / graph / read_file) ===",
    "git":         "=== GIT CONTEXT ===",
    "android":     "=== ANDROID DOCUMENTATION ===",
    "session":     "=== SESSION NOTES (this turn) ===",
    "session_log": "=== RELEVANT PRIOR TURNS (session log) ===",
    "docs":        "=== PROJECT DOCUMENTATION (already loaded — do not re-read) ===",
    "history":     "=== PRIOR TURNS (context only — do not copy prior answers) ===",
}


@dataclass
class ContextChunk:
    source: str    # one of _SOURCE_ORDER values
    label: str     # short descriptor (shown in debug logs)
    text: str      # raw chunk text
    tokens: int = 0
    score: float = 0.0

    def __post_init__(self) -> None:
        if not self.tokens:
            from .context import count_tokens
            self.tokens = count_tokens(self.text)


# ---------------------------------------------------------------------------
# Chunking helpers
# ---------------------------------------------------------------------------

def _split_on_headers(text: str) -> list[str]:
    """Split text at '### ' sub-headers; each part keeps its header line."""
    if not text.strip():
        return []
    parts = re.split(r"\n(?=### )", text.strip())
    return [p.strip() for p in parts if p.strip()]


def _chunks_for_source(source: str, text: str) -> list[ContextChunk]:
    """
    Break a formatted source section into individual scorable chunks.

    Research and session_log are split at `### ` boundaries so that each grep
    result block, semantic hit, or prior Q&A turn is scored independently.
    All other sources are kept as a single chunk.
    """
    if not text.strip():
        return []

    if source in ("research", "session_log"):
        parts = _split_on_headers(text)
        if len(parts) > 1:
            return [
                ContextChunk(
                    source=source,
                    label=p.splitlines()[0][:60] if p else source,
                    text=p,
                )
                for p in parts
            ]

    return [ContextChunk(source=source, label=source, text=text.strip())]


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _score_in_place(
    chunks: list[ContextChunk],
    query_vec: np.ndarray,
    embedder: Any,
) -> None:
    """Assign cosine-similarity scores to chunks (embeddings are already L2-normed)."""
    if not chunks:
        return
    vecs = embedder.encode([c.text for c in chunks], batch_size=64)  # (N, dim)
    sims = (vecs @ query_vec).tolist()
    for chunk, sim in zip(chunks, sims):
        chunk.score = float(sim)


# ---------------------------------------------------------------------------
# Main curator
# ---------------------------------------------------------------------------

def curate_context(
    user_query: str,
    section_map: dict[str, str],
    budget: int,
    embedder: Any,
    floors: dict[str, int] | None = None,
) -> list[ContextChunk]:
    """
    Score and select context chunks for the answer prompt.

    Args:
        user_query:  Current user question (drives relevance scoring).
        section_map: source_name → pre-rendered text (from _format_* helpers in context.py).
                     Only keys present in _SOURCE_ORDER are processed.
        budget:      Maximum total tokens for the curated context body
                     (excluding the fixed question/checklist sections).
        embedder:    Loaded Embedder instance.
        floors:      Per-source minimum token guarantees.
                     e.g. {"graph": 250, "research": 300}
                     Sources with a floor will always contribute at least that many
                     tokens (taken from their highest-scoring chunks).

    Returns:
        Ordered list of selected ContextChunk objects (sorted by _SOURCE_ORDER).
    """
    floors = floors or {}

    # 1. Build per-source chunk lists
    by_source: dict[str, list[ContextChunk]] = {}
    all_chunks: list[ContextChunk] = []
    for source in _SOURCE_ORDER:
        text = section_map.get(source, "")
        chunks = _chunks_for_source(source, text)
        if chunks:
            by_source[source] = chunks
            all_chunks.extend(chunks)

    if not all_chunks:
        return []

    # 2. Score all chunks against the query
    query_vec: np.ndarray = embedder.encode_query(user_query)
    _score_in_place(all_chunks, query_vec, embedder)

    # 3. Separate mandatory (floor-guaranteed) from optional
    mandatory: list[ContextChunk] = []
    optional: list[ContextChunk] = []

    for source, chunks in by_source.items():
        floor_tokens = floors.get(source, 0)
        if floor_tokens <= 0:
            optional.extend(chunks)
            continue
        # Take highest-scoring chunks until we hit the floor token count
        sorted_desc = sorted(chunks, key=lambda c: c.score, reverse=True)
        taken = 0
        for chunk in sorted_desc:
            if taken < floor_tokens:
                mandatory.append(chunk)
                taken += chunk.tokens
            else:
                optional.append(chunk)

    # 4. Fill remaining budget score-descending
    mandatory_tokens = sum(c.tokens for c in mandatory)
    remaining = budget - mandatory_tokens

    selected_optional: list[ContextChunk] = []
    for chunk in sorted(optional, key=lambda c: c.score, reverse=True):
        if chunk.tokens <= remaining:
            selected_optional.append(chunk)
            remaining -= chunk.tokens

    # 5. Re-order by source for coherent presentation
    order = {s: i for i, s in enumerate(_SOURCE_ORDER)}
    selected = mandatory + selected_optional
    selected.sort(key=lambda c: (order.get(c.source, 99), -c.score))
    return selected


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_context(chunks: list[ContextChunk]) -> str:
    """
    Assemble selected chunks into a prompt string.

    Chunks from the same source are grouped under one section header.
    The header is emitted once per source group, preserving _SOURCE_ORDER.
    """
    if not chunks:
        return ""

    sections: list[str] = []
    current_source: str | None = None
    group: list[str] = []

    for chunk in chunks:
        if chunk.source != current_source:
            if current_source is not None and group:
                hdr = _SECTION_HEADERS.get(current_source, f"=== {current_source.upper()} ===")
                sections.append(hdr + "\n" + "\n\n".join(group))
            current_source = chunk.source
            group = [chunk.text]
        else:
            group.append(chunk.text)

    if current_source is not None and group:
        hdr = _SECTION_HEADERS.get(current_source, f"=== {current_source.upper()} ===")
        sections.append(hdr + "\n" + "\n\n".join(group))

    return "\n\n".join(sections)
