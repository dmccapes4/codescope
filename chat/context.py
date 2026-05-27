"""Context assembly for the answer-phase LLM."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import tiktoken

from ..config import (
    GRAPH_SUMMARY_TOKENS,
    HISTORY_TOKENS,
    MAX_TOOL_RESULT_TOKENS,
    MAX_DOCS_TOOL_RESULT_TOKENS,
    PROMPT_FIELD_MAX_TOKENS,
    PREFLIGHT_DOC_MAX_TOKENS,
    ANDROID_BRIEF_MAX_TOKENS,
    BUDGET_GRAPH,
    BUDGET_RESEARCH,
    BUDGET_GIT,
    BUDGET_ANDROID,
    BUDGET_SESSION,
    BUDGET_SESSION_LOG,
    BUDGET_DOCS,
    BUDGET_HISTORY,
    FLOOR_GRAPH,
    FLOOR_RESEARCH,
    FLOOR_SESSION,
)
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
    try:
        idx = GraphIndex(cache_dir)
    except Exception:
        return "(graph not yet built — run `codescope index` first)"

    lines = idx.summary_lines(max_files=15)
    text = "\n".join(lines)
    if count_tokens(text) <= max_tokens:
        return text

    lines2 = [line.split("—")[0].rstrip() if "—" in line else line for line in lines]
    text2 = "\n".join(lines2)
    if count_tokens(text2) <= max_tokens:
        return text2

    module_lines = [l for l in lines if l.startswith("Modules:") or l.startswith("Files indexed:")]
    return "\n".join(module_lines) + "\n(use graph_lookup for details)"


def get_git_log(project_root: Path, n: int = 10) -> str:
    try:
        result = subprocess.run(
            [
                "git", "log", f"-{n}",
                "--pretty=format:%x1E%h %ad  %s",
                "--date=short", "--name-only",
            ],
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=5,
        )
        raw = result.stdout.strip()
        if not raw:
            return ""

        lines: list[str] = []
        for block in raw.split("\x1e"):
            block = block.strip()
            if not block:
                continue
            block_lines = block.splitlines()
            header = block_lines[0].strip()
            parts = header.split(" ", 1)
            sha = parts[0]
            rest = parts[1] if len(parts) > 1 else ""
            lines.append(f"[{sha}] {rest}")
            for f in block_lines[1:]:
                if f.strip():
                    lines.append(f"    {f.strip()}")
        return truncate_to_tokens("\n".join(lines), 400)
    except Exception:
        return ""


def _format_preloaded(tool_results: list[dict]) -> str:
    parts: list[str] = []
    for tr in tool_results:
        if tr.get("name") != "docs_lookup":
            continue
        result = tr.get("result")
        if not isinstance(result, dict) or not result.get("content"):
            continue
        path = result.get("file") or tr.get("args", {}).get("file", "?")
        parts.append(
            f"### {path}\n"
            + truncate_to_tokens(result["content"], MAX_DOCS_TOOL_RESULT_TOKENS)
        )
    return "\n\n".join(parts)


def _format_git(tool_results: list[dict]) -> str:
    """Format git_log and git_diff results."""
    parts: list[str] = []
    for tr in tool_results:
        name = tr.get("name")
        result = tr.get("result")
        if not isinstance(result, dict):
            continue
        if name == "git_log":
            commits = result.get("commits", "")
            if commits:
                parts.append("### git log\n" + str(commits)[:1200])
        elif name == "git_diff":
            diff = result.get("diff", "")
            if diff:
                parts.append(
                    f"### git diff ({result.get('lines', 0)} lines)\n"
                    + diff[:1500]
                )
    return "\n\n".join(parts)


def _format_session_log(tool_results: list[dict]) -> str:
    """Format session_search results — full prior Q&A turns."""
    parts: list[str] = []
    for tr in tool_results:
        if tr.get("name") != "session_search":
            continue
        result = tr.get("result")
        if not isinstance(result, list):
            continue
        for i, pair in enumerate(result):
            if not isinstance(pair, dict) or pair.get("note"):
                continue
            u = (pair.get("user") or "").strip()
            a = (pair.get("assistant") or "").strip()
            if u or a:
                parts.append(
                    f"### Prior turn {i + 1}\n"
                    f"User: {u}\n\n"
                    f"Assistant: {a}"
                )
    return "\n\n".join(parts)


def _format_research(tool_results: list[dict]) -> str:
    """Format all retrieved code evidence: grep, semantic_search, graph_lookup, read_file."""
    parts: list[str] = []
    for tr in tool_results:
        name = tr.get("name")
        if name not in ("grep", "semantic_search", "graph_lookup", "read_file"):
            continue
        result = tr.get("result")
        args = tr.get("args") or {}
        if isinstance(result, dict) and result.get("error"):
            continue  # silently drop errors from research — don't waste budget on them
        if name == "grep" and isinstance(result, list) and result:
            lines = [
                f"  {r.get('file')}:{r.get('line')}: {str(r.get('text', '')).strip()[:100]}"
                for r in result[:20]
                if isinstance(r, dict)
            ]
            if lines:
                pat = args.get("pattern", "")
                parts.append(f"### grep({pat!r})\n" + "\n".join(lines))
        elif name == "semantic_search" and isinstance(result, list) and result:
            lines = []
            for r in result[:8]:
                if not isinstance(r, dict):
                    continue
                snippet = str(r.get("snippet") or r.get("text", "")).strip().replace("\n", " ")
                lines.append(
                    f"  {r.get('file')} lines {r.get('lines','?')} "
                    f"(score {r.get('score', 0):.2f}): {snippet[:120]}"
                )
            if lines:
                parts.append("### semantic_search\n" + "\n".join(lines))
        elif name == "graph_lookup":
            node = result.get("node") if isinstance(result, dict) else None
            if node:
                nid   = node.get("id", "?")
                summ  = node.get("summary", "")
                ogre  = node.get("ogre_notes")
                out_e = [e.get("dst", "") for e in (result.get("out") or [])[:6]]
                in_e  = [e.get("src", "") for e in (result.get("in") or [])[:4]]
                block = f"  id: {nid}\n  summary: {summ}"
                if ogre:
                    block += f"\n  ogre_notes: {json.dumps(ogre, ensure_ascii=False)[:300]}"
                if out_e:
                    block += f"\n  depends_on: {', '.join(out_e)}"
                if in_e:
                    block += f"\n  used_by: {', '.join(in_e)}"
                parts.append(f"### graph_lookup({nid})\n{block}")
            elif isinstance(result, list):
                items = []
                for r in result[:6]:
                    n2 = r.get("node") if isinstance(r, dict) else None
                    if n2:
                        items.append(f"  {n2.get('id','?')}: {n2.get('summary','')}")
                if items:
                    parts.append("### graph_lookup\n" + "\n".join(items))
        elif name == "read_file" and isinstance(result, dict):
            if result.get("preloaded"):
                continue  # already in preloaded docs section
            content = (result.get("content") or "").strip()
            if content:
                p = result.get("path", args.get("path", "?"))
                start = result.get("start", "")
                end   = result.get("end", "")
                header = f"### read_file({p} lines {start}–{end})"
                parts.append(header + "\n" + content[:600] + ("…" if len(content) > 600 else ""))
    return "\n\n".join(parts)


def _android_brief(tool_results: list[dict]) -> str:
    validated = fetched = None
    topic = "Room"
    for tr in tool_results:
        if tr.get("name") == "android_docs_validate":
            r = tr.get("result")
            if isinstance(r, dict) and r.get("url"):
                validated = r
        if tr.get("name") == "android_docs":
            r = tr.get("result")
            if isinstance(r, dict) and r.get("url"):
                fetched = r
            if tr.get("args", {}).get("topic"):
                topic = tr["args"]["topic"]

    r = validated or fetched
    if not r:
        return ""

    url = r.get("url", "")
    excerpt = r.get("excerpt") or r.get("content") or r.get("summary") or ""
    excerpt = truncate_to_tokens(str(excerpt), ANDROID_BRIEF_MAX_TOKENS)
    return (
        f"{topic}: {url}\n\n"
        "Explain this for the user's task:\n"
        f"{excerpt}"
    )


def _fit(text: str, budget: int) -> tuple[str, int]:
    """Truncate text to budget tokens. Returns (fitted_text, tokens_used)."""
    if not text.strip():
        return "", 0
    fitted = truncate_to_tokens(text, budget)
    return fitted, count_tokens(fitted)


def _format_history(history: list[dict]) -> str:
    """Convert conversation history to a flat string for prompt inclusion."""
    lines: list[str] = []
    for entry in history or []:
        role    = entry.get("role", "")
        content = entry.get("content", "")
        if role == "user":
            lines.append(f"User: {content}")
        elif role == "assistant":
            lines.append(f"Assistant: {content}")
    return "\n".join(lines)


def build_answer_prompt(
    user_query: str,
    checklist: str,
    tool_results: list[dict],
    history: list[dict] | None = None,
    session_notes: str = "",
    graph_summary: str = "",
    git_summary: str = "",
    embedder: Any = None,
) -> str:
    """
    Assemble the final-answer prompt.

    When `embedder` is provided the prompt is built via hybrid-fusion curation:
      - Each context source is split into individual items (per grep block, per
        prior Q&A turn, etc.) and scored by cosine similarity to the query.
      - Minimum floors guarantee coverage for graph + research + session notes.
      - Remaining budget is filled score-descending across all sources.

    When `embedder` is None the legacy priority-waterfall is used (same as before).

    Priority order (for both paths):
      graph → research → git → android → session → session_log → docs → history
    """
    instruction = 'Respond with JSON: {"action":"final_answer","content":"<your full answer>"}'

    # ── Fixed sections (always present) ─────────────────────────────────────
    fixed = (
        "=== CURRENT QUESTION (answer this only — do not repeat a prior turn) ===\n"
        + user_query
        + "\n\n=== CHECKLIST (address each item in your answer) ===\n"
        + checklist
        + "\n\n"
        + instruction
    )
    context_budget = PROMPT_FIELD_MAX_TOKENS - count_tokens(fixed)

    # ── Pre-render each source section ───────────────────────────────────────
    research_raw    = _format_research(tool_results)
    git_raw_tools   = _format_git(tool_results)
    git_combined    = "\n\n".join(filter(None, [git_summary.strip(), git_raw_tools]))
    android_raw     = _android_brief(tool_results)
    session_log_raw = _format_session_log(tool_results)
    preloaded_raw   = _format_preloaded(tool_results)
    hist_raw        = _format_history(history or [])

    # ── Hybrid-fusion path (embedder available) ───────────────────────────────
    if embedder is not None and context_budget > 0:
        from .context_curator import curate_context, render_context

        section_map = {
            "graph":       graph_summary.strip(),
            "research":    research_raw,
            "git":         git_combined,
            "android":     android_raw,
            "session":     session_notes.strip(),
            "session_log": session_log_raw,
            "docs":        preloaded_raw,
            "history":     hist_raw,
        }
        floors = {
            "graph":    FLOOR_GRAPH,
            "research": FLOOR_RESEARCH,
            "session":  FLOOR_SESSION,
        }
        selected = curate_context(
            user_query=user_query,
            section_map=section_map,
            budget=context_budget,
            embedder=embedder,
            floors=floors,
        )
        context_body = render_context(selected)
        prompt = (context_body + "\n\n" + fixed) if context_body else fixed
        return prompt

    # ── Legacy waterfall path (no embedder) ──────────────────────────────────
    remaining = context_budget
    sections: list[str] = []

    if graph_summary.strip() and remaining > 0:
        text, used = _fit(
            "=== CODEBASE GRAPH ===\n" + graph_summary,
            min(BUDGET_GRAPH, remaining),
        )
        if text:
            sections.append(text)
            remaining -= used

    if research_raw and remaining > 0:
        text, used = _fit(
            "=== CODEBASE RESEARCH (grep / semantic / graph / read_file) ===\n" + research_raw,
            min(BUDGET_RESEARCH, remaining),
        )
        if text:
            sections.append(text)
            remaining -= used

    if git_combined and remaining > 0:
        text, used = _fit(
            "=== GIT CONTEXT ===\n" + git_combined,
            min(BUDGET_GIT, remaining),
        )
        if text:
            sections.append(text)
            remaining -= used

    if android_raw and remaining > 0:
        text, used = _fit(
            "=== ANDROID DOCUMENTATION ===\n" + android_raw,
            min(BUDGET_ANDROID, remaining),
        )
        if text:
            sections.append(text)
            remaining -= used

    if session_notes.strip() and remaining > 0:
        text, used = _fit(
            "=== SESSION NOTES (this turn) ===\n" + session_notes,
            min(BUDGET_SESSION, remaining),
        )
        if text:
            sections.append(text)
            remaining -= used

    if session_log_raw and remaining > 0:
        text, used = _fit(
            "=== RELEVANT PRIOR TURNS (session log) ===\n" + session_log_raw,
            min(BUDGET_SESSION_LOG, remaining),
        )
        if text:
            sections.append(text)
            remaining -= used

    if preloaded_raw and remaining > 0:
        doc_budget = min(BUDGET_DOCS, remaining)
        if doc_budget < 200:
            sections.append(
                "=== PROJECT DOCUMENTATION ===\n"
                "(docs pre-loaded but budget exhausted; rely on CODEBASE RESEARCH above)"
            )
        else:
            text, used = _fit(
                "=== PROJECT DOCUMENTATION (already loaded — do not re-read) ===\n"
                + preloaded_raw,
                doc_budget,
            )
            if text:
                sections.append(text)
                remaining -= used

    if hist_raw.strip() and remaining > 0:
        text, _ = _fit(
            "=== PRIOR TURNS (context only — do not copy prior answers) ===\n" + hist_raw,
            min(BUDGET_HISTORY, remaining),
        )
        if text:
            sections.append(text)

    prompt = "\n\n".join(sections) + "\n\n" + fixed if sections else fixed
    return prompt


# Legacy alias for any callers
def build_prompt(
    system_prompt: str,
    history: list[dict],
    tool_results_this_turn: list[dict],
    user_query: str,
    **kwargs: Any,
) -> str:
    return build_answer_prompt(
        user_query=user_query,
        checklist="",
        tool_results=tool_results_this_turn,
        history=history,
    )
