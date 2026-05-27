"""Codebase research: graph, grep, semantic_search — writes notes to query_session.json."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable

from ..storage import session_state
from .tools import dispatch, ToolError


# ── High-confidence explicit patterns (score = 1.0) ─────────────────────────
_PRIOR_TURN_PATTERNS = (
    "previous query",
    "previous question",
    "previous response",
    "prior query",
    "prior question",
    "prior response",
    "last question",
    "last response",
    "as discussed",
    "as i mentioned",
    "review the previous",
    "review the last",
    "review our session",
    "review the session",
    "review our prior",
    "review my last",
    "based on the previous",
    "based on our",
    "from before",
    "you said",
    "you mentioned",
    "we discussed",
    "continue from",
    "continue where",
    "resume",
    "our session",
)

# ── Heuristic signals for implied context dependency ─────────────────────────

# Pronouns / demonstratives that reference something not introduced in this query
_REF_RE = re.compile(
    r"\b("
    r"it\b|this\b|that\b|these\b|those\b|them\b|its\b"
    r"|the\s+(?:same|above|following|previous|last|file|activity|kt|class"
    r"|approach|result|output|answer|code|function|method|pattern"
    r"|manifest|gradle|build\s+file|layout|viewmodel|repository|dao|database)"
    r")\b",
    re.IGNORECASE,
)

# Words that explicitly mean "do more of what we were doing"
_CONTINUE_RE = re.compile(
    r"\b(also|additionally|furthermore|moreover|continue|proceed"
    r"|next\s+step|move\s+on|resume|again|still|as\s+well"
    r"|then\s+also|and\s+also|building\s+on)\b",
    re.IGNORECASE,
)

# Past-tense second-person: "you created/wrote/added/showed"
_PAST_YOU_RE = re.compile(
    r"\byou\s+(?:said|mentioned|created|wrote|built|added|showed"
    r"|told|explained|implemented|suggested|gave|provided)\b",
    re.IGNORECASE,
)

# Imperative with definite article on a proper noun — likely refers to prior work
# e.g. "add comments to PatientActivity", "update the manifest", "edit the file"
_DEFINITE_PROPER_RE = re.compile(
    r"\b(?:add|edit|update|fix|change|modify|improve|extend|delete|remove)"
    r"\s+(?:\w+\s+)*the\s+[A-Z]\w+",
    re.IGNORECASE,
)

# Second-person possessive ("our work", "our approach", "my last answer")
_POSSESSIVE_RE = re.compile(r"\b(?:our|my)\s+(?:last|previous|prior|earlier|work|approach|answer|code|result)\b", re.IGNORECASE)


def _session_search_score(user_query: str) -> float:
    """
    Heuristic score [0.0, 1.0] for whether this query implies knowledge
    of a prior turn.  Returns 1.0 on any exact pattern match.
    Threshold for triggering session_search is 0.35.
    """
    q = user_query.lower()

    # Exact match → definite yes
    if any(p in q for p in _PRIOR_TURN_PATTERNS):
        return 1.0

    score = 0.0

    if _PAST_YOU_RE.search(user_query):
        score += 0.6          # "you created the file" is a very strong signal

    if _POSSESSIVE_RE.search(user_query):
        score += 0.5          # "our approach", "my last answer"

    if _CONTINUE_RE.search(user_query):
        score += 0.35         # "also add", "continue building"

    ref_hits = len(_REF_RE.findall(user_query))
    if ref_hits >= 2:
        score += 0.4          # multiple dangling references
    elif ref_hits == 1:
        score += 0.2

    if _DEFINITE_PROPER_RE.search(user_query):
        score += 0.25         # "edit the PatientActivity" — definite ref to named thing

    # Short queries (<= 10 words) with at least one reference are likely contextual
    if len(user_query.split()) <= 10 and ref_hits >= 1:
        score += 0.2

    return min(score, 1.0)


_GIT_PATTERNS = (
    "git diff",
    "git log",
    "git messages",
    "commit messages",
    "what changed",
    "recent changes",
    "what was modified",
    "uncommitted",
    "working tree",
    "modified files",
    "going back",       # "git messages going back"
    "git history",
)


def query_needs_session_search(user_query: str) -> bool:
    return _session_search_score(user_query) >= 0.35


def query_needs_git_context(user_query: str) -> bool:
    q = user_query.lower()
    return any(p in q for p in _GIT_PATTERNS)


def query_needs_codebase_research(user_query: str) -> bool:
    q = user_query.lower()
    return any(
        p in q
        for p in (
            "search through",
            "search the codebase",
            "inspect",
            "explore the code",
            "grep",
            "codebase",
            "create a readme",
            "create readme",
            "write a readme",
            "read_me",
            "project structure",
            "what files",
            "where is",
            "find in the project",
        )
    )


def _note_from_result(tool: str, args: dict, result: Any, max_len: int = 600) -> str:
    if isinstance(result, dict) and result.get("error"):
        return f"{tool} error: {result['error']}"
    if tool == "grep" and isinstance(result, list):
        lines = [f"{r.get('file')}:{r.get('line')}" for r in result[:15] if isinstance(r, dict)]
        return f"{len(result)} matches. Top: " + ", ".join(lines[:8])
    if tool == "semantic_search" and isinstance(result, list):
        parts = []
        for r in result[:6]:
            if isinstance(r, dict):
                parts.append(f"{r.get('file', '?')} (score {r.get('score', 0):.2f})")
        return "Hits: " + "; ".join(parts)
    if tool == "graph_lookup":
        text = json.dumps(result, ensure_ascii=False)[:max_len]
        return text
    if tool == "read_file" and isinstance(result, dict) and result.get("content"):
        path = result.get("path", "?")
        preview = result["content"][:max_len]
        return f"{path} excerpt:\n{preview}"
    return json.dumps(result, ensure_ascii=False)[:max_len]


def run_session_search(
    user_query: str,
    session_path: Path,
    session_dir: Path,
    tool_results: list[dict],
    turn_cache: dict[str, Any],
    verbose: bool = False,
    print_fn: Callable[[str], None] = print,
) -> None:
    """Search past Q&A turns in session.jsonl for prior context."""
    if verbose:
        print_fn("[codescope] session_search…")
    args = {"query": user_query, "n": 3}
    cache_key = f"session_search:{json.dumps(args, sort_keys=True)}"
    if cache_key in turn_cache:
        result = turn_cache[cache_key]
    else:
        from .tools import tool_session_search
        result = tool_session_search(
            query=user_query,
            n=3,
            session_path=session_path,
            session_dir=session_dir,
        )
        turn_cache[cache_key] = result
    tool_results.append({"name": "session_search", "args": args, "result": result})
    session_state.append_entry(
        session_path,
        phase="research",
        task="session_search",
        notes=f"{len(result)} prior turn(s) found" if isinstance(result, list) else str(result)[:200],
        metadata={"query": user_query[:120]},
    )


def run_git_context(
    user_query: str,
    session_path: Path,
    project_root: Path,
    tool_results: list[dict],
    turn_cache: dict[str, Any],
    verbose: bool = False,
    print_fn: Callable[[str], None] = print,
) -> None:
    """Fetch git log and diff; semantic-score diff chunks against user query."""
    from .tools import tool_git_log, tool_git_diff

    for tool_name, tkwargs, label in [
        ("git_log",  {"n": 10, "stat": True},    "git_log"),
        ("git_diff", {"staged": False, "path": ".", "max_bytes": 10_000}, "git_diff"),
    ]:
        cache_key = f"{tool_name}:{json.dumps(tkwargs, sort_keys=True)}"
        if cache_key in turn_cache:
            result = turn_cache[cache_key]
        else:
            if verbose:
                print_fn(f"[codescope] {label}…")
            fn = tool_git_log if tool_name == "git_log" else tool_git_diff
            result = fn(project_root=project_root, **tkwargs)  # type: ignore[arg-type]
            turn_cache[cache_key] = result
        tool_results.append({"name": tool_name, "args": tkwargs, "result": result})
        session_state.append_entry(
            session_path,
            phase="research",
            task=label,
            notes=str(result)[:300],
            metadata={"tool": tool_name},
        )


def run_codebase_research(
    user_query: str,
    session_path: Path,
    project_root: Path,
    cache_dir: Path,
    session_dir: Path,
    embedder,
    graph_summary: str,
    preloaded_docs: dict[str, dict] | None,
    tool_results: list[dict],
    turn_cache: dict[str, Any],
    verbose: bool = False,
    print_fn: Callable[[str], None] = print,
) -> None:
    """Run graph/grep/semantic probes; append query_session.json entries + tool_results."""
    session_state.append_entry(
        session_path,
        phase="research",
        task="Start codebase research",
        notes="Using graph summary, README, grep, and semantic_search.",
        metadata={"graph_chars": len(graph_summary)},
    )

    readme = project_root / "README.md"
    if readme.is_file():
        try:
            text = readme.read_text(encoding="utf-8", errors="replace")
            session_state.append_entry(
                session_path,
                phase="research",
                task="README.md project summary",
                notes=text[:1200] + ("…" if len(text) > 1200 else ""),
                metadata={"file": "README.md", "bytes": len(text)},
            )
        except OSError as e:
            session_state.append_entry(
                session_path, phase="research", task="README.md",
                notes=f"Could not read: {e}", metadata={},
            )

    probes: list[tuple[str, dict]] = [
        # Semantic broad sweep
        ("semantic_search", {
            "query": _research_semantic_query(user_query),
            "k": 10,
        }),
        # Enumerate every Activity/Fragment/ViewModel/Entity class declaration
        ("grep", {
            "pattern": r"class\s+\w+(Activity|Fragment|ViewModel|Repository)\b",
            "path": ".",
            "regex": True,
            "max_results": 40,
        }),
        # Enumerate every Room entity and DAO
        ("grep", {
            "pattern": r"@(Entity|Dao|Database)\b",
            "path": ".",
            "regex": True,
            "max_results": 30,
        }),
    ]

    for tool_name, args in probes:
        cache_key = f"{tool_name}:{json.dumps(args, sort_keys=True)}"
        if verbose:
            print_fn(f"[codescope] research: {tool_name}")
        try:
            if cache_key in turn_cache:
                result = turn_cache[cache_key]
            else:
                result = dispatch(
                    tool_name, args,
                    project_root, cache_dir, session_dir, embedder, user_query,
                    preloaded_docs=preloaded_docs,
                    session_path=session_path,
                )
                turn_cache[cache_key] = result
            note = _note_from_result(tool_name, args, result)
            session_state.append_entry(
                session_path,
                phase="research",
                task=f"{tool_name}",
                notes=note,
                metadata={"tool": tool_name, "args": args},
            )
            tool_results.append({"name": tool_name, "args": args, "result": result})

            # Graph-lookup for every UNIQUE source file that the grep hit
            if tool_name == "grep" and isinstance(result, list) and result:
                seen_files: set[str] = set()
                for hit in result:
                    if not isinstance(hit, dict):
                        continue
                    gf = str(hit.get("file", "")).replace("\\", "/")
                    if not gf or gf in seen_files:
                        continue
                    seen_files.add(gf)
                    gargs = {"file": gf}
                    gkey = f"graph_lookup:{json.dumps(gargs, sort_keys=True)}"
                    if gkey in turn_cache:
                        continue
                    try:
                        gres = dispatch(
                            "graph_lookup", gargs,
                            project_root, cache_dir, session_dir, embedder, user_query,
                            preloaded_docs=preloaded_docs,
                            session_path=session_path,
                        )
                        turn_cache[gkey] = gres
                    except ToolError as ge:
                        gres = {"error": str(ge)}
                    tool_results.append({"name": "graph_lookup", "args": gargs, "result": gres})
                    session_state.append_entry(
                        session_path,
                        phase="research",
                        task=f"graph_lookup {gf}",
                        notes=_note_from_result("graph_lookup", gargs, gres),
                        metadata={"tool": "graph_lookup", "file": gf},
                    )

        except ToolError as e:
            session_state.append_entry(
                session_path,
                phase="research",
                task=tool_name,
                notes=str(e),
                metadata={"tool": tool_name, "error": True},
            )


def _research_semantic_query(user_query: str) -> str:
    q = user_query.lower()
    if "readme" in q:
        return "Android project structure main activity application entry point gradle"
    if "patient" in q:
        return "PatientActivity Patient dashboard Room ClinicalNode ViewModel"
    return "Android Kotlin main application architecture Room Compose"
