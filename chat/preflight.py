"""Pre-load project docs mentioned in the user query before the first LLM hop."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from ..config import PREFLIGHT_ANDROID_DOCS, PREFLIGHT_PROJECT_DOCS
from .tools import tool_docs_lookup

_DOC_PATH_RE = re.compile(
    r"(?:^|[\s\"'`(])(?P<path>(?:docs/)?[\w][\w./_-]*\.md)\b",
    re.IGNORECASE,
)


def extract_doc_paths(query: str, project_root: Path | None = None) -> list[str]:
    """Paths mentioned in the query. Only returns files that exist (avoids READ_ME.md typos on create)."""
    paths: list[str] = []
    seen: set[str] = set()
    for m in _DOC_PATH_RE.finditer(query):
        p = m.group("path").replace("\\", "/")
        if not p.startswith("docs/") and "/" not in p:
            candidates = [p, f"docs/{p}"]
        else:
            candidates = [p]
        for c in candidates:
            if c in seen:
                continue
            if project_root is not None:
                if not (project_root / c).is_file():
                    continue
            seen.add(c)
            paths.append(c)
    return paths


def preflight_project_docs(
    user_query: str,
    project_root: Path,
    tool_results: list[dict],
    turn_cache: dict[str, Any],
) -> None:
    """Load docs when the query names a full .md path (e.g. docs/GAME_PLAN.md)."""
    if not PREFLIGHT_PROJECT_DOCS:
        return
    paths = extract_doc_paths(user_query, project_root)
    if not paths:
        return
    for rel in paths:
        args = {"file": rel}
        cache_key = f"docs_lookup:{json.dumps(args, sort_keys=True)}"
        if cache_key in turn_cache:
            result = turn_cache[cache_key]
        else:
            try:
                result = tool_docs_lookup(file=rel, project_root=project_root)
            except Exception as e:
                result = {"error": str(e), "file": rel}
            turn_cache[cache_key] = result

        tool_results.append({
            "name":      "docs_lookup",
            "args":      args,
            "result":    result,
            "cached":    False,
            "preflight": True,
        })


def norm_doc_path(p: str) -> str:
    return p.replace("\\", "/").lstrip("./")


def preloaded_doc_index(tool_results: list[dict]) -> dict[str, dict]:
    """Map normalized doc path → docs_lookup result (content included)."""
    index: dict[str, dict] = {}
    for tr in tool_results:
        if tr.get("name") != "docs_lookup":
            continue
        result = tr.get("result")
        if not isinstance(result, dict) or not result.get("content"):
            continue
        path = tr.get("args", {}).get("file") or result.get("file")
        if path:
            index[norm_doc_path(str(path))] = result
    return index


def query_wants_android_docs(user_query: str) -> bool:
    """
    Return True only when the user explicitly wants to FETCH/read Android platform docs.

    Does NOT trigger on:
      "...implementation strategy... supporting Android documentation"
    (that means: mention which docs are relevant — the agent should call android_docs as a tool.)

    Does trigger on:
      "explain the Room documentation"
      "provide documentation on Room"
      "android_docs Room"
    """
    q = user_query.lower().strip()

    if q.startswith("android_docs"):
        return True

    explicit_fetch = (
        "fetch android doc",
        "look up android doc",
        "get the android documentation",
        "pull up android doc",
        "read the android documentation",
        "explain the room documentation",
        "explain room documentation",
        "provide documentation on room",
        "provide documentation on stateflow",
        "provide documentation on compose",
        "show me the room documentation",
        "what does the room documentation say",
        "documentation on room and how",
    )
    if any(p in q for p in explicit_fetch):
        return True

    return False


_CODEBASE_ONLY_PATTERNS = (
    "search through",
    "search the codebase",
    "inspect",
    "explore the code",
    "create a readme",
    "create readme",
    "write a readme",
    "write readme",
    "create a read_me",
    "project structure",
    "what files",
    "where is",
    "find in the project",
    "grep",
    "codebase",
    # Creating source files is codebase work, not platform-docs work
    "create patientactivity",
    "create doctoractivity",
    "create graphactivity",
    "create mainactivity",
    "create the activity",
    "create the file",
    "alongside the other activities",
    "alongside other activities",
    "edit androidmanifest",
    "update androidmanifest",
    "update the manifest",
    "edit the manifest",
)


def plan_needs_android_docs(user_query: str) -> bool:
    """Whether the planner should schedule android_docs for this query.

    Returns False for pure codebase/README tasks — those need grep + semantic_search,
    not Android platform docs, and a wasted android_docs call burns ~900 tokens of
    context budget.
    """
    q = user_query.lower()
    # Codebase-only tasks never need android_docs
    if any(p in q for p in _CODEBASE_ONLY_PATTERNS):
        return False
    if query_wants_android_docs(user_query):
        return True
    return any(
        p in q
        for p in (
            "supporting android documentation",
            "supporting documentation",
            "android documentation",
        )
    )


def preflight_android_docs(
    user_query: str,
    cache_dir: Path,
    tool_results: list[dict],
    turn_cache: dict[str, Any],
    doc_context_fn,
) -> tuple[bool, bool, str | None]:
    """
    Optional pre-fetch (off by default). Set CODESCOPE_PREFETCH_ANDROID_DOCS=1 to enable.
    Returns (android_docs_used, android_validate_used, last_url).
    """
    if not PREFLIGHT_ANDROID_DOCS or not query_wants_android_docs(user_query):
        return False, False, None

    from .android_docs import pick_primary_topic
    from .tools import tool_android_docs, tool_android_docs_validate

    topic = pick_primary_topic(user_query, doc_context_fn())
    args_docs = {"topic": topic}
    key_docs = f"android_docs:{json.dumps(args_docs, sort_keys=True)}"

    if key_docs in turn_cache:
        doc = turn_cache[key_docs]
    else:
        doc = tool_android_docs(topic=topic, cache_dir=cache_dir)
        turn_cache[key_docs] = doc

    tool_results.append({
        "name": "android_docs", "args": args_docs, "result": doc,
        "cached": False, "preflight": True,
    })

    url = doc.get("url") if isinstance(doc, dict) else None
    if not url:
        return True, False, None

    args_val = {"url": url, "topic": topic}
    key_val = f"android_docs_validate:{json.dumps(args_val, sort_keys=True)}"
    if key_val in turn_cache:
        validated = turn_cache[key_val]
    else:
        validated = tool_android_docs_validate(
            url=url, user_query=user_query, topic=topic, cache_dir=cache_dir,
        )
        turn_cache[key_val] = validated

    tool_results.append({
        "name": "android_docs_validate", "args": args_val, "result": validated,
        "cached": False, "preflight": True,
    })

    return True, True, url
