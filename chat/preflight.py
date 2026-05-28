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

# Matches source/config file paths explicitly named in a query.
# Handles both paths with extensions and extensionless paths that contain at least one
# directory separator (e.g. "database/entities/ClinicalNodes").
_SOURCE_FILE_RE = re.compile(
    r"(?:^|[\s\"'`(,])(?P<path>[\w][\w./\-]*\.(?:kt|java|xml|kts|toml|py|gradle))\b",
    re.IGNORECASE,
)
# Extensionless directory/ClassName paths, e.g. "database/entities/ClinicalNodes"
_SOURCE_PATH_RE = re.compile(
    r"(?:^|[\s\"'`(,])(?P<path>[\w][\w\-]+(?:/[\w][\w.\-]+)+)\b",
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


_SOURCE_EXTENSIONS = (".kt", ".java", ".xml", ".kts", ".toml", ".py", ".gradle")


def _resolve_source_path(raw: str, project_root: Path, seen: set[str]) -> str | None:
    """
    Resolve a raw path string to a project-relative path that exists on disk.

    Tries, in order:
      1. Exact path relative to project_root
      2. Exact path + each common extension (for extensionless inputs)
      3. rglob by filename across the whole project tree
    Returns a posix-relative path or None if not found.
    """
    raw = raw.replace("\\", "/").lstrip("./")
    name = Path(raw).name

    # 1. Exact path
    if (project_root / raw).is_file():
        rel = raw
        if rel not in seen:
            return rel

    # 2. Extensionless: try appending common extensions
    if not Path(raw).suffix:
        for ext in _SOURCE_EXTENSIONS:
            candidate = raw + ext
            if (project_root / candidate).is_file():
                if candidate not in seen:
                    return candidate
        # Also try just the basename + extensions (skip parent dirs)
        for ext in _SOURCE_EXTENSIONS:
            hits = list(project_root.rglob(name + ext))
            if hits:
                rel = hits[0].relative_to(project_root).as_posix()
                if rel not in seen:
                    return rel

    # 3. rglob by full filename (with extension)
    if Path(raw).suffix:
        hits = list(project_root.rglob(name))
        if hits:
            rel = hits[0].relative_to(project_root).as_posix()
            if rel not in seen:
                return rel

    return None


def extract_source_file_paths(query: str, project_root: Path | None = None) -> list[str]:
    """
    Return source/config file paths explicitly named in the query.

    Handles:
    - Paths with code extensions: database/entities/PatientEntity.kt
    - Extensionless directory paths: database/entities/ClinicalNodes
    When project_root is given, validates existence and resolves the real path.
    """
    paths: list[str] = []
    seen: set[str] = set()

    # Collect all raw candidates from both regexes
    candidates: list[str] = []
    for m in _SOURCE_FILE_RE.finditer(query):
        candidates.append(m.group("path"))
    for m in _SOURCE_PATH_RE.finditer(query):
        p = m.group("path")
        # Skip if it looks like a URL or doc path
        if p.startswith("http") or p.endswith(".md"):
            continue
        # Skip if already captured by the extension regex
        if not any(p in c or c in p for c in candidates):
            candidates.append(p)

    for raw in candidates:
        raw = raw.replace("\\", "/").lstrip("./")
        if project_root is None:
            if raw not in seen:
                seen.add(raw)
                paths.append(raw)
            continue

        resolved = _resolve_source_path(raw, project_root, seen)
        if resolved:
            seen.add(resolved)
            paths.append(resolved)

    return paths


def preflight_source_files(
    user_query: str,
    project_root: Path,
    tool_results: list[dict],
    turn_cache: dict[str, Any],
) -> list[str]:
    """
    Pre-read source files explicitly named in the query (e.g. database/entities/PatientEntity.kt).

    Results are appended to tool_results as read_file entries so they appear in
    CODEBASE RESEARCH section of the answer prompt.  Returns the list of file paths read.
    """
    from .tools import tool_read_file

    paths = extract_source_file_paths(user_query, project_root)
    read: list[str] = []
    for rel in paths:
        cache_key = f"read_file:{json.dumps({'path': rel}, sort_keys=True)}"
        if cache_key in turn_cache:
            result = turn_cache[cache_key]
        else:
            try:
                result = tool_read_file(path=rel, project_root=project_root)
            except Exception as e:
                result = {"error": str(e), "path": rel}
            turn_cache[cache_key] = result

        tool_results.append({
            "name":      "read_file",
            "args":      {"path": rel},
            "result":    result,
            "preflight": True,
        })
        read.append(rel)
    return read


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
    # File-review tasks — the agent must read the file, not fetch platform docs
    "please review",
    "review the file",
    "is this a correct implementation",
    "is this correct",
    "check this implementation",
    "does this look correct",
    "review this",
    "continuing with the implementation",
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

    Returns False for:
    - Pure codebase/README tasks (grep + semantic_search suffice)
    - File-review tasks (the named file must be read, not docs fetched)
    - Any query that explicitly names a source file (.kt/.java/.xml etc.)
    A wasted android_docs call burns ~700 tokens of context budget and adds latency.
    """
    q = user_query.lower()
    if any(p in q for p in _CODEBASE_ONLY_PATTERNS):
        return False
    # If a source file is explicitly named, this is always a codebase task
    if extract_source_file_paths(user_query):
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
