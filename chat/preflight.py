"""Pre-load project docs mentioned in the user query before the first LLM hop."""
from __future__ import annotations

import difflib
import json
import re
from pathlib import Path
from typing import Any

from ..config import PREFLIGHT_ANDROID_DOCS, PREFLIGHT_PROJECT_DOCS
from .tools import tool_docs_lookup

# Cache of every source file under each project root (basename → list of relative paths)
# so we can do cheap difflib matches on typos like "ClinicalApplicaton.kt".
_PROJECT_FILE_INDEX: dict[str, list[str]] = {}
_TYPO_SIMILARITY_CUTOFF = 0.82  # 0.0–1.0; below this we don't auto-correct
# A correction must be measurably better than picking a random file of the same
# extension; we also require the closest match to lead the runner-up by this
# much to avoid silently grabbing the wrong sibling.
_TYPO_MARGIN = 0.05


def _project_file_index(project_root: Path) -> list[str]:
    """Return all source/config files under project_root as project-relative posix paths.
    Cached per project_root for the life of the process."""
    key = str(project_root.resolve())
    cached = _PROJECT_FILE_INDEX.get(key)
    if cached is not None:
        return cached
    files: list[str] = []
    for ext in _SOURCE_EXTENSIONS:
        for p in project_root.rglob(f"*{ext}"):
            try:
                rel = p.relative_to(project_root).as_posix()
            except ValueError:
                continue
            # Skip build artifacts / IDE caches — they pollute close-match results.
            if rel.startswith((".gradle/", ".idea/", "build/", ".codescope-cache/")):
                continue
            if "/build/" in rel:
                continue
            files.append(rel)
    _PROJECT_FILE_INDEX[key] = files
    return files


def _token_sort_ratio(a: str, b: str) -> float:
    """Like fuzzy token-sort: tokenise on [_\\-\\s.], sort, then compare. Handles
    typos that ALSO reorder tokens, e.g. STARTEGY_PATIENT_ACTIVITY vs
    PATIENT_ACTIVITY_STRATEGY."""
    def _tokens(s: str) -> str:
        return " ".join(sorted(t for t in re.split(r"[_\-\s.]+", s.lower()) if t))
    return difflib.SequenceMatcher(None, _tokens(a), _tokens(b)).ratio()


def _best_filename_score(query_name: str, candidate_name: str) -> float:
    """Return max(SequenceMatcher, token-sort) — the best of both worlds for
    typos vs reorderings."""
    a = Path(query_name.lower()).stem
    b = Path(candidate_name.lower()).stem
    s1 = difflib.SequenceMatcher(None, a, b).ratio()
    s2 = _token_sort_ratio(a, b)
    return max(s1, s2)


def _closest_filename_match(name: str, project_root: Path) -> str | None:
    """Find the project-relative path whose basename is closest to `name` (typo-tolerant).
    Returns None if no match clears the similarity cutoff."""
    if not name:
        return None
    files = _project_file_index(project_root)
    if not files:
        return None
    name_lower = name.lower()
    name_ext = Path(name_lower).suffix

    # Only consider candidates with matching extension if one was provided.
    pool = [f for f in files if (not name_ext or Path(f).suffix.lower() == name_ext)]
    if not pool:
        pool = files

    # Score by basename similarity using max(SequenceMatcher, token-sort) so we
    # catch both letter-flips AND reordered tokens (e.g. STARTEGY_PATIENT_ACTIVITY
    # vs PATIENT_ACTIVITY_STRATEGY).
    scored: list[tuple[float, str]] = []
    for rel in pool:
        s = _best_filename_score(name, Path(rel).name)
        scored.append((s, rel))
    scored.sort(reverse=True)
    if not scored:
        return None
    best_score, best_rel = scored[0]
    if best_score < _TYPO_SIMILARITY_CUTOFF:
        return None
    if len(scored) > 1 and (best_score - scored[1][0]) < _TYPO_MARGIN:
        # Two files are equally close — refuse to guess.
        return None
    return best_rel

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
    """Paths mentioned in the query. Returns files that exist; for typos like
    'docs/STARTEGY_PATIENT.md' (missing 'R') we fall through to a difflib match
    against the actual files under docs/, gated by _TYPO_SIMILARITY_CUTOFF.
    """
    paths: list[str] = []
    seen: set[str] = set()
    for m in _DOC_PATH_RE.finditer(query):
        p = m.group("path").replace("\\", "/")
        if not p.startswith("docs/") and "/" not in p:
            candidates = [p, f"docs/{p}"]
        else:
            candidates = [p]
        resolved = None
        for c in candidates:
            if c in seen:
                resolved = c
                break
            if project_root is None:
                resolved = c
                break
            if (project_root / c).is_file():
                resolved = c
                break

        if resolved is None and project_root is not None:
            # Typo tolerance: try difflib against project_root/docs/*.md.
            resolved = _closest_doc_match(p, project_root)

        if resolved and resolved not in seen:
            seen.add(resolved)
            paths.append(resolved)

    return paths


def _closest_doc_match(name: str, project_root: Path) -> str | None:
    """Find the closest docs/*.md file to `name` (typo-tolerant + reorder-tolerant).
    Returns a project-relative posix path or None if no candidate clears the cutoff."""
    if not name:
        return None
    name = name.replace("\\", "/")
    bare = Path(name).name
    docs_dir = project_root / "docs"
    if not docs_dir.is_dir():
        candidates = list(project_root.rglob("*.md"))
    else:
        candidates = list(docs_dir.rglob("*.md"))
    if not candidates:
        return None
    scored: list[tuple[float, Path]] = []
    for p in candidates:
        s = _best_filename_score(bare, p.name)
        scored.append((s, p))
    scored.sort(reverse=True)
    if not scored:
        return None
    best_score, best_path = scored[0]
    if best_score < _TYPO_SIMILARITY_CUTOFF:
        return None
    if len(scored) > 1 and (best_score - scored[1][0]) < _TYPO_MARGIN:
        return None
    try:
        return best_path.relative_to(project_root).as_posix()
    except ValueError:
        return None


_SOURCE_EXTENSIONS = (".kt", ".java", ".xml", ".kts", ".toml", ".py", ".gradle")


def _resolve_source_path(raw: str, project_root: Path, seen: set[str]) -> str | None:
    """
    Resolve a raw path string to a project-relative path that exists on disk.

    Tries, in order:
      1. Exact path relative to project_root
      2. Exact path + each common extension (for extensionless inputs)
      3. rglob by filename across the whole project tree
      4. difflib close-match against project file index (typo tolerance)
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

    # 4. Typo tolerance — try difflib close-match against the file index.
    fuzzy = _closest_filename_match(name, project_root)
    if fuzzy and fuzzy not in seen:
        return fuzzy

    return None


def extract_source_file_paths(query: str, project_root: Path | None = None) -> list[str]:
    """
    Return source/config file paths explicitly named in the query.

    Handles:
    - Paths with code extensions: database/entities/PatientEntity.kt
    - Extensionless directory paths: database/entities/ClinicalNodes
    When project_root is given, validates existence and resolves the real path
    (including typo-tolerant difflib close-matches).
    """
    return [rel for rel, _raw in extract_source_file_resolutions(query, project_root)]


def extract_source_file_resolutions(
    query: str,
    project_root: Path | None = None,
) -> list[tuple[str, str]]:
    """Like extract_source_file_paths but returns (resolved_rel, original_raw) pairs
    so callers can detect typo corrections (resolved basename != raw basename)."""
    out: list[tuple[str, str]] = []
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
                out.append((raw, raw))
            continue

        resolved = _resolve_source_path(raw, project_root, seen)
        if resolved:
            seen.add(resolved)
            out.append((resolved, raw))

    return out


def preflight_source_files(
    user_query: str,
    project_root: Path,
    tool_results: list[dict],
    turn_cache: dict[str, Any],
) -> list[str]:
    """
    Pre-read source files explicitly named in the query (e.g. database/entities/PatientEntity.kt).

    Results are appended to tool_results as read_file entries with preflight=True so
    they render in the PRE-READ SOURCE FILES section of the answer prompt.
    Typo-corrected resolutions get a `_note` on the result so the LLM (and the
    user, via stdout) can see the correction.
    Returns the list of (display) file paths read.
    """
    from .tools import tool_read_file

    resolutions = extract_source_file_resolutions(user_query, project_root)
    read: list[str] = []
    for rel, raw in resolutions:
        cache_key = f"read_file:{json.dumps({'path': rel}, sort_keys=True)}"
        if cache_key in turn_cache:
            result = turn_cache[cache_key]
        else:
            try:
                result = tool_read_file(path=rel, project_root=project_root)
            except Exception as e:
                result = {"error": str(e), "path": rel}
            turn_cache[cache_key] = result

        # Detect typo correction: resolved basename differs from what the user wrote.
        raw_base = Path(raw.replace("\\", "/")).name.lower()
        rel_base = Path(rel).name.lower()
        if raw_base and raw_base != rel_base and isinstance(result, dict) and "error" not in result:
            note = f"resolved typo: '{raw}' → '{rel}' (closest match)"
            result.setdefault("_note", note)

        tool_results.append({
            "name":      "read_file",
            "args":      {"path": rel, "original_query_path": raw},
            "result":    result,
            "preflight": True,
        })
        # Display label shows correction in-line so the visible banner is honest.
        if raw_base and raw_base != rel_base:
            read.append(f"{rel} (was: {raw})")
        else:
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
