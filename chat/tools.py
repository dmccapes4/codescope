"""Tool implementations for the agent loop."""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np

from ..config import EXCLUDED_DIRS, EXCLUDED_SUFFIXES
from ..storage.graph import GraphIndex
from ..storage.vectors import load as vectors_load, search as vectors_search
from ..storage.sessions import read_all as sessions_read_all


class ToolError(ValueError):
    pass


def _assert_safe_path(base: Path, rel: str) -> Path:
    """Resolve rel inside base and assert it doesn't escape."""
    resolved = (base / rel).resolve()
    try:
        resolved.relative_to(base.resolve())
    except ValueError:
        raise ToolError(f"Path {rel!r} escapes the project root.")
    return resolved


# ---------------------------------------------------------------------------
# grep
# ---------------------------------------------------------------------------

def _grep_python(
    pattern: str,
    search_root: Path,
    case_insensitive: bool,
    is_regex: bool,
    max_results: int,
) -> list[dict]:
    flags = re.IGNORECASE if case_insensitive else 0
    try:
        rx = re.compile(pattern if is_regex else re.escape(pattern), flags)
    except re.error as e:
        raise ToolError(f"Invalid regex: {e}")

    results = []
    for dirpath, dirnames, filenames in os.walk(search_root):
        dirnames[:] = [d for d in dirnames if d not in EXCLUDED_DIRS and not d.startswith(".")]
        for name in filenames:
            ext = Path(name).suffix.lower()
            if ext in EXCLUDED_SUFFIXES:
                continue
            fp = Path(dirpath) / name
            try:
                lines = fp.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                continue
            for lineno, line in enumerate(lines, 1):
                if rx.search(line):
                    rel = fp.relative_to(search_root).as_posix()
                    results.append({"file": rel, "line": lineno, "text": line.rstrip()})
                    if len(results) >= max_results:
                        return results
    return results


def tool_grep(
    pattern: str,
    path: str = ".",
    case_insensitive: bool = False,
    regex: bool = True,
    max_results: int = 40,
    include_sessions: bool = False,
    project_root: Path | None = None,
    session_dir: Path | None = None,
) -> list[dict]:
    if not project_root:
        raise ToolError("project_root not set")

    search_root = _assert_safe_path(project_root, path)
    if not search_root.exists():
        raise ToolError(f"Path does not exist: {path}")

    rg_bin = shutil.which("rg")
    if rg_bin:
        cmd = [rg_bin, "--line-number", "--with-filename", "--no-heading",
               "--max-count", str(max_results)]
        if case_insensitive:
            cmd.append("--ignore-case")
        if not regex:
            cmd.append("--fixed-strings")
        cmd += ["--", pattern, str(search_root)]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if proc.returncode not in (0, 1):
                # rg exit 1 = no matches (fine); anything else = error → fall back
                raise RuntimeError(proc.stderr[:200])
            results = []
            for line in proc.stdout.splitlines():
                parts = line.split(":", 2)
                if len(parts) == 3:
                    try:
                        rel = str(Path(parts[0]).relative_to(project_root)).replace("\\", "/")
                        results.append({"file": rel, "line": int(parts[1]), "text": parts[2]})
                    except (ValueError, TypeError):
                        continue
                if len(results) >= max_results:
                    break
        except Exception:
            results = _grep_python(pattern, search_root, case_insensitive, regex, max_results)
    else:
        results = _grep_python(pattern, search_root, case_insensitive, regex, max_results)

    # Optionally also grep session logs
    if include_sessions and session_dir and session_dir.exists():
        results += _grep_python(pattern, session_dir, case_insensitive, regex,
                                 max(0, max_results - len(results)))

    return results[:max_results]


# ---------------------------------------------------------------------------
# semantic_search
# ---------------------------------------------------------------------------

def tool_semantic_search(
    query: str,
    k: int = 8,
    filter: dict | None = None,
    cache_dir: Path | None = None,
    embedder=None,
) -> list[dict]:
    if not cache_dir or not embedder:
        raise ToolError("cache_dir and embedder required")

    matrix, meta = vectors_load(cache_dir)
    if matrix is None or matrix.shape[0] == 0:
        return [{"error": "No embeddings found. Run `codescope index --embed --project <name>` first."}]

    query_vec = embedder.encode_query(query)
    ext_filter = (filter or {}).get("ext")

    hits = vectors_search(matrix, meta, query_vec, k=k, filter_ext=ext_filter)

    results = []
    for h in hits:
        results.append({
            "file":    h["file"],
            "lines":   h["lines"],
            "score":   round(h["score"], 4),
            "symbol":  h["symbol"],
            "snippet": h["chunk_text"][:300] if h.get("chunk_text") else "",
        })
    return results


# ---------------------------------------------------------------------------
# read_file
# ---------------------------------------------------------------------------

def tool_read_file(
    path: str,
    start: int = 1,
    end: int | None = None,
    project_root: Path | None = None,
) -> dict:
    if not project_root:
        raise ToolError("project_root not set")

    abs_path = _assert_safe_path(project_root, path)
    if not abs_path.exists():
        # If path has no extension, try common code extensions then rglob
        if not abs_path.suffix:
            _exts = (".kt", ".java", ".xml", ".kts", ".toml", ".py", ".gradle")
            for ext in _exts:
                candidate = abs_path.with_name(abs_path.name + ext)
                if candidate.is_file():
                    abs_path = candidate
                    path = str(candidate.relative_to(project_root))
                    break
            else:
                # Fall back to rglob by name
                hits = list(project_root.rglob(abs_path.name + ".*"))
                code_hits = [h for h in hits if h.suffix in _exts]
                if code_hits:
                    abs_path = code_hits[0]
                    path = str(abs_path.relative_to(project_root))
                else:
                    raise ToolError(f"File not found: {path}")
        else:
            # Try rglob by filename in case the path prefix is wrong
            hits = list(project_root.rglob(abs_path.name))
            if hits:
                abs_path = hits[0]
                path = str(abs_path.relative_to(project_root))
            else:
                raise ToolError(f"File not found: {path}")
    if not abs_path.is_file():
        raise ToolError(f"Not a file: {path}")

    try:
        lines = abs_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as e:
        raise ToolError(f"Cannot read {path}: {e}")

    total = len(lines)
    start = max(1, min(start, total))
    if end is None:
        end = min(start + 199, total)
    end = max(start, min(end, total, start + 399))  # hard cap: 400 lines

    selected = lines[start - 1 : end]
    return {
        "path":    path,
        "start":   start,
        "end":     start + len(selected) - 1,
        "total":   total,
        "content": "\n".join(selected),
    }


# ---------------------------------------------------------------------------
# list_dir
# ---------------------------------------------------------------------------

def tool_list_dir(
    path: str = ".",
    depth: int = 1,
    project_root: Path | None = None,
) -> list[dict]:
    if not project_root:
        raise ToolError("project_root not set")

    abs_path = _assert_safe_path(project_root, path)
    if not abs_path.exists():
        raise ToolError(f"Path not found: {path}")

    depth = max(1, min(depth, 3))
    results: list[dict] = []

    def _walk(p: Path, current_depth: int):
        if current_depth > depth:
            return
        try:
            entries = sorted(p.iterdir())
        except PermissionError:
            return
        for entry in entries:
            if entry.name in EXCLUDED_DIRS or entry.name.startswith("."):
                continue
            if entry.suffix.lower() in EXCLUDED_SUFFIXES:
                continue
            rel = entry.relative_to(project_root).as_posix()
            info: dict[str, Any] = {
                "path": rel,
                "type": "dir" if entry.is_dir() else "file",
            }
            if entry.is_file():
                info["size"] = entry.stat().st_size
            results.append(info)
            if entry.is_dir() and current_depth < depth:
                _walk(entry, current_depth + 1)

    _walk(abs_path, 1)
    return results


# ---------------------------------------------------------------------------
# graph_lookup
# ---------------------------------------------------------------------------

_graph_cache: dict[str, GraphIndex] = {}


def _get_graph(cache_dir: Path) -> GraphIndex:
    key = str(cache_dir)
    if key not in _graph_cache:
        _graph_cache[key] = GraphIndex(cache_dir)
    return _graph_cache[key]


def invalidate_graph_cache(cache_dir: Path) -> None:
    _graph_cache.pop(str(cache_dir), None)


def tool_graph_lookup(
    node: str | None = None,
    file: str | None = None,
    cache_dir: Path | None = None,
) -> dict | list:
    if not cache_dir:
        raise ToolError("cache_dir required")
    idx = _get_graph(cache_dir)

    if node:
        result = idx.lookup_node(node)
        if result is None:
            return {"error": f"Node {node!r} not found in graph."}
        return result

    if file:
        results = idx.lookup_file(file)
        if not results:
            return {"error": f"No nodes found for file {file!r}."}
        return results

    raise ToolError("Provide either 'node' or 'file' argument.")


# ---------------------------------------------------------------------------
# write_file / edit_file — sandboxed file authoring
# ---------------------------------------------------------------------------

# Extensions the agent is allowed to write
_WRITABLE_EXTENSIONS: frozenset[str] = frozenset({
    ".md", ".kt", ".java", ".kts", ".gradle",
    ".xml", ".json", ".toml", ".properties",
    ".py", ".txt", ".sh",
})


def _assert_writable(project_root: Path, rel: str) -> Path:
    """Resolve and validate that rel is a writable path inside the project."""
    # Strip common hallucinated prefixes like "project_root/", "./", "/"
    rel = rel.replace("\\", "/")
    for prefix in ("project_root/", "projectroot/", "root/", "./", "/"):
        if rel.startswith(prefix):
            rel = rel[len(prefix):]
    abs_path = _assert_safe_path(project_root, rel)
    norm = rel.lstrip("./")

    # Allowed locations
    in_docs     = norm.startswith("docs/")
    is_root_md  = norm.count("/") == 0 and abs_path.suffix.lower() == ".md"
    in_src      = (
        norm.startswith("app/src/")
        or norm.startswith("app/generated/")
        or norm.startswith("lib/src/")
        or norm.startswith("library/src/")
    )
    in_generated = "generated" in norm.split("/")

    if not (in_docs or is_root_md or in_src or in_generated):
        raise ToolError(
            f"write_file: {rel!r} is outside the allowed zones "
            "(docs/, project-root *.md, app/src/**, generated/**)."
        )

    if abs_path.suffix.lower() not in _WRITABLE_EXTENSIONS:
        raise ToolError(
            f"write_file: extension {abs_path.suffix!r} is not in the allowed set "
            f"({', '.join(sorted(_WRITABLE_EXTENSIONS))})."
        )
    return abs_path


def _atomic_write(abs_path: Path, content: str) -> None:
    """Write to a temp file in the same directory, then rename — avoids partial writes."""
    abs_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=abs_path.parent, prefix=".codescope_tmp_")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
        os.replace(tmp, abs_path)   # atomic on POSIX; best-effort on Windows
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def tool_write_file(
    path: str,
    content: str,
    overwrite: bool = False,
    project_root: Path | None = None,
    session_path: Path | None = None,
) -> dict:
    """
    Create or overwrite a file inside the project.
    Sandboxed to docs/, project-root *.md, app/src/**, generated/**
    and to safe extensions (.md .kt .java .kts .gradle .xml .json .toml …).
    Pass overwrite=true to replace an existing file.
    """
    if not project_root:
        raise ToolError("project_root required")
    if not content:
        raise ToolError("content must not be empty")

    abs_path = _assert_writable(project_root, path)

    if abs_path.exists() and not overwrite:
        raise ToolError(
            f"write_file: {path!r} already exists. Pass overwrite=true to replace it."
        )

    existed = abs_path.exists()
    _atomic_write(abs_path, content)

    record = {
        "op": "write_file",
        "path": path,
        "action": "overwritten" if existed else "created",
        "bytes": len(content.encode("utf-8")),
    }
    _record_file_op(session_path, record)
    return {**record, "status": "ok"}


def tool_edit_file(
    path: str,
    old_string: str,
    new_string: str,
    project_root: Path | None = None,
    session_path: Path | None = None,
) -> dict:
    """
    Patch an existing file by replacing an exact string.
    old_string must appear exactly ONCE — this prevents ambiguous edits.
    Uses an atomic write so a crash mid-edit cannot corrupt the file.
    """
    if not project_root:
        raise ToolError("project_root required")
    if not old_string:
        raise ToolError("old_string must not be empty")
    if old_string == new_string:
        raise ToolError("old_string and new_string are identical — nothing to do.")

    abs_path = _assert_writable(project_root, path)

    if not abs_path.exists():
        raise ToolError(f"edit_file: {path!r} does not exist. Use write_file to create it first.")

    try:
        original = abs_path.read_text(encoding="utf-8")
    except OSError as e:
        raise ToolError(f"edit_file: cannot read {path}: {e}")

    count = original.count(old_string)
    if count == 0:
        raise ToolError(
            f"edit_file: old_string not found in {path!r}.\n"
            "Tip: grep the file first to confirm the exact text, including whitespace."
        )
    if count > 1:
        raise ToolError(
            f"edit_file: old_string appears {count} times in {path!r} — ambiguous. "
            "Add more surrounding context to make it unique."
        )

    patched = original.replace(old_string, new_string, 1)
    _atomic_write(abs_path, patched)

    record = {
        "op": "edit_file",
        "path": path,
        "old_len": len(old_string),
        "new_len": len(new_string),
        "net_bytes": len(new_string.encode()) - len(old_string.encode()),
    }
    _record_file_op(session_path, record)
    return {**record, "status": "ok"}


def _record_file_op(session_path: Path | None, record: dict) -> None:
    """Append a file-write record to query_session.json if a session is active."""
    if not session_path:
        return
    try:
        from ..storage import session_state
        import datetime
        session_state.append_entry(
            session_path,
            phase="write",
            task=f"{record['op']} {record['path']}",
            notes=str({k: v for k, v in record.items() if k != "op"}),
            metadata=record,
        )
    except Exception:
        pass  # never let logging crash a successful write


# ---------------------------------------------------------------------------
# graph_enrich  (OGrE — Opportunistic Graph Enrichment)
# ---------------------------------------------------------------------------

def tool_graph_enrich(
    file: str | None = None,
    node: str | None = None,
    summary: str | None = None,
    notes: dict | None = None,
    cache_dir: Path | None = None,
) -> dict:
    """
    Update a graph node's summary and/or structured notes after reading it.
    Identifies the node by file path or node_id. Writes back to graph.nodes.jsonl.
    """
    if not cache_dir:
        raise ToolError("cache_dir required")
    if not file and not node:
        raise ToolError("Provide either 'file' or 'node' to identify the node to enrich.")
    if not summary and not notes:
        raise ToolError("Provide at least 'summary' or 'notes' to enrich the node.")

    from ..storage import graph as graph_store

    all_nodes = graph_store.load_nodes(cache_dir)

    # Resolve target node_id
    target_id: str | None = None
    if node and node in all_nodes:
        target_id = node
    elif file:
        norm = file.replace("\\", "/").lstrip("./")
        for nid, n in all_nodes.items():
            nf = (n.get("file") or "").replace("\\", "/").lstrip("./")
            if nf == norm or nf.endswith("/" + norm) or norm.endswith("/" + nf):
                target_id = nid
                break

    if target_id is None:
        # No existing node — create a lightweight stub so the enrichment isn't lost
        import datetime
        target_id = f"ogre:{(file or node or 'unknown').replace('/', ':')}"
        all_nodes[target_id] = {
            "id":   target_id,
            "kind": "file",
            "file": file or "",
            "summary": summary or "",
            "ogre_notes": notes or {},
            "ogre_at": datetime.datetime.utcnow().isoformat(),
        }
        graph_store.rewrite_nodes(cache_dir, all_nodes)
        invalidate_graph_cache(cache_dir)
        return {"status": "created_stub", "id": target_id}

    n = all_nodes[target_id]
    if summary:
        n["summary"] = summary
    if notes:
        existing = n.get("ogre_notes") or {}
        existing.update(notes)
        n["ogre_notes"] = existing

    import datetime
    n["ogre_at"] = datetime.datetime.utcnow().isoformat()
    all_nodes[target_id] = n

    graph_store.rewrite_nodes(cache_dir, all_nodes)
    invalidate_graph_cache(cache_dir)
    return {"status": "enriched", "id": target_id, "summary": n.get("summary", "")}


# ---------------------------------------------------------------------------
# docs_lookup
# ---------------------------------------------------------------------------

def _find_doc_files(project_root: Path) -> list[Path]:
    """Return all README.md (root) and docs/**/*.md files in the project."""
    docs: list[Path] = []
    # Root README (any case)
    for name in project_root.iterdir():
        if name.is_file() and name.suffix.lower() == ".md" and name.stem.lower() == "readme":
            docs.append(name)
    # docs/ folder (any nesting level)
    docs_dir = project_root / "docs"
    if docs_dir.is_dir():
        for md in sorted(docs_dir.rglob("*.md")):
            if md.is_file():
                docs.append(md)
    return docs


def tool_docs_lookup(
    query: str | None = None,
    file: str | None = None,
    project_root: Path | None = None,
) -> dict | list:
    """
    Search or read project documentation (README.md + docs/**/*.md).

    - file="docs/GAME_PLAN.md"  → read that file in full
    - query="architecture"       → keyword search across all doc files
    - (no args)                  → list all available doc files with sizes
    """
    if not project_root:
        raise ToolError("project_root required")

    doc_files = _find_doc_files(project_root)

    # ── List mode ───────────────────────────────────────────────────────────
    if not file and not query:
        return [
            {
                "file": str(p.relative_to(project_root).as_posix()),
                "size": p.stat().st_size,
            }
            for p in doc_files
        ]

    # ── Read specific file ───────────────────────────────────────────────────
    if file:
        abs_path = _assert_safe_path(project_root, file)
        if not abs_path.exists():
            raise ToolError(f"Documentation file not found: {file}")
        if abs_path.suffix.lower() != ".md":
            raise ToolError(f"docs_lookup only reads .md files. Use read_file for other types.")
        # Validate it is a doc file (root README or inside docs/)
        rel_parts = abs_path.relative_to(project_root).parts
        in_docs = any(p.lower() == "docs" for p in rel_parts[:-1])
        is_root_readme = len(rel_parts) == 1 and abs_path.stem.lower() == "readme"
        if not (in_docs or is_root_readme):
            raise ToolError(
                f"{file!r} is not a documentation file. "
                "docs_lookup only reads README.md (root) and docs/**/*.md. "
                "Use read_file for other files."
            )
        content = abs_path.read_text(encoding="utf-8", errors="replace")
        return {
            "file":    file,
            "size":    len(content),
            "content": content,
            "_hint":   "Full document loaded. Do NOT call docs_lookup again for this file this turn.",
        }

    # ── Keyword search ───────────────────────────────────────────────────────
    assert query
    flags = re.IGNORECASE
    try:
        rx = re.compile(re.escape(query), flags)
    except re.error as e:
        raise ToolError(f"Invalid search pattern: {e}")

    results: list[dict] = []
    for doc in doc_files:
        rel = doc.relative_to(project_root).as_posix()
        try:
            lines = doc.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        for lineno, line in enumerate(lines, 1):
            if rx.search(line):
                results.append({"file": rel, "line": lineno, "text": line.rstrip()})
                if len(results) >= 60:
                    return results

    if not results:
        available = [p.relative_to(project_root).as_posix() for p in doc_files]
        return {
            "matches": 0,
            "query":   query,
            "searched": available,
            "message": f"No matches for {query!r} in documentation files.",
        }
    return results


# ---------------------------------------------------------------------------
# android_docs  (official developer.android.com — NOT project docs)
# ---------------------------------------------------------------------------

def tool_android_docs(
    topic: str | None = None,
    topics: list[str] | str | None = None,
    cache_dir: Path | None = None,
) -> dict:
    """
    Fetch ONE official Android documentation page (developer.android.com).

    Only the primary topic is used — pass the most important topic for the user's question.
    """
    from .android_docs import fetch_android_doc, pick_primary_topic, TOPIC_URLS

    chosen = (topic or "").strip()
    if not chosen and topics:
        if isinstance(topics, str):
            parts = [t.strip() for t in re.split(r"[,;]+", topics) if t.strip()]
        else:
            parts = [str(t).strip() for t in topics if str(t).strip()]
        chosen = parts[0] if parts else ""

    if not chosen:
        return {
            "topics": sorted(set(TOPIC_URLS.keys())),
            "usage":  'android_docs with topic="Room" (one topic per turn)',
            "note":   "Official Android docs only. Next: android_docs_validate with the returned url.",
        }

    doc = fetch_android_doc(chosen, cache_dir=cache_dir)
    doc["next_step"] = (
        "Call android_docs_validate with url=<url from this result> before final_answer."
    )
    return doc


def tool_android_docs_validate(
    url: str,
    user_query: str = "",
    topic: str = "",
    cache_dir: Path | None = None,
) -> dict:
    """Verify an Android doc URL matches the user's question; returns excerpt + canonical url."""
    from .android_docs import validate_android_doc

    if not url or not url.strip():
        raise ToolError("url is required for android_docs_validate")
    return validate_android_doc(
        url=url.strip(),
        user_query=user_query,
        topic=topic,
        cache_dir=cache_dir,
    )


# ---------------------------------------------------------------------------
# web_search  (internet; HITL-required)
# ---------------------------------------------------------------------------

def tool_web_search(
    query: str,
    max_results: int = 5,
    hitl_enabled: bool = False,
) -> list[dict]:
    """
    Lightweight web search using DuckDuckGo Instant Answer API.

    This tool is intentionally gated: it can run only when HITL is enabled.
    """
    if not hitl_enabled:
        raise ToolError(
            "web_search requires HITL. Re-run chat/ask with --hitl to enable internet search."
        )
    q = (query or "").strip()
    if not q:
        raise ToolError("web_search: query is required")
    max_results = max(1, min(max_results, 10))

    url = (
        "https://api.duckduckgo.com/?"
        + urllib.parse.urlencode(
            {
                "q": q,
                "format": "json",
                "no_html": "1",
                "no_redirect": "1",
                "skip_disambig": "1",
            }
        )
    )
    req = urllib.request.Request(url, headers={"User-Agent": "codescope/0.1.0"})
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="replace"))
    except Exception as e:
        raise ToolError(f"web_search failed: {e}")

    results: list[dict] = []

    # Primary abstract result (if present)
    if data.get("AbstractURL"):
        results.append(
            {
                "title": data.get("Heading") or q,
                "url": data.get("AbstractURL"),
                "snippet": (data.get("AbstractText") or "").strip(),
                "source": "duckduckgo",
            }
        )

    def _add_topic(topic: dict) -> None:
        if len(results) >= max_results:
            return
        u = topic.get("FirstURL")
        t = topic.get("Text")
        if not u or not t:
            return
        results.append(
            {
                "title": t.split(" - ")[0][:120],
                "url": u,
                "snippet": t[:280],
                "source": "duckduckgo",
            }
        )

    # Flat and nested related topics
    for item in data.get("RelatedTopics", []) or []:
        if len(results) >= max_results:
            break
        if isinstance(item, dict) and "Topics" in item:
            for t in item.get("Topics") or []:
                if isinstance(t, dict):
                    _add_topic(t)
        elif isinstance(item, dict):
            _add_topic(item)

    if not results:
        return [{"note": f"No web results found for {q!r}."}]
    return results[:max_results]


# ---------------------------------------------------------------------------
# git_diff / git_log
# ---------------------------------------------------------------------------

def _run_git(args: list[str], cwd: Path, timeout: int = 15) -> str:
    try:
        r = subprocess.run(args, cwd=cwd, capture_output=True, text=True, timeout=timeout)
        return r.stdout.strip()
    except Exception:
        return ""


def tool_git_diff(
    staged: bool = False,
    path: str = ".",
    max_bytes: int = 12_000,
    project_root: Path | None = None,
) -> dict:
    """Return the current git diff (unstaged by default, or --cached for staged)."""
    if not project_root:
        raise ToolError("project_root required")
    cmd = ["git", "diff"]
    if staged:
        cmd.append("--cached")
    cmd += ["--", path]
    raw = _run_git(cmd, project_root)
    if not raw:
        return {"diff": "", "lines": 0, "note": "No changes detected (clean working tree)."}
    if len(raw) > max_bytes:
        raw = raw[:max_bytes] + f"\n… [truncated — {len(raw) - max_bytes} bytes omitted]"
    return {"diff": raw, "lines": raw.count("\n"), "staged": staged}


def tool_git_log(
    n: int = 10,
    stat: bool = True,
    project_root: Path | None = None,
) -> dict:
    """Return the last n git commits with optional file stats."""
    if not project_root:
        raise ToolError("project_root required")
    fmt = "%H%x09%ad%x09%an%x09%s"
    cmd = ["git", "log", f"-{n}", f"--pretty=format:{fmt}", "--date=short"]
    if stat:
        cmd.append("--stat")
    raw = _run_git(cmd, project_root)
    if not raw:
        return {"commits": [], "note": "No commits found or not a git repo."}
    return {"commits": raw, "count": n}


# ---------------------------------------------------------------------------
# session_search  (semantic keyword search over past session.jsonl turns)
# ---------------------------------------------------------------------------

def tool_session_search(
    query: str,
    n: int = 3,
    include_tools: bool = False,
    session_path: Path | None = None,
    session_dir: Path | None = None,
) -> list[dict]:
    """
    Search past Q&A turns in the current session.jsonl for content relevant to
    the query.  Returns up to n matching turns as {user, assistant} dicts,
    with the full text of each side — so the model can reference prior answers
    without re-running research.

    Scoring: simple token overlap between query words and turn content.
    """
    if not session_path or not session_path.exists():
        # Fall back to latest session in session_dir
        if session_dir:
            from ..storage.sessions import latest as _latest
            fallback = _latest(session_dir)
            if fallback and fallback.exists():
                session_path = fallback
    if not session_path or not session_path.exists():
        return [{"note": "No session log found."}]

    from ..storage.sessions import read_all as _read_all
    entries = _read_all(session_path)

    # Reconstruct Q&A pairs
    pairs: list[dict] = []
    i = 0
    while i < len(entries):
        e = entries[i]
        if e.get("role") == "user":
            pair: dict = {"user": e.get("content", ""), "assistant": "", "tools": []}
            j = i + 1
            while j < len(entries) and entries[j].get("role") != "user":
                r = entries[j]
                if r.get("role") == "assistant":
                    pair["assistant"] = r.get("content", "")
                elif r.get("role") == "tool" and include_tools:
                    pair["tools"].append({"name": r.get("name"), "args": r.get("args")})
                j += 1
            pairs.append(pair)
            i = j
        else:
            i += 1

    if not pairs:
        return [{"note": "Session log is empty."}]

    # Score by token overlap with query
    q_tokens = set(query.lower().split())

    def _score(pair: dict) -> float:
        text = (pair["user"] + " " + pair["assistant"]).lower()
        return sum(1 for t in q_tokens if t in text) / max(len(q_tokens), 1)

    ranked = sorted(pairs, key=_score, reverse=True)
    top = ranked[:n]

    results = []
    for p in top:
        results.append({
            "user":      p["user"][:2000],
            "assistant": p["assistant"][:3000],
        })
    return results


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

TOOL_REGISTRY = {
    "grep":            tool_grep,
    "semantic_search": tool_semantic_search,
    "read_file":       tool_read_file,
    "list_dir":        tool_list_dir,
    "graph_lookup":    tool_graph_lookup,
    "graph_enrich":    tool_graph_enrich,
    "write_file":      tool_write_file,
    "edit_file":       tool_edit_file,
    "git_diff":        tool_git_diff,
    "git_log":         tool_git_log,
    "session_search":  tool_session_search,
    "docs_lookup":     tool_docs_lookup,
    "android_docs":           tool_android_docs,
    "android_docs_validate":  tool_android_docs_validate,
    "web_search":      tool_web_search,
}


def normalize_tool_args(tool_name: str, args: dict) -> dict:
    """Map common LLM arg mistakes to the real parameter names."""
    args = dict(args or {})

    if tool_name == "read_file":
        for old, new in (
            ("file_path", "path"),
            ("filepath", "path"),
            ("file", "path"),
            ("start_line", "start"),
            ("end_line", "end"),
            ("line_start", "start"),
            ("line_end", "end"),
        ):
            if old in args and new not in args:
                args[new] = args.pop(old)

    if tool_name == "docs_lookup":
        if "file" not in args and "path" in args:
            args["file"] = args.pop("path")
        if "query" in args and "file" not in args:
            args.setdefault("query", args["query"])

    if tool_name == "graph_lookup":
        # LLMs often pass "path", "id", or "file_path" — normalise to "file" or "node"
        for alias in ("path", "file_path", "filepath"):
            if alias in args and "file" not in args and "node" not in args:
                val = args.pop(alias)
                # Looks like a file path if it contains a slash or has a code extension
                if "/" in str(val) or "." in str(val):
                    args["file"] = val
                else:
                    args["node"] = val
                break
        if "id" in args and "file" not in args and "node" not in args:
            val = args.pop("id")
            if "/" in str(val) or "." in str(val):
                args["file"] = val
            else:
                args["node"] = val

    if tool_name == "grep" and "query" in args and "pattern" not in args:
        args["pattern"] = args.pop("query")
    if tool_name == "web_search" and "q" in args and "query" not in args:
        args["query"] = args.pop("q")

    return args


def _lookup_preloaded(path: str, preloaded_docs: dict[str, dict] | None) -> dict | None:
    if not preloaded_docs or not path:
        return None
    p = path.replace("\\", "/").lstrip("./")
    if p in preloaded_docs:
        return preloaded_docs[p]
    for key, val in preloaded_docs.items():
        if p.endswith(key) or key.endswith(p):
            return val
    return None


def dispatch(
    tool_name: str,
    args: dict,
    project_root: Path,
    cache_dir: Path,
    session_dir: Path,
    embedder,
    user_query: str = "",
    preloaded_docs: dict[str, dict] | None = None,
    session_path: Path | None = None,
    hitl_enabled: bool = False,
) -> Any:
    """Call the named tool with its args plus injected context. Returns JSON-serialisable result."""
    if tool_name not in TOOL_REGISTRY:
        raise ToolError(f"Unknown tool: {tool_name!r}. Available: {list(TOOL_REGISTRY)}")

    args = normalize_tool_args(tool_name, args)

    if tool_name == "read_file" and args.get("path"):
        hit = _lookup_preloaded(args["path"], preloaded_docs)
        if hit:
            return {
                "path": args["path"],
                "content": hit.get("content", ""),
                "file": hit.get("file", args["path"]),
                "preloaded": True,
                "_note": "Content served from doc pre-loaded at turn start (also in prompt).",
            }

    if tool_name == "docs_lookup" and args.get("file"):
        hit = _lookup_preloaded(args["file"], preloaded_docs)
        if hit:
            return {**hit, "preloaded": True, "_note": "Already loaded at turn start."}

    # Inject runtime context into args
    ctx = {
        "project_root": project_root,
        "cache_dir":    cache_dir,
        "session_dir":  session_dir,
        "embedder":     embedder,
        "user_query":   user_query,
        "session_path": session_path,
        "hitl_enabled": hitl_enabled,
    }
    merged = {**args, **ctx}

    fn = TOOL_REGISTRY[tool_name]
    import inspect
    valid_params = set(inspect.signature(fn).parameters)
    filtered = {k: v for k, v in merged.items() if k in valid_params}

    try:
        return fn(**filtered)
    except ToolError:
        raise
    except Exception as e:
        raise ToolError(f"Tool {tool_name} failed: {e}") from e
