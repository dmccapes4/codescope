"""Agent: preflight docs → planner (fast LLM) → run tools → answer (main LLM)."""
from __future__ import annotations

import json
import os as _os
from pathlib import Path
from typing import Any, Callable

# Silence sentence-transformers' "Loading weights: …" tqdm bar and the HF Hub
# unauthenticated-request warning. We surface our own clean banner instead.
_os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
_os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
_os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

from rich.syntax import Syntax
from rich.panel import Panel
from rich.console import Console as _Console

_rich_console = _Console()


def _pretty(obj, title: str = "") -> None:
    text = json.dumps(obj, indent=2, ensure_ascii=False) if not isinstance(obj, str) else obj
    syntax = Syntax(text, "json", theme="monokai", word_wrap=True)
    if title:
        _rich_console.print(Panel(syntax, title=f"[bold]{title}[/bold]", border_style="dim"))
    else:
        _rich_console.print(syntax)

from pydantic import BaseModel, ValidationError

import re as _re

from ..config import (
    AUTO_ANDROID_VALIDATE,
    PLANNER_LLM,
    PLANNER_USE_ANSWER_MODEL,
    USE_PLANNER_LLM,
    WARM_MODEL_AT_REPL,
    ANSWER_NUM_PREDICT,
    ANSWER_NUM_PREDICT_EXPLAIN,
    ANSWER_NUM_PREDICT_WRITE,
    TWO_STAGE_WRITE,
    TWO_STAGE_PLAN_NUM_PREDICT,
)
from ..models.llm import LLM, OllamaError
from ..models.embedder import Embedder
from ..storage import sessions as session_store
from ..storage import session_state
from .context import build_answer_prompt, build_graph_summary, get_git_log
from .preflight import (
    extract_doc_paths,
    extract_source_file_paths,
    preflight_project_docs,
    preflight_source_files,
    preflight_android_docs,
    preloaded_doc_index,
)
from .planner import run_planner, format_checklist, resolve_planner_llm
from .research import (
    query_needs_codebase_research,
    query_needs_session_search,
    query_needs_git_context,
    run_codebase_research,
    run_session_search,
    run_git_context,
)
from .prompts import build_system_prompt, build_answer_system_prompt
from .tools import dispatch, ToolError, normalize_tool_args


class AgentResponse(BaseModel):
    thought:  str  = ""
    action:   str
    tool:     str | None = None
    args:     dict | None = None
    content:  str | None = None


def _answer_text(content: Any) -> str:
    """Coerce LLM content (str or nested dict) to plain answer text."""
    from .formatting import format_structured_answer

    if content is None:
        return ""
    if isinstance(content, str):
        s = content.strip()
        if s.startswith("{"):
            try:
                return format_structured_answer(json.loads(s))
            except json.JSONDecodeError:
                pass
        return s
    if isinstance(content, dict):
        return format_structured_answer(content)
    return str(content).strip()


def _normalize_llm_response(data: dict) -> dict:
    from .tools import TOOL_REGISTRY

    if not isinstance(data, dict):
        return {}

    if "response" in data and isinstance(data["response"], dict) and not data.get("action"):
        inner = data["response"]
        sections = []
        for key, val in inner.items():
            if isinstance(val, str) and val.strip():
                sections.append(f"## {key.replace('_', ' ').title()}\n{val}")
        if sections:
            return {
                "thought":  str(data.get("thought", "")),
                "action":   "final_answer",
                "tool":     None,
                "args":     None,
                "content":  "\n\n".join(sections),
            }

    if not data.get("action"):
        c = data.get("content")
        if isinstance(c, dict):
            text = _answer_text(c)
            if text:
                return {
                    "thought": str(data.get("thought", "")),
                    "action": "final_answer",
                    "tool": None,
                    "args": None,
                    "content": text,
                }
        parts = []
        for key in ("strategy", "documentation", "implementation", "answer"):
            if isinstance(data.get(key), str) and data[key].strip():
                parts.append(f"## {key.replace('_', ' ').title()}\n{data[key]}")
        if parts:
            return {
                "thought": "",
                "action":   "final_answer",
                "tool":     None,
                "args":     None,
                "content":  "\n\n".join(parts),
            }
        from .formatting import format_structured_answer
        body = format_structured_answer(data)
        if body and not body.lstrip().startswith("{"):
            return {
                "thought": str(data.get("thought", "")),
                "action": "final_answer",
                "tool": None,
                "args": None,
                "content": body,
            }

    action = (data.get("action") or "").strip()
    tool   = data.get("tool")
    args   = data.get("args") if isinstance(data.get("args"), dict) else {}

    if action in TOOL_REGISTRY:
        return {**data, "action": "tool_call", "tool": action, "args": args, "content": None}

    if action == "tool_call" and isinstance(tool, str) and tool in TOOL_REGISTRY:
        return {**data, "action": "tool_call", "args": args}

    if action not in ("tool_call", "final_answer") and data.get("content") is not None:
        return {**data, "action": "final_answer", "content": _answer_text(data.get("content"))}

    if action == "final_answer":
        text = _answer_text(data.get("content"))
        if text:
            return {**data, "content": text}
        from .formatting import format_structured_answer
        extra = {k: v for k, v in data.items() if k not in ("action", "tool", "args", "thought")}
        if extra:
            return {**data, "content": format_structured_answer(extra)}

    return data


def _append_tool(
    session_path: Path,
    tool_results: list[dict],
    name: str,
    args: dict,
    result: Any,
) -> None:
    session_store.append_tool_call(session_path, name, args, result)
    tool_results.append({"name": name, "args": args, "result": result})


def _auto_validate(
    url: str,
    topic: str,
    user_query: str,
    session_path: Path,
    tool_results: list[dict],
    turn_cache: dict,
    project_root: Path,
    cache_dir: Path,
    session_dir: Path,
    embedder: Embedder,
    hitl_enabled: bool,
    verbose: bool,
) -> None:
    val_args = {"url": url, "topic": topic or "room", "user_query": user_query}
    key = f"android_docs_validate:{json.dumps(val_args, sort_keys=True)}"
    if key in turn_cache:
        result = turn_cache[key]
    else:
        result = dispatch(
            "android_docs_validate", val_args,
            project_root, cache_dir, session_dir, embedder, user_query,
            session_path=session_path,
            hitl_enabled=hitl_enabled,
        )
        turn_cache[key] = result
    if verbose:
        _rich_console.print("\n[bold yellow]🔧 android_docs_validate[/bold yellow] [dim](auto)[/dim]")
        _pretty(val_args, title="args")
        _pretty(result, title="result")
    _append_tool(session_path, tool_results, "android_docs_validate", val_args, result)


def _execute_planned_tools(
    plan: dict,
    *,
    session_path: Path,
    tool_results: list[dict],
    turn_cache: dict,
    preloaded_docs: dict[str, dict],
    project_root: Path,
    cache_dir: Path,
    session_dir: Path,
    embedder: Embedder,
    user_query: str,
    hitl_enabled: bool,
    verbose: bool,
    print_fn: Callable[[str], None],
) -> None:
    for spec in plan.get("tools") or []:
        if not isinstance(spec, dict):
            continue
        tool_name = spec.get("tool")
        # Writing requires content generated by the answer LLM — never run during planning phase
        if tool_name in ("write_file", "edit_file"):
            if verbose:
                print_fn(f"[codescope] skipping {tool_name} in plan phase (write happens in answer phase)")
            continue
        if not tool_name:
            continue
        args = dict(spec.get("args") or {})
        args = normalize_tool_args(tool_name, args)
        cache_key = f"{tool_name}:{json.dumps(args, sort_keys=True, ensure_ascii=False)}"
        if cache_key in turn_cache:
            result = turn_cache[cache_key]
        else:
            if verbose:
                print_fn(f"\n[codescope] tool: {tool_name}")
                _pretty(args, title="args")
            try:
                result = dispatch(
                    tool_name, args,
                    project_root, cache_dir, session_dir, embedder, user_query,
                    preloaded_docs=preloaded_docs,
                    session_path=session_path,
                    hitl_enabled=hitl_enabled,
                )
            except ToolError as e:
                result = {"error": str(e)}
            # Don't cache writes — each edit must actually run
            if tool_name not in ("write_file", "edit_file"):
                turn_cache[cache_key] = result
            if verbose:
                _pretty(result, title="result")

        _append_tool(session_path, tool_results, tool_name, args, result)

        if (
            tool_name == "android_docs"
            and AUTO_ANDROID_VALIDATE
            and isinstance(result, dict)
            and result.get("url")
        ):
            _auto_validate(
                result["url"],
                str(args.get("topic") or "room"),
                user_query,
                session_path,
                tool_results,
                turn_cache,
                project_root,
                cache_dir,
                session_dir,
                embedder,
                hitl_enabled,
                verbose,
            )


# ---------------------------------------------------------------------------
# Post-answer file-write hook
# ---------------------------------------------------------------------------

_EXPLAIN_PAT = _re.compile(
    r"\b(why|how|explain|compare|difference|trade.?off|pros? and cons?|should i use)\b",
    _re.IGNORECASE,
)
# Longer, implementation-heavy responses that often include code + rationale.
_IMPLEMENT_PAT = _re.compile(
    r"\b(implement|implementation|create|add|build|fix|error|compile|dao|repository|entity|schema|migration)\b",
    _re.IGNORECASE,
)

_CLASS_NAME_PAT = _re.compile(
    r"(?:create|write|add|implement|build|generate|modify|update|patch)\s+(?:the\s+)?(?:file\s+)?"
    r"((?:Patient|Doctor|Graph|Main|Ants|Timer|News|\w+)"
    r"(?:Activity|Fragment|ViewModel|Repository|Dao|Database|Entity|Composable))",
    _re.IGNORECASE,
)
_README_PAT = _re.compile(r"(?:create|write|add|generate)\s+(?:a\s+|the\s+)?"
                           r"(README(?:\.md)?)", _re.IGNORECASE)
_GENERIC_FILE_PAT = _re.compile(
    r"(?:create|write|add|modify|update|patch)\s+(?:the\s+file\s+)?"
    r"([A-Za-z0-9_\-]+\.[a-z]{2,4})",
    _re.IGNORECASE,
)
# Catches the "please write to the file" / "save to disk" / "apply the change"
# phrasing that does not name a file by extension but clearly asks for a write.
_WRITE_INTENT_PAT = _re.compile(
    r"\b("
    r"write\s+(?:to|it\s+to|them\s+to|out)\s+(?:the\s+)?(?:file|disk|files|the\s+kt)"
    r"|save\s+(?:to|it\s+to|the\s+changes\s+to|the\s+file)"
    r"|apply\s+the\s+change(?:s)?"
    r"|(?:update|modify|patch|edit)\s+the\s+(?:file|composable|composables|activity|viewmodel|entity|dao|repository|class|fragment|gradle|module)"
    r"|implement\s+the\s+(?:composable|composables|activity|viewmodel|repository|dao|entity|ui)"
    r")\b",
    _re.IGNORECASE,
)


_FILE_REF_PAT = _re.compile(
    r"\b([A-Za-z0-9_]+\.(?:kt|java|xml|kts|toml|gradle))\b", _re.IGNORECASE,
)
_ACTION_VERB_PAT = _re.compile(
    r"\b(?:add|insert|append|include|modify|update|implement|change|put|patch|"
    r"edit|replace|inject|wire|hook\s+up|set\s+up|introduce|refactor|rewrite|"
    r"create|write|build|generate|extend|extract|"
    r"fix|repair|resolve|correct|tidy|cleanup|clean\s+up|finish|complete)\b",
    _re.IGNORECASE,
)


def _extract_write_intent(user_query: str) -> str | None:
    """Return a bare filename (e.g. 'PatientActivity.kt', 'README.md') if the
    query asks to create or modify a file.

    Resolution order:
      1. `<verb> <ClassName(Activity|Fragment|…|Composable)>` — class-name verbs.
      2. Generic `<verb> <FILENAME.ext>` — verb immediately before a filename.
      3. `<action verb>` anywhere + a named source file anywhere — covers
         phrasings like 'add a Preview of PatientDashboard to PatientActivity.kt'
         or 'inject ClinicalDatabase into ClinicalApplication.kt'.
      4. README pattern.
      5. Write-intent phrasing without an explicit filename — returns the
         sentinel '<write-target>' (caller resolves via tool_results).
    """
    m = _CLASS_NAME_PAT.search(user_query)
    if m:
        name = m.group(1)
        return name if name.endswith((".kt", ".java")) else name + ".kt"
    m = _GENERIC_FILE_PAT.search(user_query)
    if m:
        return m.group(1)
    # Action verb anywhere + a filename anywhere: "add a Preview … to PatientActivity.kt"
    if _ACTION_VERB_PAT.search(user_query):
        m = _FILE_REF_PAT.search(user_query)
        if m:
            return m.group(1)
    m = _README_PAT.search(user_query)
    if m:
        return "README.md"
    if _WRITE_INTENT_PAT.search(user_query):
        return "<write-target>"  # sentinel — write task without an explicit filename
    return None


def _extract_code_blocks(text: str) -> list[str]:
    """Return all fenced code-block bodies, longest first."""
    blocks = _re.findall(
        r"```(?:kotlin|java|xml|groovy|kts|markdown|md|)?\n?(.*?)```",
        text, _re.DOTALL,
    )
    return sorted([b.strip() for b in blocks if len(b.strip()) > 60], key=len, reverse=True)


# Matches the stage-2 file-section pattern:  ### <path>\n```<lang>\n<body>\n```
# The path heading must reference a file with a recognised source/config
# extension (excludes generic markdown ### titles like '### Files to modify').
_FILE_SECTION_PAT = _re.compile(
    r"^###\s+(?P<path>[A-Za-z0-9_\-./]+\.(?:kt|java|xml|kts|toml|gradle|md))\s*\n+"
    r"```(?P<lang>[a-zA-Z]*)\n(?P<body>.*?)\n```",
    _re.DOTALL | _re.MULTILINE,
)


def _parse_file_writes(answer: str) -> list[tuple[str, str]]:
    """Pull stage-2 style `### <path>` + fenced code block sections from
    `answer`. Returns a list of (project_relative_path, file_body) pairs.
    Returns [] if the answer doesn't follow that shape."""
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    for m in _FILE_SECTION_PAT.finditer(answer):
        path = m.group("path").replace("\\", "/").lstrip("./")
        body = m.group("body").rstrip()
        if not body or len(body) < 30:
            continue
        # Sometimes the body itself starts with an inadvertent extra fence.
        if body.startswith("```"):
            body = body.split("\n", 1)[1] if "\n" in body else body
        if path in seen:
            continue
        seen.add(path)
        out.append((path, body))
    return out


def _infer_kt_path(filename: str, tool_results: list[dict], project_root: Path) -> str | None:
    """Resolve a bare .kt filename to a full relative path using existing file hits."""
    candidate_dirs: list[str] = []
    for tr in tool_results:
        result = tr.get("result")
        hits = result if isinstance(result, list) else [result] if isinstance(result, dict) else []
        for h in hits:
            if not isinstance(h, dict):
                continue
            for key in ("file", "path"):
                f = str(h.get(key, "")).replace("\\", "/")
                if f.endswith(".kt") and "src/main/java" in f:
                    candidate_dirs.append("/".join(f.split("/")[:-1]))
    if not candidate_dirs:
        # Fall back to walking the project
        for kt in project_root.rglob("*.kt"):
            rel = kt.relative_to(project_root).as_posix()
            if "src/main/java" in rel:
                candidate_dirs.append("/".join(rel.split("/")[:-1]))
    if candidate_dirs:
        # Use the most common dir (usually the app package dir)
        from collections import Counter
        best = Counter(candidate_dirs).most_common(1)[0][0]
        return f"{best}/{filename}"
    return None


def _resolve_target_path(
    filename: str,
    tool_results: list[dict],
    project_root: Path,
) -> str | None:
    """Map a bare/sentinel filename to a project-relative path.

      • Real path (already has '/' and exists)  → use as-is.
      • README                                  → README.md at root.
      • Bare .kt/.java filename                 → search preflight reads /
                                                  graph hits, else _infer_kt_path.
      • '<write-target>' sentinel               → the most recently preflighted
                                                  source file (the file the
                                                  question is about).
    """
    if filename == "README.md":
        return "README.md"

    if filename == "<write-target>":
        # Prefer the last preflight-read source file — that's the file the
        # user almost certainly wants edited.
        for tr in reversed(tool_results):
            if tr.get("name") == "read_file" and tr.get("preflight"):
                p = (tr.get("result") or {}).get("path") or (tr.get("args") or {}).get("path")
                if p:
                    return str(p).replace("\\", "/").lstrip("./")
        return None

    cleaned = filename.replace("\\", "/").lstrip("./")
    if "/" in cleaned and (project_root / cleaned).is_file():
        return cleaned

    # Try matching against any path we've already read this turn.
    base = Path(cleaned).name
    for tr in tool_results:
        if tr.get("name") != "read_file":
            continue
        p = (tr.get("result") or {}).get("path") or (tr.get("args") or {}).get("path")
        if p and Path(str(p)).name == base:
            return str(p).replace("\\", "/").lstrip("./")

    if cleaned.endswith((".kt", ".java")):
        inferred = _infer_kt_path(base, tool_results, project_root)
        if inferred:
            return inferred

    return cleaned if "/" in cleaned else None


def _maybe_write_file(
    user_query: str,
    answer: str,
    tool_results: list[dict],
    project_root: Path,
    session_path: Path,
    verbose: bool,
    print_fn,
) -> str:
    """After final_answer, detect file-write intent and write the code block(s).

    Two paths:
      1. Stage-2 multi-file shape: `### <path>` + fenced code block, repeated.
         All matching sections are written.
      2. Single-file legacy shape: longest fenced code block goes to the file
         inferred from the query (`_extract_write_intent` + `_resolve_target_path`).
    """
    from .tools import tool_write_file, ToolError as _TE

    if not _extract_write_intent(user_query):
        return answer

    written: list[tuple[str, int, str]] = []  # (path, bytes, action)

    # ── Path 1: stage-2 multi-file sections ──────────────────────────────────
    sections = _parse_file_writes(answer)
    if sections:
        for raw_path, body in sections:
            target = _resolve_target_path(raw_path, tool_results, project_root)
            if not target:
                if verbose:
                    print_fn(f"[codescope] auto-write: cannot resolve path for {raw_path}")
                continue
            try:
                result = tool_write_file(
                    path=target,
                    content=body,
                    overwrite=True,
                    project_root=project_root,
                    session_path=session_path,
                )
                if result.get("status") == "ok":
                    written.append((target, result.get("bytes", 0), result.get("action", "written")))
                    print_fn(
                        f"[codescope] auto-wrote {target} "
                        f"({result.get('bytes', 0)} bytes)"
                    )
            except _TE as e:
                print_fn(f"[codescope] auto-write failed for {target}: {e}")

        if written:
            footer = "\n\n---\n" + "\n".join(
                f"**File {a}:** `{p}` ({b} bytes)" for p, b, a in written
            )
            return answer + footer
        # If sections found but nothing written successfully, fall through to
        # the legacy single-file path as a last resort.

    # ── Path 2: legacy single-file ───────────────────────────────────────────
    filename = _extract_write_intent(user_query) or ""
    blocks = _extract_code_blocks(answer)
    if not blocks and filename == "README.md" and answer.strip().startswith("#"):
        blocks = [answer.strip()]
    if not blocks:
        if verbose:
            print_fn(f"[codescope] auto-write: no code block found for {filename}")
        return answer

    content = blocks[0]  # longest block
    target = _resolve_target_path(filename, tool_results, project_root)
    if not target:
        if verbose:
            print_fn(f"[codescope] auto-write: cannot infer path for {filename}")
        return answer

    try:
        result = tool_write_file(
            path=target,
            content=content,
            overwrite=True,
            project_root=project_root,
            session_path=session_path,
        )
        if result.get("status") == "ok":
            action = result.get("action", "written")
            print_fn(
                f"[codescope] auto-wrote {target} ({result.get('bytes', 0)} bytes)"
            )
            return answer + f"\n\n---\n**File {action}:** `{target}` ({result.get('bytes', 0)} bytes)"
    except _TE as e:
        if verbose:
            print_fn(f"[codescope] auto-write failed: {e}")
    return answer


def _is_write_task(user_query: str) -> bool:
    return _extract_write_intent(user_query) is not None


def _extract_partial_content(raw: str) -> str | None:
    """
    Recover readable text from a truncated JSON response.

    Handles the common case where the model stops generating mid-string:
      {"action":"final_answer","content":"Here is the explanation...
    Returns the partial content string with JSON escape sequences decoded,
    or None if nothing useful can be extracted.
    """
    # Find the start of the content value
    idx = raw.find('"content"')
    if idx == -1:
        return None
    # Skip past "content": "
    idx = raw.find('"', idx + 9)   # opening quote of value
    if idx == -1:
        return None
    idx += 1  # first char of value

    # Collect chars up to the end, respecting JSON escape sequences
    chars: list[str] = []
    i = idx
    while i < len(raw):
        c = raw[i]
        if c == "\\" and i + 1 < len(raw):
            nxt = raw[i + 1]
            escape_map = {"n": "\n", "t": "\t", "r": "\r", '"': '"', "\\": "\\", "/": "/"}
            chars.append(escape_map.get(nxt, nxt))
            i += 2
        elif c == '"':
            break   # clean end of JSON string
        else:
            chars.append(c)
            i += 1

    result = "".join(chars).strip()
    return result if len(result) > 20 else None


# --------------------------------------------------------------------------
# Truncation recovery
# --------------------------------------------------------------------------

def _looks_truncated(answer: str) -> bool:
    """Heuristically detect whether an answer was cut off by num_predict.

    Triggers when:
      • Odd number of ``` fences (an open code block at end).
      • Ends without a sentence-ending character AND without a closing fence.
      • Last 200 chars contain an obviously-incomplete identifier (open `(` /
        `{` count exceeds close count by 3+).
    """
    if not answer or not answer.strip():
        return False
    text = answer.rstrip()
    fence_count = text.count("```")
    if fence_count % 2 == 1:
        return True
    last_tail = text[-200:]
    open_paren = last_tail.count("(") - last_tail.count(")")
    open_brace = last_tail.count("{") - last_tail.count("}")
    if open_paren >= 3 or open_brace >= 3:
        return True
    if not text.endswith(("```", "}", ")", "]", ".", "!", "?", "—", ":")):
        # Trailing word-character with no terminator: likely cut mid-sentence.
        if text[-1].isalnum() or text[-1] in "-_,/":
            return True
    return False


_CONTINUATION_INSTRUCTION_CLOSED = (
    "The previous response was truncated. Continue EXACTLY where it left off; "
    "do not repeat or summarize any earlier text. End the answer cleanly. "
    "Return plain markdown (no JSON wrapper this time)."
)
_CONTINUATION_INSTRUCTION_OPEN_FENCE = (
    "The previous response was truncated WHILE INSIDE AN OPEN CODE BLOCK. "
    "Your continuation MUST:\n"
    "  • Resume writing CODE on the next line — do NOT emit ``` at the start. "
    "You are still inside the open fence.\n"
    "  • When the code body is finished, close the fence with a line "
    "containing exactly ``` and nothing else.\n"
    "  • Then add at most one short closing sentence in plain markdown.\n"
    "Do not restate or summarize anything from the partial answer."
)


def _detect_open_fence_lang(partial: str) -> str | None:
    """If the partial ends inside an open ``` fence, return the language tag
    that opened it (e.g. 'kotlin'), else None."""
    fences = list(_re.finditer(r"```([a-zA-Z]*)", partial))
    if not fences:
        return None
    # Odd number of fences => last one is unclosed.
    if len(fences) % 2 == 0:
        return None
    last = fences[-1]
    return (last.group(1) or "").strip() or ""


def _strip_leading_reopened_fence(cont: str, lang: str | None) -> str:
    """Stage-2 continuation requests qwen NOT to re-open a fence, but it
    sometimes does it anyway (`\\n```kotlin\\n`). Strip such a leading reopen so
    we don't end up with `<code>\\n```kotlin\\n<more code>` glued together."""
    if not cont:
        return cont
    stripped = cont.lstrip()
    # Possible reopen patterns: ```\n / ```kotlin\n / ```kt\n / language\n
    m = _re.match(r"^```([a-zA-Z]*)\s*\n", stripped)
    if not m:
        return cont
    fence_lang = (m.group(1) or "").strip()
    # Only strip the reopen if the language matches the open fence (or both empty).
    if lang is None or fence_lang == "" or fence_lang.lower() == (lang or "").lower():
        return stripped[m.end():]
    return cont


def _attempt_continuation(
    llm: LLM,
    partial_answer: str,
    num_predict: int,
    print_fn: Callable[[str], None],
) -> str:
    """Ask the answer LLM to continue a truncated response. Returns the
    concatenated, hopefully-complete answer. On failure returns the partial
    plus a visible truncation note.
    """
    open_fence_lang = _detect_open_fence_lang(partial_answer)
    inside_fence = open_fence_lang is not None

    print_fn(
        f"[codescope] Answer truncated"
        + (" mid-code-block" if inside_fence else "")
        + f" — continuing with {llm.model} (num_predict={num_predict})…"
    )

    primer_intro = (
        "Here is the partial answer so far. Continue it from the exact point "
        "it stopped — do not restate anything that's already there."
    )
    if inside_fence:
        primer_intro += (
            f"\nThe partial ends INSIDE an open ```{open_fence_lang} code "
            "block. Resume by writing the next line of code (no opening "
            "fence). Close the fence with ``` when the code body is done."
        )

    primer = (
        primer_intro
        + "\n\n=== PARTIAL ANSWER (continue from here) ===\n"
        + partial_answer.rstrip()
        + "\n=== CONTINUE BELOW ==="
    )
    system = (
        _CONTINUATION_INSTRUCTION_OPEN_FENCE
        if inside_fence
        else _CONTINUATION_INSTRUCTION_CLOSED
    )
    try:
        cont = llm.generate(
            prompt=primer,
            system=system,
            json_mode=False,
            num_predict=num_predict,
            temperature=0.15,
        )
    except OllamaError as e:
        print_fn(f"[codescope] Continuation failed: {e}.")
        return partial_answer + (
            "\n\n*(response was truncated and continuation failed — try "
            "CODESCOPE_NUM_PREDICT_WRITE=12000 and retry)*"
        )
    cont = (cont or "")
    if not cont.strip():
        return partial_answer + (
            "\n\n*(response was truncated; continuation returned empty)*"
        )

    if inside_fence:
        cont = _strip_leading_reopened_fence(cont, open_fence_lang)

    glued = partial_answer.rstrip() + ("\n" if not partial_answer.endswith("\n") else "") + cont
    # Final safety net: ensure every fence is balanced.
    if glued.count("```") % 2 == 1:
        glued = glued.rstrip() + "\n```"
    return glued


# Keywords used to map completed phases to checklist items. The heuristic is
# coarse but useful: when "research" ran, any checklist item mentioning grep /
# semantic / graph / search gets a check mark. The user still sees raw phase
# banners above so this is just a tidy recap.
_PHASE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "preread":  ("read", "review", "inspect", "open", "load"),
    "plan":     ("plan", "draft plan", "outline"),
    "research": ("grep", "semantic", "graph", "search", "find", "locate", "explore"),
    "session":  ("prior", "session", "previous turn", "earlier"),
    "git":      ("git", "diff", "commit", "log"),
    "answer":   ("answer", "respond", "synthes", "write", "explain", "summari"),
}


def _mark_checklist_done(plan: dict, phases_done: set[str]) -> dict:
    """Return a shallow copy of `plan` with checklist statuses updated to 'done'
    for any item whose task text contains a keyword from a completed phase."""
    import copy
    new_plan = copy.deepcopy(plan)
    for item in new_plan.get("checklist") or []:
        if not isinstance(item, dict):
            continue
        if item.get("status") == "done":
            continue
        task_lower = str(item.get("task", "")).lower()
        for phase in phases_done:
            if any(kw in task_lower for kw in _PHASE_KEYWORDS.get(phase, ())):
                item["status"] = "done"
                break
    return new_plan


def _run_turn(
    user_query: str,
    session_path: Path,
    system_prompt: str,
    llm: LLM,
    project_root: Path,
    cache_dir: Path,
    session_dir: Path,
    embedder: Embedder,
    verbose: bool = False,
    print_fn: Callable[[str], None] = print,
    planner_llm: LLM | None = None,
    graph_summary: str = "",
    git_summary: str = "",
    hitl_enabled: bool = False,
) -> str:
    session_store.append_user(session_path, user_query)

    history = session_store.last_n_turns(session_path, 4)
    history = [e for e in history if not (e.get("role") == "user" and e.get("content") == user_query)]

    tool_results: list[dict] = []
    turn_cache: dict[str, Any] = {}
    phases_done: set[str] = set()  # tracks phases that ran for the final checklist update

    preflight_project_docs(user_query, project_root, tool_results, turn_cache)
    source_files_read = preflight_source_files(user_query, project_root, tool_results, turn_cache)
    if source_files_read:
        phases_done.add("preread")
    preflight_android_docs(
        user_query, cache_dir, tool_results, turn_cache,
        doc_context_fn=lambda: "",
    )
    preloaded_docs = preloaded_doc_index(tool_results)
    preloaded_files = list(preloaded_docs.keys()) or extract_doc_paths(
        user_query, project_root
    )

    if preloaded_files:
        print_fn(f"[codescope] Pre-loaded {len(preloaded_files)} project doc(s) from query.")
    if source_files_read:
        print_fn(
            f"[codescope] Pre-read {len(source_files_read)} source file(s): "
            + ", ".join(source_files_read)
        )

    if not graph_summary:
        graph_summary = build_graph_summary(cache_dir)

    # Planner: deterministic by default (no second model load). Optional USE_PLANNER_LLM=1.
    effective_planner: LLM | None = None
    if USE_PLANNER_LLM:
        if PLANNER_USE_ANSWER_MODEL:
            effective_planner = llm
        elif planner_llm is not None:
            effective_planner = planner_llm
        else:
            effective_planner = resolve_planner_llm()

    planner_label = (
        effective_planner.model if effective_planner else "deterministic (no LLM)"
    )
    print_fn(f"[codescope] Planning ({planner_label})…")

    plan = run_planner(
        user_query,
        graph_summary,
        preloaded_files,
        planner_llm=effective_planner,
        project_root=project_root,
    )

    session_state.init_turn(
        session_path, user_query, plan.get("checklist") or [], graph_summary
    )
    session_state.append_entry(
        session_path,
        phase="planner",
        task="Plan turn",
        notes=format_checklist(plan),
        metadata={"tools": [t.get("tool") for t in plan.get("tools") or [] if isinstance(t, dict)]},
    )

    # Always print the checklist so the user can see what the agent committed to.
    checklist_text = format_checklist(plan)
    if checklist_text.strip():
        print_fn(checklist_text)

    _execute_planned_tools(
        plan,
        session_path=session_path,
        tool_results=tool_results,
        turn_cache=turn_cache,
        preloaded_docs=preloaded_docs,
        project_root=project_root,
        cache_dir=cache_dir,
        session_dir=session_dir,
        embedder=embedder,
        user_query=user_query,
        hitl_enabled=hitl_enabled,
        verbose=verbose,
        print_fn=print_fn,
            )

    phases_done.add("plan")

    if query_needs_codebase_research(user_query):
        phases_done.add("research")
        print_fn("[codescope] Codebase research (grep + semantic + graph)…")
        run_codebase_research(
            user_query,
            session_path,
            project_root,
            cache_dir,
            session_dir,
            embedder,
            graph_summary,
            preloaded_docs,
            tool_results,
            turn_cache,
            verbose=verbose,
            print_fn=print_fn,
        )

    # Always pull session history when query references a prior turn
    if query_needs_session_search(user_query):
        phases_done.add("session")
        print_fn("[codescope] Searching prior session turns…")
        run_session_search(
            user_query,
            session_path,
            session_dir,
            tool_results,
            turn_cache,
            verbose=verbose,
            print_fn=print_fn,
        )

    # Pull git diff + log when the query asks about recent changes, OR always as context
    if query_needs_git_context(user_query):
        phases_done.add("git")
        print_fn("[codescope] Pulling git context…")
        run_git_context(
            user_query,
            session_path,
            project_root,
            tool_results,
            turn_cache,
            verbose=verbose,
            print_fn=print_fn,
        )

    answer_system = build_answer_system_prompt()
    prompt = build_answer_prompt(
        user_query=user_query,
        checklist=format_checklist(plan),
        tool_results=tool_results,
        history=history,
        session_notes=session_state.format_for_prompt(session_path),
        graph_summary=graph_summary,
        git_summary=git_summary,
        embedder=embedder,
    )

    is_write = _is_write_task(user_query)
    if is_write:
        answer_num_predict = ANSWER_NUM_PREDICT_WRITE
        mode_label = "write"
    elif _EXPLAIN_PAT.search(user_query) or _IMPLEMENT_PAT.search(user_query):
        # Implementation/debug prompts usually need long prose + code snippets.
        answer_num_predict = ANSWER_NUM_PREDICT_EXPLAIN
        mode_label = "explain"
    else:
        answer_num_predict = ANSWER_NUM_PREDICT
        mode_label = "answer"

    # Updated checklist: mark the items whose phases ran. The answer phase itself
    # is marked optimistically so the user sees a fully-checked list when the
    # LLM finishes — they'll see the actual answer immediately after.
    phases_done.add("answer")
    updated_plan = _mark_checklist_done(plan, phases_done)
    updated_checklist = format_checklist(updated_plan)
    if updated_checklist.strip() and updated_checklist != checklist_text:
        print_fn("[codescope] Checklist progress:")
        print_fn(updated_checklist)

    # ── Two-stage write flow ─────────────────────────────────────────────────
    # For write tasks on a capable model, first ask the same LLM for a compact
    # markdown implementation plan, pretty-print it, then re-call it with the
    # plan injected and a strict "emit full file bodies under ### <path> headings"
    # system suffix so stage 2 actually emits code (not a plan recap).
    plan_md: str | None = None
    if is_write and TWO_STAGE_WRITE:
        from .write_flow import run_plan_stage, inject_plan_into_prompt, build_stage2_system

        plan_md = run_plan_stage(
            llm=llm,
            base_prompt=prompt,
            print_fn=print_fn,
            plan_num_predict=TWO_STAGE_PLAN_NUM_PREDICT,
        )
        if plan_md:
            prompt = inject_plan_into_prompt(prompt, plan_md)
            answer_system = build_stage2_system(answer_system)
            print_fn(
                f"[codescope] Stage 2/2: implementing per plan with {llm.model} "
                f"(num_predict={answer_num_predict})…"
            )
        else:
            # Plan stage failed/empty — fall back to a normal single-stage call.
            print_fn(
                f"[codescope] Generating {mode_label} with {llm.model} "
                f"(num_predict={answer_num_predict})…"
            )
    else:
        # Always announce the answer model — this is the slow phase the user is waiting on.
        print_fn(
            f"[codescope] Generating {mode_label} with {llm.model} "
            f"(num_predict={answer_num_predict})…"
        )

    try:
        raw = llm.generate(
            prompt=prompt,
            system=answer_system,
            json_mode=True,
            num_predict=answer_num_predict,
            temperature=0.2,
        )
    except OllamaError as e:
        msg = f"LLM error: {e}"
        session_store.append_assistant(session_path, msg)
        return msg

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        import re as _re
        m = _re.search(r"\{.*\}", raw, _re.DOTALL)
        data = json.loads(m.group(0)) if m else {}

    data = _normalize_llm_response(data)
    try:
        resp = AgentResponse.model_validate(data)
        validated = True
    except ValidationError:
        resp = None
        validated = False

    thought = ""
    if validated and resp is not None and resp.action == "tool_call":
        tool_name = resp.tool or ""
        args = normalize_tool_args(tool_name, dict(resp.args or {}))
        if verbose:
            print_fn(f"\n[codescope] Optional tool: {tool_name}")
        try:
            result = dispatch(
                tool_name, args,
                project_root, cache_dir, session_dir, embedder, user_query,
                preloaded_docs=preloaded_docs,
                session_path=session_path,
                hitl_enabled=hitl_enabled,
            )
        except ToolError as e:
            result = {"error": str(e)}
        _append_tool(session_path, tool_results, tool_name, args, result)
        retry_prompt = build_answer_prompt(
            user_query=user_query,
            checklist=format_checklist(plan),
            tool_results=tool_results,
            history=history,
            session_notes=session_state.format_for_prompt(session_path),
            graph_summary=graph_summary,
            git_summary=git_summary,
            embedder=embedder,
        )
        try:
            raw2 = llm.generate(
                prompt=retry_prompt + "\n\nNow respond with final_answer only.",
                system=answer_system,
                json_mode=True,
                num_predict=answer_num_predict,
                temperature=0.2,
            )
            data2 = _normalize_llm_response(json.loads(raw2))
            answer = data2.get("content") or raw2
        except Exception:
            answer = str(result)
        thought = resp.thought or ""
    elif validated and resp is not None:
        answer = _answer_text(resp.content)
        thought = resp.thought or ""
    else:
        # Validation failed — usually because the JSON was truncated mid-string.
        # Salvage the partial `content` first, then let the truncation
        # continuation recover the rest.
        from .formatting import coerce_answer_text
        salvaged = _extract_partial_content(raw)
        if salvaged:
            answer = salvaged
        else:
            try:
                answer = coerce_answer_text(data)
            except Exception:
                answer = raw
        if not (answer or "").strip():
            answer = raw

    # Truncation recovery: applies to EVERY path (final_answer, tool_call retry,
    # AND the ValidationError salvage path). When num_predict ran out
    # mid-code-block or mid-sentence, call the LLM once more in plain-markdown
    # mode to finish.
    if mode_label in ("write", "explain") and _looks_truncated(answer):
        # Use a generous fraction of the original budget for the continuation.
        # Capped at 6000 tokens for the 64K context host.
        cont_budget = max(2000, min(answer_num_predict // 2, 6000))
        answer = _attempt_continuation(llm, answer, cont_budget, print_fn)

    # Post-answer hook: if the task asked to create a file and the model produced
    # a code block in its answer (rather than calling write_file via tool_call),
    # extract and write it automatically.
    answer = _maybe_write_file(
        user_query, answer, tool_results, project_root, session_path, verbose, print_fn
    )

    session_store.append_assistant(session_path, answer, thought=thought)
    return answer


def make_session(
    slug: str,
    cache_dir: Path,
    session_dir: Path,
    llm: LLM,
    embedder: Embedder,
    session_path: Path | None = None,
    project_root: Path | None = None,
) -> tuple[str, Path, str]:
    """Returns (system_prompt, session_path, git_summary)."""
    graph_summary = build_graph_summary(cache_dir)
    git_summary = get_git_log(project_root) if project_root else ""
    system_prompt = build_system_prompt(slug, graph_summary, git_log=git_summary)
    if session_path is None:
        latest = session_store.latest(session_dir)
        session_path = latest if latest else session_store.new_path(session_dir)
    return system_prompt, session_path, git_summary


def _setup_readline(session_dir: Path) -> None:
    """
    Activate GNU readline for the REPL input prompt.

    Provides: arrow-key cursor movement, Ctrl+A/E (line start/end),
    Ctrl+W (delete word), Ctrl+U (clear line), Up/Down history navigation,
    and persistent history across sessions stored in session_dir.

    Falls back silently on Windows or if readline is unavailable.
    """
    try:
        import readline as _rl
    except ImportError:
        return  # Windows without pyreadline — plain input() still works

    history_file = session_dir / ".repl_history"

    # Load existing history
    try:
        _rl.read_history_file(str(history_file))
    except FileNotFoundError:
        pass

    _rl.set_history_length(500)

    import atexit
    atexit.register(_rl.write_history_file, str(history_file))

    # Vi-style tab completion is off; keep default emacs bindings
    _rl.parse_and_bind("tab: complete")


def repl(
    slug: str,
    project_root: Path,
    cache_dir: Path,
    session_dir: Path,
    llm: LLM,
    embedder: Embedder,
    session_path: Path | None = None,
    verbose: bool = False,
    new_session: bool = False,
    hitl_enabled: bool = False,
) -> None:
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.syntax import Syntax
    from rich.panel import Panel

    _setup_readline(session_dir)

    console = Console()
    graph_summary = build_graph_summary(cache_dir)
    planner_llm = resolve_planner_llm() if USE_PLANNER_LLM and not PLANNER_USE_ANSWER_MODEL else None

    if not embedder.is_loaded():
        console.print(
            f"[dim]Loading embedding weights ({embedder.model_name} on {embedder.device})…[/dim]"
        )
        try:
            embedder.warm()
        except Exception as e:
            console.print(f"[red]Embedder failed to load: {e}[/red]")

    if WARM_MODEL_AT_REPL:
        remote = "127.0.0.1" not in llm.base_url and "localhost" not in llm.base_url
        hint = " (remote — first load may take 1–3 min)" if remote else ""
        console.print(f"[dim]Warming answer model {llm.model}{hint}…[/dim]")
        llm.warm()
        if planner_llm is not None:
            console.print(f"[dim]Warming planner {planner_llm.model}…[/dim]")
            planner_llm.warm()

    def _render_answer(text: str) -> None:
        from .formatting import coerce_answer_text, format_terminal_answer

        stripped = text.strip()
        if stripped.startswith("{"):
            try:
                text = coerce_answer_text(json.loads(stripped))
                stripped = text
            except Exception:
                # JSON was truncated mid-generation — try to salvage the content field
                recovered = _extract_partial_content(stripped)
                if recovered:
                    stripped = (
                        recovered
                        + "\n\n*(response was truncated — set CODESCOPE_NUM_PREDICT_EXPLAIN=6000 "
                        "or CODESCOPE_NUM_PREDICT and retry)*"
                    )
                text = stripped
        text = format_terminal_answer(stripped)
        try:
            console.print(Markdown(text))
        except Exception:
            console.print(text)

    if new_session or session_path is None:
        session_path = session_store.new_path(session_dir)
    else:
        latest = session_store.latest(session_dir)
        session_path = latest if (latest and not new_session) else session_store.new_path(session_dir)

    system_prompt, session_path, git_summary = make_session(
        slug, cache_dir, session_dir, llm, embedder, session_path,
        project_root=project_root,
    )

    from ..config import _PROFILE, OLLAMA_NUM_CTX, OLLAMA_BASE_URL, LLM_TIMEOUT
    console.print(f"\n[bold green]codescope[/bold green] — project: [cyan]{slug}[/cyan]")
    console.print(
        f"Session: [dim]{session_path.name}[/dim]  Profile: [dim]{_PROFILE}[/dim]  "
        f"ctx: [dim]{OLLAMA_NUM_CTX:,}[/dim]"
    )
    console.print(f"Ollama: [dim]{OLLAMA_BASE_URL}[/dim]  timeout: [dim]{int(LLM_TIMEOUT)}s[/dim]")
    if USE_PLANNER_LLM:
        pl = llm.model if PLANNER_USE_ANSWER_MODEL else (planner_llm.model if planner_llm else PLANNER_LLM)
        console.print(f"Planner: [dim]{pl}[/dim]  Answer: [dim]{llm.model}[/dim]")
        if PLANNER_USE_ANSWER_MODEL and _PROFILE == "workstation-24gb":
            console.print(
                "[dim]Tip: unset CODESCOPE_PLANNER_SAME_MODEL to use fast llama3.2:3b planner[/dim]"
            )
    else:
        console.print(f"Planner: [dim]deterministic[/dim]  Answer: [dim]{llm.model}[/dim]")
    console.print("Type [bold]exit[/bold] or [bold]quit[/bold] to leave.\n")

    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye.[/dim]")
            break

        if not query:
            continue
        if query.lower() in ("exit", "quit", "/exit", "/quit"):
            console.print("[dim]Goodbye.[/dim]")
            break

        import re as _re
        m = _re.match(r"^android_docs\s*[:\s]\s*(.+)$", query, _re.IGNORECASE)
        if m:
            from .android_docs import fetch_android_docs, format_doc_terminal
            topics = [t.strip() for t in _re.split(r"[,;]+", m.group(1)) if t.strip()]
            console.print(f"[dim]Fetching Android docs: {', '.join(topics)}…[/dim]\n")
            for doc in fetch_android_docs(topics, cache_dir=cache_dir):
                console.print(Markdown(format_doc_terminal(doc)))
                console.print()
            continue

        console.print("[dim]Thinking…[/dim]")
        answer = _run_turn(
            user_query=query,
            session_path=session_path,
            system_prompt=system_prompt,
            llm=llm,
            project_root=project_root,
            cache_dir=cache_dir,
            session_dir=session_dir,
            embedder=embedder,
            verbose=verbose,
            print_fn=console.print,
            planner_llm=planner_llm if USE_PLANNER_LLM else None,
            graph_summary=graph_summary,
            git_summary=git_summary,
            hitl_enabled=hitl_enabled,
        )

        console.print("\n[bold]codescope:[/bold]")
        _render_answer(answer)
        console.print()


def ask_once(
    query: str,
    slug: str,
    project_root: Path,
    cache_dir: Path,
    session_dir: Path,
    llm: LLM,
    embedder: Embedder,
    verbose: bool = False,
    hitl_enabled: bool = False,
) -> str:
    session_path = session_store.new_path(session_dir)
    system_prompt, session_path, git_summary = make_session(
        slug, cache_dir, session_dir, llm, embedder, session_path,
        project_root=project_root,
    )
    return _run_turn(
        user_query=query,
        session_path=session_path,
        system_prompt=system_prompt,
        llm=llm,
        project_root=project_root,
        cache_dir=cache_dir,
        session_dir=session_dir,
        embedder=embedder,
        verbose=verbose,
        graph_summary=build_graph_summary(cache_dir),
        git_summary=git_summary,
        hitl_enabled=hitl_enabled,
    )
