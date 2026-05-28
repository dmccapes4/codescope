"""Agent: preflight docs → planner (fast LLM) → run tools → answer (main LLM)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

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
    r"(?:create|write|add|implement|build|generate)\s+(?:the\s+)?(?:file\s+)?"
    r"((?:Patient|Doctor|Graph|Main|Ants|Timer|News|\w+)"
    r"(?:Activity|Fragment|ViewModel|Repository|Dao|Database|Entity))",
    _re.IGNORECASE,
)
_README_PAT = _re.compile(r"(?:create|write|add|generate)\s+(?:a\s+|the\s+)?"
                           r"(README(?:\.md)?)", _re.IGNORECASE)
_GENERIC_FILE_PAT = _re.compile(
    r"(?:create|write|add)\s+(?:the\s+file\s+)?([A-Za-z0-9_\-]+\.[a-z]{2,4})",
    _re.IGNORECASE,
)


def _extract_write_intent(user_query: str) -> str | None:
    """Return a bare filename (e.g. 'PatientActivity.kt', 'README.md') if the
    query asks to create a file, else None."""
    m = _CLASS_NAME_PAT.search(user_query)
    if m:
        name = m.group(1)
        return name if name.endswith((".kt", ".java")) else name + ".kt"
    m = _README_PAT.search(user_query)
    if m:
        return "README.md"
    m = _GENERIC_FILE_PAT.search(user_query)
    if m:
        return m.group(1)
    return None


def _extract_code_blocks(text: str) -> list[str]:
    """Return all fenced code-block bodies, longest first."""
    blocks = _re.findall(
        r"```(?:kotlin|java|xml|groovy|kts|markdown|md|)?\n?(.*?)```",
        text, _re.DOTALL,
    )
    return sorted([b.strip() for b in blocks if len(b.strip()) > 60], key=len, reverse=True)


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


def _maybe_write_file(
    user_query: str,
    answer: str,
    tool_results: list[dict],
    project_root: Path,
    session_path: Path,
    verbose: bool,
    print_fn,
) -> str:
    """After final_answer, detect file-creation intent and write the code block."""
    from .tools import tool_write_file, ToolError as _TE

    filename = _extract_write_intent(user_query)
    if not filename:
        return answer

    # Determine content: prefer explicit code blocks; for README accept raw markdown
    blocks = _extract_code_blocks(answer)
    if not blocks and filename == "README.md" and answer.strip().startswith("#"):
        blocks = [answer.strip()]
    if not blocks:
        if verbose:
            print_fn(f"[codescope] auto-write: no code block found for {filename}")
        return answer

    content = blocks[0]  # longest block

    # Resolve path
    if filename == "README.md":
        path = "README.md"
    elif filename.endswith(".kt") or filename.endswith(".java"):
        path = _infer_kt_path(filename, tool_results, project_root)
        if not path:
            if verbose:
                print_fn(f"[codescope] auto-write: cannot infer path for {filename}")
            return answer
    else:
        path = filename

    try:
        result = tool_write_file(
            path=path,
            content=content,
            overwrite=True,
            project_root=project_root,
            session_path=session_path,
        )
        if result.get("status") == "ok":
            action = result.get("action", "written")
            if verbose:
                print_fn(f"[codescope] auto-wrote {path} ({result.get('bytes', 0)} bytes)")
            return answer + f"\n\n---\n**File {action}:** `{path}` ({result.get('bytes', 0)} bytes)"
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

    preflight_project_docs(user_query, project_root, tool_results, turn_cache)
    source_files_read = preflight_source_files(user_query, project_root, tool_results, turn_cache)
    preflight_android_docs(
        user_query, cache_dir, tool_results, turn_cache,
        doc_context_fn=lambda: "",
    )
    preloaded_docs = preloaded_doc_index(tool_results)
    preloaded_files = list(preloaded_docs.keys()) or extract_doc_paths(
        user_query, project_root
    )

    if verbose and preloaded_files:
        print_fn(f"[codescope] Pre-loaded {len(preloaded_files)} project doc(s) from query.")
    if verbose and source_files_read:
        print_fn(f"[codescope] Pre-read {len(source_files_read)} source file(s): {', '.join(source_files_read)}")

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
    if verbose:
        print_fn(f"[codescope] Planner ({planner_label})…")

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

    if verbose:
        print_fn(format_checklist(plan))

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

    if query_needs_codebase_research(user_query):
        if verbose:
            print_fn("[codescope] Codebase research…")
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
        if verbose:
            print_fn("[codescope] Session search (prior turns)…")
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
        if verbose:
            print_fn("[codescope] Git context…")
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

    if verbose:
        print_fn(f"\n[codescope] Answer ({llm.model})…")

    if _is_write_task(user_query):
        answer_num_predict = ANSWER_NUM_PREDICT_WRITE
    elif _EXPLAIN_PAT.search(user_query) or _IMPLEMENT_PAT.search(user_query):
        # Implementation/debug prompts usually need long prose + code snippets.
        answer_num_predict = ANSWER_NUM_PREDICT_EXPLAIN
    else:
        answer_num_predict = ANSWER_NUM_PREDICT

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
    except ValidationError:
        from .formatting import coerce_answer_text
        try:
            answer = coerce_answer_text(data)
        except Exception:
            answer = raw
        if not answer.strip():
            answer = raw
        session_store.append_assistant(session_path, answer)
        return answer

    if resp.action == "tool_call":
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
    else:
        answer = _answer_text(resp.content)

    # Post-answer hook: if the task asked to create a file and the model produced
    # a code block in its answer (rather than calling write_file via tool_call),
    # extract and write it automatically.
    answer = _maybe_write_file(
        user_query, answer, tool_results, project_root, session_path, verbose, print_fn
    )

    session_store.append_assistant(session_path, answer, thought=resp.thought or "")
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
