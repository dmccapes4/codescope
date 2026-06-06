"""Simple chat pipeline.

One job: assemble the right context and let one model answer.

  per turn:
    1. resolve the files the question names (full bodies)
    2. semantic-search the codebase + graph for relevant neighbors
    3. graph summary (the repo map)
    4. one prompt -> one model -> answer

No planner, no tool loop, no two-stage write, no curation floors. If the answer
contains a fenced code block for a file the user named, we offer to write it.
"""
from __future__ import annotations

import difflib
import os as _os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

# Quiet the embedder's HF download bar / telemetry before sentence-transformers loads.
_os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
_os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

from ..config import (
    ANSWER_NUM_PREDICT_WRITE,
    OLLAMA_NUM_CTX,
    PROMPT_FIELD_MAX_TOKENS,
    SYSTEM_PROMPT_TOKENS,
    WARM_MODEL_AT_REPL,
    is_instructor_model,
)
from ..models.llm import LLM, OllamaError
from ..models.embedder import Embedder
from ..storage import sessions as session_store
from ..storage.graph import GraphIndex
from .context import build_graph_summary, count_tokens, truncate_to_tokens
from .proscope_docs import (
    format_feature_folder_list,
    load_feature_docs_context,
    persist_proscope_docs,
    resolve_feature_slug,
    save_active_feature,
    slugify_feature,
)
from .tools import tool_grep, tool_read_file, tool_semantic_search, ToolError

# --------------------------------------------------------------------------
# Prompt
# --------------------------------------------------------------------------

# Fallback system prompt when --llm points at a raw base model (not codescope:latest).
_SYSTEM_FALLBACK = (
    "You are codescope, a senior Android/Kotlin engineer. Ground answers in the "
    "provided context sections. For edits, output the complete file as "
    "`### <path>` followed by a fenced code block; codescope converts it to a "
    "git diff for the operator to apply — never claim you wrote anything."
)


def _system_for_model(model: str) -> str | None:
    """Return None for Modelfile models so their baked-in SYSTEM is used."""
    return None if is_instructor_model(model) else _SYSTEM_FALLBACK

# Files / classes named in the query
_CODE_EXTS = ("kt", "java", "xml", "kts", "gradle", "toml", "md")
_PATH_RE = re.compile(r"[\w./-]+\.(?:" + "|".join(_CODE_EXTS) + r")\b", re.IGNORECASE)
_CLASS_RE = re.compile(
    r"\b([A-Z][A-Za-z0-9]+"
    r"(?:Activity|Fragment|ViewModel|Repository|Dao|Database|Entity|Application|"
    r"Service|Screen|Module|UseCase|Manager|Helper|Provider))\b"
)

# Per-file body cap (~6 K tokens) so one huge file can't crowd everything else out.
_FILE_TOKEN_CAP = 6_000


# --------------------------------------------------------------------------
# Context assembly
# --------------------------------------------------------------------------

def _resolve_named_files(query: str, project_root: Path) -> list[str]:
    """Project-relative paths for files/classes the query names, resolved on disk."""
    wanted: list[str] = []
    for m in _PATH_RE.finditer(query):
        wanted.append(Path(m.group(0)).name)
    for m in _CLASS_RE.finditer(query):
        wanted.append(m.group(1) + ".kt")

    resolved: list[str] = []
    seen: set[str] = set()
    for name in wanted:
        hits = list(project_root.rglob(name))
        hits = [h for h in hits if "/build/" not in h.as_posix() and h.is_file()]
        if not hits:
            continue
        rel = hits[0].relative_to(project_root).as_posix()
        if rel not in seen:
            seen.add(rel)
            resolved.append(rel)
    return resolved


def _read_full(rel: str, project_root: Path) -> str | None:
    """Full file body (capped), or None if unreadable."""
    try:
        result = tool_read_file(path=rel, start=1, end=400, project_root=project_root)
    except ToolError:
        return None
    body = (result.get("content") or "").strip()
    if not body:
        return None
    return truncate_to_tokens(body, _FILE_TOKEN_CAP)


def _semantic_neighbors(
    query: str,
    cache_dir: Path,
    embedder: Embedder,
    exclude: set[str],
    k: int = 8,
) -> list[str]:
    """Top semantic hits (chunks + graph nodes), formatted, excluding already-included files."""
    try:
        hits = tool_semantic_search(
            query=query, k=k, min_score=0.30, include_graph=True,
            cache_dir=cache_dir, embedder=embedder,
        )
    except ToolError:
        return []
    lines: list[str] = []
    for h in hits:
        if not isinstance(h, dict) or h.get("info"):
            continue
        src = h.get("source", "?")
        score = h.get("score", 0)
        snippet = str(h.get("snippet") or "").strip().replace("\n", " ")[:160]
        if src == "graph":
            loc = h.get("node_id") or h.get("file", "?")
            lines.append(f"[graph] {loc} ({score:.2f}): {snippet}")
        else:
            f = h.get("file", "?")
            if f in exclude:
                continue
            ln = h.get("lines") or []
            tag = f"L{ln[0]}–{ln[1]}" if isinstance(ln, list) and len(ln) == 2 else ""
            lines.append(f"[code] {f} {tag} ({score:.2f}): {snippet}")
    return lines


def _format_history(path: Path, n: int = 3) -> str:
    turns = session_store.last_n_turns(path, n)
    out: list[str] = []
    for e in turns:
        role = e.get("role")
        content = (e.get("content") or "").strip()
        if not content:
            continue
        out.append(f"{'User' if role == 'user' else 'Assistant'}: {content[:600]}")
    return "\n".join(out)


def _git_context(project_root: Path) -> str:
    """Where the project is: branch, last commit, and dirty files. Empty if not a repo."""
    def run(*args: str) -> str:
        try:
            r = subprocess.run(
                ["git", "-C", str(project_root), *args],
                capture_output=True, text=True, timeout=5,
            )
            return r.stdout.strip() if r.returncode == 0 else ""
        except Exception:  # noqa: BLE001 — git missing / not a repo
            return ""

    branch = run("rev-parse", "--abbrev-ref", "HEAD")
    if not branch:
        return ""
    lines = [f"branch: {branch}"]
    last = run("log", "-1", "--pretty=%h %s")
    if last:
        lines.append(f"last commit: {last}")
    status = run("status", "--short")
    if status:
        lines.append("uncommitted changes:")
        lines.append(status)
    else:
        lines.append("working tree clean")
    return "\n".join(lines)


def _graph_neighborhood(named_rels: list[str], cache_dir: Path) -> str:
    """Nearness: what each named file depends on and what depends on it."""
    try:
        idx = GraphIndex(cache_dir)
    except Exception:  # noqa: BLE001
        return ""
    blocks: list[str] = []
    for rel in named_rels:
        results = idx.lookup_file(rel)
        if not results:
            continue
        outs, ins = set(), set()
        for r in results:
            for e in r.get("out", []):
                outs.add(f"{e.get('rel', 'refs')} → {e.get('dst', '?')}")
            for e in r.get("in", []):
                ins.add(f"{e.get('src', '?')} {e.get('rel', 'refs')} this")
        section = [f"{rel}:"]
        if outs:
            section.append("  depends on: " + "; ".join(sorted(outs)[:20]))
        if ins:
            section.append("  used by: " + "; ".join(sorted(ins)[:20]))
        if len(section) > 1:
            blocks.append("\n".join(section))
    return "\n".join(blocks)


def _grep_usages(named_rels: list[str], project_root: Path) -> str:
    """Where the named symbols are referenced across the project (call sites / shape)."""
    symbols = sorted({Path(r).stem for r in named_rels if Path(r).stem})
    out: list[str] = []
    seen: set[tuple] = set()
    for sym in symbols:
        try:
            hits = tool_grep(
                pattern=rf"\b{re.escape(sym)}\b",
                regex=True, max_results=12, project_root=project_root,
            )
        except ToolError:
            continue
        for h in hits:
            key = (h.get("file"), h.get("line"))
            if key in seen:
                continue
            seen.add(key)
            out.append(f"{h.get('file')}:{h.get('line')}: {str(h.get('text', '')).strip()[:120]}")
    return "\n".join(out[:40])


def build_prompt(
    query: str,
    project_root: Path,
    cache_dir: Path,
    embedder: Embedder,
    graph_summary: str,
    session_path: Path,
    print_fn: Callable[[str], None],
    *,
    proscope: bool = False,
    feature_slug: str | None = None,
    session_dir: Path | None = None,
) -> str:
    label = "proscope" if proscope else "codescope"
    active_feature = feature_slug
    if proscope and session_dir is not None:
        active_feature = resolve_feature_slug(query, project_root, session_dir, explicit=feature_slug)
        if active_feature:
            save_active_feature(session_dir, active_feature)
            print_fn(f"[proscope] Feature: docs/{active_feature}/")

    # File lookup is mandatory.
    named = _resolve_named_files(query, project_root)
    if named:
        print_fn(f"[{label}] Files: {', '.join(named)}")
    else:
        print_fn(f"[{label}] No files named — relying on search + graph.")

    file_blocks: list[str] = []
    included: set[str] = set()
    for rel in named:
        body = _read_full(rel, project_root)
        if body:
            file_blocks.append(f"### {rel}\n```\n{body}\n```")
            included.add(rel)

    print_fn(f"[{label}] git · graph · search · grep…")
    git_info     = _git_context(project_root)
    neighborhood = _graph_neighborhood(named, cache_dir)
    neighbors    = _semantic_neighbors(query, cache_dir, embedder, exclude=included)
    usages       = _grep_usages(named, project_root)
    history      = _format_history(session_path)

    feature_docs = ""
    feature_list = ""
    if proscope:
        feature_docs = load_feature_docs_context(project_root, active_feature)
        feature_list = format_feature_folder_list(project_root)

    # ── Budget ───────────────────────────────────────────────────────────────
    # Reserve the full write budget out of the context window FIRST, so that
    # however many files we pulled in, the model still has room to emit a full
    # file. The prompt can never grow into the response's headroom.
    prompt_budget = min(
        PROMPT_FIELD_MAX_TOKENS,
        OLLAMA_NUM_CTX - ANSWER_NUM_PREDICT_WRITE - SYSTEM_PROMPT_TOKENS - 512,
    )
    prompt_budget = max(prompt_budget, 2_000)

    parts: list[str] = []
    budget = prompt_budget

    def add(label: str, text: str, frac: float, hard: int) -> None:
        nonlocal budget
        text = (text or "").strip()
        if not text or budget <= 0:
            return
        cap = min(int(prompt_budget * frac), hard, budget)
        block = f"=== {label} ===\n" + truncate_to_tokens(text, cap)
        parts.append(block)
        budget -= count_tokens(block)

    # Priority order. Files come right after the tiny git header.
    add("PROJECT STATE (git)", git_info, 0.05, 800)
    if proscope:
        add("FEATURE DOCUMENTATION (source of truth)", feature_docs, 0.18, 6_000)
        add("PROSCOPE FEATURE FOLDERS", feature_list, 0.04, 800)
    if file_blocks:
        add("FILES IN QUESTION (full bodies)", "\n\n".join(file_blocks), 0.55, 24_000)
    add("GRAPH NEIGHBORHOOD (nearness)", neighborhood, 0.12, 4_000)
    add("RELATED CODE (semantic search)", "\n".join(neighbors), 0.12, 4_000)
    add("USAGES (grep — call sites / components)", usages, 0.10, 3_000)
    add("REPO MAP", graph_summary, 0.06, 1_200)
    add("RECENT CONVERSATION", history, 0.05, 1_200)

    context = "\n\n".join(parts)
    suffix = query
    if proscope and active_feature:
        suffix = (
            f"[Active feature folder: docs/{active_feature}/]\n"
            f"Emit PROSCOPE_DOC blocks for docs/{active_feature}/ when planning or updating strategy.\n\n"
            f"{query}"
        )
    return f"{context}\n\n=== QUESTION ===\n{suffix}"


# --------------------------------------------------------------------------
# Turn
# --------------------------------------------------------------------------

def answer_turn(
    query: str,
    project_root: Path,
    cache_dir: Path,
    session_dir: Path,
    session_path: Path,
    llm: LLM,
    embedder: Embedder,
    graph_summary: str,
    print_fn: Callable[[str], None] = print,
    *,
    proscope: bool = False,
    feature_slug: str | None = None,
) -> str:
    session_store.append_user(session_path, query)

    prompt = build_prompt(
        query, project_root, cache_dir, embedder, graph_summary, session_path, print_fn,
        proscope=proscope,
        feature_slug=feature_slug,
        session_dir=session_dir,
    )

    mode_label = "proscope" if proscope else "codescope"
    print_fn(f"[{mode_label}] Generating with {llm.model} (num_predict={ANSWER_NUM_PREDICT_WRITE})…")

    try:
        answer = llm.generate(
            prompt=prompt,
            system=_system_for_model(llm.model),
            json_mode=False,
            num_predict=ANSWER_NUM_PREDICT_WRITE,
            temperature=0.2,
        )
    except OllamaError as e:
        msg = f"LLM error: {e}"
        session_store.append_assistant(session_path, msg)
        return msg

    answer = (answer or "").strip()

    if proscope:
        active = resolve_feature_slug(query, project_root, session_dir, explicit=feature_slug)
        answer, _ = persist_proscope_docs(
            answer, project_root, session_dir, active, print_fn=print_fn,
        )

    answer = _propose_diffs(query, answer, project_root, cache_dir, print_fn)
    session_store.append_assistant(session_path, answer)
    return answer


# --------------------------------------------------------------------------
# Propose changes as a git diff — never apply anything
# --------------------------------------------------------------------------

_FILE_SECTION_RE = re.compile(
    r"^###\s+(?P<path>[A-Za-z0-9_\-./]+\.(?:kt|java|xml|kts|gradle|toml))\s*\n+"
    r"```[a-zA-Z]*\n(?P<body>.*?)\n```",
    re.DOTALL | re.MULTILINE,
)


def _resolve_rel(rel: str, project_root: Path) -> str:
    """Map a model-given path to a real project-relative path (resolving bare names)."""
    rel = rel.replace("\\", "/").lstrip("./")
    if (project_root / rel).exists():
        return rel
    hits = [h for h in project_root.rglob(Path(rel).name)
            if "/build/" not in h.as_posix() and h.is_file()]
    return hits[0].relative_to(project_root).as_posix() if hits else rel


def _make_patch(rel: str, old_text: str, new_text: str, existed: bool) -> str | None:
    """A git-apply-able unified diff for one file, or None if nothing changed."""
    old_norm = old_text if (old_text == "" or old_text.endswith("\n")) else old_text + "\n"
    new_norm = new_text if new_text.endswith("\n") else new_text + "\n"
    if old_norm == new_norm:
        return None

    is_new = not existed
    fromfile = "/dev/null" if is_new else f"a/{rel}"
    body = "".join(difflib.unified_diff(
        old_norm.splitlines(keepends=True),
        new_norm.splitlines(keepends=True),
        fromfile=fromfile,
        tofile=f"b/{rel}",
    ))
    header = f"diff --git a/{rel} b/{rel}\n"
    if is_new:
        header += "new file mode 100644\n"
    return header + body


def _propose_diffs(
    query: str,
    answer: str,
    project_root: Path,
    cache_dir: Path,
    print_fn: Callable[[str], None],
) -> str:
    """Turn each `### <path>` + full-file block in the answer into a reviewable git diff.

    Replaces the full-file block with a ```diff block, writes a combined patch to the
    cache, and appends the `git apply` command. Nothing is written to the project.
    The model decides whether to propose a file — if the answer has no `### <path>`
    block, this is a no-op.
    """
    patches: list[str] = []

    def _replace(m: re.Match) -> str:
        rel = _resolve_rel(m.group("path"), project_root)
        body = m.group("body").rstrip("\n")
        if len(body.strip()) < 30:
            return m.group(0)  # too small to be a real file — leave untouched

        abs_path = project_root / rel
        existed = abs_path.exists()
        try:
            old = abs_path.read_text(encoding="utf-8", errors="replace") if existed else ""
        except OSError:
            old = ""

        patch = _make_patch(rel, old, body, existed)
        if patch is None:
            return f"### {rel}\n_(no change — file on disk already matches)_"
        patches.append(patch)
        tag = "new file" if not existed else "modified"
        return f"### {rel}  _({tag})_\n```diff\n{patch.rstrip()}\n```"

    new_answer = _FILE_SECTION_RE.sub(_replace, answer)

    if not patches:
        return new_answer

    patch_dir = cache_dir / "patches"
    patch_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    patch_path = patch_dir / f"{ts}.patch"
    patch_path.write_text("\n".join(patches) + "\n", encoding="utf-8")

    apply_cmd = (
        f'cd "{project_root}"\n'
        f'git apply --check "{patch_path}"   # dry run — verify it applies cleanly\n'
        f'git apply "{patch_path}"           # apply for real'
    )
    print_fn(f"[codescope] Proposed {len(patches)} change(s) → patch saved to {patch_path}")
    new_answer += (
        "\n\n---\n"
        "**Review the diff above. Nothing has been changed.** To apply it yourself:\n\n"
        f"```bash\n{apply_cmd}\n```\n"
        f"_Patch file: `{patch_path}`_"
    )
    return new_answer


# --------------------------------------------------------------------------
# REPL + one-shot (signatures consumed by cli.py)
# --------------------------------------------------------------------------

def _open_session(session_dir: Path, new_session: bool) -> Path:
    if new_session:
        return session_store.new_path(session_dir)
    return session_store.latest(session_dir) or session_store.new_path(session_dir)


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
    *,
    proscope: bool = False,
    feature_slug: str | None = None,
) -> None:
    from rich.console import Console
    from rich.markdown import Markdown

    console = Console()
    session_dir.mkdir(parents=True, exist_ok=True)
    session_path = session_path or _open_session(session_dir, new_session)
    graph_summary = build_graph_summary(cache_dir)

    if proscope and feature_slug:
        save_active_feature(session_dir, slugify_feature(feature_slug))

    if not embedder.is_loaded():
        console.print(f"[dim]Loading embedding weights ({embedder.model_name} on {embedder.device})…[/dim]")
        try:
            embedder.warm()
        except Exception as e:  # noqa: BLE001
            console.print(f"[red]Embedder failed to load: {e}[/red]")

    if WARM_MODEL_AT_REPL:
        console.print(f"[dim]Warming {llm.model}…[/dim]")
        llm.warm()

    from ..config import OLLAMA_NUM_CTX, OLLAMA_BASE_URL
    from .proscope_docs import load_active_feature

    brand = "ProScope" if proscope else "codescope"
    console.print(f"\n[bold magenta]{brand}[/bold magenta] — project: [cyan]{slug}[/cyan]")
    console.print(
        f"Session: [dim]{session_path.name}[/dim]  "
        f"Model: [dim]{llm.model}[/dim]  ctx: [dim]{OLLAMA_NUM_CTX:,}[/dim]  "
        f"[dim]{OLLAMA_BASE_URL}[/dim]"
    )
    if proscope:
        active = load_active_feature(session_dir)
        console.print(
            f"Feature docs: [cyan]docs/{active or '(none — use /feature <slug> or feature slug: …)'}[/cyan]\n"
            f"[dim]ProScope writes docs/<feature>/IMPLEMENTATION_PLAN.md and STRATEGY_PHASE_*.md[/dim]"
        )
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

        # ProScope REPL commands
        if proscope and query.lower().startswith("/feature "):
            slug = slugify_feature(query.split(maxsplit=1)[1])
            save_active_feature(session_dir, slug)
            console.print(f"[green]Active feature → docs/{slug}/[/green]\n")
            continue

        answer = answer_turn(
            query, project_root, cache_dir, session_dir, session_path,
            llm, embedder, graph_summary, print_fn=console.print,
            proscope=proscope,
            feature_slug=None,
        )
        console.print(f"\n[bold]{brand}:[/bold]")
        try:
            console.print(Markdown(answer))
        except Exception:  # noqa: BLE001
            console.print(answer)
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
    *,
    proscope: bool = False,
    feature_slug: str | None = None,
) -> str:
    session_dir.mkdir(parents=True, exist_ok=True)
    session_path = session_store.new_path(session_dir)
    graph_summary = build_graph_summary(cache_dir)
    return answer_turn(
        query, project_root, cache_dir, session_dir, session_path,
        llm, embedder, graph_summary, print_fn=print,
        proscope=proscope,
        feature_slug=feature_slug,
    )
