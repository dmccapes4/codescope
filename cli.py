"""
codescope CLI — entry point.

Command groups
──────────────
  projects          List discovered Android projects
  status            Show cache manifest
  index             Full index pipeline (skeleton → graph → enrich → embed)
  reset             Wipe cache + sessions
  sessions          List saved chat sessions

  chat              Interactive REPL
  ask               One-shot question

  grep              Run the grep tool directly (no LLM)
  search            Run semantic search directly (no LLM)
  android-docs      Fetch official Android docs (developer.android.com)
  graph             Print the dependency graph summary
  diff              Show files changed since last index

  menu              Interactive numbered menu (FMP-style)
  export            Bundle cache + sessions to zip
  help              Show documentation (topic: index|chat|tools|setup|concepts|tips|commands)

Design conventions (FullMetalPacket-inspired)
──────────────────────────────────────────────
  • Every long action logs [HH:MM:SS] ▶️/✅/❌ timestamped lines.
  • Governance gates ([C/A/Q/S]) guard expensive phases when --hitl is set.
  • Failed phases offer [R]etry / [A]bort.
  • --suffix <name> namespaces run logs under logs/<suffix>/<run_ts>/.
  • run_meta.json is written alongside each index run.
  • Streaming output is dual-written to the terminal AND the log file.
"""
from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

app     = typer.Typer(
    help="Agentic code-ingestion and search for Android Studio projects.",
    no_args_is_help=True,
)
console = Console()

_VERSION = "0.1.0"

def _version_callback(value: bool):
    if value:
        console.print(f"codescope [bold cyan]{_VERSION}[/bold cyan]")
        raise typer.Exit()

@app.callback()
def _main(
    version: bool = typer.Option(
        False, "--version", "-V",
        callback=_version_callback,
        is_eager=True,
        help="Show version and exit.",
    ),
):
    pass

# ── shared option factories ───────────────────────────────────────────────────

def _proj_opt():
    return typer.Option(..., "--project", "-p", help="Project slug (folder name)")

def _llm_opt():
    from .config import DEFAULT_LLM
    return typer.Option(DEFAULT_LLM, "--llm", help="Ollama model name")

def _embedder_opt():
    from .config import DEFAULT_EMBEDDER
    return typer.Option(DEFAULT_EMBEDDER, "--embedder", help="sentence-transformers model")

def _verbose_opt():
    return typer.Option(False, "--verbose", "-v", help="Show debug output")

def _suffix_opt():
    return typer.Option("default", "--suffix", "-s", help="Run label — namespaces logs/")

def _hitl_opt():
    return typer.Option(False, "--hitl", help="Enable HITL governance gates before each phase")

def _force_opt():
    return typer.Option(False, "--force", help="Ignore content-hash cache; re-index everything")


# ── helpers ───────────────────────────────────────────────────────────────────

def _print_answer(answer: str) -> None:
    """Render the agent's final answer with Markdown + JSON syntax highlighting."""
    import re
    from rich.markdown import Markdown
    from rich.syntax import Syntax
    from rich.panel import Panel

    # If the whole answer is a JSON object/array, render it as syntax-highlighted JSON
    stripped = answer.strip()
    if stripped.startswith(("{", "[")):
        try:
            import json as _json
            obj = _json.loads(stripped)
            pretty = _json.dumps(obj, indent=2, ensure_ascii=False)
            console.print(Panel(
                Syntax(pretty, "json", theme="monokai", word_wrap=True),
                border_style="dim",
            ))
            return
        except Exception:
            pass

    from .chat.formatting import format_terminal_answer

    answer = format_terminal_answer(answer)

    # Plain text first (URLs visible); optional light markdown for headings
    try:
        console.print(Markdown(answer))
    except Exception:
        console.print(answer)


def _check_index(project: str, cache_dir: Path) -> None:
    if not (cache_dir / "manifest.json").exists():
        console.print(
            f"[yellow]Project {project!r} is not indexed yet.[/yellow]\n"
            f"Run: [bold]codescope index --project {project}[/bold]"
        )
        raise typer.Exit(1)


def _check_ollama(llm) -> None:
    if not llm.is_available():
        console.print(
            f"[red]Ollama not reachable at {llm.base_url}[/red]\n"
            "Start with: [bold]ollama serve &[/bold]"
        )
        raise typer.Exit(1)


def _resolve(project: str) -> Path:
    from . import projects as P
    try:
        return P.resolve(project)
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)


# ═════════════════════════════════════════════════════════════════════════════
# codescope projects
# ═════════════════════════════════════════════════════════════════════════════

@app.command("projects")
def cmd_projects():
    """List all discovered Android projects and their cache status."""
    from . import projects as P
    from .config import CACHE_ROOT

    rows = P.discover()
    if not rows:
        console.print("[yellow]No Android projects found alongside the workspace root.[/yellow]")
        raise typer.Exit(1)

    t = Table(title="Discovered Android projects", show_lines=True)
    t.add_column("Project",    style="cyan",  no_wrap=True)
    t.add_column("Path",       style="dim")
    t.add_column("Cache",      justify="right")
    t.add_column("Indexed?",   justify="center")
    t.add_column("Files",      justify="right")
    t.add_column("Chunks",     justify="right")

    for slug, path in rows:
        size = P.cache_size_bytes(slug)
        size_s = f"{size/1024:.0f} KB" if size < 1_048_576 else f"{size/1_048_576:.1f} MB"
        mp = CACHE_ROOT / slug / "manifest.json"
        if mp.exists():
            mf = json.loads(mp.read_text())
            indexed = "[green]✓[/green]"
            files  = str(mf.get("file_count", "–"))
            chunks = str(mf.get("chunk_count", "–"))
        else:
            indexed, files, chunks = "[red]✗[/red]", "–", "–"
        t.add_row(slug, str(path), size_s, indexed, files, chunks)

    console.print(t)


# ═════════════════════════════════════════════════════════════════════════════
# codescope status
# ═════════════════════════════════════════════════════════════════════════════

@app.command("status")
def cmd_status(project: str = _proj_opt()):
    """Show the cache manifest for a project."""
    from .config import CACHE_ROOT
    from .storage.manifest import load as manifest_load
    _resolve(project)
    data = manifest_load(CACHE_ROOT / project)
    if not data:
        console.print(f"[yellow]No cache for {project!r}. Run `codescope index --project {project}`[/yellow]")
        raise typer.Exit(1)
    t = Table(title=f"Cache manifest — {project}", show_lines=False)
    t.add_column("Key",   style="cyan")
    t.add_column("Value", style="white")
    for k, v in data.items():
        t.add_row(str(k), str(v))
    console.print(t)


# ═════════════════════════════════════════════════════════════════════════════
# codescope index
# ═════════════════════════════════════════════════════════════════════════════

@app.command("index")
def cmd_index(
    project:       str        = _proj_opt(),
    skeleton_only: bool       = typer.Option(False, "--skeleton-only"),
    no_enrich:     bool       = typer.Option(False, "--no-enrich"),
    no_embed:      bool       = typer.Option(False, "--no-embed"),
    force:         bool       = _force_opt(),
    hitl:          bool       = _hitl_opt(),
    suffix:        str        = _suffix_opt(),
    llm_model:     str        = _llm_opt(),
    embedder_name: str        = _embedder_opt(),
    verbose:       bool       = _verbose_opt(),
    enrich_path:   list[str]  = typer.Option(
        None, "--enrich-path", "-P",
        help=(
            "Restrict LLM enrichment (and embedding) to files under this path. "
            "Can be a folder prefix or an exact file path. Repeatable: "
            "-P app/src/main/java/com/example -P app/src/main/res/layout/activity_main.xml"
        ),
    ),
):
    """Index a project: skeleton → static graph → LLM enrich → embed.

    By default all files are enriched and embedded.  Use --enrich-path / -P to
    restrict phases 3 and 4 to a specific folder or file:

      codescope index -p MyApp -P app/src/main/java/com/example/feature/

    Multiple paths can be specified:

      codescope index -p MyApp -P app/src/main/java -P docs/
    """
    from . import projects as P
    from .config import CACHE_ROOT, WORKSPACE_ROOT
    from .storage.manifest import load_hashes, build as mf_build, save as mf_save
    from .ingest import skeleton, static_parse, llm_enrich, embed as embed_mod
    from .models.llm import LLM
    from .models.embedder import Embedder
    from .log_utils import (
        log_start, log_ok, log_fail, log_warn, log_info, log_event,
        phase_banner, section_line, print_banner,
        governance_gate, run_log_dir, write_run_meta,
        _elapsed, _ts,
    )

    project_root = _resolve(project)
    cache_dir    = P.cache_dir(project)
    run_ts       = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_dir      = run_log_dir(WORKSPACE_ROOT, suffix, run_ts)
    phases_run:  list[str] = []
    run_start    = datetime.now(timezone.utc)

    print_banner(project)
    log_info("index", f"suffix={suffix}  run_ts={run_ts}  logs → {log_dir.relative_to(WORKSPACE_ROOT)}")
    log_info("index", f"llm={llm_model}  embedder={embedder_name}  force={force}  hitl={hitl}")
    if enrich_path:
        for ep in enrich_path:
            log_info("index", f"  enrich-path filter: {ep}")

    # ── Phase 1: Skeleton ──────────────────────────────────────────────────

    phase_banner("Phase 1 / 4  —  Skeleton Scan")

    if hitl:
        choice = governance_gate(
            "Skeleton Scan",
            f"Walk {project_root} and classify every source file.\n"
            "Fast — no network, no model. Updates files.jsonl.",
        )
        if choice == "abort":
            raise typer.Exit(1)

    t1 = log_start("skeleton", f"walking {project_root.name}")
    existing_hashes = load_hashes(cache_dir)

    _STATUS_ICON = {"new": "🆕", "changed": "✏️ ", "unchanged": "·  ", "oversize": "⚠️ "}

    def _skel_progress(rel, category, status):
        icon = _STATUS_ICON.get(status, "   ")
        log_event("skeleton", f"{icon} [{status:<9}] {rel}  ({category})", emoji="")

    added, unchanged = skeleton.run(
        project_root, cache_dir, existing_hashes, force=force,
        progress_cb=_skel_progress,
    )
    log_ok("skeleton", f"+{added} new/changed  {unchanged} unchanged", start=t1)
    phases_run.append("skeleton")

    files_jsonl = cache_dir / "files.jsonl"
    if not files_jsonl.exists():
        log_fail("skeleton", "files.jsonl was not created — aborting")
        raise typer.Exit(1)

    if skeleton_only:
        log_info("index", "--skeleton-only: stopping here")
        _finalize(project, project_root, cache_dir, log_dir, run_start, run_ts,
                  phases_run, llm_model, embedder_name, suffix, success=True)
        return

    # ── Phase 2: Static parse ──────────────────────────────────────────────

    phase_banner("Phase 2 / 4  —  Static Graph")

    if hitl:
        choice = governance_gate(
            "Static Parse",
            "Regex-extract imports, class declarations, Gradle deps, Manifest entries.\n"
            "Writes graph.nodes.jsonl + graph.edges.jsonl. No LLM.",
        )
        if choice == "abort":
            raise typer.Exit(1)
        if choice == "skip":
            log_info("static", "skipped by user")
        else:
            _run_static(project_root, cache_dir, files_jsonl, force, verbose, phases_run, log_dir)
    else:
        _run_static(project_root, cache_dir, files_jsonl, force, verbose, phases_run, log_dir)

    # ── Phase 3: LLM enrichment ────────────────────────────────────────────

    if not no_enrich:
        phase_banner("Phase 3 / 4  —  LLM Enrichment")
        llm = LLM(model=llm_model)

        if not llm.is_available():
            log_warn("enrich", "Ollama not reachable — skipping. Start: ollama serve &")
        else:
            if hitl:
                choice = governance_gate(
                    "LLM Enrichment",
                    f"Send each source file to {llm_model} one-at-a-time.\n"
                    "Adds summaries, call-edges, and tags to the graph.\n"
                    "Responses are cached by content-hash — re-runs are instant.",
                    allow_skip=True,
                )
                if choice == "abort":
                    raise typer.Exit(1)
                if choice == "skip":
                    log_info("enrich", "skipped by user")
                    no_enrich = True

            if not no_enrich:
                _run_enrich_with_retry(
                    project_root, cache_dir, files_jsonl, llm,
                    force, verbose, hitl, phases_run, log_dir,
                    enrich_paths=enrich_path or [],
                )
    else:
        log_info("enrich", "skipped (--no-enrich)")

    # ── Phase 4: Embed ─────────────────────────────────────────────────────

    if not no_embed:
        phase_banner("Phase 4 / 4  —  Embedding")
        embedder = Embedder(model_name=embedder_name)

        if hitl:
            choice = governance_gate(
                "Embedding",
                f"Chunk every source file and encode with {embedder_name}.\n"
                "Writes embeddings.npy + embeddings.meta.jsonl.\n"
                "GPU-accelerated when CUDA is available.",
                allow_skip=True,
            )
            if choice == "abort":
                raise typer.Exit(1)
            if choice == "skip":
                log_info("embed", "skipped by user")
                no_embed = True

        if not no_embed:
            _run_embed_with_retry(
                project_root, cache_dir, files_jsonl, embedder,
                force, verbose, hitl, phases_run, log_dir,
                enrich_paths=enrich_path or [],
            )
    else:
        log_info("embed", "skipped (--no-embed)")

    # ── Manifest + run_meta ────────────────────────────────────────────────

    embedder_dim = 0
    if not no_embed:
        from .models.embedder import Embedder as E2
        try:
            embedder_dim = E2(model_name=embedder_name).dim
        except Exception:
            pass

    mf = mf_build(project, project_root, cache_dir, embedder_name, embedder_dim, llm_model)
    mf_save(cache_dir, mf)

    _finalize(project, project_root, cache_dir, log_dir, run_start, run_ts,
              phases_run, llm_model, embedder_name, suffix, success=True,
              file_count=mf["file_count"], node_count=mf["node_count"],
              edge_count=mf["edge_count"], chunk_count=mf["chunk_count"])


def _run_static(project_root, cache_dir, files_jsonl, force, verbose, phases_run, log_dir):
    from .ingest import static_parse
    from .log_utils import log_start, log_ok, log_event

    _STATIC_ICON = {"parsed": "✅", "cached": "·  ", "oversize": "⚠️ ", "node-only": "📄"}

    def _static_progress(rel, category, status):
        icon = _STATIC_ICON.get(status, "   ")
        log_event("static", f"{icon} [{status:<9}] {rel}  ({category})", emoji="")

    t = log_start("static", "parsing Kotlin, Gradle, Manifest, TOML")
    proc, skip = static_parse.run(
        project_root, cache_dir, files_jsonl, force=force,
        progress_cb=_static_progress,
    )
    log_ok("static", f"{proc} parsed  {skip} skipped", start=t)
    phases_run.append("static")


def _run_enrich_with_retry(
    project_root, cache_dir, files_jsonl, llm, force, verbose, hitl, phases_run, log_dir,
    enrich_paths: list = [],
):
    from .ingest import llm_enrich
    from .log_utils import log_start, log_ok, log_fail, log_info, log_event, governance_gate

    while True:
        label = f"model={llm.model}"
        if enrich_paths:
            label += f"  paths={enrich_paths}"
        t = log_start("enrich", label)
        counts = {"enriched": 0, "skipped": 0, "total": 0}

        def progress(done, total, rel, skipped=False):
            counts["total"] = total
            if not skipped:
                counts["enriched"] += 1
            else:
                counts["skipped"] += 1
            icon = "·  " if skipped else "🧠"
            tag  = "skip" if skipped else "enriched"
            log_event("enrich", f"{icon} [{done:>3}/{total}] [{tag:<8}] {rel}", emoji="")

        enriched, skipped = llm_enrich.run(
            project_root, cache_dir, files_jsonl, llm,
            force=force, progress_cb=progress, include_paths=enrich_paths,
        )
        log_ok("enrich", f"{enriched} enriched  {skipped} skipped", start=t)
        phases_run.append("enrich")

        if hitl:
            choice = governance_gate(
                "Post-Enrichment Review",
                f"LLM enrichment complete.\n"
                f"  enriched={enriched}  skipped={skipped}\n"
                "Choose [C]ontinue to proceed to embedding, [R]etry enrichment, or [A]bort.",
                allow_retry=True,
            )
            if choice == "abort":
                raise typer.Exit(1)
            if choice == "retry":
                phases_run.remove("enrich")
                continue
        break


def _run_embed_with_retry(
    project_root, cache_dir, files_jsonl, embedder, force, verbose, hitl, phases_run, log_dir,
    enrich_paths: list = [],
):
    from .ingest import embed as embed_mod
    from .log_utils import log_start, log_ok, log_fail, log_event, governance_gate

    while True:
        t = log_start("embed", f"model={embedder.model_name}")

        def progress(done, total, rel, n_chunks=0, skipped=False):
            icon = "·  " if skipped else "✨"
            tag  = "skip" if skipped else f"{n_chunks} chunks"
            log_event("embed", f"{icon} [{done:>3}/{total}] [{tag:<10}] {rel}", emoji="")

        chunks, skipped = embed_mod.run(
            project_root, cache_dir, files_jsonl, embedder,
            force=force, progress_cb=progress, include_paths=enrich_paths,
        )
        log_ok("embed", f"{chunks} chunks written  {skipped} files skipped", start=t)
        phases_run.append("embed")

        if hitl:
            choice = governance_gate(
                "Post-Embedding Review",
                f"Embedding complete.\n  chunks={chunks}  files_skipped={skipped}\n"
                "Choose [C]ontinue to finalise, [R]etry embedding, or [A]bort.",
                allow_retry=True,
            )
            if choice == "abort":
                raise typer.Exit(1)
            if choice == "retry":
                phases_run.remove("embed")
                continue
        break


def _finalize(slug, project_root, cache_dir, log_dir, run_start, run_ts,
              phases_run, llm, embedder, suffix, success,
              file_count=0, node_count=0, edge_count=0, chunk_count=0):
    from .log_utils import log_ok, write_run_meta, _elapsed, phase_banner
    from .config import WORKSPACE_ROOT

    elapsed = (datetime.now(timezone.utc) - run_start).total_seconds()
    run_meta_path = log_dir / "run_meta.json"
    write_run_meta(
        path=run_meta_path,
        slug=slug,
        suffix=suffix,
        phases_run=phases_run,
        llm=llm,
        embedder=embedder,
        elapsed_s=elapsed,
        success=success,
        extra={
            "file_count":  file_count,
            "node_count":  node_count,
            "edge_count":  edge_count,
            "chunk_count": chunk_count,
        },
    )

    phase_banner("✅  Index Complete")
    print(f"  project  : {slug}")
    print(f"  files    : {file_count}   nodes : {node_count}   edges : {edge_count}   chunks : {chunk_count}")
    print(f"  phases   : {', '.join(phases_run)}")
    print(f"  elapsed  : {elapsed:.1f}s")
    print(f"  run_meta : {run_meta_path.relative_to(WORKSPACE_ROOT)}")
    print()


# ═════════════════════════════════════════════════════════════════════════════
# codescope reset
# ═════════════════════════════════════════════════════════════════════════════

@app.command("reset")
def cmd_reset(
    project: str  = _proj_opt(),
    yes:     bool = typer.Option(False, "--yes", "-y"),
):
    """Delete cache and sessions for a project."""
    from .config import CACHE_ROOT, SESSIONS_ROOT
    _resolve(project)
    if not yes:
        if not typer.confirm(f"Delete all cache + sessions for {project!r}?"):
            raise typer.Abort()
    for d in (CACHE_ROOT / project, SESSIONS_ROOT / project):
        if d.exists():
            shutil.rmtree(d)
            console.print(f"  [dim]deleted {d}[/dim]")
    console.print(f"[green]✓ Reset complete for {project!r}[/green]")


# ═════════════════════════════════════════════════════════════════════════════
# codescope sessions
# ═════════════════════════════════════════════════════════════════════════════

@app.command("sessions")
def cmd_sessions(project: str = _proj_opt()):
    """List saved chat sessions for a project."""
    from . import projects as P
    from .storage.sessions import list_sessions
    sess = list_sessions(P.session_dir(project))
    if not sess:
        console.print(f"[yellow]No sessions for {project!r}[/yellow]")
        return
    t = Table(title=f"Sessions — {project}", show_lines=True)
    t.add_column("File",     style="cyan")
    t.add_column("Modified", style="dim")
    t.add_column("Entries",  justify="right")
    t.add_column("Size",     justify="right")
    for r in sess:
        t.add_row(r["file"], r["modified"][:19], str(r["entries"]), f"{r['size']/1024:.1f} KB")
    console.print(t)


# ═════════════════════════════════════════════════════════════════════════════
# codescope diff
# ═════════════════════════════════════════════════════════════════════════════

@app.command("diff")
def cmd_diff(project: str = _proj_opt()):
    """Show files changed on disk since the last index run."""
    from . import projects as P
    from .config import CACHE_ROOT
    from .storage.manifest import load_hashes
    import hashlib

    project_root = _resolve(project)
    cache_dir    = P.cache_dir(project)
    old_hashes   = load_hashes(cache_dir)

    if not old_hashes:
        console.print(f"[yellow]No index found for {project!r}. Nothing to diff.[/yellow]")
        raise typer.Exit(1)

    added, modified, deleted = [], [], []

    # Walk current state
    current: dict[str, str] = {}
    from .config import EXCLUDED_DIRS, EXCLUDED_SUFFIXES, EXCLUDED_FILENAMES
    import os
    for dirpath, dirnames, filenames in os.walk(project_root):
        dirnames[:] = [d for d in dirnames if d not in EXCLUDED_DIRS and not d.startswith(".")]
        for name in filenames:
            if name in EXCLUDED_FILENAMES:
                continue
            if Path(name).suffix.lower() in EXCLUDED_SUFFIXES:
                continue
            fp  = Path(dirpath) / name
            rel = fp.relative_to(project_root).as_posix()
            try:
                h = hashlib.sha256(fp.read_bytes()).hexdigest()
                current[rel] = h
            except OSError:
                pass

    for rel, h in current.items():
        if rel not in old_hashes:
            added.append(rel)
        elif old_hashes[rel] != h:
            modified.append(rel)

    for rel in old_hashes:
        if rel not in current:
            deleted.append(rel)

    total = len(added) + len(modified) + len(deleted)
    if total == 0:
        console.print(f"[green]✓ No changes since last index ({len(old_hashes)} files)[/green]")
        return

    if added:
        console.print(f"\n[green]+{len(added)} added[/green]")
        for r in sorted(added):
            console.print(f"  [green]+[/green] {r}")
    if modified:
        console.print(f"\n[yellow]~{len(modified)} modified[/yellow]")
        for r in sorted(modified):
            console.print(f"  [yellow]~[/yellow] {r}")
    if deleted:
        console.print(f"\n[red]-{len(deleted)} deleted[/red]")
        for r in sorted(deleted):
            console.print(f"  [red]-[/red] {r}")

    console.print(f"\n[dim]{total} change(s) — run `codescope index --project {project}` to update[/dim]")


# ═════════════════════════════════════════════════════════════════════════════
# codescope grep
# ═════════════════════════════════════════════════════════════════════════════

@app.command("grep")
def cmd_grep(
    pattern:    str  = typer.Argument(..., help="Regex or literal pattern"),
    project:    str  = _proj_opt(),
    path:       str  = typer.Option(".", "--path",  help="Sub-path inside the project"),
    ignore_case:bool = typer.Option(False, "--ignore-case", "-i"),
    fixed:      bool = typer.Option(False, "--fixed",       "-F", help="Treat pattern as literal"),
    max_results:int  = typer.Option(60,    "--max",  "-n"),
    sessions:   bool = typer.Option(False, "--sessions", help="Also search session logs"),
):
    """Run the grep tool directly — no LLM, no chat."""
    from . import projects as P
    from .chat.tools import tool_grep

    project_root = _resolve(project)
    session_dir  = P.session_dir(project)

    results = tool_grep(
        pattern=pattern,
        path=path,
        case_insensitive=ignore_case,
        regex=not fixed,
        max_results=max_results,
        include_sessions=sessions,
        project_root=project_root,
        session_dir=session_dir,
    )

    if not results:
        console.print("[yellow]No matches.[/yellow]")
        return

    console.print(f"[dim]{len(results)} match(es)[/dim]\n")
    prev_file = None
    for r in results:
        if r["file"] != prev_file:
            console.print(f"[cyan]{r['file']}[/cyan]")
            prev_file = r["file"]
        console.print(f"  [dim]{r['line']:>5}[/dim]  {r['text']}")


# ═════════════════════════════════════════════════════════════════════════════
# codescope search
# ═════════════════════════════════════════════════════════════════════════════

@app.command("search")
def cmd_search(
    query:        str  = typer.Argument(..., help="Semantic search query"),
    project:      str  = _proj_opt(),
    k:            int  = typer.Option(8,   "--k",   help="Number of results"),
    ext:          str  = typer.Option("",  "--ext", help="Filter by extension, e.g. .kt"),
    embedder_name:str  = _embedder_opt(),
):
    """Run semantic search directly — no LLM, no chat."""
    from . import projects as P
    from .chat.tools import tool_semantic_search
    from .models.embedder import Embedder

    project_root = _resolve(project)
    cache_dir    = P.cache_dir(project)
    _check_index(project, cache_dir)

    embedder = Embedder(model_name=embedder_name)
    filt     = {"ext": [ext]} if ext else None

    results = tool_semantic_search(
        query=query,
        k=k,
        filter=filt,
        cache_dir=cache_dir,
        embedder=embedder,
    )

    if not results:
        console.print("[yellow]No results.[/yellow]")
        return

    console.print(f"\n[dim]Top {len(results)} matches for:[/dim] [bold]{query}[/bold]\n")
    for i, r in enumerate(results, 1):
        lines = r.get("lines") or []
        line_s = f"L{lines[0]}–{lines[1]}" if len(lines) == 2 else ""
        console.print(
            f"  [bold cyan]{i:>2}.[/bold cyan] [cyan]{r['file']}[/cyan]  "
            f"[dim]{line_s}[/dim]  [yellow]score={r['score']:.3f}[/yellow]"
        )
        if r.get("snippet"):
            snippet = r["snippet"][:200].replace("\n", " ")
            console.print(f"      [dim]{snippet}[/dim]")
        console.print()


# ═════════════════════════════════════════════════════════════════════════════
# codescope android-docs
# ═════════════════════════════════════════════════════════════════════════════

@app.command("android-docs")
def cmd_android_docs(
    topics: list[str] = typer.Argument(..., help="Topics e.g. Room StateFlow Compose"),
    project: str = typer.Option("", "--project", "-p", help="Project (for doc cache dir)"),
):
    """Fetch official Android documentation from developer.android.com."""
    from rich.markdown import Markdown
    from .chat.android_docs import fetch_android_docs, format_doc_terminal
    from . import projects as P

    cache_dir = P.cache_dir(project) if project else None

    console.print(f"[dim]Fetching {len(topics)} topic(s) from developer.android.com…[/dim]\n")
    for doc in fetch_android_docs(topics, cache_dir=cache_dir):
        console.print(Markdown(format_doc_terminal(doc)))
        console.print()


# ═════════════════════════════════════════════════════════════════════════════
# codescope graph
# ═════════════════════════════════════════════════════════════════════════════

@app.command("graph")
def cmd_graph(
    project: str = _proj_opt(),
    full:    bool = typer.Option(False, "--full", help="Print all nodes, not just summary"),
):
    """Print the dependency graph summary for a project."""
    from . import projects as P
    from .chat.context import build_graph_summary
    from .storage.graph import load_nodes, load_edges

    cache_dir = P.cache_dir(project)
    _check_index(project, cache_dir)

    if full:
        nodes = load_nodes(cache_dir)
        edges = load_edges(cache_dir)
        t = Table(title=f"Graph nodes — {project}", show_lines=True)
        t.add_column("ID",     style="cyan",  no_wrap=False, max_width=60)
        t.add_column("Kind",   style="yellow")
        t.add_column("Module", style="dim")
        for n in sorted(nodes.values(), key=lambda x: x.get("kind", "")):
            t.add_row(n.get("id", ""), n.get("kind", ""), n.get("module", ""))
        console.print(t)
        console.print(f"\n[dim]{len(nodes)} nodes  {len(edges)} edges[/dim]")
    else:
        summary = build_graph_summary(cache_dir, max_tokens=4000)
        console.print(f"\n[bold cyan]Graph summary — {project}[/bold cyan]\n")
        console.print(summary)


# ═════════════════════════════════════════════════════════════════════════════
# codescope chat
# ═════════════════════════════════════════════════════════════════════════════

@app.command("chat")
def cmd_chat(
    project:       str  = _proj_opt(),
    new:           bool = typer.Option(False, "--new",  help="Force a new session"),
    llm_model:     str  = _llm_opt(),
    embedder_name: str  = _embedder_opt(),
    verbose:       bool = _verbose_opt(),
):
    """Start an interactive chat REPL for a project."""
    from . import projects as P
    from .models.llm import LLM
    from .models.embedder import Embedder
    from .chat.agent import repl

    project_root = _resolve(project)
    cache_dir    = P.cache_dir(project)
    session_dir  = P.session_dir(project)
    _check_index(project, cache_dir)

    llm      = LLM(model=llm_model)
    embedder = Embedder(model_name=embedder_name)
    _check_ollama(llm)

    repl(
        slug=project, project_root=project_root,
        cache_dir=cache_dir, session_dir=session_dir,
        llm=llm, embedder=embedder,
        new_session=new, verbose=verbose,
    )


# ═════════════════════════════════════════════════════════════════════════════
# codescope ask
# ═════════════════════════════════════════════════════════════════════════════

@app.command("ask")
def cmd_ask(
    query:         str  = typer.Argument(..., help="Question to ask"),
    project:       str  = _proj_opt(),
    llm_model:     str  = _llm_opt(),
    embedder_name: str  = _embedder_opt(),
    verbose:       bool = _verbose_opt(),
):
    """Ask a single question and print the answer (no REPL)."""
    from . import projects as P
    from .models.llm import LLM
    from .models.embedder import Embedder
    from .chat.agent import ask_once

    project_root = _resolve(project)
    cache_dir    = P.cache_dir(project)
    session_dir  = P.session_dir(project)
    _check_index(project, cache_dir)

    llm      = LLM(model=llm_model)
    embedder = Embedder(model_name=embedder_name)
    _check_ollama(llm)

    answer = ask_once(
        query=query, slug=project,
        project_root=project_root, cache_dir=cache_dir, session_dir=session_dir,
        llm=llm, embedder=embedder, verbose=verbose,
    )
    _print_answer(answer)


# ═════════════════════════════════════════════════════════════════════════════
# codescope export
# ═════════════════════════════════════════════════════════════════════════════

@app.command("export")
def cmd_export(
    project: str = _proj_opt(),
    to:      str = typer.Option(None, "--to", help="Output zip path"),
):
    """Bundle cache and sessions for a project into a zip file."""
    import zipfile
    from .config import CACHE_ROOT, SESSIONS_ROOT

    out = Path(to) if to else Path(f"./{project}.zip")
    _resolve(project)
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for root_dir in (CACHE_ROOT / project, SESSIONS_ROOT / project):
            if root_dir.exists():
                for fp in root_dir.rglob("*"):
                    if fp.is_file():
                        zf.write(fp, fp.relative_to(CACHE_ROOT.parent))
    console.print(f"[green]✓ Exported {project!r} → {out}  ({out.stat().st_size/1024:.0f} KB)[/green]")


# ═════════════════════════════════════════════════════════════════════════════
# codescope menu   (interactive FMP-style numbered menu)
# ═════════════════════════════════════════════════════════════════════════════

@app.command("menu")
def cmd_menu(
    llm_model:     str = _llm_opt(),
    embedder_name: str = _embedder_opt(),
):
    """Interactive numbered menu — select project and action without typing commands."""
    from . import projects as P
    from .log_utils import print_banner, safe_input, section_line

    from codescope import __version__
    print_banner("(menu)", version=__version__)

    projects = P.discover()
    if not projects:
        console.print("[yellow]No Android projects found.[/yellow]")
        raise typer.Exit(1)

    active_project: str | None = None

    ACTIONS = [
        ("Index project (skeleton + graph + enrich + embed)",  "index"),
        ("Index project — skeleton only",                       "index-skel"),
        ("Index project — skip LLM enrichment",                 "index-noenrich"),
        ("Index project — skip embedding",                      "index-noembed"),
        ("Index project — HITL governance mode",                "index-hitl"),
        ("Status (show cache manifest)",                        "status"),
        ("Diff (files changed since last index)",               "diff"),
        ("Grep (search source files)",                          "grep"),
        ("Semantic search",                                     "search"),
        ("Graph summary",                                       "graph"),
        ("Chat (interactive REPL)",                             "chat"),
        ("Ask one question",                                    "ask"),
        ("Sessions (list saved chats)",                         "sessions"),
        ("Reset (delete cache + sessions)",                     "reset"),
        ("Export cache + sessions to zip",                      "export"),
        ("Help — show documentation",                           "help"),
    ]

    while True:
        section_line()
        console.print(f"\n[bold cyan]Active project:[/bold cyan] {active_project or '[dim]none selected[/dim]'}\n")

        console.print("[bold]Select project:[/bold]")
        for i, (slug, path) in enumerate(projects, 1):
            from .config import CACHE_ROOT
            indexed = "[green]✓[/green]" if (CACHE_ROOT / slug / "manifest.json").exists() else "[red]✗[/red]"
            marker  = " [bold cyan]◀[/bold cyan]" if slug == active_project else ""
            console.print(f"  {i:>2}. {indexed} {slug}{marker}")

        console.print("\n[bold]Actions:[/bold]")
        for i, (label, _) in enumerate(ACTIONS, len(projects) + 1):
            console.print(f"  {i:>2}. {label}")

        console.print(f"\n  {'Q':>2}. Quit")
        section_line()

        raw = safe_input("\nChoice: ").strip().upper()

        if raw in ("Q", "QUIT", "EXIT"):
            console.print("[dim]Goodbye.[/dim]")
            break

        try:
            choice = int(raw)
        except ValueError:
            console.print("[red]Enter a number or Q.[/red]")
            continue

        # Project selection
        if 1 <= choice <= len(projects):
            active_project = projects[choice - 1][0]
            console.print(f"[green]✓ Active project: {active_project}[/green]")
            continue

        # Action
        action_idx = choice - len(projects) - 1
        if 0 <= action_idx < len(ACTIONS):
            if active_project is None:
                console.print("[yellow]Select a project first.[/yellow]")
                continue
            _, action = ACTIONS[action_idx]
            _menu_dispatch(action, active_project, llm_model, embedder_name)
        else:
            console.print("[red]Invalid choice.[/red]")


def _menu_dispatch(action: str, project: str, llm_model: str, embedder_name: str) -> None:
    """Execute an action from the menu by calling the appropriate Typer command logic."""
    from .log_utils import safe_input
    ctx = {"project": project, "llm_model": llm_model, "embedder_name": embedder_name}

    if action == "index":
        cmd_index(**ctx, skeleton_only=False, no_enrich=False, no_embed=False,
                  force=False, hitl=False, suffix="menu", verbose=False)
    elif action == "index-skel":
        cmd_index(**ctx, skeleton_only=True, no_enrich=False, no_embed=False,
                  force=False, hitl=False, suffix="menu", verbose=False)
    elif action == "index-noenrich":
        cmd_index(**ctx, skeleton_only=False, no_enrich=True, no_embed=False,
                  force=False, hitl=False, suffix="menu", verbose=False)
    elif action == "index-noembed":
        cmd_index(**ctx, skeleton_only=False, no_enrich=False, no_embed=True,
                  force=False, hitl=False, suffix="menu", verbose=False)
    elif action == "index-hitl":
        cmd_index(**ctx, skeleton_only=False, no_enrich=False, no_embed=False,
                  force=False, hitl=True, suffix="menu", verbose=True)
    elif action == "status":
        cmd_status(project=project)
    elif action == "diff":
        cmd_diff(project=project)
    elif action == "grep":
        pattern = safe_input("  Grep pattern: ").strip()
        if pattern:
            cmd_grep(pattern=pattern, project=project, path=".",
                     ignore_case=False, fixed=False, max_results=60, sessions=False)
    elif action == "search":
        query = safe_input("  Search query: ").strip()
        if query:
            cmd_search(query=query, project=project, k=8, ext="", embedder_name=embedder_name)
    elif action == "graph":
        cmd_graph(project=project, full=False)
    elif action == "chat":
        cmd_chat(project=project, new=False, llm_model=llm_model,
                 embedder_name=embedder_name, verbose=False)
    elif action == "ask":
        query = safe_input("  Question: ").strip()
        if query:
            cmd_ask(query=query, project=project, llm_model=llm_model,
                    embedder_name=embedder_name, verbose=False)
    elif action == "sessions":
        cmd_sessions(project=project)
    elif action == "reset":
        cmd_reset(project=project, yes=False)
    elif action == "export":
        cmd_export(project=project, to=None)
    elif action == "help":
        cmd_help(topic="", pager=False)


# ═════════════════════════════════════════════════════════════════════════════
# codescope help
# ═════════════════════════════════════════════════════════════════════════════

# Map of topic keyword → anchor / section heading substring in codescope.md
_TOPICS: dict[str, str] = {
    "index":    "## Indexing pipeline",
    "chat":     "### chat",
    "ask":      "### ask",
    "tools":    "## Agent and tools",
    "setup":    "## Quick start",
    "concepts": "## Storage layout",
    "tips":     "## Tips",
    "commands": "## Commands",
    "storage":  "## Storage layout",
    "models":   "## Models",
    "config":   "## Configuration",
    "hitl":     "## HITL governance",
    "sessions": "## Session logs",
    "logs":     "## Run logs",
    "launcher": "## Bash launcher",
}


@app.command("help")
def cmd_help(
    topic: str = typer.Argument("", help=(
        "Optional topic: index|chat|ask|tools|setup|concepts|tips|commands|"
        "storage|models|config|hitl|sessions|logs|launcher"
    )),
    pager: bool = typer.Option(False, "--pager/--no-pager",
                                help="Pipe through 'less -R' (auto-handles ANSI colours)"),
):
    """
    Show codescope documentation in the terminal.

    Examples:
      codescope help
      codescope help index
      codescope help tools
      codescope help tips
      codescope help --pager
    """
    import os
    from rich.markdown import Markdown
    from rich.console import Console as _Console
    from .config import WORKSPACE_ROOT

    doc_path = WORKSPACE_ROOT / "docs" / "codescope.md"
    if not doc_path.exists():
        console.print(
            f"[red]Documentation file not found at {doc_path}[/red]\n"
            "It should have been created alongside this package.\n"
            "If you deleted it, run:\n"
            "  git checkout docs/codescope.md"
        )
        raise typer.Exit(1)

    full_text = doc_path.read_text(encoding="utf-8")

    # Topic filtering: find the matching section and extract until the next same-level heading
    if topic:
        topic_lower = topic.strip().lower()
        anchor = _TOPICS.get(topic_lower)

        if anchor is None:
            console.print(
                f"[yellow]Unknown topic: {topic!r}[/yellow]\n"
                f"Available topics: {', '.join(sorted(_TOPICS))}"
            )
            raise typer.Exit(1)

        lines          = full_text.splitlines()
        start          = None
        heading_prefix = anchor.split()[0]  # "##" or "###"

        for i, line in enumerate(lines):
            if start is None:
                if anchor.lower() in line.lower():
                    start = i
            else:
                if line.startswith(heading_prefix + " ") and i > start:
                    full_text = "\n".join(lines[start:i])
                    break
        else:
            if start is not None:
                full_text = "\n".join(lines[start:])
            else:
                console.print(
                    f"[yellow]Section for topic {topic!r} not found in docs.[/yellow]"
                )
                raise typer.Exit(1)

    md = Markdown(full_text)

    if pager:
        # Ensure less renders ANSI colour codes properly
        os.environ.setdefault("LESS", "")
        if "-R" not in os.environ["LESS"]:
            os.environ["LESS"] = "-R " + os.environ["LESS"]
        with console.pager(styles=True):
            console.print(md)
    else:
        # Print directly — Rich handles colour rendering natively in the terminal
        console.print(md)
