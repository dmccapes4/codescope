"""ProScope — feature folders under project docs/ and doc extraction from model output."""
from __future__ import annotations

import re
from pathlib import Path

from .tools import tool_write_file, ToolError

# Model emits: ### PROSCOPE_DOC: docs/<slug>/FILE.md\n```markdown\n...\n```
_PROSCOPE_DOC_RE = re.compile(
    r"^###\s+PROSCOPE_DOC:\s+(?P<path>docs/[A-Za-z0-9_./-]+\.md)\s*\n+"
    r"```(?:markdown|md)?\n(?P<body>.*?)\n```",
    re.DOTALL | re.MULTILINE,
)

_ACTIVE_FEATURE_FILE = "proscope.active"


def slugify_feature(text: str) -> str:
    """Turn free text into a docs/ folder slug."""
    s = text.lower().strip()
    s = re.sub(r"^feature\s*:\s*", "", s)
    s = re.sub(r"^new\s+feature\s*:\s*", "", s)
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return (s.strip("_")[:60] or "feature")


def list_feature_folders(project_root: Path) -> list[str]:
    docs = project_root / "docs"
    if not docs.is_dir():
        return []
    return sorted(
        p.name for p in docs.iterdir()
        if p.is_dir() and not p.name.startswith(".")
    )


def ensure_docs_root(project_root: Path) -> Path:
    docs = project_root / "docs"
    docs.mkdir(parents=True, exist_ok=True)
    return docs


def feature_dir(project_root: Path, slug: str) -> Path:
    d = ensure_docs_root(project_root) / slug
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_active_feature(session_dir: Path) -> str | None:
    p = session_dir / _ACTIVE_FEATURE_FILE
    if p.is_file():
        slug = p.read_text(encoding="utf-8").strip()
        return slug or None
    return None


def save_active_feature(session_dir: Path, slug: str) -> None:
    session_dir.mkdir(parents=True, exist_ok=True)
    (session_dir / _ACTIVE_FEATURE_FILE).write_text(slug.strip(), encoding="utf-8")


def resolve_feature_slug(
    query: str,
    project_root: Path,
    session_dir: Path,
    explicit: str | None = None,
) -> str | None:
    """Pick the active feature slug for this turn."""
    if explicit:
        return slugify_feature(explicit)

    # Explicit slug in query: "feature slug: wire_backend" or "docs/wire_backend/"
    m = re.search(r"(?:feature\s+slug|feature)\s*:\s*([a-z0-9_/-]+)", query, re.I)
    if m:
        return slugify_feature(m.group(1).split("/")[0])

    m = re.search(r"\bdocs/([a-z0-9_]+)", query, re.I)
    if m:
        return m.group(1).lower()

    # Continue session feature unless operator starts a new one
    if re.search(r"\bnew\s+feature\b", query, re.I):
        # "New feature: simplify onboarding" → slugify remainder after colon
        m = re.search(r"new\s+feature\s*:\s*(.+)", query, re.I)
        if m:
            return slugify_feature(m.group(1))
        return None

    return load_active_feature(session_dir)


def load_feature_docs_context(project_root: Path, slug: str | None, max_chars: int = 12_000) -> str:
    """Read IMPLEMENTATION_PLAN + STRATEGY_PHASE_* for prompt injection."""
    if not slug:
        return "(no active feature — set with --feature or: feature slug: my_feature)"
    folder = project_root / "docs" / slug
    if not folder.is_dir():
        return f"(feature docs/{slug}/ not created yet — propose IMPLEMENTATION_PLAN.md this turn)"

    chunks: list[str] = []
    plan = folder / "IMPLEMENTATION_PLAN.md"
    if plan.is_file():
        chunks.append(f"### {plan.relative_to(project_root).as_posix()}\n{plan.read_text(encoding='utf-8')}")
    for p in sorted(folder.glob("STRATEGY_PHASE_*.md")):
        chunks.append(f"### {p.relative_to(project_root).as_posix()}\n{p.read_text(encoding='utf-8')}")

    if not chunks:
        return f"(docs/{slug}/ exists but empty — propose IMPLEMENTATION_PLAN.md)"

    text = "\n\n".join(chunks)
    if len(text) > max_chars:
        text = text[:max_chars] + "\n… [truncated — read full files on disk]"
    return text


def format_feature_folder_list(project_root: Path) -> str:
    folders = list_feature_folders(project_root)
    if not folders:
        return "(none yet — ProScope creates docs/<feature_slug>/ per initiative)"
    return "\n".join(f"- docs/{f}/" for f in folders)


def parse_proscope_docs(answer: str) -> list[tuple[str, str]]:
    """Return (project-relative path, body) pairs from PROSCOPE_DOC blocks."""
    out: list[tuple[str, str]] = []
    for m in _PROSCOPE_DOC_RE.finditer(answer):
        rel = m.group("path").replace("\\", "/").lstrip("./")
        body = m.group("body").rstrip() + "\n"
        out.append((rel, body))
    return out


def persist_proscope_docs(
    answer: str,
    project_root: Path,
    session_dir: Path,
    feature_slug: str | None,
    print_fn=print,
) -> tuple[str, list[str]]:
    """Write PROSCOPE_DOC blocks to disk; return (updated answer, written paths)."""
    docs = parse_proscope_docs(answer)
    if not docs:
        return answer, []

    written: list[str] = []
    slug_from_path: str | None = feature_slug

    for rel, body in docs:
        rel = rel.replace("\\", "/")
        if not rel.startswith("docs/") or not rel.endswith(".md"):
            continue
        parts = rel.split("/")
        if len(parts) >= 2:
            slug_from_path = parts[1]

        try:
            tool_write_file(
                path=rel,
                content=body,
                overwrite=True,
                project_root=project_root,
            )
            written.append(rel)
            print_fn(f"[proscope] Wrote {rel} ({len(body)} bytes)")
        except ToolError as e:
            print_fn(f"[proscope] Failed to write {rel}: {e}")

    if slug_from_path:
        save_active_feature(session_dir, slug_from_path)

    if written:
        summary = "\n\n---\n**ProScope wrote feature docs:** " + ", ".join(f"`{w}`" for w in written)
        answer = answer + summary

    return answer, written


def strip_proscope_doc_blocks_for_display(answer: str) -> str:
    """Optional: keep full blocks in terminal (operator reviews content). No-op for now."""
    return answer
