"""Phase 1 — Walk a project tree, classify files, hash them, write files.jsonl."""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Generator

from ..config import (
    EXCLUDED_DIRS,
    EXCLUDED_FILENAMES,
    EXCLUDED_SUFFIXES,
    MAX_FILE_BYTES,
)


# ---------------------------------------------------------------------------
# File classification
# ---------------------------------------------------------------------------

_EXT_TO_CATEGORY = {
    ".kt":      "source.kotlin",
    ".kts":     "gradle.kts",
    ".java":    "source.java",
    ".xml":     "resource.xml",   # refined below
    ".toml":    "versions.toml",
    ".gradle":  "gradle.groovy",
    ".proto":   "proto",
    ".md":      "markdown",
    ".txt":     "text.plain",
    ".json":    "data.json",
    ".properties": "properties",
}


def _classify(rel: str, name: str, size: int) -> str:
    ext  = Path(name).suffix.lower()
    stem = Path(name).stem.lower()
    cat  = _EXT_TO_CATEGORY.get(ext, "other")

    # Refine XML category
    if cat == "resource.xml":
        if name == "AndroidManifest.xml":
            return "manifest.xml"
        if "layout" in rel:
            return "layout.xml"

    # Markdown: only README.md at project root, or any .md inside a docs/ folder.
    # All other .md files (changelogs, generated docs, etc.) are excluded.
    if cat == "markdown":
        parts = Path(rel).parts
        in_docs     = any(p.lower() == "docs" for p in parts[:-1])
        is_root_readme = len(parts) == 1 and Path(name).stem.lower() == "readme"
        if not (in_docs or is_root_readme):
            return "other"

    # Large text
    if cat == "text.plain" and size > 50_000:
        return "text.large"

    # gradle.kts — distinguish settings vs module
    if cat == "gradle.kts":
        if stem == "settings":
            return "gradle.kts.settings"
        if stem == "libs" or "versions" in rel:
            return "versions.toml"

    return cat


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _count_lines(path: Path) -> int:
    try:
        with open(path, "rb") as f:
            return f.read().count(b"\n")
    except OSError:
        return 0


def _module_from_path(rel_posix: str, module_map: dict[str, str]) -> str:
    """Find the deepest module prefix that matches this relative path."""
    best = ""
    best_mod = ""
    for prefix, mod in module_map.items():
        if rel_posix.startswith(prefix) and len(prefix) > len(best):
            best = prefix
            best_mod = mod
    return best_mod or ":app"


# ---------------------------------------------------------------------------
# Module map: parse settings.gradle.kts for include(":mod") lines
# ---------------------------------------------------------------------------

_INCLUDE_RE = re.compile(r'include\s*\(\s*["\']([^"\']+)["\']')


def _build_module_map(project_root: Path) -> dict[str, str]:
    """Return {relative_dir_prefix: gradle_module_name}."""
    settings = project_root / "settings.gradle.kts"
    if not settings.exists():
        settings = project_root / "settings.gradle"
    if not settings.exists():
        return {"app/": ":app"}

    text = settings.read_text(encoding="utf-8", errors="replace")
    result: dict[str, str] = {}
    for m in _INCLUDE_RE.finditer(text):
        mod = m.group(1)              # e.g. ":feature:bookmarks:impl"
        # convert ":feature:bookmarks:impl" → "feature/bookmarks/impl/"
        prefix = mod.lstrip(":").replace(":", "/") + "/"
        result[prefix] = mod
    if not result:
        result["app/"] = ":app"
    return result


# ---------------------------------------------------------------------------
# Main walk
# ---------------------------------------------------------------------------

def walk(project_root: Path) -> Generator[dict, None, None]:
    """Yield file records for every indexable file in project_root."""
    module_map = _build_module_map(project_root)

    for dirpath, dirnames, filenames in os.walk(project_root):
        # Prune excluded directories in-place
        dirnames[:] = [
            d for d in dirnames
            if d not in EXCLUDED_DIRS and not d.startswith(".")
        ]

        for name in filenames:
            if name in EXCLUDED_FILENAMES:
                continue
            ext = Path(name).suffix.lower()
            if ext in EXCLUDED_SUFFIXES:
                continue

            abs_path = Path(dirpath) / name
            try:
                size = abs_path.stat().st_size
            except OSError:
                continue

            if size > MAX_FILE_BYTES:
                # Still record it as a stub, but mark it oversize
                rel = abs_path.relative_to(project_root).as_posix()
                yield {
                    "path":     rel,
                    "size":     size,
                    "ext":      ext,
                    "loc":      0,
                    "sha256":   "",
                    "category": "binary.large",
                    "module":   _module_from_path(rel, module_map),
                    "oversize": True,
                }
                continue

            rel = abs_path.relative_to(project_root).as_posix()
            category = _classify(rel, name, size)
            sha = _sha256(abs_path)
            loc = _count_lines(abs_path)

            yield {
                "path":     rel,
                "size":     size,
                "ext":      ext,
                "loc":      loc,
                "sha256":   sha,
                "category": category,
                "module":   _module_from_path(rel, module_map),
                "oversize": False,
            }


# ---------------------------------------------------------------------------
# Write files.jsonl
# ---------------------------------------------------------------------------

def run(
    project_root: Path,
    cache_dir: Path,
    existing_hashes: dict[str, str],
    force: bool = False,
    progress_cb=None,   # callable(rel, category, status) where status in {"new","changed","unchanged","oversize"}
) -> tuple[int, int]:
    """
    Walk the project, write/update cache/files.jsonl.
    Returns (added_count, unchanged_count).
    """
    from ..storage.manifest import repair_jsonl

    out_path = cache_dir / "files.jsonl"
    repair_jsonl(out_path)

    # Load existing records
    existing: dict[str, dict] = {}
    if out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        r = json.loads(line)
                        existing[r["path"]] = r
                    except (json.JSONDecodeError, KeyError):
                        pass

    added = 0
    unchanged = 0
    new_records: dict[str, dict] = {}

    for rec in walk(project_root):
        rel      = rec["path"]
        category = rec.get("category", "")
        cached_sha = existing.get(rel, {}).get("sha256", "")

        if rec.get("oversize"):
            new_records[rel] = rec
            added += 1
            if progress_cb:
                progress_cb(rel, category, "oversize")
        elif not force and cached_sha and cached_sha == rec.get("sha256", ""):
            new_records[rel] = existing[rel]
            unchanged += 1
            if progress_cb:
                progress_cb(rel, category, "unchanged")
        else:
            new_records[rel] = rec
            added += 1
            status = "new" if rel not in existing else "changed"
            if progress_cb:
                progress_cb(rel, category, status)

    # Rewrite the whole file (fast, typically <5 MB)
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in new_records.values():
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return added, unchanged
