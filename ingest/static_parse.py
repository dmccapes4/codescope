"""Phase 2 — Cheap regex/XML extractors: emit graph nodes and edges without an LLM."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

from ..storage.graph import append_node, append_edge, load_nodes, rewrite_nodes

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

_KT_PACKAGE    = re.compile(r"^\s*package\s+([\w.]+)", re.MULTILINE)
_KT_IMPORT     = re.compile(r"^\s*import\s+([\w.*]+)", re.MULTILINE)
_KT_CLASS      = re.compile(
    r"^\s*(?:(?:@\w+\s+)*)"
    r"(?:(?:abstract|sealed|open|data|inner|enum|annotation|value)\s+)*"
    r"(class|interface|object|fun)\s+(\w+)",
    re.MULTILINE,
)
_COMPOSABLE    = re.compile(r"@Composable", re.MULTILINE)
_GRADLE_DEP    = re.compile(
    r"""(?:implementation|api|ksp|kapt|testImplementation|androidTestImplementation)\s*\(?\s*["']([\w.\-:]+)["']""",
    re.MULTILINE,
)
_GRADLE_PROJ   = re.compile(r"""project\s*\(\s*["'](:[^"']+)["']""", re.MULTILINE)
_SETTINGS_INCL = re.compile(r"""include\s*\(\s*["'](:[^"']+)["']""", re.MULTILINE)
_TOML_LIB      = re.compile(r"""^(\w[\w-]*)\s*=\s*\{[^}]*module\s*=\s*["']([\w.\-:]+)["']""", re.MULTILINE)

# Android XML namespaces
_ANDROID_NS = "http://schemas.android.com/apk/res/android"
_TOOLS_NS   = "http://schemas.android.com/tools"


def _attrib(el, local: str) -> str:
    return (
        el.attrib.get(f"{{{_ANDROID_NS}}}{local}")
        or el.attrib.get(local)
        or ""
    )


# ---------------------------------------------------------------------------
# Kotlin / Java
# ---------------------------------------------------------------------------

def _parse_kotlin(path: Path, rel: str, module: str, cache_dir: Path) -> None:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return

    pkg_m = _KT_PACKAGE.search(text)
    package = pkg_m.group(1) if pkg_m else ""

    file_id = f"file::{rel}"
    node: dict[str, Any] = {
        "id":       file_id,
        "kind":     "file",
        "file":     rel,
        "module":   module,
        "ext":      path.suffix,
        "package":  package,
        "lines":    [1, text.count("\n") + 1],
    }
    append_node(cache_dir, node)

    # Import edges: file → package prefix
    for m in _KT_IMPORT.finditer(text):
        imp = m.group(1).rstrip(".*")
        append_edge(cache_dir, {
            "src": file_id,
            "dst": f"pkg::{imp}",
            "rel": "imports",
            "via": "static",
        })

    # Symbol nodes (classes, top-level objects, @Composable funs)
    has_composable = bool(_COMPOSABLE.search(text))
    for m in _KT_CLASS.finditer(text):
        kind   = m.group(1)      # class / interface / object / fun
        sym    = m.group(2)
        sym_id = f"kt::{package}.{sym}" if package else f"kt::{sym}"
        sym_kind = "composable" if (kind == "fun" and has_composable) else kind
        append_node(cache_dir, {
            "id":     sym_id,
            "kind":   sym_kind,
            "file":   rel,
            "module": module,
            "lines":  [m.start(0), m.start(0)],   # approximate start
        })
        append_edge(cache_dir, {
            "src": file_id,
            "dst": sym_id,
            "rel": "declares",
            "via": "static",
        })


# ---------------------------------------------------------------------------
# Gradle build files
# ---------------------------------------------------------------------------

def _parse_gradle(path: Path, rel: str, module: str, cache_dir: Path) -> None:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return

    file_id = f"file::{rel}"
    node: dict[str, Any] = {
        "id":     file_id,
        "kind":   "file",
        "file":   rel,
        "module": module,
        "ext":    path.suffix,
        "lines":  [1, text.count("\n") + 1],
    }
    append_node(cache_dir, node)

    # External dependencies
    for m in _GRADLE_DEP.finditer(text):
        dep = m.group(1)
        append_edge(cache_dir, {
            "src": file_id,
            "dst": f"lib::{dep}",
            "rel": "depends_on",
            "via": "static",
        })

    # Project module dependencies
    for m in _GRADLE_PROJ.finditer(text):
        dep_mod = m.group(1)
        append_edge(cache_dir, {
            "src": module or file_id,
            "dst": dep_mod,
            "rel": "module_dep",
            "via": "static",
        })


def _parse_settings(path: Path, cache_dir: Path) -> None:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return

    root_id = "module::root"
    append_node(cache_dir, {"id": root_id, "kind": "module", "file": path.name, "module": ":root"})

    for m in _SETTINGS_INCL.finditer(text):
        mod = m.group(1)
        append_node(cache_dir, {"id": f"module::{mod}", "kind": "module", "module": mod})
        append_edge(cache_dir, {
            "src": root_id,
            "dst": f"module::{mod}",
            "rel": "includes",
            "via": "static",
        })


# ---------------------------------------------------------------------------
# AndroidManifest.xml
# ---------------------------------------------------------------------------

def _parse_manifest(path: Path, rel: str, module: str, cache_dir: Path) -> None:
    try:
        tree = ET.parse(path)
    except ET.ParseError:
        return

    root = tree.getroot()
    app_el = root.find("application")
    if app_el is None:
        return

    app_class = _attrib(app_el, "name")
    if app_class:
        append_node(cache_dir, {
            "id":     f"manifest::application::{app_class}",
            "kind":   "application",
            "file":   rel,
            "module": module,
            "name":   app_class,
        })

    for tag in ("activity", "service", "receiver", "provider"):
        for el in app_el.findall(tag):
            name = _attrib(el, "name")
            if not name:
                continue
            exported = _attrib(el, "exported")
            node_id  = f"manifest::{tag}::{name}"
            append_node(cache_dir, {
                "id":       node_id,
                "kind":     tag,
                "file":     rel,
                "module":   module,
                "name":     name,
                "exported": exported,
            })

    for perm_el in root.findall("uses-permission"):
        perm = _attrib(perm_el, "name")
        if perm:
            append_node(cache_dir, {
                "id":     f"manifest::permission::{perm}",
                "kind":   "permission",
                "file":   rel,
                "module": module,
                "name":   perm,
            })


# ---------------------------------------------------------------------------
# libs.versions.toml
# ---------------------------------------------------------------------------

def _parse_toml(path: Path, rel: str, cache_dir: Path) -> None:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return

    # Python 3.11+ has tomllib
    if sys.version_info >= (3, 11):
        import tomllib
        try:
            data = tomllib.loads(text)
        except Exception:
            data = {}
    else:
        data = {}

    for alias, mod_str in _TOML_LIB.findall(text):
        append_node(cache_dir, {
            "id":    f"lib::{mod_str}",
            "kind":  "library",
            "file":  rel,
            "alias": alias,
        })


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def run(
    project_root: Path,
    cache_dir: Path,
    files_jsonl: Path,
    force: bool = False,
    progress_cb=None,   # callable(rel, category, status) where status in {"parsed","cached","oversize","node-only"}
) -> tuple[int, int]:
    """
    Parse every file listed in files.jsonl and emit graph nodes/edges.
    Skips files that already have a node in graph.nodes.jsonl (unless force=True).
    Returns (processed, skipped).
    """
    from ..storage.manifest import repair_jsonl
    repair_jsonl(cache_dir / "graph.nodes.jsonl")
    repair_jsonl(cache_dir / "graph.edges.jsonl")

    existing_nodes: set[str] = set()
    if not force:
        nodes_path = cache_dir / "graph.nodes.jsonl"
        if nodes_path.exists():
            with open(nodes_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            r = json.loads(line)
                            fp = r.get("file", "")
                            if fp:
                                existing_nodes.add(fp)
                        except json.JSONDecodeError:
                            pass

    if force:
        (cache_dir / "graph.nodes.jsonl").unlink(missing_ok=True)
        (cache_dir / "graph.edges.jsonl").unlink(missing_ok=True)

    processed = skipped = 0

    with open(files_jsonl, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            rel      = rec["path"]
            category = rec["category"]
            module   = rec.get("module", ":app")
            abs_path = project_root / rel

            if not force and rel in existing_nodes:
                skipped += 1
                if progress_cb:
                    progress_cb(rel, category, "cached")
                continue
            if rec.get("oversize"):
                skipped += 1
                if progress_cb:
                    progress_cb(rel, category, "oversize")
                continue

            if category == "source.kotlin":
                _parse_kotlin(abs_path, rel, module, cache_dir)
                processed += 1
                if progress_cb:
                    progress_cb(rel, category, "parsed")
            elif category in ("gradle.kts", "gradle.kts.settings"):
                if "settings" in Path(rel).stem.lower():
                    _parse_settings(abs_path, cache_dir)
                else:
                    _parse_gradle(abs_path, rel, module, cache_dir)
                processed += 1
                if progress_cb:
                    progress_cb(rel, category, "parsed")
            elif category == "manifest.xml":
                _parse_manifest(abs_path, rel, module, cache_dir)
                processed += 1
                if progress_cb:
                    progress_cb(rel, category, "parsed")
            elif category == "versions.toml":
                _parse_toml(abs_path, rel, cache_dir)
                processed += 1
                if progress_cb:
                    progress_cb(rel, category, "parsed")
            elif category == "source.java":
                _parse_kotlin(abs_path, rel, module, cache_dir)  # Java regex is close enough
                processed += 1
                if progress_cb:
                    progress_cb(rel, category, "parsed")
            else:
                # Non-parseable: still emit a file node so the embedder can reference it
                append_node(cache_dir, {
                    "id":     f"file::{rel}",
                    "kind":   "file",
                    "file":   rel,
                    "module": module,
                    "ext":    rec["ext"],
                })
                skipped += 1
                if progress_cb:
                    progress_cb(rel, category, "node-only")

    return processed, skipped
