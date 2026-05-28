"""Read/write graph.nodes.jsonl and graph.edges.jsonl."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Generator, Optional


# ---------------------------------------------------------------------------
# Low-level streaming I/O
# ---------------------------------------------------------------------------

def _iter_jsonl(path: Path) -> Generator[dict, None, None]:
    if not path.exists():
        return
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def _append_jsonl(path: Path, record: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def iter_nodes(cache_dir: Path) -> Generator[dict, None, None]:
    yield from _iter_jsonl(cache_dir / "graph.nodes.jsonl")


def append_node(cache_dir: Path, node: dict[str, Any]) -> None:
    _append_jsonl(cache_dir / "graph.nodes.jsonl", node)


def load_nodes(cache_dir: Path) -> dict[str, dict]:
    """Return {node_id: node_record} — loads entire file into memory."""
    return {n["id"]: n for n in iter_nodes(cache_dir)}


def rewrite_nodes(cache_dir: Path, nodes: dict[str, dict]) -> None:
    """Overwrite nodes file with the given dict (used after enrichment updates)."""
    path = cache_dir / "graph.nodes.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for node in nodes.values():
            f.write(json.dumps(node, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Edges
# ---------------------------------------------------------------------------

def iter_edges(cache_dir: Path) -> Generator[dict, None, None]:
    yield from _iter_jsonl(cache_dir / "graph.edges.jsonl")


def append_edge(cache_dir: Path, edge: dict[str, Any]) -> None:
    _append_jsonl(cache_dir / "graph.edges.jsonl", edge)


def load_edges(cache_dir: Path) -> list[dict]:
    return list(iter_edges(cache_dir))


# ---------------------------------------------------------------------------
# In-memory graph for agent graph_lookup tool
# ---------------------------------------------------------------------------

class GraphIndex:
    def __init__(self, cache_dir: Path):
        self._nodes = load_nodes(cache_dir)
        edges = load_edges(cache_dir)
        # adjacency: node_id → {in: [...], out: [...]}
        self._adj: dict[str, dict[str, list]] = {}
        for e in edges:
            src, dst = e.get("src", ""), e.get("dst", "")
            self._adj.setdefault(src, {"in": [], "out": []})["out"].append(e)
            self._adj.setdefault(dst, {"in": [], "out": []})["in"].append(e)
        # file-path → node_id index
        self._by_file: dict[str, list[str]] = {}
        for nid, node in self._nodes.items():
            fp = node.get("file", "")
            if fp:
                self._by_file.setdefault(fp, []).append(nid)

    def lookup_node(self, node_id: str) -> Optional[dict]:
        node = self._nodes.get(node_id)
        if node is None:
            return None
        adj = self._adj.get(node_id, {"in": [], "out": []})
        return {"node": node, "in": adj["in"], "out": adj["out"]}

    def lookup_file(self, file_path: str) -> list[dict]:
        """
        Look up all graph nodes associated with a file path.

        Accepts:
        - Exact project-relative path: "app/src/.../PatientEntity.kt"
        - Partial suffix path:         "database/entities/PatientEntity.kt"
        - Bare filename:               "PatientEntity.kt"
        """
        fp = file_path.replace("\\", "/").lstrip("./")

        # 1. Exact match
        if fp in self._by_file:
            return [r for nid in self._by_file[fp] if (r := self.lookup_node(nid))]

        # 2. Suffix match: any indexed path that ends with fp
        matches: list[str] = [k for k in self._by_file if k.endswith(fp) or k.endswith("/" + fp)]
        if not matches:
            # 3. Bare filename match (e.g. "PatientEntity.kt" matches any path ending in that name)
            name = fp.split("/")[-1]
            matches = [k for k in self._by_file if k.split("/")[-1] == name]

        results = []
        for key in matches:
            for nid in self._by_file[key]:
                r = self.lookup_node(nid)
                if r:
                    results.append(r)
        return results

    def list_doc_files(self) -> list[str]:
        """README.md and docs/**/*.md paths present in the graph."""
        paths: list[str] = []
        for n in self._nodes.values():
            if n.get("kind") != "file":
                continue
            path = (n.get("file") or "").replace("\\", "/")
            if not path:
                continue
            low = path.lower()
            if low.endswith(".md") and (low == "readme.md" or low.startswith("docs/")):
                paths.append(path)
        return sorted(set(paths))

    def summary_lines(self, max_files: int = 30) -> list[str]:
        """Return lines for the context graph summary.

        All files are always listed (split into connected + zero-edge groups)
        so the agent knows every file that exists in the project.
        """
        from collections import Counter

        # edge count per node
        counts: Counter = Counter()
        for nid, adj in self._adj.items():
            counts[nid] = len(adj["in"]) + len(adj["out"])

        modules = sorted({
            n.get("module", "") for n in self._nodes.values()
            if n.get("kind") == "module" or n.get("module")
        } - {""})

        file_nodes = [n for n in self._nodes.values() if n.get("kind") == "file"]

        # Split: connected (has edges) vs standalone (docs, configs, etc.)
        connected   = sorted(
            [n for n in file_nodes if counts.get(n["id"], 0) > 0],
            key=lambda n: counts.get(n["id"], 0), reverse=True,
        )
        standalone  = sorted(
            [n for n in file_nodes if counts.get(n["id"], 0) == 0],
            key=lambda n: n.get("file", ""),
        )

        manifest_nodes = [
            n for n in self._nodes.values()
            if n.get("kind") in ("activity", "service", "receiver", "provider", "permission")
        ]

        lines = []
        if modules:
            lines.append(f"Modules: {', '.join(modules)}")
        lines.append(f"Files indexed: {len(file_nodes)} ({len(connected)} with graph edges, {len(standalone)} standalone)")
        lines.append("")

        if connected:
            lines.append("Connected files (sorted by graph edges):")
            for n in connected[:max_files]:
                summary = f"  — {n['summary']}" if n.get("summary") else ""
                lines.append(f"  {n.get('file', n['id'])}{summary}")

        if standalone:
            lines.append("")
            lines.append("Standalone files (docs, configs, resources):")
            for n in standalone:
                summary = f"  — {n['summary']}" if n.get("summary") else ""
                lines.append(f"  {n.get('file', n['id'])}{summary}")

        if manifest_nodes:
            lines.append("")
            lines.append("Manifest entries:")
            for n in manifest_nodes:
                lines.append(f"  [{n['kind']}] {n['id']}")
        return lines
