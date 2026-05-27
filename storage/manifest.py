"""Read/write cache/<project>/manifest.json and track per-file sha256 hashes."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Manifest (the summary JSON written at the end of each index run)
# ---------------------------------------------------------------------------

def load(cache_dir: Path) -> dict[str, Any]:
    p = cache_dir / "manifest.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def save(cache_dir: Path, data: dict[str, Any]) -> None:
    p = cache_dir / "manifest.json"
    p.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def build(
    slug: str,
    project_root: Path,
    cache_dir: Path,
    embedder: str,
    embedder_dim: int,
    llm: str,
) -> dict[str, Any]:
    """Compute statistics from cache files and return a fresh manifest dict."""
    file_count  = _jsonl_count(cache_dir / "files.jsonl")
    chunk_count = _jsonl_count(cache_dir / "embeddings.meta.jsonl")
    node_count  = _jsonl_count(cache_dir / "graph.nodes.jsonl")
    edge_count  = _jsonl_count(cache_dir / "graph.edges.jsonl")

    files_jsonl = cache_dir / "files.jsonl"
    sha = _sha256(files_jsonl) if files_jsonl.exists() else ""

    return {
        "project":              slug,
        "project_root":         project_root.as_posix(),
        "indexed_at":           datetime.now(timezone.utc).isoformat(),
        "embedder":             embedder,
        "embedder_dim":         embedder_dim,
        "llm":                  llm,
        "prompt_version":       1,
        "file_count":           file_count,
        "chunk_count":          chunk_count,
        "node_count":           node_count,
        "edge_count":           edge_count,
        "sha256_of_files_jsonl": sha,
    }


def _jsonl_count(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


# ---------------------------------------------------------------------------
# Per-file hash index (used by ingestion to skip unchanged files)
# ---------------------------------------------------------------------------

def load_hashes(cache_dir: Path) -> dict[str, str]:
    """Return {relative_posix_path: sha256} from files.jsonl."""
    p = cache_dir / "files.jsonl"
    if not p.exists():
        return {}
    result: dict[str, str] = {}
    with open(p, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                result[rec["path"]] = rec["sha256"]
            except (json.JSONDecodeError, KeyError):
                continue
    return result


def repair_jsonl(path: Path) -> None:
    """Truncate a JSONL file at the last successfully parsed line (crash recovery)."""
    if not path.exists():
        return
    lines = path.read_bytes().splitlines()
    good = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            json.loads(line)
            good.append(line)
        except json.JSONDecodeError:
            break   # stop at first bad line; discard the rest
    path.write_bytes(b"\n".join(good) + (b"\n" if good else b""))
