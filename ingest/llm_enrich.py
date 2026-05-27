"""Phase 3 — LLM enrichment: add summaries, calls, ref_files, and tags to graph nodes."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from ..config import (
    MAX_LLM_FILE_BYTES,
    MIN_LLM_FILE_BYTES,
    PROMPT_VERSION,
    EMBEDDABLE_CATEGORIES,
)
from ..models.llm import LLM, OllamaError
from ..storage.graph import load_nodes, rewrite_nodes, append_edge


# ---------------------------------------------------------------------------
# Pydantic schema for one LLM enrichment response
# ---------------------------------------------------------------------------

class EnrichResult(BaseModel):
    file:      str
    summary:   str = ""
    exports:   list[str] = Field(default_factory=list)
    calls:     list[str] = Field(default_factory=list)
    ref_files: list[str] = Field(default_factory=list)
    tags:      list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_SYSTEM = """\
You are a static-analysis assistant for Android codebases.
You will be given the contents of a single source file.
Respond with a single JSON object (no markdown, no explanation) matching exactly:
{
  "file": "<the relative path you were given>",
  "summary": "<1-2 sentence plain-English description of this file's purpose>",
  "exports": ["<ClassName or functionName this file exposes publicly>", ...],
  "calls":   ["<ClassName or functionName this file calls or instantiates>", ...],
  "ref_files": ["<relative path of another file that is clearly referenced>", ...],
  "tags":    ["<short keyword>", ...]
}
Only include ref_files you can directly see in imports or string literals. Do not invent paths.
Keep exports, calls, and tags lists under 10 items each.
"""

_FILE_TEMPLATE = """\
Analyse this file and return JSON as instructed.

=== FILE: {rel} ===
{content}
"""


def _truncate(text: str, max_bytes: int = MAX_LLM_FILE_BYTES) -> str:
    """Return head + tail of text so total length ≤ max_bytes."""
    if len(text.encode("utf-8")) <= max_bytes:
        return text
    half = max_bytes // 2 - 50
    head = text[:half]
    tail = text[-half:]
    return head + "\n\n[... truncated ...]\n\n" + tail


def _cache_key(sha256: str, llm_name: str) -> str:
    return f"{sha256}|{llm_name}|v{PROMPT_VERSION}"


# ---------------------------------------------------------------------------
# Per-project response cache (enrich_cache.jsonl)
# ---------------------------------------------------------------------------

def _load_enrich_cache(cache_dir: Path) -> dict[str, dict]:
    path = cache_dir / "enrich_cache.jsonl"
    if not path.exists():
        return {}
    result: dict[str, dict] = {}
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                result[r["key"]] = r["response"]
            except (json.JSONDecodeError, KeyError):
                continue
    return result


def _save_enrich_entry(cache_dir: Path, key: str, response: dict) -> None:
    path = cache_dir / "enrich_cache.jsonl"
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"key": key, "response": response}, ensure_ascii=False) + "\n")
        f.flush()


# ---------------------------------------------------------------------------
# Path filter helper
# ---------------------------------------------------------------------------

def _path_matches(rel: str, include_paths: list[str]) -> bool:
    """Return True if rel is under any include_paths (prefix or exact match)."""
    if not include_paths:
        return True
    rel_norm = rel.replace("\\", "/")
    for p in include_paths:
        p_norm = p.replace("\\", "/").rstrip("/")
        if rel_norm == p_norm or rel_norm.startswith(p_norm + "/"):
            return True
    return False


# ---------------------------------------------------------------------------
# Main run
# ---------------------------------------------------------------------------

def run(
    project_root: Path,
    cache_dir: Path,
    files_jsonl: Path,
    llm: LLM,
    force: bool = False,
    progress_cb=None,                        # callable(done, total, rel, skipped=False)
    include_paths: list[str] | None = None,  # if set, only enrich matching files
) -> tuple[int, int]:
    """
    Enrich file nodes with LLM-generated summaries, calls, and ref_file edges.
    Returns (enriched_count, skipped_count).
    """
    include_paths = include_paths or []

    # Load file records — filtered to include_paths if specified
    file_records: list[dict] = []
    with open(files_jsonl, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                if r["category"] in EMBEDDABLE_CATEGORIES and not r.get("oversize"):
                    if r.get("size", 0) >= MIN_LLM_FILE_BYTES:
                        if _path_matches(r["path"], include_paths):
                            file_records.append(r)
            except (json.JSONDecodeError, KeyError):
                continue

    # Load existing graph nodes (to check which already have summaries)
    nodes = load_nodes(cache_dir)

    # Load LLM response cache
    enrich_cache = _load_enrich_cache(cache_dir)

    enriched = skipped = 0
    total = len(file_records)

    for i, rec in enumerate(file_records):
        rel    = rec["path"]
        sha256 = rec.get("sha256", "")

        # Check if already enriched with same content
        existing_node = nodes.get(f"file::{rel}", {})
        if not force and existing_node.get("summary") and existing_node.get("sha256") == sha256:
            skipped += 1
            if progress_cb:
                progress_cb(i + 1, total, rel, skipped=True)
            continue

        cache_key = _cache_key(sha256, llm.model)

        # Check LLM response cache
        if not force and cache_key in enrich_cache:
            raw_resp = enrich_cache[cache_key]
        else:
            abs_path = project_root / rel
            try:
                content = abs_path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                skipped += 1
                continue

            prompt = _FILE_TEMPLATE.format(rel=rel, content=_truncate(content))

            raw_resp = None
            for attempt in range(2):
                try:
                    raw_resp = llm.generate_json(prompt, system=_SYSTEM, num_predict=600, temperature=0.1)
                    break
                except OllamaError as e:
                    if attempt == 0:
                        continue
                    # Give up on this file
                    break

            if raw_resp is None:
                skipped += 1
                if progress_cb:
                    progress_cb(i + 1, total, rel, skipped=True)
                continue

            _save_enrich_entry(cache_dir, cache_key, raw_resp)

        # Validate with pydantic
        try:
            result = EnrichResult.model_validate(raw_resp)
        except ValidationError:
            skipped += 1
            if progress_cb:
                progress_cb(i + 1, total, rel, skipped=True)
            continue

        # Update the file node with summary + tags
        node_id = f"file::{rel}"
        if node_id not in nodes:
            nodes[node_id] = {
                "id":     node_id,
                "kind":   "file",
                "file":   rel,
                "module": rec.get("module", ""),
                "ext":    rec.get("ext", ""),
            }
        nodes[node_id]["summary"] = result.summary
        nodes[node_id]["tags"]    = result.tags
        nodes[node_id]["sha256"]  = sha256

        # Emit call edges
        for sym in result.calls:
            append_edge(cache_dir, {
                "src": node_id,
                "dst": f"kt::{sym}",
                "rel": "calls",
                "via": "llm",
            })

        # Emit ref_file edges
        for ref_rel in result.ref_files:
            append_edge(cache_dir, {
                "src": node_id,
                "dst": f"file::{ref_rel}",
                "rel": "refs",
                "via": "llm",
            })

        enriched += 1
        if progress_cb:
            progress_cb(i + 1, total, rel, skipped=False)

    # Write back updated nodes
    rewrite_nodes(cache_dir, nodes)

    return enriched, skipped
