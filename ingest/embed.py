"""Phase 4 — Chunk files and embed them into embeddings.npy + embeddings.meta.jsonl."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..config import EMBEDDABLE_CATEGORIES, EMBED_BATCH_SIZE
from ..models.embedder import Embedder
from ..storage.vectors import append as vectors_append, reset as vectors_reset
from .chunk import chunk_file


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


def run(
    project_root: Path,
    cache_dir: Path,
    files_jsonl: Path,
    embedder: Embedder,
    force: bool = False,
    progress_cb=None,                        # callable(done, total, rel, n_chunks, skipped)
    include_paths: list[str] | None = None,  # if set, only embed matching files
) -> tuple[int, int]:
    """
    Chunk every embeddable file and write vectors + meta.
    Returns (chunks_written, files_skipped).
    """
    # Build set of already-embedded files (from meta.jsonl)
    embedded_files: set[str] = set()
    meta_path = cache_dir / "embeddings.meta.jsonl"
    if not force and meta_path.exists():
        with open(meta_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        embedded_files.add(json.loads(line)["file"])
                    except (json.JSONDecodeError, KeyError):
                        pass

    if force:
        vectors_reset(cache_dir)
        embedded_files.clear()

    # Also build sha256 map from files.jsonl for change detection
    sha_map: dict[str, str] = {}
    with open(files_jsonl, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                sha_map[r["path"]] = r.get("sha256", "")
            except (json.JSONDecodeError, KeyError):
                pass

    include_paths = include_paths or []

    # Collect files to embed
    to_embed: list[dict] = []
    with open(files_jsonl, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            if rec["category"] not in EMBEDDABLE_CATEGORIES:
                continue
            if rec.get("oversize"):
                continue
            if rec["path"] in embedded_files:
                continue
            if not _path_matches(rec["path"], include_paths):
                continue
            to_embed.append(rec)

    if not to_embed:
        return 0, 0

    # Process in batches of files (not chunks) to limit memory
    FILE_BATCH = 20
    total_chunks = 0
    files_skipped = 0
    total_files = len(to_embed)

    for batch_start in range(0, total_files, FILE_BATCH):
        batch = to_embed[batch_start : batch_start + FILE_BATCH]
        batch_texts: list[str] = []
        batch_meta:  list[dict] = []

        file_chunk_counts: list[tuple[str, int]] = []   # (rel, n_chunks) for logging

        for rec in batch:
            rel      = rec["path"]
            category = rec["category"]
            abs_path = project_root / rel

            chunks = chunk_file(abs_path, rel, category)
            if not chunks:
                files_skipped += 1
                if progress_cb:
                    done = batch_start + batch.index(rec) + 1
                    progress_cb(done, total_files, rel, n_chunks=0, skipped=True)
                continue

            n = 0
            for ch in chunks:
                text = ch["chunk_text"].strip()
                if not text:
                    continue
                batch_texts.append(text)
                meta = dict(ch)
                meta.pop("chunk_text", None)
                meta["sha256"] = sha_map.get(rel, "")
                batch_meta.append(meta)
                n += 1
            file_chunk_counts.append((rel, n))

        if not batch_texts:
            continue

        # Embed this batch of text strings
        vecs = embedder.encode(batch_texts, batch_size=EMBED_BATCH_SIZE)

        # Store chunk_text in meta for snippet retrieval later
        for i, text in enumerate(batch_texts):
            batch_meta[i]["chunk_text"] = text[:400]   # store trimmed snippet

        vectors_append(cache_dir, vecs, batch_meta)
        total_chunks += len(batch_texts)

        if progress_cb:
            for idx, (rel, n_chunks) in enumerate(file_chunk_counts):
                file_idx = batch_start + idx + 1
                progress_cb(file_idx, total_files, rel, n_chunks=n_chunks, skipped=False)

    return total_chunks, files_skipped
