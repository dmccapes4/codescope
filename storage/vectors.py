"""Dense embedding storage: embeddings.npy (float32) + embeddings.meta.jsonl."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def load(cache_dir: Path) -> tuple[np.ndarray | None, list[dict]]:
    """Return (matrix, meta_list). Returns (None, []) if no embeddings exist yet."""
    npy_path  = cache_dir / "embeddings.npy"
    meta_path = cache_dir / "embeddings.meta.jsonl"

    if not npy_path.exists() or not meta_path.exists():
        return None, []

    matrix = np.load(str(npy_path))
    meta: list[dict] = []
    with open(meta_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    meta.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    if len(meta) != matrix.shape[0]:
        # Mismatch — truncate to the shorter side for safety
        n = min(len(meta), matrix.shape[0])
        meta   = meta[:n]
        matrix = matrix[:n]

    return matrix, meta


def append(cache_dir: Path, new_vectors: np.ndarray, new_meta: list[dict[str, Any]]) -> None:
    """Append new_vectors (shape N×dim) and corresponding meta records."""
    assert len(new_meta) == new_vectors.shape[0], "vectors and meta must have same length"

    npy_path  = cache_dir / "embeddings.npy"
    meta_path = cache_dir / "embeddings.meta.jsonl"

    if npy_path.exists():
        existing = np.load(str(npy_path))
        combined = np.vstack([existing, new_vectors]).astype(np.float32)
    else:
        combined = new_vectors.astype(np.float32)

    np.save(str(npy_path), combined)

    offset = combined.shape[0] - len(new_meta)
    with open(meta_path, "a", encoding="utf-8") as f:
        for i, m in enumerate(new_meta):
            m = dict(m)
            m["row"] = offset + i
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
            f.flush()


def reset(cache_dir: Path) -> None:
    for name in ("embeddings.npy", "embeddings.meta.jsonl"):
        p = cache_dir / name
        if p.exists():
            p.unlink()


def search(
    matrix: np.ndarray,
    meta: list[dict],
    query_vec: np.ndarray,
    k: int = 8,
    filter_ext: list[str] | None = None,
) -> list[dict]:
    """Brute-force cosine similarity search. query_vec must already be normalised."""
    if matrix is None or matrix.shape[0] == 0:
        return []

    scores = matrix @ query_vec          # shape (N,)
    order  = np.argsort(scores)[::-1]    # descending

    results = []
    for idx in order:
        m = meta[idx]
        if filter_ext and not any(m.get("file", "").endswith(e) for e in filter_ext):
            continue
        results.append({
            "file":   m.get("file", ""),
            "lines":  m.get("lines", []),
            "score":  float(scores[idx]),
            "symbol": m.get("symbol", ""),
            "chunk_text": m.get("chunk_text", ""),
        })
        if len(results) >= k:
            break

    return results
