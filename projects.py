"""Discover and resolve Android Studio projects alongside the workspace root."""
from pathlib import Path
from typing import List, Tuple

from .config import WORKSPACE_ROOT, CACHE_ROOT, SESSIONS_ROOT


def _is_android_project(path: Path) -> bool:
    return (
        (path / "settings.gradle.kts").exists()
        or (path / "settings.gradle").exists()
        or (path / "app" / "build.gradle.kts").exists()
        or (path / "app" / "build.gradle").exists()
    )


def discover() -> List[Tuple[str, Path]]:
    """Return sorted list of (slug, absolute_path) for all Android projects."""
    results = []
    for entry in sorted(WORKSPACE_ROOT.iterdir()):
        if not entry.is_dir():
            continue
        if entry.name.startswith("."):
            continue
        if entry.name in {"codescope", "cache", "sessions", "reports", "docs"}:
            continue
        if _is_android_project(entry):
            results.append((entry.name, entry))
    return results


def resolve(slug: str) -> Path:
    """Return the absolute path for a project slug, raising ValueError if unknown."""
    mapping = dict(discover())
    if slug not in mapping:
        available = list(mapping.keys())
        raise ValueError(
            f"Project {slug!r} not found.\n"
            f"Available projects: {available}\n"
            f"Run `codescope projects` for details."
        )
    return mapping[slug]


def cache_dir(slug: str) -> Path:
    d = CACHE_ROOT / slug
    d.mkdir(parents=True, exist_ok=True)
    return d


def session_dir(slug: str) -> Path:
    d = SESSIONS_ROOT / slug
    d.mkdir(parents=True, exist_ok=True)
    return d


def cache_size_bytes(slug: str) -> int:
    d = CACHE_ROOT / slug
    if not d.exists():
        return 0
    return sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
