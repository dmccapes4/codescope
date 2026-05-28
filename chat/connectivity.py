"""Probe the configured Ollama host; fall back to a local server when unreachable.

Designed for the scenario where `OLLAMA_HOST` points to a tunnel (laptop → mac →
WSL → workstation) and the upstream goes away (power outage, dropped SSH, etc.).
We probe once at command start, and if the primary is dead AND a local fallback
is alive, the caller is given a `ResolvedHost` it can pass straight into `LLM(...)`.
No retry-on-every-turn; this is a startup decision so the user sees a clear banner.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable

import httpx


PROBE_TIMEOUT = 3.0  # seconds — anything past this on /api/tags means the host is unhealthy

# Preferred small-VRAM fallback answer models, in priority order. We pick the first
# tag that the fallback host actually has installed (no surprise pulls during a chat).
LAPTOP_MODEL_CANDIDATES: tuple[str, ...] = (
    "qwen2.5-coder:7b-instruct-q4_K_M",
    "qwen2.5-coder:7b",
    "qwen2.5:7b-instruct-q4_K_M",
    "qwen2.5:7b",
    "llama3.1:8b-instruct-q4_K_M",
    "llama3.1:8b",
    "llama3.2:3b",
)

DEFAULT_FALLBACK_HOST = "http://127.0.0.1:11434"
LAPTOP_NUM_CTX        = 12288


@dataclass
class ResolvedHost:
    base_url: str
    model:    str
    source:   str               # "primary" | "fallback"
    profile:  str               # "workstation-24gb" | "laptop-6gb" — hint for env overrides
    num_ctx:  int
    reason:   str = ""
    models_available: list[str] = None  # type: ignore[assignment]

    def is_degraded(self) -> bool:
        return self.source == "fallback"


def _normalize_url(url: str) -> str:
    return url.rstrip("/")


def _list_models(base_url: str, timeout: float = PROBE_TIMEOUT) -> list[str] | None:
    """Return the model tags installed on this Ollama server, or None if it's not reachable."""
    try:
        r = httpx.get(f"{_normalize_url(base_url)}/api/tags", timeout=timeout)
        if r.status_code != 200:
            return None
        return [m.get("name", "") for m in r.json().get("models", []) if m.get("name")]
    except Exception:
        return None


def _pick_model(preferred: str | None, available: list[str], candidates: Iterable[str]) -> str | None:
    """Choose the best model tag actually present on the host."""
    if preferred and preferred in available:
        return preferred
    for cand in candidates:
        if cand in available:
            return cand
    # Last resort: any qwen* tag, then anything at all.
    for tag in available:
        if "qwen" in tag.lower():
            return tag
    return available[0] if available else None


def resolve_ollama_host(
    requested_model: str | None,
    primary_url: str | None = None,
    fallback_url: str | None = None,
) -> ResolvedHost:
    """Decide which Ollama server (and which model) the chat should use right now.

    Inputs come from env / CLI flags so this function is pure-by-construction:
      primary_url    – OLLAMA_HOST (or whatever the caller currently expects to use)
      fallback_url   – CODESCOPE_FALLBACK_HOST (default 127.0.0.1:11434)
      requested_model – the model the user asked for (e.g. via --llm); may be None

    Returns a ResolvedHost. If primary is healthy this is just the primary echoed
    back. Otherwise we probe the fallback; if it has any usable model we degrade
    to it, and the caller mutates os.environ accordingly before importing config.
    """
    primary  = _normalize_url(primary_url  or os.environ.get("OLLAMA_HOST", DEFAULT_FALLBACK_HOST))
    fallback = _normalize_url(fallback_url or os.environ.get("CODESCOPE_FALLBACK_HOST", DEFAULT_FALLBACK_HOST))

    primary_models = _list_models(primary)
    if primary_models is not None:
        # Primary is alive — keep whatever ctx the caller already has via CODESCOPE_NUM_CTX.
        try:
            num_ctx = int(os.environ.get("CODESCOPE_NUM_CTX", "0"))
        except ValueError:
            num_ctx = 0
        return ResolvedHost(
            base_url = primary,
            model    = requested_model or "",
            source   = "primary",
            profile  = os.environ.get("CODESCOPE_PROFILE", "laptop-6gb"),
            num_ctx  = num_ctx,  # 0 = let config.py decide (don't override)
            reason   = "primary host reachable",
            models_available = primary_models,
        )

    # Primary is dead. If the fallback is the same host, there's no escape — surface that.
    if fallback == primary:
        return ResolvedHost(
            base_url = primary,
            model    = requested_model or "",
            source   = "primary",
            profile  = os.environ.get("CODESCOPE_PROFILE", "laptop-6gb"),
            num_ctx  = 0,
            reason   = "primary unreachable and no separate fallback configured",
            models_available = [],
        )

    fallback_models = _list_models(fallback)
    if fallback_models is None:
        return ResolvedHost(
            base_url = primary,
            model    = requested_model or "",
            source   = "primary",
            profile  = os.environ.get("CODESCOPE_PROFILE", "laptop-6gb"),
            num_ctx  = 0,
            reason   = f"both primary ({primary}) and fallback ({fallback}) unreachable",
            models_available = [],
        )

    chosen = _pick_model(requested_model, fallback_models, LAPTOP_MODEL_CANDIDATES)
    return ResolvedHost(
        base_url = fallback,
        model    = chosen or (requested_model or ""),
        source   = "fallback",
        profile  = "laptop-6gb",
        num_ctx  = LAPTOP_NUM_CTX,
        reason   = f"primary {primary} unreachable; using local Ollama at {fallback}",
        models_available = fallback_models,
    )


def apply_resolution_to_env(resolved: ResolvedHost, user_supplied_model: bool) -> None:
    """Mutate os.environ so a later `from .config import ...` reads the fallback config.

    Only override env vars the caller hasn't explicitly set, EXCEPT for OLLAMA_HOST
    (the whole point of the fallback is to redirect that one). `user_supplied_model`
    tells us whether to respect a `--llm` flag the user typed at the CLI.
    """
    if not resolved.is_degraded():
        return

    os.environ["OLLAMA_HOST"] = resolved.base_url
    if resolved.model and not user_supplied_model:
        os.environ["CODESCOPE_LLM"] = resolved.model
    os.environ.setdefault("CODESCOPE_PROFILE", resolved.profile)
    # On 6 GB don't load a second planner model — reuse the answer model instead.
    os.environ.setdefault("CODESCOPE_PLANNER_SAME_MODEL", "1")
    os.environ.setdefault("CODESCOPE_USE_PLANNER_LLM", "0")
    if resolved.num_ctx > 0:
        os.environ.setdefault("CODESCOPE_NUM_CTX", str(resolved.num_ctx))
    # Keep request timeouts modest — local 7B is much faster than remote tunneled 14B.
    os.environ.setdefault("CODESCOPE_LLM_TIMEOUT", "120")
