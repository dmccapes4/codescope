"""Local Ollama health check — codescope does not use remote hosts or fallbacks."""
from __future__ import annotations

import httpx

PROBE_TIMEOUT = 3.0
DEFAULT_LOCAL_HOST = "http://127.0.0.1:11434"


def probe_ollama(base_url: str, timeout: float = PROBE_TIMEOUT) -> list[str] | None:
    """Return installed model tags, or None if the server is unreachable."""
    try:
        r = httpx.get(f"{base_url.rstrip('/')}/api/tags", timeout=timeout)
        if r.status_code != 200:
            return None
        return [m.get("name", "") for m in r.json().get("models", []) if m.get("name")]
    except Exception:
        return None
