"""Thin async-capable Ollama HTTP client."""
from __future__ import annotations

import json
from typing import Any

import httpx

from ..config import (
    OLLAMA_BASE_URL,
    DEFAULT_LLM,
    OLLAMA_NUM_GPU,
    OLLAMA_NUM_CTX,
    OLLAMA_KEEP_ALIVE,
    LLM_TIMEOUT,
)


class OllamaError(RuntimeError):
    pass


class LLM:
    def __init__(
        self,
        model: str = DEFAULT_LLM,
        base_url: str = OLLAMA_BASE_URL,
        timeout: float | None = None,
        num_gpu: int | None = None,
        num_ctx: int | None = None,
    ):
        self.model    = model
        self.base_url = base_url.rstrip("/")
        self.timeout  = LLM_TIMEOUT if timeout is None else timeout
        # Per-instance GPU layer override.  None → use OLLAMA_NUM_GPU global.
        # Set to 0 to run a model entirely on CPU (e.g. planner on 6 GB VRAM).
        self._num_gpu = num_gpu
        # Per-instance context window override (None → fall back to OLLAMA_NUM_CTX
        # Per-instance num_ctx override (None → use OLLAMA_NUM_CTX from config).
        self._num_ctx = num_ctx

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        try:
            r = httpx.get(f"{self.base_url}/api/tags", timeout=5)
            return r.status_code == 200
        except Exception:
            return False

    def list_models(self) -> list[str]:
        r = httpx.get(f"{self.base_url}/api/tags", timeout=10)
        r.raise_for_status()
        return [m["name"] for m in r.json().get("models", [])]

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        system: str | None = None,
        json_mode: bool = False,
        num_predict: int = 1500,
        temperature: float = 0.1,
        keep_alive: int | str | None = None,
    ) -> str:
        """Return the model's response as a string. Raises OllamaError on failure."""
        payload: dict[str, Any] = {
            "model":      self.model,
            "prompt":     prompt,
            "stream":     False,
            "keep_alive": OLLAMA_KEEP_ALIVE if keep_alive is None else keep_alive,
            "options": {
                "num_predict": num_predict,
                "temperature": temperature,
                "num_gpu":     OLLAMA_NUM_GPU if self._num_gpu is None else self._num_gpu,
                "num_ctx":     OLLAMA_NUM_CTX if self._num_ctx is None else self._num_ctx,
            },
        }
        if system:
            payload["system"] = system
        if json_mode:
            payload["format"] = "json"

        try:
            r = httpx.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout,
            )
            r.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise OllamaError(f"Ollama HTTP {e.response.status_code}: {e.response.text[:400]}")
        except httpx.RequestError as e:
            raise OllamaError(f"Ollama connection error: {e}")

        return r.json().get("response", "").strip()

    def warm(self) -> None:
        """Load weights once at REPL start so the first user question is faster."""
        try:
            self.generate("{}", system="Reply with {}", json_mode=True, num_predict=8, temperature=0)
        except OllamaError:
            pass

    def generate_json(
        self,
        prompt: str,
        system: str | None = None,
        num_predict: int = 1500,
        temperature: float = 0.1,
    ) -> dict | list:
        """Generate and parse JSON. Raises OllamaError if parsing fails."""
        raw = self.generate(
            prompt,
            system=system,
            json_mode=True,
            num_predict=num_predict,
            temperature=temperature,
        )
        try:
            return json.loads(raw)
        except json.JSONDecodeError as e:
            raise OllamaError(f"LLM returned invalid JSON: {e}\nRaw: {raw[:300]}")
