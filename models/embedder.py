"""sentence-transformers embedding wrapper with lazy loading and GPU auto-detect."""
from __future__ import annotations

import numpy as np
from functools import lru_cache
from typing import List

from ..config import DEFAULT_EMBEDDER, EMBED_BATCH_SIZE, EMBED_DEVICE


@lru_cache(maxsize=4)
def _load_model(model_name: str, device: str):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name, device=device)


def _best_device() -> str:
    """
    Resolve the embedding device.

    Respects CODESCOPE_EMBED_DEVICE (config.EMBED_DEVICE).  Default is "cpu"
    so that the sentence-transformer does not consume VRAM that qwen needs to
    keep all 29 layers on-GPU.  Set CODESCOPE_EMBED_DEVICE=cuda if you have
    ≥ 8 GB VRAM and want maximum embedding throughput.
    """
    if EMBED_DEVICE and EMBED_DEVICE.lower() != "auto":
        return EMBED_DEVICE.lower()
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


class Embedder:
    def __init__(self, model_name: str = DEFAULT_EMBEDDER, device: str | None = None):
        self.model_name = model_name
        self.device = device or _best_device()
        self._model = None

    def _ensure_loaded(self):
        if self._model is None:
            self._model = _load_model(self.model_name, self.device)

    def is_loaded(self) -> bool:
        return self._model is not None

    def warm(self) -> None:
        """Force the embedding weights into memory now (so the cost is visible
        at REPL startup rather than mid-turn). Safe to call multiple times."""
        self._ensure_loaded()

    @property
    def dim(self) -> int:
        self._ensure_loaded()
        try:
            return self._model.get_embedding_dimension()
        except AttributeError:
            return self._model.get_sentence_embedding_dimension()

    def encode(self, texts: List[str], batch_size: int = EMBED_BATCH_SIZE) -> np.ndarray:
        """Return float32 array of shape (N, dim), L2-normalised."""
        self._ensure_loaded()
        vecs = self._model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.array(vecs, dtype=np.float32)

    def encode_query(self, text: str) -> np.ndarray:
        """Encode a single query string, return shape (dim,)."""
        return self.encode([text])[0]
