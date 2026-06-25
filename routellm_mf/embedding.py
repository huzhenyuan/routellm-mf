"""
Local embedding module.

Wraps sentence-transformers so the rest of the codebase only sees
a simple ``embed(text) -> np.ndarray`` interface.
"""

from __future__ import annotations

from typing import List, Union

import numpy as np

from routellm_mf.config import EmbeddingConfig


class LocalEmbedder:
    """Thin wrapper around a sentence-transformers model.

    The underlying model is loaded lazily on first use so that importing
    the package is fast even when sentence-transformers is large.
    """

    def __init__(self, config: EmbeddingConfig) -> None:
        self._config = config
        self._model = None  # lazy

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def embed(self, text: str) -> np.ndarray:
        """Return the L2-normalised embedding vector for *text*."""
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Return an (N, D) array of L2-normalised embeddings."""
        model = self._get_model()
        truncated = [t[: self._config.max_chars] for t in texts]
        embeddings = model.encode(
            truncated,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return np.array(embeddings, dtype=np.float32)

    @property
    def dim(self) -> int:
        """Embedding dimensionality."""
        return self._get_model().get_sentence_embedding_dimension()

    # ------------------------------------------------------------------ #
    # Internal
    # ------------------------------------------------------------------ #

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # noqa: PLC0415

            self._model = SentenceTransformer(
                self._config.model_name,
                device=self._config.device,
            )
        return self._model
