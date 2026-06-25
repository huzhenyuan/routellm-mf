"""
Pytest fixtures shared across the test suite.

The embedding model (sentence-transformers) requires a network download on
first use.  When running in environments without internet access we substitute
a deterministic mock that returns unit-norm random vectors of the correct
dimensionality (384, matching all-MiniLM-L6-v2).
"""

import numpy as np
import pytest

MOCK_DIM = 384


class _MockEmbedder:
    """Deterministic unit-norm embedder that needs no network."""

    def __init__(self):
        self.dim = MOCK_DIM

    def embed(self, text: str) -> np.ndarray:
        rng = np.random.default_rng(abs(hash(text)) % (2**31))
        v = rng.standard_normal(self.dim).astype(np.float32)
        return v / np.linalg.norm(v)

    def embed_batch(self, texts) -> np.ndarray:
        return np.stack([self.embed(t) for t in texts])


@pytest.fixture()
def mock_embedder(monkeypatch):
    """Replace LocalEmbedder with a no-network mock for unit tests."""
    from routellm_mf import embedding as emb_module

    mock = _MockEmbedder()

    monkeypatch.setattr(emb_module.LocalEmbedder, "embed", lambda self, text: mock.embed(text))
    monkeypatch.setattr(
        emb_module.LocalEmbedder, "embed_batch", lambda self, texts: mock.embed_batch(texts)
    )
    monkeypatch.setattr(emb_module.LocalEmbedder, "dim", property(lambda self: MOCK_DIM))
    return mock
