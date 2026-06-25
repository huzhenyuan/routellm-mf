"""
Tests for the core modules (no LLM calls required).

Embedding-dependent tests use the ``mock_embedder`` fixture (conftest.py) so
they work in offline / CI environments without downloading a model.
"""

import numpy as np
import pytest

from routellm_mf.classifier import DifficultyClassifier
from routellm_mf.config import ClassifierConfig, EmbeddingConfig, RouterConfig
from routellm_mf.embedding import LocalEmbedder
from routellm_mf.router import Router


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------


def test_router_config_defaults():
    cfg = RouterConfig()
    assert cfg.strong_model.model
    assert cfg.weak_model.model
    assert cfg.classifier.threshold == 0.5
    assert cfg.classifier.heuristic_weight == 1.0


def test_router_config_from_dict():
    cfg = RouterConfig._from_dict(
        {
            "strong_model": {"model": "big-llm", "base_url": "http://localhost:9999/v1"},
            "weak_model": {"model": "small-llm"},
            "classifier": {"threshold": 0.6},
            "verbose": True,
        }
    )
    assert cfg.strong_model.model == "big-llm"
    assert cfg.strong_model.base_url == "http://localhost:9999/v1"
    assert cfg.weak_model.model == "small-llm"
    assert cfg.classifier.threshold == 0.6
    assert cfg.verbose is True


def test_router_config_yaml_roundtrip(tmp_path):
    cfg = RouterConfig()
    cfg.classifier.threshold = 0.72
    path = str(tmp_path / "cfg.yaml")
    cfg.to_yaml(path)
    cfg2 = RouterConfig.from_yaml(path)
    assert abs(cfg2.classifier.threshold - 0.72) < 1e-6


# ---------------------------------------------------------------------------
# Embedding tests — use mock_embedder fixture (no network required)
# ---------------------------------------------------------------------------


def test_embedder_shape(mock_embedder):
    emb = LocalEmbedder(EmbeddingConfig())
    vec = emb.embed("Hello world")
    assert vec.ndim == 1
    assert vec.shape[0] > 0


def test_embedder_batch(mock_embedder):
    emb = LocalEmbedder(EmbeddingConfig())
    vecs = emb.embed_batch(["Hello", "World"])
    assert vecs.shape[0] == 2
    assert vecs.shape[1] > 0


def test_embedder_normalised(mock_embedder):
    emb = LocalEmbedder(EmbeddingConfig())
    vec = emb.embed("Test normalisation")
    norm = float(np.linalg.norm(vec))
    assert abs(norm - 1.0) < 1e-4


# ---------------------------------------------------------------------------
# Classifier / heuristic tests  (pure Python — no embedding needed)
# ---------------------------------------------------------------------------


def test_classifier_easy_prompt():
    clf = DifficultyClassifier(ClassifierConfig(threshold=0.5))
    result = clf.classify("Hi!")
    assert result.score < 0.5
    assert not result.use_strong_model


def test_classifier_hard_prompt():
    # A clearly complex prompt should exceed the threshold.
    clf = DifficultyClassifier(ClassifierConfig(threshold=0.4))
    result = clf.classify(
        "Prove the Riemann hypothesis step by step and derive all intermediate lemmas."
    )
    assert result.score >= 0.4
    assert result.use_strong_model


def test_classifier_very_hard_prompt():
    # A very complex prompt must reach a high score.
    clf = DifficultyClassifier(ClassifierConfig(threshold=0.5))
    result = clf.classify(
        "Implement a red-black tree in Python, prove its O(log n) insertion complexity, "
        "and compare it with AVL trees and B-trees for database indexing use cases. "
        "Additionally, derive the amortised cost using the potential method."
    )
    assert result.score >= 0.5
    assert result.use_strong_model


def test_classifier_reasoning_populated():
    clf = DifficultyClassifier(ClassifierConfig(threshold=0.5))
    result = clf.classify("Implement a recursive merge sort algorithm.")
    assert isinstance(result.reasoning, dict)
    assert len(result.reasoning) > 0


def test_classifier_threshold_override():
    # Very low threshold — even simple prompts route to strong
    clf = DifficultyClassifier(ClassifierConfig(threshold=0.0))
    result = clf.classify("Hello")
    assert result.use_strong_model  # score >= 0.0 always true


def test_classifier_score_clamped():
    clf = DifficultyClassifier(ClassifierConfig(threshold=0.5))
    for text in ["Hi", "a" * 5000, "Prove " * 100]:
        result = clf.classify(text)
        assert 0.0 <= result.score <= 1.0


def test_classifier_fit_predict(mock_embedder, tmp_path):
    """Train on mock embeddings, verify save/load round-trip."""
    easy = ["Hi", "Hello", "What is 2+2?", "Define noun."]
    hard = [
        "Prove that sqrt(2) is irrational using contradiction.",
        "Derive the gradient of a sigmoid through backpropagation.",
        "Implement a red-black tree insertion algorithm in Python.",
        "Compare transformer vs RNN for NLP sequence tasks.",
    ]
    emb = LocalEmbedder(EmbeddingConfig())
    embeddings = emb.embed_batch(easy + hard)
    labels = np.array([0] * len(easy) + [1] * len(hard))

    clf = DifficultyClassifier(ClassifierConfig(threshold=0.5, heuristic_weight=0.0))
    clf.fit(embeddings, labels)

    # Save and reload
    path = str(tmp_path / "clf.joblib")
    clf.save_trained_model(path)

    clf2 = DifficultyClassifier(
        ClassifierConfig(threshold=0.5, heuristic_weight=0.0, model_path=path)
    )
    result = clf2.classify("", embedding=emb.embed("What is 1+1?"))
    assert isinstance(result.score, float)
    assert 0.0 <= result.score <= 1.0


# ---------------------------------------------------------------------------
# Router tests (no LLM calls, mock embedder)
# ---------------------------------------------------------------------------


def test_router_score_returns_difficulty_result(mock_embedder):
    router = Router(RouterConfig())
    result = router.score("Hello, how are you?")
    assert 0.0 <= result.score <= 1.0


def test_router_routes_correctly(mock_embedder):
    cfg = RouterConfig()
    cfg.classifier.threshold = 0.0  # Force everything to strong
    router = Router(cfg)
    result = router.score("Hi")
    assert result.use_strong_model


def test_router_decision_model_names(mock_embedder):
    cfg = RouterConfig()
    cfg.strong_model.model = "big"
    cfg.weak_model.model = "small"
    cfg.classifier.threshold = 0.99  # Force everything to weak
    router = Router(cfg)
    decision = router._decide("Hello world")
    assert decision.model_used == "small"
