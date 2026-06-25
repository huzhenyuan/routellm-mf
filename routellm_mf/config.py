"""
Configuration dataclass for the router.

Supports loading from a YAML file or constructing directly in Python.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

import yaml


@dataclass
class ModelConfig:
    """Connection settings for one LLM endpoint (strong or weak model)."""

    # Model identifier sent to the API (e.g. "llama3:8b", "gpt-4o")
    model: str
    # Base URL for an OpenAI-compatible API.
    # Ollama default : http://localhost:11434/v1
    # LM Studio default: http://localhost:1234/v1
    # OpenAI           : https://api.openai.com/v1  (or omit)
    base_url: Optional[str] = None
    # API key — for local models any non-empty string works
    api_key: str = "local"
    # Generation parameters
    temperature: float = 0.7
    max_tokens: int = 2048


@dataclass
class EmbeddingConfig:
    """Settings for the local sentence-transformer embedding model."""

    # Any model from https://www.sbert.net/docs/pretrained_models.html
    # Recommended fast models:
    #   "all-MiniLM-L6-v2"   — 22 MB, English, very fast
    #   "paraphrase-multilingual-MiniLM-L12-v2" — multilingual
    model_name: str = "all-MiniLM-L6-v2"
    # Device: "cpu", "cuda", or "mps"
    device: str = "cpu"
    # Truncate input to this many characters before embedding
    max_chars: int = 2000


@dataclass
class ClassifierConfig:
    """Settings for the difficulty classifier."""

    # Path to a persisted sklearn classifier (.joblib).
    # Leave empty to use the built-in rule-based heuristic.
    model_path: Optional[str] = None
    # Routing threshold: score >= threshold → strong model
    threshold: float = 0.5
    # Weight for the rule-based heuristic vs. the trained model [0..1].
    # 1.0 = pure heuristic, 0.0 = pure trained model (requires model_path)
    heuristic_weight: float = 1.0


@dataclass
class RouterConfig:
    """Top-level configuration for the router."""

    strong_model: ModelConfig = field(
        default_factory=lambda: ModelConfig(
            model="llama3:70b",
            base_url="http://localhost:11434/v1",
        )
    )
    weak_model: ModelConfig = field(
        default_factory=lambda: ModelConfig(
            model="llama3:8b",
            base_url="http://localhost:11434/v1",
        )
    )
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    classifier: ClassifierConfig = field(default_factory=ClassifierConfig)

    # If True, print routing decisions to stdout
    verbose: bool = False

    # ------------------------------------------------------------------ #
    @classmethod
    def from_yaml(cls, path: str) -> "RouterConfig":
        """Load configuration from a YAML file."""
        with open(path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
        return cls._from_dict(raw or {})

    @classmethod
    def _from_dict(cls, d: dict) -> "RouterConfig":
        def _model(key: str, defaults: dict) -> ModelConfig:
            cfg = {**defaults, **d.get(key, {})}
            return ModelConfig(**{k: v for k, v in cfg.items() if k in ModelConfig.__dataclass_fields__})

        strong_defaults = {
            "model": "llama3:70b",
            "base_url": "http://localhost:11434/v1",
            "api_key": "local",
            "temperature": 0.7,
            "max_tokens": 2048,
        }
        weak_defaults = {
            "model": "llama3:8b",
            "base_url": "http://localhost:11434/v1",
            "api_key": "local",
            "temperature": 0.7,
            "max_tokens": 2048,
        }

        emb_raw = d.get("embedding", {})
        clf_raw = d.get("classifier", {})

        return cls(
            strong_model=_model("strong_model", strong_defaults),
            weak_model=_model("weak_model", weak_defaults),
            embedding=EmbeddingConfig(
                **{k: v for k, v in emb_raw.items() if k in EmbeddingConfig.__dataclass_fields__}
            )
            if emb_raw
            else EmbeddingConfig(),
            classifier=ClassifierConfig(
                **{k: v for k, v in clf_raw.items() if k in ClassifierConfig.__dataclass_fields__}
            )
            if clf_raw
            else ClassifierConfig(),
            verbose=d.get("verbose", False),
        )

    def to_yaml(self, path: str) -> None:
        """Persist this configuration to a YAML file."""
        import dataclasses

        with open(path, "w", encoding="utf-8") as fh:
            yaml.dump(dataclasses.asdict(self), fh, default_flow_style=False)
