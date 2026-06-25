"""
Difficulty classifier.

Architecture
============
The classifier produces a *difficulty score* in [0, 1] for a given prompt
embedding.  A score >= threshold means "hard → strong model".

Two modes are supported and can be blended:

1. **Rule-based heuristic** (always available, zero training required)
   Combines several prompt-surface signals:
   - token length
   - presence of reasoning / multi-step keywords
   - presence of code / math / science indicators
   - sentence count
   - number of sub-questions

2. **Trained logistic-regression model** (optional, loaded from ``model_path``)
   Trained on (embedding, label) pairs via ``Calibrator``.
   When both modes are active the final score is a weighted blend controlled
   by ``ClassifierConfig.heuristic_weight``.

Decision explanation
====================
``classify()`` returns a ``DifficultyResult`` with a ``reasoning`` dict so
callers can inspect exactly why a prompt was routed to a given model.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

from routellm_mf.config import ClassifierConfig

# ---------------------------------------------------------------------------
# Keywords that signal a hard / complex prompt
# ---------------------------------------------------------------------------
_HARD_PATTERNS: list[tuple[str, float]] = [
    # reasoning / logic
    (r"\b(prove|proof|derive|formal(ly)?|theorem|lemma|corollary)\b", 0.20),
    (r"\b(step[- ]by[- ]step|walk me through|explain how|reason(ing)?)\b", 0.10),
    (r"\b(compare and contrast|trade[- ]?off|pros and cons)\b", 0.10),
    # code / technical
    (r"\b(implement|algorithm|complexity|big[- ]?o|runtime|optimize|debug)\b", 0.15),
    (r"\b(sql|regex|recursion|dynamic programming|graph|tree|sort)\b", 0.12),
    # math / science
    (r"\b(integral|derivative|matrix|eigen|probability|statistics|calculus)\b", 0.18),
    (r"\b(chemistry|physics|biology|quantum|thermodynamics|relativity)\b", 0.15),
    # multi-part questions
    (r"\?.*\?", 0.08),                     # two or more question marks
    (r"\b(first|second|third|finally|lastly|additionally)\b", 0.06),
]

_EASY_PATTERNS: list[tuple[str, float]] = [
    (r"^(hi|hello|hey|greetings)[,!.]?\s*$", 0.40),
    (r"\b(what is|what are|who is|where is|when is|define)\b", 0.10),
    (r"\b(capital of|synonym|antonym|translate|how do you spell)\b", 0.12),
    (r"\b(thank you|thanks|bye|goodbye)\b", 0.30),
]

# ---------------------------------------------------------------------------
# Data class returned by classify()
# ---------------------------------------------------------------------------


@dataclass
class DifficultyResult:
    """Full result of a routing decision."""

    score: float
    """Difficulty score in [0, 1].  Higher → more complex."""

    use_strong_model: bool
    """True if the router should use the strong (large) model."""

    threshold: float
    """The configured threshold that was applied."""

    reasoning: Dict[str, float] = field(default_factory=dict)
    """Per-signal contributions to the final score (for explainability)."""

    heuristic_score: Optional[float] = None
    """Raw rule-based heuristic score before blending."""

    model_score: Optional[float] = None
    """Raw trained-model score before blending (None if no model loaded)."""

    # Convenience -----------------------------------------------------------
    def __str__(self) -> str:  # pragma: no cover
        model_label = "STRONG" if self.use_strong_model else "weak"
        lines = [f"Difficulty score: {self.score:.3f} (threshold={self.threshold:.2f}) → {model_label}"]
        if self.reasoning:
            lines.append("  Signal breakdown:")
            for k, v in sorted(self.reasoning.items(), key=lambda x: -abs(x[1])):
                lines.append(f"    {k:40s}: {v:+.3f}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------


class DifficultyClassifier:
    """Classify prompt difficulty using heuristics and/or a trained model."""

    def __init__(self, config: ClassifierConfig) -> None:
        self._config = config
        self._trained_model = None
        if config.model_path:
            self._load_trained_model(config.model_path)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def classify(self, prompt: str, embedding: Optional[np.ndarray] = None) -> DifficultyResult:
        """Score *prompt* and return a :class:`DifficultyResult`.

        Parameters
        ----------
        prompt:
            The raw user prompt text.
        embedding:
            Pre-computed embedding vector.  Required when a trained model is
            loaded and ``heuristic_weight < 1``.
        """
        hw = self._config.heuristic_weight
        threshold = self._config.threshold

        h_score, reasoning = self._heuristic_score(prompt)

        m_score: Optional[float] = None
        if self._trained_model is not None and embedding is not None and hw < 1.0:
            m_score = float(
                self._trained_model.predict_proba(embedding.reshape(1, -1))[0, 1]
            )
            reasoning["trained_model"] = m_score * (1.0 - hw)
            final_score = hw * h_score + (1.0 - hw) * m_score
        else:
            final_score = h_score

        # Clamp to [0, 1]
        final_score = float(np.clip(final_score, 0.0, 1.0))

        return DifficultyResult(
            score=final_score,
            use_strong_model=final_score >= threshold,
            threshold=threshold,
            reasoning=reasoning,
            heuristic_score=h_score,
            model_score=m_score,
        )

    def load_trained_model(self, path: str) -> None:
        """Load (or replace) the trained sklearn classifier from *path*."""
        self._load_trained_model(path)

    # ------------------------------------------------------------------ #
    # Heuristic scoring
    # ------------------------------------------------------------------ #

    def _heuristic_score(self, prompt: str) -> tuple[float, Dict[str, float]]:
        """Compute a difficulty score in [0, 1] from surface features."""
        text = prompt.strip().lower()
        reasoning: Dict[str, float] = {}
        score = 0.0

        # --- Length signal -----------------------------------------------
        word_count = len(text.split())
        # Short prompts (< 10 words) are likely easy; long ones more complex.
        length_score = min(word_count / 100.0, 0.30)
        reasoning["length"] = length_score
        score += length_score

        # --- Hard-keyword patterns ---------------------------------------
        for pattern, weight in _HARD_PATTERNS:
            if re.search(pattern, text):
                reasoning[f"hard:{pattern[:30]}"] = weight
                score += weight

        # --- Easy-keyword patterns (subtract) ----------------------------
        for pattern, weight in _EASY_PATTERNS:
            if re.search(pattern, text):
                reasoning[f"easy:{pattern[:30]}"] = -weight
                score -= weight

        # --- Sentence-count heuristic ------------------------------------
        sentence_count = max(1, len(re.split(r"[.!?]+", text)))
        if sentence_count >= 4:
            extra = min((sentence_count - 3) * 0.03, 0.12)
            reasoning["sentence_count"] = extra
            score += extra

        return float(np.clip(score, 0.0, 1.0)), reasoning

    # ------------------------------------------------------------------ #
    # Trained model I/O
    # ------------------------------------------------------------------ #

    def _load_trained_model(self, path: str) -> None:
        import joblib  # noqa: PLC0415

        self._trained_model = joblib.load(path)

    def save_trained_model(self, path: str) -> None:
        """Persist the trained sklearn model to *path*."""
        if self._trained_model is None:
            raise RuntimeError("No trained model to save — call fit() first.")
        import joblib  # noqa: PLC0415

        joblib.dump(self._trained_model, path)

    def fit(self, embeddings: np.ndarray, labels: np.ndarray) -> None:
        """Train a logistic-regression classifier on labelled embeddings.

        Parameters
        ----------
        embeddings:
            (N, D) float array of L2-normalised prompt embeddings.
        labels:
            (N,) int array where 1 = hard (strong model) / 0 = easy (weak).
        """
        from sklearn.linear_model import LogisticRegression  # noqa: PLC0415
        from sklearn.preprocessing import StandardScaler  # noqa: PLC0415
        from sklearn.pipeline import Pipeline  # noqa: PLC0415

        clf = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("lr", LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced")),
            ]
        )
        clf.fit(embeddings, labels)
        self._trained_model = clf
