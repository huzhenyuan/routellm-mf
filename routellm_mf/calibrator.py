"""
Calibrator — train the logistic-regression difficulty classifier on your own data.

Usage (Python)::

    from routellm_mf.calibrator import Calibrator
    from routellm_mf.config import RouterConfig

    cfg = RouterConfig.from_yaml("config.yaml")
    cal = Calibrator(cfg)

    # Option A — load a CSV with columns: "prompt", "label"  (1=hard, 0=easy)
    cal.load_csv("my_labels.csv")

    # Option B — add examples programmatically
    cal.add("What is 2+2?", label=0)
    cal.add("Prove the Riemann hypothesis step by step.", label=1)

    # Fit and save
    cal.fit()
    cal.save("classifier.joblib")

    # Evaluate (accuracy, classification report)
    cal.evaluate()

Usage (CLI)::

    routellm-calibrate --data my_labels.csv --out classifier.joblib --config config.yaml
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from routellm_mf.classifier import DifficultyClassifier
from routellm_mf.config import RouterConfig
from routellm_mf.embedding import LocalEmbedder


class Calibrator:
    """Collect labelled prompts, embed them, and train the classifier."""

    def __init__(self, config: RouterConfig) -> None:
        self._config = config
        self._embedder = LocalEmbedder(config.embedding)
        self._classifier = DifficultyClassifier(config.classifier)
        self._prompts: List[str] = []
        self._labels: List[int] = []

    # ------------------------------------------------------------------ #
    # Data loading
    # ------------------------------------------------------------------ #

    def add(self, prompt: str, label: int) -> None:
        """Add a single labelled example.

        Parameters
        ----------
        prompt: str
            The user prompt.
        label: int
            1 = hard (use strong model), 0 = easy (use weak model).
        """
        self._prompts.append(prompt)
        self._labels.append(label)

    def load_csv(self, path: str, prompt_col: str = "prompt", label_col: str = "label") -> None:
        """Load labelled examples from a CSV file.

        The CSV must have at least two columns: *prompt_col* and *label_col*.
        """
        with open(path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                self._prompts.append(row[prompt_col])
                self._labels.append(int(row[label_col]))

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #

    def fit(self) -> None:
        """Embed all prompts and train the logistic-regression classifier."""
        if not self._prompts:
            raise ValueError("No training data — call add() or load_csv() first.")
        print(f"[Calibrator] Embedding {len(self._prompts)} prompts …")
        embeddings = self._embedder.embed_batch(self._prompts)
        labels = np.array(self._labels, dtype=int)
        print("[Calibrator] Training logistic-regression classifier …")
        self._classifier.fit(embeddings, labels)
        print("[Calibrator] Training complete.")

    def save(self, path: str) -> None:
        """Save the trained classifier to *path* (.joblib)."""
        self._classifier.save_trained_model(path)
        print(f"[Calibrator] Classifier saved to {path}")

    # ------------------------------------------------------------------ #
    # Evaluation
    # ------------------------------------------------------------------ #

    def evaluate(self, test_csv: Optional[str] = None) -> dict:
        """Evaluate the classifier.

        If *test_csv* is given, evaluate on that held-out set; otherwise
        report training-set metrics (useful for sanity-checking).

        Returns a dict with accuracy, precision, recall, f1.
        """
        from sklearn.metrics import classification_report, accuracy_score  # noqa: PLC0415

        if test_csv:
            test_prompts: List[str] = []
            test_labels: List[int] = []
            with open(test_csv, newline="", encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    test_prompts.append(row["prompt"])
                    test_labels.append(int(row["label"]))
        else:
            test_prompts = self._prompts
            test_labels = self._labels

        embeddings = self._embedder.embed_batch(test_prompts)
        labels = np.array(test_labels, dtype=int)

        # Score using the classifier in pure-model mode
        cfg_backup = self._classifier._config.heuristic_weight
        self._classifier._config.heuristic_weight = 0.0
        preds = []
        for emb in embeddings:
            result = self._classifier.classify("", embedding=emb)
            preds.append(int(result.use_strong_model))
        self._classifier._config.heuristic_weight = cfg_backup

        acc = accuracy_score(labels, preds)
        report = classification_report(labels, preds, target_names=["easy", "hard"])
        print(f"\nAccuracy: {acc:.3f}\n{report}")
        return {"accuracy": acc, "report": report}

    # ------------------------------------------------------------------ #
    # Cross-validation
    # ------------------------------------------------------------------ #

    def cross_validate(self, cv: int = 5) -> Tuple[float, float]:
        """K-fold cross-validation.  Returns (mean_accuracy, std_accuracy)."""
        from sklearn.linear_model import LogisticRegression  # noqa: PLC0415
        from sklearn.preprocessing import StandardScaler  # noqa: PLC0415
        from sklearn.pipeline import Pipeline  # noqa: PLC0415
        from sklearn.model_selection import cross_val_score  # noqa: PLC0415

        if not self._prompts:
            raise ValueError("No training data.")
        embeddings = self._embedder.embed_batch(self._prompts)
        labels = np.array(self._labels, dtype=int)

        clf = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("lr", LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced")),
            ]
        )
        scores = cross_val_score(clf, embeddings, labels, cv=cv, scoring="accuracy")
        mean, std = float(scores.mean()), float(scores.std())
        print(f"[Calibrator] {cv}-fold CV accuracy: {mean:.3f} ± {std:.3f}")
        return mean, std
