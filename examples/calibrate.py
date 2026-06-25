"""
Calibration example — train the classifier on labelled prompt data.

Run:
    python examples/calibrate.py

This script:
1. Creates a small synthetic labelled dataset (prompt, label).
2. Trains the logistic-regression classifier.
3. Saves it to /tmp/classifier.joblib.
4. Evaluates training-set accuracy.
5. Shows how to use the saved model in the router.
"""

import csv
import os
import tempfile

from routellm_mf import Router, RouterConfig
from routellm_mf.calibrator import Calibrator

# -----------------------------------------------------------------------
# 1.  Create synthetic training data
# -----------------------------------------------------------------------
EASY_PROMPTS = [
    ("Hi, how are you?", 0),
    ("What is the capital of Japan?", 0),
    ("Who wrote Romeo and Juliet?", 0),
    ("What does HTTP stand for?", 0),
    ("Translate 'hello' to Spanish.", 0),
    ("How many days are in a week?", 0),
    ("What color is the sky?", 0),
    ("What is 15 + 27?", 0),
    ("Define the word 'eloquent'.", 0),
    ("When was the Eiffel Tower built?", 0),
]

HARD_PROMPTS = [
    ("Prove that the square root of 2 is irrational using a proof by contradiction.", 1),
    ("Implement a red-black tree in Python and explain the insertion algorithm.", 1),
    ("Explain the differences between transformer and RNN architectures for NLP. Compare their pros and cons for long sequences.", 1),
    ("Derive the time complexity of Dijkstra's algorithm with a Fibonacci heap.", 1),
    ("Write a SQL query to find the second highest salary in each department, handling ties correctly.", 1),
    ("Explain quantum entanglement and its implications for faster-than-light communication.", 1),
    ("How does backpropagation work through an LSTM cell? Derive the gradient equations.", 1),
    ("Compare dynamic programming and memoization. When is each preferred?", 1),
    ("Prove the Cauchy-Schwarz inequality and give three applications in machine learning.", 1),
    ("Design a distributed key-value store with eventual consistency. Discuss CAP theorem trade-offs.", 1),
]

ALL_DATA = EASY_PROMPTS + HARD_PROMPTS

# -----------------------------------------------------------------------
# 2.  Train the classifier
# -----------------------------------------------------------------------
cfg = RouterConfig()
cal = Calibrator(cfg)

for prompt, label in ALL_DATA:
    cal.add(prompt, label)

print(f"Training on {len(ALL_DATA)} examples …")
cal.fit()

# -----------------------------------------------------------------------
# 3.  Cross-validate (k=5)
# -----------------------------------------------------------------------
cal.cross_validate(cv=5)

# -----------------------------------------------------------------------
# 4.  Save the classifier
# -----------------------------------------------------------------------
clf_path = os.path.join(tempfile.gettempdir(), "routellm_classifier.joblib")
cal.save(clf_path)

# -----------------------------------------------------------------------
# 5.  Evaluate
# -----------------------------------------------------------------------
print("\nTraining-set evaluation:")
cal.evaluate()

# -----------------------------------------------------------------------
# 6.  Use the saved classifier in the router
# -----------------------------------------------------------------------
cfg2 = RouterConfig()
cfg2.classifier.model_path = clf_path
cfg2.classifier.heuristic_weight = 0.3   # 70% ML, 30% heuristic
cfg2.classifier.threshold = 0.50
cfg2.verbose = True

router = Router(cfg2)

test_prompts = [
    "What is the speed of light?",
    "Implement a lock-free concurrent queue in C++ and analyse its ABA problem.",
]
print("\n" + "=" * 60)
print("Test routing with trained classifier:")
for p in test_prompts:
    result = router.score(p)
    print(f"\n  {p[:70]!r}")
    print(f"  Score={result.score:.3f}  → {'STRONG' if result.use_strong_model else 'weak'}")
