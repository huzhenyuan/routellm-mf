"""
Basic usage example — no real LLM needed for the routing/scoring part.

Run:
    python examples/basic_usage.py
"""

from routellm_mf import Router, RouterConfig

# Build a config directly in Python (no YAML file needed)
cfg = RouterConfig()
cfg.verbose = True  # Print routing decisions to stdout

# Lower the threshold so some prompts route to strong model
cfg.classifier.threshold = 0.40

router = Router(cfg)

# -----------------------------------------------------------------------
# 1.  Just score prompts (no model call)
# -----------------------------------------------------------------------
prompts = [
    "Hi!",
    "What is the capital of France?",
    "Explain the difference between TCP and UDP.",
    "Write a Python function that implements merge sort and analyse its time complexity.",
    "Prove that there are infinitely many prime numbers using Euclid's proof.",
    "Integrate x^3 * ln(x) dx using integration by parts and verify the result.",
]

print("=" * 70)
print("ROUTING DECISIONS (no model call)")
print("=" * 70)
for p in prompts:
    result = router.score(p)
    model = cfg.strong_model.model if result.use_strong_model else cfg.weak_model.model
    print(f"\nPrompt : {p[:60]!r}")
    print(f"  Score : {result.score:.3f}  →  {model}")

# -----------------------------------------------------------------------
# 2.  Calling a model (requires a running ollama / lmstudio / OpenAI)
# -----------------------------------------------------------------------
# Uncomment to actually call the model:
#
# reply, decision = router.chat(
#     "What sorting algorithm works best for nearly-sorted data?",
#     return_decision=True,
# )
# print("\n" + str(decision))
# print("\nReply:", reply)

# -----------------------------------------------------------------------
# 3.  Multi-turn conversation
# -----------------------------------------------------------------------
# messages = [
#     {"role": "system", "content": "You are a helpful assistant."},
#     {"role": "user", "content": "Prove that sqrt(2) is irrational."},
# ]
# reply, decision = router.chat_messages(messages, return_decision=True)
# print(decision)
# print(reply)
