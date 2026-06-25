# routellm-mf

A **local-embedding-based LLM prompt difficulty router** — a controllable, explainable alternative to [RouteLLM MF](https://github.com/lm-sys/RouteLLM).

## What it does

Every incoming prompt is automatically scored for difficulty using a local embedding model.  
Simple prompts go to your **weak (fast/cheap) model**; complex prompts go to your **strong (capable) model**.  
No OpenAI dependency at any stage — works entirely offline with [Ollama](https://ollama.com/), [LM Studio](https://lmstudio.ai/), or any OpenAI-compatible server.

```
User prompt
    │
    ▼
LocalEmbedder (sentence-transformers, runs on CPU/GPU)
    │
    ▼
DifficultyClassifier ──► score in [0, 1]
    │                         │
    │            score < threshold       score ≥ threshold
    │                 │                        │
    ▼                 ▼                        ▼
 Router         Weak model               Strong model
              (llama3:8b, …)         (llama3:70b, GPT-4, …)
```

---

## Features

| Feature | Detail |
|---|---|
| **No OpenAI for routing** | Uses `sentence-transformers` locally |
| **Zero-config heuristic** | Works out of the box, no training data needed |
| **Trainable classifier** | Optionally train a logistic-regression model on your own labelled prompts |
| **Explainable decisions** | Every routing decision comes with a per-signal score breakdown |
| **Any local model** | Ollama, LM Studio, vLLM, llama-cpp-python, or real OpenAI |
| **Blended scoring** | Configurable mix of heuristic + ML model weights |
| **CLI tools** | `routellm-route` and `routellm-calibrate` |

---

## Installation

```bash
pip install routellm-mf
# or, from source:
git clone https://github.com/huzhenyuan/routellm-mf
cd routellm-mf
pip install -e .
```

> **Embedding model** — `all-MiniLM-L6-v2` (~22 MB) is downloaded from Hugging Face on first use.  
> Swap to a multilingual model (`paraphrase-multilingual-MiniLM-L12-v2`) or a higher-quality one (`all-mpnet-base-v2`) by changing `embedding.model_name` in your config.

---

## Quick start

### 1. Copy and edit the config

```bash
cp config.example.yaml config.yaml
# Edit strong_model and weak_model to match your setup
```

### 2. Start your local models

```bash
# Ollama example
ollama run llama3:70b
ollama run llama3:8b
```

### 3. Use the router

```python
from routellm_mf import Router, RouterConfig

router = Router(RouterConfig.from_yaml("config.yaml"))

# Simple string reply
reply = router.chat("What is the capital of France?")

# With routing explanation
reply, decision = router.chat(
    "Prove that √2 is irrational using contradiction.",
    return_decision=True,
)
print(decision)
# Routing decision: STRONG
#   Score     : 0.520  (threshold=0.5)
#   Signal breakdown:
#     hard:\b(prove|proof|…)               : +0.200
#     hard:\b(step[- ]by[- ]step|…)        : +0.100
#     length                               : +0.120
#     …
```

### 4. Score without calling any model

```python
result = router.score("Integrate x³ ln(x) dx using integration by parts.")
print(result.score)           # e.g. 0.62
print(result.use_strong_model)  # True
print(result.reasoning)       # dict of signal → contribution
```

### 5. CLI scoring

```bash
routellm-route "Explain quantum entanglement."
routellm-route --config config.yaml --threshold 0.4 "Hello!"
echo "Prove Fermat's last theorem." | routellm-route
```

---

## Configuration

```yaml
# config.yaml

strong_model:
  model: "llama3:70b"
  base_url: "http://localhost:11434/v1"   # Ollama
  api_key: "local"
  temperature: 0.7
  max_tokens: 2048

weak_model:
  model: "llama3:8b"
  base_url: "http://localhost:11434/v1"
  api_key: "local"

embedding:
  model_name: "all-MiniLM-L6-v2"   # any sentence-transformers model
  device: "cpu"                      # "cuda" or "mps" for GPU
  max_chars: 2000

classifier:
  threshold: 0.50          # score ≥ threshold → strong model
  heuristic_weight: 1.0    # 1.0 = pure heuristic; 0.0 = pure ML model
  # model_path: "classifier.joblib"   # optional trained model

verbose: false
```

### Using OpenAI (or any other cloud)

```yaml
strong_model:
  model: "gpt-4o"
  base_url: null           # uses OpenAI's default
  api_key: "sk-…"

weak_model:
  model: "gpt-4o-mini"
  api_key: "sk-…"
```

---

## Training your own classifier

The built-in heuristic works well out of the box. For higher accuracy on your
specific workload, train a logistic-regression classifier on your own labelled
prompts.

### Label format (CSV)

```csv
prompt,label
"What is the capital of Japan?",0
"Prove that there are infinitely many primes.",1
```

`label = 0` → easy (weak model), `label = 1` → hard (strong model).

### Train with Python

```python
from routellm_mf import RouterConfig
from routellm_mf.calibrator import Calibrator

cfg = RouterConfig.from_yaml("config.yaml")
cal = Calibrator(cfg)
cal.load_csv("my_labels.csv")
cal.cross_validate(cv=5)        # optional: check generalisation
cal.fit()
cal.save("classifier.joblib")
cal.evaluate()                  # print accuracy / classification report
```

### Train with the CLI

```bash
routellm-calibrate \
  --data my_labels.csv \
  --out  classifier.joblib \
  --config config.yaml \
  --cv 5
```

Then update your config:

```yaml
classifier:
  model_path: "classifier.joblib"
  heuristic_weight: 0.3    # 70% ML model, 30% heuristic
  threshold: 0.50
```

---

## Architecture

```
routellm_mf/
├── __init__.py        — public API (Router, RouterConfig)
├── config.py          — dataclasses: RouterConfig, ModelConfig, EmbeddingConfig, ClassifierConfig
├── embedding.py       — LocalEmbedder wrapping sentence-transformers
├── classifier.py      — DifficultyClassifier (heuristic + optional logistic regression)
├── router.py          — Router: ties embedder + classifier + model clients together
├── models.py          — ModelClient: thin OpenAI-SDK wrapper (ollama / lmstudio / OpenAI)
├── calibrator.py      — Calibrator: collect labels, train, evaluate, cross-validate
└── cli.py             — CLI entry points (routellm-route, routellm-calibrate)
```

### Difficulty scoring signals (heuristic)

| Signal | Examples | Weight |
|---|---|---|
| Prompt length | >100 words | up to +0.30 |
| Proof / derivation | "prove", "derive", "theorem" | +0.20 |
| Math / science | "integral", "eigen", "quantum" | +0.15–0.18 |
| Code / algorithms | "implement", "complexity", "SQL" | +0.12–0.15 |
| Reasoning steps | "step-by-step", "walk me through" | +0.10 |
| Multiple questions | two or more `?` | +0.08 |
| Simple greetings | "hi", "hello" | −0.40 |
| Factual lookups | "what is", "capital of" | −0.10 |

---

## Development

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

---

## License

MIT
