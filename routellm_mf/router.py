"""
Core router.

The Router ties together:
- LocalEmbedder  — produce a vector for each incoming prompt
- DifficultyClassifier — score the vector and decide which model to use
- ModelClient (strong / weak) — send the prompt and return the answer

Usage::

    from routellm_mf import Router, RouterConfig

    router = Router(RouterConfig.from_yaml("config.yaml"))

    # Simple: returns a string
    reply = router.chat("Solve the integral ∫x²dx")

    # With routing metadata:
    reply, result = router.chat("Hello!", return_result=True)
    print(result)          # DifficultyResult with score & reasoning
"""

from __future__ import annotations

from typing import Iterator, List, Optional, Tuple, Union

from routellm_mf.classifier import DifficultyClassifier, DifficultyResult
from routellm_mf.config import RouterConfig
from routellm_mf.embedding import LocalEmbedder
from routellm_mf.models import Message, ModelClient


class RoutingDecision:
    """Holds the outcome of a single routing decision."""

    def __init__(
        self,
        prompt: str,
        result: DifficultyResult,
        model_used: str,
    ) -> None:
        self.prompt = prompt
        self.result = result
        self.model_used = model_used

    def __str__(self) -> str:  # pragma: no cover
        return (
            f"[Router] → {self.model_used}\n"
            f"{self.result}"
        )


class Router:
    """LLM router that directs prompts to strong or weak models based on difficulty.

    Parameters
    ----------
    config:
        A fully populated :class:`~routellm_mf.config.RouterConfig`.

    Examples
    --------
    >>> router = Router(RouterConfig())
    >>> reply = router.chat("What is 2+2?")
    """

    def __init__(self, config: RouterConfig) -> None:
        self._config = config
        self._embedder = LocalEmbedder(config.embedding)
        self._classifier = DifficultyClassifier(config.classifier)
        self._strong = ModelClient(config.strong_model)
        self._weak = ModelClient(config.weak_model)

    # ------------------------------------------------------------------ #
    # Public API — single-turn
    # ------------------------------------------------------------------ #

    def chat(
        self,
        prompt: str,
        *,
        stream: bool = False,
        return_decision: bool = False,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Union[str, Iterator[str], Tuple[str, RoutingDecision], Tuple[Iterator[str], RoutingDecision]]:
        """Route *prompt* to the appropriate model and return the reply.

        Parameters
        ----------
        prompt:
            The user's message.
        stream:
            Stream the response token-by-token.
        return_decision:
            If True, returns a ``(reply, RoutingDecision)`` tuple instead of
            just the reply, allowing callers to inspect the routing choice.
        temperature / max_tokens:
            Override the per-model defaults.

        Returns
        -------
        str | Iterator[str] | Tuple[str | Iterator[str], RoutingDecision]
        """
        decision = self._decide(prompt)

        if self._config.verbose:
            print(decision)

        client = self._strong if decision.result.use_strong_model else self._weak
        reply = client.chat_simple(
            prompt,
            stream=stream,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        if return_decision:
            return reply, decision
        return reply

    def chat_messages(
        self,
        messages: List[Message],
        *,
        stream: bool = False,
        return_decision: bool = False,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ):
        """Route a full message list (multi-turn conversation support).

        The difficulty is evaluated on the **last user message** only.
        """
        user_content = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                user_content = msg.get("content", "")
                break

        decision = self._decide(user_content)

        if self._config.verbose:
            print(decision)

        client = self._strong if decision.result.use_strong_model else self._weak
        reply = client.chat(
            messages,
            stream=stream,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        if return_decision:
            return reply, decision
        return reply

    # ------------------------------------------------------------------ #
    # Direct score access (useful for calibration / testing)
    # ------------------------------------------------------------------ #

    def score(self, prompt: str) -> DifficultyResult:
        """Return the difficulty result for *prompt* without calling any model."""
        return self._decide(prompt).result

    # ------------------------------------------------------------------ #
    # Internal
    # ------------------------------------------------------------------ #

    def _decide(self, prompt: str) -> RoutingDecision:
        embedding = self._embedder.embed(prompt)
        result = self._classifier.classify(prompt, embedding=embedding)
        model_used = (
            self._config.strong_model.model
            if result.use_strong_model
            else self._config.weak_model.model
        )
        return RoutingDecision(prompt=prompt, result=result, model_used=model_used)
