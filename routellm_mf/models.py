"""
OpenAI-compatible model client.

Works with any server that exposes the OpenAI Chat Completions API:
- Ollama   (http://localhost:11434/v1)
- LM Studio (http://localhost:1234/v1)
- OpenAI    (https://api.openai.com/v1)
- vLLM, llama-cpp-python, etc.
"""

from __future__ import annotations

from typing import Iterator, List, Optional

from openai import OpenAI
from openai.types.chat import ChatCompletion, ChatCompletionChunk

from routellm_mf.config import ModelConfig

Message = dict  # {"role": str, "content": str}


class ModelClient:
    """Thin wrapper around the OpenAI SDK for a single model endpoint."""

    def __init__(self, config: ModelConfig) -> None:
        self._config = config
        self._client = OpenAI(
            api_key=config.api_key or "local",
            base_url=config.base_url,
        )

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def chat(
        self,
        messages: List[Message],
        *,
        stream: bool = False,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str | Iterator[str]:
        """Send *messages* to the model and return the assistant reply.

        Parameters
        ----------
        messages:
            Standard OpenAI message list, e.g.
            ``[{"role": "user", "content": "Hello!"}]``
        stream:
            If True, returns a generator that yields text chunks.
        temperature / max_tokens:
            Override the per-model defaults from config.

        Returns
        -------
        str or Iterator[str]
            When ``stream=False`` the full reply string.
            When ``stream=True`` a generator of text deltas.
        """
        kwargs = dict(
            model=self._config.model,
            messages=messages,
            temperature=temperature if temperature is not None else self._config.temperature,
            max_tokens=max_tokens if max_tokens is not None else self._config.max_tokens,
        )
        if stream:
            return self._stream(**kwargs)
        response: ChatCompletion = self._client.chat.completions.create(**kwargs, stream=False)
        return response.choices[0].message.content or ""

    def chat_simple(self, prompt: str, **kwargs) -> str:
        """Convenience wrapper: send a single user message, return the reply."""
        return self.chat([{"role": "user", "content": prompt}], **kwargs)

    @property
    def model_name(self) -> str:
        return self._config.model

    # ------------------------------------------------------------------ #
    # Internal
    # ------------------------------------------------------------------ #

    def _stream(self, **kwargs) -> Iterator[str]:
        response = self._client.chat.completions.create(**kwargs, stream=True)
        for chunk in response:
            chunk: ChatCompletionChunk
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
