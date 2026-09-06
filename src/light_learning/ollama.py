"""Small, testable client for Ollama's local native chat API."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Mapping, Protocol, Sequence

import httpx


class OllamaError(RuntimeError):
    """Raised when the local Ollama service cannot satisfy a request."""


@dataclass(frozen=True, slots=True)
class ChatResponse:
    """The response fields used by the room agent and its diagnostics."""

    content: str
    thinking: str | None
    raw: Mapping[str, Any]
    latency_seconds: float


class ChatClient(Protocol):
    """Dependency-inversion seam used by the LLM agent and unit tests."""

    def complete(
        self,
        *,
        model: str,
        messages: Sequence[Mapping[str, str]],
        json_schema: Mapping[str, Any],
        options: Mapping[str, Any],
        think: bool,
    ) -> ChatResponse: ...

    def model_metadata(self, model: str) -> Mapping[str, Any]: ...


class OllamaChatClient:
    """Synchronous client for `/api/chat`, deliberately without auto-pull."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:11434",
        *,
        timeout_seconds: float = 120.0,
        client: httpx.Client | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._owns_client = client is None
        self._client = client or httpx.Client(timeout=timeout_seconds)

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> "OllamaChatClient":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def _request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        try:
            response = self._client.request(method, f"{self.base_url}{path}", **kwargs)
            response.raise_for_status()
            payload = response.json()
        except (httpx.HTTPError, ValueError) as error:
            raise OllamaError(f"Ollama request {method} {path} failed: {error}") from error
        if not isinstance(payload, dict):
            raise OllamaError(f"Ollama request {method} {path} returned a non-object JSON body")
        return payload

    def complete(
        self,
        *,
        model: str,
        messages: Sequence[Mapping[str, str]],
        json_schema: Mapping[str, Any],
        options: Mapping[str, Any],
        think: bool,
    ) -> ChatResponse:
        """Request one non-streaming schema-constrained chat completion."""

        started = perf_counter()
        payload = self._request(
            "POST",
            "/api/chat",
            json={
                "model": model,
                "messages": [dict(message) for message in messages],
                "stream": False,
                "format": dict(json_schema),
                "think": think,
                "options": dict(options),
            },
        )
        latency = perf_counter() - started
        message = payload.get("message")
        if not isinstance(message, dict) or not isinstance(message.get("content"), str):
            raise OllamaError("Ollama chat response did not contain message.content")
        thinking = message.get("thinking")
        if thinking is not None and not isinstance(thinking, str):
            thinking = str(thinking)
        return ChatResponse(
            content=message["content"],
            thinking=thinking,
            raw=payload,
            latency_seconds=latency,
        )

    def list_models(self) -> tuple[Mapping[str, Any], ...]:
        """List locally installed models without downloading anything."""

        payload = self._request("GET", "/api/tags")
        models = payload.get("models", [])
        if not isinstance(models, list):
            raise OllamaError("Ollama tags response did not contain a models list")
        return tuple(model for model in models if isinstance(model, dict))

    def model_metadata(self, model: str) -> Mapping[str, Any]:
        """Capture the installed model's local digest and show metadata."""

        installed = next(
            (
                item
                for item in self.list_models()
                if item.get("name") == model or item.get("model") == model
            ),
            None,
        )
        if installed is None:
            raise OllamaError(f"model {model!r} is not installed locally")
        show = self._request("POST", "/api/show", json={"name": model, "verbose": True})
        return {
            "backend": "ollama-native-api",
            "base_url": self.base_url,
            "model": model,
            "digest": installed.get("digest"),
            "installed_model": dict(installed),
            "show": show,
        }
