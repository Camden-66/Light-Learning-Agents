from __future__ import annotations

import json
from typing import Any

import httpx

from light_learning.ollama import OllamaChatClient


def test_native_chat_request_uses_schema_and_thinking() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["json"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={"message": {"content": '{"kind":"observe","time_slot":2}', "thinking": "x"}},
        )

    client = OllamaChatClient(client=httpx.Client(transport=httpx.MockTransport(handler)))
    response = client.complete(
        model="qwen3:1.7b",
        messages=[{"role": "user", "content": "hello"}],
        json_schema={"type": "object"},
        options={"seed": 1},
        think=True,
    )
    assert response.thinking == "x"
    assert captured["json"]["stream"] is False
    assert captured["json"]["think"] is True
    assert captured["json"]["format"] == {"type": "object"}
