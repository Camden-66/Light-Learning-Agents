from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from light_learning.llm_agent import LLMRoomAgent, LLMRoomAgentConfig
from light_learning.ollama import ChatResponse
from light_learning.room import RoomEnv


class FakeChatClient:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.calls: list[dict[str, Any]] = []

    def complete(
        self,
        *,
        model: str,
        messages: Sequence[Mapping[str, str]],
        json_schema: Mapping[str, Any],
        options: Mapping[str, Any],
        think: bool,
    ) -> ChatResponse:
        self.calls.append(
            {
                "model": model,
                "messages": list(messages),
                "schema": dict(json_schema),
                "options": dict(options),
                "think": think,
            }
        )
        content = self.responses.pop(0)
        return ChatResponse(
            content=content,
            thinking="considering observations",
            raw={"message": {"content": content}},
            latency_seconds=0.25,
        )

    def model_metadata(self, model: str) -> Mapping[str, Any]:
        return {"model": model, "digest": "test-digest"}


def test_qualitative_agent_only_receives_public_history() -> None:
    client = FakeChatClient(['{"kind":"observe","time_slot":7}'])
    agent = LLMRoomAgent(client, LLMRoomAgentConfig(model="qwen3:1.7b"))
    state = RoomEnv(budget=4).reset(seed=99, theta=13)
    agent.start_episode(state)
    decision = agent.act(state)

    assert decision.action == 7
    assert decision.thinking == "considering observations"
    serialized_messages = json.dumps(client.calls[0]["messages"])
    assert "theta" not in serialized_messages.lower()
    assert "exp(" not in serialized_messages
    assert client.calls[0]["think"] is True


def test_terminal_schema_is_distinct_from_observation_schema() -> None:
    client = FakeChatClient(['{"kind":"estimate","theta_hat":16}'])
    agent = LLMRoomAgent(client, LLMRoomAgentConfig(model="qwen3:1.7b"))
    env = RoomEnv(budget=4)
    state = env.reset(seed=99, theta=13)
    for _ in range(4):
        state = env.observe(10)
    decision = agent.act(state)
    assert decision.action == 16
    assert client.calls[0]["schema"]["required"] == ["kind", "theta_hat"]


def test_invalid_responses_retry_then_use_public_fallback() -> None:
    client = FakeChatClient(["not json", "[]", '{"kind":"estimate","theta_hat":1}'])
    agent = LLMRoomAgent(client, LLMRoomAgentConfig(model="qwen3:1.7b"))
    state = RoomEnv(budget=4).reset(seed=99, theta=13)
    decision = agent.act(state)
    assert decision.action == 0
    assert decision.fallback_used
    assert decision.protocol_failure
    assert decision.metadata["attempts"] == 3
    assert len(client.calls) == 3
    assert len(client.calls[1]["messages"]) == 3


def test_run_metadata_caches_model_digest() -> None:
    client = FakeChatClient([])
    agent = LLMRoomAgent(client, LLMRoomAgentConfig(model="qwen3:1.7b"))
    first = agent.run_metadata()
    second = agent.run_metadata()
    assert first is second
    assert first["backend"]["digest"] == "test-digest"
