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


def _system_prompt(condition: str) -> str:
    agent = LLMRoomAgent(
        FakeChatClient([]), LLMRoomAgentConfig(model="qwen3:1.7b", condition=condition)
    )
    return agent._system_prompt()


def test_both_conditions_state_the_answer_range() -> None:
    """Withholding the support would confound discovery with guessing outside it."""

    for condition in ("qualitative", "disclosed"):
        prompt = _system_prompt(condition)
        assert "from 4 through 27" in prompt
        assert "from 0 through 31" in prompt


def test_conditions_differ_only_by_the_disclosed_likelihood() -> None:
    """The ablation must vary exactly one thing: the likelihood."""

    import os

    qualitative = _system_prompt("qualitative")
    disclosed = _system_prompt("disclosed")
    shared = os.path.commonprefix([qualitative, disclosed])

    # Everything up to the condition-specific sentence is byte-identical.
    assert "from 4 through 27" in shared
    assert "full observation budget" in shared
    assert qualitative[len(shared):].startswith("The room has a stable but unknown")
    assert disclosed[len(shared):].startswith("Writing theta")


def test_qualitative_prompt_withholds_the_likelihood() -> None:
    prompt = _system_prompt("qualitative")
    assert "exp(" not in prompt
    assert "P(on" not in prompt
    for constant in ("0.05", "0.9", "18"):
        assert constant not in prompt
    assert "No probability formula is supplied." in prompt


def test_disclosed_prompt_tracks_the_room_configuration() -> None:
    """The formula is rendered from RoomConfig, not hardcoded."""

    from light_learning.config import RoomConfig

    room = RoomConfig(background_probability=0.1, peak_amplitude=0.5)
    agent = LLMRoomAgent(
        FakeChatClient([]),
        LLMRoomAgentConfig(model="qwen3:1.7b", condition="disclosed"),
        room,
    )
    assert "0.1 + 0.5 * exp(-(t - theta)^2 / 18)" in agent._system_prompt()


def test_run_metadata_pins_the_exact_prompt() -> None:
    """A manifest must show which prompt wording produced a run."""

    import hashlib

    agent = LLMRoomAgent(
        FakeChatClient([]), LLMRoomAgentConfig(model="qwen3:1.7b", condition="qualitative")
    )
    metadata = agent.run_metadata()
    prompt = metadata["system_prompt"]
    assert "from 4 through 27" in prompt
    assert metadata["system_prompt_sha256"] == hashlib.sha256(
        prompt.encode("utf-8")
    ).hexdigest()

    disclosed = LLMRoomAgent(
        FakeChatClient([]), LLMRoomAgentConfig(model="qwen3:1.7b", condition="disclosed")
    ).run_metadata()
    assert disclosed["system_prompt_sha256"] != metadata["system_prompt_sha256"]
