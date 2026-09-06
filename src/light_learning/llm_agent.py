"""Schema-constrained, stateless-per-turn Ollama room agent."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal, Mapping

from .config import RoomConfig
from .ollama import ChatClient
from .room import fallback_estimate, fallback_observation_slot
from .types import AgentDecision, RoomState

PromptCondition = Literal["qualitative", "disclosed"]


@dataclass(frozen=True, slots=True)
class LLMRoomAgentConfig:
    """Pinned behavioral settings for one reproducible LLM condition."""

    model: str
    condition: PromptCondition = "qualitative"
    thinking_enabled: bool = True
    temperature: float = 0.6
    top_p: float = 0.95
    context_tokens: int = 4096
    max_output_tokens: int = 256
    seed: int = 0
    schema_retries: int = 2

    def __post_init__(self) -> None:
        if self.condition not in {"qualitative", "disclosed"}:
            raise ValueError("condition must be qualitative or disclosed")
        if not self.model:
            raise ValueError("model must be non-empty")
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError("temperature must be in [0, 2]")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")
        if self.context_tokens <= 0 or self.max_output_tokens <= 0:
            raise ValueError("token limits must be positive")
        if self.schema_retries < 0:
            raise ValueError("schema_retries may not be negative")


class LLMRoomAgent:
    """A fresh-context LLM policy over only public room history.

    Every request is stateless except for history supplied in the current
    `RoomState`. Native thinking is saved in diagnostics but never fed back to
    the model on a later turn.
    """

    agent_id = "ollama-llm-room-agent"

    def __init__(
        self,
        client: ChatClient,
        config: LLMRoomAgentConfig,
        room_config: RoomConfig | None = None,
    ) -> None:
        self.client = client
        self.config = config
        self.room_config = room_config or RoomConfig()
        self.agent_version = config.model
        self.condition = config.condition
        self._metadata_cache: Mapping[str, Any] | None = None

    def start_episode(self, state: RoomState) -> None:
        """Assert a clean initial state; no prior episode state is retained."""

        if state.history:
            raise ValueError("an LLM episode must begin with an empty history")

    @property
    def _options(self) -> Mapping[str, Any]:
        return {
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "num_ctx": self.config.context_tokens,
            "num_predict": self.config.max_output_tokens,
            "seed": self.config.seed,
        }

    def _system_prompt(self) -> str:
        common = (
            "You are an experimental agent outside a room with a light. "
            "You may inspect one time slot at a time and receive only on or off. "
            "You must use the full observation budget, then name the time slot "
            "where the light is most likely to be on. The score is unavailable "
            "until after your final answer. Return only the requested JSON object."
        )
        if self.config.condition == "qualitative":
            return (
                common
                + " The room has a stable but unknown relationship between time and "
                "the chance that the light is on. No probability formula is supplied."
            )
        return (
            common
            + " The hidden peak theta is one integer from 4 through 27. For a time "
            "slot t from 0 through 31, P(on | t, theta) = 0.05 + 0.90 * "
            "exp(-(t - theta)^2 / 18). Theta itself remains hidden."
        )

    @staticmethod
    def _history_payload(state: RoomState) -> list[dict[str, Any]]:
        return [
            {
                "time_slot": observation.time_slot,
                "light": "on" if observation.light_on else "off",
            }
            for observation in state.history
        ]

    def _messages(self, state: RoomState, repair: bool = False) -> list[dict[str, str]]:
        expected = (
            'an observation action: {"kind":"observe","time_slot":INTEGER}'
            if state.phase == "observe"
            else 'a final estimate: {"kind":"estimate","theta_hat":INTEGER}'
        )
        user = {
            "observation_history": self._history_payload(state),
            "remaining_observations": state.remaining_observations,
            "valid_time_slots": [0, self.room_config.slot_count - 1],
            "required_action": expected,
        }
        messages = [
            {"role": "system", "content": self._system_prompt()},
            {"role": "user", "content": json.dumps(user, separators=(",", ":"))},
        ]
        if repair:
            messages.append(
                {
                    "role": "user",
                    "content": "Return only one JSON object matching the required action schema.",
                }
            )
        return messages

    def _schema(self, state: RoomState) -> Mapping[str, Any]:
        if state.phase == "observe":
            return {
                "type": "object",
                "additionalProperties": False,
                "required": ["kind", "time_slot"],
                "properties": {
                    "kind": {"const": "observe"},
                    "time_slot": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": self.room_config.slot_count - 1,
                    },
                },
            }
        return {
            "type": "object",
            "additionalProperties": False,
            "required": ["kind", "theta_hat"],
            "properties": {
                "kind": {"const": "estimate"},
                "theta_hat": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": self.room_config.slot_count - 1,
                },
            },
        }

    def _parse_action(self, content: str, state: RoomState) -> int:
        try:
            payload = json.loads(content)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSON: {error.msg}") from error
        if not isinstance(payload, dict):
            raise ValueError("JSON response must be an object")
        if state.phase == "observe":
            expected_kind, action_key = "observe", "time_slot"
        else:
            expected_kind, action_key = "estimate", "theta_hat"
        if set(payload) != {"kind", action_key}:
            raise ValueError(f"response keys must be kind and {action_key}")
        if payload.get("kind") != expected_kind:
            raise ValueError(f"response kind must be {expected_kind}")
        action = payload.get(action_key)
        if isinstance(action, bool) or not isinstance(action, int):
            raise ValueError(f"{action_key} must be an integer")
        return self.room_config.validate_time_slot(action)

    def _fallback(self, state: RoomState) -> int:
        if state.phase == "observe":
            return fallback_observation_slot(state, self.room_config)
        return fallback_estimate(state, self.room_config)

    def act(self, state: RoomState) -> AgentDecision:
        """Get a valid action or finish the episode with a logged fallback."""

        errors: list[str] = []
        total_latency = 0.0
        last_content: str | None = None
        last_thinking: str | None = None
        for attempt in range(self.config.schema_retries + 1):
            try:
                response = self.client.complete(
                    model=self.config.model,
                    messages=self._messages(state, repair=attempt > 0),
                    json_schema=self._schema(state),
                    options=self._options,
                    think=self.config.thinking_enabled,
                )
                total_latency += response.latency_seconds
                last_content = response.content
                last_thinking = response.thinking
                action = self._parse_action(response.content, state)
                return AgentDecision(
                    action=action,
                    raw_response=response.content,
                    thinking=response.thinking,
                    latency_seconds=total_latency,
                    metadata={
                        "attempts": attempt + 1,
                        "model": self.config.model,
                        "thinking_enabled": self.config.thinking_enabled,
                    },
                )
            except Exception as error:
                errors.append(f"{type(error).__name__}: {error}")
        return AgentDecision(
            action=self._fallback(state),
            raw_response=last_content,
            thinking=last_thinking,
            latency_seconds=total_latency,
            fallback_used=True,
            protocol_failure=True,
            metadata={
                "attempts": self.config.schema_retries + 1,
                "errors": errors,
                "model": self.config.model,
                "thinking_enabled": self.config.thinking_enabled,
            },
        )

    def run_metadata(self) -> Mapping[str, Any]:
        """Return pinned configuration plus best-effort local model metadata."""

        if self._metadata_cache is None:
            try:
                backend_metadata = dict(self.client.model_metadata(self.config.model))
            except Exception as error:
                backend_metadata = {"model_metadata_error": f"{type(error).__name__}: {error}"}
            self._metadata_cache = {
                "agent": self.agent_id,
                "model": self.config.model,
                "condition": self.config.condition,
                "generation": dict(self._options),
                "thinking_enabled": self.config.thinking_enabled,
                "schema_retries": self.config.schema_retries,
                "backend": backend_metadata,
            }
        return self._metadata_cache
