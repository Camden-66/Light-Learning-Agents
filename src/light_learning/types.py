"""Public data contracts shared by room agents and evaluators."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from math import isclose, isfinite
from typing import Any, Literal, Mapping

from .config import BUDGETS, SLOT_COUNT

RECORD_SCHEMA_VERSION = 2


@dataclass(frozen=True, slots=True)
class Observation:
    """One public light observation made by an agent."""

    time_slot: int
    light_on: bool
    repeat_index: int


@dataclass(frozen=True, slots=True)
class RoomState:
    """Only the information visible to an agent before terminal scoring."""

    history: tuple[Observation, ...]
    budget: int
    remaining_observations: int

    @property
    def phase(self) -> Literal["observe", "estimate"]:
        return "observe" if self.remaining_observations > 0 else "estimate"


@dataclass(frozen=True, slots=True)
class TerminalOutcome:
    """Private-to-terminal result returned after a final estimate."""

    theta: int
    theta_hat: int
    absolute_error: int
    reward: float


@dataclass(frozen=True, slots=True)
class EpisodeDefinition:
    """Evaluator-owned hidden parameters for one reproducible episode."""

    episode_id: str
    theta: int
    episode_seed: int

    def __post_init__(self) -> None:
        if not self.episode_id:
            raise ValueError("episode_id must not be empty")
        if isinstance(self.episode_seed, bool) or not isinstance(self.episode_seed, int):
            raise TypeError("episode_seed must be an integer")
        if self.episode_seed < 0:
            raise ValueError("episode_seed must be non-negative")


@dataclass(frozen=True, slots=True)
class AgentDecision:
    """A structured agent decision plus optional diagnostic material."""

    action: int
    raw_response: str | None = None
    thinking: str | None = None
    rationale: str | None = None
    latency_seconds: float = 0.0
    fallback_used: bool = False
    protocol_failure: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class EpisodeRecord:
    """Schema persisted for every completed evaluation episode.

    The trailing provenance fields are optional so that a record can be built
    from the terminal outcome alone, but the evaluator always populates them.
    Consistency checks assume the canonical room's ``SLOT_COUNT`` reward scale.
    """

    agent_id: str
    agent_version: str
    condition: str
    budget: int
    episode_id: str
    episode_seed: int
    theta: int
    theta_hat: int
    absolute_error: int
    reward: float
    trace: tuple[Mapping[str, Any], ...]
    completed: bool
    protocol_failure: bool
    fallback_used: bool
    latency_seconds: float
    metadata: Mapping[str, Any] = field(default_factory=dict)
    profile: str | None = None
    profile_version: str | None = None
    configuration_digest: str | None = None
    training_seed: int | None = None
    run_id: str | None = None
    schema_version: int = RECORD_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.agent_id or not self.agent_version or not self.condition:
            raise ValueError("agent_id, agent_version, and condition are required")
        if self.budget not in BUDGETS:
            raise ValueError(f"budget must be one of {BUDGETS}")
        if not 0 <= self.theta_hat < SLOT_COUNT:
            raise ValueError("theta_hat must be a room slot")
        if self.absolute_error != abs(self.theta - self.theta_hat):
            raise ValueError("absolute_error does not match theta and theta_hat")
        expected_reward = -self.absolute_error / (SLOT_COUNT - 1)
        if not isclose(self.reward, expected_reward, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("reward does not match absolute_error")
        if len(self.trace) != self.budget + 1:
            raise ValueError(
                "trace must contain one entry per observation plus the terminal estimate"
            )
        if not isfinite(self.latency_seconds) or self.latency_seconds < 0:
            raise ValueError("latency_seconds must be finite and non-negative")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready representation of this record."""

        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> EpisodeRecord:
        """Rebuild a record from its JSON form, rejecting unknown fields."""

        known = {item.name for item in fields(cls)}
        unknown = set(value) - known
        if unknown:
            raise ValueError(f"unknown record fields: {sorted(unknown)!r}")
        payload = dict(value)
        if "trace" in payload:
            payload["trace"] = tuple(payload["trace"])
        return cls(**payload)


@dataclass(frozen=True, slots=True)
class ExperimentProfile:
    """Named evaluation scale used by the CLI and reports."""

    name: str
    episodes_per_cell: int
    preliminary: bool


PILOT_PROFILE = ExperimentProfile("pilot", episodes_per_cell=10, preliminary=True)
FULL_PROFILE = ExperimentProfile("full", episodes_per_cell=100, preliminary=False)
