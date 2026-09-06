"""Public data contracts shared by room agents and evaluators."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Mapping


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
    """Schema persisted for every completed evaluation episode."""

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

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready representation of this record."""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class ExperimentProfile:
    """Named evaluation scale used by the CLI and reports."""

    name: str
    episodes_per_cell: int
    preliminary: bool


PILOT_PROFILE = ExperimentProfile("pilot", episodes_per_cell=10, preliminary=True)
FULL_PROFILE = ExperimentProfile("full", episodes_per_cell=100, preliminary=False)
