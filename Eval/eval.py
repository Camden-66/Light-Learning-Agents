"""Room benchmark evaluator.

This module drives the shared ``RoomEnv`` contract, validates the shared
``AgentDecision`` protocol, persists comparable records, and computes metrics.
It intentionally does not implement PPO, MLE, or an Ollama client.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import statistics
import time
import uuid
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

try:  # Works both as ``python Eval/eval.py`` and as a package import.
    from .environment import (
        BUDGETS,
        SLOT_COUNT,
        THETA_MAX,
        THETA_MIN,
        Observation,
        RoomConfig,
        RoomEnv,
        RoomState,
        deterministic_light_outcome,
    )
except ImportError:  # pragma: no cover - exercised by the CLI invocation.
    from environment import (  # type: ignore[no-redef]
        BUDGETS,
        SLOT_COUNT,
        THETA_MAX,
        THETA_MIN,
        Observation,
        RoomConfig,
        RoomEnv,
        RoomState,
        deterministic_light_outcome,
    )


EVALUATOR_VERSION = "evaluator-v2"
RECORD_SCHEMA_VERSION = 2
PROFILE_MANIFEST = Path(__file__).with_name("episode_profiles.v1.json")


class InvalidAction(ValueError):
    """Raised when an agent decision does not contain one valid integer action."""


@dataclass(frozen=True)
class AgentDecision:
    """One agent response in the shared evaluator protocol."""

    action: int
    raw_response: str | None = None
    thinking: str | None = None
    rationale: str | None = None
    latency_seconds: float = 0.0
    fallback_used: bool = False
    protocol_failure: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.action, bool) or not isinstance(self.action, int):
            raise InvalidAction("AgentDecision.action must be an integer")
        if self.raw_response is not None and not isinstance(self.raw_response, str):
            raise ValueError("raw_response must be a string or None")
        if self.thinking is not None and not isinstance(self.thinking, str):
            raise ValueError("thinking must be a string or None")
        if self.rationale is not None and not isinstance(self.rationale, str):
            raise ValueError("rationale must be a string or None")
        if not math.isfinite(self.latency_seconds) or self.latency_seconds < 0:
            raise ValueError("latency_seconds must be a finite non-negative number")
        if not isinstance(self.fallback_used, bool):
            raise ValueError("fallback_used must be boolean")
        if not isinstance(self.protocol_failure, bool):
            raise ValueError("protocol_failure must be boolean")
        if not isinstance(self.metadata, Mapping):
            raise ValueError("metadata must be a mapping")
        json.dumps(dict(self.metadata))


@dataclass(frozen=True)
class EpisodeDefinition:
    episode_id: str
    theta: int
    episode_seed: int

    def __post_init__(self) -> None:
        if not self.episode_id:
            raise ValueError("episode_id must not be empty")
        if not THETA_MIN <= self.theta <= THETA_MAX:
            raise ValueError(f"theta must be in [{THETA_MIN}, {THETA_MAX}]")
        if self.episode_seed < 0:
            raise ValueError("episode_seed must be non-negative")


@dataclass
class EpisodeRecord:
    """The JSONL record persisted for every completed episode."""

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
    trace: list[dict[str, Any]]
    completed: bool
    protocol_failure: bool
    fallback_used: bool
    latency_seconds: float
    metadata: dict[str, Any]
    profile: str
    profile_version: str
    configuration_digest: str
    training_seed: int | None = None
    run_id: str | None = None
    schema_version: int = RECORD_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.agent_id or not self.agent_version or not self.condition:
            raise ValueError("agent_id, agent_version, and condition are required")
        if self.budget not in BUDGETS:
            raise ValueError(f"budget must be one of {BUDGETS}")
        if not THETA_MIN <= self.theta <= THETA_MAX:
            raise ValueError("theta is outside the canonical hidden range")
        if not 0 <= self.theta_hat < SLOT_COUNT:
            raise ValueError("theta_hat must be a room slot")
        if self.absolute_error != abs(self.theta - self.theta_hat):
            raise ValueError("absolute_error does not match theta and theta_hat")
        expected_reward = -self.absolute_error / (SLOT_COUNT - 1)
        if not math.isclose(self.reward, expected_reward, rel_tol=0, abs_tol=1e-12):
            raise ValueError("reward does not match absolute_error")
        if len(self.trace) != self.budget + 1:
            raise ValueError("trace must contain B observations and one terminal decision")
        if self.latency_seconds < 0 or not math.isfinite(self.latency_seconds):
            raise ValueError("latency_seconds must be finite and non-negative")
        json.dumps(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EpisodeRecord":
        return cls(**dict(value))


def configuration_digest(configuration: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        configuration, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _manifest(manifest_path: str | Path = PROFILE_MANIFEST) -> dict[str, Any]:
    value = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if value.get("schema_version") != 1:
        raise ValueError("unsupported episode profile schema")
    if tuple(value.get("budgets", ())) != BUDGETS:
        raise ValueError(f"profile budgets must be {BUDGETS}")
    if not isinstance(value.get("master_seed"), int):
        raise ValueError("profile manifest needs an integer master_seed")
    return value


def profile_master_seed(
    profile: str, manifest_path: str | Path = PROFILE_MANIFEST
) -> int:
    manifest = _manifest(manifest_path)
    if profile not in manifest["profiles"]:
        raise ValueError(f"unknown evaluation profile: {profile!r}")
    return int(manifest["master_seed"])


def load_episode_definitions(
    profile: str, manifest_path: str | Path = PROFILE_MANIFEST
) -> tuple[str, dict[int, list[EpisodeDefinition]]]:
    """Create deterministic theta-stratified definitions from the master seed."""

    manifest = _manifest(manifest_path)
    if profile not in manifest["profiles"]:
        raise ValueError(f"unknown evaluation profile: {profile!r}")
    spec = manifest["profiles"][profile]
    count = int(spec["episodes_per_budget"])
    version = str(spec["version"])
    master_seed = int(manifest["master_seed"])

    theta_order = sorted(
        range(THETA_MIN, THETA_MAX + 1),
        key=lambda theta: hashlib.sha256(
            f"{master_seed}|{profile}|theta|{theta}".encode()
        ).digest(),
    )
    definitions: dict[int, list[EpisodeDefinition]] = {}
    for budget in BUDGETS:
        episodes: list[EpisodeDefinition] = []
        for index in range(count):
            theta = theta_order[index % len(theta_order)]
            digest = hashlib.sha256(
                f"{master_seed}|{profile}|budget={budget}|episode={index}".encode()
            ).digest()
            episode_seed = int.from_bytes(digest[:8], "big")
            episode_id = f"{version}-b{budget:02d}-{index:03d}"
            episodes.append(EpisodeDefinition(episode_id, theta, episode_seed))
        definitions[budget] = episodes
    return version, definitions


def assert_training_separation(
    training_episode_ids: Iterable[str],
    training_episode_seeds: Iterable[int],
    definitions: Mapping[int, Sequence[EpisodeDefinition]],
) -> None:
    """Reject any overlap between RL training cases and held-out cases."""

    heldout_ids = {episode.episode_id for cases in definitions.values() for episode in cases}
    heldout_seeds = {episode.episode_seed for cases in definitions.values() for episode in cases}
    duplicate_ids = heldout_ids.intersection(training_episode_ids)
    duplicate_seeds = heldout_seeds.intersection(training_episode_seeds)
    if duplicate_ids or duplicate_seeds:
        raise ValueError(
            f"training/evaluation overlap: ids={sorted(duplicate_ids)!r}, "
            f"seeds={sorted(duplicate_seeds)!r}"
        )


def room_state_to_gym_observation(state: RoomState) -> tuple[float, ...]:
    """Adapt a public RoomState to the 66-value PPO observation when needed."""

    on_counts = [0] * SLOT_COUNT
    visit_counts = [0] * SLOT_COUNT
    for observation in state.history:
        visit_counts[observation.time_slot] += 1
        on_counts[observation.time_slot] += int(observation.light_on)
    denominator = state.budget
    return tuple(
        [count / denominator for count in on_counts]
        + [count / denominator for count in visit_counts]
        + [state.remaining_observations / denominator, float(state.phase == "estimate")]
    )


def _strict_action(value: Any) -> int:
    if hasattr(value, "item"):
        try:
            value = value.item()
        except (TypeError, ValueError):
            pass
    if isinstance(value, bool) or not isinstance(value, int):
        raise InvalidAction("action must be an integer")
    if not 0 <= value < SLOT_COUNT:
        raise InvalidAction(f"action {value} is outside [0, {SLOT_COUNT - 1}]")
    return value


class GymPolicyAdapter:
    """Adapter for a deterministic SB3-style ``predict`` policy."""

    def __init__(self, policy: Any):
        self.policy = policy

    def start_episode(self, state: RoomState) -> None:
        del state

    def act(self, state: RoomState) -> AgentDecision:
        started = time.perf_counter()
        raw = self.policy.predict(room_state_to_gym_observation(state), deterministic=True)
        action_value = raw[0] if isinstance(raw, tuple) and len(raw) == 2 else raw
        action = _strict_action(action_value)
        return AgentDecision(
            action=action,
            latency_seconds=time.perf_counter() - started,
            metadata={"adapter": "gym-policy", "deterministic": True},
        )

    def run_metadata(self) -> Mapping[str, Any]:
        return {"adapter": "gym-policy"}


class _ConstructionFailureAgent:
    def __init__(
        self,
        error: Exception,
        *,
        agent_id: str | None = None,
        agent_version: str | None = None,
        condition: str | None = None,
    ):
        self.error = error
        self.agent_id = agent_id or "unknown-agent"
        self.agent_version = agent_version or "unknown-version"
        self.condition = condition or "unknown-condition"

    def start_episode(self, state: RoomState) -> None:
        del state
        raise RuntimeError(
            f"agent construction failed: {type(self.error).__name__}: {self.error}"
        ) from self.error

    def act(self, state: RoomState) -> AgentDecision:
        del state
        raise RuntimeError("agent unavailable after construction failure")

    def run_metadata(self) -> Mapping[str, Any]:
        return {"construction_failure": str(self.error)}


def _adapt_agent(agent: Any) -> Any:
    if hasattr(agent, "start_episode") and hasattr(agent, "act"):
        return agent
    if hasattr(agent, "predict"):
        return GymPolicyAdapter(agent)
    raise TypeError("agent must implement start_episode/act or expose SB3 predict")


def _identity(
    agent: Any,
    *,
    agent_id: str | None,
    agent_version: str | None,
    condition: str | None,
) -> tuple[str, str, str]:
    values = (
        agent_id or getattr(agent, "agent_id", None),
        agent_version or getattr(agent, "agent_version", None),
        condition or getattr(agent, "condition", None),
    )
    if any(not isinstance(value, str) or not value for value in values):
        raise ValueError("agent_id, agent_version, and condition must be supplied")
    return values  # type: ignore[return-value]


def _fallback_observation(state: RoomState) -> int:
    visits = [0] * SLOT_COUNT
    for observation in state.history:
        visits[observation.time_slot] += 1
    return min(range(SLOT_COUNT), key=lambda slot: (visits[slot], slot))


def _fallback_estimate(state: RoomState) -> int:
    if not state.history:
        return 16
    grouped: dict[int, list[bool]] = defaultdict(list)
    for observation in state.history:
        grouped[observation.time_slot].append(observation.light_on)
    return max(
        sorted(grouped),
        key=lambda slot: sum(grouped[slot]) / len(grouped[slot]),
    )


def _decision_trace(
    decision: AgentDecision | None,
    *,
    action: int,
    phase: str,
    latency_seconds: float,
    fallback_used: bool,
    protocol_failure: bool,
    outcome: Observation | None = None,
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "phase": phase,
        "action": action,
        "raw_response": decision.raw_response if decision else None,
        "thinking": decision.thinking if decision else None,
        "rationale": decision.rationale if decision else None,
        "decision_latency_seconds": latency_seconds,
        "fallback_used": fallback_used,
        "protocol_failure": protocol_failure,
        "metadata": dict(decision.metadata) if decision else {},
    }
    if outcome is not None:
        value.update(
            {
                "time_slot": outcome.time_slot,
                "light_on": outcome.light_on,
                "repeat_index": outcome.repeat_index,
            }
        )
    return value


def _agent_metadata(agent: Any) -> dict[str, Any]:
    getter = getattr(agent, "run_metadata", None)
    if getter is None:
        raise TypeError("agent must implement run_metadata()")
    value = getter()
    if not isinstance(value, Mapping):
        raise TypeError("run_metadata() must return a mapping")
    json.dumps(dict(value), default=str)
    return dict(value)


def run_episode(
    agent: Any,
    episode: EpisodeDefinition,
    budget: int,
    *,
    config: RoomConfig | None = None,
    agent_id: str | None = None,
    agent_version: str | None = None,
    condition: str | None = None,
    profile: str = "full",
    profile_version: str = "room-v1-full-v1",
    configuration: Mapping[str, Any] | None = None,
    training_seed: int | None = None,
    run_id: str | None = None,
    runtime_metadata: Mapping[str, Any] | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> EpisodeRecord:
    """Run exactly B observations and one terminal estimate."""

    if budget not in BUDGETS:
        raise ValueError(f"budget must be one of {BUDGETS}")
    config = config or RoomConfig()
    configuration = configuration or {}
    try:
        agent = _adapt_agent(agent)
    except Exception as error:
        agent = _ConstructionFailureAgent(
            error,
            agent_id=agent_id,
            agent_version=agent_version,
            condition=condition,
        )
    resolved_id, resolved_version, resolved_condition = _identity(
        agent,
        agent_id=agent_id,
        agent_version=agent_version,
        condition=condition,
    )
    env = RoomEnv(config, budget=budget)
    state = env.reset(seed=episode.episode_seed, theta=episode.theta)
    protocol_failure = False
    fallback_used = False
    trace: list[dict[str, Any]] = []
    latency_seconds = 0.0
    agent_available = True
    start_error: str | None = None

    try:
        agent.start_episode(state)
    except Exception as error:
        protocol_failure = True
        agent_available = False
        start_error = f"{type(error).__name__}: {error}"
        fallback_used = True

    for _ in range(budget):
        decision: AgentDecision | None = None
        decision_failure = False
        used_fallback = False
        started = clock()
        if agent_available:
            try:
                raw_decision = agent.act(state)
                if not isinstance(raw_decision, AgentDecision):
                    raise TypeError("agent.act() must return AgentDecision")
                decision = raw_decision
                action = _strict_action(decision.action)
                decision_failure = decision.protocol_failure
                used_fallback = decision.fallback_used
                decision_latency = decision.latency_seconds
            except Exception as error:
                action = _fallback_observation(state)
                decision_failure = True
                used_fallback = True
                protocol_failure = True
                decision_latency = clock() - started
                decision = AgentDecision(
                    action=action,
                    raw_response=None,
                    metadata={"error": f"{type(error).__name__}: {error}"},
                    latency_seconds=decision_latency,
                    fallback_used=True,
                    protocol_failure=True,
                )
        else:
            action = _fallback_observation(state)
            decision_failure = True
            used_fallback = True
            decision_latency = 0.0

        if decision_failure:
            protocol_failure = True
        if used_fallback:
            fallback_used = True
        latency_seconds += decision_latency
        state = env.observe(action)
        observation = state.history[-1]
        event = _decision_trace(
            decision,
            action=action,
            phase="observe",
            latency_seconds=decision_latency,
            fallback_used=used_fallback,
            protocol_failure=decision_failure,
            outcome=observation,
        )
        event["step"] = len(trace)
        trace.append(event)

    decision = None
    decision_failure = False
    used_fallback = False
    started = clock()
    if agent_available:
        try:
            raw_decision = agent.act(state)
            if not isinstance(raw_decision, AgentDecision):
                raise TypeError("agent.act() must return AgentDecision")
            decision = raw_decision
            theta_hat = _strict_action(decision.action)
            decision_failure = decision.protocol_failure
            used_fallback = decision.fallback_used
            decision_latency = decision.latency_seconds
        except Exception as error:
            theta_hat = _fallback_estimate(state)
            decision_failure = True
            used_fallback = True
            protocol_failure = True
            decision_latency = clock() - started
            decision = AgentDecision(
                action=theta_hat,
                metadata={"error": f"{type(error).__name__}: {error}"},
                latency_seconds=decision_latency,
                fallback_used=True,
                protocol_failure=True,
            )
    else:
        theta_hat = _fallback_estimate(state)
        decision_failure = True
        used_fallback = True
        decision_latency = 0.0

    if decision_failure:
        protocol_failure = True
    if used_fallback:
        fallback_used = True
    latency_seconds += decision_latency
    terminal = env.estimate(theta_hat)
    terminal_event = _decision_trace(
        decision,
        action=theta_hat,
        phase="estimate",
        latency_seconds=decision_latency,
        fallback_used=used_fallback,
        protocol_failure=decision_failure,
    )
    terminal_event.update(
        {
            "step": budget,
            "theta": terminal.theta,
            "theta_hat": terminal.theta_hat,
            "absolute_error": terminal.absolute_error,
            "reward": terminal.reward,
        }
    )
    trace.append(terminal_event)

    try:
        metadata = _agent_metadata(agent)
    except Exception as error:
        protocol_failure = True
        metadata = {"run_metadata_error": f"{type(error).__name__}: {error}"}
    if start_error:
        metadata["start_episode_error"] = start_error
    if runtime_metadata:
        metadata.update(dict(runtime_metadata))
    return EpisodeRecord(
        agent_id=resolved_id,
        agent_version=resolved_version,
        condition=resolved_condition,
        budget=budget,
        episode_id=episode.episode_id,
        episode_seed=episode.episode_seed,
        theta=terminal.theta,
        theta_hat=terminal.theta_hat,
        absolute_error=terminal.absolute_error,
        reward=terminal.reward,
        trace=trace,
        completed=True,
        protocol_failure=protocol_failure,
        fallback_used=fallback_used,
        latency_seconds=latency_seconds,
        metadata=metadata,
        profile=profile,
        profile_version=profile_version,
        configuration_digest=configuration_digest(configuration),
        training_seed=training_seed,
        run_id=run_id,
    )


def _iter_profile_records(
    agent_factory: Callable[[], Any],
    *,
    profile: str,
    definitions: Mapping[int, Sequence[EpisodeDefinition]],
    config: RoomConfig,
    agent_id: str | None,
    agent_version: str | None,
    condition: str | None,
    configuration: Mapping[str, Any],
    training_seed: int | None,
    run_id: str | None,
    runtime_metadata: Mapping[str, Any] | None,
    skip_keys: set[tuple[str, int, str, str, str]],
    profile_version: str,
) -> Iterator[EpisodeRecord]:
    for budget in BUDGETS:
        for episode in definitions[budget]:
            if agent_id and agent_version and condition:
                expected_key = (
                    agent_id,
                    budget,
                    episode.episode_id,
                    agent_version,
                    condition,
                )
                if expected_key in skip_keys:
                    continue
            try:
                agent = _adapt_agent(agent_factory())
            except Exception as error:
                agent = _ConstructionFailureAgent(
                    error,
                    agent_id=agent_id,
                    agent_version=agent_version,
                    condition=condition,
                )
            resolved_id, resolved_version, resolved_condition = _identity(
                agent,
                agent_id=agent_id,
                agent_version=agent_version,
                condition=condition,
            )
            key = (
                resolved_id,
                budget,
                episode.episode_id,
                resolved_version,
                resolved_condition,
            )
            if key in skip_keys:
                continue
            yield run_episode(
                agent,
                episode,
                budget,
                config=config,
                agent_id=resolved_id,
                agent_version=resolved_version,
                condition=resolved_condition,
                profile=profile,
                profile_version=profile_version,
                configuration=configuration,
                training_seed=training_seed,
                run_id=run_id,
                runtime_metadata=runtime_metadata,
            )


def evaluate_profile(
    agent_factory: Callable[[], Any],
    *,
    profile: str = "full",
    config: RoomConfig | None = None,
    agent_id: str | None = None,
    agent_version: str | None = None,
    condition: str | None = None,
    configuration: Mapping[str, Any] | None = None,
    training_seed: int | None = None,
    run_id: str | None = None,
    runtime_metadata: Mapping[str, Any] | None = None,
    manifest_path: str | Path = PROFILE_MANIFEST,
) -> list[EpisodeRecord]:
    """Evaluate all four budgets using one fixed versioned profile."""

    profile_version, definitions = load_episode_definitions(profile, manifest_path)
    records = list(
        _iter_profile_records(
            agent_factory,
            profile=profile,
            definitions=definitions,
            config=config or RoomConfig(),
            agent_id=agent_id,
            agent_version=agent_version,
            condition=condition,
            configuration=configuration or {},
            training_seed=training_seed,
            run_id=run_id,
            runtime_metadata=runtime_metadata,
            skip_keys=set(),
            profile_version=profile_version,
        )
    )
    return records


def _record_key(record: EpisodeRecord) -> tuple[str, int, str, str, str]:
    return (
        record.agent_id,
        record.budget,
        record.episode_id,
        record.agent_version,
        record.condition,
    )


def write_records(
    path: str | Path, records: Iterable[EpisodeRecord], *, mode: str = "w"
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open(mode, encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict(), sort_keys=True) + "\n")
            handle.flush()


def read_records(paths: Iterable[str | Path]) -> list[EpisodeRecord]:
    records: list[EpisodeRecord] = []
    for path in paths:
        with Path(path).open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    records.append(EpisodeRecord.from_dict(json.loads(line)))
                except Exception as error:
                    raise ValueError(f"invalid record at {path}:{line_number}: {error}") from error
    return records


def _metrics(records: Sequence[EpisodeRecord], *, scope: str, seed: int | None) -> dict[str, Any]:
    if not records:
        raise ValueError("cannot compute metrics for an empty cell")
    errors = [record.absolute_error for record in records]
    by_theta: dict[int, list[int]] = defaultdict(list)
    for record in records:
        by_theta[record.theta].append(record.absolute_error)
    slices = [
        (theta, statistics.fmean(values), len(values))
        for theta, values in sorted(by_theta.items())
    ]
    # Sorted theta order makes max's first-equal behavior the required tie break.
    worst_theta, worst_mae, worst_count = max(slices, key=lambda item: item[1])
    return {
        "agent_id": records[0].agent_id,
        "agent_version": records[0].agent_version,
        "condition": records[0].condition,
        "profile": records[0].profile,
        "profile_version": records[0].profile_version,
        "budget": records[0].budget,
        "scope": scope,
        "training_seed": seed,
        "episode_count": len(records),
        "mae": statistics.fmean(errors),
        "mean_absolute_error": statistics.fmean(errors),
        "median_absolute_error": statistics.median(errors),
        "hit_within_one_slot_rate": sum(error <= 1 for error in errors) / len(errors),
        "protocol_failure_rate": sum(record.protocol_failure for record in records)
        / len(records),
        "fallback_rate": sum(record.fallback_used for record in records) / len(records),
        "mean_latency_seconds": statistics.fmean(
            record.latency_seconds for record in records
        ),
        "worst_theta_slice_mae": worst_mae,
        "worst_theta": worst_theta,
        "worst_theta_group_size": worst_count,
        "worst_theta_slice_preliminary": records[0].profile == "pilot",
        "configuration_digests": sorted(
            {record.configuration_digest for record in records}
        ),
    }


def aggregate_metrics(records: Iterable[EpisodeRecord]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str, str, int], list[EpisodeRecord]] = defaultdict(list)
    for record in records:
        groups[
            (
                record.agent_id,
                record.agent_version,
                record.condition,
                record.profile,
                record.profile_version,
                record.budget,
            )
        ].append(record)
    reports: list[dict[str, Any]] = []
    for key in sorted(groups):
        group = groups[key]
        reports.append(_metrics(group, scope="aggregate", seed=None))
        for seed in sorted({record.training_seed for record in group if record.training_seed is not None}):
            reports.append(
                _metrics(
                    [record for record in group if record.training_seed == seed],
                    scope="training_seed",
                    seed=seed,
                )
            )
    return reports


def write_summary(path: str | Path, reports: Iterable[Mapping[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(list(reports), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


# Backwards-compatible name for callers that used the earlier evaluator API.
write_report = write_summary


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_evaluation(
    agent_factory: Callable[[], Any],
    *,
    output_dir: str | Path,
    run_id: str | None = None,
    resume: bool = True,
    profile: str = "full",
    config: RoomConfig | None = None,
    agent_id: str | None = None,
    agent_version: str | None = None,
    condition: str | None = None,
    configuration: Mapping[str, Any] | None = None,
    training_seed: int | None = None,
    runtime_metadata: Mapping[str, Any] | None = None,
    manifest_path: str | Path = PROFILE_MANIFEST,
) -> list[EpisodeRecord]:
    """Run, persist, summarize, and optionally resume one evaluation run."""

    profile_version, definitions = load_episode_definitions(profile, manifest_path)
    master_seed = profile_master_seed(profile, manifest_path)
    config = config or RoomConfig()
    configuration = configuration or {}
    output_root = Path(output_dir)
    run_id = run_id or f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
    run_root = output_root / run_id
    records_path = run_root / "records.jsonl"
    summary_path = run_root / "summary.json"
    manifest_output = run_root / "manifest.json"
    run_root.mkdir(parents=True, exist_ok=True)

    existing: list[EpisodeRecord] = []
    if resume and records_path.exists():
        existing = read_records([records_path])
        existing_keys = [_record_key(record) for record in existing]
        if len(existing_keys) != len(set(existing_keys)):
            raise ValueError("records contain duplicate episodes; cannot resume safely")
    if manifest_output.exists():
        previous = json.loads(manifest_output.read_text(encoding="utf-8"))
        expected = {
            "run_id": run_id,
            "profile": profile,
            "profile_version": profile_version,
            "master_seed": master_seed,
            "room_config": asdict(config),
        }
        for field_name, expected_value in expected.items():
            if previous.get(field_name) != expected_value:
                raise ValueError(f"run manifest mismatch for {field_name}")
        previous_agent = previous.get("agent", {})
        agent_id = agent_id or previous_agent.get("agent_id")
        agent_version = agent_version or previous_agent.get("agent_version")
        condition = condition or previous_agent.get("condition")
        previous_digest = previous_agent.get("configuration_digest")
        if previous_digest and previous_digest != configuration_digest(configuration):
            raise ValueError("run manifest mismatch for agent configuration")
        previous_training_seed = previous_agent.get("training_seed")
        if previous_training_seed != training_seed:
            raise ValueError("run manifest mismatch for training_seed")
    elif existing:
        raise ValueError("records exist without a manifest; cannot safely resume")

    manifest_data = {
        "evaluator_version": EVALUATOR_VERSION,
        "schema_version": RECORD_SCHEMA_VERSION,
        "run_id": run_id,
        "status": "running",
        "created_at": _utc_now(),
        "updated_at": _utc_now(),
        "profile": profile,
        "profile_version": profile_version,
        "profile_status": _manifest(manifest_path)["profiles"][profile].get("status"),
        "master_seed": master_seed,
        "room_config": asdict(config),
        "expected_episode_count": sum(len(cases) for cases in definitions.values()),
        "completed_episode_count": len(existing),
        "agent": {
            "agent_id": agent_id,
            "agent_version": agent_version,
            "condition": condition,
            "configuration_digest": configuration_digest(configuration),
            "runtime_metadata": dict(runtime_metadata or {}),
            "training_seed": training_seed,
        },
    }
    manifest_output.write_text(
        json.dumps(manifest_data, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    existing_keys = {_record_key(record) for record in existing}
    new_records: list[EpisodeRecord] = []
    for record in _iter_profile_records(
        agent_factory,
        profile=profile,
        definitions=definitions,
        config=config,
        agent_id=agent_id,
        agent_version=agent_version,
        condition=condition,
        configuration=configuration,
        training_seed=training_seed,
        run_id=run_id,
        runtime_metadata=runtime_metadata,
        skip_keys=existing_keys,
        profile_version=profile_version,
    ):
        write_records(records_path, [record], mode="a" if existing or new_records else "w")
        new_records.append(record)
        current_records = existing + new_records
        write_summary(summary_path, aggregate_metrics(current_records))
        manifest_data["completed_episode_count"] = len(current_records)
        manifest_data["updated_at"] = _utc_now()
        manifest_output.write_text(
            json.dumps(manifest_data, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    records = existing + new_records
    if records:
        manifest_data["agent"].update(
            {
                "agent_id": records[0].agent_id,
                "agent_version": records[0].agent_version,
                "condition": records[0].condition,
                "metadata": records[0].metadata,
            }
        )
    write_summary(summary_path, aggregate_metrics(records))
    manifest_data["status"] = "complete"
    manifest_data["completed_episode_count"] = len(records)
    manifest_data["updated_at"] = _utc_now()
    manifest_output.write_text(
        json.dumps(manifest_data, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return records


def _validate_profiles(manifest_path: str | Path = PROFILE_MANIFEST) -> dict[str, Any]:
    manifest = _manifest(manifest_path)
    result: dict[str, Any] = {"master_seed": manifest["master_seed"]}
    for profile, expected in (("pilot", 10), ("full", 100)):
        version, definitions = load_episode_definitions(profile, manifest_path)
        all_episodes = [episode for cases in definitions.values() for episode in cases]
        if any(len(cases) != expected for cases in definitions.values()):
            raise ValueError(f"{profile} has the wrong episodes-per-budget count")
        if len({episode.episode_id for episode in all_episodes}) != len(all_episodes):
            raise ValueError(f"{profile} episode IDs are not unique")
        if len({episode.episode_seed for episode in all_episodes}) != len(all_episodes):
            raise ValueError(f"{profile} episode seeds are not unique")
        result[profile] = {
            "version": version,
            "status": manifest["profiles"][profile].get("status"),
            "episodes_per_budget": expected,
            "total_episodes": len(all_episodes),
            "theta_counts_by_budget": {
                str(budget): {
                    str(theta): sum(episode.theta == theta for episode in definitions[budget])
                    for theta in range(THETA_MIN, THETA_MAX + 1)
                }
                for budget in BUDGETS
            },
        }
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate-profiles")
    validate.add_argument("--manifest", default=str(PROFILE_MANIFEST))
    report = subparsers.add_parser("report")
    report.add_argument("records", nargs="+")
    report.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.command == "validate-profiles":
        print(json.dumps(_validate_profiles(args.manifest), indent=2, sort_keys=True))
        return 0
    write_summary(args.output, aggregate_metrics(read_records(args.records)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
