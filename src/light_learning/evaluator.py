"""Room benchmark evaluator.

This module owns episode definitions, the agent-driving loop, the fallback
policy, episode records, metrics, and JSONL/summary/manifest artifacts.

It owns no environment code: the canonical ``RoomConfig``, ``RoomEnv``,
deterministic light draws, public ``RoomState``, and the documented fallback
policy all come from :mod:`light_learning.room` and :mod:`light_learning.types`.
It implements neither PPO, oracle MLE, nor an Ollama client.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import statistics
import time
import uuid
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from math import isfinite
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

from .config import BUDGETS, SLOT_COUNT, RoomConfig
from .room import RoomEnv, fallback_estimate, fallback_observation_slot
from .types import (
    RECORD_SCHEMA_VERSION,
    AgentDecision,
    EpisodeDefinition,
    EpisodeRecord,
    Observation,
    RoomState,
    TerminalOutcome,
)

__all__ = [
    "EVALUATOR_VERSION",
    "RECORD_SCHEMA_VERSION",
    "AgentDecision",
    "EpisodeDefinition",
    "EpisodeRecord",
    "GymPolicyAdapter",
    "InvalidAction",
    "Observation",
    "RoomState",
    "TerminalOutcome",
    "aggregate_metrics",
    "assert_training_separation",
    "configuration_digest",
    "derive_policy_seed",
    "evaluate_profile",
    "load_episode_definitions",
    "profile_master_seed",
    "read_records",
    "run_episode",
    "run_evaluation",
    "validate_profiles",
    "write_records",
    "write_summary",
]

EVALUATOR_VERSION = "evaluator-v3"
PROFILE_MANIFEST = Path(__file__).with_name("episode_profiles.v1.json")
POLICY_SEED_NAMESPACE = "light-learning:policy-seed:v1"
PRELIMINARY_PROFILES = frozenset({"pilot"})
ARTIFACT_FLUSH_INTERVAL = 25


class InvalidAction(ValueError):
    """Raised when an agent decision does not contain one valid integer action."""


# --------------------------------------------------------------------------
# Profiles and reproducible episode definitions
# --------------------------------------------------------------------------


def configuration_digest(configuration: Mapping[str, Any]) -> str:
    """Return a stable digest of an agent's evaluated configuration."""

    encoded = json.dumps(
        configuration, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _jsonable(value: Any) -> Any:
    """Round-trip through JSON so tuples compare equal to their stored lists."""

    return json.loads(json.dumps(value, default=str))


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


def _evenly_spaced(candidates: Sequence[int], count: int) -> list[int]:
    """Pick ``count`` candidates spread across the whole range, endpoints included.

    A profile with fewer episodes than theta values must still probe the entire
    hidden range, otherwise the worst-theta slice only ever sees a lopsided
    subset of the room.
    """

    if count >= len(candidates):
        return list(candidates)
    if count == 1:
        return [candidates[0]]
    last = len(candidates) - 1
    return [candidates[round(index * last / (count - 1))] for index in range(count)]


def load_episode_definitions(
    profile: str,
    manifest_path: str | Path = PROFILE_MANIFEST,
    *,
    config: RoomConfig | None = None,
) -> tuple[str, dict[int, list[EpisodeDefinition]]]:
    """Create deterministic, theta-stratified definitions from the master seed."""

    config = config or RoomConfig()
    manifest = _manifest(manifest_path)
    if profile not in manifest["profiles"]:
        raise ValueError(f"unknown evaluation profile: {profile!r}")
    spec = manifest["profiles"][profile]
    count = int(spec["episodes_per_budget"])
    version = str(spec["version"])
    master_seed = int(manifest["master_seed"])

    pool = _evenly_spaced(config.theta_candidates, count)
    theta_order = sorted(
        pool,
        key=lambda theta: hashlib.sha256(
            f"{master_seed}|{profile}|theta|{theta}".encode()
        ).digest(),
    )
    definitions: dict[int, list[EpisodeDefinition]] = {}
    for budget in BUDGETS:
        episodes: list[EpisodeDefinition] = []
        for index in range(count):
            theta = config.validate_theta(theta_order[index % len(theta_order)])
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
    heldout_seeds = {
        episode.episode_seed for cases in definitions.values() for episode in cases
    }
    duplicate_ids = heldout_ids.intersection(training_episode_ids)
    duplicate_seeds = heldout_seeds.intersection(training_episode_seeds)
    if duplicate_ids or duplicate_seeds:
        raise ValueError(
            f"training/evaluation overlap: ids={sorted(duplicate_ids)!r}, "
            f"seeds={sorted(duplicate_seeds)!r}"
        )


# --------------------------------------------------------------------------
# Agent-policy seeding
# --------------------------------------------------------------------------


def derive_policy_seed(
    *,
    master_seed: int,
    agent_version: str,
    condition: str,
    budget: int,
    episode_id: str,
) -> int:
    """Derive one episode's agent-policy seed from public inputs only.

    Deliberately excludes the hidden ``theta`` and ``episode_seed``: a stochastic
    agent must never be able to correlate its own randomness with the room's.
    Because every input is stable per episode, a resumed run reproduces the same
    seed without replaying earlier episodes.
    """

    payload = (
        f"{POLICY_SEED_NAMESPACE}:{master_seed}:{agent_version}:"
        f"{condition}:{budget}:{episode_id}"
    ).encode("utf-8")
    return int.from_bytes(
        hashlib.blake2b(payload, digest_size=8).digest(), byteorder="big", signed=False
    )


def _accepts_policy_seed(agent_factory: Callable[..., Any]) -> bool:
    """Report whether a factory wants the evaluator-owned policy seed."""

    try:
        signature = inspect.signature(agent_factory)
    except (TypeError, ValueError):  # pragma: no cover - exotic callables
        return False
    for parameter in signature.parameters.values():
        if parameter.kind is inspect.Parameter.VAR_KEYWORD:
            return True
        if parameter.name == "policy_seed" and parameter.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ):
            return True
    return False


# --------------------------------------------------------------------------
# Agent adapters and protocol validation
# --------------------------------------------------------------------------


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


def _validate_decision(decision: AgentDecision) -> None:
    """Enforce the diagnostic contract on a returned decision."""

    for name in ("raw_response", "thinking", "rationale"):
        value = getattr(decision, name)
        if value is not None and not isinstance(value, str):
            raise InvalidAction(f"{name} must be a string or None")
    latency = decision.latency_seconds
    if isinstance(latency, bool) or not isinstance(latency, (int, float)):
        raise InvalidAction("latency_seconds must be a number")
    if not isfinite(latency) or latency < 0:
        raise InvalidAction("latency_seconds must be finite and non-negative")
    if not isinstance(decision.fallback_used, bool):
        raise InvalidAction("fallback_used must be boolean")
    if not isinstance(decision.protocol_failure, bool):
        raise InvalidAction("protocol_failure must be boolean")
    if not isinstance(decision.metadata, Mapping):
        raise InvalidAction("metadata must be a mapping")
    json.dumps(dict(decision.metadata))


class GymPolicyAdapter:
    """Adapter for a deterministic SB3-style ``predict`` policy.

    ``PPORoomAgent`` already implements the agent protocol directly; this exists
    only for bare policies that expose nothing but ``predict``.
    """

    def __init__(self, policy: Any, *, config: RoomConfig | None = None):
        self.policy = policy
        self.config = config or RoomConfig()

    def start_episode(self, state: RoomState) -> None:
        del state

    def act(self, state: RoomState) -> AgentDecision:
        from .ppo import room_state_vector

        started = time.perf_counter()
        observation = room_state_vector(state, config=self.config)
        raw = self.policy.predict(observation, deterministic=True)
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
    """Stand-in that turns agent construction failure into a scored episode."""

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


def _construct_agent(
    agent_factory: Callable[..., Any],
    *,
    policy_seed: int | None,
    agent_id: str | None,
    agent_version: str | None,
    condition: str | None,
) -> Any:
    """Build one agent, converting any failure into a scorable stand-in."""

    try:
        if policy_seed is not None and _accepts_policy_seed(agent_factory):
            return _adapt_agent(agent_factory(policy_seed=policy_seed))
        return _adapt_agent(agent_factory())
    except Exception as error:
        return _ConstructionFailureAgent(
            error,
            agent_id=agent_id,
            agent_version=agent_version,
            condition=condition,
        )


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


def _agent_metadata(agent: Any) -> dict[str, Any]:
    getter = getattr(agent, "run_metadata", None)
    if getter is None:
        raise TypeError("agent must implement run_metadata()")
    value = getter()
    if not isinstance(value, Mapping):
        raise TypeError("run_metadata() must return a mapping")
    json.dumps(dict(value), default=str)
    return dict(value)


# --------------------------------------------------------------------------
# Episode loop
# --------------------------------------------------------------------------


def _decision_trace(
    decision: AgentDecision | None,
    *,
    action: int,
    phase: str,
    latency_seconds: float,
    measured_latency_seconds: float,
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
        "measured_latency_seconds": measured_latency_seconds,
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
    policy_seed: int | None = None,
    runtime_metadata: Mapping[str, Any] | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> EpisodeRecord:
    """Run exactly ``budget`` observations and one terminal estimate."""

    config = config or RoomConfig()
    budget = config.validate_budget(budget)
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
                    raise TypeError(
                        "agent.act() must return light_learning.types.AgentDecision"
                    )
                _validate_decision(raw_decision)
                decision = raw_decision
                action = _strict_action(decision.action)
                decision_failure = decision.protocol_failure
                used_fallback = decision.fallback_used
                decision_latency = float(decision.latency_seconds)
            except Exception as error:
                action = fallback_observation_slot(state, config)
                decision_failure = True
                used_fallback = True
                decision_latency = clock() - started
                decision = AgentDecision(
                    action=action,
                    metadata={"error": f"{type(error).__name__}: {error}"},
                    latency_seconds=decision_latency,
                    fallback_used=True,
                    protocol_failure=True,
                )
        else:
            action = fallback_observation_slot(state, config)
            decision_failure = True
            used_fallback = True
            decision_latency = 0.0
        measured = clock() - started

        if decision_failure:
            protocol_failure = True
        if used_fallback:
            fallback_used = True
        latency_seconds += decision_latency
        state = env.observe(action)
        event = _decision_trace(
            decision,
            action=action,
            phase="observe",
            latency_seconds=decision_latency,
            measured_latency_seconds=measured,
            fallback_used=used_fallback,
            protocol_failure=decision_failure,
            outcome=state.history[-1],
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
                raise TypeError(
                    "agent.act() must return light_learning.types.AgentDecision"
                )
            _validate_decision(raw_decision)
            decision = raw_decision
            theta_hat = _strict_action(decision.action)
            decision_failure = decision.protocol_failure
            used_fallback = decision.fallback_used
            decision_latency = float(decision.latency_seconds)
        except Exception as error:
            theta_hat = fallback_estimate(state, config)
            decision_failure = True
            used_fallback = True
            decision_latency = clock() - started
            decision = AgentDecision(
                action=theta_hat,
                metadata={"error": f"{type(error).__name__}: {error}"},
                latency_seconds=decision_latency,
                fallback_used=True,
                protocol_failure=True,
            )
    else:
        theta_hat = fallback_estimate(state, config)
        decision_failure = True
        used_fallback = True
        decision_latency = 0.0
    measured = clock() - started

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
        measured_latency_seconds=measured,
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
    if policy_seed is not None:
        metadata["policy_seed"] = policy_seed
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
        trace=tuple(trace),
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
    agent_factory: Callable[..., Any],
    *,
    profile: str,
    definitions: Mapping[int, Sequence[EpisodeDefinition]],
    config: RoomConfig,
    master_seed: int,
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
    seeded = _accepts_policy_seed(agent_factory)
    if seeded and not (agent_version and condition):
        raise ValueError(
            "agent_version and condition must be supplied when the agent factory "
            "accepts a policy_seed, because the seed is derived from them"
        )
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
            policy_seed = (
                derive_policy_seed(
                    master_seed=master_seed,
                    agent_version=agent_version,  # type: ignore[arg-type]
                    condition=condition,  # type: ignore[arg-type]
                    budget=budget,
                    episode_id=episode.episode_id,
                )
                if seeded
                else None
            )
            agent = _construct_agent(
                agent_factory,
                policy_seed=policy_seed,
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
                policy_seed=policy_seed,
                runtime_metadata=runtime_metadata,
            )


def evaluate_profile(
    agent_factory: Callable[..., Any],
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

    config = config or RoomConfig()
    profile_version, definitions = load_episode_definitions(
        profile, manifest_path, config=config
    )
    return list(
        _iter_profile_records(
            agent_factory,
            profile=profile,
            definitions=definitions,
            config=config,
            master_seed=profile_master_seed(profile, manifest_path),
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


# --------------------------------------------------------------------------
# Artifacts
# --------------------------------------------------------------------------


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
                    raise ValueError(
                        f"invalid record at {path}:{line_number}: {error}"
                    ) from error
    return records


def _metrics(
    records: Sequence[EpisodeRecord], *, scope: str, seed: int | None
) -> dict[str, Any]:
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
    # Ascending theta order makes max's first-equal behaviour the required tie break.
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
        "worst_theta_slice_preliminary": records[0].profile in PRELIMINARY_PROFILES,
        "configuration_digests": sorted(
            {record.configuration_digest for record in records}
        ),
    }


def aggregate_metrics(records: Iterable[EpisodeRecord]) -> list[dict[str, Any]]:
    """Group records into evaluation cells and compute every required metric."""

    groups: dict[tuple[Any, ...], list[EpisodeRecord]] = defaultdict(list)
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
    for key in sorted(groups, key=lambda item: tuple(str(part) for part in item)):
        group = groups[key]
        reports.append(_metrics(group, scope="aggregate", seed=None))
        for seed in sorted(
            {record.training_seed for record in group if record.training_seed is not None}
        ):
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
    agent_factory: Callable[..., Any],
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

    config = config or RoomConfig()
    profile_version, definitions = load_episode_definitions(
        profile, manifest_path, config=config
    )
    master_seed = profile_master_seed(profile, manifest_path)
    configuration = configuration or {}
    output_root = Path(output_dir)
    run_id = run_id or (
        f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
    )
    run_root = output_root / run_id
    records_path = run_root / "records.jsonl"
    summary_path = run_root / "summary.json"
    manifest_output = run_root / "manifest.json"
    run_root.mkdir(parents=True, exist_ok=True)
    room_config = _jsonable(asdict(config))

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
            "room_config": room_config,
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
        if previous_agent.get("training_seed") != training_seed:
            raise ValueError("run manifest mismatch for training_seed")
    elif existing:
        raise ValueError("records exist without a manifest; cannot safely resume")

    manifest_data: dict[str, Any] = {
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
        "room_config": room_config,
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

    def flush_manifest(status: str, completed: int) -> None:
        manifest_data["status"] = status
        manifest_data["completed_episode_count"] = completed
        manifest_data["updated_at"] = _utc_now()
        manifest_output.write_text(
            json.dumps(manifest_data, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    flush_manifest("running", len(existing))

    existing_keys = {_record_key(record) for record in existing}
    new_records: list[EpisodeRecord] = []
    for record in _iter_profile_records(
        agent_factory,
        profile=profile,
        definitions=definitions,
        config=config,
        master_seed=master_seed,
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
        # Records are the durable artifact, so they are flushed every episode.
        # The derived summary/manifest are refreshed on an interval instead.
        write_records(records_path, [record], mode="a" if existing or new_records else "w")
        new_records.append(record)
        if len(new_records) % ARTIFACT_FLUSH_INTERVAL == 0:
            write_summary(summary_path, aggregate_metrics(existing + new_records))
            flush_manifest("running", len(existing) + len(new_records))

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
    write_summary(summary_path, aggregate_metrics(records) if records else [])
    flush_manifest("complete", len(records))
    return records


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def validate_profiles(manifest_path: str | Path = PROFILE_MANIFEST) -> dict[str, Any]:
    """Check that both profiles are deterministic, unique, and stratified."""

    manifest = _manifest(manifest_path)
    config = RoomConfig()
    result: dict[str, Any] = {"master_seed": manifest["master_seed"]}
    for profile, expected in (("pilot", 10), ("full", 100)):
        version, definitions = load_episode_definitions(
            profile, manifest_path, config=config
        )
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
            "distinct_theta_values": len({episode.theta for episode in all_episodes}),
            "theta_counts_by_budget": {
                str(budget): {
                    str(theta): sum(
                        episode.theta == theta for episode in definitions[budget]
                    )
                    for theta in config.theta_candidates
                    if any(episode.theta == theta for episode in definitions[budget])
                }
                for budget in BUDGETS
            },
        }
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Room benchmark evaluator.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser(
        "validate-profiles", help="check profile determinism and stratification"
    )
    validate.add_argument("--manifest", default=str(PROFILE_MANIFEST))
    report = subparsers.add_parser("report", help="summarize JSONL episode records")
    report.add_argument("records", nargs="+")
    report.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.command == "validate-profiles":
        print(json.dumps(validate_profiles(args.manifest), indent=2, sort_keys=True))
        return 0
    write_summary(args.output, aggregate_metrics(read_records(args.records)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
