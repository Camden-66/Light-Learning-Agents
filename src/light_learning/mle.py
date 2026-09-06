"""Passive uniform-query oracle likelihood MLE (baseline spec)."""

from __future__ import annotations

import hashlib
import json
from math import log
from typing import Any, Sequence

from .config import RoomConfig
from .room import RoomEnv
from .types import AgentDecision, RoomState, TerminalOutcome


History = Sequence[tuple[int, int | bool]]


def log_likelihood(
    history: History, candidate: int, config: RoomConfig
) -> float:
    """Return the exact canonical Bernoulli log likelihood for ``candidate``."""

    total = 0.0
    for slot, y in history:
        if y not in (0, 1):
            raise ValueError("observations must be binary (0/1 or bool)")
        p = min(max(config.light_probability(slot, candidate), 1e-15), 1.0 - 1e-15)
        total += log(p) if y else log(1.0 - p)
    return total


def mle_theta_hat(history: History, config: RoomConfig | None = None) -> int:
    """Maximize the oracle likelihood, breaking exact ties toward smaller theta."""

    config = config or RoomConfig()
    best_ll = -float("inf")
    best = config.theta_min
    for candidate in config.theta_candidates:
        ll = log_likelihood(history, candidate, config)
        if ll > best_ll or (ll == best_ll and candidate < best):
            best_ll = ll
            best = candidate
    return int(best)


def likelihood_profile(
    history: History, config: RoomConfig | None = None
) -> list[dict[str, float | int | bool]]:
    config = config or RoomConfig()
    best = mle_theta_hat(history, config)
    rows: list[dict[str, float | int | bool]] = []
    for candidate in config.theta_candidates:
        rows.append(
            {
                "theta": candidate,
                "ll": log_likelihood(history, candidate, config),
                "best": candidate == best,
            }
        )
    return rows


class PassiveUniformOracleMLE:
    """Non-adaptive uniform queries, then MLE on the oracle likelihood.

    ``query_seed`` is public policy randomness and must be derived independently
    by the evaluator; it must never be the room's hidden ``episode_seed``.  The
    evaluator should construct one agent per episode with a seed derived from
    its evaluation master seed and public episode ID.  That factory pattern is
    necessary because the shared ``start_episode(RoomState)`` protocol does not
    carry an episode ID.

    Query draw ``i`` is keyed directly by ``(query_seed, i)``.  Consequently a
    retry or resumed episode produces the same schedule without depending on
    evaluation order or mutable random-generator state.
    """

    agent_id = "passive-uniform-oracle-likelihood-mle"
    agent_version = "0.2.0"
    condition = "oracle_mle"
    _QUERY_GENERATOR = "blake2b_u64_rejection_v1"
    _QUERY_NAMESPACE = "light-learning:mle-query:v1"

    def __init__(self, config: RoomConfig | None = None, *, query_seed: int) -> None:
        if isinstance(query_seed, bool) or not isinstance(query_seed, int):
            raise TypeError("query_seed must be an integer")
        self.config = config or RoomConfig()
        self.query_seed = query_seed
        self._queries: tuple[int, ...] | None = None
        self._query_index = 0

    @classmethod
    def _uniform_query(cls, query_seed: int, draw_index: int, slot_count: int) -> int:
        """Map one keyed pseudorandom draw to a slot without modulo bias."""

        sample_space = 1 << 64
        rejection_limit = sample_space - (sample_space % slot_count)
        attempt = 0
        while True:
            payload = (
                f"{cls._QUERY_NAMESPACE}:{query_seed}:{draw_index}:{attempt}"
            ).encode("utf-8")
            sample = int.from_bytes(
                hashlib.blake2b(payload, digest_size=8).digest(),
                byteorder="big",
                signed=False,
            )
            if sample < rejection_limit:
                return sample % slot_count
            attempt += 1

    def query_schedule(self, budget: int) -> tuple[int, ...]:
        """Return the reproducible IID-with-replacement schedule for ``budget``."""

        validated_budget = self.config.validate_budget(budget)
        return tuple(
            self._uniform_query(self.query_seed, draw, self.config.slot_count)
            for draw in range(validated_budget)
        )

    def start_episode(self, state: RoomState) -> None:
        queries = self.query_schedule(state.budget)
        completed_queries = state.budget - state.remaining_observations
        if not 0 <= completed_queries <= state.budget:
            raise ValueError("room state has an invalid remaining-observation count")
        if len(state.history) != completed_queries:
            raise ValueError("room state history and remaining budget are inconsistent")

        observed_slots = tuple(observation.time_slot for observation in state.history)
        if observed_slots != queries[:completed_queries]:
            raise ValueError("room history is not a prefix of this query-seed schedule")

        self._queries = queries
        self._query_index = completed_queries

    def act(self, state: RoomState) -> AgentDecision:
        if self._queries is None:
            raise RuntimeError("start_episode must be called before act")
        if state.budget != len(self._queries):
            raise ValueError("room budget changed after start_episode")

        completed_queries = state.budget - state.remaining_observations
        if completed_queries != len(state.history):
            raise ValueError("room state history and remaining budget are inconsistent")
        if completed_queries != self._query_index:
            raise RuntimeError("act received a stale or unexpected room state")

        if state.phase == "observe":
            action = self._queries[self._query_index]
            self._query_index += 1
            return AgentDecision(action=action)
        history = [(obs.time_slot, 1 if obs.light_on else 0) for obs in state.history]
        return AgentDecision(action=mle_theta_hat(history, self.config))

    def run_metadata(self) -> dict[str, Any]:
        """Return JSON-ready policy/model configuration and its stable digest."""

        metadata: dict[str, Any] = {
            "query_policy": {
                "distribution": "discrete_uniform",
                "sampling": "iid_with_replacement",
                "slot_count": self.config.slot_count,
                "query_seed": self.query_seed,
                "generator": self._QUERY_GENERATOR,
                "namespace": self._QUERY_NAMESPACE,
            },
            "estimator": {
                "name": "oracle_bernoulli_maximum_likelihood",
                "theta_candidates": list(self.config.theta_candidates),
                "tie_break": "smallest_theta",
            },
            "likelihood": {
                "background_probability": self.config.background_probability,
                "peak_amplitude": self.config.peak_amplitude,
                "exponent_denominator": self.config.exponent_denominator,
            },
        }
        canonical = json.dumps(metadata, sort_keys=True, separators=(",", ":"))
        metadata["configuration_digest"] = (
            "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        )
        return metadata


def run_mle_episode(
    env: RoomEnv,
    agent: PassiveUniformOracleMLE | None = None,
    *,
    query_seed: int | None = None,
) -> TerminalOutcome:
    """Drive a reset RoomEnv to termination with the oracle MLE policy."""

    if agent is None:
        if query_seed is None:
            raise ValueError("provide an agent or an evaluator-owned query_seed")
        agent = PassiveUniformOracleMLE(env.config, query_seed=query_seed)
    elif query_seed is not None:
        raise ValueError("provide either an agent or query_seed, not both")

    state = env.state
    agent.start_episode(state)
    while state.phase == "observe":
        decision = agent.act(state)
        state = env.observe(decision.action)
    decision = agent.act(state)
    return env.estimate(decision.action)
