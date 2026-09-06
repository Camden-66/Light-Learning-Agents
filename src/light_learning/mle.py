"""Passive uniform-query oracle likelihood MLE (baseline spec)."""

from __future__ import annotations

from math import log

import numpy as np

from .config import RoomConfig
from .room import RoomEnv
from .types import AgentDecision, RoomState, TerminalOutcome


def log_likelihood(
    history: list[tuple[int, int]], candidate: int, config: RoomConfig
) -> float:
    total = 0.0
    for slot, y in history:
        p = min(max(config.light_probability(slot, candidate), 1e-15), 1.0 - 1e-15)
        total += log(p) if y else log(1.0 - p)
    return total


def mle_theta_hat(history: list[tuple[int, int]], config: RoomConfig | None = None) -> int:
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
    history: list[tuple[int, int]], config: RoomConfig | None = None
) -> list[dict]:
    config = config or RoomConfig()
    best = mle_theta_hat(history, config)
    rows = []
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
    """Non-adaptive uniform queries, then Bernoulli MLE on the true likelihood."""

    agent_id = "passive-uniform-oracle-likelihood-mle"
    agent_version = "0.1.0"
    condition = "oracle_mle"

    def __init__(self, config: RoomConfig | None = None, *, rng: np.random.Generator | None = None) -> None:
        self.config = config or RoomConfig()
        self._rng = rng if rng is not None else np.random.default_rng()
        self._queries: list[int] = []
        self._query_index = 0

    def start_episode(self, state: RoomState) -> None:
        self._queries = [int(s) for s in self._rng.integers(0, self.config.slot_count, size=state.budget)]
        self._query_index = 0

    def act(self, state: RoomState) -> AgentDecision:
        if state.phase == "observe":
            action = self._queries[self._query_index]
            self._query_index += 1
            return AgentDecision(action=action)
        history = [(obs.time_slot, 1 if obs.light_on else 0) for obs in state.history]
        return AgentDecision(action=mle_theta_hat(history, self.config))

    def run_metadata(self) -> dict:
        return {"query": "uniform_iid_slots", "estimator": "oracle_bernoulli_mle"}


def run_mle_episode(env: RoomEnv, agent: PassiveUniformOracleMLE | None = None) -> TerminalOutcome:
    """Drive a reset RoomEnv to termination with the oracle MLE policy."""
    agent = agent or PassiveUniformOracleMLE(env.config)
    state = env.state
    agent.start_episode(state)
    while state.phase == "observe":
        decision = agent.act(state)
        state = env.observe(decision.action)
    decision = agent.act(state)
    return env.estimate(decision.action)
