"""Deterministic canonical room environment independent of any agent runtime."""

from __future__ import annotations

import hashlib
from collections import Counter

from .config import RoomConfig
from .types import Observation, RoomState, TerminalOutcome


class RoomPhaseError(RuntimeError):
    """Raised when an observation or estimate occurs in the wrong phase."""


class RoomNotResetError(RuntimeError):
    """Raised when an episode method is called before reset."""


class RoomEnv:
    """A sequential room task with order-independent seeded outcomes.

    The evaluator owns hidden theta. The only state returned to agents is a
    public history of their own time-slot/outcome pairs and remaining budget.
    """

    def __init__(self, config: RoomConfig | None = None, *, budget: int = 8) -> None:
        self.config = config or RoomConfig()
        self.budget = self.config.validate_budget(budget)
        self._history: list[Observation] = []
        self._episode_seed: int | None = None
        self._theta: int | None = None
        self._outcome: TerminalOutcome | None = None

    @staticmethod
    def _hash_u64(*parts: int | str) -> int:
        payload = ":".join(str(part) for part in parts).encode("utf-8")
        digest = hashlib.blake2b(payload, digest_size=8).digest()
        return int.from_bytes(digest, byteorder="big", signed=False)

    def _derive_theta(self, episode_seed: int) -> int:
        candidates = self.config.theta_candidates
        index = self._hash_u64("theta", episode_seed) % len(candidates)
        return candidates[index]

    def _draw_uniform(self, time_slot: int, repeat_index: int) -> float:
        if self._episode_seed is None:
            raise RoomNotResetError("reset must be called before observing")
        value = self._hash_u64("light", self._episode_seed, time_slot, repeat_index)
        return value / float(1 << 64)

    def reset(self, *, seed: int, theta: int | None = None) -> RoomState:
        """Start an episode without exposing its hidden theta."""

        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("seed must be an integer")
        self._episode_seed = seed
        self._theta = self.config.validate_theta(theta) if theta is not None else self._derive_theta(seed)
        self._history = []
        self._outcome = None
        return self.state

    @property
    def state(self) -> RoomState:
        if self._episode_seed is None:
            raise RoomNotResetError("reset must be called before reading state")
        return RoomState(
            history=tuple(self._history),
            budget=self.budget,
            remaining_observations=self.budget - len(self._history),
        )

    def observe(self, time_slot: int) -> RoomState:
        """Inspect one slot and return the next public state."""

        if self._episode_seed is None or self._theta is None:
            raise RoomNotResetError("reset must be called before observing")
        if self._outcome is not None:
            raise RoomPhaseError("episode already terminated")
        if self.state.phase != "observe":
            raise RoomPhaseError("observation budget is exhausted; submit an estimate")

        slot = self.config.validate_time_slot(time_slot)
        repeat_index = sum(item.time_slot == slot for item in self._history)
        light_on = self._draw_uniform(slot, repeat_index) < self.config.light_probability(
            slot, self._theta
        )
        self._history.append(
            Observation(time_slot=slot, light_on=light_on, repeat_index=repeat_index)
        )
        return self.state

    def estimate(self, theta_hat: int) -> TerminalOutcome:
        """Submit the terminal estimate and reveal the score."""

        if self._episode_seed is None or self._theta is None:
            raise RoomNotResetError("reset must be called before estimating")
        if self._outcome is not None:
            raise RoomPhaseError("episode already terminated")
        if self.state.phase != "estimate":
            raise RoomPhaseError("all observations must be used before estimating")

        estimate = self.config.validate_time_slot(theta_hat)
        error = abs(self._theta - estimate)
        self._outcome = TerminalOutcome(
            theta=self._theta,
            theta_hat=estimate,
            absolute_error=error,
            reward=-error / (self.config.slot_count - 1),
        )
        return self._outcome

    @property
    def terminal_outcome(self) -> TerminalOutcome | None:
        """Return a score only after successful termination."""

        return self._outcome


def fallback_observation_slot(state: RoomState, config: RoomConfig) -> int:
    """Choose the least-sampled slot, resolving ties toward the smallest slot."""

    counts = Counter(observation.time_slot for observation in state.history)
    minimum = min(counts.get(slot, 0) for slot in range(config.slot_count))
    return next(slot for slot in range(config.slot_count) if counts.get(slot, 0) == minimum)


def fallback_estimate(state: RoomState, config: RoomConfig) -> int:
    """Return the best observed empirical rate without using hidden information."""

    if not state.history:
        return config.slot_count // 2
    visits = Counter(observation.time_slot for observation in state.history)
    on_counts = Counter(
        observation.time_slot for observation in state.history if observation.light_on
    )
    observed_slots = sorted(visits)
    return max(observed_slots, key=lambda slot: (on_counts[slot] / visits[slot], -slot))
