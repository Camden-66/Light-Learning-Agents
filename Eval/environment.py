"""Canonical room contracts used by the evaluator.

The environment deliberately keeps ``theta`` and the episode seed private.
Only ``RoomState`` is returned before the terminal estimate.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass


SLOT_COUNT = 32
THETA_MIN = 4
THETA_MAX = 27
BUDGETS = (4, 8, 16, 32)


@dataclass(frozen=True)
class RoomConfig:
    slot_count: int = SLOT_COUNT
    theta_min: int = THETA_MIN
    theta_max: int = THETA_MAX
    baseline_probability: float = 0.05
    peak_probability: float = 0.90
    width: float = 18.0
    version: str = "room-v1"

    def __post_init__(self) -> None:
        if self.slot_count != SLOT_COUNT:
            raise ValueError(f"slot_count must be {SLOT_COUNT}")
        if (self.theta_min, self.theta_max) != (THETA_MIN, THETA_MAX):
            raise ValueError("theta range must be 4..27")
        if not 0 <= self.baseline_probability <= 1:
            raise ValueError("baseline_probability must be in [0, 1]")
        if not 0 <= self.peak_probability <= 1:
            raise ValueError("peak_probability must be in [0, 1]")
        if self.baseline_probability + self.peak_probability > 1:
            raise ValueError("baseline plus peak probability must be at most 1")
        if self.width <= 0:
            raise ValueError("width must be positive")

    def probability(self, slot: int, theta: int) -> float:
        if not 0 <= slot < self.slot_count:
            raise ValueError("slot is outside the room")
        if not self.theta_min <= theta <= self.theta_max:
            raise ValueError("theta is outside the room's hidden range")
        return self.baseline_probability + self.peak_probability * math.exp(
            -((slot - theta) ** 2) / self.width
        )


@dataclass(frozen=True)
class Observation:
    time_slot: int
    light_on: bool
    repeat_index: int


@dataclass(frozen=True)
class RoomState:
    history: tuple[Observation, ...]
    budget: int
    remaining_observations: int
    phase: str

    def __post_init__(self) -> None:
        if self.budget not in BUDGETS:
            raise ValueError(f"budget must be one of {BUDGETS}")
        if not 0 <= self.remaining_observations <= self.budget:
            raise ValueError("remaining_observations is outside the budget")
        expected_phase = "observe" if self.remaining_observations else "estimate"
        if self.phase != expected_phase:
            raise ValueError(f"phase must be {expected_phase!r}")


@dataclass(frozen=True)
class TerminalOutcome:
    theta: int
    theta_hat: int
    absolute_error: int
    reward: float


def deterministic_light_outcome(
    episode_seed: int,
    slot: int,
    repeat_index: int,
    theta: int,
    config: RoomConfig | None = None,
) -> bool:
    """Return the stable Bernoulli draw for one public query."""

    config = config or RoomConfig()
    if episode_seed < 0:
        raise ValueError("episode_seed must be non-negative")
    if repeat_index < 0:
        raise ValueError("repeat_index must be non-negative")
    probability = config.probability(slot, theta)
    material = f"{episode_seed}|{slot}|{repeat_index}".encode()
    random_bits = int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
    return random_bits < int(probability * 2**64)


class RoomEnv:
    """A minimal deterministic room environment with a public state boundary."""

    def __init__(self, config: RoomConfig | None = None, *, budget: int):
        self.config = config or RoomConfig()
        if budget not in BUDGETS:
            raise ValueError(f"budget must be one of {BUDGETS}")
        self.budget = budget
        self._episode_seed: int | None = None
        self._theta: int | None = None
        self._history: list[Observation] = []
        self._state: RoomState | None = None

    def reset(self, *, seed: int, theta: int) -> RoomState:
        if seed < 0:
            raise ValueError("seed must be non-negative")
        if not self.config.theta_min <= theta <= self.config.theta_max:
            raise ValueError("theta is outside the configured hidden range")
        self._episode_seed = seed
        self._theta = theta
        self._history = []
        self._state = RoomState((), self.budget, self.budget, "observe")
        return self._state

    def _require_state(self) -> RoomState:
        if self._state is None or self._episode_seed is None or self._theta is None:
            raise RuntimeError("room must be reset before use")
        return self._state

    def observe(self, time_slot: int) -> RoomState:
        state = self._require_state()
        if state.phase != "observe":
            raise RuntimeError("cannot observe during estimate phase")
        if isinstance(time_slot, bool) or not isinstance(time_slot, int):
            raise ValueError("time_slot must be an integer")
        if not 0 <= time_slot < self.config.slot_count:
            raise ValueError("time_slot is outside the room")
        repeat_index = sum(item.time_slot == time_slot for item in self._history)
        light_on = deterministic_light_outcome(
            self._episode_seed, time_slot, repeat_index, self._theta, self.config
        )
        self._history.append(Observation(time_slot, light_on, repeat_index))
        remaining = state.remaining_observations - 1
        self._state = RoomState(
            tuple(self._history), self.budget, remaining, "observe" if remaining else "estimate"
        )
        return self._state

    def estimate(self, theta_hat: int) -> TerminalOutcome:
        state = self._require_state()
        if state.phase != "estimate":
            raise RuntimeError("cannot estimate before the observation budget is exhausted")
        if isinstance(theta_hat, bool) or not isinstance(theta_hat, int):
            raise ValueError("theta_hat must be an integer")
        if not 0 <= theta_hat < self.config.slot_count:
            raise ValueError("theta_hat is outside the room")
        assert self._theta is not None
        absolute_error = abs(self._theta - theta_hat)
        return TerminalOutcome(
            self._theta,
            theta_hat,
            absolute_error,
            -absolute_error / (self.config.slot_count - 1),
        )
