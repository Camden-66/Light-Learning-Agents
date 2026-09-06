"""Canonical room-task configuration and likelihood."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp

SLOT_COUNT = 32
BUDGETS = (4, 8, 16, 32)


@dataclass(frozen=True, slots=True)
class RoomConfig:
    """Immutable parameters of the canonical hidden-peak room."""

    slot_count: int = SLOT_COUNT
    theta_min: int = 4
    theta_max: int = 27
    background_probability: float = 0.05
    peak_amplitude: float = 0.90
    exponent_denominator: float = 18.0
    budgets: tuple[int, ...] = BUDGETS

    def __post_init__(self) -> None:
        if self.slot_count < 2:
            raise ValueError("slot_count must be at least 2")
        if not 0 <= self.theta_min <= self.theta_max < self.slot_count:
            raise ValueError("theta support must be within the time-slot range")
        if not 0.0 <= self.background_probability <= 1.0:
            raise ValueError("background_probability must be in [0, 1]")
        if not 0.0 <= self.peak_amplitude <= 1.0:
            raise ValueError("peak_amplitude must be in [0, 1]")
        if self.background_probability + self.peak_amplitude > 1.0:
            raise ValueError("peak probability may not exceed 1")
        if self.exponent_denominator <= 0.0:
            raise ValueError("exponent_denominator must be positive")
        if not self.budgets or any(budget <= 0 for budget in self.budgets):
            raise ValueError("budgets must contain positive values")

    @property
    def theta_candidates(self) -> tuple[int, ...]:
        return tuple(range(self.theta_min, self.theta_max + 1))

    def validate_time_slot(self, time_slot: int) -> int:
        if isinstance(time_slot, bool) or not isinstance(time_slot, int):
            raise TypeError("time_slot must be an integer")
        if not 0 <= time_slot < self.slot_count:
            raise ValueError(f"time_slot must be in [0, {self.slot_count - 1}]")
        return time_slot

    def validate_theta(self, theta: int) -> int:
        if isinstance(theta, bool) or not isinstance(theta, int):
            raise TypeError("theta must be an integer")
        if theta not in self.theta_candidates:
            raise ValueError(
                f"theta must be in [{self.theta_min}, {self.theta_max}]"
            )
        return theta

    def validate_budget(self, budget: int) -> int:
        if isinstance(budget, bool) or not isinstance(budget, int):
            raise TypeError("budget must be an integer")
        if budget not in self.budgets:
            raise ValueError(f"budget must be one of {self.budgets}")
        return budget

    def light_probability(self, time_slot: int, theta: int) -> float:
        """Return the canonical probability of an on observation."""

        self.validate_time_slot(time_slot)
        self.validate_theta(theta)
        distance = time_slot - theta
        return self.background_probability + self.peak_amplitude * exp(
            -(distance * distance) / self.exponent_denominator
        )
