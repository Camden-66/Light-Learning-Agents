"""Gymnasium adapter over the canonical room environment."""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np

from .config import RoomConfig
from .room import RoomEnv
from .types import RoomState

try:  # Keep import errors actionable for users who have not run `uv sync` yet.
    import gymnasium as gym
    from gymnasium import spaces
except ModuleNotFoundError as error:  # pragma: no cover - dependency contract
    gym = None  # type: ignore[assignment]
    spaces = None  # type: ignore[assignment]
    _GYM_IMPORT_ERROR = error
else:
    _GYM_IMPORT_ERROR = None


if gym is not None:

    class GymRoomEnv(gym.Env[np.ndarray, int]):
        """32-action RL adapter with a fixed 66-float public observation."""

        metadata = {"render_modes": []}

        def __init__(self, config: RoomConfig | None = None, *, budget: int = 8) -> None:
            super().__init__()
            self.config = config or RoomConfig()
            self.budget = self.config.validate_budget(budget)
            self._core = RoomEnv(self.config, budget=self.budget)
            self.action_space = spaces.Discrete(self.config.slot_count)
            self.observation_space = spaces.Box(
                low=0.0,
                high=1.0,
                shape=(2 * self.config.slot_count + 2,),
                dtype=np.float32,
            )
            self._state: RoomState | None = None
            self._episode_seed: int | None = None

        def _vector(self, state: RoomState) -> np.ndarray:
            on_counts = Counter(
                observation.time_slot for observation in state.history if observation.light_on
            )
            visit_counts = Counter(observation.time_slot for observation in state.history)
            values = np.zeros(2 * self.config.slot_count + 2, dtype=np.float32)
            for slot in range(self.config.slot_count):
                values[slot] = on_counts[slot] / self.budget
                values[self.config.slot_count + slot] = visit_counts[slot] / self.budget
            values[-2] = state.remaining_observations / self.budget
            values[-1] = 1.0 if state.phase == "estimate" else 0.0
            return values

        def reset(
            self,
            *,
            seed: int | None = None,
            options: dict[str, Any] | None = None,
        ) -> tuple[np.ndarray, dict[str, Any]]:
            super().reset(seed=seed)
            options = options or {}
            episode_seed = int(seed) if seed is not None else int(
                self.np_random.integers(0, (1 << 63) - 1)
            )
            theta = options.get("theta")
            self._state = self._core.reset(seed=episode_seed, theta=theta)
            self._episode_seed = episode_seed
            return self._vector(self._state), {
                "budget": self.budget,
                "remaining_observations": self._state.remaining_observations,
            }

        def step(
            self, action: int
        ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
            if self._state is None:
                raise RuntimeError("reset must be called before step")
            if not self.action_space.contains(action):
                raise ValueError(f"action must be in [0, {self.config.slot_count - 1}]")
            action = int(action)

            if self._state.phase == "observe":
                self._state = self._core.observe(action)
                return self._vector(self._state), 0.0, False, False, {
                    "time_slot": action,
                    "light_on": self._state.history[-1].light_on,
                    "remaining_observations": self._state.remaining_observations,
                }

            outcome = self._core.estimate(action)
            return self._vector(self._state), outcome.reward, True, False, {
                "theta": outcome.theta,
                "theta_hat": outcome.theta_hat,
                "absolute_error": outcome.absolute_error,
                "episode_seed": self._episode_seed,
            }

else:

    class GymRoomEnv:  # pragma: no cover - only reached without project dependencies
        """Placeholder that explains how to enable the optional runtime."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError(
                "GymRoomEnv requires gymnasium and numpy. Run `uv sync` first."
            ) from _GYM_IMPORT_ERROR
