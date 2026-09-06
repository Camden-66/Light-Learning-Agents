"""Stable-Baselines3 PPO training and evaluator-facing inference support.

The evaluator owns held-out episodes, records, fallbacks, and metrics. This
module only trains canonical PPO checkpoints and adapts one checkpoint to the
shared ``RoomAgent`` protocol.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import asdict
from importlib.metadata import PackageNotFoundError, version
from platform import python_version
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping, Sequence

import numpy as np

from .config import BUDGETS, RoomConfig
from .gym_env import GymRoomEnv
from .types import AgentDecision, RoomState

PPO_AGENT_ID = "stable-baselines3-ppo"
PPO_AGENT_VERSION = "0.2.0"
PPO_CONDITION = "trained_distribution_ppo"
PPO_TRAINING_SEEDS = (0, 1, 2, 3, 4)
PPO_MANIFEST_NAME = "ppo-training-manifest.json"


def _validate_seed(seed: int) -> int:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("training seed must be an integer")
    if seed < 0:
        raise ValueError("training seed must be non-negative")
    return seed


def _config_payload(config: RoomConfig) -> dict[str, Any]:
    return asdict(config)


def _canonical_config(config: RoomConfig | None) -> RoomConfig:
    supplied = config or RoomConfig()
    if supplied != RoomConfig():
        raise ValueError("the PPO benchmark requires the canonical RoomConfig")
    return supplied


def room_config_digest(config: RoomConfig) -> str:
    """Return a stable digest for checkpoint/config compatibility checks."""

    encoded = json.dumps(
        _config_payload(config), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _checkpoint_path(path: str | Path) -> Path:
    checkpoint = Path(path)
    if checkpoint.exists():
        return checkpoint
    if checkpoint.suffix != ".zip":
        zipped = checkpoint.with_suffix(".zip")
        if zipped.exists():
            return zipped
    raise FileNotFoundError(f"PPO checkpoint not found: {checkpoint}")


def checkpoint_digest(path: str | Path) -> str:
    """Hash one saved checkpoint without loading or executing it."""

    checkpoint = _checkpoint_path(path)
    digest = hashlib.sha256()
    with checkpoint.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def room_state_vector(
    state: RoomState,
    *,
    config: RoomConfig | None = None,
    expected_budget: int | None = None,
) -> np.ndarray:
    """Encode only public ``RoomState`` fields using the 66-value Gym contract."""

    room_config = _canonical_config(config)
    budget = room_config.validate_budget(state.budget)
    if expected_budget is not None and budget != room_config.validate_budget(
        expected_budget
    ):
        raise ValueError(
            f"state budget {budget} does not match PPO policy budget {expected_budget}"
        )
    if isinstance(state.remaining_observations, bool) or not isinstance(
        state.remaining_observations, int
    ):
        raise TypeError("remaining_observations must be an integer")
    if not 0 <= state.remaining_observations <= budget:
        raise ValueError("remaining_observations must be between zero and budget")
    if len(state.history) + state.remaining_observations != budget:
        raise ValueError("history length and remaining observations must sum to budget")

    visits: Counter[int] = Counter()
    on_counts: Counter[int] = Counter()
    repeat_counts: Counter[int] = Counter()
    for observation in state.history:
        slot = room_config.validate_time_slot(observation.time_slot)
        if observation.repeat_index != repeat_counts[slot]:
            raise ValueError("observation repeat indices must be chronological per slot")
        repeat_counts[slot] += 1
        visits[slot] += 1
        if observation.light_on:
            on_counts[slot] += 1

    values = np.zeros(2 * room_config.slot_count + 2, dtype=np.float32)
    for slot in range(room_config.slot_count):
        values[slot] = on_counts[slot] / budget
        values[room_config.slot_count + slot] = visits[slot] / budget
    values[-2] = state.remaining_observations / budget
    values[-1] = 1.0 if state.phase == "estimate" else 0.0
    return values


def _runtime_version(package: str) -> str | None:
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def _runtime_metadata() -> dict[str, str | None]:
    return {
        "python": python_version(),
        "stable_baselines3": _runtime_version("stable-baselines3"),
        "torch": _runtime_version("torch"),
        "gymnasium": _runtime_version("gymnasium"),
        "numpy": np.__version__,
        "device": "cpu",
    }


def _validate_model_spaces(model: Any, config: RoomConfig) -> None:
    observation_space = getattr(model, "observation_space", None)
    action_space = getattr(model, "action_space", None)
    expected_shape = (2 * config.slot_count + 2,)
    actual_shape = getattr(observation_space, "shape", None)
    if tuple(actual_shape or ()) != expected_shape:
        raise ValueError(
            f"checkpoint observation shape {actual_shape!r} does not match {expected_shape}"
        )
    action_count = getattr(action_space, "n", None)
    if action_count != config.slot_count:
        raise ValueError(
            f"checkpoint action count {action_count!r} does not match {config.slot_count}"
        )


class PPORoomAgent:
    """Expose a trained PPO policy through the shared evaluator agent protocol."""

    agent_id = PPO_AGENT_ID
    condition = PPO_CONDITION

    def __init__(
        self,
        model: Any,
        *,
        budget: int,
        training_seed: int,
        config: RoomConfig | None = None,
        checkpoint_path: str | Path | None = None,
        checkpoint_sha256: str | None = None,
    ) -> None:
        self.config = _canonical_config(config)
        self.budget = self.config.validate_budget(budget)
        self.training_seed = _validate_seed(training_seed)
        _validate_model_spaces(model, self.config)
        self._model = model
        self._started = False
        self._checkpoint_path = (
            str(Path(checkpoint_path).resolve()) if checkpoint_path is not None else None
        )
        self._checkpoint_sha256 = checkpoint_sha256
        checkpoint_tag = checkpoint_sha256[:12] if checkpoint_sha256 else "in-memory"
        self.agent_version = (
            f"{PPO_AGENT_VERSION}+b{self.budget}.s{self.training_seed}.{checkpoint_tag}"
        )

    @classmethod
    def load_checkpoint(
        cls,
        path: str | Path,
        *,
        budget: int,
        training_seed: int,
        config: RoomConfig | None = None,
        expected_sha256: str | None = None,
    ) -> PPORoomAgent:
        """Load and validate an SB3 checkpoint for deterministic evaluation."""

        from stable_baselines3 import PPO

        room_config = _canonical_config(config)
        room_config.validate_budget(budget)
        _validate_seed(training_seed)
        checkpoint = _checkpoint_path(path)
        actual_sha256 = checkpoint_digest(checkpoint)
        if expected_sha256 is not None and actual_sha256 != expected_sha256:
            raise ValueError(
                "checkpoint SHA-256 does not match the evaluator-provided manifest"
            )
        model = PPO.load(str(checkpoint), device="cpu")
        return cls(
            model,
            budget=budget,
            training_seed=training_seed,
            config=room_config,
            checkpoint_path=checkpoint,
            checkpoint_sha256=actual_sha256,
        )

    def start_episode(self, state: RoomState) -> None:
        if state.history:
            raise ValueError("a PPO episode must begin with an empty history")
        room_state_vector(state, config=self.config, expected_budget=self.budget)
        if state.remaining_observations != self.budget:
            raise ValueError("a PPO episode must begin with its full observation budget")
        self._started = True

    def act(self, state: RoomState) -> AgentDecision:
        if not self._started:
            raise RuntimeError("start_episode must be called before act")
        observation = room_state_vector(
            state, config=self.config, expected_budget=self.budget
        )
        started_at = perf_counter()
        action, _state = self._model.predict(observation, deterministic=True)
        latency_seconds = perf_counter() - started_at
        action_values = np.asarray(action)
        if action_values.size != 1:
            raise ValueError("PPO policy must return exactly one discrete action")
        scalar = action_values.reshape(-1)[0]
        if isinstance(scalar, (bool, np.bool_)) or not np.issubdtype(
            np.asarray(scalar).dtype, np.integer
        ):
            raise TypeError("PPO policy action must be an integer")
        selected = self.config.validate_time_slot(int(scalar))
        return AgentDecision(
            action=selected,
            latency_seconds=latency_seconds,
            metadata={
                "algorithm": "PPO",
                "deterministic": True,
                "training_seed": self.training_seed,
                "phase": state.phase,
            },
        )

    def run_metadata(self) -> Mapping[str, Any]:
        return {
            "agent": self.agent_id,
            "agent_version": self.agent_version,
            "algorithm": "Stable-Baselines3 PPO",
            "condition": self.condition,
            "budget": self.budget,
            "training_seed": self.training_seed,
            "deterministic_evaluation": True,
            "checkpoint_path": self._checkpoint_path,
            "checkpoint_sha256": self._checkpoint_sha256,
            "room_config": _config_payload(self.config),
            "room_config_digest": room_config_digest(self.config),
            "runtime": _runtime_metadata(),
        }


def _training_pairs(
    budgets: Sequence[int], seeds: Sequence[int], config: RoomConfig
) -> list[tuple[int, int]]:
    checked_budgets = [config.validate_budget(budget) for budget in budgets]
    checked_seeds = [_validate_seed(seed) for seed in seeds]
    if len(set(checked_budgets)) != len(checked_budgets):
        raise ValueError("training budgets must be unique")
    if len(set(checked_seeds)) != len(checked_seeds):
        raise ValueError("training seeds must be unique")
    if not checked_budgets or not checked_seeds:
        raise ValueError("training budgets and seeds may not be empty")
    return [
        (budget, seed) for budget in checked_budgets for seed in checked_seeds
    ]


def build_training_manifest(
    *,
    output_dir: str | Path,
    budgets: Sequence[int] = BUDGETS,
    seeds: Sequence[int] = PPO_TRAINING_SEEDS,
    total_timesteps: int = 20_000,
    config: RoomConfig | None = None,
    ppo_kwargs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Describe the complete, auditable training matrix without training it."""

    room_config = _canonical_config(config)
    if isinstance(total_timesteps, bool) or not isinstance(total_timesteps, int):
        raise TypeError("total_timesteps must be an integer")
    if total_timesteps <= 0:
        raise ValueError("total_timesteps must be positive")
    pairs = _training_pairs(budgets, seeds, room_config)
    destination = Path(output_dir)
    kwargs = dict(ppo_kwargs or {})
    try:
        json.dumps(kwargs)
    except (TypeError, ValueError) as error:
        raise TypeError("ppo_kwargs must be JSON-serializable for the manifest") from error
    return {
        "schema_version": 1,
        "agent_id": PPO_AGENT_ID,
        "agent_code_version": PPO_AGENT_VERSION,
        "algorithm": "Stable-Baselines3 PPO",
        "condition": PPO_CONDITION,
        "deterministic_evaluation": True,
        "training_distribution": "canonical_room_distribution",
        "held_out_episode_definitions_used_for_training": False,
        "room_config": _config_payload(room_config),
        "room_config_digest": room_config_digest(room_config),
        "total_timesteps_per_run": total_timesteps,
        "ppo_kwargs": kwargs,
        "runtime": _runtime_metadata(),
        "runs": [
            {
                "budget": budget,
                "training_seed": seed,
                "checkpoint": str(destination / f"ppo_b{budget}_seed{seed}.zip"),
                "status": "planned",
                "checkpoint_sha256": None,
            }
            for budget, seed in pairs
        ],
    }


def train_ppo(
    budget: int,
    seed: int,
    total_timesteps: int = 20_000,
    save_dir: str | Path | None = None,
    *,
    config: RoomConfig | None = None,
    ppo_kwargs: Mapping[str, Any] | None = None,
):
    """Train one independently seeded PPO policy on the canonical Gym task."""

    from stable_baselines3 import PPO

    room_config = _canonical_config(config)
    checked_budget = room_config.validate_budget(budget)
    checked_seed = _validate_seed(seed)
    if isinstance(total_timesteps, bool) or not isinstance(total_timesteps, int):
        raise TypeError("total_timesteps must be an integer")
    if total_timesteps <= 0:
        raise ValueError("total_timesteps must be positive")
    kwargs = dict(ppo_kwargs or {})
    reserved = {"policy", "env", "seed", "verbose", "device"}.intersection(kwargs)
    if reserved:
        names = ", ".join(sorted(reserved))
        raise ValueError(f"ppo_kwargs may not override reproducibility fields: {names}")

    env = GymRoomEnv(config=room_config, budget=checked_budget)
    try:
        model = PPO(
            "MlpPolicy",
            env,
            seed=checked_seed,
            verbose=0,
            device="cpu",
            **kwargs,
        )
        model.learn(total_timesteps=total_timesteps)
    except Exception:
        env.close()
        raise
    if save_dir is not None:
        checkpoint = Path(save_dir) / f"ppo_b{checked_budget}_seed{checked_seed}.zip"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(checkpoint))
    return model


def train_ppo_suite(
    output_dir: str | Path,
    *,
    budgets: Sequence[int] = BUDGETS,
    seeds: Sequence[int] = PPO_TRAINING_SEEDS,
    total_timesteps: int = 20_000,
    config: RoomConfig | None = None,
    ppo_kwargs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Train every budget/seed pair and write the evaluator-consumable manifest."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    manifest = build_training_manifest(
        output_dir=destination,
        budgets=budgets,
        seeds=seeds,
        total_timesteps=total_timesteps,
        config=config,
        ppo_kwargs=ppo_kwargs,
    )
    room_config = _canonical_config(config)
    for run in manifest["runs"]:
        budget = int(run["budget"])
        seed = int(run["training_seed"])
        model = train_ppo(
            budget,
            seed,
            total_timesteps=total_timesteps,
            save_dir=destination,
            config=room_config,
            ppo_kwargs=ppo_kwargs,
        )
        env = model.get_env()
        if env is not None:
            env.close()
        checkpoint = _checkpoint_path(run["checkpoint"])
        run["checkpoint"] = str(checkpoint.resolve())
        run["checkpoint_sha256"] = checkpoint_digest(checkpoint)
        run["status"] = "complete"

    manifest_path = destination / PPO_MANIFEST_NAME
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest
