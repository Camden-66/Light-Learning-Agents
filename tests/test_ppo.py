from __future__ import annotations

import importlib.util
from types import SimpleNamespace

import numpy as np
import pytest

from light_learning.config import BUDGETS, RoomConfig
from light_learning.ppo import (
    PPO_TRAINING_SEEDS,
    PPORoomAgent,
    build_training_manifest,
    room_config_digest,
    room_state_vector,
)
from light_learning.room import RoomEnv
from light_learning.types import Observation, RoomState


class FakeModel:
    observation_space = SimpleNamespace(shape=(66,))
    action_space = SimpleNamespace(n=32)

    def __init__(self, action: int = 7) -> None:
        self.action = action
        self.calls: list[tuple[np.ndarray, bool]] = []

    def predict(self, observation: np.ndarray, *, deterministic: bool):
        self.calls.append((observation, deterministic))
        return np.asarray(self.action), None


def test_room_state_vector_matches_public_gym_contract() -> None:
    state = RoomState(
        history=(
            Observation(time_slot=3, light_on=True, repeat_index=0),
            Observation(time_slot=3, light_on=False, repeat_index=1),
        ),
        budget=4,
        remaining_observations=2,
    )

    vector = room_state_vector(state, expected_budget=4)

    assert vector.shape == (66,)
    assert vector.dtype == np.float32
    assert vector[3] == pytest.approx(0.25)
    assert vector[32 + 3] == pytest.approx(0.5)
    assert vector[-2] == pytest.approx(0.5)
    assert vector[-1] == 0.0
    assert np.count_nonzero(vector) == 3


def test_ppo_agent_uses_deterministic_prediction_and_public_state() -> None:
    model = FakeModel(action=11)
    agent = PPORoomAgent(model, budget=4, training_seed=2)
    state = RoomState(history=(), budget=4, remaining_observations=4)
    agent.start_episode(state)

    decision = agent.act(state)

    assert decision.action == 11
    assert model.calls[0][1] is True
    assert model.calls[0][0].shape == (66,)
    assert decision.metadata["deterministic"] is True
    assert decision.latency_seconds >= 0.0
    metadata = agent.run_metadata()
    assert metadata["training_seed"] == 2
    assert metadata["room_config_digest"] == room_config_digest(RoomConfig())
    expected_runtime_fields = {
        "python",
        "stable_baselines3",
        "torch",
        "gymnasium",
        "numpy",
        "device",
    }
    assert expected_runtime_fields <= set(metadata["runtime"])
    assert metadata["runtime"]["python"]
    assert metadata["runtime"]["gymnasium"]
    assert metadata["runtime"]["device"] == "cpu"


def test_ppo_agent_rejects_budget_and_model_space_mismatches() -> None:
    agent = PPORoomAgent(FakeModel(), budget=4, training_seed=0)
    wrong_budget = RoomState(history=(), budget=8, remaining_observations=8)
    with pytest.raises(ValueError, match="does not match PPO policy budget"):
        agent.start_episode(wrong_budget)

    bad_model = FakeModel()
    bad_model.action_space = SimpleNamespace(n=31)
    with pytest.raises(ValueError, match="action count"):
        PPORoomAgent(bad_model, budget=4, training_seed=0)


@pytest.mark.parametrize("budget", BUDGETS)
def test_public_state_encoder_matches_gym_observations(budget: int) -> None:
    from light_learning.gym_env import GymRoomEnv

    core = RoomEnv(budget=budget)
    gym_env = GymRoomEnv(budget=budget)
    state = core.reset(seed=8675309, theta=17)
    gym_observation, _ = gym_env.reset(seed=8675309, options={"theta": 17})
    np.testing.assert_array_equal(room_state_vector(state), gym_observation)

    for step in range(budget):
        action = (step * 7) % 32
        state = core.observe(action)
        gym_observation, reward, terminated, truncated, info = gym_env.step(action)
        np.testing.assert_array_equal(room_state_vector(state), gym_observation)
        assert reward == 0.0
        assert terminated is False
        assert truncated is False
        assert info["light_on"] == state.history[-1].light_on

    gym_env.close()


def test_default_training_manifest_has_four_budgets_and_five_seeds(tmp_path) -> None:
    manifest = build_training_manifest(output_dir=tmp_path)

    expected = {(budget, seed) for budget in BUDGETS for seed in PPO_TRAINING_SEEDS}
    actual = {
        (run["budget"], run["training_seed"])
        for run in manifest["runs"]
    }
    assert actual == expected
    assert len(manifest["runs"]) == 20
    assert manifest["deterministic_evaluation"] is True
    assert manifest["held_out_episode_definitions_used_for_training"] is False
    assert all(run["status"] == "planned" for run in manifest["runs"])


def test_manifest_rejects_duplicate_seeds(tmp_path) -> None:
    with pytest.raises(ValueError, match="seeds must be unique"):
        build_training_manifest(output_dir=tmp_path, seeds=(3, 3))


def test_manifest_rejects_non_json_ppo_options(tmp_path) -> None:
    with pytest.raises(TypeError, match="JSON-serializable"):
        build_training_manifest(output_dir=tmp_path, ppo_kwargs={"schedule": object()})


def test_ppo_agent_rejects_noncanonical_room_config() -> None:
    with pytest.raises(ValueError, match="canonical RoomConfig"):
        PPORoomAgent(
            FakeModel(),
            budget=4,
            training_seed=0,
            config=RoomConfig(background_probability=0.06),
        )


@pytest.mark.skipif(
    importlib.util.find_spec("stable_baselines3") is None,
    reason="Stable-Baselines3 is not installed",
)
def test_ppo_smoke_training_checkpoint_and_evaluator_rollout(tmp_path) -> None:
    from light_learning.ppo import PPO_MANIFEST_NAME, train_ppo_suite

    manifest = train_ppo_suite(
        tmp_path,
        budgets=(4,),
        seeds=(3,),
        total_timesteps=16,
        ppo_kwargs={"n_steps": 8, "batch_size": 8, "n_epochs": 1},
    )
    run = manifest["runs"][0]
    assert run["status"] == "complete"
    assert run["checkpoint_sha256"]
    assert manifest["runtime"]["stable_baselines3"]
    assert manifest["runtime"]["torch"]
    assert (tmp_path / PPO_MANIFEST_NAME).is_file()

    agent = PPORoomAgent.load_checkpoint(
        run["checkpoint"],
        budget=4,
        training_seed=3,
        expected_sha256=run["checkpoint_sha256"],
    )
    env = RoomEnv(budget=4)
    state = env.reset(seed=91, theta=17)
    agent.start_episode(state)
    while state.phase == "observe":
        decision = agent.act(state)
        assert 0 <= decision.action < 32
        state = env.observe(decision.action)
    terminal_decision = agent.act(state)
    outcome = env.estimate(terminal_decision.action)
    assert 0 <= outcome.theta_hat < 32
    assert outcome.theta == 17
