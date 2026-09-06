from __future__ import annotations

import pytest

gymnasium = pytest.importorskip("gymnasium")

from light_learning.gym_env import GymRoomEnv


def test_gym_adapter_has_public_vector_and_terminal_only_score() -> None:
    env = GymRoomEnv(budget=4)
    observation, info = env.reset(seed=123, options={"theta": 12})
    assert observation.shape == (66,)
    assert env.observation_space.contains(observation)
    assert "theta" not in info
    for _ in range(4):
        observation, reward, terminated, truncated, info = env.step(12)
        assert reward == 0.0
        assert not terminated
        assert not truncated
        assert "theta" not in info
    observation, reward, terminated, truncated, info = env.step(12)
    assert env.observation_space.contains(observation)
    assert terminated
    assert not truncated
    assert info["theta"] == 12
    assert info["theta_hat"] == 12
    assert reward == 0.0


def test_gym_adapter_rejects_invalid_action() -> None:
    env = GymRoomEnv(budget=4)
    env.reset(seed=123)
    with pytest.raises(ValueError):
        env.step(32)
