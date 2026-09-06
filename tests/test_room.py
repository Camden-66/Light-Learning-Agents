from __future__ import annotations

from dataclasses import asdict

import pytest

from light_learning.config import RoomConfig
from light_learning.room import (
    RoomEnv,
    RoomPhaseError,
    fallback_estimate,
    fallback_observation_slot,
)


def test_canonical_probability_matches_specification() -> None:
    config = RoomConfig()
    assert config.light_probability(10, 10) == pytest.approx(0.95)
    assert config.light_probability(0, 10) == pytest.approx(
        0.05 + 0.90 * __import__("math").exp(-100 / 18)
    )


def test_public_state_never_contains_hidden_theta_before_terminal() -> None:
    env = RoomEnv(budget=4)
    state = env.reset(seed=7, theta=12)
    assert "theta" not in asdict(state)
    state = env.observe(10)
    assert "theta" not in asdict(state)
    assert env.terminal_outcome is None


def test_seeded_slot_outcomes_do_not_depend_on_cross_slot_order() -> None:
    first = RoomEnv(budget=4)
    first.reset(seed=991, theta=13)
    first.observe(8)
    first_state = first.observe(12)

    second = RoomEnv(budget=4)
    second.reset(seed=991, theta=13)
    second_state = second.observe(12)

    assert first_state.history[-1].light_on == second_state.history[-1].light_on
    assert first_state.history[-1].repeat_index == second_state.history[-1].repeat_index == 0


def test_episode_requires_full_budget_then_reveals_terminal_score() -> None:
    env = RoomEnv(budget=4)
    state = env.reset(seed=2, theta=14)
    for _ in range(4):
        state = env.observe(14)
    assert state.phase == "estimate"
    with pytest.raises(RoomPhaseError):
        env.observe(14)
    outcome = env.estimate(13)
    assert outcome.theta == 14
    assert outcome.theta_hat == 13
    assert outcome.absolute_error == 1
    assert outcome.reward == pytest.approx(-1 / 31)
    with pytest.raises(RoomPhaseError):
        env.estimate(13)


def test_fallbacks_use_only_public_history() -> None:
    env = RoomEnv(budget=4)
    state = env.reset(seed=9, theta=20)
    assert fallback_observation_slot(state, env.config) == 0
    state = env.observe(5)
    state = env.observe(5)
    state = env.observe(8)
    state = env.observe(8)
    assert fallback_estimate(state, env.config) in {5, 8}
