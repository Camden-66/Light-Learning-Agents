from __future__ import annotations

import json
import math

import pytest

from light_learning.config import RoomConfig
from light_learning.mle import (
    PassiveUniformOracleMLE,
    log_likelihood,
    mle_theta_hat,
    run_mle_episode,
)
from light_learning.room import RoomEnv


def test_hand_calculated_likelihood_prefers_true_peak() -> None:
    config = RoomConfig()
    history = [(10, 1)]
    assert config.light_probability(10, 10) == pytest.approx(0.95)
    ll_true = math.log(0.95)
    p_far = config.light_probability(10, 4)
    assert math.log(p_far) < ll_true
    assert mle_theta_hat(history, config) == 10


def test_mixed_on_off_log_likelihood_matches_hand_calculation() -> None:
    config = RoomConfig()
    candidate = 12
    history = [(10, 1), (10, 0), (15, 0), (12, 1)]

    p_10 = config.light_probability(10, candidate)
    p_15 = config.light_probability(15, candidate)
    p_12 = config.light_probability(12, candidate)
    expected = math.log(p_10) + math.log(1 - p_10)
    expected += math.log(1 - p_15) + math.log(p_12)

    assert log_likelihood(history, candidate, config) == pytest.approx(expected)


def test_log_likelihood_rejects_non_binary_observations() -> None:
    with pytest.raises(ValueError, match="binary"):
        log_likelihood([(10, 2)], 10, RoomConfig())


def test_empty_history_tie_breaks_to_smallest_candidate() -> None:
    assert mle_theta_hat([], RoomConfig()) == 4


def collect_schedule(*, query_seed: int, room_seed: int, budget: int) -> tuple[int, ...]:
    env = RoomEnv(budget=budget)
    state = env.reset(seed=room_seed, theta=18)
    agent = PassiveUniformOracleMLE(query_seed=query_seed)
    agent.start_episode(state)
    actions: list[int] = []
    while state.phase == "observe":
        action = agent.act(state).action
        actions.append(action)
        state = env.observe(action)
    return tuple(actions)


@pytest.mark.parametrize("budget", RoomConfig().budgets)
def test_uniform_query_schedule_completes_every_budget(budget: int) -> None:
    schedule = collect_schedule(query_seed=2026, room_seed=91, budget=budget)

    assert len(schedule) == budget
    assert all(0 <= slot < RoomConfig().slot_count for slot in schedule)


def test_query_schedule_is_with_replacement_and_has_stable_known_draws() -> None:
    schedule = PassiveUniformOracleMLE(query_seed=2026).query_schedule(32)

    assert len(set(schedule)) < len(schedule)
    assert schedule[:8] == (18, 27, 15, 20, 10, 9, 21, 30)


def test_query_seed_is_independent_of_hidden_room_seed() -> None:
    first = collect_schedule(query_seed=77, room_seed=1, budget=16)
    retry = collect_schedule(query_seed=77, room_seed=999_999, budget=16)
    other_policy_seed = collect_schedule(query_seed=78, room_seed=1, budget=16)

    assert retry == first
    assert other_policy_seed != first


def test_partial_episode_can_resume_from_same_query_seed() -> None:
    query_seed = 314159
    full_schedule = PassiveUniformOracleMLE(query_seed=query_seed).query_schedule(8)
    env = RoomEnv(budget=8)
    state = env.reset(seed=2718, theta=18)
    for action in full_schedule[:3]:
        state = env.observe(action)

    resumed = PassiveUniformOracleMLE(query_seed=query_seed)
    resumed.start_episode(state)

    assert resumed.act(state).action == full_schedule[3]


def test_resume_rejects_history_from_another_query_schedule() -> None:
    env = RoomEnv(budget=4)
    state = env.reset(seed=100, theta=18)
    expected_first = PassiveUniformOracleMLE(query_seed=1).query_schedule(4)[0]
    state = env.observe((expected_first + 1) % 32)

    with pytest.raises(ValueError, match="not a prefix"):
        PassiveUniformOracleMLE(query_seed=1).start_episode(state)


def test_run_metadata_is_json_serializable_and_digestible() -> None:
    agent = PassiveUniformOracleMLE(query_seed=1234)
    metadata = agent.run_metadata()

    assert json.loads(json.dumps(metadata)) == metadata
    assert metadata["query_policy"]["query_seed"] == 1234
    assert metadata["query_policy"]["sampling"] == "iid_with_replacement"
    assert metadata["estimator"]["theta_candidates"] == list(range(4, 28))
    assert metadata["likelihood"] == {
        "background_probability": 0.05,
        "peak_amplitude": 0.90,
        "exponent_denominator": 18.0,
    }
    assert metadata["configuration_digest"].startswith("sha256:")
    assert len(metadata["configuration_digest"]) == len("sha256:") + 64
    assert "episode_seed" not in json.dumps(metadata)
    assert metadata == agent.run_metadata()
    assert metadata["configuration_digest"] != PassiveUniformOracleMLE(
        query_seed=1235
    ).run_metadata()["configuration_digest"]


def test_query_seed_is_required_and_must_be_an_integer() -> None:
    with pytest.raises(TypeError, match="query_seed"):
        PassiveUniformOracleMLE(query_seed=True)
    with pytest.raises(TypeError, match="query_seed"):
        PassiveUniformOracleMLE(query_seed="123")  # type: ignore[arg-type]


@pytest.mark.parametrize("budget", RoomConfig().budgets)
def test_mle_completes_a_room_episode_at_every_budget(budget: int) -> None:
    env = RoomEnv(budget=budget)
    env.reset(seed=11, theta=18)
    outcome = run_mle_episode(env, query_seed=2026 + budget)

    assert 4 <= outcome.theta_hat <= 27
    assert outcome.absolute_error == abs(18 - outcome.theta_hat)
    assert env.terminal_outcome is not None


def test_run_mle_episode_refuses_implicit_policy_randomness() -> None:
    env = RoomEnv(budget=4)
    env.reset(seed=11, theta=18)

    with pytest.raises(ValueError, match="query_seed"):
        run_mle_episode(env)


def test_run_mle_episode_rejects_two_randomness_sources() -> None:
    env = RoomEnv(budget=4)
    env.reset(seed=11, theta=18)
    agent = PassiveUniformOracleMLE(query_seed=1)

    with pytest.raises(ValueError, match="either"):
        run_mle_episode(env, agent, query_seed=2)
