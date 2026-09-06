from __future__ import annotations

import math

import pytest

from light_learning.config import RoomConfig
from light_learning.mle import mle_theta_hat, run_mle_episode, PassiveUniformOracleMLE
from light_learning.room import RoomEnv


def test_hand_calculated_likelihood_prefers_true_peak() -> None:
    config = RoomConfig()
    history = [(10, 1)]
    assert config.light_probability(10, 10) == pytest.approx(0.95)
    ll_true = math.log(0.95)
    p_far = config.light_probability(10, 4)
    assert math.log(p_far) < ll_true
    assert mle_theta_hat(history, config) == 10


def test_empty_history_tie_breaks_to_smallest_candidate() -> None:
    assert mle_theta_hat([], RoomConfig()) == 4


def test_mle_completes_a_room_episode() -> None:
    env = RoomEnv(budget=8)
    env.reset(seed=11, theta=18)
    outcome = run_mle_episode(env, PassiveUniformOracleMLE(rng=__import__("numpy").random.default_rng(0)))
    assert 4 <= outcome.theta_hat <= 27
    assert outcome.absolute_error == abs(18 - outcome.theta_hat)
    assert env.terminal_outcome is not None
