from __future__ import annotations

from light_learning.config import RoomConfig
from light_learning.emergent import (
    TRUE_SIGMA,
    EmergentBumpAgent,
    cached_grid,
    history_pairs,
    map_hypothesis,
    pool_shape_log_prior,
    shape_posterior_mass,
)
from light_learning.evaluator import run_episode
from light_learning.room import RoomEnv
from light_learning.types import EpisodeDefinition


def _dense_history(theta: int, seed: int) -> list[tuple[int, int]]:
    env = RoomEnv(budget=32)
    env.reset(seed=seed, theta=theta)
    for slot in range(32):
        env.observe(slot)
    return history_pairs(env.state.history)


def test_bump_family_recovers_true_width_from_a_full_scan() -> None:
    history = _dense_history(theta=16, seed=2026)
    hyps, table = cached_grid()
    fitted = map_hypothesis(history, hyps, table=table)
    mass = shape_posterior_mass(history, hyps, sigma=TRUE_SIGMA, table=table)

    assert fitted.mu == 16
    assert fitted.background == 0.05
    assert fitted.amplitude == 0.90
    assert fitted.sigma in {2.5, 3.0, 3.5}
    assert mass > shape_posterior_mass(history, hyps, sigma=8.0, table=table)


def test_pooled_shape_prior_moves_mass_onto_true_width() -> None:
    hyps, table = cached_grid()
    labeled = [
        (_dense_history(theta, seed=3000 + theta), theta) for theta in (8, 12, 20, 24)
    ]
    prior = pool_shape_log_prior(labeled, hyps)
    empty_mass = shape_posterior_mass([], hyps, sigma=TRUE_SIGMA, table=table)
    pooled_mass = shape_posterior_mass(
        [], hyps, sigma=TRUE_SIGMA, shape_log_prior=prior, table=table
    )
    assert pooled_mass > empty_mass
    assert pooled_mass > 0.15


def test_emergent_agent_completes_an_evaluator_episode() -> None:
    agent = EmergentBumpAgent(policy_seed=7)
    record = run_episode(
        agent,
        EpisodeDefinition(episode_id="emergent-smoke", theta=18, episode_seed=11),
        budget=4,
    )
    assert record.completed is True
    assert record.protocol_failure is False
    assert 0 <= record.theta_hat < 32
    assert record.trace[-1]["metadata"]["sigma"] in {1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 6.0, 8.0}


def test_in_episode_agent_does_not_read_canonical_constants() -> None:
    meta = EmergentBumpAgent(policy_seed=1).run_metadata()
    assert meta["canonical_formula_disclosed"] is False
    config = RoomConfig()
    assert config.exponent_denominator == 18.0
    assert 18.0 not in meta["sigmas"]
