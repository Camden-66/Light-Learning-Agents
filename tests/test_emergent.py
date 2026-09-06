from __future__ import annotations

import pytest

from light_learning.config import RoomConfig
from light_learning.emergent import (
    SIGMA_TOLERANCE,
    TRUE_SIGMA,
    EmergentBumpAgent,
    cached_grid,
    history_pairs,
    hypothesis_grid,
    map_hypothesis,
    pool_shape_log_prior,
    shape_posterior_mass,
    sigma_mass_chance_level,
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


def test_hypothesis_grid_stays_inside_the_theta_support() -> None:
    """The answer space is stated up front, so the grid must not exceed it.

    Guessing outside [theta_min, theta_max] is an answer-space handicap, not a
    failure of model discovery -- the same confound the LLM prompt conditions
    were fixed for.
    """

    config = RoomConfig()
    grid = hypothesis_grid(config)
    candidates = set(config.theta_candidates)

    assert {hyp.mu for hyp in grid} == candidates
    assert all(hyp.mu in candidates for hyp in grid)


def test_emergent_estimates_never_land_outside_the_theta_support() -> None:
    config = RoomConfig()
    for offset in range(12):
        env = RoomEnv(budget=8)
        state = env.reset(seed=9100 + offset)
        agent = EmergentBumpAgent(policy_seed=offset, config=config)
        agent.start_episode(state)
        while state.phase == "observe":
            state = env.observe(agent.act(state).action)
        outcome = env.estimate(agent.act(state).action)
        assert outcome.theta_hat in config.theta_candidates


def test_looks_are_not_restricted_to_the_theta_support() -> None:
    """Only the terminal estimate is constrained; observation stays full-range."""

    config = RoomConfig()
    looked: set[int] = set()
    for offset in range(12):
        env = RoomEnv(budget=8)
        state = env.reset(seed=4242 + offset)
        agent = EmergentBumpAgent(policy_seed=offset, config=config)
        agent.start_episode(state)
        while state.phase == "observe":
            action = agent.act(state).action
            looked.add(action)
            state = env.observe(action)

    assert any(slot not in config.theta_candidates for slot in looked)
    assert all(0 <= slot < config.slot_count for slot in looked)


def test_sigma_mass_chance_level_matches_an_unlearned_posterior() -> None:
    hyps, table = cached_grid()
    chance = sigma_mass_chance_level(hyps)
    unlearned = shape_posterior_mass(
        [], hyps, sigma=TRUE_SIGMA, atol=SIGMA_TOLERANCE, table=table
    )

    assert chance == pytest.approx(unlearned, abs=1e-9)
    assert 0.0 < chance < 1.0


def test_pooled_runs_are_distinguishable_in_run_metadata() -> None:
    """Two different training pools must not produce identical manifests."""

    hyps, _ = cached_grid()
    prior_a = pool_shape_log_prior(
        [(_dense_history(theta, seed=4100 + theta), theta) for theta in (8, 12)], hyps
    )
    prior_b = pool_shape_log_prior(
        [(_dense_history(theta, seed=4200 + theta), theta) for theta in (20, 24)], hyps
    )

    meta_a = EmergentBumpAgent(policy_seed=1, shape_log_prior=prior_a).run_metadata()
    meta_b = EmergentBumpAgent(policy_seed=1, shape_log_prior=prior_b).run_metadata()
    meta_none = EmergentBumpAgent(policy_seed=1).run_metadata()

    assert meta_a["shape_prior_digest"] != meta_b["shape_prior_digest"]
    assert meta_a["configuration_digest"] != meta_b["configuration_digest"]
    assert meta_none["shape_prior_digest"] is None
    assert meta_none["configuration_digest"].startswith("sha256:")
    assert meta_a["theta_candidates"] == list(RoomConfig().theta_candidates)
