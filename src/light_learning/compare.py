"""Held-out comparison: emergent world model vs basic RL vs oracle MLE."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable

from .config import BUDGETS, RoomConfig
from .emergent import (
    TRUE_SIGMA,
    EmergentBumpAgent,
    collect_labeled_episode,
    hypothesis_grid,
    pool_shape_log_prior,
)
from .evaluator import (
    assert_training_separation,
    derive_policy_seed,
    load_episode_definitions,
    profile_master_seed,
    run_episode,
)
from .mle import PassiveUniformOracleMLE
from .rl_train import train_reinforce
from .room import RoomEnv
from .types import EpisodeDefinition, EpisodeRecord, RoomState


def _mae(records: list[EpisodeRecord]) -> dict[int, float]:
    grouped: dict[int, list[int]] = defaultdict(list)
    for record in records:
        grouped[record.budget].append(record.absolute_error)
    return {
        budget: (sum(errors) / len(errors) if errors else float("nan"))
        for budget, errors in sorted(grouped.items())
    }


def _curve_rmse(predicted: list[float], theta: int, config: RoomConfig) -> float:
    se = 0.0
    for slot, value in enumerate(predicted):
        truth = config.light_probability(slot, theta)
        delta = value - truth
        se += delta * delta
    return (se / len(predicted)) ** 0.5


def _reinforce_agent(weights: list[list[float]], budget: int):
    import numpy as np

    from .ppo import room_state_vector
    from .rl_train import LinearSoftmax
    from .types import AgentDecision

    policy = LinearSoftmax(n_actions=32, n_features=66, rng=np.random.default_rng(0))
    policy.W = np.asarray(weights, dtype=np.float64)

    class _Agent:
        agent_id = "linear-softmax-reinforce"
        agent_version = "0.1.0"
        condition = "basic_rl_reinforce"

        def start_episode(self, state: RoomState) -> None:
            del state

        def act(self, state: RoomState) -> AgentDecision:
            vector = room_state_vector(state, expected_budget=budget)
            action = policy.greedy(vector.astype("float64"))
            return AgentDecision(action=int(action))

        def run_metadata(self) -> dict[str, Any]:
            return {"algo": "reinforce", "budget": budget, "reportable": False}

    return _Agent()


def _eval_cells(
    factory: Callable[..., Any],
    *,
    heldout: dict[int, tuple[EpisodeDefinition, ...]],
    budgets: tuple[int, ...],
    profile: str,
    master_seed: int,
    accepts_policy_seed: bool,
) -> list[EpisodeRecord]:
    records: list[EpisodeRecord] = []
    for budget in budgets:
        for episode in heldout[budget]:
            if accepts_policy_seed:
                probe = factory(policy_seed=0)
                policy_seed = derive_policy_seed(
                    master_seed=master_seed,
                    agent_version=probe.agent_version,
                    condition=probe.condition,
                    budget=budget,
                    episode_id=episode.episode_id,
                )
                agent = factory(policy_seed=policy_seed)
            else:
                agent = factory()
                policy_seed = None
            records.append(
                run_episode(
                    agent,
                    episode,
                    budget,
                    profile=profile,
                    policy_seed=policy_seed,
                )
            )
    return records


def summarize_emergent_pattern(
    records: list[EpisodeRecord],
    config: RoomConfig | None = None,
) -> dict[str, float]:
    """How well terminal MAP bumps reconstruct the hidden lamp."""

    config = config or RoomConfig()
    sigmas: list[float] = []
    rmse: list[float] = []
    masses: list[float] = []
    for record in records:
        meta = dict(record.trace[-1].get("metadata") or {})
        if "sigma" not in meta:
            continue
        sigmas.append(float(meta["sigma"]))
        masses.append(float(meta.get("sigma_mass_at_3", 0.0)))
        curve = meta.get("curve")
        if isinstance(curve, list) and curve:
            rmse.append(_curve_rmse([float(x) for x in curve], record.theta, config))
    return {
        "mean_sigma": (sum(sigmas) / len(sigmas)) if sigmas else float("nan"),
        "abs_sigma_error": (
            sum(abs(s - TRUE_SIGMA) for s in sigmas) / len(sigmas) if sigmas else float("nan")
        ),
        "mean_sigma_mass_at_3": (sum(masses) / len(masses)) if masses else float("nan"),
        "mean_curve_rmse": (sum(rmse) / len(rmse)) if rmse else float("nan"),
    }


def fit_pooled_prior(
    *,
    n_episodes: int,
    budget: int,
    seed: int,
    heldout: dict[int, tuple[EpisodeDefinition, ...]],
    config: RoomConfig,
) -> list[float]:
    held_seeds = {ep.episode_seed for rows in heldout.values() for ep in rows}
    labeled: list[tuple[list[tuple[int, int]], int]] = []
    used_seeds: list[int] = []
    cursor = 2_000_000 + seed * 10_000 + budget
    while len(labeled) < n_episodes:
        if cursor in held_seeds:
            cursor += 1
            continue
        agent = EmergentBumpAgent(policy_seed=seed + len(labeled), config=config)
        history, theta, _outcome = collect_labeled_episode(
            lambda: RoomEnv(config, budget=budget),
            agent,
            seed=cursor,
        )
        labeled.append((history, theta))
        used_seeds.append(cursor)
        cursor += 1
    train_ids = [f"emergent-pool-{budget}-{i}" for i in range(n_episodes)]
    assert_training_separation(train_ids, used_seeds, heldout)
    return pool_shape_log_prior(labeled, hypothesis_grid(config))


def run_organic_comparison(
    *,
    profile: str = "pilot",
    reinforce_episodes: int = 200,
    pool_episodes: int = 16,
    seed: int = 0,
    config: RoomConfig | None = None,
    budgets: tuple[int, ...] | None = None,
) -> dict[str, Any]:
    """Evaluate world-model vs REINFORCE vs oracle MLE on the same profile."""

    config = config or RoomConfig()
    budgets = budgets or BUDGETS
    version, heldout = load_episode_definitions(profile, config=config)
    master = profile_master_seed(profile)
    del version
    report: dict[str, Any] = {
        "profile": profile,
        "reportable": False,
        "evaluation_label": "non_reportable_organic_comparison",
        "agents": {},
    }

    def mle_factory(*, policy_seed: int):
        return PassiveUniformOracleMLE(config, query_seed=policy_seed)

    mle_records = _eval_cells(
        mle_factory,
        heldout=heldout,
        budgets=budgets,
        profile=profile,
        master_seed=master,
        accepts_policy_seed=True,
    )
    report["agents"]["oracle_mle"] = {
        "mae": _mae(mle_records),
        "pattern": None,
        "note": (
            "Knows the canonical formula but queries passively and uniformly. "
            "Not a performance ceiling: roughly a third to a half of its error "
            "is query strategy, so an adaptive agent can beat it without "
            "knowing more. Different information regime, not a ladder rung."
        ),
    }

    def emergent_factory(*, policy_seed: int):
        return EmergentBumpAgent(policy_seed=policy_seed, config=config)

    in_ep = _eval_cells(
        emergent_factory,
        heldout=heldout,
        budgets=budgets,
        profile=profile,
        master_seed=master,
        accepts_policy_seed=True,
    )
    report["agents"]["emergent_in_episode"] = {
        "mae": _mae(in_ep),
        "pattern": summarize_emergent_pattern(in_ep, config),
        "note": "Infers (mu, sigma, a, b) from the current episode only.",
    }

    pooled_records: list[EpisodeRecord] = []
    for budget in budgets:
        prior = fit_pooled_prior(
            n_episodes=pool_episodes,
            budget=budget,
            seed=seed,
            heldout=heldout,
            config=config,
        )

        def pooled_factory(*, policy_seed: int, _prior=prior):
            return EmergentBumpAgent(
                policy_seed=policy_seed,
                config=config,
                shape_log_prior=_prior,
                condition="emergent_pooled_shape",
            )

        pooled_records.extend(
            _eval_cells(
                pooled_factory,
                heldout=heldout,
                budgets=(budget,),
                profile=profile,
                master_seed=master,
                accepts_policy_seed=True,
            )
        )
    report["agents"]["emergent_pooled"] = {
        "mae": _mae(pooled_records),
        "pattern": summarize_emergent_pattern(pooled_records, config),
        "note": (
            "After training episodes, keeps the learned lamp shape and infers "
            "a new peak on held-out rooms. This is the emergence check."
        ),
        "pool_episodes_per_budget": pool_episodes,
    }

    rl_records: list[EpisodeRecord] = []
    for budget in budgets:
        held_seeds = {ep.episode_seed for rows in heldout.values() for ep in rows}
        train_ids = [f"reinforce-train-{budget}-{i}" for i in range(reinforce_episodes)]
        train_seeds = []
        cursor = 1_000_000 + seed * 10_000 + budget
        while len(train_seeds) < reinforce_episodes:
            if cursor not in held_seeds:
                train_seeds.append(cursor)
            cursor += 1
        assert_training_separation(train_ids, train_seeds, heldout)
        weights = train_reinforce(
            budget=budget, n_episodes=reinforce_episodes, seed=seed
        )["W"]

        def rl_factory(_weights=weights, _budget=budget):
            return _reinforce_agent(_weights, _budget)

        rl_records.extend(
            _eval_cells(
                rl_factory,
                heldout=heldout,
                budgets=(budget,),
                profile=profile,
                master_seed=master,
                accepts_policy_seed=False,
            )
        )
    report["agents"]["basic_rl_reinforce"] = {
        "mae": _mae(rl_records),
        "pattern": None,
        "note": "Linear softmax REINFORCE; a look/guess policy, not a lamp model.",
        "train_episodes_per_budget": reinforce_episodes,
    }
    return report
