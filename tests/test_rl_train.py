from __future__ import annotations

import numpy as np

from light_learning.rl_train import LinearSoftmax, train_reinforce


def test_reinforce_smoke_records_history() -> None:
    result = train_reinforce(budget=4, n_episodes=8, seed=0, eval_episodes=4)
    assert result["n_episodes"] == 8
    assert result["history"]
    assert result["eval_mae"] >= 0
    assert len(result["W"]) == 32
    assert result["evaluation_scope"] == "non_reportable_smoke_demo"
    assert "not PPO results" in result["note"]


def test_entropy_regularization_moves_policy_toward_higher_entropy() -> None:
    policy = LinearSoftmax(
        n_actions=2,
        n_features=1,
        rng=np.random.default_rng(0),
        lr=0.1,
        entropy_coef=1.0,
    )
    policy.W = np.asarray([[2.0], [-2.0]])
    features = np.asarray([1.0])
    probabilities = policy._pi(features)
    entropy_before = -float(np.sum(probabilities * np.log(probabilities)))

    policy.update([(features, 0, probabilities)], advantage=0.0)

    probabilities_after = policy._pi(features)
    entropy_after = -float(np.sum(probabilities_after * np.log(probabilities_after)))
    assert entropy_after > entropy_before
