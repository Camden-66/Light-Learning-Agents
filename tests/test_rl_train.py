from __future__ import annotations

from light_learning.rl_train import train_reinforce


def test_reinforce_smoke_records_history() -> None:
    result = train_reinforce(budget=4, n_episodes=8, seed=0, eval_episodes=4)
    assert result["n_episodes"] == 8
    assert result["history"]
    assert result["eval_mae"] >= 0
    assert len(result["W"]) == 32
