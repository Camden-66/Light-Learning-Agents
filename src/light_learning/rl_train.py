"""On-policy REINFORCE for the shared GymRoomEnv (same contract as PPO)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .gym_env import GymRoomEnv


@dataclass
class LinearSoftmax:
    n_actions: int
    n_features: int
    rng: np.random.Generator
    lr: float = 0.06
    entropy_coef: float = 0.02

    def __post_init__(self) -> None:
        self.W = 0.01 * self.rng.normal(size=(self.n_actions, self.n_features))

    def _pi(self, phi: np.ndarray) -> np.ndarray:
        logits = self.W @ phi
        logits = logits - np.max(logits)
        exp = np.exp(logits)
        return exp / np.sum(exp)

    def sample(self, phi: np.ndarray) -> tuple[int, np.ndarray]:
        pi = self._pi(phi)
        action = int(self.rng.choice(self.n_actions, p=pi))
        return action, pi

    def greedy(self, phi: np.ndarray) -> int:
        return int(np.argmax(self._pi(phi)))

    def update(self, trajectory: list[tuple[np.ndarray, int, np.ndarray]], advantage: float) -> None:
        if not trajectory:
            return
        g = np.zeros_like(self.W)
        for phi, action, pi in trajectory:
            indic = np.zeros(self.n_actions)
            indic[action] = 1.0
            g += advantage * np.outer(indic - pi, phi)
        g -= self.entropy_coef * self.W
        self.W += self.lr * g / len(trajectory)
        np.clip(self.W, -8.0, 8.0, out=self.W)


def train_reinforce(
    budget: int = 8,
    n_episodes: int = 250,
    seed: int = 0,
    eval_episodes: int = 20,
) -> dict:
    rng = np.random.default_rng(seed)
    env = GymRoomEnv(budget=budget)
    policy = LinearSoftmax(n_actions=32, n_features=66, rng=rng)
    baseline = 0.0
    momentum = 0.9
    history: list[dict] = []
    window: list[int] = []
    for ep in range(n_episodes):
        env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        result = _rollout(env, policy, greedy=False)
        advantage = result["reward"] - baseline
        baseline = momentum * baseline + (1.0 - momentum) * result["reward"]
        policy.update(result["trajectory"], advantage)
        window.append(result["absolute_error"])
        if len(window) > 25:
            window.pop(0)
        if ep % max(1, n_episodes // 40) == 0 or ep == n_episodes - 1:
            history.append(
                {
                    "episode": ep,
                    "error": result["absolute_error"],
                    "reward": result["reward"],
                    "mae_window": float(np.mean(window)),
                }
            )
    eval_errors = []
    for _ in range(eval_episodes):
        env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        eval_errors.append(_rollout(env, policy, greedy=True)["absolute_error"])
    return {
        "budget": budget,
        "n_episodes": n_episodes,
        "history": history,
        "eval_mae": float(np.mean(eval_errors)),
        "eval_hit_within_one": float(np.mean([e <= 1 for e in eval_errors])),
        "W": policy.W.tolist(),
        "algo": "reinforce",
        "note": "Linear softmax REINFORCE on GymRoomEnv; same observation/reward as PPO.",
    }


def _rollout(env: GymRoomEnv, policy: LinearSoftmax, *, greedy: bool) -> dict:
    obs = env._vector(env._state) if env._state is not None else None
    if obs is None:
        obs, _ = env.reset()
    trajectory: list[tuple[np.ndarray, int, np.ndarray]] = []
    terminated = False
    info: dict = {}
    reward = 0.0
    while not terminated:
        phi = np.asarray(obs, dtype=np.float64)
        if greedy:
            action = policy.greedy(phi)
            pi = policy._pi(phi)
        else:
            action, pi = policy.sample(phi)
        if not greedy:
            trajectory.append((phi, action, pi))
        obs, reward, terminated, _, info = env.step(action)
    return {
        "trajectory": trajectory,
        "reward": float(reward),
        "absolute_error": int(info["absolute_error"]),
        "theta": int(info["theta"]),
        "theta_hat": int(info["theta_hat"]),
    }


def play_policy(
    W: list[list[float]],
    *,
    budget: int,
    seed: int,
    theta: int,
) -> dict:
    policy = LinearSoftmax(n_actions=32, n_features=66, rng=np.random.default_rng(0))
    policy.W = np.asarray(W, dtype=np.float64)
    env = GymRoomEnv(budget=budget)
    obs, _ = env.reset(seed=seed, options={"theta": theta})
    trace: list[dict] = []
    terminated = False
    info: dict = {}
    reward = 0.0
    while not terminated:
        action = policy.greedy(np.asarray(obs, dtype=np.float64))
        obs, reward, terminated, _, info = env.step(action)
        if terminated:
            trace.append({"phase": "terminal", "action": action, "light_on": None})
        else:
            trace.append(
                {
                    "phase": "observe",
                    "action": action,
                    "light_on": int(info["light_on"]),
                }
            )
    return {
        "condition": "rl_reinforce",
        "theta": int(info["theta"]),
        "theta_hat": int(info["theta_hat"]),
        "absolute_error": int(info["absolute_error"]),
        "reward": float(reward),
        "trace": trace,
    }
