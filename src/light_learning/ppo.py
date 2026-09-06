"""Stable-Baselines3 PPO baseline on the shared GymRoomEnv."""

from __future__ import annotations

from pathlib import Path

from .gym_env import GymRoomEnv


def train_ppo(
    budget: int,
    seed: int,
    total_timesteps: int = 20_000,
    save_dir: str | Path | None = None,
):
    from stable_baselines3 import PPO

    env = GymRoomEnv(budget=budget)
    model = PPO("MlpPolicy", env, seed=seed, verbose=0)
    model.learn(total_timesteps=total_timesteps)
    if save_dir is not None:
        path = Path(save_dir) / f"ppo_b{budget}_seed{seed}"
        path.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(path))
    return model
