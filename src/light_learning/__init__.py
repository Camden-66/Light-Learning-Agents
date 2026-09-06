"""Shared infrastructure for the Light Learning Agents room benchmark."""

from .config import BUDGETS, SLOT_COUNT, RoomConfig
from .evaluator import (
    EVALUATOR_VERSION,
    aggregate_metrics,
    assert_training_separation,
    derive_policy_seed,
    evaluate_profile,
    load_episode_definitions,
    run_episode,
    run_evaluation,
    validate_profiles,
)
from .gym_env import GymRoomEnv
from .llm_agent import LLMRoomAgent, LLMRoomAgentConfig
from .ollama import ChatResponse, OllamaChatClient, OllamaError
from .emergent import EmergentBumpAgent
from .mle import PassiveUniformOracleMLE
from .ppo import PPORoomAgent, train_ppo, train_ppo_suite
from .room import RoomEnv, fallback_estimate, fallback_observation_slot
from .types import (
    AgentDecision,
    EpisodeDefinition,
    EpisodeRecord,
    Observation,
    RoomState,
    TerminalOutcome,
)

__all__ = [
    "BUDGETS",
    "EVALUATOR_VERSION",
    "SLOT_COUNT",
    "AgentDecision",
    "EpisodeDefinition",
    "EpisodeRecord",
    "EmergentBumpAgent",
    "GymRoomEnv",
    "LLMRoomAgent",
    "LLMRoomAgentConfig",
    "PassiveUniformOracleMLE",
    "PPORoomAgent",
    "Observation",
    "OllamaChatClient",
    "OllamaError",
    "ChatResponse",
    "RoomConfig",
    "RoomEnv",
    "RoomState",
    "TerminalOutcome",
    "aggregate_metrics",
    "assert_training_separation",
    "derive_policy_seed",
    "evaluate_profile",
    "fallback_estimate",
    "fallback_observation_slot",
    "load_episode_definitions",
    "run_episode",
    "run_evaluation",
    "validate_profiles",
    "train_ppo",
    "train_ppo_suite",
]
