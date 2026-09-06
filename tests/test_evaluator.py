"""Acceptance tests for the evaluator contract in docs/evaluator-build-spec.md.

Tests marked "regression" cover defects found while integrating the evaluator
from the ``add-eval`` branch; each of them fails against that implementation.
"""

from __future__ import annotations

import inspect
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from light_learning.config import BUDGETS, SLOT_COUNT, RoomConfig
from light_learning.evaluator import (
    GymPolicyAdapter,
    aggregate_metrics,
    assert_training_separation,
    derive_policy_seed,
    evaluate_profile,
    load_episode_definitions,
    profile_master_seed,
    read_records,
    run_episode,
    run_evaluation,
    write_records,
)
from light_learning.mle import PassiveUniformOracleMLE, mle_theta_hat
from light_learning.ppo import PPORoomAgent
from light_learning.room import RoomEnv, fallback_estimate
from light_learning.types import (
    AgentDecision,
    EpisodeDefinition,
    EpisodeRecord,
    Observation,
    RoomState,
)

CONFIG = RoomConfig()


class ConformingAgent:
    """A minimal fixed policy that satisfies the shared agent protocol."""

    agent_id = "agent-1"
    agent_version = "agent-v1"
    condition = "baseline"

    def __init__(self, query_slot: int = 7, estimate: int = 12) -> None:
        self.query_slot = query_slot
        self.estimate = estimate
        self.states: list[RoomState] = []

    def start_episode(self, state: RoomState) -> None:
        self.states.append(state)

    def act(self, state: RoomState) -> AgentDecision:
        self.states.append(state)
        action = self.estimate if state.phase == "estimate" else self.query_slot
        return AgentDecision(
            action=action,
            raw_response=str(action),
            rationale="fixed test policy",
            latency_seconds=0.001,
            metadata={"test": True},
        )

    def run_metadata(self) -> dict[str, object]:
        return {"model": "test"}


def make_record(theta, theta_hat, *, protocol=False, fallback=False, seed=None):
    budget = 4
    trace = [
        {"step": step, "phase": "observe", "action": 0, "light_on": False}
        for step in range(budget)
    ]
    trace.append({"step": budget, "phase": "estimate", "action": theta_hat})
    return EpisodeRecord(
        agent_id="agent-1",
        agent_version="agent-v1",
        condition="baseline",
        budget=budget,
        episode_id=f"e-{theta}-{theta_hat}-{seed}",
        episode_seed=1,
        theta=theta,
        theta_hat=theta_hat,
        absolute_error=abs(theta - theta_hat),
        reward=-abs(theta - theta_hat) / (SLOT_COUNT - 1),
        trace=tuple(trace),
        completed=True,
        protocol_failure=protocol,
        fallback_used=fallback,
        latency_seconds=0.5,
        metadata={"test": True},
        profile="full",
        profile_version="room-v1-full-v1",
        configuration_digest="digest",
        training_seed=seed,
    )


def mle_factory(*, policy_seed: int) -> PassiveUniformOracleMLE:
    return PassiveUniformOracleMLE(query_seed=policy_seed)


# ---------------------------------------------------------------------------
# 1. Deterministic episode definitions and theta stratification
# ---------------------------------------------------------------------------


def test_episode_definitions_are_deterministic():
    for profile in ("pilot", "full"):
        first = load_episode_definitions(profile)
        second = load_episode_definitions(profile)
        assert first == second
        assert tuple(first[1]) == BUDGETS


def test_full_profile_covers_every_theta_evenly():
    _, full = load_episode_definitions("full")
    for budget in BUDGETS:
        counts = Counter(episode.theta for episode in full[budget])
        assert len(full[budget]) == 100
        assert set(counts) == set(CONFIG.theta_candidates)
        assert max(counts.values()) - min(counts.values()) <= 1


def test_pilot_profile_spans_the_whole_theta_range():
    """A 10-episode pilot must still probe the ends of the hidden range."""

    _, pilot = load_episode_definitions("pilot")
    for budget in BUDGETS:
        thetas = sorted(episode.theta for episode in pilot[budget])
        assert len(pilot[budget]) == 10
        assert len(set(thetas)) == 10
        assert thetas[0] == CONFIG.theta_min
        assert thetas[-1] == CONFIG.theta_max
        # Evenly spread, not clustered at one end of the range.
        gaps = [b - a for a, b in zip(thetas, thetas[1:])]
        assert max(gaps) <= 3


def test_pilot_and_full_definitions_are_disjoint():
    _, pilot = load_episode_definitions("pilot")
    _, full = load_episode_definitions("full")
    pilot_ids = {e.episode_id for cases in pilot.values() for e in cases}
    full_ids = {e.episode_id for cases in full.values() for e in cases}
    pilot_seeds = {e.episode_seed for cases in pilot.values() for e in cases}
    full_seeds = {e.episode_seed for cases in full.values() for e in cases}
    assert pilot_ids.isdisjoint(full_ids)
    assert pilot_seeds.isdisjoint(full_seeds)


def test_training_overlap_is_rejected():
    _, definitions = load_episode_definitions("full")
    episode = definitions[4][0]
    with pytest.raises(ValueError):
        assert_training_separation([episode.episode_id], [], definitions)
    with pytest.raises(ValueError):
        assert_training_separation([], [episode.episode_seed], definitions)
    assert_training_separation(["train-1"], [-1], definitions) is None


# ---------------------------------------------------------------------------
# 2. Deterministic agent-policy seeding, independent of hidden room seeds
# ---------------------------------------------------------------------------


def test_policy_seed_never_sees_hidden_room_inputs():
    parameters = set(inspect.signature(derive_policy_seed).parameters)
    assert parameters == {
        "master_seed",
        "agent_version",
        "condition",
        "budget",
        "episode_id",
    }
    assert "theta" not in parameters and "episode_seed" not in parameters


def test_policy_seed_is_deterministic_and_input_sensitive():
    base = dict(
        master_seed=20260906,
        agent_version="agent-v1",
        condition="baseline",
        budget=8,
        episode_id="ep-000",
    )
    assert derive_policy_seed(**base) == derive_policy_seed(**base)
    seeds = {derive_policy_seed(**base)}
    for field, value in (
        ("master_seed", 1),
        ("agent_version", "agent-v2"),
        ("condition", "other"),
        ("budget", 16),
        ("episode_id", "ep-001"),
    ):
        seeds.add(derive_policy_seed(**{**base, field: value}))
    assert len(seeds) == 6


def test_recorded_policy_seed_matches_independent_derivation():
    records = evaluate_profile(
        mle_factory,
        profile="pilot",
        agent_id="mle",
        agent_version="0.2.0",
        condition="oracle_mle",
    )
    master_seed = profile_master_seed("pilot")
    for record in records:
        assert record.metadata["policy_seed"] == derive_policy_seed(
            master_seed=master_seed,
            agent_version="0.2.0",
            condition="oracle_mle",
            budget=record.budget,
            episode_id=record.episode_id,
        )
        # Never reuse the hidden room seed as policy randomness.
        assert record.metadata["policy_seed"] != record.episode_seed


def test_policy_seeded_factory_requires_explicit_identity():
    with pytest.raises(ValueError, match="policy_seed"):
        evaluate_profile(mle_factory, profile="pilot", agent_id="mle")


# ---------------------------------------------------------------------------
# 3. Room outcomes are keyed by (seed, slot, repeat), not by query order
# ---------------------------------------------------------------------------


def test_outcomes_are_independent_of_query_order():
    forward_order = [0, 5, 5, 9, 20, 20, 31, 5]
    reverse_order = [31, 20, 5, 0, 5, 20, 5, 9]

    def draws(order):
        env = RoomEnv(CONFIG, budget=8)
        env.reset(seed=555, theta=15)
        result = {}
        for slot in order:
            observation = env.observe(slot).history[-1]
            result[(observation.time_slot, observation.repeat_index)] = observation.light_on
        return result

    forward = draws(forward_order)
    assert len(forward) == 8
    assert forward == draws(reverse_order)


# ---------------------------------------------------------------------------
# 4. No hidden information leaks before the terminal estimate
# ---------------------------------------------------------------------------


def test_public_room_state_exposes_only_public_fields():
    assert set(RoomState.__dataclass_fields__) == {
        "history",
        "budget",
        "remaining_observations",
    }


def test_no_preterminal_state_or_trace_entry_leaks_theta_or_score():
    agent = ConformingAgent()
    record = run_episode(
        agent,
        EpisodeDefinition("leak-check", 21, 4242),
        8,
        agent_id="agent-1",
        agent_version="agent-v1",
        condition="baseline",
    )
    hidden = {"theta", "reward", "absolute_error", "likelihood", "score"}
    for event in record.trace[:-1]:
        assert hidden.isdisjoint(event)
        assert hidden.isdisjoint(event["metadata"])
    assert record.trace[-1]["theta"] == 21
    for state in agent.states:
        assert not hasattr(state, "theta")
        assert not hasattr(state, "episode_seed")
        for observation in state.history:
            assert isinstance(observation, Observation)
    assert set(Observation.__dataclass_fields__) == {
        "time_slot",
        "light_on",
        "repeat_index",
    }


# ---------------------------------------------------------------------------
# 5. Exactly B observations plus one terminal estimate
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("budget", BUDGETS)
def test_exact_observation_count_for_every_budget(budget):
    record = run_episode(
        ConformingAgent(),
        EpisodeDefinition(f"episode-{budget}", 12, budget),
        budget,
        agent_id="agent-1",
        agent_version="agent-v1",
        condition="baseline",
    )
    assert record.budget == budget
    assert len(record.trace) == budget + 1
    assert sum(event["phase"] == "observe" for event in record.trace) == budget
    assert record.trace[-1]["phase"] == "estimate"
    assert record.completed


# ---------------------------------------------------------------------------
# 6. Broken agents complete through the documented fallback and stay scored
# ---------------------------------------------------------------------------


class RaisingAgent(ConformingAgent):
    def act(self, state):
        raise RuntimeError("policy exploded")


class WrongTypeAgent(ConformingAgent):
    def act(self, state):
        return "not an AgentDecision"


class OutOfRangeAgent(ConformingAgent):
    def act(self, state):
        return AgentDecision(action=SLOT_COUNT + 5)


class BadStartAgent(ConformingAgent):
    def start_episode(self, state):
        raise RuntimeError("cannot start")


@pytest.mark.parametrize(
    "agent_class", [RaisingAgent, WrongTypeAgent, OutOfRangeAgent, BadStartAgent]
)
def test_invalid_agents_complete_via_fallback(agent_class):
    record = run_episode(
        agent_class(),
        EpisodeDefinition("invalid-1", 10, 99),
        4,
        agent_id="invalid",
        agent_version="invalid-v1",
        condition="test",
    )
    assert record.completed
    assert record.protocol_failure
    assert record.fallback_used
    # Observation fallback: smallest least-visited slot, so a clean 0,1,2,3 sweep.
    assert [event["action"] for event in record.trace[:4]] == [0, 1, 2, 3]
    assert 0 <= record.theta_hat < SLOT_COUNT


def test_construction_failure_is_scored_not_dropped():
    def exploding_factory():
        raise RuntimeError("no model available")

    records = evaluate_profile(
        exploding_factory,
        profile="pilot",
        agent_id="broken",
        agent_version="broken-v1",
        condition="test",
    )
    assert len(records) == 40
    assert all(record.completed for record in records)
    assert all(record.protocol_failure for record in records)
    reports = aggregate_metrics(records)
    assert all(report["protocol_failure_rate"] == 1.0 for report in reports)
    assert sum(report["episode_count"] for report in reports) == 40


def test_terminal_fallback_matches_the_specified_rule():
    # Highest empirical on-rate, ties broken toward the smallest slot.
    tied = RoomState(
        (
            Observation(5, False, 0),
            Observation(5, True, 1),
            Observation(2, True, 0),
            Observation(2, False, 1),
        ),
        4,
        0,
    )
    assert fallback_estimate(tied, CONFIG) == 2

    clear = RoomState(
        (Observation(9, True, 0), Observation(3, False, 0)),
        4,
        2,
    )
    assert fallback_estimate(clear, CONFIG) == 9

    # Empty history falls back to the middle slot.
    assert fallback_estimate(RoomState((), 4, 4), CONFIG) == 16 == SLOT_COUNT // 2


# ---------------------------------------------------------------------------
# 7. Metrics over hand-built records
# ---------------------------------------------------------------------------


def test_metrics_cover_every_required_quantity():
    records = [
        make_record(4, 4, seed=1),
        make_record(4, 5, seed=2),
        make_record(20, 10, protocol=True, fallback=True, seed=1),
        make_record(20, 11, protocol=False, fallback=True, seed=2),
    ]
    reports = aggregate_metrics(records)
    aggregate = next(r for r in reports if r["scope"] == "aggregate")
    assert aggregate["episode_count"] == 4
    assert aggregate["mae"] == pytest.approx((0 + 1 + 10 + 9) / 4)
    assert aggregate["median_absolute_error"] == pytest.approx(5.0)
    assert aggregate["hit_within_one_slot_rate"] == 0.5
    assert aggregate["protocol_failure_rate"] == 0.25
    assert aggregate["fallback_rate"] == 0.5
    assert aggregate["mean_latency_seconds"] == pytest.approx(0.5)
    assert aggregate["worst_theta"] == 20
    assert aggregate["worst_theta_slice_mae"] == pytest.approx(9.5)
    assert aggregate["worst_theta_group_size"] == 2
    assert {r["training_seed"] for r in reports if r["scope"] == "training_seed"} == {1, 2}


def test_worst_theta_ties_break_toward_the_smallest_theta():
    records = [make_record(20, 25), make_record(7, 12), make_record(7, 2)]
    aggregate = next(
        r for r in aggregate_metrics(records) if r["scope"] == "aggregate"
    )
    assert aggregate["worst_theta_slice_mae"] == pytest.approx(5.0)
    assert aggregate["worst_theta"] == 7


def test_pilot_worst_theta_slice_is_labelled_preliminary():
    records = evaluate_profile(
        ConformingAgent,
        profile="pilot",
        agent_id="agent-1",
        agent_version="agent-v1",
        condition="baseline",
    )
    assert all(r["worst_theta_slice_preliminary"] for r in aggregate_metrics(records))


# ---------------------------------------------------------------------------
# 8. JSONL, summary, manifest, and resume
# ---------------------------------------------------------------------------


def test_records_round_trip_through_jsonl(tmp_path):
    records = evaluate_profile(
        ConformingAgent,
        profile="pilot",
        agent_id="agent-1",
        agent_version="agent-v1",
        condition="baseline",
    )
    assert len(records) == 40
    path = tmp_path / "records.jsonl"
    write_records(path, records)
    assert read_records([path]) == records


def test_run_artifacts_and_resume_by_run_id(tmp_path):
    calls = {"count": 0}

    def factory():
        calls["count"] += 1
        return ConformingAgent()

    kwargs = dict(
        output_dir=tmp_path,
        run_id="resume-test",
        profile="pilot",
        agent_id="agent-1",
        agent_version="agent-v1",
        condition="baseline",
    )
    first = run_evaluation(factory, **kwargs)
    first_calls = calls["count"]
    records_path = tmp_path / "resume-test" / "records.jsonl"
    before = records_path.read_bytes()

    second = run_evaluation(factory, **kwargs)
    assert len(first) == len(second) == 40
    assert calls["count"] == first_calls, "resume must not re-run completed episodes"
    assert records_path.read_bytes() == before

    manifest = json.loads((tmp_path / "resume-test" / "manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert manifest["completed_episode_count"] == 40
    assert manifest["expected_episode_count"] == 40
    assert manifest["profile_version"] == "room-v1-pilot-v3"
    assert manifest["profile_status"] == "preliminary"
    assert manifest["master_seed"] == profile_master_seed("pilot")
    assert manifest["room_config"]["slot_count"] == SLOT_COUNT

    summary = json.loads((tmp_path / "resume-test" / "summary.json").read_text())
    assert {report["budget"] for report in summary} == set(BUDGETS)
    json.dumps(summary)  # summary must stay JSON-serializable


def test_resume_rejects_a_mismatched_manifest(tmp_path):
    kwargs = dict(
        output_dir=tmp_path,
        run_id="mismatch",
        agent_id="agent-1",
        agent_version="agent-v1",
        condition="baseline",
    )
    run_evaluation(ConformingAgent, profile="pilot", **kwargs)
    with pytest.raises(ValueError, match="manifest mismatch"):
        run_evaluation(ConformingAgent, profile="full", **kwargs)


def test_resume_rejects_duplicate_records(tmp_path):
    run_dir = tmp_path / "dupes"
    run_evaluation(
        ConformingAgent,
        output_dir=tmp_path,
        run_id="dupes",
        profile="pilot",
        agent_id="agent-1",
        agent_version="agent-v1",
        condition="baseline",
    )
    records_path = run_dir / "records.jsonl"
    lines = records_path.read_text().splitlines(keepends=True)
    records_path.write_text("".join(lines + [lines[0]]))
    with pytest.raises(ValueError, match="duplicate"):
        run_evaluation(
            ConformingAgent,
            output_dir=tmp_path,
            run_id="dupes",
            profile="pilot",
            agent_id="agent-1",
            agent_version="agent-v1",
            condition="baseline",
        )


# ---------------------------------------------------------------------------
# 9. Regression: the oracle MLE baseline runs for real, not through fallback
# ---------------------------------------------------------------------------


def test_oracle_mle_baseline_runs_without_protocol_failure():
    records = evaluate_profile(
        mle_factory,
        profile="pilot",
        agent_id="passive-uniform-oracle-likelihood-mle",
        agent_version="0.2.0",
        condition="oracle_mle",
    )
    assert len(records) == 40
    assert not any(record.protocol_failure for record in records)
    assert not any(record.fallback_used for record in records)


def test_oracle_mle_actions_follow_its_own_query_schedule():
    records = evaluate_profile(
        mle_factory,
        profile="pilot",
        agent_id="passive-uniform-oracle-likelihood-mle",
        agent_version="0.2.0",
        condition="oracle_mle",
    )
    for record in records:
        agent = PassiveUniformOracleMLE(query_seed=record.metadata["policy_seed"])
        expected = agent.query_schedule(record.budget)
        assert tuple(e["action"] for e in record.trace[:-1]) == expected
        history = [
            (event["time_slot"], int(event["light_on"])) for event in record.trace[:-1]
        ]
        assert record.theta_hat == mle_theta_hat(history, CONFIG)


def test_oracle_mle_beats_the_fallback_policy_and_improves_with_budget():
    mle = evaluate_profile(
        mle_factory,
        profile="pilot",
        agent_id="mle",
        agent_version="0.2.0",
        condition="oracle_mle",
    )
    broken = evaluate_profile(
        RaisingAgent,
        profile="pilot",
        agent_id="broken",
        agent_version="broken-v1",
        condition="test",
    )

    def mae(records, budget):
        chosen = [r.absolute_error for r in records if r.budget == budget]
        return sum(chosen) / len(chosen)

    assert mae(mle, 32) < mae(mle, 4), "more observations must help the oracle MLE"
    for budget in BUDGETS:
        assert mae(mle, budget) < mae(broken, budget)


# ---------------------------------------------------------------------------
# 10. Regression: records replay exactly against the canonical room
# ---------------------------------------------------------------------------


def test_trace_replays_exactly_against_the_canonical_room():
    records = evaluate_profile(
        mle_factory,
        profile="pilot",
        agent_id="mle",
        agent_version="0.2.0",
        condition="oracle_mle",
    )
    for record in records[:8]:
        env = RoomEnv(CONFIG, budget=record.budget)
        env.reset(seed=record.episode_seed, theta=record.theta)
        for event in record.trace[:-1]:
            observation = env.observe(event["action"]).history[-1]
            assert observation.light_on == event["light_on"]
            assert observation.repeat_index == event["repeat_index"]
        outcome = env.estimate(record.theta_hat)
        assert outcome.theta == record.theta
        assert outcome.absolute_error == record.absolute_error
        assert outcome.reward == pytest.approx(record.reward)


# ---------------------------------------------------------------------------
# 11. Regression: SB3-shaped policies run through the protocol
# ---------------------------------------------------------------------------


class FakeModel:
    observation_space = SimpleNamespace(shape=(2 * SLOT_COUNT + 2,))
    action_space = SimpleNamespace(n=SLOT_COUNT)

    def predict(self, observation, *, deterministic: bool):
        assert deterministic
        assert isinstance(observation, np.ndarray)
        assert observation.shape == (2 * SLOT_COUNT + 2,)
        assert observation.dtype == np.float32
        return np.int64(11), None


def test_ppo_room_agent_runs_through_the_evaluator():
    agent = PPORoomAgent(FakeModel(), budget=8, training_seed=3)
    record = run_episode(
        agent,
        EpisodeDefinition("ppo-1", 12, 77),
        8,
        training_seed=3,
    )
    assert record.completed
    assert not record.protocol_failure
    assert not record.fallback_used
    assert record.theta_hat == 11
    assert record.training_seed == 3
    assert record.metadata["algorithm"] == "Stable-Baselines3 PPO"


def test_bare_predict_policy_is_adapted_with_the_canonical_encoding():
    record = run_episode(
        GymPolicyAdapter(FakeModel()),
        EpisodeDefinition("gym-1", 12, 78),
        8,
        agent_id="gym",
        agent_version="gym-v1",
        condition="test",
    )
    assert not record.protocol_failure
    assert record.theta_hat == 11


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_validate_profiles_cli(capsys):
    from light_learning.cli import main as cli_main

    assert cli_main(["validate-profiles"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["pilot"]["version"] == "room-v1-pilot-v3"
    assert result["pilot"]["distinct_theta_values"] == 10
    assert result["full"]["distinct_theta_values"] == len(CONFIG.theta_candidates)
    assert result["full"]["total_episodes"] == 400


def test_report_cli_summarizes_records(tmp_path):
    from light_learning.cli import main as cli_main

    records_path = tmp_path / "records.jsonl"
    write_records(records_path, [make_record(4, 6), make_record(20, 20)])
    output = tmp_path / "summary.json"
    assert cli_main(["report", str(records_path), "--output", str(output)]) == 0
    summary = json.loads(output.read_text())
    assert summary[0]["episode_count"] == 2
    assert summary[0]["mae"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Regression: the LLM agent runs through the evaluator against a stub server
# ---------------------------------------------------------------------------


class ScriptedChatClient:
    """A ChatClient that answers every turn with a schema-valid JSON action."""

    def __init__(self, slot: int = 14, theta_hat: int = 19) -> None:
        self.slot = slot
        self.theta_hat = theta_hat
        self.calls = 0

    def complete(self, *, model, messages, json_schema, options, think):
        from light_learning.ollama import ChatResponse

        self.calls += 1
        if any("theta_hat" in str(key) for key in json_schema.get("properties", {})):
            content = json.dumps({"kind": "estimate", "theta_hat": self.theta_hat})
        else:
            content = json.dumps({"kind": "observe", "time_slot": self.slot})
        return ChatResponse(
            content=content,
            thinking="weighing the observed slots",
            raw={"message": {"content": content}},
            latency_seconds=0.25,
        )

    def model_metadata(self, model):
        return {"model": model, "digest": "test-digest"}


def test_llm_room_agent_runs_through_the_evaluator():
    from light_learning.llm_agent import LLMRoomAgent, LLMRoomAgentConfig

    client = ScriptedChatClient()
    agent = LLMRoomAgent(client, LLMRoomAgentConfig(model="qwen3:1.7b"))
    record = run_episode(agent, EpisodeDefinition("llm-1", 18, 31337), 4)

    assert record.completed
    assert not record.protocol_failure
    assert not record.fallback_used
    assert record.agent_id == "ollama-llm-room-agent"
    assert record.condition == "qualitative"
    assert record.theta_hat == 19
    assert [event["action"] for event in record.trace[:-1]] == [14, 14, 14, 14]
    # LLM diagnostics survive into the trace, and thinking is never fed back.
    assert all(event["thinking"] for event in record.trace)
    assert all(json.loads(event["raw_response"]) for event in record.trace)
    assert record.latency_seconds == pytest.approx(0.25 * 5)


def test_llm_agent_self_reported_fallback_is_honoured():
    """An agent that flags its own fallback is recorded without being re-run."""

    from light_learning.llm_agent import LLMRoomAgent, LLMRoomAgentConfig

    class BrokenChatClient(ScriptedChatClient):
        def complete(self, **kwargs):
            raise RuntimeError("ollama unreachable")

    agent = LLMRoomAgent(
        BrokenChatClient(), LLMRoomAgentConfig(model="qwen3:1.7b", schema_retries=0)
    )
    record = run_episode(agent, EpisodeDefinition("llm-2", 18, 31338), 4)

    assert record.completed
    assert record.protocol_failure
    assert record.fallback_used
    assert [event["action"] for event in record.trace[:-1]] == [0, 1, 2, 3]
    assert all(event["metadata"]["errors"] for event in record.trace)


# ---------------------------------------------------------------------------
# Degenerate-policy detection
# ---------------------------------------------------------------------------


class ConstantAgent:
    """Ignores every observation and always names the same slot."""

    agent_id, agent_version, condition = "constant", "v1", "degenerate"

    def __init__(self, slot: int = 15) -> None:
        self.slot = slot

    def start_episode(self, state): pass

    def act(self, state):
        return AgentDecision(action=self.slot)

    def run_metadata(self):
        return {}


def test_constant_policy_shows_as_carrying_no_information():
    """MAE alone cannot separate a collapsed policy from a real one."""

    records = evaluate_profile(
        ConstantAgent, profile="pilot",
        agent_id="constant", agent_version="v1", condition="degenerate",
    )
    for report in aggregate_metrics(records):
        assert report["skill_over_constant"] <= 0.0
        assert report["theta_hat_correlation"] == 0.0
        assert report["distinct_estimates"] == 1
        assert report["distinct_observation_slots"] == 1
        assert report["mae"] >= report["best_constant_mae"]


def test_real_policy_shows_as_carrying_information():
    records = evaluate_profile(
        mle_factory, profile="pilot",
        agent_id="mle", agent_version="0.2.0", condition="oracle_mle",
    )
    for report in aggregate_metrics(records):
        assert report["distinct_estimates"] > 1
        assert report["skill_over_constant"] > 0.3
        # The estimate tracks the hidden peak, which a collapsed policy cannot
        # do. The bound is loose because a 10-episode pilot cell is noisy.
        assert report["theta_hat_correlation"] > 0.4
        if report["budget"] == 32:
            assert report["skill_over_constant"] > 0.8
            assert report["theta_hat_correlation"] > 0.95


def test_correlation_is_zero_when_either_side_never_varies():
    from light_learning.evaluator import _correlation

    assert _correlation([1, 2, 3], [5, 5, 5]) == 0.0
    assert _correlation([5, 5, 5], [1, 2, 3]) == 0.0
    assert _correlation([1, 2, 3], [2, 4, 6]) == pytest.approx(1.0)
    assert _correlation([1], [1]) == 0.0


def test_best_constant_mae_is_the_true_minimum():
    # thetas 4 and 20 -> the best constant sits between them.
    records = [make_record(4, 4), make_record(20, 20)]
    report = next(r for r in aggregate_metrics(records) if r["scope"] == "aggregate")
    expected = min(
        (abs(4 - slot) + abs(20 - slot)) / 2 for slot in range(SLOT_COUNT)
    )
    assert report["best_constant_mae"] == pytest.approx(expected)


def test_aggregate_row_exposes_per_seed_spread():
    """A bimodal cell must not hide behind its mean."""

    records = [
        make_record(10, 10, seed=0),   # seed 0 nails it
        make_record(20, 20, seed=0),
        make_record(10, 25, seed=1),   # seed 1 is far off
        make_record(20, 5, seed=1),
    ]
    aggregate = next(
        r for r in aggregate_metrics(records) if r["scope"] == "aggregate"
    )
    assert aggregate["per_training_seed_mae"] == {"0": 0.0, "1": 15.0}
    assert aggregate["training_seed_mae_spread"] == pytest.approx(15.0)
    # Seed 1 removes none of the constant-answer error; seed 0 removes all of it.
    assert aggregate["per_training_seed_skill"]["0"] > 0.9
    assert aggregate["per_training_seed_skill"]["1"] < 0.0
