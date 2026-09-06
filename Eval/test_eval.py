import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from environment import Observation, RoomConfig, RoomEnv, RoomState
from eval import (
    BUDGETS,
    AgentDecision,
    EpisodeDefinition,
    EpisodeRecord,
    aggregate_metrics,
    assert_training_separation,
    evaluate_profile,
    load_episode_definitions,
    read_records,
    run_episode,
    run_evaluation,
    write_records,
)


class ConformingAgent:
    agent_id = "agent-1"
    agent_version = "agent-v1"
    condition = "baseline"

    def __init__(self, query_slot=7, estimate=12):
        self.query_slot = query_slot
        self.estimate = estimate
        self.states = []

    def start_episode(self, state):
        self.states.append(state)

    def act(self, state):
        self.states.append(state)
        action = self.estimate if state.phase == "estimate" else self.query_slot
        return AgentDecision(
            action=action,
            raw_response=str(action),
            rationale="fixed test policy",
            latency_seconds=0.001,
            metadata={"test": True},
        )

    def run_metadata(self):
        return {"model": "test"}


def make_record(theta, theta_hat, *, protocol=False, fallback=False, seed=None):
    budget = 4
    trace = [
        {
            "step": step,
            "phase": "observe",
            "action": 0,
            "raw_response": "0",
            "thinking": None,
            "rationale": None,
            "decision_latency_seconds": 0.1,
            "fallback_used": False,
            "protocol_failure": False,
            "metadata": {},
            "time_slot": 0,
            "light_on": False,
            "repeat_index": step,
        }
        for step in range(budget)
    ]
    trace.append(
        {
            "step": budget,
            "phase": "estimate",
            "action": theta_hat,
            "raw_response": str(theta_hat),
            "thinking": None,
            "rationale": None,
            "decision_latency_seconds": 0.1,
            "fallback_used": fallback,
            "protocol_failure": protocol,
            "metadata": {},
            "theta": theta,
            "theta_hat": theta_hat,
            "absolute_error": abs(theta - theta_hat),
            "reward": -abs(theta - theta_hat) / 31,
        }
    )
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
        reward=-abs(theta - theta_hat) / 31,
        trace=trace,
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


class EvaluationTests(unittest.TestCase):
    def test_environment_contract_and_private_boundary(self):
        env = RoomEnv(RoomConfig(), budget=4)
        state = env.reset(seed=12, theta=10)
        self.assertEqual(state, RoomState((), 4, 4, "observe"))
        for slot in (0, 0, 3, 8):
            state = env.observe(slot)
        self.assertEqual(state.phase, "estimate")
        self.assertEqual(state.history[1].repeat_index, 1)
        outcome = env.estimate(10)
        self.assertEqual((outcome.theta, outcome.theta_hat, outcome.absolute_error), (10, 10, 0))
        self.assertEqual(outcome.reward, 0)

    def test_profiles_are_deterministic_stratified_and_named_full(self):
        pilot_version, pilot = load_episode_definitions("pilot")
        full_version, full = load_episode_definitions("full")
        self.assertEqual(tuple(pilot), BUDGETS)
        self.assertEqual((len(pilot[4]), len(full[4])), (10, 100))
        self.assertNotEqual(pilot_version, full_version)
        for budget in BUDGETS:
            counts = Counter(episode.theta for episode in full[budget])
            self.assertEqual(set(counts), set(range(4, 28)))
            self.assertLessEqual(max(counts.values()) - min(counts.values()), 1)
            self.assertEqual(full[budget], load_episode_definitions("full")[1][budget])
        self.assertTrue(
            {
                episode.episode_id for cases in pilot.values() for episode in cases
            }.isdisjoint(
                {episode.episode_id for cases in full.values() for episode in cases}
            )
        )

    def test_training_overlap_is_rejected(self):
        _, definitions = load_episode_definitions("full")
        episode = definitions[4][0]
        with self.assertRaises(ValueError):
            assert_training_separation([episode.episode_id], [], definitions)
        with self.assertRaises(ValueError):
            assert_training_separation([], [episode.episode_seed], definitions)

    def test_agent_protocol_trace_and_terminal_fields(self):
        record = run_episode(
            ConformingAgent(),
            EpisodeDefinition("episode-1", 12, 12345),
            4,
            agent_id="agent-1",
            agent_version="agent-v1",
            condition="baseline",
            profile="full",
            profile_version="room-v1-full-v1",
        )
        self.assertTrue(record.completed)
        self.assertFalse(record.protocol_failure)
        self.assertFalse(record.fallback_used)
        self.assertEqual(len(record.trace), 5)
        self.assertTrue(all(event["phase"] == "observe" for event in record.trace[:4]))
        self.assertNotIn("theta", record.trace[0])
        self.assertNotIn("reward", record.trace[0])
        self.assertEqual(record.trace[-1]["theta"], 12)
        self.assertEqual(record.reward, 0)

    def test_exact_observation_counts_for_every_budget(self):
        for budget in BUDGETS:
            record = run_episode(
                ConformingAgent(),
                EpisodeDefinition(f"episode-{budget}", 12, budget),
                budget,
                agent_id="agent-1",
                agent_version="agent-v1",
                condition="baseline",
                profile="full",
                profile_version="room-v1-full-v1",
            )
            self.assertEqual(len(record.trace), budget + 1)
            self.assertEqual(sum(event["phase"] == "observe" for event in record.trace), budget)

    def test_fallbacks_are_exact_and_failures_are_scored(self):
        class InvalidAgent(ConformingAgent):
            def act(self, state):
                self.states.append(state)
                return "not an AgentDecision"

        record = run_episode(
            InvalidAgent(),
            EpisodeDefinition("invalid-1", 10, 99),
            4,
            agent_id="invalid",
            agent_version="invalid-v1",
            condition="test",
            profile="pilot",
            profile_version="room-v1-pilot-v2",
        )
        self.assertTrue(record.completed)
        self.assertTrue(record.protocol_failure)
        self.assertTrue(record.fallback_used)
        self.assertEqual([event["action"] for event in record.trace[:4]], [0, 1, 2, 3])

        state = RoomState(
            (
                Observation(5, False, 0),
                Observation(5, True, 1),
                Observation(2, True, 0),
                Observation(2, False, 1),
            ),
            4,
            0,
            "estimate",
        )
        from eval import _fallback_estimate

        self.assertEqual(_fallback_estimate(state), 2)

    def test_metrics_include_all_required_rates_and_tie_break(self):
        records = [
            make_record(4, 4, seed=1),
            make_record(4, 5, seed=2),
            make_record(20, 10, protocol=True, fallback=True, seed=1),
            make_record(20, 11, protocol=False, fallback=True, seed=2),
        ]
        reports = aggregate_metrics(records)
        aggregate = next(report for report in reports if report["scope"] == "aggregate")
        self.assertEqual(aggregate["episode_count"], 4)
        self.assertEqual(aggregate["protocol_failure_rate"], 1 / 4)
        self.assertEqual(aggregate["fallback_rate"], 2 / 4)
        self.assertEqual(aggregate["worst_theta"], 20)
        self.assertEqual(aggregate["worst_theta_group_size"], 2)
        self.assertEqual(
            {report["training_seed"] for report in reports if report["scope"] == "training_seed"},
            {1, 2},
        )

    def test_jsonl_summary_manifest_and_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            records = evaluate_profile(
                ConformingAgent,
                profile="pilot",
                agent_id="agent-1",
                agent_version="agent-v1",
                condition="baseline",
            )
            path = Path(directory) / "records.jsonl"
            write_records(path, records)
            self.assertEqual(len(read_records([path])), 40)

            calls = {"count": 0}

            def factory():
                calls["count"] += 1
                return ConformingAgent()

            first = run_evaluation(
                factory,
                output_dir=directory,
                run_id="resume-test",
                profile="pilot",
                agent_id="agent-1",
                agent_version="agent-v1",
                condition="baseline",
            )
            first_calls = calls["count"]
            second = run_evaluation(
                factory,
                output_dir=directory,
                run_id="resume-test",
                profile="pilot",
                agent_id="agent-1",
                agent_version="agent-v1",
                condition="baseline",
            )
            self.assertEqual(len(first), 40)
            self.assertEqual(len(second), 40)
            self.assertEqual(calls["count"], first_calls)
            manifest = json.loads((Path(directory) / "resume-test" / "manifest.json").read_text())
            self.assertEqual(manifest["status"], "complete")
            self.assertEqual(manifest["completed_episode_count"], 40)
            self.assertTrue((Path(directory) / "resume-test" / "summary.json").exists())


if __name__ == "__main__":
    unittest.main()
