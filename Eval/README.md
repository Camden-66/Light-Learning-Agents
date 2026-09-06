# Evaluation

This folder implements the evaluator contract in
`docs/evaluator-build-spec.md`. It contains the public room environment
contract, agent protocol, fallback policy, fixed profiles, episode records,
metrics, JSONL artifacts, summaries, manifests, and resumable run support. It
does not implement PPO, oracle MLE, or the Ollama client.

## Agent contract

An evaluated agent supplies:

```python
class MyAgent:
    agent_id = "my-agent"
    agent_version = "my-agent-v1"
    condition = "baseline"

    def start_episode(self, state):
        pass

    def act(self, state):
        return AgentDecision(
            action=0,
            raw_response=None,
            thinking=None,
            rationale="query slot 0",
            latency_seconds=0.01,
            metadata={},
        )

    def run_metadata(self):
        return {"model": "example"}
```

The evaluator gives the agent a public `RoomState`. It calls `act` exactly `B`
times during `observe`, then once during `estimate`. Hidden theta, episode seed,
likelihood values, curve parameters, and score are never present in preterminal
state or trace entries.

SB3-style policies exposing `predict(observation, deterministic=True)` are
supported through `GymPolicyAdapter`; all other agents should implement the
shared protocol directly.

## Profiles and definitions

`episode_profiles.v1.json` defines a numeric master seed and these fixed
profiles:

- `pilot`: 10 episodes per budget cell, marked preliminary.
- `full`: 100 episodes per budget cell, reportable.

Theta values are deliberately stratified. Use
`assert_training_separation` to verify RL training IDs and seeds do not overlap
with evaluation definitions.

## In-memory evaluation

```python
from Eval.eval import AgentDecision, evaluate_profile

records = evaluate_profile(
    lambda: MyAgent(),
    profile="full",
    agent_id="my-agent",
    agent_version="my-agent-v1",
    condition="baseline",
)
```

Every `EpisodeRecord` contains agent identity, score and reward, completion and
failure flags, fallback use, latency, metadata, and a chronological trace with
LLM diagnostics when supplied.

## Artifactful, resumable runs

```python
from Eval.eval import run_evaluation

records = run_evaluation(
    lambda: MyAgent(),
    output_dir="evaluation-runs",
    run_id="my-agent-full-v1",
    profile="full",
    agent_id="my-agent",
    agent_version="my-agent-v1",
    condition="baseline",
)
```

The run directory contains:

- `records.jsonl`, written after each completed episode;
- `summary.json`, with aggregate and per-training-seed metrics; and
- `manifest.json`, with evaluator version, room configuration, profile, master
  seed, agent metadata, run status, and progress.

Re-running with the same `run_id` resumes from existing records and skips
already completed episode keys. Duplicate records or a mismatched manifest are
rejected.

## CLI

```sh
python Eval/eval.py validate-profiles
python Eval/eval.py report records.jsonl --output summary.json
```

Run tests from the repository root:

```sh
python -m unittest discover -s Eval -p 'test_*.py' -v
```
