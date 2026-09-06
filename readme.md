# Light Learning Agents

A reproducible benchmark for studying in-context, sequential learning in a
small active-observation task. Agents choose when to inspect a room light and
then estimate the hidden time slot where it is most likely to be on.

## What is implemented here

- a deterministic canonical room environment and Gymnasium adapter;
- the shared room-state, decision, and episode-record contracts;
- a local Ollama-backed LLM room agent using structured JSON actions;
- the passive uniform-query oracle-likelihood MLE baseline;
- a Stable-Baselines3 PPO training matrix and evaluator-facing agent adapter; and
- the evaluator: held-out episode banks, the agent-driving loop, fallbacks,
  metrics, and resumable JSONL/summary/manifest artifacts.

The evaluator implements
[`docs/evaluator-build-spec.md`](docs/evaluator-build-spec.md). It owns no
environment code — the canonical `RoomEnv`, light draws, public `RoomState`, and
fallback policy all live in `light_learning.room`, so evaluation results replay
exactly against the environment the baselines train in.

## Running an evaluation

Agents supply `agent_id`, `agent_version`, `condition`, `start_episode(state)`,
`act(state) -> AgentDecision`, and `run_metadata()`. `AgentDecision` must be
`light_learning.types.AgentDecision`; anything else is scored as a protocol
failure and driven through the documented fallback.

```python
from light_learning.evaluator import run_evaluation
from light_learning.mle import PassiveUniformOracleMLE

records = run_evaluation(
    lambda *, policy_seed: PassiveUniformOracleMLE(query_seed=policy_seed),
    output_dir="evaluation-runs",
    run_id="mle-pilot",
    profile="pilot",            # 10 episodes/cell, preliminary; "full" is 100
    agent_id="passive-uniform-oracle-likelihood-mle",
    agent_version="0.2.0",
    condition="oracle_mle",
)
```

A factory that accepts a `policy_seed` keyword receives a deterministic,
evaluator-owned seed per episode, derived from the master seed, agent version,
condition, budget, and episode ID — never from the hidden `theta` or
`episode_seed`. It is recorded in `metadata["policy_seed"]`, so a resumed run
reproduces each episode without replaying the ones before it. Factories that
take no arguments keep working unchanged.

The run directory holds `records.jsonl` (flushed per episode), `summary.json`,
and `manifest.json`. Re-running the same `run_id` resumes and skips completed
episodes; a mismatched manifest or duplicate records are rejected.

```bash
uv run light-learning validate-profiles
uv run light-learning report evaluation-runs/mle-pilot/records.jsonl --output summary.json
```

## Interactive explainer

```bash
uv sync --extra dev
uv run light-learning web --port 8767
```

Open http://127.0.0.1:8767/ — it drives the same `RoomEnv` as the rest of the
package. **Train RL** runs on-policy REINFORCE on `GymRoomEnv` and plots rolling
MAE; **Run trained RL on this episode** compares that policy to oracle MLE only
after you commit. This small REINFORCE trainer and its smoke-sample metrics are
an explainer, not the reportable PPO benchmark.

## Setup

This project targets Python 3.12 and `uv`:

```bash
uv sync --extra dev --extra rl
uv run pytest
```

The `rl` extra is separately declared so non-RL users can omit it; include it
whenever you train PPO or run the complete test suite.

`light_learning.ppo.train_ppo_suite(...)` defaults to the required four budgets
and five independent training seeds, saves one checkpoint per run, and writes a
manifest. The evaluator loads each checkpoint through `PPORoomAgent` and owns
held-out episodes, `EpisodeRecord` construction, and all reported metrics. Use
`assert_training_separation` to prove RL training IDs and seeds never overlap
the held-out banks.

The MLE agent requires an explicit evaluator-owned `query_seed` for each
episode, supplied by the `policy_seed` factory contract above. It must be
independent of the hidden room seed; this makes its uniform query schedule
reproducible across retries and resumed runs.

The local LLM experiment needs a running Ollama server and both requested
models installed. The CLI never pulls models automatically. Check readiness
first:

```bash
uv run light-learning preflight
```

After freeing sufficient disk space and installing a model yourself, hand the
constructed `LLMRoomAgent` to `run_evaluation` as shown above. The evaluator
owns pilot/full execution and run-directory output.

## Experiment interpretation

The primary LLM prompt gives only qualitative room rules, so it measures
zero-shot in-context model discovery rather than online weight learning. The
disclosed-likelihood condition is an ablation over exactly one variable: both
arms state the answer range, and only the disclosed arm supplies the Bernoulli
likelihood.

The oracle MLE and trained RL baselines have different information regimes;
report their labels faithfully. In particular the three are not a ladder. Oracle
MLE has the exact likelihood but zero task exposure and a deliberately
non-adaptive uniform query schedule; PPO has large task exposure and an adaptive
policy but no analytic knowledge of the likelihood. Oracle MLE is therefore not
a performance ceiling: an adaptive querier using the same exact likelihood
reaches MAE 2.54/1.23/0.55/0.30 at budgets 4/8/16/32 against passive uniform
MLE's 3.62/2.33/1.24/0.59, so roughly a third to a half of the oracle MLE's
error is query strategy rather than model knowledge.
