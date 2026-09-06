# Light Learning Agents

A reproducible benchmark for studying in-context, sequential learning in a
small active-observation task. Agents choose when to inspect a room light and
then estimate the hidden time slot where it is most likely to be on.

## What is implemented here

- a deterministic canonical room environment and Gymnasium adapter;
- the shared room-state, decision, and episode-record contracts;
- a local Ollama-backed LLM room agent using structured JSON actions;
- the passive uniform-query oracle-likelihood MLE baseline;
- a Stable-Baselines3 PPO training matrix and evaluator-facing agent adapter;
- the evaluator: held-out episode banks, the agent-driving loop, fallbacks,
  metrics, and resumable JSONL/summary/manifest artifacts; and
- an English interactive explainer (`src/light_learning/web/` + `web_server.py`)
  with oracle MLE comparison and a small on-page REINFORCE trainer.

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

## Interactive explainer (MLE + RL)

```bash
git checkout feat/baselines-and-explainer
uv sync --extra dev
uv run light-learning web
```

Without `uv`: `python -m pip install -e ".[dev]"` then `light-learning web`.

Open **http://127.0.0.1:8768/**. Do not use `python -m http.server`. If the
port is taken, stop the old process (`lsof -i :8768`) and start again.

### What to click

- **32 windows** — only for *you* playing: first `B` clicks look (yellow = ON,
  gray = OFF); the next click is your peak guess. Skip the windows for MLE/RL.
- **New episode** — new hidden θ (the page also starts one on load).
- **Compare oracle MLE** — enabled after you commit in the current UI. MLE
  samples its own uniform slots. Green = its guess.
- **Train REINFORCE demo (250 episodes)** — on-policy REINFORCE on
  `GymRoomEnv` (looks give reward 0; commit gives −|θ−θ̂|/31). Wait a few
  seconds for a blue rolling-MAE curve. Smoke-sample metrics are **not** the
  official PPO benchmark; that is `light_learning.ppo`.
- **Compare REINFORCE demo** — greedy policy on the current episode after
  train (and, in the packaged UI, after you commit). Blue = its guess.

Orange = true peak. White dashed = your guess. Green = MLE. Blue = REINFORCE.

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
disclosed-likelihood condition is an ablation. The oracle MLE and trained RL
baselines have different information regimes; report their labels faithfully.
