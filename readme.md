# Light Learning Agents

A reproducible benchmark for studying in-context, sequential learning in a
small active-observation task. Agents choose when to inspect a room light and
then estimate the hidden time slot where it is most likely to be on.

## What is implemented here

- a deterministic canonical room environment and Gymnasium adapter;
- shared state and episode-record contracts for the evaluator owner;
- a local Ollama-backed LLM room agent using structured JSON actions;
- the passive uniform-query oracle-likelihood MLE baseline;
- a Stable-Baselines3 PPO training matrix and evaluator-facing agent adapter; and
- an English interactive explainer (`src/light_learning/web/` + `web_server.py`)
  with oracle MLE comparison and a small on-page REINFORCE trainer.

The evaluator (held-out banks, JSONL artifacts) is still a separate teammate
deliverable: [`docs/evaluator-build-spec.md`](docs/evaluator-build-spec.md).

## Interactive explainer (MLE + RL)

Install once, then start the app (not `python -m http.server`):

```bash
git checkout feat/baselines-and-explainer
python -m pip install -e ".[dev]"
light-learning web
```

Open **http://127.0.0.1:8768/**. If that URL 404s, something else is already
bound to 8768 — stop it (`lsof -i :8768`) and start `light-learning web` again.

With `uv`: `uv sync --extra dev` then `uv run light-learning web`.

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
held-out episodes, `EpisodeRecord` construction, and all reported metrics.

The MLE agent requires an explicit evaluator-owned `query_seed` for each
episode. It must be independent of the hidden room seed; this makes its uniform
query schedule reproducible across retries and resumed runs.

The local LLM experiment needs a running Ollama server and both requested
models installed. The CLI never pulls models automatically. Check readiness
first:

```bash
uv run light-learning preflight
```

After freeing sufficient disk space and installing a model yourself, hand the
constructed `LLMRoomAgent` to the evaluator implementation described in
[`docs/evaluator-build-spec.md`](docs/evaluator-build-spec.md). The evaluator
owns pilot/full execution and `artifacts/` output.

## Experiment interpretation

The primary LLM prompt gives only qualitative room rules, so it measures
zero-shot in-context model discovery rather than online weight learning. The
disclosed-likelihood condition is an ablation. The oracle MLE and trained RL
baselines have different information regimes; report their labels faithfully.
