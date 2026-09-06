# Light Learning Agents

A reproducible benchmark for studying in-context, sequential learning in a
small active-observation task. Agents choose when to inspect a room light and
then estimate the hidden time slot where it is most likely to be on.

## What is implemented here

- a deterministic canonical room environment and Gymnasium adapter;
- shared state and episode-record contracts for the evaluator owner;
- a local Ollama-backed LLM room agent using structured JSON actions;
- the passive uniform-query oracle-likelihood MLE baseline; and
- a PPO helper on `GymRoomEnv`.

The evaluator (held-out banks, JSONL artifacts) is still a separate teammate
deliverable: [`docs/evaluator-build-spec.md`](docs/evaluator-build-spec.md).

## Interactive explainer

```bash
uv sync --extra dev
uv run light-learning web --port 8767
```

Open http://127.0.0.1:8767/ — it drives the same `RoomEnv` as the rest of the
package. **Train RL** runs on-policy REINFORCE on `GymRoomEnv` and plots rolling
MAE; **Run trained RL on this episode** compares that policy to oracle MLE.

## Setup

This project targets Python 3.12 and `uv`:

```bash
uv sync --extra dev
uv run pytest
```

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
