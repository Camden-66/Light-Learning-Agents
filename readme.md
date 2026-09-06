# Light Learning Agents

A reproducible benchmark for studying in-context, sequential learning in a
small active-observation task. Agents choose when to inspect a room light and
then estimate the hidden time slot where it is most likely to be on.

## What is implemented here

- a deterministic canonical room environment and Gymnasium adapter;
- shared state and episode-record contracts for the evaluator owner;
- a local Ollama-backed LLM room agent using structured JSON actions; and
- the teammate handoff for RL and MLE baseline implementations in
  [`docs/baseline-build-spec.md`](docs/baseline-build-spec.md).

RL, oracle-MLE, and evaluator implementations are intentionally not included.

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
