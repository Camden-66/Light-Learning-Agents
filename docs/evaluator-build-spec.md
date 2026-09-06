# Evaluator Implementation Specification

This is the normative build contract for the evaluator owner. The evaluator is
an independent teammate deliverable: it must orchestrate room episodes,
produce comparable records, calculate metrics, and write artifacts. It must
not implement RL, MLE, or the Ollama agent.

## Scope and ownership

- **Environment owner:** canonical `RoomConfig`, `RoomEnv`, public room
  state, deterministic light draws, and the Gymnasium adapter.
- **Evaluator owner:** episode definitions, agent-driving loop, fallback
  policy, records, metrics, JSONL/summary artifacts, and evaluation profiles.
- **Baseline owners:** RL and oracle-MLE implementations consuming the shared
  contracts; see [`baseline-build-spec.md`](baseline-build-spec.md).
- **LLM owner:** local Ollama client and LLM room agent implementing the same
  agent protocol below.

Do not change the canonical room likelihood, public observations, action
semantics, scoring, or information boundaries.

## Environment-facing contract

The evaluator receives a `RoomConfig` and constructs a fresh room for every
episode:

```python
env = RoomEnv(config, budget=budget)
state = env.reset(seed=episode_seed, theta=theta)
```

`theta` and `episode_seed` are evaluator-owned hidden inputs. They must not be
passed to an agent.

If an agent has a stochastic policy, instantiate it per episode with a
deterministic evaluator-owned policy seed derived independently of `theta` and
`episode_seed`. The seed may depend on the evaluator master seed, agent version,
condition, budget, and opaque episode ID. Record it in agent metadata. This
must make a resumed episode identical without replaying earlier episodes; never
reuse the hidden room seed as an agent-policy seed.

`RoomState` contains only:

```python
history: tuple[Observation, ...]  # Observation(time_slot, light_on, repeat_index)
budget: int
remaining_observations: int
phase: "observe" | "estimate"  # derived from remaining observations
```

During `observe`, call `env.observe(time_slot)`, which returns the next public
`RoomState`. During `estimate`, call `env.estimate(theta_hat)`, which returns:

```python
TerminalOutcome(theta, theta_hat, absolute_error, reward)
```

The evaluator may access terminal fields only after `estimate`; no trace or
pre-terminal `info` may contain theta, likelihood values, curve constants,
future outcomes, or score.

## Agent-facing contract

Agents provide these members:

```python
agent_id: str
agent_version: str
condition: str

start_episode(state: RoomState) -> None
act(state: RoomState) -> AgentDecision
run_metadata() -> Mapping[str, Any]
```

`AgentDecision` is:

```python
action: int                         # slot while observing; theta_hat at terminal
raw_response: str | None            # optional, for LLM diagnostics
thinking: str | None                 # optional, never replayed as hidden memory
rationale: str | None
latency_seconds: float
fallback_used: bool
protocol_failure: bool
metadata: Mapping[str, Any]
```

The evaluator validates every action as an integer in `{0, ..., 31}`. An
invalid action, invalid return type, or agent exception must not drop an
episode. Instead set `protocol_failure=True` and use the deterministic,
non-oracle fallback below.

## Episode loop and fallback policy

For exactly `B in {4, 8, 16, 32}` observation turns:

1. pass the current public state to `agent.act`;
2. validate the action or substitute a fallback;
3. call `env.observe(action)`; and
4. append the decision and returned on/off outcome to the chronological trace.

After exactly `B` observations, invoke `agent.act` once more and treat its
action as `theta_hat`. Call `env.estimate(theta_hat)`, append terminal fields,
and mark the episode completed.

Fallbacks must use only public history:

- observation phase: choose the smallest least-visited slot;
- terminal phase: choose the observed slot with the highest empirical on rate,
  breaking ties toward the smallest slot; if history is empty, use slot 16.

Record fallback/protocol failure on the episode and never exclude such episodes
from summaries.

## Reproducible episode definitions

Define:

```python
EpisodeDefinition(
    episode_id: str,
    theta: int,          # one of 4..27
    episode_seed: int,
)
```

Create deterministic, theta-stratified sequences from a master seed. Use fixed
held-out episode definitions for every agent/version evaluated in a cell. The
room's outcome stream is keyed by `(episode_seed, slot, repeat_index)` so a
different query order cannot alter an already defined outcome.

Provide profiles:

| Profile | Episodes per model/baseline × condition × budget cell | Status |
| --- | ---: | --- |
| `pilot` | 10 | preliminary |
| `full` | 100 | reportable |

Training episode IDs/seeds used by RL must be separate from held-out IDs/seeds.

## Required record schema

Persist one JSONL `EpisodeRecord` per completed episode with these fields:

```text
agent_id, agent_version, condition,
budget, episode_id, episode_seed,
theta, theta_hat, absolute_error, reward,
trace, completed, protocol_failure, fallback_used,
latency_seconds, metadata
```

`trace` contains each chronological observation/terminal decision, action,
light outcome when applicable, raw LLM diagnostics when applicable, fallback
status, and per-decision latency. Run artifacts should additionally contain a
JSON summary and a manifest with the evaluator version, room configuration,
profile, master seed, and agent metadata.

## Metrics

Compute metrics over **all** records in one evaluation cell:

1. MAE: mean `absolute_error` (primary);
2. median absolute error;
3. hit-within-one-slot rate;
4. protocol-failure rate;
5. fallback rate;
6. mean latency; and
7. worst-theta-slice MAE.

For worst-theta-slice MAE, group records by true theta, calculate MAE within
each theta group, then report the maximum group MAE, the responsible theta,
and that group's record count. Break equal worst-MAE ties toward the smallest
theta. The pilot report must label this metric preliminary.

### Degeneracy diagnostics

MAE alone cannot distinguish a real policy from one that has collapsed onto a
near-constant answer: a collapsed agent still completes every episode, raises no
protocol failure, uses no fallback, and writes well-formed records. Each cell
must therefore also report:

- `best_constant_mae` and `best_constant_slot` — the best single answer that
  ignores every observation, over that cell's own true thetas;
- `skill_over_constant` — the fraction of `best_constant_mae` the agent removes;
- `skill_over_constant_z` — the same comparison paired per episode and divided
  by its standard error;
- `theta_hat_correlation` — Pearson correlation between theta and theta_hat,
  defined as zero when either never varies;
- `distinct_estimates` and `distinct_observation_slots` — action-space coverage.

**Report these numbers; do not reduce them to a verdict.** No single threshold
separates collapsed from real policies. Measured on 100 held-out episodes at
budget 8:

| policy | MAE | skill | z | corr | estimates | slots |
|---|---:|---:|---:|---:|---:|---:|
| PPO, 20k steps | 5.37 | +0.09 | 2.56 | 0.467 | 2/32 | 3/32 |
| PPO, 1M steps | 4.19 | +0.29 | 7.44 | 0.828 | 3/32 | 10/32 |
| oracle MLE | 2.06 | +0.65 | 8.88 | 0.867 | 24/32 | 32/32 |

The 20k policy answers with two slots out of thirty-two and is plainly
degenerate, yet it clears a `z >= 2` significance bar — a two-constant policy
really is better than the best one-constant policy — and its correlation is not
near zero. Only action-space coverage separates it cleanly here, and coverage
alone would misjudge a genuinely sharp policy on an easy cell. Read the row
together.

An aggregate row over multiple training seeds must additionally carry
`per_training_seed_mae`, `per_training_seed_skill`, and
`training_seed_mae_spread`. Seed outcomes can be bimodal — some seeds escape
collapse and some do not — and a mean across a bimodal cell describes neither
mode, so the per-seed rows are the reportable ones.
