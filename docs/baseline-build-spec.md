# Baseline Implementation Specification

This document is the normative integration contract for the RL and MLE
baselines in the Light Learning Agents benchmark. It intentionally specifies
baseline behavior without providing either implementation.

## Ownership and scope

- The room environment, evaluator, record schema, and LLM agent are shared
  infrastructure.
- The RL owner implements and trains the PPO baseline.
- The MLE owner implements the passive uniform-query oracle likelihood MLE.
- Baseline code must not change room dynamics, public observations, scoring,
  or held-out evaluation cases.

## Canonical room task

The room has 32 integer time slots, `t in {0, ..., 31}`. Each episode samples
a hidden peak location `theta` uniformly from `{4, ..., 27}`. At an observed
slot `t`, the light is on with probability:

```text
P(light=on | t, theta) = 0.05 + 0.90 * exp(-(t - theta)^2 / 18)
```

An episode has an observation budget `B` selected from `{4, 8, 16, 32}`.
For exactly `B` steps, an agent chooses a slot and receives a binary outcome.
Repeated slots are valid and count against the budget. Once the budget is
exhausted, the agent submits `theta_hat in {0, ..., 31}`.

The primary score is raw absolute error:

```text
absolute_error = |theta - theta_hat|
```

The environment's terminal reward is `-absolute_error / 31`. Observation
steps receive reward zero. `theta`, the score, and all curve parameters are
private until the terminal action has completed.

## Gymnasium contract for RL

`GymRoomEnv` has:

- `action_space = Discrete(32)`;
- a fixed-size float32 observation vector of length 66;
- the first 32 values: on-count for each slot, divided by `B`;
- the next 32 values: visit-count for each slot, divided by `B`;
- value 64: remaining observations divided by `B`;
- value 65: `1.0` when the next action is the terminal estimate, otherwise
  `0.0`.

During the observation phase, the action selects the slot to inspect. During
the terminal phase, the action is interpreted as `theta_hat`. The terminal
step returns `terminated=True`, `truncated=False`; it is the only step whose
`info` may include `theta`, `theta_hat`, or scoring fields.

The public observation and `info` fields must never reveal the hidden theta,
the likelihood formula, any curve constants, a future outcome, or the score
before termination.

## RL baseline requirements

Implement Stable-Baselines3 PPO with one independently trained policy for
each observation budget (`4`, `8`, `16`, and `32`). The policy may consume
only the Gymnasium observation, action space, and reward supplied above.

- Train five policies per budget with independent training seeds.
- Sample theta and observation randomness from the canonical training
  distribution; never train on held-out evaluation episode IDs or seeds.
- Evaluate each trained policy deterministically (`deterministic=True`).
- Emit one shared `EpisodeRecord` per held-out episode.
- Report results by training seed as well as their aggregate; do not retain
  only the strongest training run.

PPO is a trained-distribution reference. It may learn the task structure from
episodes and terminal rewards, but must not receive direct oracle access to
the likelihood.

## Passive uniform-query oracle likelihood MLE

The MLE reference is deliberately an oracle-information baseline. Its query
policy samples each of its `B` observation slots independently and uniformly
from `{0, ..., 31}`. Given history `(t_i, y_i)`, compute, for every candidate
`candidate in {4, ..., 27}`:

```text
log_likelihood(candidate)
  = sum_i [y_i * log(p(t_i, candidate))
           + (1 - y_i) * log(1 - p(t_i, candidate))]
```

where `p` is the canonical room probability above. Return the candidate with
the largest log-likelihood; choose the smallest candidate on an exact tie.

The historical name “IID MLE” may appear in discussion, but the implementation
and reports must call it **passive uniform-query oracle likelihood MLE**. The
query slots are IID; the Bernoulli outcomes are not identically distributed
when sampled at different slots.

Do not make this baseline adaptive, Bayesian, or nonparametric in the first
benchmark version. Those are separate future baselines.

## Evaluation protocol and records

Every agent version is evaluated against the same versioned held-out episode
definitions. An episode definition contains at least `episode_id`, `theta`,
and `episode_seed`. Room outcomes must be reproducible from episode seed,
slot, and repeat number so an agent's query order cannot change a previously
defined outcome.

Each `EpisodeRecord` must contain:

- agent/version identifier and configuration digest;
- prompt condition or baseline condition;
- budget, episode ID, seed, theta, theta hat, and absolute error;
- chronological observation/action trace;
- completion/fallback/protocol-failure status;
- latency and model/runtime metadata when applicable.

Use a 10-episode-per-cell pilot profile and a 100-episode-per-cell held-out
profile. For each version, prompt/baseline condition, and budget, report:

1. mean absolute error (primary metric);
2. median absolute error;
3. hit-within-one-slot rate;
4. invalid-output or protocol-failure rate;
5. latency where applicable; and
6. **worst-theta-slice MAE**: group held-out episodes by true theta, calculate
   MAE for each group, then report the largest group MAE together with its
   theta value and group size.

Do not discard fallback-completed or invalid-output episodes from metrics.
Mark pilot worst-theta-slice MAE as preliminary because its per-theta sample
size is small.

## Required validation for baseline pull requests

- Gymnasium environment compatibility and a complete rollout at every budget.
- No pre-terminal leakage of theta, curve parameters, future outcomes, or
  terminal score.
- Correct phase transition, terminal reward, score, and tie behavior.
- Hand-calculated likelihood tests for the MLE reference.
- PPO smoke training that produces valid terminal estimates and records.
- Held-out evaluator run that produces schema-valid records and aggregate
  metrics, including worst-theta-slice MAE.

## Fairness interpretation

The oracle MLE knows the likelihood and is an upper/reference point, not an
information-matched cognitive baseline. The RL baseline learns from the
training distribution. The primary LLM condition receives only qualitative
room rules, while the disclosed-likelihood LLM condition is a separate
ablation. Reports must retain these labels rather than presenting all methods
as equivalent-information comparisons.
