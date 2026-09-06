# Emergent World-Model Integration

This document is the normative contract for `light_learning.emergent`, the
closed-book bump world-model. It sits alongside
[`baseline-build-spec.md`](baseline-build-spec.md) and
[`evaluator-build-spec.md`](evaluator-build-spec.md) and does not restate the
canonical room task.

## Scope

`EmergentBumpAgent` is a world model, not a policy and not a second copy of
`RoomEnv`. It is the condition that tests whether an agent can recover the
lamp's generative structure from on/off bits alone.

## Information boundary

The agent **must not** receive the canonical constants `0.05`, `0.90`, or `18`,
in any form, including as a prior spike. It assumes only a unimodal Bernoulli
bump:

```text
p(on | t) = a + b * exp(-(t - mu)^2 / (2 * sigma^2))
```

and maintains a posterior over `(mu, sigma, a, b)` from public history.
`TRUE_SIGMA` exists in the module for **diagnostics only** and must never enter
a likelihood, prior, or tie-break.

The agent **does** receive the theta support. `hypothesis_grid` ranges `mu`
over `config.theta_candidates`, not over every slot. The support is task
framing that the room states up front — the human explainer prints it, and both
LLM prompt conditions state it — so withholding it measures answer-space
handicap rather than model discovery. Looks remain unrestricted over all
`slot_count` slots; only the terminal estimate is constrained.

This distinction is load-bearing. An agent whose grid spans all 32 slots places
about a quarter of its hypotheses on peaks the room can never generate, emits
an out-of-support estimate in roughly 6% of episodes at budget 8, and carries
about 9% of its measured error as answer-space cost. The terminal schema still
accepts any slot in `[0, slot_count - 1]`, so a violation stays visible in the
trace as a diagnostic rather than being silently prevented.

## Conditions

| condition | shape prior | what it measures |
|---|---|---|
| `emergent_in_episode` | none | in-episode structure discovery |
| `emergent_pooled_shape` | `pool_shape_log_prior` over revealed peaks | cross-episode transfer of lamp shape |

Pooling transfers **shape only** — `(sigma, a, b)` scores are copied onto every
`mu`, so a later episode still infers its own peak. Training episodes must come
from outside the held-out bank; call `assert_training_separation`.

## Reporting structure recovery

Report width recovery **against its chance level, in a band**:

- `sigma_mass_near_3` — posterior mass within `SIGMA_TOLERANCE` (one grid step)
  of `TRUE_SIGMA`.
- `sigma_mass_chance` — what a uniform, unlearned posterior already puts in the
  same band (`sigma_mass_chance_level`). Currently `33%`.
- `mean_sigma` — posterior mean width.

`sigma_mass_at_3` (exact match) is retained but is **not** the headline number.
The width grid is coarse, so with finite data the posterior legitimately splits
between the `3.0` and `3.5` columns; exact mass is non-monotonic in pool size
and can fall while the fit improves. A width number published without its
chance level is not interpretable.

## Reporting peak accuracy

Structure recovery and peak accuracy are **separate claims**. Pooling currently
improves the first without improving the second; a paired run of 100 episodes
at budget 8 shows pooled mean absolute error above in-episode by `+0.71` slots
(`SE 0.46`, not significant). Do not report pooling as an accuracy win without
a run large enough to support it.

## Reproducibility

`run_metadata()` must pin everything that changes results: the hypothesis grid
axes, `theta_candidates`, `condition`, `policy_seed`, a `shape_prior_digest`
over the pooled prior, and a `configuration_digest` over the whole record,
following the pattern in `mle.py`. A bare `pooled_shape_prior: true` is
insufficient — two runs over different training pools must not produce
identical manifests.

## Demo surfaces are not evaluations

The explainer endpoints and `light_learning.compare` are smoke surfaces. They
carry `reportable: false` and a `non_reportable_*` label, and must keep doing
so. `compare` defaults to `profile="pilot"` (10 episodes per cell), which
cannot rank conditions; use `profile="full"` before drawing conclusions.
