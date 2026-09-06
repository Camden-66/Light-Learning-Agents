"""In-episode and pooled world models that *infer* the light bump.

Oracle MLE is given the canonical formula and only searches for theta.
These agents never see those constants. They maintain a family of unimodal
Bernoulli bumps ``p(on|t) = a + b * exp(-(t-mu)^2 / (2 sigma^2))`` and update
a posterior over ``(mu, sigma, a, b)`` from public on/off history.

That is the emergent-learning claim: if the posterior on ``sigma`` concentrates
near the true width (``sigma = 3``, because the room uses ``exp(-d^2 / 18)``)
and the implied curve tracks ``P(on|t, theta)``, the agent recovered the
pattern rather than a reward-trained look/guess policy.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from math import exp, log
from typing import Any, Iterable, Sequence

from .config import RoomConfig, SLOT_COUNT
from .types import AgentDecision, Observation, RoomState, TerminalOutcome

# Misspecified-on-purpose grid: the true room is a=0.05, b=0.90, 2*sigma^2=18.
SIGMAS = (1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 6.0, 8.0)
BACKGROUNDS = (0.02, 0.05, 0.10, 0.20)
AMPLITUDES = (0.50, 0.70, 0.85, 0.90)
TRUE_SIGMA = 3.0  # diagnostics only; never used as an agent prior spike
# One grid step either side of TRUE_SIGMA. Exact-match mass on a coarse grid is
# a brittle read: with finite data the posterior legitimately splits between the
# 3.0 and 3.5 columns, so exact mass can fall while the fit is improving.
SIGMA_TOLERANCE = 0.5


History = Sequence[tuple[int, int]]


def bump_probability(
    time_slot: int,
    mu: int,
    sigma: float,
    background: float,
    amplitude: float,
) -> float:
    """Bernoulli mean for one hypothesized bump. Not the canonical formula."""

    width = 2.0 * sigma * sigma
    value = background + amplitude * exp(-((time_slot - mu) ** 2) / width)
    return min(max(value, 1e-6), 1.0 - 1e-6)


def _log_bernoulli(y: int, p: float) -> float:
    return log(p) if y else log(1.0 - p)


def history_pairs(history: Sequence[Observation] | History) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for item in history:
        if isinstance(item, Observation):
            pairs.append((item.time_slot, 1 if item.light_on else 0))
        else:
            slot, y = item
            pairs.append((int(slot), 1 if y else 0))
    return pairs


@dataclass(frozen=True, slots=True)
class Hypothesis:
    mu: int
    sigma: float
    background: float
    amplitude: float


def hypothesis_grid(config: RoomConfig | None = None) -> tuple[Hypothesis, ...]:
    """Enumerate bump hypotheses over the *answerable* peak locations.

    ``mu`` ranges over ``config.theta_candidates`` rather than every slot. The
    theta support is task framing that the room states up front, not part of
    the hidden generator, so withholding it would only add an answer-space
    handicap on top of the model-discovery problem this agent is meant to
    measure. Looks still range over all ``slot_count`` slots; only the terminal
    estimate is constrained. Same reasoning as the LLM prompt conditions.
    """

    room = config or RoomConfig()
    rows: list[Hypothesis] = []
    for mu in room.theta_candidates:
        for sigma in SIGMAS:
            for background in BACKGROUNDS:
                for amplitude in AMPLITUDES:
                    if background + amplitude >= 0.995:
                        continue
                    rows.append(Hypothesis(mu, sigma, background, amplitude))
    return tuple(rows)


def probability_table(
    hypotheses: Sequence[Hypothesis], slot_count: int = SLOT_COUNT
) -> tuple[tuple[float, ...], ...]:
    return tuple(
        tuple(
            bump_probability(slot, hyp.mu, hyp.sigma, hyp.background, hyp.amplitude)
            for slot in range(slot_count)
        )
        for hyp in hypotheses
    )


_GridKey = tuple[int, tuple[int, ...]]
_GRID_CACHE: dict[
    _GridKey, tuple[tuple[Hypothesis, ...], tuple[tuple[float, ...], ...]]
] = {}


def cached_grid(
    config: RoomConfig | None = None,
) -> tuple[tuple[Hypothesis, ...], tuple[tuple[float, ...], ...]]:
    room = config or RoomConfig()
    key: _GridKey = (room.slot_count, room.theta_candidates)
    packed = _GRID_CACHE.get(key)
    if packed is None:
        hyps = hypothesis_grid(room)
        packed = (hyps, probability_table(hyps, room.slot_count))
        _GRID_CACHE[key] = packed
    return packed


def log_likelihood_hyp(
    history: History,
    hyp: Hypothesis,
    table_row: Sequence[float] | None = None,
) -> float:
    total = 0.0
    for slot, y in history:
        p = table_row[slot] if table_row is not None else bump_probability(
            slot, hyp.mu, hyp.sigma, hyp.background, hyp.amplitude
        )
        total += _log_bernoulli(y, p)
    return total


def log_posterior(
    history: History,
    hypotheses: Sequence[Hypothesis],
    shape_log_prior: Sequence[float] | None = None,
    table: Sequence[Sequence[float]] | None = None,
) -> list[float]:
    scores: list[float] = []
    for index, hyp in enumerate(hypotheses):
        extra = 0.0 if shape_log_prior is None else float(shape_log_prior[index])
        row = table[index] if table is not None else None
        scores.append(log_likelihood_hyp(history, hyp, row) + extra)
    return scores


def _softmax(log_scores: Sequence[float]) -> list[float]:
    peak = max(log_scores)
    weights = [exp(value - peak) for value in log_scores]
    total = sum(weights) or 1.0
    return [weight / total for weight in weights]


def map_hypothesis(
    history: History,
    hypotheses: Sequence[Hypothesis],
    shape_log_prior: Sequence[float] | None = None,
    table: Sequence[Sequence[float]] | None = None,
) -> Hypothesis:
    scores = log_posterior(history, hypotheses, shape_log_prior, table)
    best = 0
    for index, score in enumerate(scores):
        if score > scores[best]:
            best = index
        elif score == scores[best]:
            current = hypotheses[index]
            champ = hypotheses[best]
            if (current.mu, current.sigma, current.background, current.amplitude) < (
                champ.mu,
                champ.sigma,
                champ.background,
                champ.amplitude,
            ):
                best = index
    return hypotheses[best]


def implied_curve(hyp: Hypothesis, slot_count: int = SLOT_COUNT) -> list[float]:
    return [
        bump_probability(slot, hyp.mu, hyp.sigma, hyp.background, hyp.amplitude)
        for slot in range(slot_count)
    ]


def posterior_mean_curve(
    history: History,
    hypotheses: Sequence[Hypothesis],
    shape_log_prior: Sequence[float] | None = None,
    slot_count: int = SLOT_COUNT,
    table: Sequence[Sequence[float]] | None = None,
) -> list[float]:
    probs = _softmax(log_posterior(history, hypotheses, shape_log_prior, table))
    curve = [0.0] * slot_count
    for weight, hyp in zip(probs, hypotheses):
        for slot in range(slot_count):
            curve[slot] += weight * bump_probability(
                slot, hyp.mu, hyp.sigma, hyp.background, hyp.amplitude
            )
    return curve


def shape_posterior_mass(
    history: History,
    hypotheses: Sequence[Hypothesis],
    *,
    sigma: float,
    atol: float = 1e-9,
    shape_log_prior: Sequence[float] | None = None,
    table: Sequence[Sequence[float]] | None = None,
) -> float:
    probs = _softmax(log_posterior(history, hypotheses, shape_log_prior, table))
    return float(
        sum(
            weight
            for weight, hyp in zip(probs, hypotheses)
            if abs(hyp.sigma - sigma) <= atol
        )
    )


def sigma_marginal(
    history: History,
    hypotheses: Sequence[Hypothesis],
    shape_log_prior: Sequence[float] | None = None,
    table: Sequence[Sequence[float]] | None = None,
) -> dict[float, float]:
    """Posterior marginal over bump width, summed across mu, a and b."""

    probs = _softmax(log_posterior(history, hypotheses, shape_log_prior, table))
    marginal: dict[float, float] = {}
    for weight, hyp in zip(probs, hypotheses):
        marginal[hyp.sigma] = marginal.get(hyp.sigma, 0.0) + weight
    return marginal


def posterior_mean_sigma(
    history: History,
    hypotheses: Sequence[Hypothesis],
    shape_log_prior: Sequence[float] | None = None,
    table: Sequence[Sequence[float]] | None = None,
) -> float:
    marginal = sigma_marginal(history, hypotheses, shape_log_prior, table)
    return float(sum(sigma * mass for sigma, mass in marginal.items()))


def sigma_mass_chance_level(
    hypotheses: Sequence[Hypothesis],
    *,
    sigma: float = TRUE_SIGMA,
    atol: float = SIGMA_TOLERANCE,
) -> float:
    """Mass a *non-learning* uniform posterior already puts in the same band.

    Reporting a width-recovery number without this is meaningless: the reader
    cannot tell 10% concentrated from 10% chance.
    """

    if not hypotheses:
        return 0.0
    hits = sum(1 for hyp in hypotheses if abs(hyp.sigma - sigma) <= atol)
    return hits / len(hypotheses)


def best_look_slot(
    history: History,
    hypotheses: Sequence[Hypothesis],
    slot_count: int,
    shape_log_prior: Sequence[float] | None = None,
    table: Sequence[Sequence[float]] | None = None,
    *,
    policy_seed: int,
    step: int,
) -> int:
    log_scores = log_posterior(history, hypotheses, shape_log_prior, table)
    prior = _softmax(log_scores)
    current_entropy = _entropy(prior)
    best_slot = 0
    best_gain = -1.0
    for slot in range(slot_count):
        expected = 0.0
        for y in (0, 1):
            p_y = 0.0
            child_logs: list[float] = []
            for index, (weight, score) in enumerate(zip(prior, log_scores)):
                p = table[index][slot] if table is not None else bump_probability(
                    slot,
                    hypotheses[index].mu,
                    hypotheses[index].sigma,
                    hypotheses[index].background,
                    hypotheses[index].amplitude,
                )
                pyh = p if y else (1.0 - p)
                p_y += weight * pyh
                child_logs.append(score + log(max(pyh, 1e-15)))
            expected += p_y * _entropy(_softmax(child_logs))
        gain = current_entropy - expected
        if gain > best_gain:
            best_gain = gain
            best_slot = slot
        elif abs(gain - best_gain) < 1e-12:
            tie = _hash_u64("emergent-tie", policy_seed, step, slot)
            champ = _hash_u64("emergent-tie", policy_seed, step, best_slot)
            if tie > champ:
                best_slot = slot
    return best_slot


def _entropy(probs: Sequence[float]) -> float:
    return -sum(p * log(p) for p in probs if p > 1e-15)


def _hash_u64(*parts: int | str) -> int:
    payload = ":".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")


def pool_shape_log_prior(
    labeled: Iterable[tuple[History, int]],
    hypotheses: Sequence[Hypothesis],
) -> list[float]:
    """Copy a shape-only log-prior onto every peak location.

    Completed episodes contribute Bernoulli likelihood given the *revealed*
    peak. The resulting ``(sigma, a, b)`` scores are reused for every ``mu``
    so a later episode can infer a new peak with the learned lamp physics.
    """

    shape_score: dict[tuple[float, float, float], float] = {}
    for history, theta in labeled:
        pairs = history_pairs(history)
        for hyp in hypotheses:
            if hyp.mu != theta:
                continue
            key = (hyp.sigma, hyp.background, hyp.amplitude)
            shape_score[key] = shape_score.get(key, 0.0) + log_likelihood_hyp(pairs, hyp)
    return [
        shape_score.get((hyp.sigma, hyp.background, hyp.amplitude), 0.0)
        for hyp in hypotheses
    ]


class EmergentBumpAgent:
    """Active world-model agent. No canonical likelihood constants.

    ``shape_log_prior`` may be produced by :func:`pool_shape_log_prior` after a
    training phase. When omitted, every ``(mu, sigma, a, b)`` starts equal and
    learning is strictly in-episode.
    """

    agent_id = "emergent-bump-world-model"
    agent_version = "0.1.0"

    def __init__(
        self,
        *,
        policy_seed: int,
        config: RoomConfig | None = None,
        shape_log_prior: Sequence[float] | None = None,
        condition: str = "emergent_in_episode",
    ) -> None:
        if isinstance(policy_seed, bool) or not isinstance(policy_seed, int):
            raise TypeError("policy_seed must be an integer")
        self.config = config or RoomConfig()
        self.policy_seed = policy_seed
        self.hypotheses, self.table = cached_grid(self.config)
        if shape_log_prior is not None and len(shape_log_prior) != len(self.hypotheses):
            raise ValueError("shape_log_prior must match the hypothesis grid")
        self.shape_log_prior = (
            None if shape_log_prior is None else tuple(float(x) for x in shape_log_prior)
        )
        self.shape_prior_digest = (
            None
            if self.shape_log_prior is None
            else "sha256:"
            + hashlib.sha256(
                json.dumps([round(x, 9) for x in self.shape_log_prior]).encode("utf-8")
            ).hexdigest()
        )
        self.condition = condition

    def start_episode(self, state: RoomState) -> None:
        del state

    def _choose_look(self, state: RoomState) -> int:
        history = history_pairs(state.history)
        return best_look_slot(
            history,
            self.hypotheses,
            self.config.slot_count,
            self.shape_log_prior,
            self.table,
            policy_seed=self.policy_seed,
            step=len(history),
        )

    def diagnose(self, state: RoomState) -> dict[str, Any]:
        history = history_pairs(state.history)
        hyp = map_hypothesis(
            history, self.hypotheses, self.shape_log_prior, self.table
        )
        curve = posterior_mean_curve(
            history,
            self.hypotheses,
            self.shape_log_prior,
            self.config.slot_count,
            self.table,
        )
        return {
            "mu": hyp.mu,
            "sigma": hyp.sigma,
            "background": hyp.background,
            "amplitude": hyp.amplitude,
            "sigma_mass_at_3": shape_posterior_mass(
                history,
                self.hypotheses,
                sigma=TRUE_SIGMA,
                shape_log_prior=self.shape_log_prior,
                table=self.table,
            ),
            "sigma_mass_near_3": shape_posterior_mass(
                history,
                self.hypotheses,
                sigma=TRUE_SIGMA,
                atol=SIGMA_TOLERANCE,
                shape_log_prior=self.shape_log_prior,
                table=self.table,
            ),
            "sigma_mass_chance": sigma_mass_chance_level(self.hypotheses),
            "mean_sigma": posterior_mean_sigma(
                history, self.hypotheses, self.shape_log_prior, self.table
            ),
            "curve": curve,
        }

    def act(self, state: RoomState) -> AgentDecision:
        if state.phase == "observe":
            action = self._choose_look(state)
            return AgentDecision(
                action=action,
                rationale="information-gain look under the inferred bump family",
                metadata={"step": len(state.history)},
            )
        diagnosis = self.diagnose(state)
        return AgentDecision(
            action=int(diagnosis["mu"]),
            rationale="MAP peak of the inferred bump family",
            metadata=diagnosis,
        )

    def run_metadata(self) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "hypothesis_family": "unimodal_gaussian_bump",
            "sigmas": list(SIGMAS),
            "backgrounds": list(BACKGROUNDS),
            "amplitudes": list(AMPLITUDES),
            "theta_candidates": list(self.config.theta_candidates),
            "canonical_formula_disclosed": False,
            "condition": self.condition,
            "pooled_shape_prior": self.shape_log_prior is not None,
            "shape_prior_digest": self.shape_prior_digest,
            "policy_seed": self.policy_seed,
        }
        # The pooled prior is this agent's main variable. Without pinning it,
        # two runs over different training pools are indistinguishable in the
        # manifest -- same failure the LLM prompt digest exists to prevent.
        canonical = json.dumps(metadata, sort_keys=True, separators=(",", ":"))
        metadata["configuration_digest"] = (
            "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        )
        return metadata


def collect_labeled_episode(
    env_factory,
    agent: EmergentBumpAgent,
    *,
    seed: int,
    theta: int | None = None,
) -> tuple[list[tuple[int, int]], int, TerminalOutcome]:
    """Run one episode and return (history, revealed theta, outcome)."""

    env = env_factory()
    state = env.reset(seed=seed, theta=theta)
    agent.start_episode(state)
    while state.phase == "observe":
        state = env.observe(agent.act(state).action)
    decision = agent.act(state)
    outcome = env.estimate(decision.action)
    history = [(obs.time_slot, 1 if obs.light_on else 0) for obs in state.history]
    return history, outcome.theta, outcome
