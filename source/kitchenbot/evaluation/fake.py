"""The ``fake`` scenario: synthetic episodes generated from the seed, no simulation.

It exists so CI can exercise the whole evaluator path (runner, aggregation, schema, CLI) without
Isaac Sim. **Its numbers describe nothing**: the result document marks ``run.synthetic`` true and
nothing from a ``fake`` run may be copied into ``docs/experiments.md``.

Determinism: an episode is a pure function of ``(seed, split)``. All randomness comes from
``random.Random(seed)`` and only ``random()`` is used, whose output Python guarantees. The same
number of draws is made for every seed so adding a branch never shifts the others.

Each episode also carries a synthetic 10 Hz torso tilt trace (``traces``), so the plots have
per-step data on CI.

The generator mimics S4: each failure cause of docs/evaluation.md section 7 is hit independently
with a small probability, so several causes can occur in one trial and the precedence in
``failure_tags.py`` decides the primary tag. Unseen seeds fail 1.5 times as often.
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from typing import Any

from kitchenbot.evaluation.failure_tags import FAILURE_PRECEDENCE, FailureTag, primary_failure_tag
from kitchenbot.evaluation.runner import EpisodeResult, EvalConfig

#: Part of ``resolved_config`` and so of ``run.config_hash``. fake-v3 (#78): a success's
#: ``duration_s`` equals its completion time instead of 2 s more.
GENERATOR_VERSION = "fake-v3"

#: Probability that each failure cause is hit in a seen trial.
SEEN_CAUSE_PROBABILITY: dict[FailureTag, float] = {
    FailureTag.FALL: 0.02,
    FailureTag.IK_INFEASIBLE: 0.02,
    FailureTag.GRASP_MISS: 0.06,
    FailureTag.DROP: 0.03,
    FailureTag.RACK_COLLISION: 0.04,
    FailureTag.MISPLACE: 0.04,
    FailureTag.TIMEOUT: 0.03,
}

#: Multiplier on the failure probabilities for the unseen split.
UNSEEN_FACTOR = 1.5

#: Probability of touching the rack above the force threshold in a trial that is not tagged so.
BENIGN_CONTACT_PROBABILITY = 0.05

#: S4 timeout in seconds (docs/evaluation.md section 3).
TIMEOUT_S = 30.0

#: Sample rate of the synthetic torso trace.
TRACE_RATE_HZ = 10.0

_NO_GRASP = frozenset({FailureTag.FALL, FailureTag.IK_INFEASIBLE, FailureTag.GRASP_MISS})


def synthetic_episode(seed: int, split: str) -> EpisodeResult:
    """The synthetic S4-like episode for ``seed``. Deterministic; see the module docstring."""
    rng = random.Random(seed)
    factor = UNSEEN_FACTOR if split == "unseen" else 1.0
    observed = [
        tag
        for tag in FAILURE_PRECEDENCE
        if rng.random() < min(1.0, SEEN_CAUSE_PROBABILITY[tag] * factor)
    ]
    benign_contact = rng.random() < BENIGN_CONTACT_PROBABILITY
    torso_u = rng.random()
    completion_u = rng.random()
    failure_time_u = rng.random()
    # Drawn after everything above, so the fields above do not change when the trace does.
    trace_phase_u = rng.random()
    trace_period_u = rng.random()
    trace_peak_u = rng.random()

    success = not observed
    tag = FailureTag.SUCCESS if success else primary_failure_tag(observed)
    fell = FailureTag.FALL in observed

    completion = round(8.0 + 10.0 * completion_u, 3) if success else None
    if success:
        # completion_time_s is when the judges' 2 s stability window after release closes
        # (judges.PlacementJudge), so it already holds the 2 s; the episode ends there (#78).
        duration = completion
    elif tag is FailureTag.TIMEOUT:
        duration = TIMEOUT_S
    else:
        duration = 3.0 + 25.0 * failure_time_u

    metrics: dict[str, Any] = {
        "fell": fell,
        "torso_orientation_error_deg": round(1.5 + 3.0 * torso_u + (10.0 if fell else 0.0), 3),
        "grasp_success": not (_NO_GRASP & set(observed)),
        "placement_success": success,
        "rack_collision": FailureTag.RACK_COLLISION in observed or benign_contact,
        "completion_time_s": completion,
    }
    traces = {
        "sample_rate_hz": TRACE_RATE_HZ,
        "torso_orientation_error_deg": _torso_trace(
            level=metrics["torso_orientation_error_deg"],
            duration=duration,
            phase=trace_phase_u,
            period_s=2.0 + 4.0 * trace_period_u,
            fall_peak_deg=45.0 + 15.0 * trace_peak_u if fell else None,
        ),
    }
    return EpisodeResult(
        seed=seed,
        success=success,
        failure_tag=tag,
        duration_s=duration,
        metrics=metrics,
        traces=traces,
    )


def _torso_trace(
    level: float, duration: float, phase: float, period_s: float, fall_peak_deg: float | None
) -> list[float]:
    """A synthetic torso tilt series at ``TRACE_RATE_HZ``: a triangle wave around ``level``.

    Only exact arithmetic (no sin or exp), so the output is identical on every platform. A fall
    ramps the last 1.5 s linearly up to ``fall_peak_deg``.
    """
    n = max(2, int(duration * TRACE_RATE_HZ))
    series = []
    for k in range(n):
        t = k / TRACE_RATE_HZ
        x = (t / period_s + phase) % 1.0
        triangle = 4.0 * abs(x - 0.5) - 1.0  # -1 to 1
        value = level * (1.0 + 0.4 * triangle)
        if fall_peak_deg is not None and t > duration - 1.5:
            fraction = (t - (duration - 1.5)) / 1.5
            value += fraction * (fall_peak_deg - value)
        series.append(round(max(0.0, value), 3))
    return series


class FakeBackend:
    """Backend for the ``fake`` scenario. Holds no resources."""

    def __init__(self, config: EvalConfig) -> None:
        self._split = config.split

    def resolved_config(self) -> Mapping[str, Any]:
        return {
            "generator": GENERATOR_VERSION,
            "seen_cause_probability": {t.value: p for t, p in SEEN_CAUSE_PROBABILITY.items()},
            "unseen_factor": UNSEEN_FACTOR,
            "benign_contact_probability": BENIGN_CONTACT_PROBABILITY,
        }

    def run_batch(self, seeds: Sequence[int]) -> list[EpisodeResult]:
        return [synthetic_episode(seed, self._split) for seed in seeds]

    def close(self) -> None:
        return None
