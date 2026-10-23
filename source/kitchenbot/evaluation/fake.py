"""The ``fake`` scenario: synthetic episodes generated from the seed, no simulation.

It exists so CI can exercise the whole evaluator path (runner, aggregation, schema, CLI) without
Isaac Sim. **Its numbers describe nothing**: the result document marks ``run.synthetic`` true and
nothing from a ``fake`` run may be copied into ``docs/experiments.md``.

Determinism: an episode is a pure function of ``(seed, split)``. All randomness comes from
``random.Random(seed)`` and only ``random()`` is used, whose output Python guarantees. The same
number of draws is made for every seed so adding a branch never shifts the others.

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

GENERATOR_VERSION = "fake-v1"

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

    success = not observed
    tag = FailureTag.SUCCESS if success else primary_failure_tag(observed)
    fell = FailureTag.FALL in observed

    completion = round(8.0 + 10.0 * completion_u, 3) if success else None
    if success:
        duration = completion + 2.0  # the 2 s stability hold after release
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
    return EpisodeResult(
        seed=seed,
        success=success,
        failure_tag=tag,
        duration_s=duration,
        metrics=metrics,
    )


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
