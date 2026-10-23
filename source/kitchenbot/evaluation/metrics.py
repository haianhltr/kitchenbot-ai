"""Metric catalogue (docs/evaluation.md section 3): which metrics each scenario reports.

This is the single place that says which per-episode metrics exist and which scenarios they apply
to. ``schema.json`` repeats the same sets in JSON Schema form and a unit test checks that the two
agree. This module only names the metrics.

Deciding them is :mod:`kitchenbot.evaluation.judges` (#57), the reference implementation of the
section 3 definitions, written from state alone: ``GraspJudge`` gives ``grasp_success``,
``PlacementJudge`` gives ``placement_success``, ``rack_collision`` and ``completion_time_s``, and
``episode_outcome`` turns those and the fall, IK, drop and timeout signals into ``success`` and
the primary failure tag. The simulation backends (#22) feed the judges scene state; the CI dry run
(#60) feeds them synthetic trajectories; both therefore mean the same thing by success.

Per-episode ``metrics`` keys
----------------------------
- ``fell`` (bool, all scenarios): pelvis below 0.5 m or torso tilt above 60 deg.
- ``standing_duration_s`` (number, S0): seconds until termination.
- ``torso_orientation_error_deg`` (number, S0, S1, S4): RMS of torso roll and pitch.
- ``hand_target_error_cm`` (number, S1): final hand-to-target distance.
- ``grasp_success`` (bool, S2, S4): plate lifted 15 cm and held for 2 s.
- ``placement_success`` (bool, S3, S4): plate in the slot volume, stable for 2 s.
- ``rack_collision`` (bool, S3, S4): contact force above 20 N before release.
- ``completion_time_s`` (number, S4): seconds to stable placement, ``None`` unless the trial
  succeeded.

``rack_collision`` is a metric, not the failure tag of the same name: a trial can touch the rack
above the force threshold and still succeed, and the collision rate counts it.

The scenario ``fake`` reports the S4 set, so CI exercises every kind of metric except the S0 and
S1 ones.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Section 3: "fraction under 3 cm" for the S1 hand target error.
HAND_TARGET_THRESHOLD_CM = 3.0

#: Name of the derived rate computed from ``hand_target_error_cm``.
HAND_TARGET_RATE = "hand_target_under_3cm_rate"

#: Scenarios the evaluator knows. ``fake`` is the synthetic scenario that runs on CI.
REAL_SCENARIOS: tuple[str, ...] = ("S0", "S1", "S2", "S3", "S4")
SCENARIO_NAMES: tuple[str, ...] = (*REAL_SCENARIOS, "fake")

#: What ``success`` means per scenario, quoted from docs/evaluation.md sections 3 and 5. The
#: scenario backend decides it from simulation state; the runner only checks that ``success`` and
#: ``failure_tag`` agree.
SUCCESS_DEFINITIONS: dict[str, str] = {
    "S0": "no fall until the timeout (section 5: 60 s without a fall)",
    "S1": "final hand error under 3 cm and no fall",
    "S2": "grasp success: plate lifted 15 cm and held for 2 s",
    "S3": "placement success: plate in the slot volume, within 15 deg of vertical, stable 2 s",
    "S4": "task success: grasp and placement both succeed, no fall, within the 30 s timeout",
    "fake": "synthetic: no failure cause drawn for the seed (mirrors S4)",
}


@dataclass(frozen=True)
class MetricSpec:
    """One per-episode metric.

    ``kind`` is ``"bool"`` (aggregated as a rate with a Wilson interval, under ``rate_name``) or
    ``"number"`` (aggregated as n, mean, std, median). ``nullable`` numbers may be ``None``.
    """

    name: str
    kind: str
    scenarios: frozenset[str]
    rate_name: str | None = None
    nullable: bool = False


_S4_LIKE = {"S4", "fake"}

METRICS: dict[str, MetricSpec] = {
    m.name: m
    for m in (
        MetricSpec("fell", "bool", frozenset((*REAL_SCENARIOS, "fake")), rate_name="fall_rate"),
        MetricSpec("standing_duration_s", "number", frozenset({"S0"})),
        MetricSpec("torso_orientation_error_deg", "number", frozenset({"S0", "S1"} | _S4_LIKE)),
        MetricSpec("hand_target_error_cm", "number", frozenset({"S1"})),
        MetricSpec(
            "grasp_success",
            "bool",
            frozenset({"S2"} | _S4_LIKE),
            rate_name="grasp_success_rate",
        ),
        MetricSpec(
            "placement_success",
            "bool",
            frozenset({"S3"} | _S4_LIKE),
            rate_name="placement_success_rate",
        ),
        MetricSpec(
            "rack_collision", "bool", frozenset({"S3"} | _S4_LIKE), rate_name="collision_rate"
        ),
        MetricSpec("completion_time_s", "number", frozenset(_S4_LIKE), nullable=True),
    )
}


def metrics_for(scenario: str) -> tuple[str, ...]:
    """Names of the per-episode metrics ``scenario`` reports, in catalogue order."""
    if scenario not in SCENARIO_NAMES:
        raise ValueError(f"unknown scenario {scenario!r}; expected one of {SCENARIO_NAMES}")
    return tuple(name for name, spec in METRICS.items() if scenario in spec.scenarios)


def rate_names_for(scenario: str) -> tuple[str, ...]:
    """Names of the rates in ``aggregates.rates`` for ``scenario``, in catalogue order."""
    names: list[str] = []
    for name in metrics_for(scenario):
        spec = METRICS[name]
        if spec.rate_name:
            names.append(spec.rate_name)
        if name == "hand_target_error_cm":
            names.append(HAND_TARGET_RATE)
    return tuple(names)


def continuous_names_for(scenario: str) -> tuple[str, ...]:
    """Names of the numeric metrics in ``aggregates.continuous`` for ``scenario``."""
    return tuple(n for n in metrics_for(scenario) if METRICS[n].kind == "number")
