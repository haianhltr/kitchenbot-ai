"""Fake arm, hand, plate and rack for the plate-to-rack planner tests. No simulation.

The fakes moved to ``kitchenbot.planning.fakes`` in #60 so that the evaluator's dry run can use
them on CI; this module only re-exports the names the #43 and #58 tests import from here, so those
tests stay unchanged. New tests import from ``kitchenbot.planning.fakes``.
"""

from kitchenbot.planning.fakes import (
    ARM_ANGULAR_SPEED_RPS,
    ARM_LINEAR_SPEED_MPS,
    ATTACH_RADIUS_M,
    CLOSE_DURATION_S,
    COUNTER_EDGE,
    PALM_START,
    PLATE_GRASP,
    SHOULDER,
    SLOT_ENTRY,
    EpisodeLog,
    FakeArm,
    FakeHand,
    FakePlate,
    FakeRack,
    Fakes,
    run_episode,
)

__all__ = [
    "ARM_ANGULAR_SPEED_RPS",
    "ARM_LINEAR_SPEED_MPS",
    "ATTACH_RADIUS_M",
    "CLOSE_DURATION_S",
    "COUNTER_EDGE",
    "PALM_START",
    "PLATE_GRASP",
    "SHOULDER",
    "SLOT_ENTRY",
    "EpisodeLog",
    "FakeArm",
    "FakeHand",
    "FakePlate",
    "FakeRack",
    "Fakes",
    "run_episode",
]
