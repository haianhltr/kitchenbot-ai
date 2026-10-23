"""The lift waypoint is aimed one position tolerance above the protocol's 15 cm (#60).

The ``lift`` state exits within ``position_tolerance_m`` of its waypoint. Aimed at exactly 15 cm,
the exit could leave the plate at 13 cm and a straight slew to the next waypoint never rose above
14.9 cm, below what the grasp judge asks for (docs/evaluation.md section 3). The waypoint is now
``lift_height_m + position_tolerance_m`` above the grasp frame, so the exit height is at least
``lift_height_m``. ``PlannerConfig.lift_height_m`` stays 0.15, the number the judges read.
"""

import dataclasses

import numpy as np
import pytest
from planner_helpers import COUNTER_EDGE, PLATE_GRASP, SLOT_ENTRY, Fakes

from kitchenbot.evaluation.judges import GRASP_HOLD_S, GRASP_LIFT_HEIGHT_M, GraspJudge
from kitchenbot.geometry import Pose
from kitchenbot.planning.plate_to_rack import (
    PlannerConfig,
    PlannerState,
    PlateToRackPlanner,
    plan_waypoints,
)

UP = np.array([0.0, 0.0, 1.0])
CONFIGS = [
    PlannerConfig(),
    PlannerConfig(position_tolerance_m=0.03),
    PlannerConfig(lift_height_m=0.20, position_tolerance_m=0.01),
]
IDS = ["default", "tolerance-3cm", "lift-20cm-tolerance-1cm"]


def _waypoints(cfg):
    return plan_waypoints(PLATE_GRASP, SLOT_ENTRY, COUNTER_EDGE, cfg)


def test_the_protocol_lift_stays_15_cm_and_the_judge_reads_it():
    cfg = PlannerConfig()
    assert cfg.lift_height_m == 0.15 == GRASP_LIFT_HEIGHT_M
    assert cfg.position_tolerance_m == 0.02
    assert (
        "one position tolerance higher"
        in {f.name: f for f in dataclasses.fields(cfg)}["lift_height_m"].metadata["doc"]
    )


@pytest.mark.parametrize("cfg", CONFIGS, ids=IDS)
def test_the_lift_waypoint_is_the_lift_height_plus_the_tolerance_above_the_grasp(cfg):
    wp = _waypoints(cfg)
    expected = PLATE_GRASP.position + (cfg.lift_height_m + cfg.position_tolerance_m) * UP
    np.testing.assert_allclose(wp["lift"].position, expected, atol=1e-12)
    np.testing.assert_allclose(wp["lift"].quat_wxyz, PLATE_GRASP.quat_wxyz)  # carried flat
    # the next waypoint is at least as high, so the plate does not sink back below the lift
    assert wp["counter_edge"].position[2] >= wp["lift"].position[2] - 1e-12


def test_a_palm_a_full_tolerance_below_the_waypoint_still_satisfies_the_grasp_judge():
    cfg = PlannerConfig()
    lift = _waypoints(cfg)["lift"]
    palm_z = lift.position[2] - cfg.position_tolerance_m  # the lowest the exit test accepts
    plate_rise = palm_z - PLATE_GRASP.position[2]  # the plate rides the palm rigidly
    assert plate_rise == pytest.approx(GRASP_LIFT_HEIGHT_M, abs=1e-12)
    judge = GraspJudge()
    start = Pose(PLATE_GRASP.position, PLATE_GRASP.quat_wxyz)
    judge.reset(start)
    plate = Pose(start.position + plate_rise * UP, start.quat_wxyz)
    t = 0.0
    while t <= GRASP_HOLD_S + 0.1 and not judge.grasp_success:
        judge.step(t, plate, holding=True)
        t = round(t + 0.02, 6)
    assert judge.grasp_success, judge.why()
    assert judge.lifted_at_s == 0.0


def test_the_lift_state_exits_at_a_full_tolerance_below_and_not_beyond_it():
    cfg = PlannerConfig()
    tol = cfg.position_tolerance_m
    dt = 0.02
    for margin, state in ((tol - 1e-9, PlannerState.TRANSPORT), (tol + 1e-3, PlannerState.LIFT)):
        fakes = Fakes()
        planner = PlateToRackPlanner(cfg)
        planner.reset(fakes.observe())
        for _ in range(2000):  # up to the lift state
            if planner.state is PlannerState.LIFT:
                break
            fakes.apply(planner.step(fakes.observe(), dt), dt)
        assert planner.state is PlannerState.LIFT
        lift = planner.waypoints["lift"]
        palm = Pose(lift.position - margin * UP, lift.quat_wxyz)
        obs = dataclasses.replace(fakes.observe(), palm_pose=palm, hand_holding=True)
        planner.step(obs, dt)
        assert planner.state is state, margin
        if state is PlannerState.TRANSPORT:  # the exit height is at least the protocol's 15 cm
            assert palm.position[2] - PLATE_GRASP.position[2] >= cfg.lift_height_m - 1e-9
