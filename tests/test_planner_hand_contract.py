"""The planner's hand contract: its verbs drive the #42 ``HandController`` (architecture sections 8
and 9), and the rim point the hand closes on is the one convention of #67.

The second half asserts that convention in one place: the grasp frame is the top rim point nearest
the robot (``kitchenbot.planning.geometry``), and the four consequences its docstring lists hold
(the quarter turn puts the point on top, the slot entry is a diameter above the tine tops, the
kinematic-follow attach distance, the judges' slot volume).
"""

import math

import numpy as np
import pytest
from eval_helpers import load_script
from planner_helpers import Fakes, run_episode

from kitchenbot.controllers.hand.base import HandController
from kitchenbot.controllers.hand.kinematic_follow import DEFAULT_ATTACH_DISTANCE_M, rim_distance
from kitchenbot.controllers.hand.mock import MockHand
from kitchenbot.evaluation.failure_tags import FailureTag
from kitchenbot.evaluation.judges import plate_in_slot
from kitchenbot.geometry import (
    Pose,
    plate_vertical_quat,
    pose_compose,
    quat_apply,
    quat_from_yaw,
    to_frame,
    yaw_from_quat,
)
from kitchenbot.planning.geometry import (
    GRASP_RIM_HEIGHT_M,
    TOWARD_ROBOT_KITCHEN,
    approach_body_pose,
    body_pose_from_grasp,
    grasp_in_body,
    grasp_pose_from_body,
    slot_entry_grasp_pose,
)
from kitchenbot.planning.plate_to_rack import HandAction, PlannerConfig, PlannerState
from kitchenbot.scene import kitchen_dims as kd

LAYOUT = kd.RackSlotLayout()
DIAMETERS = kd.PLATE_DIAMETERS_M
H = kd.PLATE_HEIGHT_M


def test_every_hand_action_is_an_abstract_hand_controller_method():
    for action in HandAction:
        assert action.value in HandController.__abstractmethods__, action
        assert callable(getattr(MockHand(), action.value)), action


def _mock_hand_episode(plate_in_reach: bool):
    hand = MockHand(plate_in_reach=plate_in_reach, close_duration=0.5)
    log = run_episode(fakes=Fakes(hand=hand, hand_reach_from_geometry=False))
    return hand, log


def test_nominal_episode_with_the_mock_hand_reaches_done():
    hand, log = _mock_hand_episode(plate_in_reach=True)
    assert log.final.state is PlannerState.DONE and log.final.failure is None
    assert log.entered_at(PlannerState.DONE) <= PlannerConfig().total_budget_s
    first = {verb: hand.commands.index(verb) for verb in ("pre_grasp", "close", "open")}
    assert first["pre_grasp"] < first["close"] < first["open"]


def test_mock_hand_out_of_reach_ends_in_grasp_miss():
    _, log = _mock_hand_episode(plate_in_reach=False)
    assert log.final.state is PlannerState.FAILED
    assert log.final.failure is FailureTag.GRASP_MISS
    assert log.planner.failed_in is PlannerState.GRASP


# ----------------------------------------------------------------------------- the rim convention


@pytest.mark.parametrize("diameter", DIAMETERS)
def test_the_grasp_point_is_the_top_rim_point_on_the_robot_side_in_body_coordinates(diameter):
    # (+D/2, 0, PLATE_HEIGHT_M) in the body frame turned so that its +x faces the robot; the frame's
    # x points back at the plate centre (a half turn about z) and z is the plate normal
    g = grasp_in_body(diameter)
    np.testing.assert_allclose(g.position, (diameter / 2.0, 0.0, H))
    assert GRASP_RIM_HEIGHT_M == H == 0.020
    np.testing.assert_allclose(quat_apply(g.quat_wxyz, (1, 0, 0)), (-1, 0, 0), atol=1e-12)
    np.testing.assert_allclose(quat_apply(g.quat_wxyz, (0, 0, 1)), (0, 0, 1), atol=1e-12)


@pytest.mark.parametrize("diameter", DIAMETERS)
def test_the_grasp_point_is_the_outer_top_corner_of_the_plate_asset(diameter):
    profile = load_script("make_plate_usd").plate_profile(diameter)
    radius, top = profile[3]  # outer top corner of the #30 cross-section
    np.testing.assert_allclose((radius, 0.0, top), grasp_in_body(diameter).position)


@pytest.mark.parametrize("yaw_deg", [-30.0, 0.0, 17.0, 30.0, 180.0])
@pytest.mark.parametrize("diameter", DIAMETERS)
def test_on_the_counter_the_grasp_frame_ignores_the_plate_yaw(diameter, yaw_deg):
    body = Pose((0.12, 0.30, kd.COUNTER_HEIGHT_M), quat_from_yaw(math.radians(yaw_deg)))
    grasp = grasp_pose_from_body(body, diameter)
    # kitchen -y is the robot's side; the point is the nearest top rim point to it
    assert TOWARD_ROBOT_KITCHEN == (0.0, -1.0, 0.0)
    np.testing.assert_allclose(grasp.position, (0.12, 0.30 - diameter / 2.0, 0.92), atol=1e-12)
    assert math.degrees(yaw_from_quat(grasp.quat_wxyz)) == pytest.approx(90.0)  # x into the counter
    rim = [
        body.position + np.array([diameter / 2 * math.cos(a), diameter / 2 * math.sin(a), H])
        for a in np.linspace(0.0, 2.0 * math.pi, 361)
    ]
    robot_far_away = np.array([0.12, -100.0, 0.92])
    nearest = min(rim, key=lambda p: float(np.linalg.norm(p - robot_far_away)))
    np.testing.assert_allclose(grasp.position, nearest, atol=1e-6)
    turned = approach_body_pose(body)
    np.testing.assert_allclose(quat_apply(turned.quat_wxyz, (1, 0, 0)), (0, -1, 0), atol=1e-12)


def test_a_direction_along_the_plate_normal_has_no_nearest_rim_point():
    flat = Pose((0.0, 0.3, 0.9))
    with pytest.raises(ValueError, match="normal"):
        grasp_pose_from_body(flat, 0.25, toward_robot=(0.0, 0.0, -1.0))
    with pytest.raises(ValueError, match="diameter"):
        grasp_in_body(0.0)


@pytest.mark.parametrize("slot_yaw_deg", [0.0, 90.0, -35.0])
@pytest.mark.parametrize("diameter", DIAMETERS)
def test_the_quarter_turn_puts_the_grasp_point_on_top_of_the_vertical_plate(diameter, slot_yaw_deg):
    # consequence 1: plate_vertical_quat tips the grasp frame's x down, so the centre is below it
    yaw = math.radians(slot_yaw_deg)
    grasp = Pose((0.4, -0.1, 1.0), plate_vertical_quat(yaw))
    body = body_pose_from_grasp(grasp, diameter)
    normal = quat_apply(body.quat_wxyz, (0, 0, 1))
    np.testing.assert_allclose(normal, (math.cos(yaw), math.sin(yaw), 0.0), atol=1e-12)
    top_centre = body.position + H * normal  # the centre of the plate's top face
    np.testing.assert_allclose(top_centre, grasp.position - np.array([0, 0, diameter / 2]))
    # every point of the top rim circle is at or below the grasp point
    u = np.cross(normal, (0, 0, 1))
    for a in np.linspace(0.0, 2.0 * math.pi, 73):
        p = top_centre + diameter / 2 * (math.cos(a) * u + math.sin(a) * np.array([0, 0, 1.0]))
        assert p[2] <= grasp.position[2] + 1e-12


@pytest.mark.parametrize("diameter", DIAMETERS)
@pytest.mark.parametrize("slot", range(1, kd.RACK_SLOTS + 1))
def test_the_slot_entry_is_a_diameter_above_the_tine_tops(slot, diameter):
    # consequence 2: RackSlotLayout.slot_pose keeps its meaning (gap centre at the tine tops)
    sx, sy, sz, yaw = LAYOUT.slot_pose(slot)
    entry = slot_entry_grasp_pose(LAYOUT, slot, diameter)
    np.testing.assert_allclose(entry.position, (sx + H / 2.0, sy, sz + diameter), atol=1e-12)
    assert yaw_from_quat(entry.quat_wxyz) == pytest.approx(yaw)
    frame = Pose((0.55, 0.0, 0.09), quat_from_yaw(0.3))
    in_world = slot_entry_grasp_pose(LAYOUT, slot, diameter, frame)
    np.testing.assert_allclose(in_world.position, pose_compose(frame, entry).position)


def _body_in_rack_at(entry: Pose, depth: float, diameter: float) -> Pose:
    """The plate body (rack frame) when the planner's grasp frame is ``depth`` below ``entry``."""
    grasp = Pose(entry.position - np.array([0.0, 0.0, depth]), plate_vertical_quat(entry.yaw))
    return body_pose_from_grasp(grasp, diameter)


@pytest.mark.parametrize("diameter", DIAMETERS)
@pytest.mark.parametrize("slot", range(1, kd.RACK_SLOTS + 1))
def test_the_judges_see_the_lowest_point_at_the_tine_tops_then_inside_the_slot(slot, diameter):
    # consequence 4: at the slot entry the judges' lowest rim point is at the tine tops, centred;
    # insertion_depth_m further down it is inside the slot volume, that far below the tine tops
    entry = slot_entry_grasp_pose(LAYOUT, slot, diameter)
    at_entry = plate_in_slot(_body_in_rack_at(entry, 0.0, diameter), slot, diameter, LAYOUT)
    assert at_entry.across_m == pytest.approx(0.0, abs=1e-12)
    assert at_entry.along_m == pytest.approx(0.0, abs=1e-12)
    assert at_entry.rim_bottom_m == pytest.approx(0.0, abs=1e-12)
    above = plate_in_slot(_body_in_rack_at(entry, -0.001, diameter), slot, diameter, LAYOUT)
    assert not above.in_slot  # 1 mm above the tine tops is not in the slot
    depth = PlannerConfig().insertion_depth_m
    seated = plate_in_slot(_body_in_rack_at(entry, depth, diameter), slot, diameter, LAYOUT)
    assert seated.in_slot, seated.explain()
    assert seated.rim_bottom_m == pytest.approx(-depth)
    assert seated.across_m == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("diameter", DIAMETERS)
def test_the_kinematic_follow_attach_distance_at_the_grasp_point(diameter):
    # consequence 3: measured to the rim circle of the root (body) pose, the underside plane, the
    # top rim point is one plate height away: inside the 3 cm attach distance with 1 cm to spare
    body = Pose((0.1, 0.3, kd.COUNTER_HEIGHT_M), quat_from_yaw(0.4))
    grasp = grasp_pose_from_body(body, diameter)
    assert rim_distance(grasp.position, body, diameter) == pytest.approx(H)
    assert rim_distance(grasp.position, body, diameter) <= DEFAULT_ATTACH_DISTANCE_M - 0.01 + 1e-12
    # measured to the top rim circle (a pose raised one plate height) it would be zero
    top = pose_compose(body, Pose((0.0, 0.0, H)))
    assert rim_distance(grasp.position, top, diameter) == pytest.approx(0.0, abs=1e-12)
    # and the plate centre in the grasp frame is a radius ahead and one plate height down
    np.testing.assert_allclose(to_frame(body, grasp).position, (diameter / 2, 0.0, -H), atol=1e-12)
