"""The planner's hand verbs drive KinematicFollowHand through the fakes (#58, contract of #43).

``planner_helpers.Fakes`` holds the plate's grasp frame on the rim (what the planner observes,
docs/architecture.md section 9); the hand works on the plate's root (body) pose. The wiring below
converts between the two with the grasp convention of ``kitchenbot.planning.geometry`` (#67: the
top rim point nearest the robot), the way the environment layer will, and lets the hand write the
plate pose itself (``plate_follows_palm=False``). Still no simulation: this checks that the verbs,
the attach rule and the follow compose with the planner, not that anything is grasped.
"""

import numpy as np
import pytest
from planner_helpers import PLATE_GRASP, Fakes, run_episode

from kitchenbot.controllers.hand.base import HandController, HandPosture
from kitchenbot.controllers.hand.kinematic_follow import (
    DEFAULT_ATTACH_DISTANCE_M,
    KinematicFollowHand,
    rim_distance,
)
from kitchenbot.evaluation.failure_tags import FailureTag
from kitchenbot.geometry import Pose, pose_compose, pose_inverse, quat_angle
from kitchenbot.planning.geometry import body_pose_from_grasp
from kitchenbot.planning.plate_to_rack import HandAction, PlannerConfig, PlannerState
from kitchenbot.scene.kitchen_dims import PLATE_HEIGHT_M

DIAMETER = 0.25
#: The plate's root (body) pose in the grasp frame, from the one convention (#67): a radius along
#: the grasp frame's +x (toward the centre) and one plate height down, ``(D/2, 0, -0.02)``.
CENTRE_IN_GRASP = body_pose_from_grasp(Pose((0.0, 0.0, 0.0)), DIAMETER)
PALM_VELOCITY = ((0.10, 0.00, 0.05), (0.0, 0.0, 0.20))


class Wiring:
    """A KinematicFollowHand connected to a ``Fakes``, with a log of what it did."""

    def __init__(self, fakes: Fakes):
        self.fakes = fakes
        self.calls: list[tuple] = []
        self.follow: list[tuple[Pose, Pose]] = []  # (palm, plate grasp frame) after each write
        self.hand = KinematicFollowHand(
            "right",
            palm_pose=lambda: fakes.arm.palm,
            palm_velocity=lambda: PALM_VELOCITY,
            plate_pose=lambda: pose_compose(fakes.plate.pose, CENTRE_IN_GRASP),
            plate_diameter_m=DIAMETER,
            set_plate_kinematic=lambda kinematic: self.calls.append(("kinematic", kinematic)),
            write_plate_pose=self._write,
            set_plate_velocity=lambda lin, ang: self.calls.append(("velocity", lin, ang)),
        )
        fakes.hand = self.hand
        fakes.hand_reach_from_geometry = False  # the hand decides from its own rim distance
        fakes.plate_follows_palm = False  # the hand writes the plate pose

    def _write(self, centre: Pose) -> None:
        self.fakes.plate.pose = pose_compose(centre, pose_inverse(CENTRE_IN_GRASP))
        self.follow.append((self.fakes.arm.palm, self.fakes.plate.pose))


def wired(**fakes_kwargs) -> Wiring:
    return Wiring(Fakes(**fakes_kwargs))


def test_the_centre_offset_is_the_top_rim_convention():
    np.testing.assert_allclose(
        CENTRE_IN_GRASP.position, (DIAMETER / 2.0, 0.0, -PLATE_HEIGHT_M), atol=1e-12
    )


def test_a_palm_exactly_at_the_grasp_frame_is_one_plate_height_from_the_hands_rim_circle():
    # the hand measures to the rim circle in the plane of the pose it is given, here the root pose
    # (underside): the top rim point is 2 cm above it, inside the 3 cm attach distance by 1 cm
    fakes = Fakes()
    root = pose_compose(fakes.plate.pose, CENTRE_IN_GRASP)
    distance = rim_distance(fakes.plate.pose.position, root, DIAMETER)
    assert distance == pytest.approx(PLATE_HEIGHT_M)
    assert DEFAULT_ATTACH_DISTANCE_M - distance == pytest.approx(0.01)


def test_the_hand_exposes_every_planner_verb():
    hand = wired().hand
    assert isinstance(hand, HandController)
    for action in HandAction:
        assert callable(getattr(hand, action.value)), action


def test_nominal_episode_reaches_done_and_the_plate_is_released_in_the_slot():
    wiring = wired()
    log = run_episode(fakes=wiring.fakes)
    assert log.final.state is PlannerState.DONE and log.final.failure is None
    assert log.entered_at(PlannerState.DONE) <= PlannerConfig().total_budget_s
    assert wiring.fakes.rack.in_slot(wiring.fakes.plate.pose.position)
    assert not wiring.hand.is_holding()
    assert wiring.hand.posture is HandPosture.OPEN


def test_the_plate_is_kinematic_only_between_the_grasp_and_the_release():
    wiring = wired()
    log = run_episode(fakes=wiring.fakes)
    handed_over = ("velocity", *PALM_VELOCITY)
    assert wiring.calls == [("kinematic", True), ("kinematic", False), handed_over]
    states = log.states
    assert states.index(PlannerState.GRASP) < states.index(PlannerState.RELEASE)


def test_the_plate_rides_rigidly_with_the_palm_through_the_lift():
    wiring = wired()
    start = PLATE_GRASP.position.copy()
    log = run_episode(fakes=wiring.fakes)
    assert log.final.state is PlannerState.DONE
    assert len(wiring.follow) > 100  # written every step from the grasp to the release
    offsets = [pose_compose(pose_inverse(palm), plate) for palm, plate in wiring.follow]
    for offset in offsets:
        assert np.linalg.norm(offset.position - offsets[0].position) < 1e-9
        assert quat_angle(offset.quat_wxyz, offsets[0].quat_wxyz) < 1e-9
    # the lift state exits within the planner's 2 cm of the +15 cm waypoint, so at least 13 cm
    lifted = max(plate.position[2] for _, plate in wiring.follow) - start[2]
    assert lifted >= 0.15 - 0.02 - 1e-9
    # the palm stopped within the planner's 2 cm of the grasp frame, so the offset is that small
    assert float(np.linalg.norm(offsets[0].position)) <= 0.02 + 1e-9


def test_a_grasp_frame_5_cm_off_is_a_grasp_miss_and_nothing_is_attached():
    # the planner is told the plate is 5 cm away from where it is, so the palm closes 5 cm from the
    # rim: beyond the 3 cm attach distance
    wiring = wired(plate_observation_offset=(0.05, 0.0, 0.0))
    log = run_episode(fakes=wiring.fakes)
    assert log.final.state is PlannerState.FAILED
    assert log.final.failure is FailureTag.GRASP_MISS
    assert log.planner.failed_in is PlannerState.GRASP
    assert wiring.hand.posture is HandPosture.CLOSED
    assert not wiring.hand.is_holding()
    assert wiring.calls == []  # the plate never became kinematic
    assert wiring.follow == []


@pytest.mark.parametrize("offset", [(0.0, 0.015, 0.0), (0.01, 0.0, -0.01)])
def test_a_small_observation_error_still_attaches(offset):
    wiring = wired(plate_observation_offset=offset)
    log = run_episode(fakes=wiring.fakes)
    assert log.final.state is PlannerState.DONE
    assert wiring.calls[0] == ("kinematic", True)
