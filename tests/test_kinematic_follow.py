"""KinematicFollowHand logic with fake callables (#58). No simulator needed.

The hand is a labelled baseline, not a grasp: these tests pin the attach rule, the follow
mathematics and the release, not any physical behaviour.
"""

import math
import subprocess
import sys

import numpy as np
import pytest

from kitchenbot.controllers.hand.base import HandController, HandPosture
from kitchenbot.controllers.hand.kinematic_follow import (
    DEFAULT_ATTACH_DISTANCE_M,
    KinematicFollowHand,
    rim_distance,
)
from kitchenbot.geometry import (
    Pose,
    pose_compose,
    pose_inverse,
    quat_angle,
    quat_from_axis_angle,
    quat_from_yaw,
)

DT = 0.02  # 50 Hz control step, docs/architecture.md section 5
DIAMETER = 0.25
RADIUS = DIAMETER / 2.0
VERB = {HandPosture.OPEN: "open", HandPosture.PRE_GRASP: "pre_grasp", HandPosture.CLOSED: "close"}

#: A plate lying yawed on the counter; its local z is the plate normal.
PLATE = Pose((0.50, 0.10, 0.90), quat_from_yaw(0.3))
#: A palm tilted off every axis, so a frame mix-up cannot hide behind an identity rotation.
TILTED_PALM = Pose((0.40, -0.05, 1.05), quat_from_axis_angle((1.0, -2.0, 0.5), 1.1))
VELOCITY = ((0.10, -0.20, 0.30), (0.0, 0.50, -0.40))


def local_point(plate: Pose, x: float, y: float, z: float) -> np.ndarray:
    """World position of the point ``(x, y, z)`` given in the plate frame."""
    return pose_compose(plate, Pose((x, y, z))).position


def near_rim(plate: Pose, height: float, angle: float = 0.0) -> Pose:
    """A palm pose ``height`` above the rim point at ``angle`` around the plate axis."""
    point = local_point(plate, RADIUS * math.cos(angle), RADIUS * math.sin(angle), height)
    return Pose(point, TILTED_PALM.quat_wxyz)


def assert_same_pose(a: Pose, b: Pose, tol: float = 1e-9) -> None:
    assert np.linalg.norm(a.position - b.position) < tol, (a, b)
    assert quat_angle(a.quat_wxyz, b.quat_wxyz) < tol, (a, b)


class Fakes:
    """Sensing values the test sets, and every call the hand made, in order."""

    def __init__(self, palm: Pose | None = None, plate: Pose | None = None):
        self.palm = palm if palm is not None else near_rim(PLATE, 0.0)
        self.plate = plate if plate is not None else PLATE
        self.velocity = VELOCITY
        self.calls: list[tuple] = []

    def write(self, pose: Pose) -> None:
        self.calls.append(("pose", pose))
        self.plate = pose  # a kinematic body goes where it is told

    def hand(self, side="right", **kwargs) -> KinematicFollowHand:
        kwargs.setdefault("plate_diameter_m", DIAMETER)
        return KinematicFollowHand(
            side,
            palm_pose=lambda: self.palm,
            palm_velocity=lambda: self.velocity,
            plate_pose=lambda: self.plate,
            set_plate_kinematic=lambda kinematic: self.calls.append(("kinematic", kinematic)),
            write_plate_pose=self.write,
            set_plate_velocity=lambda lin, ang: self.calls.append(("velocity", lin, ang)),
            **kwargs,
        )

    def verbs(self, kind: str) -> list[tuple]:
        return [c for c in self.calls if c[0] == kind]


# --- rim_distance ---------------------------------------------------------------------------


PLATE_POSES = {
    "identity": Pose((0.0, 0.0, 0.0)),
    "yawed": PLATE,
    "tilted": Pose((-0.3, 0.7, 1.2), quat_from_axis_angle((1.0, 2.0, -0.5), 2.0)),
    "vertical": Pose((0.1, 0.2, 0.3), quat_from_axis_angle((0.0, 1.0, 0.0), math.pi / 2)),
}


@pytest.mark.parametrize("plate", PLATE_POSES.values(), ids=PLATE_POSES.keys())
def test_rim_distance_is_zero_on_the_rim(plate):
    for angle in np.linspace(0.0, 2.0 * math.pi, 9):
        point = local_point(plate, RADIUS * math.cos(angle), RADIUS * math.sin(angle), 0.0)
        assert rim_distance(point, plate, DIAMETER) == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("plate", PLATE_POSES.values(), ids=PLATE_POSES.keys())
def test_rim_distance_at_the_centre_is_the_radius(plate):
    assert rim_distance(plate.position, plate, DIAMETER) == pytest.approx(RADIUS)
    assert rim_distance(plate.position, plate, 0.30) == pytest.approx(0.15)


@pytest.mark.parametrize("plate", PLATE_POSES.values(), ids=PLATE_POSES.keys())
def test_rim_distance_above_and_below_the_rim_is_the_height(plate):
    for angle in (0.0, 1.0, 2.5, -2.0):
        c, s = math.cos(angle), math.sin(angle)
        for height in (0.01, 0.07, -0.07):
            point = local_point(plate, RADIUS * c, RADIUS * s, height)
            assert rim_distance(point, plate, DIAMETER) == pytest.approx(abs(height))


@pytest.mark.parametrize("plate", PLATE_POSES.values(), ids=PLATE_POSES.keys())
def test_rim_distance_combines_radial_and_axial_offsets(plate):
    # (radial - radius, axial) = (+3 cm, 4 cm) and (-3 cm, 4 cm): both are 5 cm from the ring
    for radial in (RADIUS + 0.03, RADIUS - 0.03):
        point = local_point(plate, 0.6 * radial, 0.8 * radial, -0.04)
        assert rim_distance(point, plate, DIAMETER) == pytest.approx(0.05)
    # in the plate plane, off the rim, the distance is the radial gap only
    assert rim_distance(local_point(plate, 0.0, RADIUS + 0.02, 0.0), plate, DIAMETER) == (
        pytest.approx(0.02)
    )
    assert rim_distance(local_point(plate, 0.05, 0.0, 0.0), plate, DIAMETER) == (
        pytest.approx(RADIUS - 0.05)
    )


def test_rim_distance_follows_the_plate_pose_not_the_world_axes():
    flat = PLATE_POSES["identity"]
    vertical = PLATE_POSES["vertical"]  # normal along world x, local x along world -z
    # the same world offset, 0.3 m straight up from the plate centre, is over the middle of a flat
    # plate but in the plane of a vertical one
    up = (0.0, 0.0, 0.3)
    over_centre = rim_distance(flat.position + up, flat, DIAMETER)
    assert over_centre == pytest.approx(math.hypot(RADIUS, 0.3))
    assert rim_distance(vertical.position + up, vertical, DIAMETER) == pytest.approx(0.3 - RADIUS)
    # a point on the vertical plate's rim, then 7 cm off its face (along world x)
    rim = vertical.position + np.array([0.0, 0.0, -RADIUS])
    assert rim_distance(rim, vertical, DIAMETER) == pytest.approx(0.0, abs=1e-12)
    assert rim_distance(rim + np.array([0.07, 0.0, 0.0]), vertical, DIAMETER) == pytest.approx(0.07)


def test_rim_distance_returns_a_float_and_validates_the_diameter():
    assert isinstance(rim_distance((0.0, 0.0, 0.0), PLATE, DIAMETER), float)
    for bad in (0.0, -0.25, math.nan, math.inf):
        with pytest.raises(ValueError, match="diameter_m"):
            rim_distance((0.0, 0.0, 0.0), PLATE, bad)
    with pytest.raises(ValueError):
        rim_distance((0.0, 0.0), PLATE, DIAMETER)


# --- the HandController contract ------------------------------------------------------------


def test_starts_open_and_calls_nothing():
    fakes = Fakes()
    hand = fakes.hand()
    assert isinstance(hand, HandController)
    assert hand.posture is HandPosture.OPEN
    assert hand.is_holding() is False
    assert hand.offset is None
    hand.update(DT)
    hand.update(DT)
    assert fakes.calls == []  # constructing and updating while open touch nothing


def test_posture_is_read_only_and_verbs_map_to_postures():
    hand = Fakes().hand("left")
    assert hand.side == "left"
    with pytest.raises(AttributeError):
        hand.posture = HandPosture.CLOSED
    for posture, verb in VERB.items():
        getattr(hand, verb)()
        assert hand.posture is posture


def test_negative_dt_is_rejected_zero_is_fine():
    hand = Fakes().hand()
    with pytest.raises(ValueError):
        hand.update(-0.01)
    hand.update(0.0)


def test_constructor_validates_its_arguments():
    fakes = Fakes()
    with pytest.raises(ValueError, match="side"):
        fakes.hand("middle")
    for bad in (-0.01, math.nan, math.inf):
        with pytest.raises(ValueError, match="attach_distance_m"):
            fakes.hand(attach_distance_m=bad)
    for bad in (0.0, -0.25, math.nan):
        with pytest.raises(ValueError, match="plate_diameter_m"):
            fakes.hand(plate_diameter_m=bad)
    assert fakes.hand().attach_distance_m == DEFAULT_ATTACH_DISTANCE_M == 0.03
    assert fakes.hand(attach_distance_m=0.0).attach_distance_m == 0.0


def test_module_imports_without_simulator_packages():
    code = (
        "import sys\n"
        "import kitchenbot.controllers.hand.kinematic_follow\n"
        "bad = ('isaaclab', 'isaacsim', 'omni', 'torch', 'pxr')\n"
        "print(sorted(m for m in sys.modules if m.split('.')[0] in bad))\n"
    )
    out = subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)
    assert out.stdout.strip() == "[]"


# --- the attach rule ------------------------------------------------------------------------


@pytest.mark.parametrize("start", [HandPosture.OPEN, HandPosture.PRE_GRASP], ids=str)
def test_close_on_the_rim_attaches_and_switches_the_plate_to_kinematic(start):
    fakes = Fakes()
    hand = fakes.hand()
    getattr(hand, VERB[start])()
    assert not hand.is_holding()
    hand.close()
    assert hand.posture is HandPosture.CLOSED
    assert hand.is_holding()
    assert fakes.calls == [("kinematic", True)]


def test_attach_distance_is_a_closed_bound_of_the_rim_distance():
    for height in (0.0, 0.025, 0.029):
        fakes = Fakes(palm=near_rim(PLATE, height))
        hand = fakes.hand()
        hand.close()
        assert hand.is_holding(), height
    for height in (0.031, 0.035, 0.08, -0.05):
        fakes = Fakes(palm=near_rim(PLATE, height))
        hand = fakes.hand()
        hand.close()
        assert not hand.is_holding(), height
        assert fakes.calls == []


def test_attach_at_exactly_the_distance_attaches_and_just_beyond_does_not():
    palm = near_rim(PLATE, 0.021, angle=0.7)
    exact = rim_distance(palm.position, PLATE, DIAMETER)
    at = Fakes(palm=palm).hand(attach_distance_m=exact)
    at.close()
    assert at.is_holding()
    beyond = Fakes(palm=palm).hand(attach_distance_m=np.nextafter(exact, 0.0))
    beyond.close()
    assert not beyond.is_holding()


def test_attach_distance_is_configurable():
    palm = near_rim(PLATE, 0.05)
    default = Fakes(palm=palm).hand()
    default.close()
    assert not default.is_holding()  # 5 cm is beyond the default 3 cm
    wide = Fakes(palm=palm).hand(attach_distance_m=0.06)
    wide.close()
    assert wide.is_holding()


def test_close_at_the_plate_centre_does_not_attach():
    # the centre of the disc is a radius away from the rim ring: 12.5 cm, far above 3 cm
    fakes = Fakes(palm=Pose(PLATE.position, TILTED_PALM.quat_wxyz))
    hand = fakes.hand()
    hand.close()
    assert not hand.is_holding()
    assert fakes.calls == []


def test_refused_close_commands_closed_attaches_nothing_and_writes_nothing():
    fakes = Fakes(palm=near_rim(PLATE, 0.10))
    hand = fakes.hand()
    hand.pre_grasp()
    hand.close()
    assert hand.posture is HandPosture.CLOSED
    assert hand.is_holding() is False  # the planner's grasp timeout tags grasp_miss
    assert hand.offset is None
    for _ in range(25):
        hand.update(DT)
    assert fakes.calls == []


def test_a_missed_close_is_final_until_the_posture_leaves_closed():
    fakes = Fakes(palm=near_rim(PLATE, 0.10))
    hand = fakes.hand()
    hand.close()
    fakes.palm = near_rim(PLATE, 0.0)  # the palm now sits on the rim
    for _ in range(10):
        hand.close()  # the planner repeats its verb every step: not a new attempt
        hand.update(DT)
    assert not hand.is_holding()
    assert fakes.calls == []
    hand.pre_grasp()  # leaving closed re-arms the attempt
    hand.close()
    assert hand.is_holding()
    assert fakes.calls == [("kinematic", True)]


def test_a_repeated_close_while_attached_keeps_the_captured_offset():
    fakes = Fakes()
    hand = fakes.hand()
    hand.close()
    offset = hand.offset
    fakes.palm = Pose(fakes.palm.position + np.array([0.0, 0.0, 0.10]), fakes.palm.quat_wxyz)
    hand.close()
    hand.close()
    assert hand.offset is offset
    assert fakes.calls == [("kinematic", True)]


def test_the_hand_uses_the_current_plate_diameter():
    palm = near_rim(PLATE, 0.0)
    palm = Pose(local_point(PLATE, RADIUS + 0.05, 0.0, 0.0), palm.quat_wxyz)  # 5 cm outside
    fakes = Fakes(palm=palm)
    hand = fakes.hand()
    assert hand.plate_diameter_m == DIAMETER
    hand.close()
    assert not hand.is_holding()
    hand.open()
    hand.plate_diameter_m = 0.31  # a larger plate: the same palm is 2 cm outside its rim
    assert hand.plate_diameter_m == 0.31
    hand.close()
    assert hand.is_holding()
    for bad in (0.0, -1.0, math.nan):
        with pytest.raises(ValueError):
            hand.plate_diameter_m = bad
    assert hand.plate_diameter_m == 0.31


# --- the offset and the follow ---------------------------------------------------------------


def test_offset_is_the_plate_pose_in_the_palm_frame():
    palm = near_rim(PLATE, 0.012, angle=1.3)
    fakes = Fakes(palm=palm)
    hand = fakes.hand()
    hand.close()
    assert_same_pose(hand.offset, pose_compose(pose_inverse(palm), PLATE))
    assert_same_pose(pose_compose(palm, hand.offset), PLATE)


def test_the_plate_pose_is_reproduced_at_capture():
    fakes = Fakes(palm=near_rim(PLATE, 0.012, angle=1.3))
    hand = fakes.hand()
    hand.close()
    hand.update(DT)
    (written,) = [c[1] for c in fakes.verbs("pose")]
    assert_same_pose(written, PLATE)


def test_the_plate_follows_a_moved_and_rotated_palm_rigidly():
    fakes = Fakes(palm=near_rim(PLATE, 0.012, angle=1.3))
    hand = fakes.hand()
    hand.close()
    offset = hand.offset
    path = [
        Pose((0.40, -0.05, 1.05), quat_from_axis_angle((0.0, 0.0, 1.0), 0.9)),
        Pose((0.10, 0.35, 1.30), quat_from_axis_angle((0.0, 1.0, 0.0), -0.6)),
        Pose((-0.20, 0.00, 0.80), quat_from_axis_angle((1.0, 1.0, 1.0), 2.4)),
    ]
    for palm in path:
        fakes.palm = palm
        hand.update(DT)
        assert_same_pose(fakes.plate, pose_compose(palm, offset))
        # the plate stays rigid in the palm frame whatever the palm does
        assert_same_pose(pose_compose(pose_inverse(palm), fakes.plate), offset)
    assert len(fakes.verbs("pose")) == len(path)


def test_the_plate_stays_where_the_palm_leaves_it_after_release():
    fakes = Fakes()
    hand = fakes.hand()
    hand.close()
    fakes.palm = TILTED_PALM
    hand.update(DT)
    last = fakes.plate
    hand.open()
    fakes.palm = Pose((9.0, 9.0, 9.0))
    for _ in range(5):
        hand.update(DT)
    assert fakes.plate is last
    assert len(fakes.verbs("pose")) == 1


# --- release ---------------------------------------------------------------------------------


def test_open_while_attached_restores_dynamics_then_hands_over_the_palm_velocity():
    fakes = Fakes()
    hand = fakes.hand()
    hand.close()
    hand.update(DT)
    fakes.velocity = ((0.4, 0.0, -0.1), (0.0, 0.0, 1.5))  # read at the moment of release
    hand.open()
    assert hand.posture is HandPosture.OPEN
    assert not hand.is_holding()
    assert hand.offset is None
    handed_over = ("velocity", (0.4, 0.0, -0.1), (0.0, 0.0, 1.5))
    assert fakes.calls[-2:] == [("kinematic", False), handed_over]
    assert [c[0] for c in fakes.calls] == ["kinematic", "pose", "kinematic", "velocity"]


def test_release_velocity_equals_the_palm_velocity():
    fakes = Fakes()
    hand = fakes.hand()
    hand.close()
    hand.open()
    (_, linear, angular) = fakes.verbs("velocity")[0]
    assert tuple(linear) == VELOCITY[0]
    assert tuple(angular) == VELOCITY[1]


def test_pre_grasp_while_attached_releases_like_open():
    fakes = Fakes()
    hand = fakes.hand()
    hand.close()
    hand.pre_grasp()
    assert hand.posture is HandPosture.PRE_GRASP
    assert not hand.is_holding()
    assert fakes.calls == [("kinematic", True), ("kinematic", False), ("velocity", *VELOCITY)]


def test_open_while_not_attached_only_changes_the_posture():
    fresh = Fakes()
    hand = fresh.hand()
    hand.open()
    hand.open()
    assert hand.posture is HandPosture.OPEN
    assert fresh.calls == []
    missed = Fakes(palm=near_rim(PLATE, 0.10))
    hand = missed.hand()
    hand.close()
    hand.open()
    assert hand.posture is HandPosture.OPEN
    assert missed.calls == []  # nothing was attached, so nothing is restored or handed over


def test_a_second_open_after_release_does_not_hand_over_again():
    fakes = Fakes()
    hand = fakes.hand()
    hand.close()
    hand.open()
    hand.open()
    hand.pre_grasp()
    assert fakes.calls == [("kinematic", True), ("kinematic", False), ("velocity", *VELOCITY)]


def test_re_attach_after_a_release_captures_a_new_offset():
    fakes = Fakes()
    hand = fakes.hand()
    hand.close()
    first = hand.offset
    hand.update(DT)
    hand.open()
    # the simulator now owns the plate: it falls and lands at another pose, the palm moves back
    fakes.plate = Pose((0.52, 0.12, 0.90), quat_from_yaw(-0.4))
    fakes.palm = near_rim(fakes.plate, 0.02, angle=-2.0)
    hand.close()
    assert hand.is_holding()
    assert_same_pose(hand.offset, pose_compose(pose_inverse(fakes.palm), fakes.plate))
    assert np.linalg.norm(hand.offset.position - first.position) > 1e-3
    kinds = [c[0:2] for c in fakes.calls if c[0] == "kinematic"]
    assert kinds == [("kinematic", True), ("kinematic", False), ("kinematic", True)]
    fakes.palm = TILTED_PALM
    hand.update(DT)
    assert_same_pose(fakes.plate, pose_compose(TILTED_PALM, hand.offset))


def test_re_attach_is_refused_when_the_plate_has_fallen_away():
    fakes = Fakes()
    hand = fakes.hand()
    hand.close()
    hand.open()
    fakes.plate = Pose((1.0, 1.0, 0.0), quat_from_yaw(0.0))  # far from the palm
    hand.close()
    assert not hand.is_holding()
    assert hand.posture is HandPosture.CLOSED
    assert fakes.verbs("kinematic") == [("kinematic", True), ("kinematic", False)]
