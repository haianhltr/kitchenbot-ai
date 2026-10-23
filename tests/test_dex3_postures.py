"""Dex3 joint limits against the URDF fixture, and the posture table built on them (#42)."""

import json
from pathlib import Path

import pytest

from kitchenbot.controllers.hand import dex3_postures as dp
from kitchenbot.controllers.hand.base import HandPosture
from kitchenbot.robots import joint_maps as jm
from kitchenbot.robots.joint_limits import DEX3_LIMITS, JointLimit

REPO = Path(__file__).resolve().parents[1]
FIXTURE = REPO / "tests" / "fixtures" / "g1_dex3_joints_urdf.json"

POSTURES = (HandPosture.OPEN, HandPosture.PRE_GRASP, HandPosture.CLOSED)
HANDS = {"left": jm.HAND_LEFT, "right": jm.HAND_RIGHT}


@pytest.fixture(scope="module")
def urdf_joints():
    dump = json.loads(FIXTURE.read_text(encoding="utf-8"))
    return {j["name"]: j for j in dump["joints"]}


def fraction_of(side, role, angle):
    start, end = dp.open_end(side, role), dp.closed_end(side, role)
    return (angle - start) / (end - start)


# --- limits -----------------------------------------------------------------------------------


def test_limits_equal_the_fixture_on_all_fields(urdf_joints):
    assert list(DEX3_LIMITS) == [*jm.HAND_LEFT, *jm.HAND_RIGHT]
    for name, limit in DEX3_LIMITS.items():
        joint = urdf_joints[name]
        fields = (limit.lower, limit.upper, limit.effort, limit.velocity)
        assert fields == (joint["lower"], joint["upper"], joint["effort"], joint["velocity"]), name


def test_joint_limit_contains_and_clamp():
    limit = JointLimit(-1.0, 0.5, 1.4, 12.0)
    assert limit.contains(-1.0) and limit.contains(0.5) and not limit.contains(0.51)
    assert (limit.clamp(-2.0), limit.clamp(0.2), limit.clamp(3.0)) == (-1.0, 0.2, 0.5)
    with pytest.raises(AttributeError):
        limit.lower = 0.0  # frozen


# --- roles and directions ---------------------------------------------------------------------


def test_roles_follow_the_joint_map_order():
    for side, names in HANDS.items():
        assert tuple(dp.joint_name(side, role) for role in dp.ROLES) == names


def test_flexion_roles_have_one_zero_limit_and_thumb_roles_none():
    for side in dp.SIDES:
        for role in dp.ROLES:
            limit = DEX3_LIMITS[dp.joint_name(side, role)]
            has_zero = 0.0 in (limit.lower, limit.upper)
            assert has_zero == (role in dp.FLEXION_ROLES), (side, role)
    assert set(dp.FLEXION_ROLES) | set(dp.LEFT_THUMB_CLOSING_SIGN) == set(dp.ROLES)


def test_flexion_directions_are_read_from_the_limits():
    expected_left = {"thumb_2": 1, "middle_0": -1, "middle_1": -1, "index_0": -1, "index_1": -1}
    for role, sign in expected_left.items():
        assert dp.closing_direction("left", role) == sign, role
        assert dp.closing_direction("right", role) == -sign, role
        assert dp.open_end("left", role) == 0.0 == dp.open_end("right", role)


def test_thumb_directions_come_from_the_one_constant():
    assert set(dp.LEFT_THUMB_CLOSING_SIGN) == {"thumb_0", "thumb_1"}
    assert set(dp.LEFT_THUMB_CLOSING_SIGN.values()) <= {1, -1}
    for role, sign in dp.LEFT_THUMB_CLOSING_SIGN.items():
        assert dp.closing_direction("left", role) == sign
    # thumb_0 turns about y: same sign on both hands; thumb_1 about z: mirrored
    assert dp.closing_direction("right", "thumb_0") == dp.LEFT_THUMB_CLOSING_SIGN["thumb_0"]
    assert dp.closing_direction("right", "thumb_1") == -dp.LEFT_THUMB_CLOSING_SIGN["thumb_1"]


def test_left_thumb_1_closes_with_left_thumb_2():
    assert dp.closing_direction("left", "thumb_1") == dp.closing_direction("left", "thumb_2")
    assert dp.closing_direction("right", "thumb_1") == dp.closing_direction("right", "thumb_2")


@pytest.mark.parametrize("role", ["thumb_0", "thumb_1"])
def test_flipping_one_sign_flips_that_thumb_joint_on_both_hands(monkeypatch, role):
    before = {side: dp.posture_targets(side, HandPosture.OPEN) for side in dp.SIDES}
    flipped = -dp.LEFT_THUMB_CLOSING_SIGN[role]
    monkeypatch.setitem(dp.LEFT_THUMB_CLOSING_SIGN, role, flipped)
    i = dp.ROLES.index(role)
    for side in dp.SIDES:
        after = dp.posture_targets(side, HandPosture.OPEN)
        limit = DEX3_LIMITS[dp.joint_name(side, role)]
        assert {before[side][i], after[i]} == {limit.lower, limit.upper}
        assert [a for j, a in enumerate(after) if j != i] == [
            b for j, b in enumerate(before[side]) if j != i
        ]


def test_a_bad_constant_value_is_rejected(monkeypatch):
    monkeypatch.setitem(dp.LEFT_THUMB_CLOSING_SIGN, "thumb_1", 0)
    with pytest.raises(ValueError):
        dp.closing_direction("left", "thumb_1")


# --- posture table and targets ----------------------------------------------------------------


def test_table_has_every_posture_and_fractions_in_unit_interval():
    assert set(dp.POSTURE_FRACTIONS) == set(HandPosture)
    for posture, fractions in dp.POSTURE_FRACTIONS.items():
        assert len(fractions) == len(dp.ROLES), posture
        assert all(0.0 <= f <= 1.0 for f in fractions), posture
    assert all(f == 0.0 for f in dp.POSTURE_FRACTIONS[HandPosture.OPEN])
    assert max(dp.POSTURE_FRACTIONS[HandPosture.CLOSED]) < 1.0  # margin for the controller


@pytest.mark.parametrize("side", dp.SIDES)
@pytest.mark.parametrize("posture", POSTURES)
def test_targets_lie_inside_the_limits(side, posture):
    targets = dp.posture_targets(side, posture)
    assert len(targets) == 7
    for name, angle in zip(HANDS[side], targets, strict=True):
        assert DEX3_LIMITS[name].contains(angle), (name, angle)


@pytest.mark.parametrize("side", dp.SIDES)
def test_closed_targets_stay_off_the_limits(side):
    for name, angle in zip(HANDS[side], dp.posture_targets(side, "closed"), strict=True):
        limit = DEX3_LIMITS[name]
        assert limit.lower < angle < limit.upper, name


@pytest.mark.parametrize("posture", POSTURES)
def test_left_and_right_share_fractions_with_mirrored_signs(posture):
    left = dp.posture_targets("left", posture)
    right = dp.posture_targets("right", posture)
    for i, role in enumerate(dp.ROLES):
        assert fraction_of("left", role, left[i]) == pytest.approx(
            fraction_of("right", role, right[i]), abs=1e-12
        ), role
        if role == "thumb_0":
            assert right[i] == left[i]  # turns about y: the same angle mirrors the motion
        else:
            assert right[i] == -left[i], role  # turns about z: mirrored sign


def test_flexion_strictly_increases_open_to_pre_grasp_to_closed():
    for side in dp.SIDES:
        for role in (*dp.FLEXION_ROLES, "thumb_1"):
            i = dp.ROLES.index(role)
            sign = dp.closing_direction(side, role)
            flex = [sign * dp.posture_targets(side, p)[i] for p in POSTURES]
            assert flex[0] < flex[1] < flex[2], (side, role, flex)


def test_thumb_0_sits_in_the_pinch_plane_once_the_hand_leaves_open():
    i = dp.ROLES.index("thumb_0")
    for side in dp.SIDES:
        assert dp.posture_targets(side, "open")[i] == dp.open_end(side, "thumb_0")
        for posture in (HandPosture.PRE_GRASP, HandPosture.CLOSED):
            assert dp.posture_targets(side, posture)[i] == pytest.approx(0.0, abs=1e-12)


def test_known_values():
    # left middle_0 [-1.57079632, 0]: pre-grasp 0.35 and closed 0.8 of the way to the lower limit
    i = dp.ROLES.index("middle_0")
    assert dp.posture_targets("left", "pre_grasp")[i] == pytest.approx(-0.35 * 1.57079632)
    assert dp.posture_targets("right", "closed")[i] == pytest.approx(0.8 * 1.57079632)


def test_bad_inputs_are_rejected():
    with pytest.raises(ValueError):
        dp.posture_targets("middle", "open")
    with pytest.raises(ValueError):
        dp.posture_targets("left", "fist")
    with pytest.raises(ValueError):
        dp.closing_direction("left", "ring_0")
    with pytest.raises(ValueError):
        dp.fraction_to_angle("left", "index_0", 1.2)
