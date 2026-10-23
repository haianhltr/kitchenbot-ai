"""Pose and quaternion helpers of the task planner (wxyz order, numpy only)."""

import math

import numpy as np
import pytest

from kitchenbot.planning.geometry import (
    Pose,
    plate_vertical_quat,
    position_distance,
    quat_angle,
    quat_apply,
    quat_conjugate,
    quat_from_axis_angle,
    quat_from_yaw,
    quat_multiply,
    quat_slerp,
    tilt_from_vertical,
    yaw_from_quat,
)

X, Y, Z = np.eye(3)


def test_yaw_quaternion_is_wxyz_and_round_trips():
    q = quat_from_yaw(math.pi / 2)
    np.testing.assert_allclose(q, [math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)])
    for yaw in (-3.0, -1.0, 0.0, 0.4, 2.5):
        assert yaw_from_quat(quat_from_yaw(yaw)) == pytest.approx(yaw)


def test_apply_multiply_and_conjugate():
    qz = quat_from_yaw(math.pi / 2)
    np.testing.assert_allclose(quat_apply(qz, X), Y, atol=1e-12)
    qy = quat_from_axis_angle(Y, math.pi / 2)
    np.testing.assert_allclose(quat_apply(qy, Z), X, atol=1e-12)
    # a * b applies b first
    np.testing.assert_allclose(quat_apply(quat_multiply(qz, qy), Z), Y, atol=1e-12)
    np.testing.assert_allclose(
        quat_multiply(qz, quat_conjugate(qz)), [1.0, 0.0, 0.0, 0.0], atol=1e-12
    )


def test_quat_angle_is_sign_invariant_and_takes_the_short_way():
    a = quat_from_yaw(0.3)
    assert quat_angle(a, a) == pytest.approx(0.0, abs=1e-9)
    assert quat_angle(a, -a) == pytest.approx(0.0, abs=1e-9)
    assert quat_angle(quat_from_yaw(0.0), quat_from_yaw(1.2)) == pytest.approx(1.2)
    assert quat_angle(quat_from_yaw(-3.0), quat_from_yaw(3.0)) == pytest.approx(2 * math.pi - 6.0)


def test_slerp_endpoints_and_midpoint():
    a, b = quat_from_yaw(0.0), quat_from_yaw(1.0)
    np.testing.assert_allclose(quat_slerp(a, b, 0.0), a, atol=1e-12)
    np.testing.assert_allclose(quat_slerp(a, b, 1.0), b, atol=1e-12)
    assert yaw_from_quat(quat_slerp(a, b, 0.5)) == pytest.approx(0.5)
    assert yaw_from_quat(quat_slerp(a, -b, 0.25)) == pytest.approx(0.25)  # short arc


@pytest.mark.parametrize("slot_yaw_deg", [-120.0, -30.0, 0.0, 45.0, 90.0, 180.0])
def test_plate_vertical_quat_puts_the_normal_horizontal_along_the_slot_x(slot_yaw_deg):
    yaw = math.radians(slot_yaw_deg)
    q = plate_vertical_quat(yaw)
    assert np.linalg.norm(q) == pytest.approx(1.0)
    normal = quat_apply(q, Z)
    np.testing.assert_allclose(normal, [math.cos(yaw), math.sin(yaw), 0.0], atol=1e-12)
    assert tilt_from_vertical(q) == pytest.approx(0.0, abs=1e-9)


def test_tilt_from_vertical():
    assert tilt_from_vertical(quat_from_yaw(0.7)) == pytest.approx(math.pi / 2)  # flat plate
    tipped = quat_multiply(plate_vertical_quat(0.0), quat_from_axis_angle(Y, math.radians(-20)))
    assert math.degrees(tilt_from_vertical(tipped)) == pytest.approx(20.0)


def test_pose_is_frozen_read_only_and_normalised():
    source = np.array([1.0, 2.0, 3.0])
    pose = Pose(source, (2.0, 0.0, 0.0, 0.0))
    source[0] = 99.0
    assert pose.position[0] == 1.0
    np.testing.assert_allclose(pose.quat_wxyz, [1.0, 0.0, 0.0, 0.0])
    with pytest.raises(ValueError):
        pose.position[0] = 5.0
    with pytest.raises(AttributeError):
        pose.position = np.zeros(3)  # type: ignore[misc]
    assert pose == Pose((1.0, 2.0, 3.0))
    assert pose != Pose((1.0, 2.0, 3.5))
    assert pose.raised(0.1).position[2] == pytest.approx(3.1)
    assert position_distance(pose.position, (1.0, 2.0, 4.0)) == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("position", "quat"),
    [((0.0, 0.0), (1, 0, 0, 0)), ((0.0, 0.0, math.nan), (1, 0, 0, 0)), ((0, 0, 0), (0, 0, 0, 0))],
)
def test_pose_rejects_bad_input(position, quat):
    with pytest.raises(ValueError):
        Pose(position, quat)
