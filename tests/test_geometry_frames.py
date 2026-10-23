"""Frame helpers of kitchenbot.geometry (#48): matrices, pose composition, the pelvis frame."""

import math

import numpy as np
import pytest

import kitchenbot.geometry as geo
import kitchenbot.planning.geometry as planning_geo
from kitchenbot.geometry import (
    Pose,
    matrix_to_quat,
    pose_compose,
    pose_inverse,
    quat_angle,
    quat_apply,
    quat_from_axis_angle,
    quat_from_yaw,
    quat_multiply,
    quat_to_matrix,
    to_frame,
)

X, Y, Z = np.eye(3)
RNG_SEED = 48


def random_quats(n: int, seed: int = RNG_SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)
    q = rng.normal(size=(n, 4))
    return q / np.linalg.norm(q, axis=1, keepdims=True)


def random_poses(n: int, seed: int = RNG_SEED) -> list[Pose]:
    rng = np.random.default_rng(seed + 1)
    return [Pose(rng.uniform(-2.0, 2.0, 3), q) for q in random_quats(n, seed)]


def assert_pose_close(a: Pose, b: Pose, atol: float = 1e-12) -> None:
    np.testing.assert_allclose(a.position, b.position, atol=atol)
    assert quat_angle(a.quat_wxyz, b.quat_wxyz) == pytest.approx(0.0, abs=1e-6)


def test_planning_geometry_re_exports_the_same_objects():
    assert planning_geo.__all__ == geo.__all__
    for name in geo.__all__:
        assert getattr(planning_geo, name) is getattr(geo, name), name


def test_quat_to_matrix_agrees_with_quat_apply_and_is_a_rotation():
    rng = np.random.default_rng(RNG_SEED)
    for q in random_quats(50):
        m = quat_to_matrix(q)
        np.testing.assert_allclose(m @ m.T, np.eye(3), atol=1e-12)
        assert np.linalg.det(m) == pytest.approx(1.0)
        v = rng.normal(size=3)
        np.testing.assert_allclose(m @ v, quat_apply(q, v), atol=1e-12)
    np.testing.assert_allclose(quat_to_matrix(quat_from_yaw(math.pi / 2)) @ X, Y, atol=1e-12)


@pytest.mark.parametrize(
    "q",
    [
        (1.0, 0.0, 0.0, 0.0),
        tuple(quat_from_axis_angle(X, math.pi)),  # half-turns: w = 0, one branch per axis
        tuple(quat_from_axis_angle(Y, math.pi)),
        tuple(quat_from_axis_angle(Z, math.pi)),
        tuple(quat_from_axis_angle((1.0, 1.0, 0.0), math.pi - 1e-9)),
        tuple(quat_from_axis_angle((0.2, -0.3, 0.9), 4.0)),  # w < 0: comes back as -q
    ],
)
def test_matrix_to_quat_round_trips_on_every_branch(q):
    back = matrix_to_quat(quat_to_matrix(q))
    assert back[0] >= 0.0
    assert np.linalg.norm(back) == pytest.approx(1.0)
    assert quat_angle(back, q) == pytest.approx(0.0, abs=1e-7)


def test_matrix_to_quat_round_trips_random_rotations_and_rejects_non_rotations():
    for q in random_quats(200):
        back = matrix_to_quat(quat_to_matrix(q))
        sign = 1.0 if q[0] >= 0.0 else -1.0
        np.testing.assert_allclose(back, sign * q, atol=1e-12)
    with pytest.raises(ValueError):
        matrix_to_quat(np.diag([1.0, 1.0, -1.0]))  # a reflection
    with pytest.raises(ValueError):
        matrix_to_quat(2.0 * np.eye(3))
    with pytest.raises(ValueError):
        matrix_to_quat(np.eye(4))


def test_pose_compose_matches_homogeneous_matrices():
    def homogeneous(p: Pose) -> np.ndarray:
        t = np.eye(4)
        t[:3, :3] = quat_to_matrix(p.quat_wxyz)
        t[:3, 3] = p.position
        return t

    poses = random_poses(20)
    for a, b in zip(poses[:-1], poses[1:], strict=True):
        ab = pose_compose(a, b)
        expected = homogeneous(a) @ homogeneous(b)
        np.testing.assert_allclose(homogeneous(ab), expected, atol=1e-12)


def test_pose_inverse_and_associativity():
    identity = Pose((0.0, 0.0, 0.0))
    a, b, c = random_poses(3)
    assert_pose_close(pose_compose(a, pose_inverse(a)), identity)
    assert_pose_close(pose_compose(pose_inverse(a), a), identity)
    assert_pose_close(pose_inverse(pose_inverse(a)), a)
    assert_pose_close(pose_compose(pose_compose(a, b), c), pose_compose(a, pose_compose(b, c)))


def test_to_frame_of_a_yawed_pelvis_by_hand():
    # pelvis at (1, 2, 0.8) facing world +y; a target 1 m along +y and 0.2 m up from it is 1 m
    # straight ahead (pelvis +x) and 0.2 m up in the pelvis frame
    pelvis = Pose((1.0, 2.0, 0.8), quat_from_yaw(math.pi / 2))
    target = Pose((1.0, 3.0, 1.0), quat_from_yaw(math.pi / 2))
    local = to_frame(target, pelvis)
    np.testing.assert_allclose(local.position, [1.0, 0.0, 0.2], atol=1e-12)
    assert quat_angle(local.quat_wxyz, (1.0, 0.0, 0.0, 0.0)) == pytest.approx(0.0, abs=1e-9)


def test_to_frame_of_a_tilted_pelvis_and_round_trip():
    # a pelvis pitched forward by 10 degrees (x tipped down) sees a level target above its x axis
    pelvis = Pose((0.0, 0.0, 0.75), quat_from_axis_angle(Y, math.radians(10.0)))
    target = Pose((0.5, -0.2, 0.75))
    local = to_frame(target, pelvis)
    c, s = math.cos(math.radians(10.0)), math.sin(math.radians(10.0))
    np.testing.assert_allclose(local.position, [0.5 * c, -0.2, 0.5 * s], atol=1e-12)
    assert quat_angle(local.quat_wxyz, quat_from_axis_angle(Y, math.radians(-10.0))) < 1e-9
    for frame, world in zip(random_poses(10, seed=1), random_poses(10, seed=2), strict=True):
        assert_pose_close(pose_compose(frame, to_frame(world, frame)), world)
        assert_pose_close(to_frame(pose_compose(frame, world), frame), world)


def test_to_frame_compensates_body_sway():
    # the pelvis sways; the world target is fixed; the pelvis-frame target moves the other way
    target = Pose((0.45, -0.15, 0.95), quat_from_yaw(0.3))
    still = to_frame(target, Pose((0.0, 0.0, 0.75)))
    swayed = to_frame(target, Pose((0.02, 0.0, 0.75)))
    np.testing.assert_allclose(swayed.position - still.position, [-0.02, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(swayed.quat_wxyz, still.quat_wxyz, atol=1e-12)
    rotated = to_frame(target, Pose((0.0, 0.0, 0.75), quat_from_yaw(0.1)))
    expected = quat_multiply(quat_from_yaw(-0.1), target.quat_wxyz)
    assert quat_angle(rotated.quat_wxyz, expected) < 1e-9
