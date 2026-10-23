"""Arm target slew filter and the world-to-pelvis pipeline (#48). No simulator."""

import json
import math
from pathlib import Path

import numpy as np
import pytest

from kitchenbot.controllers.reachability import ReachabilityEnvelope
from kitchenbot.controllers.target_filter import (
    MAX_ANG_SPEED_RPS,
    MAX_LIN_SPEED_MPS,
    ArmTargetPipeline,
    TargetSlewFilter,
)
from kitchenbot.geometry import (
    Pose,
    pose_compose,
    quat_angle,
    quat_from_axis_angle,
    quat_from_yaw,
    to_frame,
)
from kitchenbot.robots.kinematics import right_arm_chain

REPO = Path(__file__).resolve().parents[1]
FIXTURE = REPO / "tests" / "fixtures" / "g1_dex3_joints_urdf.json"

DT = 0.02  # 50 Hz, the arm controller rate (docs/architecture.md section 5)
TOL = 1e-9
PELVIS = Pose((0.0, 0.0, 0.75))


@pytest.fixture(scope="module")
def envelope():
    return ReachabilityEnvelope.load()


@pytest.fixture(scope="module")
def palm_zero():
    """The right palm at the zero configuration, pelvis frame."""
    chain = right_arm_chain(json.loads(FIXTURE.read_text(encoding="utf-8"))["joints"])
    return Pose(*chain.fk(np.zeros(7)))


def steps_to_arrive(filt: TargetSlewFilter, target: Pose, dt: float, limit: int = 10_000) -> int:
    for n in range(1, limit + 1):
        pose = filt.step(target, dt)
        arrived_at = float(np.linalg.norm(pose.position - target.position))
        if arrived_at < 1e-12 and quat_angle(pose.quat_wxyz, target.quat_wxyz) < 1e-12:
            return n
    raise AssertionError("never arrived")


# --- TargetSlewFilter -------------------------------------------------------------------------


def test_section_7_limits_are_the_defaults():
    assert (MAX_LIN_SPEED_MPS, MAX_ANG_SPEED_RPS) == (0.3, 1.0)
    filt = TargetSlewFilter()
    assert (filt.max_lin_speed, filt.max_ang_speed) == (0.3, 1.0)


def test_no_step_exceeds_either_cap():
    rng = np.random.default_rng(48)
    filt = TargetSlewFilter()
    filt.reset(Pose((0.2, -0.2, 0.1)))
    target = Pose((0.0, 0.0, 0.0))
    for _ in range(2000):
        if rng.random() < 0.05:  # a new random target now and then, some far away
            target = Pose(rng.uniform(-1.0, 1.0, 3), rng.normal(size=4))
        dt = float(rng.choice([0.005, 0.01, 0.02, 0.05]))
        before = filt.pose
        after = filt.step(target, dt)
        moved = float(np.linalg.norm(after.position - before.position))
        turned = quat_angle(before.quat_wxyz, after.quat_wxyz)
        assert moved <= MAX_LIN_SPEED_MPS * dt + TOL
        assert turned <= MAX_ANG_SPEED_RPS * dt + 1e-7


def test_reaches_a_fixed_target_in_distance_over_speed():
    # 0.3 m at 0.3 m/s: 1.0 s, 50 steps at 50 Hz
    filt = TargetSlewFilter()
    filt.reset(Pose((0.0, 0.0, 0.0)))
    assert steps_to_arrive(filt, Pose((0.0, 0.3, 0.0)), DT) == 50
    # 1.2 rad at 1 rad/s: 1.2 s, 60 steps
    filt.reset(Pose((0.0, 0.0, 0.0)))
    assert steps_to_arrive(filt, Pose((0.0, 0.0, 0.0), quat_from_yaw(1.2)), DT) == 60
    # both at once: the slower of the two, 1.2 s; position arrives first, at 1.0 s
    filt.reset(Pose((0.1, 0.1, 0.1)))
    target = Pose((0.1 + 0.3 / math.sqrt(3),) * 3, quat_from_axis_angle((1, 1, 0), 1.2))
    for _ in range(50):
        pose = filt.step(target, DT)
    np.testing.assert_allclose(pose.position, target.position, atol=1e-12)
    assert quat_angle(pose.quat_wxyz, target.quat_wxyz) == pytest.approx(0.2, abs=1e-9)
    assert steps_to_arrive(filt, target, DT) == 10


def test_moves_along_the_straight_line_and_the_short_arc():
    filt = TargetSlewFilter()
    start = Pose((0.0, 0.0, 0.0), quat_from_yaw(3.0))
    target = Pose((0.3, 0.4, 0.0), quat_from_yaw(-3.0))  # 0.283 rad the short way
    filt.reset(start)
    pose = filt.step(target, 0.1)
    np.testing.assert_allclose(pose.position, [0.018, 0.024, 0.0], atol=1e-12)
    assert quat_angle(pose.quat_wxyz, quat_from_yaw(3.1)) == pytest.approx(0.0, abs=1e-9)


def test_filter_input_checks():
    filt = TargetSlewFilter()
    with pytest.raises(RuntimeError):
        filt.step(PELVIS, DT)
    filt.reset(PELVIS)
    assert filt.step(Pose((1.0, 0.0, 0.0)), 0.0) == PELVIS
    with pytest.raises(ValueError):
        filt.step(PELVIS, -0.01)
    with pytest.raises(ValueError):
        TargetSlewFilter(max_lin_speed=0.0)


# --- ArmTargetPipeline ------------------------------------------------------------------------


def test_pipeline_expresses_the_target_in_the_pelvis_frame(envelope, palm_zero):
    pelvis = Pose((1.0, 2.0, 0.75), quat_from_yaw(math.pi / 2))
    pipe = ArmTargetPipeline(envelope)
    pipe.reset(palm_zero, pelvis)
    local_goal = Pose((0.30, -0.20, 0.25), quat_from_yaw(0.2))
    world_goal = pose_compose(pelvis, local_goal)
    for _ in range(200):
        out = pipe.step(world_goal, pelvis, DT)
        assert not out.infeasible
    np.testing.assert_allclose(out.pose.position, local_goal.position, atol=1e-12)
    assert quat_angle(out.pose.quat_wxyz, local_goal.quat_wxyz) < 1e-9


def test_unreachable_target_sets_infeasible_and_holds_the_last_feasible_target(envelope, palm_zero):
    pipe = ArmTargetPipeline(envelope)
    pipe.reset(palm_zero, PELVIS)
    feasible = pose_compose(PELVIS, Pose((0.30, -0.25, 0.20)))
    unreachable = pose_compose(PELVIS, Pose((2.0, -0.25, 0.20)))  # 2 m ahead of the pelvis
    assert pipe.reachable(to_frame(feasible, PELVIS).position)
    assert not pipe.reachable(to_frame(unreachable, PELVIS).position)

    for _ in range(5):
        out = pipe.step(feasible, PELVIS, DT)
        assert not out.infeasible
    for _ in range(300):
        out = pipe.step(unreachable, PELVIS, DT)
        assert out.infeasible
        assert out.pose.position[0] <= 0.30 + 1e-12  # never heads for the unreachable target
    assert pipe.held_target == feasible
    np.testing.assert_allclose(out.pose.position, (0.30, -0.25, 0.20), atol=1e-12)

    # the held target is re-expressed in the current pelvis frame: sway is compensated
    swayed = Pose((0.03, 0.0, 0.75))
    for _ in range(50):
        out = pipe.step(unreachable, swayed, DT)
        assert out.infeasible
    np.testing.assert_allclose(out.pose.position, (0.27, -0.25, 0.20), atol=1e-12)

    # a feasible target clears the flag and becomes the new held target
    out = pipe.step(feasible, PELVIS, DT)
    assert not out.infeasible


def test_before_any_feasible_target_the_pipeline_holds_the_reset_pose(envelope, palm_zero):
    pipe = ArmTargetPipeline(envelope)
    pipe.reset(palm_zero, PELVIS)
    out = pipe.step(pose_compose(PELVIS, Pose((0.0, 0.0, 2.0))), PELVIS, DT)
    assert out.infeasible
    np.testing.assert_allclose(out.pose.position, palm_zero.position, atol=1e-12)
    assert quat_angle(out.pose.quat_wxyz, palm_zero.quat_wxyz) < 1e-9


def test_margin_rejects_targets_near_the_boundary(envelope, palm_zero):
    occupied = np.argwhere(envelope.occupied)
    top = occupied[occupied[:, 2].argmax()]  # nothing occupied above this voxel
    edge = Pose(envelope.origin + (top + 0.5) * envelope.spacing)
    plain = ArmTargetPipeline(envelope)
    with_margin = ArmTargetPipeline(envelope, margin=0.05)
    assert plain.reachable(edge.position) and not with_margin.reachable(edge.position)
    with pytest.raises(ValueError):
        ArmTargetPipeline(envelope, margin=-0.01)


def test_pipeline_caps_hold_while_the_pelvis_moves(envelope, palm_zero):
    rng = np.random.default_rng(7)
    pipe = ArmTargetPipeline(envelope)
    pipe.reset(palm_zero, PELVIS)
    goal = pose_compose(PELVIS, Pose((0.25, -0.30, 0.35), quat_from_yaw(0.8)))
    previous = palm_zero
    for _ in range(300):
        pelvis = Pose(
            (0.0, 0.0, 0.75) + rng.uniform(-0.02, 0.02, 3),
            quat_from_axis_angle(rng.normal(size=3), rng.uniform(0.0, 0.05)),
        )
        out = pipe.step(goal, pelvis, DT)
        moved = float(np.linalg.norm(out.pose.position - previous.position))
        assert moved <= MAX_LIN_SPEED_MPS * DT + TOL
        assert quat_angle(previous.quat_wxyz, out.pose.quat_wxyz) <= MAX_ANG_SPEED_RPS * DT + 1e-7
        previous = out.pose


def test_pipeline_is_deterministic(envelope, palm_zero):
    def run():
        rng = np.random.default_rng(3)
        pipe = ArmTargetPipeline(envelope)
        pipe.reset(palm_zero, PELVIS)
        outputs = []
        for i in range(400):
            if i % 40 == 0:
                goal = pose_compose(PELVIS, Pose(rng.uniform(-0.6, 0.6, 3), rng.normal(size=4)))
            out = pipe.step(goal, PELVIS, DT)
            outputs.append((out.pose.position.copy(), out.pose.quat_wxyz.copy(), out.infeasible))
        return outputs

    first, second = run(), run()
    assert any(flag for _, _, flag in first) and not all(flag for _, _, flag in first)
    for (p1, q1, f1), (p2, q2, f2) in zip(first, second, strict=True):
        assert np.array_equal(p1, p2) and np.array_equal(q1, q2) and f1 == f2


def test_pipeline_loads_the_committed_envelope_by_default(envelope, palm_zero):
    pipe = ArmTargetPipeline()
    assert pipe.envelope.same_as(envelope)
    with pytest.raises(RuntimeError):
        pipe.step(PELVIS, PELVIS, DT)
