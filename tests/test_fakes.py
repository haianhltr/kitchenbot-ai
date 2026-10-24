"""The fakes in ``kitchenbot.planning.fakes`` (moved from tests/planner_helpers.py in #60).

Covers what #60 added: the re-export, the gate in the fake arm, and building the plate, rack and
counter edge from an ``EpisodeSpec`` and a ``RackSlotLayout`` with the frames the module
docstring defines. The planner's own behaviour against the fakes stays in test_plate_to_rack.py.
"""

import math
from pathlib import Path

import numpy as np
import pytest

from kitchenbot.controllers.reachability import ReachabilityEnvelope
from kitchenbot.controllers.target_filter import ArmTargetPipeline
from kitchenbot.evaluation.failure_tags import FailureTag
from kitchenbot.geometry import (
    Pose,
    pose_compose,
    position_distance,
    quat_angle,
    quat_apply,
    quat_from_yaw,
    to_frame,
    yaw_from_quat,
)
from kitchenbot.planning import fakes
from kitchenbot.planning import geometry as planning_geometry
from kitchenbot.planning.fakes import (
    DRY_RUN_RACK_FRAME,
    PALM_START,
    PALM_START_IN_PELVIS,
    PELVIS_HEIGHT_M,
    FakeArm,
    FakeHand,
    FakePlate,
    FakeRack,
    Fakes,
    body_pose_from_grasp,
    grasp_pose_from_body,
    pelvis_pose,
    run_episode,
)
from kitchenbot.planning.plate_to_rack import PlannerState
from kitchenbot.scene import dishwasher_frames
from kitchenbot.scene import kitchen_dims as kd
from kitchenbot.scene.randomization import sample_episode

LAYOUT = kd.RackSlotLayout()
SEEN = [sample_episode(seed, "seen", "S4") for seed in range(20)]
UNSEEN = [sample_episode(seed, "unseen", "S4") for seed in range(1000, 1020)]
SPECS = SEEN + UNSEEN


def _everything_envelope() -> ReachabilityEnvelope:
    """A 2 m cube around the pelvis that is all reachable: the gate on, but never refusing."""
    return ReachabilityEnvelope(
        occupied=np.ones((100, 100, 100), dtype=bool),
        origin=np.array([-1.0, -1.0, -1.0]),
        spacing=0.02,
        seed=0,
        samples=0,
    )


# ----------------------------------------------------------------------------- the re-export


def test_planner_helpers_is_a_thin_reexport_of_the_moved_fakes():
    import planner_helpers

    assert set(planner_helpers.__all__) >= {
        "FakeArm",
        "FakeHand",
        "FakePlate",
        "FakeRack",
        "Fakes",
        "EpisodeLog",
        "run_episode",
        "PALM_START",
        "PLATE_GRASP",
        "COUNTER_EDGE",
        "SLOT_ENTRY",
        "SHOULDER",
        "ARM_LINEAR_SPEED_MPS",
        "ARM_ANGULAR_SPEED_RPS",
        "ATTACH_RADIUS_M",
        "CLOSE_DURATION_S",
    }
    for name in planner_helpers.__all__:
        assert getattr(planner_helpers, name) is getattr(fakes, name), name
    source = Path(planner_helpers.__file__).read_text(encoding="utf-8")
    assert "class " not in source and "def " not in source  # nothing is defined there any more


def test_the_moved_defaults_are_unchanged():
    arm = FakeArm(PALM_START)
    assert (arm.linear_speed, arm.angular_speed) == (0.3, 1.0)
    assert arm.shoulder == (0.0, -0.20, 1.15) and arm.reach_radius == math.inf
    assert arm.pipeline is None and arm.pelvis is None and not arm.infeasible
    hand = FakeHand()
    assert hand.close_duration == 0.5 and hand.slip_after_s is None and hand.posture == "open"
    rack = FakeRack()
    assert (rack.half_width_m, rack.floor_depth_m, rack.floor_slack_m) == (0.02, 0.08, 0.01)
    assert (rack.floor_stiffness_n_per_m, rack.rack_half_extent_m, rack.tine_force_n) == (
        1000.0,
        0.15,
        50.0,
    )
    assert rack.frame is None and rack.slot_index is None and rack.layout is None
    assert FakePlate().diameter_m is None
    fk = Fakes()
    assert fk.hand_reach_from_geometry and fk.plate_follows_palm
    assert run_episode().final.state is PlannerState.DONE  # the default fakes still work


def test_the_legacy_reach_radius_path_counts_its_refusals():
    arm = FakeArm(PALM_START, reach_radius=0.5)
    start = arm.palm.position.copy()
    arm.step(Pose((2.0, 2.0, 2.0)), 0.02)
    assert arm.infeasible and arm.refusals == 1
    assert np.array_equal(arm.palm.position, start)  # a refused target does not move the palm
    arm.step(Pose(start + np.array([0.0, 0.0, 0.05])), 0.02)
    assert not arm.infeasible and arm.refusals == 1


# ----------------------------------------------------------------------------- frames


def test_the_fakes_use_the_one_grasp_convention_of_planning_geometry():
    # #67: the rule is written once, in planning.geometry; the fakes re-export it, not restate it
    assert body_pose_from_grasp is planning_geometry.body_pose_from_grasp
    assert grasp_pose_from_body is planning_geometry.grasp_pose_from_body
    assert {"body_pose_from_grasp", "grasp_pose_from_body"} <= set(fakes.__all__)


@pytest.mark.parametrize("diameter", [0.22, 0.25, 0.27])
def test_grasp_frame_and_body_pose_convert_both_ways(diameter):
    body = Pose((0.2, 0.3, 0.9), quat_from_yaw(math.radians(25.0)))
    grasp = grasp_pose_from_body(body, diameter)
    back = body_pose_from_grasp(grasp, diameter)
    # the same plate: same origin and normal; the yaw about the normal is the convention's
    assert np.allclose(back.position, body.position)
    assert np.allclose(quat_apply(back.quat_wxyz, (0, 0, 1)), quat_apply(body.quat_wxyz, (0, 0, 1)))
    again = grasp_pose_from_body(back, diameter)
    assert np.allclose(again.position, grasp.position)
    assert quat_angle(again.quat_wxyz, grasp.quat_wxyz) == pytest.approx(0.0, abs=1e-9)
    # the top rim point on the robot's side (kitchen -y) of the plate centre
    expected = body.position + np.array([0.0, -diameter / 2.0, kd.PLATE_HEIGHT_M])
    assert np.allclose(grasp.position, expected)
    centre_in_grasp = to_frame(body, grasp)
    assert np.allclose(centre_in_grasp.position, (diameter / 2.0, 0.0, -kd.PLATE_HEIGHT_M))


def test_the_nominal_pelvis_stands_at_the_standoff_facing_plus_y():
    spec = SEEN[0]
    pelvis = pelvis_pose(spec)
    dx, dy = spec.pelvis_offset_m
    assert np.allclose(pelvis.position, (dx, -kd.ROBOT_STANDOFF_M + dy, PELVIS_HEIGHT_M))
    forward = quat_apply(pelvis.quat_wxyz, (1.0, 0.0, 0.0))
    left = quat_apply(pelvis.quat_wxyz, (0.0, 1.0, 0.0))
    yaw = math.pi / 2.0 + math.radians(spec.pelvis_yaw_offset_deg)
    assert np.allclose(forward, (math.cos(yaw), math.sin(yaw), 0.0))
    assert np.allclose(left, (-math.sin(yaw), math.cos(yaw), 0.0))
    assert yaw_from_quat(pelvis.quat_wxyz) == pytest.approx(yaw)
    assert pelvis_pose(spec, stand_off_m=0.2).position[1] == pytest.approx(-0.2 + dy)


@pytest.mark.parametrize("spec", SPECS, ids=lambda s: f"{s.split}-{s.seed}")
def test_the_plate_is_built_from_the_spec_in_the_kitchen_frame(spec):
    plate = FakePlate.from_spec(spec)
    body = plate.body_pose()
    assert np.allclose(body.position, spec.plate_position_m)
    assert body.position[2] == pytest.approx(kd.COUNTER_HEIGHT_M)
    assert np.allclose(quat_apply(body.quat_wxyz, (0.0, 0.0, 1.0)), (0.0, 0.0, 1.0))  # flat
    # the body comes back in the convention's turned frame, +x toward the robot (kitchen -y)
    assert math.degrees(yaw_from_quat(body.quat_wxyz)) == pytest.approx(-90.0)
    assert plate.diameter_m == spec.plate_diameter_m
    # the grasp frame the planner observes is the top rim point nearest the robot, whatever the
    # sampled yaw (the plate is a solid of revolution)
    sampled = Pose(spec.plate_position_m, quat_from_yaw(math.radians(spec.plate_yaw_deg)))
    assert plate.pose == grasp_pose_from_body(sampled, spec.plate_diameter_m)
    x, y, z = spec.plate_position_m
    rim = (x, y - spec.plate_diameter_m / 2.0, z + kd.PLATE_HEIGHT_M)
    assert np.allclose(plate.pose.position, rim)
    assert position_distance(plate.pose.position, body.position) == pytest.approx(
        math.hypot(spec.plate_diameter_m / 2.0, kd.PLATE_HEIGHT_M)
    )


def test_a_plate_without_a_diameter_has_no_body_pose():
    with pytest.raises(ValueError, match="diameter_m"):
        FakePlate().body_pose()


# ----------------------------------------------------------------------------- the rack


@pytest.mark.parametrize("diameter", kd.PLATE_DIAMETERS_M)
@pytest.mark.parametrize("slot", range(1, kd.RACK_SLOTS + 1))
def test_the_rack_is_built_from_the_layout_and_the_slot(slot, diameter):
    frame = Pose((0.45, -0.20, 0.51))
    rack = FakeRack.from_layout(LAYOUT, slot, frame, diameter_m=diameter)
    sx, sy, sz, _ = LAYOUT.slot_pose(slot)
    thickness = kd.PLATE_HEIGHT_M
    # the grasp frame holds the top of the vertical plate: a diameter above the tine tops, half a
    # thickness along the slot x (#67, planning.geometry.slot_entry_grasp_pose)
    assert np.allclose(
        rack.slot_entry.position,
        (0.45 + sx + thickness / 2.0, -0.20 + sy, 0.51 + sz + diameter),
    )
    assert yaw_from_quat(rack.slot_entry.quat_wxyz) == pytest.approx(0.0)
    assert rack.frame is frame and rack.slot_index == slot and rack.layout is LAYOUT
    assert rack.half_width_m == pytest.approx((LAYOUT.gap_width - thickness) / 2.0)
    assert rack.floor_depth_m == pytest.approx(LAYOUT.tine_height)
    entry = rack.slot_entry.position
    seated = entry + np.array([0.0, 0.0, -0.05])  # the planner's insertion depth
    assert rack.in_slot(seated) and rack.contact_force(seated) == 0.0
    off = seated + np.array([rack.half_width_m + 0.005, 0.0, 0.0])
    assert not rack.in_slot(off) and rack.contact_force(off) == rack.tine_force_n
    assert not rack.in_slot(entry + np.array([0.0, 0.0, 0.05]))  # above the entry


def test_a_rotated_rack_frame_rotates_the_slot_entry():
    frame = Pose((1.0, 0.5, 0.4), quat_from_yaw(math.pi / 2.0))
    rack = FakeRack.from_layout(LAYOUT, 4, frame, diameter_m=0.25)
    sx, sy, sz, _ = LAYOUT.slot_pose(4)
    local = Pose((sx + kd.PLATE_HEIGHT_M / 2.0, sy, sz + 0.25))
    assert np.allclose(rack.slot_entry.position, pose_compose(frame, local).position)
    assert yaw_from_quat(rack.slot_entry.quat_wxyz) == pytest.approx(math.pi / 2.0)


def test_the_slot_index_is_validated_by_the_layout():
    with pytest.raises(ValueError, match="slot index"):
        FakeRack.from_layout(LAYOUT, 6, DRY_RUN_RACK_FRAME, diameter_m=0.25)


def test_the_dry_run_rack_frame_is_the_dishwasher_pose_with_the_rack_pulled_out():
    # #67: the #59 raised compact dishwasher (D013), not the old floor-standing placeholder
    np.testing.assert_allclose(DRY_RUN_RACK_FRAME.position, (0.3502, -0.2600, 0.6496), atol=1e-4)
    assert math.degrees(yaw_from_quat(DRY_RUN_RACK_FRAME.quat_wxyz)) == pytest.approx(-90.0)
    # the fakes' slot entries are the dishwasher_frames slot poses, lifted by the convention
    for slot, (x, y, z, yaw) in enumerate(dishwasher_frames.slot_poses_kitchen(), start=1):
        rack = FakeRack.from_layout(LAYOUT, slot, DRY_RUN_RACK_FRAME, diameter_m=0.25)
        # rack +x (across the tines, half a thickness) is kitchen -y at yaw -90 deg
        expected = (x, y - kd.PLATE_HEIGHT_M / 2.0, z + 0.25)
        np.testing.assert_allclose(rack.slot_entry.position, expected, atol=1e-9)
        assert yaw_from_quat(rack.slot_entry.quat_wxyz) == pytest.approx(yaw)


def test_the_rack_needs_the_plate_diameter():
    # the slot entry of the grasp frame depends on it, so it has no default
    with pytest.raises(TypeError):
        FakeRack.from_layout(LAYOUT, 3, DRY_RUN_RACK_FRAME)  # type: ignore[call-arg]


@pytest.mark.parametrize("spec", [SEEN[0], UNSEEN[0]], ids=lambda s: f"{s.split}-{s.seed}")
def test_from_episode_builds_the_slot_entry_for_the_episode_plate(spec):
    fk = Fakes.from_episode(spec, LAYOUT, gate=False)
    expected = FakeRack.from_layout(
        LAYOUT, spec.rack_slot, DRY_RUN_RACK_FRAME, diameter_m=spec.plate_diameter_m
    )
    assert fk.rack.slot_entry == expected.slot_entry


# ----------------------------------------------------------------------------- from_episode


def test_from_episode_needs_the_gate_flag():
    with pytest.raises(TypeError):
        Fakes.from_episode(SEEN[0], LAYOUT)  # type: ignore[call-arg]


@pytest.mark.parametrize("spec", SPECS, ids=lambda s: f"{s.split}-{s.seed}")
def test_from_episode_puts_everything_in_one_frame(spec):
    fk = Fakes.from_episode(spec, LAYOUT, gate=False)
    assert np.allclose(fk.plate.body_pose().position, spec.plate_position_m)
    assert fk.rack.slot_index == spec.rack_slot
    assert LAYOUT.slot_pose(spec.rack_slot) == pytest.approx(spec.rack_slot_pose)  # same rack
    edge = fk.counter_edge_pose.position
    assert edge[1] == 0.0 and edge[2] == pytest.approx(kd.COUNTER_HEIGHT_M)  # the counter's edge
    assert edge[0] == pytest.approx(fk.plate.pose.position[0])  # at the grasp x (#67)
    assert fk.hand_reach_from_geometry and fk.plate_follows_palm
    obs = fk.observe()
    assert not obs.hand_holding and not obs.arm_infeasible and not obs.plate_in_slot
    assert obs.rack_contact_force == 0.0


def test_gate_selects_whether_the_arm_has_a_pipeline():
    spec = SEEN[0]
    off = Fakes.from_episode(spec, LAYOUT, gate=False)
    on = Fakes.from_episode(spec, LAYOUT, gate=True)
    assert off.arm.pipeline is None and off.arm.pelvis is not None
    assert isinstance(on.arm.pipeline, ArmTargetPipeline) and on.arm.pelvis is not None
    assert np.allclose(off.arm.palm.position, on.arm.palm.position)
    assert on.arm.pipeline.filter.max_lin_speed == fakes.ARM_LINEAR_SPEED_MPS
    assert on.arm.pipeline.filter.max_ang_speed == fakes.ARM_ANGULAR_SPEED_RPS


def test_a_pipeline_needs_the_pelvis_pose():
    with pytest.raises(ValueError, match="pelvis"):
        FakeArm(PALM_START, pipeline=ArmTargetPipeline())


@pytest.mark.parametrize("spec", SPECS, ids=lambda s: f"{s.split}-{s.seed}")
def test_the_palm_starts_inside_the_committed_envelope(spec):
    fk = Fakes.from_episode(spec, LAYOUT, gate=True)
    in_pelvis = to_frame(fk.arm.palm, fk.arm.pelvis)
    assert np.allclose(in_pelvis.position, PALM_START_IN_PELVIS.position)
    assert ReachabilityEnvelope.load().contains(in_pelvis.position)


def test_with_the_committed_envelope_the_gate_refuses_the_first_reach():
    # the status.md reach gap: the first target of pre_grasp is out of the palm's envelope
    fk = Fakes.from_episode(SEEN[0], LAYOUT, gate=True)
    log = run_episode(fakes=fk)
    assert log.final.state is PlannerState.FAILED
    assert log.final.failure is FailureTag.IK_INFEASIBLE
    assert log.planner.failed_in is PlannerState.PRE_GRASP
    assert fk.arm.refusals >= 1 and len(log.commands) < 10


@pytest.mark.parametrize(
    "spec", [SEEN[0], SEEN[7], UNSEEN[0], UNSEEN[11]], ids=lambda s: f"{s.split}-{s.seed}"
)
def test_with_a_gate_that_never_refuses_the_pipeline_reaches_done_like_the_ungated_arm(spec):
    gated = Fakes.from_episode(spec, LAYOUT, gate=True, envelope=_everything_envelope())
    free = Fakes.from_episode(spec, LAYOUT, gate=False)
    log_gated, log_free = run_episode(fakes=gated), run_episode(fakes=free)
    assert log_gated.final.state is PlannerState.DONE and gated.arm.refusals == 0
    assert log_free.final.state is PlannerState.DONE
    # the pipeline filters the same slew, so the two arms end at the same palm pose
    assert np.allclose(gated.arm.palm.position, free.arm.palm.position, atol=1e-6)
    assert np.allclose(gated.plate.pose.position, free.plate.pose.position, atol=1e-6)
    assert gated.rack.in_slot(gated.plate.pose.position)


def test_a_closer_stand_off_moves_the_pelvis_and_the_palm_start():
    near = Fakes.from_episode(SEEN[0], LAYOUT, gate=False, stand_off_m=0.10)
    far = Fakes.from_episode(SEEN[0], LAYOUT, gate=False)
    assert near.arm.palm.position[1] - far.arm.palm.position[1] == pytest.approx(
        kd.ROBOT_STANDOFF_M - 0.10
    )
