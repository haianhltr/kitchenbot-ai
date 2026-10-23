"""Grasp and placement judges against synthetic trajectories (docs/evaluation.md section 3, #57).

No simulation: every trajectory is a function of time sampled at 50 Hz. Placement trajectories are
written in the rack frame of ``kitchen_dims`` and handed to the judge as world poses of a rack that
is turned and moved away from the origin, so every placement test also exercises the conversion.
"""

import dataclasses
import itertools
import math
import subprocess
import sys

import numpy as np
import pytest

from kitchenbot.evaluation.failure_tags import FAILURE_PRECEDENCE, FailureTag
from kitchenbot.evaluation.judges import (
    GRASP_HOLD_S,
    GRASP_LIFT_HEIGHT_M,
    PLACEMENT_STABLE_S,
    PLACEMENT_VERTICAL_TOL_DEG,
    RACK_COLLISION_FORCE_N,
    STABLE_ANG_TOL_RAD_S,
    STABLE_LIN_TOL_M_S,
    THRESHOLDS,
    GraspJudge,
    PlacementJudge,
    SlotFit,
    StabilityWindow,
    episode_outcome,
    plate_in_slot,
    vertical_error_deg,
)
from kitchenbot.evaluation.metrics import SUCCESS_DEFINITIONS
from kitchenbot.geometry import (
    Pose,
    plate_vertical_quat,
    pose_compose,
    quat_apply,
    quat_from_axis_angle,
    quat_from_yaw,
    quat_multiply,
)
from kitchenbot.planning.plate_to_rack import EVAL_3, PROVISIONAL, PlannerConfig
from kitchenbot.scene.kitchen_dims import PLATE_DIAMETERS_M, PLATE_HEIGHT_M, RackSlotLayout

HZ = 50.0
LAYOUT = RackSlotLayout()
SLOT = 3
DIAMETER = 0.22
#: Where the rack frame is in the world: level, turned 90 deg, 18 cm off the floor.
RACK_WORLD = Pose((0.80, 0.35, 0.18), quat_from_yaw(math.radians(90.0)))
#: Plate origin height on the counter top (kitchen_dims COUNTER_HEIGHT_M; the origin is the
#: centre of the underside, #30).
COUNTER_Z = 0.90


def times(t0: float, t1: float) -> list[float]:
    """Sample times from ``t0`` to ``t1`` inclusive at ``HZ``, exact multiples of the step."""
    return [k / HZ for k in range(round(t0 * HZ), round(t1 * HZ) + 1)]


def seated(
    slot: int = SLOT,
    diameter: float = DIAMETER,
    *,
    lean_deg: float = 0.0,
    bottom_offset=(0.0, 0.0, 0.0),
    facing: int = 1,
    layout: RackSlotLayout = LAYOUT,
) -> Pose:
    """Rack-frame body pose of a plate standing in ``slot`` on the base frame.

    Its lowest rim point is at the slot centre on the base top plus ``bottom_offset``; its normal
    points along ``+x`` (``facing=1``) or ``-x``; ``lean_deg`` tips it across the gap about the
    rack ``y`` axis through that lowest point, which therefore stays put.
    """
    x, y, _, _ = layout.slot_pose(slot)
    bottom = np.array([x, y, layout.base_thickness]) + np.asarray(bottom_offset, dtype=float)
    upright = plate_vertical_quat(0.0 if facing > 0 else math.pi)
    normal = quat_apply(upright, (0.0, 0.0, 1.0))
    origin = bottom + np.array([0.0, 0.0, diameter / 2.0]) - (PLATE_HEIGHT_M / 2.0) * normal
    lean = quat_from_axis_angle((0.0, 1.0, 0.0), math.radians(lean_deg))
    return Pose(bottom + quat_apply(lean, origin - bottom), quat_multiply(lean, upright))


def raised(pose: Pose, dz: float) -> Pose:
    return Pose(pose.position + np.array([0.0, 0.0, dz]), pose.quat_wxyz)


def lowered_into_slot(final: Pose, seat_s: float):
    """The plate comes down at 10 cm/s and is seated at ``seat_s``, then follows ``final``."""
    return lambda t: raised(final, max(0.0, 0.10 * (seat_s - t)))


def run_placement(
    plate_rack_at,
    release_s: float,
    end_s: float,
    *,
    force_at=lambda t: 0.0,
    judge: PlacementJudge | None = None,
    rack_world: Pose = RACK_WORLD,
    held_from_s: float = 0.0,
) -> PlacementJudge:
    """Step a placement judge from 0 to ``end_s``; the hand holds from ``held_from_s`` to
    ``release_s`` (S3 starts holding at 0; S4 holds only after the grasp)."""
    judge = judge if judge is not None else PlacementJudge(LAYOUT, SLOT, DIAMETER)
    for t in times(0.0, end_s):
        holding = held_from_s <= t < release_s
        plate_world = pose_compose(rack_world, plate_rack_at(t))
        judge.step(t, plate_world, rack_world, force_at(t), not holding)
    return judge


BASE = dict(
    fell=False,
    ik_infeasible=False,
    grasp=None,
    placement=None,
    dropped=False,
    rack_collision_before_release=False,
    timed_out=False,
)


def outcome(scenario: str, **kwargs) -> tuple[bool, FailureTag]:
    return episode_outcome(scenario, **{**BASE, **kwargs})


# ----------------------------------------------------------------------------- thresholds


def test_planner_numbers_are_read_from_the_planner_config_not_copied():
    cfg = PlannerConfig()
    assert GRASP_LIFT_HEIGHT_M == cfg.lift_height_m == 0.15
    assert PLACEMENT_VERTICAL_TOL_DEG == cfg.vertical_tolerance_deg == 15.0
    assert PLACEMENT_STABLE_S == cfg.hold_check_s == 2.0
    assert RACK_COLLISION_FORCE_N == cfg.rack_collision_force_n == 20.0
    assert GRASP_HOLD_S == 2.0  # docs/evaluation.md section 3; the planner has no grasp hold


def test_every_threshold_names_its_source():
    planner_fields = {f.name: f for f in dataclasses.fields(PlannerConfig)}
    for name, threshold in THRESHOLDS.items():
        assert threshold.doc and threshold.unit, name
        if threshold.source.startswith("PlannerConfig."):
            field_name = threshold.source.split()[0].removeprefix("PlannerConfig.")
            field = planner_fields[field_name]
            assert threshold.value == getattr(PlannerConfig(), field_name), name
            assert field.metadata["source"] in threshold.source, name
        else:
            assert threshold.source in (EVAL_3, PROVISIONAL), name
    provisional = {n for n, t in THRESHOLDS.items() if t.source == PROVISIONAL}
    assert provisional == {"stable_lin_tol_m_s", "stable_ang_tol_rad_s"}
    assert (STABLE_LIN_TOL_M_S, STABLE_ANG_TOL_RAD_S) == (0.01, 0.1)


def test_judge_defaults_are_the_thresholds():
    grasp = GraspJudge()
    assert (grasp.lift_height_m, grasp.hold_s) == (GRASP_LIFT_HEIGHT_M, GRASP_HOLD_S)
    place = PlacementJudge(LAYOUT, SLOT, DIAMETER)
    assert place.vertical_tol_deg == PLACEMENT_VERTICAL_TOL_DEG
    assert place.stable_s == PLACEMENT_STABLE_S
    assert place.rack_force_n == RACK_COLLISION_FORCE_N
    assert place.thickness_m == PLATE_HEIGHT_M


# ----------------------------------------------------------------------------- vertical error


def test_vertical_error_of_seated_flat_and_leaning_plates():
    for yaw in (0.0, 0.7, math.pi):  # normal along the rack x axis or any other horizontal
        assert vertical_error_deg(plate_vertical_quat(yaw)) == pytest.approx(0.0, abs=1e-9)
    assert vertical_error_deg((1.0, 0.0, 0.0, 0.0)) == pytest.approx(90.0)  # lying flat
    for lean in (5.0, 14.9, 20.0, -20.0, 60.0):
        assert vertical_error_deg(seated(lean_deg=lean).quat_wxyz) == pytest.approx(abs(lean))


def test_vertical_error_ignores_spin_in_the_plate_plane_and_quaternion_sign():
    upright = plate_vertical_quat(0.0)
    spun = quat_multiply(upright, quat_from_axis_angle((0.0, 0.0, 1.0), 1.0))  # about the normal
    assert vertical_error_deg(spun) == pytest.approx(0.0, abs=1e-9)
    q = seated(lean_deg=12.0).quat_wxyz
    assert vertical_error_deg(-q) == pytest.approx(vertical_error_deg(q))


# ----------------------------------------------------------------------------- slot volume


@pytest.mark.parametrize("slot", range(1, LAYOUT.slots + 1))
@pytest.mark.parametrize("diameter", PLATE_DIAMETERS_M)
@pytest.mark.parametrize("facing", (1, -1))
def test_a_seated_plate_is_in_its_slot_whichever_way_it_faces(slot, diameter, facing):
    fit = plate_in_slot(seated(slot, diameter, facing=facing), slot, diameter, LAYOUT)
    assert fit.in_slot and fit.explain() == f"in slot {slot}"
    assert fit.across_m == pytest.approx(0.0, abs=1e-12)
    assert fit.along_m == pytest.approx(0.0, abs=1e-12)
    assert fit.rim_bottom_m == pytest.approx(-LAYOUT.tine_height)  # standing on the base top


def test_the_mid_plane_not_the_underside_origin_is_what_sits_in_the_gap():
    plate = seated(facing=1)
    x_slot = LAYOUT.slot_pose(SLOT)[0]
    # the body origin (underside centre, #30) is half the 20 mm plate thickness off the slot centre
    assert plate.position[0] - x_slot == pytest.approx(-PLATE_HEIGHT_M / 2.0)
    assert plate_in_slot(plate, SLOT, DIAMETER, LAYOUT).across_m == pytest.approx(0.0, abs=1e-12)


def test_slot_fit_offsets_are_signed_and_limits_come_from_the_layout():
    fit = plate_in_slot(seated(bottom_offset=(0.015, -0.03, 0.02)), SLOT, DIAMETER, LAYOUT)
    assert fit.across_m == pytest.approx(0.015)
    assert fit.along_m == pytest.approx(-0.03)
    assert fit.rim_bottom_m == pytest.approx(-LAYOUT.tine_height + 0.02)
    assert fit.in_slot
    assert fit.across_limit_m == pytest.approx(LAYOUT.gap_width / 2.0)
    assert fit.along_limit_m == pytest.approx(LAYOUT.row_spacing / 2.0)
    assert fit.depth_limit_m == pytest.approx(LAYOUT.slot_entry_z)
    scaled = RackSlotLayout(scale=1.4)
    fit = plate_in_slot(seated(layout=scaled), SLOT, DIAMETER, scaled)
    assert fit.in_slot and fit.across_limit_m == pytest.approx(scaled.gap_width / 2.0)
    assert fit.rim_bottom_m == pytest.approx(-scaled.tine_height)


def test_slot_volume_bounds():
    half_gap = LAYOUT.gap_width / 2.0
    half_rows = LAYOUT.row_spacing / 2.0

    def fit(offset):
        return plate_in_slot(seated(bottom_offset=offset), SLOT, DIAMETER, LAYOUT)

    assert fit((half_gap - 1e-4, 0.0, 0.0)).in_slot
    assert not fit((half_gap + 1e-4, 0.0, 0.0)).in_slot
    assert fit((0.0, -(half_rows - 1e-4), 0.0)).in_slot
    assert not fit((0.0, half_rows + 1e-4, 0.0)).in_slot
    # resting on the tine tops: the rim bottom is at the slot entry, not below it
    on_top = fit((0.0, 0.0, LAYOUT.tine_height))
    assert not on_top.in_slot and "not below the slot entry" in on_top.explain()
    # below the underside of the base frame (rack z = 0)
    below = fit((0.0, 0.0, -LAYOUT.base_thickness - 1e-3))
    assert not below.in_slot and "below the rack base" in below.explain()
    assert fit((0.0, 0.0, -LAYOUT.base_thickness + 1e-3)).in_slot


def test_a_plate_in_the_next_slot_is_not_in_this_one():
    fit = plate_in_slot(seated(SLOT + 1), SLOT, DIAMETER, LAYOUT)
    assert not fit.in_slot
    assert fit.across_m == pytest.approx(LAYOUT.pitch)
    assert fit.explain().startswith(f"not in slot {SLOT}: across +0.050 m")


def test_a_flat_plate_is_judged_at_its_mid_plane_centre():
    x, y, _, _ = LAYOUT.slot_pose(SLOT)
    flat = Pose((x, y, LAYOUT.slot_entry_z + 0.001))  # lying on the tine tops
    fit = plate_in_slot(flat, SLOT, DIAMETER, LAYOUT)
    assert fit.across_m == pytest.approx(0.0) and fit.along_m == pytest.approx(0.0)
    assert fit.rim_bottom_m == pytest.approx(0.001 + PLATE_HEIGHT_M / 2.0)
    assert not fit.in_slot


def test_plate_in_slot_rejects_bad_arguments():
    with pytest.raises(ValueError):
        plate_in_slot(seated(), 0, DIAMETER, LAYOUT)
    with pytest.raises(ValueError):
        plate_in_slot(seated(), LAYOUT.slots + 1, DIAMETER, LAYOUT)
    with pytest.raises(ValueError):
        plate_in_slot(seated(), SLOT, 0.0, LAYOUT)
    assert isinstance(plate_in_slot(seated(), SLOT, DIAMETER, LAYOUT), SlotFit)


# ----------------------------------------------------------------------------- stability window


def test_still_samples_close_the_window_after_its_duration():
    window = StabilityWindow(2.0, 0.01, 0.1)
    pose = Pose((0.1, 0.2, 0.3))
    closed = [t for t in times(5.0, 8.0) if window.step(t, pose)]
    assert closed[0] == pytest.approx(7.0)
    assert window.closed_at_s == pytest.approx(7.0) and window.stable_since_s == 5.0
    assert window.still_for_s == pytest.approx(3.0)


def test_motion_restarts_the_window_where_it_stopped():
    window = StabilityWindow(2.0, 0.01, 0.1)
    for t in times(0.0, 4.0):
        x = 0.05 * min(t, 1.0)  # moves at 5 cm/s until 1.0 s, then still
        window.step(t, Pose((x, 0.0, 0.0)))
    assert window.stable_since_s == pytest.approx(1.0)
    assert window.closed_at_s == pytest.approx(3.0)
    window.step(4.02, Pose((0.06, 0.0, 0.0)))  # a 1 cm jump in one step reopens it
    assert not window.stable and window.closed_at_s is None and window.stable_since_s == 4.02


def test_window_tolerances_are_rates():
    window = StabilityWindow(1.0, 0.01, 0.1)
    window.step(0.0, Pose((0.0, 0.0, 0.0)))
    window.step(0.5, Pose((0.004, 0.0, 0.0)))  # 8 mm/s: still
    assert window.stable_since_s == 0.0 and window.linear_speed_m_s == pytest.approx(0.008)
    window.step(1.0, Pose((0.004, 0.0, 0.0), quat_from_yaw(0.06)))  # 0.12 rad/s: moving
    assert window.stable_since_s == 1.0 and window.angular_speed_rad_s == pytest.approx(0.12)
    window.reset()
    assert window.stable_since_s is None and not window.stable


def test_window_rejects_bad_input():
    with pytest.raises(ValueError):
        StabilityWindow(0.0, 0.01, 0.1)
    with pytest.raises(ValueError):
        StabilityWindow(2.0, -0.01, 0.1)
    window = StabilityWindow(2.0, 0.01, 0.1)
    window.step(1.0, Pose((0.0, 0.0, 0.0)))
    with pytest.raises(ValueError):
        window.step(1.0, Pose((0.0, 0.0, 0.0)))


# ----------------------------------------------------------------------------- placement


def test_seated_plate_succeeds_and_completes_when_the_2s_window_closes():
    judge = run_placement(lowered_into_slot(seated(), seat_s=9.0), release_s=10.0, end_s=14.0)
    assert judge.placement_success
    assert judge.released_at_s == pytest.approx(10.0)
    assert judge.completion_time_s == pytest.approx(10.0 + PLACEMENT_STABLE_S)
    assert not judge.rack_collision
    assert judge.slot_fit.in_slot and judge.last_vertical_error_deg == pytest.approx(0, abs=1e-6)
    assert judge.why().startswith(f"placed in slot {SLOT}") and "12.00 s" in judge.why()
    assert outcome("S3", placement=judge.placement_success) == (True, FailureTag.SUCCESS)


def test_completion_is_measured_from_episode_time_zero_not_from_release():
    judge = run_placement(lambda t: seated(), release_s=3.0, end_s=6.0)
    assert judge.completion_time_s == pytest.approx(5.0)


def test_plate_leaning_20_deg_is_a_misplace():
    judge = run_placement(lambda t: seated(lean_deg=20.0), release_s=10.0, end_s=16.0)
    assert not judge.placement_success and judge.completion_time_s is None
    assert judge.slot_fit.in_slot  # its lowest point is in the slot; only the lean fails
    assert judge.last_vertical_error_deg == pytest.approx(20.0)
    assert "20.0 deg from vertical, over 15.0" in judge.why()
    assert outcome("S3", placement=False, released=judge.released) == (False, FailureTag.MISPLACE)


def test_plate_leaning_14_deg_is_still_placed():
    judge = run_placement(lambda t: seated(lean_deg=14.0), release_s=10.0, end_s=13.0)
    assert judge.placement_success and judge.completion_time_s == pytest.approx(12.0)


def test_plate_dropped_to_the_floor_is_a_drop():
    g = 9.81
    drop_s, carry = 5.0, np.array([0.45, 0.35, 0.50])  # held flat 30 cm in front of the rack

    def plate_world(t):
        z = carry[2] - 0.5 * g * max(0.0, t - drop_s) ** 2
        return Pose((carry[0], carry[1], max(z, 0.0)))  # comes to rest flat on the floor

    judge = PlacementJudge(LAYOUT, SLOT, DIAMETER)
    for t in times(0.0, 10.0):
        judge.step(t, plate_world(t), RACK_WORLD, 0.0, released=t >= drop_s)
    assert not judge.placement_success
    # the judge cannot tell a drop from a release: the hand stopped holding, that is all it sees
    assert judge.released_at_s == pytest.approx(drop_s)
    fit = judge.slot_fit
    assert not fit.in_slot and fit.rim_bottom_m < -fit.depth_limit_m
    assert judge.last_vertical_error_deg == pytest.approx(90.0)
    assert f"not in slot {SLOT}" in judge.why()
    flags = dict(placement=False, released=judge.released)
    assert outcome("S3", dropped=True, **flags) == (False, FailureTag.DROP)
    # without the caller's drop signal the same trajectory reads as a release outside the slot
    assert outcome("S3", **flags) == (False, FailureTag.MISPLACE)


def test_25_n_bump_before_release_then_a_clean_seat_succeeds_with_rack_collision():
    judge = run_placement(
        lowered_into_slot(seated(), seat_s=9.0),
        release_s=10.0,
        end_s=13.0,
        force_at=lambda t: 25.0 if t == 8.0 else 1.0,
    )
    assert judge.placement_success and judge.completion_time_s == pytest.approx(12.0)
    assert judge.rack_collision and judge.rack_collision_at_s == pytest.approx(8.0)
    assert judge.max_force_before_release_n == pytest.approx(25.0)
    assert "rack contact 25.0 N over 20.0 N before release" in judge.why()
    ok = outcome("S3", placement=True, rack_collision_before_release=judge.rack_collision)
    assert ok == (True, FailureTag.SUCCESS)


def test_a_bump_after_release_is_ignored():
    judge = run_placement(
        lowered_into_slot(seated(), seat_s=9.0),
        release_s=10.0,
        end_s=13.0,
        force_at=lambda t: 60.0 if t in (10.0, 10.5) else (20.0 if t < 10.0 else 0.0),
    )
    assert not judge.rack_collision  # 20 N is not above 20 N; the 60 N came at and after release
    assert judge.max_force_before_release_n == pytest.approx(20.0)
    assert judge.placement_success


def test_wobbling_plate_completes_2s_after_it_settles():
    release_s, settle_s = 10.0, 11.5  # a 5 deg, 2 Hz wobble across the gap that stops at 11.5 s

    def wobble(t):
        lean = 5.0 * math.sin(2.0 * math.pi * 2.0 * (t - release_s)) if t < settle_s else 0.0
        return seated(lean_deg=lean)

    judge = run_placement(wobble, release_s=release_s, end_s=15.0)
    assert judge.placement_success
    assert judge.completion_time_s == pytest.approx(settle_s + PLACEMENT_STABLE_S)


def test_contacts_before_the_grasp_count_and_the_release_comes_after_holding():
    """S4: the hand does not hold the plate before the grasp, which is not a release."""
    judge = run_placement(
        lambda t: seated() if t >= 6.0 else raised(seated(), 0.5),
        release_s=6.0,
        end_s=9.0,
        held_from_s=2.0,
        force_at=lambda t: 30.0 if t == 1.0 else 0.0,  # the arm bumps the rack while reaching
    )
    assert judge.rack_collision and judge.rack_collision_at_s == pytest.approx(1.0)
    assert judge.released_at_s == pytest.approx(6.0)
    assert judge.completion_time_s == pytest.approx(8.0)


def test_never_held_and_still_held_are_not_placements():
    never = run_placement(lambda t: seated(), release_s=0.0, end_s=4.0, held_from_s=0.0)
    assert not never.released and not never.placement_success
    assert "never held" in never.why()
    holding = run_placement(lambda t: seated(), release_s=99.0, end_s=4.0)
    assert not holding.released and "still holds" in holding.why()
    assert outcome("S3", placement=False, released=holding.released, timed_out=True) == (
        False,
        FailureTag.TIMEOUT,
    )


def test_regrasp_after_release_restarts_the_window():
    judge = PlacementJudge(LAYOUT, SLOT, DIAMETER)
    plate = pose_compose(RACK_WORLD, seated())
    for t in times(0.0, 6.0):
        holding = t < 1.0 or 2.0 <= t < 2.5  # released at 1.0, held again 2.0 to 2.5
        judge.step(t, plate, RACK_WORLD, 50.0 if t == 2.2 else 0.0, not holding)
        if t == 2.2:
            assert "held again" in judge.why()
    assert judge.released_at_s == pytest.approx(1.0)
    assert not judge.rack_collision  # only contacts before the first release count
    assert judge.completion_time_s == pytest.approx(4.5)


def test_placement_verdict_does_not_depend_on_where_the_rack_is():
    trajectory = lowered_into_slot(seated(lean_deg=3.0, bottom_offset=(0.01, 0.02, 0.0)), 9.0)
    elsewhere = Pose((-1.3, 2.0, 0.25), quat_from_yaw(-2.4))
    judges = [
        run_placement(trajectory, 10.0, 13.0, rack_world=rack)
        for rack in (Pose((0.0, 0.0, 0.0)), RACK_WORLD, elsewhere)
    ]
    for judge in judges:
        assert judge.completion_time_s == pytest.approx(12.0)
        assert judge.slot_fit.across_m == pytest.approx(judges[0].slot_fit.across_m)
        assert judge.slot_fit.along_m == pytest.approx(0.02)


def test_the_window_is_judged_in_the_rack_frame():
    """A rack drawer sliding out carries the seated plate with it: still relative to the rack."""
    judge = PlacementJudge(LAYOUT, SLOT, DIAMETER)
    for t in times(0.0, 4.0):
        rack = Pose(RACK_WORLD.position + np.array([0.05 * t, 0.0, 0.0]), RACK_WORLD.quat_wxyz)
        judge.step(t, pose_compose(rack, seated()), rack, 0.0, released=t >= 1.0)
    assert judge.completion_time_s == pytest.approx(3.0)


def test_placement_judge_rejects_bad_input():
    with pytest.raises(ValueError):
        PlacementJudge(LAYOUT, 6, DIAMETER)
    with pytest.raises(ValueError):
        PlacementJudge(LAYOUT, SLOT, -0.22)
    judge = PlacementJudge(LAYOUT, SLOT, DIAMETER)
    with pytest.raises(ValueError):
        judge.step(0.0, RACK_WORLD, RACK_WORLD, -1.0, False)
    judge.step(0.5, RACK_WORLD, RACK_WORLD, 0.0, False)
    with pytest.raises(ValueError):
        judge.step(0.5, RACK_WORLD, RACK_WORLD, 0.0, False)
    judge.reset()
    judge.step(0.0, RACK_WORLD, RACK_WORLD, 0.0, False)  # a new episode starts at time zero


# ----------------------------------------------------------------------------- grasp


START = Pose((0.40, 0.25, COUNTER_Z), quat_from_yaw(0.3))


def run_grasp(lift_at, holding_at, end_s: float, contact_at=lambda t: False) -> GraspJudge:
    """Step a grasp judge from 0 to ``end_s``; ``lift_at(t)`` is the rise above the start."""
    judge = GraspJudge()
    judge.reset(START)
    for t in times(0.0, end_s):
        judge.step(t, raised(START, lift_at(t)), holding_at(t), counter_contact=contact_at(t))
    return judge


def ramp(t0: float, t1: float, height: float):
    """Rise linearly from 0 at ``t0`` to ``height`` at ``t1``, then stay."""
    return lambda t: height * min(1.0, max(0.0, (t - t0) / (t1 - t0)))


def test_lift_15_cm_and_hold_2_s_is_a_grasp():
    judge = run_grasp(ramp(0.5, 1.5, 0.15), lambda t: t >= 0.5, end_s=5.0)
    assert judge.grasp_success
    assert judge.held_at_s == pytest.approx(0.5) and judge.lifted_at_s == pytest.approx(1.5)
    assert judge.grasped_at_s == pytest.approx(3.5)
    assert judge.why().startswith("grasped") and not judge.dropped
    assert outcome("S2", grasp=True) == (True, FailureTag.SUCCESS)


def test_plate_may_go_lower_during_the_hold():
    """S4: the transport descends towards the rack right after the lift."""

    def lift(t):
        return ramp(0.5, 1.5, 0.15)(t) - (0.4 * (t - 2.0) if t > 2.0 else 0.0)

    judge = run_grasp(lift, lambda t: t >= 0.5, end_s=4.0)
    assert judge.grasp_success and judge.grasped_at_s == pytest.approx(3.5)


def test_lift_of_10_cm_only_is_no_grasp():
    judge = run_grasp(ramp(1.0, 2.0, 0.10), lambda t: t >= 1.0, end_s=30.0)
    assert not judge.grasp_success and judge.grasped_at_s is None
    assert judge.held and not judge.dropped
    assert "lifted at most 0.100 m of 0.150 m" in judge.why()
    flags = dict(grasp=False, held=judge.held, dropped=judge.dropped, timed_out=True)
    assert outcome("S2", **flags) == (False, FailureTag.TIMEOUT)


def test_lift_of_15_cm_then_drop_at_1_5_s_is_no_grasp():
    judge = run_grasp(ramp(0.5, 1.5, 0.15), lambda t: 0.5 <= t < 3.0, end_s=6.0)
    assert not judge.grasp_success
    assert judge.dropped and judge.dropped_at_s == pytest.approx(3.0)  # 1.5 s into the hold
    assert "dropped at 3.00 s" in judge.why()
    flags = dict(grasp=False, held=judge.held, dropped=judge.dropped)
    assert outcome("S2", **flags) == (False, FailureTag.DROP)


def test_never_held_is_a_grasp_miss():
    judge = run_grasp(lambda t: 0.0, lambda t: False, end_s=3.0)
    assert not judge.grasp_success and not judge.held and not judge.dropped
    assert "never reported holding" in judge.why()
    flags = dict(grasp=False, held=judge.held, dropped=judge.dropped, timed_out=True)
    assert outcome("S2", **flags) == (False, FailureTag.GRASP_MISS)


def test_counter_contact_during_the_hold_restarts_it():
    judge = run_grasp(
        ramp(0.5, 1.5, 0.15), lambda t: t >= 0.5, end_s=6.0, contact_at=lambda t: t == 2.5
    )
    assert judge.lifted_at_s == pytest.approx(2.52) and judge.grasped_at_s == pytest.approx(4.52)


def test_a_new_lift_after_a_drop_can_still_succeed():
    judge = run_grasp(ramp(0.5, 1.5, 0.15), lambda t: 0.5 <= t < 2.0 or t >= 2.5, end_s=6.0)
    assert judge.dropped and judge.grasp_success and judge.grasped_at_s == pytest.approx(4.5)
    assert outcome("S2", grasp=True, dropped=True) == (True, FailureTag.SUCCESS)


def test_grasp_verdict_is_final_and_input_is_checked():
    judge = run_grasp(ramp(0.5, 1.5, 0.15), lambda t: 0.5 <= t < 4.0, end_s=6.0)
    assert judge.grasp_success and not judge.dropped  # the loss at 4.0 s came after success
    with pytest.raises(ValueError):
        judge.step(6.0, START, True)
    with pytest.raises(RuntimeError):
        GraspJudge().step(0.0, START, True)
    with pytest.raises(ValueError):
        GraspJudge(lift_height_m=0.0)


# ----------------------------------------------------------------------------- episode outcome

S, F = FailureTag.SUCCESS, FailureTag
OUTCOME_TABLE = [
    # S2: grasp and lift; success is the grasp verdict, with no fall and no timeout
    ("S2", dict(grasp=True), (True, S)),
    ("S2", dict(grasp=True, placement=False, rack_collision_before_release=True), (True, S)),
    ("S2", dict(grasp=True, fell=True), (False, F.FALL)),
    ("S2", dict(grasp=True, timed_out=True), (False, F.TIMEOUT)),
    ("S2", dict(grasp=False), (False, F.GRASP_MISS)),
    ("S2", dict(grasp=False, held=False, timed_out=True), (False, F.GRASP_MISS)),
    ("S2", dict(grasp=False, held=True, timed_out=True), (False, F.TIMEOUT)),
    ("S2", dict(grasp=False, held=True, dropped=True), (False, F.DROP)),
    ("S2", dict(grasp=False, dropped=True, ik_infeasible=True), (False, F.IK_INFEASIBLE)),
    ("S2", dict(grasp=False, dropped=True, ik_infeasible=True, fell=True), (False, F.FALL)),
    (
        "S2",
        dict(grasp=False, held=True, rack_collision_before_release=True, timed_out=True),
        (False, F.TIMEOUT),
    ),
    # S3: place in rack; the plate starts in the hand, the grasp verdict is not read
    ("S3", dict(placement=True), (True, S)),
    ("S3", dict(placement=True, grasp=False), (True, S)),
    ("S3", dict(placement=True, rack_collision_before_release=True), (True, S)),
    ("S3", dict(placement=True, fell=True), (False, F.FALL)),
    ("S3", dict(placement=False), (False, F.MISPLACE)),
    ("S3", dict(placement=False, released=True, timed_out=True), (False, F.MISPLACE)),
    ("S3", dict(placement=False, released=False, timed_out=True), (False, F.TIMEOUT)),
    ("S3", dict(placement=False, rack_collision_before_release=True), (False, F.RACK_COLLISION)),
    (
        "S3",
        dict(placement=False, rack_collision_before_release=True, dropped=True),
        (False, F.DROP),
    ),
    ("S3", dict(placement=False, ik_infeasible=True, released=False), (False, F.IK_INFEASIBLE)),
    # S4: full task; grasp and placement, no fall, within the 30 s timeout
    ("S4", dict(grasp=True, placement=True), (True, S)),
    ("S4", dict(grasp=True, placement=True, rack_collision_before_release=True), (True, S)),
    ("S4", dict(grasp=True, placement=True, timed_out=True), (False, F.TIMEOUT)),
    ("S4", dict(grasp=True, placement=True, fell=True), (False, F.FALL)),
    ("S4", dict(grasp=False, placement=False), (False, F.GRASP_MISS)),
    ("S4", dict(grasp=False, placement=False, held=True, timed_out=True), (False, F.TIMEOUT)),
    ("S4", dict(grasp=True, placement=False, dropped=True), (False, F.DROP)),
    (
        "S4",
        dict(grasp=True, placement=False, rack_collision_before_release=True),
        (False, F.RACK_COLLISION),
    ),
    ("S4", dict(grasp=True, placement=False, released=True), (False, F.MISPLACE)),
    ("S4", dict(grasp=True, placement=False, released=False, timed_out=True), (False, F.TIMEOUT)),
    ("fake", dict(grasp=True, placement=False, released=True), (False, F.MISPLACE)),
]


@pytest.mark.parametrize(("scenario", "kwargs", "expected"), OUTCOME_TABLE)
def test_episode_outcome_table(scenario, kwargs, expected):
    assert outcome(scenario, **kwargs) == expected


def test_every_cause_at_once_resolves_by_the_section_7_precedence():
    everything = dict(
        fell=True,
        ik_infeasible=True,
        grasp=False,
        placement=False,
        dropped=True,
        rack_collision_before_release=True,
        timed_out=True,
    )
    assert episode_outcome("S4", **everything) == (False, FAILURE_PRECEDENCE[0])
    # peel the causes off in precedence order
    assert outcome("S4", grasp=False, placement=False, ik_infeasible=True) == (
        False,
        FAILURE_PRECEDENCE[1],
    )


@pytest.mark.parametrize("scenario", ["S2", "S3", "S4"])
def test_success_matches_the_success_definitions(scenario):
    """Exhaustively over the two verdicts with no other cause: success is the definition."""
    needs = {"S2": ("grasp",), "S3": ("placement",), "S4": ("grasp", "placement")}[scenario]
    for name in needs:
        assert name in SUCCESS_DEFINITIONS[scenario]
    for grasp, placement in itertools.product((False, True), repeat=2):
        verdicts = {"grasp": grasp, "placement": placement}
        expected = all(verdicts[n] for n in needs)
        success, tag = outcome(scenario, grasp=grasp, placement=placement, timed_out=not expected)
        assert success is expected and (tag is S) is expected
    assert "no fall" in SUCCESS_DEFINITIONS["S4"] and "30 s" in SUCCESS_DEFINITIONS["S4"]


def test_episode_outcome_rejects_what_it_cannot_judge():
    for scenario in ("S0", "S1", "S9"):
        with pytest.raises(ValueError):
            outcome(scenario, grasp=True, placement=True)
    with pytest.raises(ValueError, match="grasp phase"):
        outcome("S2", placement=True)
    with pytest.raises(ValueError, match="placement phase"):
        outcome("S4", grasp=True)
    with pytest.raises(ValueError, match="no cause"):
        outcome("S2", grasp=False, held=True)


# ----------------------------------------------------------------------------- imports


def test_importing_the_judges_pulls_in_no_simulator():
    code = (
        "import sys\n"
        "import kitchenbot.evaluation.judges\n"
        "banned = {'isaaclab', 'isaacsim', 'omni', 'pxr', 'torch', 'carb', 'warp'}\n"
        "hits = sorted(m for m in sys.modules if m.split('.')[0] in banned)\n"
        "assert not hits, hits\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
